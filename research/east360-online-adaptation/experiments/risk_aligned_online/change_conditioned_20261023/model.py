"""Compare absolute-state and change-conditioned fast diagnostic predictors."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from ..local_transition_20261023.model import fit_local
from ..context_separation_20261023.model import mean_shift


def patches(base,values,available):
    if values.ndim!=3 or values.shape[1:]!=(128,23) or available.shape!=values.shape or available.dtype!=torch.bool:
        raise ValueError('W128x23 and explicit boolean availability required')
    mask=available & torch.isfinite(values)
    level=torch.asinh((torch.where(mask,values,base.center)-base.center)/base.scale)
    b=len(values);pm=mask.reshape(b,16,8,23);count=pm.sum(2);complete=count==8
    mean=level.reshape(b,16,8,23).masked_fill(~pm,0.).sum(2)/count.clamp_min(1)
    mean=mean.masked_fill(~complete,0.)
    valid=complete[:,1:] & complete[:,:-1]
    delta=(mean[:,1:]-mean[:,:-1]).masked_fill(~valid,0.)
    return dict(level=level,mask=mask,mean=mean,delta=delta,valid=valid)


def fixed_scale(rows):
    if not rows:raise ValueError('explicit historical training observations required')
    sums=torch.stack([p['delta'].double().square().sum((0,1)) for p in rows]).sum(0)
    counts=torch.stack([p['valid'].sum((0,1)) for p in rows]).sum(0)
    if not torch.isfinite(sums).all():raise FloatingPointError('nonfinite training differences')
    # A zero-observation channel cannot define a scale: identity is recorded
    # explicitly and remains masked. The fixed floor prevents amplifying
    # near-constant training noise; it is not a physical detection threshold.
    scale=torch.where(counts>0,(sums/counts.clamp_min(1)).sqrt().clamp_min(.001),1.)
    return scale.float(),counts


class ChangeConditioned(nn.Module):
    def __init__(self,base,scale,key_kind):
        super().__init__()
        if base.cfg.window!=128 or base.cfg.patch!=8 or base.cfg.input_view!='source':
            raise ValueError('source-coordinate fixed window/patch contract required')
        if key_kind not in ('level','change') or scale.shape!=(23,) or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('explicit positive fixed scale and level/change key required')
        self.base=base;self.base.requires_grad_(False);self.base.eval();self.key_kind=key_kind
        self.register_buffer('change_scale',scale.detach().clone().float())
        self.key=nn.Linear(46,8);self.prior=nn.Parameter(torch.zeros(9,23))
        self.readout=nn.Linear(23,base.cfg.dim,bias=False);nn.init.zeros_(self.readout.weight)

    def components(self,values,available):
        p=patches(self.base,values,available)
        changes=torch.asinh(p['delta']/self.change_scale)
        feature=p['mean'][:,1:-1] if self.key_kind=='level' else changes[:,:-1]
        k=self.key(torch.cat((feature,p['valid'][:,:-1].to(feature.dtype)),-1)).tanh()
        k=torch.cat((k,torch.ones_like(k[:,:,:1])),-1)
        return p,k,changes[:,1:],p['valid'][:,:-1]&p['valid'][:,1:]

    def forward(self,values,available,*,mode):
        if mode not in ('prior','fit','shuffled','mean'):raise ValueError('explicit mode required')
        p,k,target,valid=self.components(values,available)
        t,m=target[:,:13],valid[:,:13]
        if mode=='shuffled':t,m=t.flip(1),m.flip(1)
        b=len(values)
        if mode=='prior':fast=self.prior[None].expand(b,-1,-1)
        elif mode=='mean':
            shift=mean_shift(k[:,:13],t,m,self.prior)
            fast=self.prior[None]+F.pad(shift[:,None],(0,0,8,0))
        else:fast=fit_local(k[:,:13],t,m,self.prior)
        prediction=torch.einsum('br,brc->bc',k[:,13],fast)
        query_mask=valid[:,13]
        innovation=(target[:,13]-prediction).masked_fill(~query_mask,0.)
        error=F.smooth_l1_loss(prediction,target[:,13],reduction='none').masked_fill(~query_mask,0.)
        correction=self.readout(innovation.tanh())
        base=self.base;level=p['level'];mask=p['mask']
        x=base.embedding(torch.cat((level,mask.to(level.dtype)),-1).reshape(b,16,-1))
        x=torch.cat((x[:,:-1],(x[:,-1]+correction)[:,None]),1)
        if not x.requires_grad:x=x.detach().requires_grad_(True)
        x=torch.cat((x,base.cls.expand(b,-1,-1)),1)+base.position
        tv=mask.reshape(b,16,-1).any(-1)
        tv=torch.cat((tv,torch.ones(b,1,dtype=torch.bool,device=tv.device)),1)
        with sdpa_kernel(SDPBackend.MATH):x=base.encoder(x,src_key_padding_mask=~tv)
        gain,bias=base.initial_fast(b).chunk(2,-1)
        h=base.norm(x)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
        return dict(logit=base.risk_head(h[:,-1]).squeeze(-1),fast=fast,
            forecast_loss=error.sum()/query_mask.sum().clamp_min(1),forecast_error_sum=error.sum(),
            target_mask=query_mask,innovation=innovation,correction=correction,
            local_observations=m.sum((1,2)))
