"""Fit local diagnostic transitions INSIDE an otherwise frozen risk network.

Fast weights minimize an observed-diagnostic ridge objective, not a risk-label
loss. The last transition is held out of fitting. All samples are in the past
window, and each call starts from the same learned prior, not another shot.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def fit_local(keys, target, mask, prior):
    if (keys.ndim != 3 or target.ndim != 3 or keys.shape[:2] != target.shape[:2]
            or mask.shape != target.shape or mask.dtype != torch.bool
            or prior.shape != (keys.shape[-1],target.shape[-1])):
        raise ValueError('matching batched keys, targets, masks and prior required')
    if not torch.isfinite(keys).all() or not torch.isfinite(target[mask]).all() or not torch.isfinite(prior).all():
        raise FloatingPointError('nonfinite observed transition')
    valid = mask.to(keys.dtype); n = valid.sum(1).clamp_min(1.)
    gram = torch.einsum('btr,btc,bts->bcrs',keys,valid,keys)/n[:,:,None,None]
    residual = torch.where(mask,target-keys@prior,0.)
    rhs = torch.einsum('btr,btc->bcr',keys,residual)/n[:,:,None]
    system = gram+.1*torch.eye(keys.shape[-1],device=keys.device,dtype=keys.dtype)
    correction = torch.linalg.solve(system,rhs.unsqueeze(-1)).squeeze(-1).transpose(1,2)
    return prior[None]+correction


class LocalTransition(nn.Module):
    def __init__(self,base):
        super().__init__()
        if base.cfg.window != 128 or base.cfg.patch != 8 or base.cfg.input_view != 'source':
            raise ValueError('fixed source-coordinate W128/patch8 base required')
        self.base = base; self.base.requires_grad_(False); self.base.eval()
        self.key = nn.Linear(46,8)
        self.prior = nn.Parameter(torch.zeros(9,23))
        self.readout = nn.Linear(23,base.cfg.dim,bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self,values,available,*,mode):
        if mode not in ('prior','fit','shuffled'):
            raise ValueError('explicit prior/fit/shuffled mode required')
        if values.ndim != 3 or values.shape[1:] != (128,23) or available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError('W128x23 observations and boolean availability required')
        mask = available & torch.isfinite(values)
        clean = torch.where(mask,values,self.base.center)
        level = torch.asinh((clean-self.base.center)/self.base.scale)
        batch = len(values); pm = mask.reshape(batch,16,8,23)
        pl = level.reshape(batch,16,8,23)
        count = pm.sum(2); complete = count == 8
        mean = pl.masked_fill(~pm,0.).sum(2)/count.clamp_min(1)
        mean = mean.masked_fill(~complete,0.)
        features = torch.cat((mean,complete.to(mean.dtype)),-1)
        k = self.key(features).tanh()
        k = torch.cat((k,torch.ones_like(k[:,:,:1])),-1)
        changes = mean[:,1:]-mean[:,:-1]
        valid = complete[:,1:] & complete[:,:-1]
        target, fit_mask = changes[:,:14],valid[:,:14]
        if mode == 'shuffled':
            target,fit_mask = target.flip(1),fit_mask.flip(1)
        fast = (self.prior[None].expand(batch,-1,-1) if mode == 'prior'
                else fit_local(k[:,:14],target,fit_mask,self.prior))
        prediction = torch.einsum('br,brc->bc',k[:,14],fast)
        query_mask = valid[:,14]
        innovation = (changes[:,14]-prediction).masked_fill(~query_mask,0.)
        error = F.smooth_l1_loss(prediction,changes[:,14],reduction='none').masked_fill(~query_mask,0.)
        forecast_loss = error.sum()/query_mask.sum().clamp_min(1)
        correction = self.readout(innovation.tanh())
        x = torch.cat((level,mask.to(level.dtype)),-1).reshape(batch,16,-1)
        x = self.base.embedding(x)
        x = torch.cat((x[:,:-1],(x[:,-1]+correction)[:,None]),1)
        # Keep the same unfused, grad-enabled Transformer path as the archived
        # paired predictor, including when all slow parameters are frozen.
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        x = torch.cat((x,self.base.cls.expand(batch,-1,-1)),1)+self.base.position
        tv = mask.reshape(batch,16,-1).any(-1)
        tv = torch.cat((tv,torch.ones(batch,1,dtype=torch.bool,device=tv.device)),1)
        with sdpa_kernel(SDPBackend.MATH):
            x = self.base.encoder(x,src_key_padding_mask=~tv)
        gain,bias = self.base.initial_fast(batch).chunk(2,-1)
        h = self.base.norm(x)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
        return dict(logit=self.base.risk_head(h[:,-1]).squeeze(-1),fast=fast,
                    forecast_loss=forecast_loss,forecast_error_sum=error.sum(),
                    target_mask=query_mask,innovation=innovation,correction=correction,
                    local_observations=fit_mask.sum((1,2)))
