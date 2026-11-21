"""Separate local mean correction from state-dependent transition fitting."""
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from ..local_transition_20261023.model import fit_local


def parts(model,values,available):
    if values.ndim!=3 or values.shape[1:]!=(128,23) or available.shape!=values.shape or available.dtype!=torch.bool:
        raise ValueError('explicit W128x23 signal and availability required')
    mask=available & torch.isfinite(values)
    level=torch.asinh((torch.where(mask,values,model.base.center)-model.base.center)/model.base.scale)
    b=len(values);pm=mask.reshape(b,16,8,23);pl=level.reshape(b,16,8,23)
    count=pm.sum(2);complete=count==8
    mean=pl.masked_fill(~pm,0.).sum(2)/count.clamp_min(1)
    mean=mean.masked_fill(~complete,0.)
    k=model.key(torch.cat((mean,complete.to(mean.dtype)),-1)).tanh()
    k=torch.cat((k,torch.ones_like(k[:,:,:1])),-1)
    return dict(level=level,mask=mask,key=k,changes=mean[:,1:]-mean[:,:-1],
                valid=complete[:,1:] & complete[:,:-1])


def mean_shift(keys,target,mask,prior):
    if keys.ndim!=3 or target.shape[:2]!=keys.shape[:2] or mask.shape!=target.shape or mask.dtype!=torch.bool or prior.shape!=(keys.shape[-1],target.shape[-1]):
        raise ValueError('matching keys, targets, masks and prior required')
    if not torch.isfinite(keys).all() or not torch.isfinite(target[mask]).all() or not torch.isfinite(prior).all():
        raise FloatingPointError('nonfinite observed transition')
    valid=mask.to(keys.dtype);n=valid.sum(1).clamp_min(1)
    residual=torch.where(mask,target-keys@prior,0.).sum(1)/n
    mean_key=torch.einsum('btr,btc->bcr',keys,valid)/n[:,:,None]
    energy=mean_key.square().sum(-1)
    return residual*energy/(energy+.1)


def forward(model,values,available,reference,reference_mask,*,kind):
    if kind not in ('ridge','mean'):raise ValueError('explicit ridge or mean required')
    current=parts(model,values,available);past=parts(model,reference,reference_mask)
    k,target,mask=past['key'][:,:14],past['changes'][:,:14],past['valid'][:,:14]
    query=current['key'][:,14]
    if kind=='ridge':
        fitted=fit_local(k,target,mask,model.prior)
        prediction=torch.einsum('br,brc->bc',query,fitted)
    else:
        fitted=mean_shift(k,target,mask,model.prior)
        prediction=query@model.prior+fitted
    valid=current['valid'][:,14]
    innovation=(current['changes'][:,14]-prediction).masked_fill(~valid,0.)
    error=F.smooth_l1_loss(prediction,current['changes'][:,14],reduction='none').masked_fill(~valid,0.)
    correction=model.readout(innovation.tanh())
    b=len(values);level=current['level'];observed=current['mask'];base=model.base
    x=base.embedding(torch.cat((level,observed.to(level.dtype)),-1).reshape(b,16,-1))
    x=torch.cat((x[:,:-1],(x[:,-1]+correction)[:,None]),1)
    if not x.requires_grad:x=x.detach().requires_grad_(True)
    x=torch.cat((x,base.cls.expand(b,-1,-1)),1)+base.position
    tv=observed.reshape(b,16,-1).any(-1)
    tv=torch.cat((tv,torch.ones(b,1,dtype=torch.bool,device=tv.device)),1)
    with sdpa_kernel(SDPBackend.MATH):x=base.encoder(x,src_key_padding_mask=~tv)
    gain,bias=base.initial_fast(b).chunk(2,-1)
    h=base.norm(x)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
    return dict(logit=base.risk_head(h[:,-1]).squeeze(-1),fitted=fitted,innovation=innovation,
        forecast_error_sum=error.sum(),target_count=valid.sum(),correction=correction,
        fitted_windows=(mask.sum((1,2))>0).sum())
