"""Gradient-path ablation of a fixed imputer, with disposable neural state."""
import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from ..latent_completion_20261028.model import tokens,masked,latent_loss,readout

SPECS={'stop_target':('stop_target',False),
       'stop_target_reverse':('stop_target',True),
       'stop_prediction':('stop_prediction',False)}


def objective(prediction,target,valid,kind):
    if kind=='stop_target':target=target.detach()
    elif kind=='stop_prediction':prediction=prediction.detach()
    else:raise ValueError('explicit gradient path required')
    return latent_loss(prediction,target,valid).mean()


def step(base,net,x,a,state,kind,reverse):
    if len(x)!=1:raise ValueError('one arrived window from current shot required')
    if not (a & torch.isfinite(x)).any():return state,dict(updates=0,gradient_norm=0.,loss=0.)
    fast=state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        h,v=tokens(base,x,a,fast);xx,aa=masked(x,a);mh,mv=tokens(base,xx,aa,fast)
        if reverse:
            mh=torch.cat((mh[:,:-1].flip(1),mh[:,-1:]),1)
            mv=torch.cat((mv[:,:-1].flip(1),mv[:,-1:]),1)
        loss=objective(net(mh,mv),h.expand(2,-1,-1),v.expand(2,-1),kind)
        gradient,=torch.autograd.grad(loss,fast);norm=gradient.norm()
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient-path update')
        proposed=fast-.05*gradient/norm.clamp_min(1e-12)
        proposed=proposed/proposed.norm().clamp_min(1.)
    return proposed.detach(),dict(updates=int(norm>0),gradient_norm=float(norm),loss=float(loss.detach()))


def predict(base,net,signal,slow):
    if set(signal)!={'values','available'}:raise ValueError('signal-only prediction required')
    states={k:base.center.new_zeros(128) for k in SPECS}
    curves={k:[] for k in ('cold',*SPECS)};names=list(curves)
    info={k:dict(updates=0,gradient_norm_sum=0.,loss_sum=0.,max_state_norm=0.) for k in SPECS}
    for lo in range(0,len(signal['values']),32):
        hi=min(lo+32,len(signal['values']));x,a=batch(signal,range(lo,hi),base.center.device)
        fast=torch.cat([base.center.new_zeros(len(x),128)]+[states[k].expand(len(x),-1) for k in SPECS])
        with torch.no_grad():
            h,v=tokens(base,x.repeat(len(names),1,1),a.repeat(len(names),1,1),fast)
            scores=readout(base,h,v,slow).sigmoid().reshape(len(names),len(x)).cpu().numpy()
        if not np.isfinite(scores).all():raise FloatingPointError('nonfinite risk')
        for name,p in zip(names,scores):curves[name].append(p)
        # No learning can change this already-scored prefix.
        if hi%128==0 and hi<len(signal['values']):
            for name,(kind,reverse) in SPECS.items():
                states[name],r=step(base,net,x[-1:],a[-1:],states[name],kind,reverse)
                q=info[name];q['updates']+=r['updates'];q['gradient_norm_sum']+=r['gradient_norm'];q['loss_sum']+=r['loss']
                q['max_state_norm']=max(q['max_state_norm'],float(states[name].norm()))
    return {k:np.concatenate(v) for k,v in curves.items()},info
