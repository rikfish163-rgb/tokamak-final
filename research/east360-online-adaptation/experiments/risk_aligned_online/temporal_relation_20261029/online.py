"""Disposable internal adaptation from arrived order pairs, never outcomes."""
import numpy as np
import torch
from torch.nn import functional as F

from ..delayed_recurrent_memory_20261025.learning import batch
from ..latent_completion_20261028.model import tokens,readout
from .model import margin,permutation

ARMS=('cold','relation','wrong_label','smooth')


def step(base,net,x,a,p,state,kind):
    if len(x)!=1 or kind not in ARMS[1:]:raise ValueError('one arrived window and explicit task required')
    fast=state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        delta,eligible,tv=margin(base,net,x,a,p,fast)
        if not eligible.item():
            return state,dict(updates=0,eligible=False,gradient_norm=0.,loss=0.)
        value=tv if kind=='smooth' else (-delta if kind=='wrong_label' else delta)
        loss=F.softplus(-value).mean();gradient,=torch.autograd.grad(loss,fast);norm=gradient.norm()
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite relation update')
        proposed=fast-.05*gradient/norm.clamp_min(1e-12)
        proposed=proposed/proposed.norm().clamp_min(1.)
    return proposed.detach(),dict(updates=int(norm>0),eligible=True,gradient_norm=float(norm),loss=float(loss.detach()))


def predict(base,net,signal,slow):
    if set(signal)!={'values','available'}:raise ValueError('observed signal-only prediction required')
    states={k:base.center.new_zeros(128) for k in ARMS[1:]};curves={k:[] for k in ARMS}
    info={k:dict(updates=0,eligible=0,opportunities=0,gradient_norm_sum=0.,loss_sum=0.,max_state_norm=0.) for k in states}
    rng=np.random.default_rng(131)
    for lo in range(0,len(signal['values']),32):
        hi=min(lo+32,len(signal['values']));x,a=batch(signal,range(lo,hi),base.center.device)
        fast=torch.cat([base.center.new_zeros(len(x),128)]+[s.expand(len(x),-1) for s in states.values()])
        with torch.no_grad():
            h,v=tokens(base,x.repeat(len(ARMS),1,1),a.repeat(len(ARMS),1,1),fast)
            scores=readout(base,h,v,slow).sigmoid().reshape(len(ARMS),len(x)).cpu().numpy()
        if not np.isfinite(scores).all():raise FloatingPointError('nonfinite risk')
        for name,score in zip(ARMS,scores):curves[name].append(score)
        # All predictions up to hi are sealed before this arrived-window step.
        if hi%128==0 and hi<len(signal['values']):
            p=permutation(1,rng)
            for name in states:
                states[name],r=step(base,net,x[-1:],a[-1:],p,states[name],name)
                q=info[name];q['opportunities']+=1;q['eligible']+=int(r['eligible']);q['updates']+=r['updates']
                q['gradient_norm_sum']+=r['gradient_norm'];q['loss_sum']+=r['loss']
                q['max_state_norm']=max(q['max_state_norm'],float(states[name].norm()))
    return {k:np.concatenate(v) for k,v in curves.items()},info
