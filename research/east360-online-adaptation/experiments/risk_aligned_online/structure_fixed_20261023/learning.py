"""Fixed positional/CLS parameters versus an unlabelled step-matched control."""
import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..forecast_token_scope_20261023.learning import SUMMARY
from ..history_tangent_20261023.learning import logit
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('observation','matched_all')


def update_pair(model,prefix,count,states,*,device):
    if set(prefix)!={'values','available'} or len(prefix['values'])!=count or set(states)!=set(ARMS):
        raise ValueError('exact arrived prefix and two declared states required')
    x,m,truth,tm,ends=forecast_examples(prefix,count,model.cfg,device)
    mask=target_mask(x,m,truth,tm);observed=int(mask.sum());out={};receipts={};length=None
    for arm in ARMS:
        fast=states[arm]
        r=dict(last_observation=count-1,first_affected_prediction=count,forecast_origins=ends,
            observed_targets=observed,gradient_steps=0,parameter_delta_norm=0.,summary_delta_norm=0.,requested_norm=0.)
        if not observed:
            # No arrived measurement, hence no update direction is available.
            out[arm]=fast;receipts[arm]=r;continue
        selected={k:v for k,v in fast.items() if arm=='matched_all' or k not in SUMMARY}
        loss=model.base.forecast_loss(model.predict_fast(x,m,fast)['forecast'],truth,mask)
        grads=torch.autograd.grad(loss,tuple(selected.values()),allow_unused=False)
        norm=sum(g.square().sum() for g in grads).sqrt()
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite signal update')
        if arm=='observation':
            factor=.001/norm.clamp_min(1.)
            length=float((factor*norm).detach())
        elif float(norm)>0:
            factor=length/norm
        else:
            # A zero gradient has no direction. Do not invent a matched vector.
            factor=norm.new_zeros(())
            r['zero_gradient_cannot_match']=True
        new={k:v.detach().requires_grad_(True) for k,v in fast.items()}
        for (k,v),g in zip(selected.items(),grads):new[k]=(v-factor.detach()*g).detach().requires_grad_(True)
        delta=float(sum((new[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt())
        summary=float(sum((new[k].detach()-fast[k].detach()).double().square().sum() for k in SUMMARY).sqrt())
        if arm=='observation':assert summary==0
        with torch.no_grad():after=model.base.forecast_loss(model.predict_fast(x,m,new)['forecast'],truth,mask)
        r.update(gradient_steps=int(delta>0),parameter_delta_norm=delta,summary_delta_norm=summary,
            requested_norm=length,gradient_norm=float(norm),loss_before=float(loss.detach()),loss_after=float(after))
        out[arm]=new;receipts[arm]=r
    return out,receipts


def predict(model,signal,*,device='cuda'):
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2 or signal['values'].shape[1]!=23
            or signal['values'].shape!=signal['available'].shape or signal['available'].dtype!=bool
            or not len(signal['values']) or model.cfg.window!=128 or model.cfg.horizon!=25):
        raise ValueError('W128/H25 flat23 signal-only stream required')
    states={a:initial_encoder(model) for a in ARMS};scores={a:[] for a in ARMS};receipts={a:[] for a in ARMS}
    n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for a in ARMS:scores[a].append(logit(model,x,m,states[a]).sigmoid().cpu().numpy())
        if 256<=hi<n:
            states,rs=update_pair(model,{k:v[:hi] for k,v in signal.items()},hi,states,device=device)
            for a in ARMS:receipts[a].append(rs[a])
    scores={a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in scores.values()):raise FloatingPointError('nonfinite probability')
    return scores,receipts
