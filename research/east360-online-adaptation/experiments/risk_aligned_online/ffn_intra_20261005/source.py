"""Source TRAIN-only episodes for a signal-update rule, NOT a new classifier.

The outer objective explicitly targets a >30 to <=125 ms warning segment. This
auxiliary goal is different from the frozen source's max(25,AdvancedTime) point
labels. Episode losses are a screening objective, never whole-shot TPR/FPR.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..redesign_20261004.intra import _batch_windows
from .stream import signal_step

STRIDE=26  # H25 must mature before the next update origin; source records are short.


def split_shots(shots,seed=131):
    if any(s['device']!='J-TEXT' for s in shots):
        raise ValueError('offline JTEXT TRAIN only')
    rng=np.random.default_rng(seed)
    train,validation=[],[]
    for disruptive in (False,True):
        pool=[s for s in shots if s['is_disrupt']==disruptive]
        order=rng.permutation(len(pool))
        n=max(1,len(pool)//5)
        validation.extend(pool[i] for i in order[:n])
        train.extend(pool[i] for i in order[n:])
    return train,validation


def make_episode(shot,rng):
    if shot['device']!='J-TEXT':
        raise ValueError('source episodes cannot read EAST outcomes')
    # Use the SAME 0,26,... event clock as full-shot replay. The original model
    # permits left padding, but the temporary state may not reset at a late crop.
    earliest=3*STRIDE
    if shot['is_disrupt']:
        lead=np.round((shot['down']-shot['times'])*1000.,9)
        timely=np.flatnonzero(shot['valid'] & (lead>30.) & (lead<=125.))
        if not len(timely):
            return None
        last_origin=((int(timely[-1])-26)//STRIDE)*STRIDE
        queries=np.arange(last_origin+26,min(last_origin+52,len(shot['times'])))
        queries=queries[shot['valid'][queries] & (lead[queries]>30.) & (lead[queries]<=125.)]
    else:
        last=((len(shot['times'])-28)//STRIDE)*STRIDE
        if last<earliest:
            return None
        last_origin=int(rng.choice(np.arange(earliest,last+1,STRIDE)))
        queries=np.arange(last_origin+26,min(last_origin+52,len(shot['times'])))
        if not shot['valid'][queries].all():
            return None
    if last_origin<earliest or len(queries)<2:
        return None
    origins=np.arange(last_origin-3*STRIDE,last_origin+1,STRIDE)
    return dict(shot=shot['shot'],origins=origins.tolist(),queries=queries.tolist(),
                burn_in_origins=list(range(0,int(origins[0]),STRIDE)),
                label=int(shot['is_disrupt']))


def episode_loss(model,shot,episode,*,step,radius,rule=None,create_graph=False,adapt=True,boundary=0.):
    device=model.center.device
    delta=model.initial_delta()
    safe_logits=[]
    losses=[]
    # Burn in from the real beginning of this shot with the current rule.
    # Truncate meta-gradients only at the boundary of the four trained events.
    for origin in episode['burn_in_origins']:
        if adapt:
            x,a=_batch_windows(simple_data,{k:shot[k] for k in ('values','available')},[origin],model,device)
            truth=torch.as_tensor(shot['values'][origin+1:origin+26],device=device)[None]
            observed=torch.as_tensor(shot['available'][origin+1:origin+26],device=device)[None]
            delta,_,_=signal_step(model,delta,x,a,truth,observed,step=step,radius=radius,rule=rule)
            delta=delta.detach().requires_grad_()
    for origin in episode['origins']:
        x,a=_batch_windows(simple_data,{k:shot[k] for k in ('values','available')},[origin],model,device)
        truth=torch.as_tensor(shot['values'][origin+1:origin+26],device=device)[None]
        observed=torch.as_tensor(shot['available'][origin+1:origin+26],device=device)[None]
        if adapt:
            delta,loss,_=signal_step(model,delta,x,a,truth,observed,step=step,radius=radius,
                                    rule=rule,create_graph=create_graph)
            losses.append(float(loss.detach()))
            if not create_graph:
                delta=delta.detach().requires_grad_()
        # Include intermediate safe segments; no outer label is used inside
        # signal_step. These are source OFFLINE labels for learning the rule.
        if not episode['label']:
            ends=[origin+26]
            qx,qa=_batch_windows(simple_data,{k:shot[k] for k in ('values','available')},ends,model,device)
            safe_logits.append(model(qx,qa,delta)['logit'])
    qx,qa=_batch_windows(simple_data,{k:shot[k] for k in ('values','available')},
                         episode['queries'],model,device)
    logits=model(qx,qa,delta)['logit']
    if safe_logits:
        logits=torch.cat([*safe_logits,logits])
    peak=logits.max()
    loss=F.binary_cross_entropy_with_logits(peak-boundary,peak.new_tensor(float(episode['label'])))
    return loss,dict(shot=shot['shot'],label=episode['label'],peak_probability=float(peak.detach().sigmoid()),
        outer_segment_loss=float(loss.detach()),delta_norm=float(delta.detach().double().norm()),
        mean_signal_loss=float(np.mean(losses)) if losses else None)
