"""Delayed finite-horizon negative feedback, conditional on live event status.

This is NOT a signal-only self-supervised loss. A caller must explicitly certify
that the event has not occurred as of the last observed timestamp. Never derive
that certificate from the length of an arbitrary acquisition file.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..feedback_gradient_20261023.geometry import lead_ms, trial_step
from ..history_tangent_20261023.learning import logit
from ..paired_feedback_20261022.model import initial_encoder

MODES = ('matured_negative', 'positive_replay', 'matured_plus_replay')


def combine_losses(parts):
    if not parts or not set(parts)<= {'negative','replay'}:
        raise ValueError('one or both named feedback components required')
    # Fixed coefficients in every arm: adding current-shot feedback must not
    # silently halve the historical-replay contribution.
    return .5*sum(parts.values())


def matured_indices(times, *, event_free_through):
    times = np.asarray(times)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or not np.all(np.diff(times)>0)):
        raise ValueError('finite strictly increasing observed timestamps required')
    if (event_free_through is None or not np.isfinite(event_free_through)
            or event_free_through != times[-1]):
        raise ValueError('explicit event-free status exactly at latest observation required')
    age = lead_ms(times, event_free_through)
    eligible = np.flatnonzero((age>=125.) & (age<253.) & (np.arange(len(times))>=127))
    if len(eligible)>8:
        eligible = eligible[np.linspace(0,len(eligible)-1,8).astype(int)]
    return eligible


def tensors(signal, indices, device):
    windows = [simple_data.window(signal,int(e),128) for e in indices]
    return tuple(torch.as_tensor(np.stack(v),device=device) for v in zip(*windows))


def adapt(model, prefix, fast, *, mode, replay, event_free_through, device):
    if (set(prefix)!={'values','available','times'} or mode not in MODES
            or prefix['values'].ndim!=2 or prefix['values'].shape[1]!=23
            or prefix['available'].shape!=prefix['values'].shape
            or prefix['available'].dtype!=bool or len(prefix['times'])!=len(prefix['values'])):
        raise ValueError('observed flat23 prefix with times only, and explicit mode required')
    ends = matured_indices(prefix['times'],event_free_through=event_free_through)
    if mode!='matured_negative' and (replay is None or len(replay)!=2 or len(replay[0])!=8):
        raise ValueError('fixed eight positive historical windows required')
    count = len(prefix['values'])
    r = dict(mode=mode, last_observation=count-1, first_affected_prediction=count,
             certified_event_free_through=float(event_free_through),
             negative_indices=ends.tolist() if mode!='positive_replay' else [],
             historical_positive_windows=0 if mode=='matured_negative' else 8,
             gradient_steps=0,parameter_delta_norm=0.)
    if not len(ends) and mode!='positive_replay':
        r['non_update_reason']='no_fully_observed_mature_negative_window'
        return fast,r
    x,m=tensors(prefix,ends,device) if len(ends) else (None,None)
    def objective(state):
        parts={}
        if mode!='positive_replay': parts['negative']=F.softplus(logit(model,x,m,state)).mean()
        if mode!='matured_negative': parts['replay']=F.softplus(-logit(model,*replay,state)).mean()
        return combine_losses(parts),parts
    loss,parts=objective(fast)
    gradients=torch.autograd.grad(loss,tuple(fast.values()))
    norm=sum(g.square().sum() for g in gradients).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite delayed-negative update')
    new=trial_step(fast,gradients)
    distance=sum((new[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    with torch.no_grad(): after,_=objective(new)
    r.update(gradient_steps=1,gradient_norm=float(norm),parameter_delta_norm=float(distance),
             loss_before=float(loss.detach()),loss_after=float(after),
             loss_parts={k:float(v.detach()) for k,v in parts.items()})
    return new,r


def predict_conditional(model, signal, replay, *, device='cuda'):
    """OFFLINE conditional simulation: every supplied packet is pre-event.

    The caller, not the adapter, owns this assumption. This routine does not
    infer event status from a raw file and is not the deployed online API.
    """
    if set(signal)!={'values','available','times'}:
        raise ValueError('signal-only record with physical timestamps required')
    states={a:initial_encoder(model) for a in ('static',)+MODES}
    scores={a:[] for a in states};receipts={a:[] for a in MODES}
    for lo in range(0,len(signal['values']),128):
        hi=min(lo+128,len(signal['values']))
        for start in range(lo,hi,32):
            x,m=tensors(signal,range(start,min(start+32,hi)),device)
            with torch.no_grad():
                for arm,state in states.items():
                    scores[arm].append(logit(model,x,m,state).sigmoid().cpu().numpy())
        if hi<256 or hi==len(signal['values']): continue
        prefix={k:v[:hi] for k,v in signal.items()}
        for arm in MODES:
            states[arm],r=adapt(model,prefix,states[arm],mode=arm,replay=replay,
                                event_free_through=float(prefix['times'][-1]),device=device)
            receipts[arm].append(r)
    curves={k:np.concatenate(v) for k,v in scores.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError('nonfinite risk curve')
    for arm in MODES: np.testing.assert_array_equal(curves['static'][:256],curves[arm][:256])
    return curves,receipts
