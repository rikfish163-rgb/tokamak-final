"""Conditional matured-window ranking; frozen risk head, ephemeral encoder only."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..feedback_gradient_20261023.geometry import trial_step
from ..history_tangent_20261023.learning import flat,logit,matched_step,apply_step
from ..matured_negative_20261023.learning import matured_indices,tensors
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('rank_current','bce_matched','rank_history_matched')


def relative_loss(negative,positive):
    if (negative.ndim!=1 or positive.ndim!=1 or not len(negative) or not len(positive)
            or not torch.isfinite(negative).all() or not torch.isfinite(positive).all()):
        raise ValueError('two nonempty finite score vectors required')
    return F.softplus(negative[:,None]-positive[None,:]).mean()


def step(fast,gradients,*,length):
    if len(gradients)!=len(fast) or any(a.shape!=b.shape for a,b in zip(gradients,fast.values())):
        raise ValueError('gradient/parameter shapes must match')
    g=flat(gradients).detach();norm=float(g.double().norm())
    if not math.isfinite(norm):raise FloatingPointError('nonfinite gradient')
    if length is None:new=trial_step(fast,gradients)
    else:
        if norm==0 and length>0:raise ValueError('cannot match a nonzero step without a direction')
        new=apply_step(fast,matched_step(g,length))
    actual=float(flat([new[k].detach()-v.detach() for k,v in fast.items()]).double().norm())
    return new,actual,norm


def update(model,prefix,states,historical_negative,positive,*,event_free_through,device):
    if (set(prefix)!={'values','available','times'} or set(states)!=set(ARMS)
            or prefix['values'].ndim!=2 or prefix['values'].shape[1]!=23
            or prefix['available'].shape!=prefix['values'].shape or prefix['available'].dtype!=bool
            or len(prefix['times'])!=len(prefix['values'])):
        raise ValueError('flat23 observed prefix and named ephemeral states required')
    ends=matured_indices(prefix['times'],event_free_through=event_free_through)
    for buffer in (historical_negative,positive):
        if buffer is None or len(buffer)!=2 or buffer[0].shape!=(8,128,23) or buffer[1].shape!=buffer[0].shape:
            raise ValueError('eight historical windows and masks required')
    count=len(prefix['times']);rows={};newstates={};length=None
    current=tensors(prefix,ends,device) if len(ends) else None
    for arm in ARMS:
        base=dict(first_affected_prediction=count,last_observation=count-1,
                  certified_event_free_through=float(event_free_through),
                  negative_indices=ends.tolist() if arm!='rank_history_matched' else [],
                  step_length_reference='self' if arm=='rank_current' else 'rank_current',
                  gradient_steps=0,parameter_delta_norm=0.)
        if current is None:
            newstates[arm]=states[arm];rows[arm]=dict(base,non_update_reason='no_matured_complete_window');continue
        negative=historical_negative if arm=='rank_history_matched' else current
        def objective(state):
            zn=logit(model,*negative,state);zp=logit(model,*positive,state)
            loss=.5*(F.softplus(zn).mean()+F.softplus(-zp).mean()) if arm=='bce_matched' else relative_loss(zn,zp)
            return loss,F.softplus(-zp)
        loss,posbefore=objective(states[arm])
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
        g=torch.autograd.grad(loss,tuple(states[arm].values()))
        new,actual,norm=step(states[arm],g,length=length)
        if arm=='rank_current':length=actual
        else:
            if abs(actual-length)>2e-7:raise AssertionError('matched step differs beyond float32 tolerance')
        with torch.no_grad():after,posafter=objective(new)
        if not torch.isfinite(after):raise FloatingPointError('nonfinite updated objective')
        newstates[arm]=new
        rows[arm]=dict(base,gradient_steps=1,parameter_delta_norm=actual,reference_step_length=length,
                      gradient_norm=norm,loss_before=float(loss.detach()),loss_after=float(after),
                      positive_bce_before=posbefore.detach().cpu().tolist(),positive_bce_after=posafter.cpu().tolist())
    return newstates,rows


def predict(model,signal,historical_negative,positive,*,device='cuda'):
    """Offline conditional replay, NOT an event detector or deployment status source."""
    if set(signal)!={'values','available','times'}:raise ValueError('observed signal only')
    states={a:initial_encoder(model) for a in ARMS};origin=initial_encoder(model)
    scores={a:[] for a in ('static',)+ARMS};rows={a:[] for a in ARMS}
    for lo in range(0,len(signal['times']),128):
        hi=min(lo+128,len(signal['times']))
        for start in range(lo,hi,32):
            x,m=tensors(signal,range(start,min(start+32,hi)),device)
            with torch.no_grad():
                for arm,state in dict(static=origin,**states).items():scores[arm].append(logit(model,x,m,state).sigmoid().cpu().numpy())
        if hi<256 or hi==len(signal['times']):continue
        prefix={k:v[:hi] for k,v in signal.items()}
        states,rs=update(model,prefix,states,historical_negative,positive,
                         event_free_through=float(prefix['times'][-1]),device=device)
        for arm in ARMS:rows[arm].append(rs[arm])
    curves={k:np.concatenate(v) for k,v in scores.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):raise FloatingPointError('nonfinite risk curve')
    for arm in ARMS:np.testing.assert_array_equal(curves['static'][:256],curves[arm][:256])
    return curves,rows
