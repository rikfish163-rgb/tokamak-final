"""Ordered historical retrieval and pairwise encoder learning.

No current outcome, output correction or change of the fixed risk head.
The primary candidate is trajectory_rank; all arms are endpoint-reset screens.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..history_retrieval_20261023.learning import normalize, query_key, select_memory
from ..history_tangent_20261023.learning import flat, logit, matched_step, apply_step
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('state_bce','state_rank','trajectory_bce','trajectory_rank','pooled_rank','reversed_rank')


def pairwise_loss(z, labels):
    if (z.shape!=(4,) or labels.shape!=z.shape or not torch.isfinite(z).all()
            or not ((labels==0)|(labels==1)).all() or int((labels==1).sum())!=2):
        raise ValueError('four finite scores with two positive and two negative historical labels required')
    return F.softplus(z[labels==0][:,None]-z[labels==1][None,:]).mean()


def trajectory_vectors(states):
    states=np.asarray(states,dtype=np.float64)
    if states.ndim!=2 or states.shape[0]!=3:raise ValueError('three ordered states required')
    unit=normalize(states)
    return dict(trajectory=normalize(unit.reshape(-1)),pooled=normalize(unit.mean(0)))


def observed_trajectory(model,signal,count,*,device):
    prefix={k:v[:count] for k,v in signal.items()}
    ends=[count-51,count-26,count-1]
    if ends[0]<127:raise ValueError('three complete causal windows required')
    windows=[simple_data.window(prefix,e,128) for e in ends]
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
    h=model.predict_fast(x,m,initial_encoder(model),mode='risk_state')['risk_state']
    return trajectory_vectors(h.detach().cpu().numpy()),ends


def step_all(model,signal,count,memory,*,excluded_shot=None,device='cuda'):
    if (set(signal)!={'values','available'} or type(count)!=int or count<256
            or count>len(signal['values']) or signal['values'].ndim!=2 or signal['values'].shape[1]!=23
            or signal['available'].shape!=signal['values'].shape or signal['available'].dtype!=bool
            or model.cfg.window!=128):
        raise ValueError('signal-only W128 prefix with at least256 observed points required')
    latest=query_key(model,signal,count,device=device)
    trajectory,ends=observed_trajectory(model,signal,count,device=device)
    origin=initial_encoder(model);states={};receipts={};gradients={};length=None
    for arm in ARMS:
        if arm.startswith('state_'):keys,key=memory['keys'],latest
        elif arm=='pooled_rank':keys,key=memory['pooled_keys'],trajectory['pooled']
        else:
            keys,key=memory['trajectory_keys'],trajectory['trajectory']
            if arm=='reversed_rank':key=key.reshape(3,-1)[::-1].reshape(-1)
        selected,distances=select_memory(dict(memory,keys=keys),key,'nearest',excluded_shot=excluded_shot)
        x,m=memory['x'][selected],memory['mask'][selected]
        labels=torch.as_tensor(memory['labels'][selected],device=device,dtype=x.dtype)
        fast={k:v.detach().clone().requires_grad_(True) for k,v in origin.items()}
        objective=F.binary_cross_entropy_with_logits if arm.endswith('_bce') else pairwise_loss
        before=logit(model,x,m,fast);loss=objective(before,labels)
        g=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach();norm=float(g.double().norm())
        if not torch.isfinite(g).all():raise FloatingPointError('nonfinite historical gradient')
        if arm=='state_bce':length=.001*norm/max(1.,norm)
        updated=apply_step(fast,matched_step(g,length))
        after=logit(model,x,m,updated);after_loss=objective(after,labels)
        if not torch.isfinite(after_loss):raise FloatingPointError('nonfinite history loss after update')
        delta=flat([fast[k].detach()-updated[k].detach() for k in fast])
        receipts[arm]=dict(selected_indices=selected,selected_labels=memory['labels'][selected].tolist(),
            selected_shots=memory['shots'][selected].tolist(),distances=distances,excluded_shot=excluded_shot,
            last_observation=count-1,first_affected_prediction=count,retrieval_ends=ends,
            loss_kind='BCE' if arm.endswith('_bce') else 'pairwise',
            gradient_steps=int(bool(delta.any())),gradient_norm=norm,requested_step_norm=length,
            parameter_delta_norm=float(delta.double().norm()),loss_before=float(loss.detach()),
            loss_after=float(after_loss.detach()),historical_logits_before=before.detach().cpu().tolist(),
            historical_logits_after=after.detach().cpu().tolist())
        states[arm]=updated;gradients[arm]=g
    return states,receipts,gradients
