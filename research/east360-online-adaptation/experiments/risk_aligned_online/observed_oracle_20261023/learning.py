"""OFFLINE LABEL ORACLE: observed inputs but future-informed risk labels.

Never import this module into an online predictor. This is not a rigorous upper
bound. It measures a specific internal-parameter interface with an optimistic
supervision source, while retaining the old update clock and reset semantics.
"""
import numpy as np
import torch
from torch.nn import functional as F

from ..history_tangent_20261023.learning import logit
from ..matured_negative_20261023.learning import tensors
from ..paired_feedback_20261022.model import initial_encoder


def step(model, prefix, labels, valid, state, *, steps, device):
    count=len(prefix['values'])
    labels,valid=np.asarray(labels),np.asarray(valid)
    if (set(prefix)!={'values','available'} or count<256 or count%128
            or prefix['values'].shape!=prefix['available'].shape
            or labels.shape!=(count,) or valid.shape!=(count,) or valid.dtype!=bool
            or not np.isin(labels[valid],[0.,1.]).all() or steps not in (1,4)):
        raise ValueError('exact observed prefix, binary ORACLE labels, valid mask and declared budget required')
    ends=np.arange(count-113,count,16);ends=ends[valid[ends]]
    receipt=dict(scope='NONDEPLOYABLE_LABEL_ORACLE',last_observation=count-1,
        first_affected_prediction=count,supervised_ends=ends.tolist(),gradient_steps=0,parameter_delta_norm=0.)
    if not len(ends):
        return state,dict(receipt,reason='no_valid_supervision')
    dtype=next(iter(state.values())).dtype
    x,m=tensors(prefix,ends,device);x=x.to(dtype)
    y=torch.as_tensor(labels[ends],device=device,dtype=dtype)
    original={k:v.detach().clone() for k,v in state.items()};norms=[];before=None
    for _ in range(steps):
        loss=F.binary_cross_entropy_with_logits(logit(model,x,m,state),y)
        if before is None:before=float(loss.detach())
        gradients=torch.autograd.grad(loss,tuple(state.values()))
        norm=sum(g.square().sum() for g in gradients).sqrt()
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('nonfinite oracle step')
        factor=norm.clamp_min(1).reciprocal()
        state={k:(v-.001*factor*g).detach().requires_grad_(True)
               for (k,v),g in zip(state.items(),gradients)}
        norms.append(float(norm))
    with torch.no_grad():after=float(F.binary_cross_entropy_with_logits(logit(model,x,m,state),y))
    delta=sum((v.detach()-original[k]).double().square().sum() for k,v in state.items()).sqrt()
    receipt.update(gradient_steps=steps,loss_before=before,loss_after=after,
                   parameter_delta_norm=float(delta),gradient_norms=norms)
    return state,receipt


def predict(model,signal,labels,valid,*,device='cuda'):
    if set(signal)!={'values','available'}:
        raise ValueError('signal-only input plus explicitly separate oracle supervision')
    states={a:initial_encoder(model) for a in ('oracle1','oracle4')}
    scores={a:[] for a in states};receipts={a:[] for a in states}
    dtype=next(iter(states['oracle1'].values())).dtype;n=len(signal['values'])
    if len(labels)!=n or len(valid)!=n:raise ValueError('matched oracle time grids required')
    for lo in range(0,n,128):
        hi=min(n,lo+128)
        for start in range(lo,hi,32):
            x,m=tensors(signal,range(start,min(hi,start+32)),device)
            with torch.no_grad():
                for a in states:scores[a].append(logit(model,x.to(dtype),m,states[a]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        prefix={k:v[:hi] for k,v in signal.items()}
        for a in states:
            states[a],receipt=step(model,prefix,labels[:hi],valid[:hi],states[a],
                                   steps=1 if a=='oracle1' else 4,device=device)
            receipts[a].append(receipt)
    curves={a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):raise FloatingPointError('nonfinite oracle risk')
    return curves,receipts
