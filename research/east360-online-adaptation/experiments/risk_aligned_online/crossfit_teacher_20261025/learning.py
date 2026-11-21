"""One signal-feedback policy learned across four shot-disjoint teachers.

The teachers are training instruments, not a deployed ensemble. The deployed
parent, threshold and finite neural update are exactly the previous experiment.
"""
import numpy as np
import torch

from ..finite_conditional_loss_20261025.learning import LossPolicy
from ..operational_bag_loss_20261025.learning import (
    evaluate_policy, historical_gate, predict as previous_predict,
    eligible as previous_eligible,
)

RENAME=dict(off='off',stage_old='insample',bag_global='crossfit_global',
            bag_context='crossfit_context',uniform='uniform',permuted='permuted',hold='hold')
ARMS=tuple(v for k,v in RENAME.items() if k!='off')


def evaluate_teachers(teachers,policy,threshold,*,device='cuda'):
    if len(teachers)!=4:raise ValueError('exactly four equal-size teaching folds required')
    seen=set()
    for teacher in teachers:
        shots=teacher['shots']
        if len(shots)!=5 or sum(bool(s['flag']) for s in shots.values())!=1:
            raise ValueError('each held fold must contain1 disruptive and4 safe shots')
        if seen.intersection(shots):raise ValueError('held teaching shots must be disjoint')
        seen.update(shots)
    losses=[];gradients=[];details=[]
    counts={k:dict(tp=0,fp=0,very_early_over3000=0) for k in ('current','baseline')}
    for teacher in teachers:
        loss,g,detail,gate=evaluate_policy(teacher['model'],policy,teacher['shots'],threshold,device=device)
        losses.append(loss);gradients.append(g);details.extend(detail)
        for kind in counts:
            for key in counts[kind]:counts[kind][key]+=gate[kind][key]
    # Equal fold sizes and class composition make this identical to an equal
    # shot, then equal class reduction over all20 unique held-out shots.
    objective=float(np.mean(losses));gradient=np.mean(gradients,axis=0)
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError('nonfinite crossfit policy objective or derivative')
    counts['passed']=historical_gate(details,counts['current'],counts['baseline'])
    return objective,gradient,details,counts


def train_policy(teachers,kind,threshold,*,device='cuda',progress=None):
    policy=LossPolicy(kind,device=device);optimizer=torch.optim.Adam(policy.parameters(),lr=.05)
    logs=[];best=None
    for epoch in range(11):
        loss,gradient,detail,gate=evaluate_teachers(teachers,policy,threshold,device=device)
        record=dict(epoch=epoch,objective=loss,per_shot=detail,history_gate=gate,
            strength=float(policy.strength.detach().sigmoid()),gradient_norm=float(np.linalg.norm(gradient)),
            policy=policy.export())
        logs.append(record)
        if best is None or loss<best['objective']:best=record
        if progress is not None:progress(kind,record)
        if epoch==10:break
        offset=0
        for parameter in policy.parameters():
            parameter.grad=torch.as_tensor(gradient[offset:offset+parameter.numel()],
                device=device,dtype=parameter.dtype).reshape_as(parameter).clone()
            offset+=parameter.numel()
        assert offset==len(gradient)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    return best,logs


def rename_arms(payload):
    if set(payload)!=set(RENAME):raise ValueError('all seven original arms required')
    return {RENAME[k]:v for k,v in payload.items()}


def predict(model,signal,records,insample,*,device='cuda'):
    curves,updates=previous_predict(model,signal,records,insample,device=device)
    return rename_arms(curves),{RENAME[k]:v for k,v in updates.items()}


def eligible(results,history_passed):
    return previous_eligible({old:results[new] for old,new in RENAME.items()},history_passed)
