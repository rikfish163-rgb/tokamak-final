"""Matured feature feedback; no labels accepted by the neural update.

This is a one-step diagnostic, not a continuous stream result. A frozen
teacher's later representation is an estimated state, not physical truth.
"""
import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..paired_feedback_20261022.model import initial_encoder

ARMS = ('advance', 'reverse', 'forecast')


def pairs(count):
    if type(count) is not int or count < 153:
        raise ValueError('128-point context and 25 arrived points required')
    return [(count-26-25*j, count-1-25*j) for j in range(4) if count-26-25*j >= 127]


def encode(model, signal, ends, fast, device):
    windows = [simple_data.window(signal, int(end), model.cfg.window) for end in ends]
    x, m = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
    return model.predict_fast(x, m, fast, mode='risk_state')['risk_state']


def step(model, signal, count, arm, *, device='cuda'):
    if (set(signal) != {'values', 'available'} or arm not in (*ARMS, 'same')
            or model.cfg.window != 128 or model.cfg.horizon != 25
            or type(count) is not int or count > len(signal['values'])
            or signal['values'].shape != signal['available'].shape
            or signal['available'].dtype != bool):
        raise ValueError('signal-only input, declared arm, and W128/H25 required')
    indices = pairs(count)
    # Enforce the boundary before any window or target is constructed.
    prefix = {k:v[:count] for k,v in signal.items()}
    observed = prefix['available'] & np.isfinite(prefix['values'])
    indices = [(a,b) for a,b in indices if observed[a-127:a+1].any()
               and observed[b-127:b+1].any() and observed[a+1:b+1].any()]
    fast = initial_encoder(model)
    receipt = dict(last_observation=count-1, first_affected_prediction=count,
                   earlier_ends=[a for a,b in indices], target_ends=[b for a,b in indices],
                   gradient_steps=0, parameter_delta_norm=0., loss_before=None, loss_after=None)
    if not indices:
        receipt['reason'] = 'no_observed_pairs'
        return fast, receipt, {k:torch.zeros_like(v) for k,v in fast.items()}
    if arm == 'forecast':
        x,m,target,tm,ends = forecast_examples(prefix,count,model.cfg,device)
        def objective(params):
            return model.base.forecast_loss(model.predict_fast(x,m,params)['forecast'],target,tm)
    else:
        early,late = [a for a,b in indices], [b for a,b in indices]
        student_ends,teacher_ends = (late,early) if arm == 'reverse' else (early,late)
        if arm == 'same': teacher_ends = student_ends
        teacher = encode(model,prefix,teacher_ends,initial_encoder(model),device).detach()
        def objective(params):
            h = encode(model,prefix,student_ends,params,device)
            return (h-teacher).square().mean()
    loss = objective(fast)
    grads = torch.autograd.grad(loss,tuple(fast.values()),allow_unused=False)
    norm = sum(g.square().sum() for g in grads).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite matured neural feedback')
    factor = (1./norm.clamp_min(1.)).detach()
    updated = {k:(v-.001*factor*g).detach().requires_grad_(True)
               for (k,v),g in zip(fast.items(),grads)}
    distance = sum((updated[k].detach()-v.detach()).double().square().sum()
                   for k,v in fast.items()).sqrt()
    # Use the same autograd execution path for before/after and teacher.
    after = objective(updated)
    receipt.update(gradient_steps=1,gradient_norm=float(norm),clip_factor=float(factor),
                   parameter_delta_norm=float(distance),loss_before=float(loss.detach()),
                   loss_after=float(after.detach()))
    return updated,receipt,{k:g.detach() for k,g in zip(fast,grads)}


def sample_queries(shot):
    candidates = np.flatnonzero(shot['valid'] & (np.arange(len(shot['times'])) >= 177))
    if shot['is_disrupt']:
        lead = np.round((shot['down']-shot['times'])*1000.,9)
        groups = [('disruptive_timely',candidates[(lead[candidates]>30)&(lead[candidates]<=125)],16),
                  ('disruptive_early',candidates[shot['y'][candidates]==0],16)]
    else:
        groups = [('safe',candidates,32)]
    rows=[]
    for group,ends,size in groups:
        if not len(ends): raise ValueError(f'no eligible {group} queries')
        chosen = ends[np.linspace(0,len(ends)-1,size).astype(int)]
        rows.extend(dict(query=int(q),count=int(q)-24,group=group) for q in chosen)
    return rows


def summarize(rows):
    result={}
    for arm in ARMS:
        per_shot=[]
        for group in ('safe','disruptive_timely','disruptive_early'):
            selected=[r for r in rows if r['arm']==arm and r['group']==group]
            for sid in sorted({r['shot'] for r in selected}):
                part=[r for r in selected if r['shot']==sid]
                per_shot.append(dict(shot=sid,group=group,n=len(part),
                    **{key:float(np.mean([r[key] for r in part])) for key in
                       ('bce_before','bce_after','cosine','gradient_dot','actual_delta')}))
        groups={}
        for group in ('safe','disruptive_timely','disruptive_early'):
            part=[r for r in per_shot if r['group']==group]
            if part:
                groups[group]=dict(shots=len(part),
                    **{key:float(np.mean([r[key] for r in part])) for key in
                       ('bce_before','bce_after','cosine','gradient_dot','actual_delta')})
        if not {'safe','disruptive_timely'}<=set(groups):
            raise ValueError('both safe and timely disruptive probes required')
        selected=[groups[g] for g in ('safe','disruptive_timely')]
        better=sum(r['bce_after']<r['bce_before'] for r in per_shot if r['group']=='disruptive_timely')
        result[arm]=dict(groups=groups,per_shot=per_shot,improved_disruptive_shots=int(better),
            balanced_before=float(np.mean([r['bce_before'] for r in selected])),
            balanced_after=float(np.mean([r['bce_after'] for r in selected])),
            passes_screen=bool(groups['disruptive_timely']['shots']==4 and better>=3
                and all(r['bce_after']<=r['bce_before'] for r in selected)
                and sum(r['bce_after'] for r in selected)<sum(r['bce_before'] for r in selected)))
    return result
