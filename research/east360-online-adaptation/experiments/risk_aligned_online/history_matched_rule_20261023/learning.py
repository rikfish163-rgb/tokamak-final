"""Train signal-update weights on the same accumulated fast state as deployment.

The forward replay includes every earlier update in this shot. Gradients are
truncated to the last two updates; context summaries remain stopped covariates
as in the archived rule. This is NOT full-trajectory backpropagation.
"""
import torch
from torch.nn import functional as F

from ..contextual_feedback_20261023.learning import step
from ..history_tangent_20261023.learning import logit
from ..matured_negative_20261023.learning import tensors
from ..paired_feedback_20261022.model import initial_encoder


def replay(model, policy, signal, count, *, device, tail_steps=2, create_graph=False):
    if (set(signal) != {'values','available'} or type(count) is not int
            or count < 256 or count % 128 or len(signal['values']) != count
            or len(signal['available']) != count or type(tail_steps) is not int or tail_steps < 1):
        raise ValueError('exact observed prefix and positive matched clock required')
    counts = list(range(256,count+1,128))
    differentiated = counts[-tail_steps:] if create_graph else []
    fast = initial_encoder(model)
    receipts = []
    for seen in counts:
        prefix = {k:v[:seen] for k,v in signal.items()}
        fast,r = step(model,policy,prefix,seen,fast,device=device,
                      create_graph=seen in differentiated)
        receipts.append(r)
    return fast, dict(all_counts=counts,differentiated_counts=differentiated,
                      updates=receipts,context_gradient='stopped')


def query_loss(model, policy, s, e, *, device, tail_steps=2):
    count,end = e['count'],e['end']
    if (any(p.requires_grad for p in model.parameters()) or count % 128
            or not 256 <= count <= end < min(count+128,len(s['values']))
            or float(s['y'][end]) != e['label'] or not bool(s['valid'][end])):
        raise ValueError('frozen model and authorized matching ended episode required')
    signal = {k:s[k][:count] for k in ('values','available')}
    fast,r = replay(model,policy,signal,count,device=device,tail_steps=tail_steps,create_graph=True)
    if not r['updates'][-1]['gradient_steps']:
        raise ValueError('teaching endpoint lacks measured signal feedback')
    x,m = tensors(s,[end],device)
    x = x.to(next(iter(fast.values())).dtype)
    y = torch.tensor([e['label']],dtype=x.dtype,device=device)
    with torch.no_grad():
        before = F.binary_cross_entropy_with_logits(logit(model,x,m,initial_encoder(model)),y)
    after = F.binary_cross_entropy_with_logits(logit(model,x,m,fast),y)
    return after, dict(shot=int(s['shot']),**e,**r,before_bce=float(before),after_bce=float(after.detach()))


def train_policy(model, policy, optimizer, episodes, *, device):
    if not episodes:
        raise ValueError('at least one authorized historical episode required')
    previous = [p.detach().clone() for p in policy.parameters()]
    optimizer.zero_grad(set_to_none=True)
    losses,rows = [],[]
    for s,e in episodes:
        loss,row = query_loss(model,policy,s,e,device=device)
        losses.append(loss);rows.append(row)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite history-matched rule loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError('nonfinite history-matched policy')
    distance = sum((p.detach()-q).double().square().sum() for p,q in zip(policy.parameters(),previous)).sqrt()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),
                parameter_delta_norm=float(distance),episodes=rows)
