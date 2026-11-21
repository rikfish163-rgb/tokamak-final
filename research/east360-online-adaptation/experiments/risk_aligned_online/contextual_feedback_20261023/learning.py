"""Learn which arrived diagnostic errors should adapt the internal encoder.

Persistent classifier and diagnostic decoder are immutable. A post-shot risk
target reaches the policy only THROUGH a differentiable signal-gradient step.
This is not a risk-score weighting or expert-mixture module.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..feedback_selector_20261023.policy import make_context
from ..history_tangent_20261023.learning import logit
from ..increment_forecast_20261023.learning import target_mask
from ..matured_negative_20261023.learning import tensors
from ..paired_feedback_20261022.model import initial_encoder

ARMS = ('fixed', 'online', 'hold', 'mean')


class WeightPolicy(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        mean, scale = torch.as_tensor(mean).float(), torch.as_tensor(scale).float()
        if mean.shape != (185,) or scale.shape != (185,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('185 finite observed context coordinates required')
        self.register_buffer('center', mean)
        self.register_buffer('scale', scale)
        self.network = nn.Sequential(nn.Linear(185, 32), nn.Tanh(), nn.Linear(32, 23))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context, *, mean_context=False):
        if context.shape != (185,) or not torch.isfinite(context).all():
            raise ValueError('one finite causal context required')
        z = torch.zeros_like(context) if mean_context else ((context-self.center)/self.scale).clamp(-8, 8)
        return (2*self.network(z).tanh()).exp()


def observed(model, prefix, count, fast, *, device):
    if set(prefix) != {'values', 'available'} or len(prefix['values']) != count or count < 256:
        raise ValueError('exact arrived signal prefix, at least256 points required')
    dtype = next(iter(fast.values())).dtype
    x, m, truth, tm, origins = forecast_examples(prefix, count, model.cfg, device)
    x, truth = x.to(dtype), truth.to(dtype)
    mask = target_mask(x, m, truth, tm)
    prediction = model.predict_fast(x, m, fast)['forecast']
    clean = torch.where(mask, truth.detach(), model.base.center)
    target = torch.asinh((clean-model.base.center)/model.base.scale)
    cx, cm = tensors(prefix, [count-1], device)
    cx = cx.to(dtype)
    with torch.no_grad():
        current_logit = float(logit(model, cx, cm, fast)[0])
    valid = cm[0] & torch.isfinite(cx[0])
    level = torch.asinh((torch.where(valid, cx[0], model.base.center)-model.base.center)/model.base.scale)
    # forecast_examples returns newest first; context's old/new descriptors
    # require chronological order. No future window contributes to context.
    context = make_context(prediction.flip(0), target.flip(0), mask.flip(0), level, valid, current_logit)
    return prediction, target, mask, x, m, origins, context


def feedback(model, policy, prefix, count, fast, *, device, mode='real', learn_policy=False):
    if mode not in ('real', 'hold', 'mean'):
        raise ValueError('declared feedback control required')
    pred, target, mask, x, m, origins, context = observed(model, prefix, count, fast, device=device)
    c = torch.as_tensor(context, device=device, dtype=policy.center.dtype)
    weights = policy(c, mean_context=mode == 'mean')
    if not learn_policy:
        weights = weights.detach()
    receipt = dict(last_observation=count-1, first_affected_prediction=count,
                   forecast_origins=origins, observed_targets=int(mask.sum()),
                   weights=weights.detach().cpu().tolist(), gradient_steps=0, parameter_delta_norm=0.)
    if not mask.any():
        return None, dict(receipt, reason='no_measured_forecast_target')
    if mode == 'hold':
        clean = torch.where(m[:, -1:] & torch.isfinite(x[:, -1:]), x[:, -1:], model.base.center)
        target = torch.asinh((clean-model.base.center)/model.base.scale).expand_as(target)
    loss = F.smooth_l1_loss(pred, target.detach(), reduction='none').masked_fill(~mask, 0.)
    weighted = (loss*weights).sum()/(mask.to(loss.dtype)*weights).sum()
    return weighted, receipt


def step(model, policy, prefix, count, fast, *, device, mode='real', create_graph=False):
    loss, receipt = feedback(model, policy, prefix, count, fast, device=device,
                            mode=mode, learn_policy=create_graph)
    if loss is None:
        return fast, receipt
    gradients = torch.autograd.grad(loss, tuple(fast.values()), create_graph=create_graph)
    norm = sum(g.square().sum() for g in gradients).clamp_min(1e-30).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite contextual feedback')
    factor = norm.clamp_min(1).reciprocal()
    updated = {k:v-.001*factor*g for (k,v), g in zip(fast.items(), gradients)}
    if not create_graph:
        updated = {k:v.detach().requires_grad_(True) for k,v in updated.items()}
    distance = sum((v.detach()-fast[k].detach()).double().square().sum() for k,v in updated.items()).sqrt()
    receipt.update(gradient_steps=1, loss_before=float(loss.detach()),
                   gradient_norm=float(norm.detach()), parameter_delta_norm=float(distance))
    return updated, receipt


def policy_loss(model, policy, episodes, *, device):
    if not episodes or any(p.requires_grad for p in model.parameters()):
        raise ValueError('frozen classifier and authorized ended episodes required')
    losses, rows = [], []
    for s,e in episodes:
        count, end = e['count'], e['end']
        if not 256 <= count <= end < count+128 or float(s['y'][end]) != e['label']:
            raise ValueError('ordered ended support/query episode required')
        fast = initial_encoder(model)
        prefix = {k:s[k][:count] for k in ('values', 'available')}
        adapted, receipt = step(model, policy, prefix, count, fast, device=device, create_graph=True)
        if not receipt['gradient_steps']:
            raise ValueError('historical teaching episode lacks measured signal feedback')
        x,m = tensors(s, [end], device)
        x = x.to(next(iter(fast.values())).dtype)
        y = torch.tensor([e['label']], dtype=x.dtype, device=device)
        with torch.no_grad():
            before = F.binary_cross_entropy_with_logits(logit(model,x,m,fast),y)
        after = F.binary_cross_entropy_with_logits(logit(model,x,m,adapted),y)
        losses.append(after)
        rows.append(dict(shot=int(s['shot']), **e, before_bce=float(before), after_bce=float(after.detach()), **receipt))
    return torch.stack(losses).mean(), rows


def train_policy(model, policy, optimizer, episodes, *, device):
    previous = [v.detach().clone() for v in policy.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss, rows = policy_loss(model, policy, episodes, device=device)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite policy objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError('nonfinite learned policy')
    delta = sum((v.detach()-q).double().square().sum() for v,q in zip(policy.parameters(),previous)).sqrt()
    return dict(loss=float(loss.detach()), gradient_norm=float(norm), parameter_delta_norm=float(delta), episodes=rows)


def predict(model, fixed, online, signal, *, device='cuda'):
    if set(signal) != {'values', 'available'}:
        raise ValueError('signal-only current shot required')
    states = {arm:initial_encoder(model) for arm in ARMS}
    scores = {arm:[] for arm in ARMS}
    receipts = {arm:[] for arm in ARMS}
    n = len(signal['values'])
    dtype = next(iter(states['fixed'].values())).dtype
    for lo in range(0,n,128):
        hi = min(n,lo+128)
        for start in range(lo,hi,32):
            x,m = tensors(signal,range(start,min(hi,start+32)),device)
            with torch.no_grad():
                for arm in ARMS:
                    scores[arm].append(logit(model,x.to(dtype),m,states[arm]).sigmoid().cpu().numpy())
        if hi < 256 or hi == n:
            continue
        prefix = {k:v[:hi] for k,v in signal.items()}
        for arm in ARMS:
            policy = fixed if arm == 'fixed' else online
            mode = arm if arm in ('hold','mean') else 'real'
            states[arm], receipt = step(model,policy,prefix,hi,states[arm],device=device,mode=mode)
            receipts[arm].append(receipt)
    curves = {arm:np.concatenate(parts) for arm,parts in scores.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError('nonfinite risk')
    for arm in ARMS:
        np.testing.assert_array_equal(curves[arm][:256],curves['fixed'][:256])
    return curves,receipts
