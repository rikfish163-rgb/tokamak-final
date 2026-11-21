"""Move only the first full-H25 update; never suppress early alarms.

The early/late single-target arms have the same number of gradient steps for
records longer than256 points. All subsequent update counts and losses match
the archived increment-forecast implementation. No outcome enters this file.
"""
import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import logit
from ..increment_forecast_20261023.learning import target_mask, update
from ..paired_feedback_20261022.model import initial_encoder

ARMS = ('static', 'late4', 'late1', 'early1')


def schedule(length, arm):
    if type(length) is not int or length < 1 or arm not in ARMS[1:]:
        raise ValueError('positive record length and declared intra arm required')
    first = 153 if arm == 'early1' else 256
    return ([first] if first < length else []) + list(range(384, length, 128))


def update_latest(model, prefix, count, fast, *, device):
    if set(prefix) != {'values', 'available'} or len(prefix['values']) != count:
        raise ValueError('exact observed signal prefix required')
    x, m, truth, truth_mask, ends = forecast_examples(prefix, count, model.cfg, device)
    x, m, truth, truth_mask = [v[:1] for v in (x, m, truth, truth_mask)]
    mask = target_mask(x, m, truth, truth_mask)
    receipt = dict(last_observation=count-1, first_affected_prediction=count,
        forecast_origins=ends[:1], observed_targets=int(mask.sum()),
        omitted_unobserved_last=int((truth_mask & ~mask).sum()),
        gradient_steps=0, parameter_delta_norm=0.)
    if not mask.any():
        # No measured target exists, so there is no supervised-by-signal gradient.
        return fast, receipt
    loss = model.base.forecast_loss(model.predict_fast(x, m, fast)['forecast'], truth, mask)
    grads = torch.autograd.grad(loss, tuple(fast.values()), allow_unused=False)
    norm = sum(g.square().sum() for g in grads).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite first-feedback loss or gradient')
    scale = (1. / norm.clamp_min(1.)).detach()
    new = {k:(v-.001*scale*g).detach().requires_grad_(True) for (k,v),g in zip(fast.items(),grads)}
    delta = sum((new[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    with torch.no_grad():
        after = model.base.forecast_loss(model.predict_fast(x,m,new)['forecast'],truth,mask)
    receipt.update(gradient_steps=1, parameter_delta_norm=float(delta), gradient_norm=float(norm),
                   loss_before=float(loss.detach()), loss_after=float(after))
    return new, receipt


def predict(model, signal, *, device='cuda'):
    if (set(signal) != {'values','available'} or signal['values'].ndim != 2
            or signal['values'].shape[1] != 23 or signal['available'].shape != signal['values'].shape
            or signal['available'].dtype != bool or not len(signal['values'])):
        raise ValueError('nonempty signal-only flat23 input required')
    if model.cfg.window != 128 or model.cfg.horizon != 25:
        raise ValueError('fixed W128/H25 experiment required')
    n = len(signal['values'])
    schedules = {a:set(schedule(n,a)) for a in ARMS[1:]}
    # Union of all update boundaries AND original32-example batch boundaries.
    # Same shapes and no_grad execution for all arms, including the static arm.
    boundaries = sorted({0,n,*range(32,n,32),*(c for v in schedules.values() for c in v)})
    states = {a:initial_encoder(model) for a in ARMS}
    scores = {a:[] for a in ARMS}; receipts = {a:[] for a in ARMS[1:]}
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        windows = [simple_data.window(signal,e,128) for e in range(lo,hi)]
        x,m = [torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        with torch.no_grad():
            for a in ARMS:
                scores[a].append(logit(model,x,m,states[a]).sigmoid().cpu().numpy())
        for a in ARMS[1:]:
            if hi in schedules[a]:
                prefix = {k:v[:hi] for k,v in signal.items()}
                fn = update_latest if (hi < 384 and a != 'late4') else update
                states[a], r = fn(model,prefix,hi,states[a],device=device)
                receipts[a].append(r)
    result = {a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite risk prediction')
    return result, receipts
