"""Historical risk sensitivity changes an arrived-signal update direction.

No risk output correction or test-outcome input. This is an empirical diagonal
preconditioner, not an exact Fisher matrix or a reproduction of EATA.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..paired_feedback_20261022.model import initial_encoder

ARMS = ('importance', 'shuffled', 'omit_uncertain_targets')
OMITTED = (0, 12, 13)  # Bt, ne_nG, qa_proxy: targets only, never inputs.


def select_endpoints(shot):
    valid, y = np.asarray(shot['valid']), np.asarray(shot['y'])
    if valid.dtype != bool or valid.ndim != 1 or valid.shape != y.shape:
        raise ValueError('aligned historical labels and validity required')
    groups = [(valid & (y == k), 16) for k in (0, 1)] if shot['is_disrupt'] else [(valid, 32)]
    rows = []
    for mask, count in groups:
        ends = np.flatnonzero(mask)
        if not len(ends):
            raise ValueError('historical label stratum has no valid endpoints')
        # Repetition is intentional for a short positive stratum: retain its
        # declared mass without pretending that repeated points are independent.
        rows.extend(ends[np.linspace(0, len(ends)-1, count).astype(int)].tolist())
    return rows


def estimate_importance(model, shots, *, device):
    if not shots or len({s['shot'] for s in shots}) != len(shots):
        raise ValueError('unique authorized historical shots required')
    classes = {k:sum(bool(s['is_disrupt']) == k for s in shots) for k in (False, True)}
    if not all(classes.values()):
        raise ValueError('both historical outcome classes required')
    fast = initial_encoder(model)
    fisher = {k:torch.zeros_like(v, dtype=torch.float64) for k, v in fast.items()}
    receipts = []
    for s in shots:
        ends = select_endpoints(s)
        weight = .5/classes[bool(s['is_disrupt'])]/len(ends)
        for end in ends:
            x, mask = simple_data.window(s, end, model.cfg.window)
            x = torch.as_tensor(x[None], device=device)
            mask = torch.as_tensor(mask[None], device=device)
            h = model.predict_fast(x, mask, fast, mode='risk_state')['risk_state']
            z = model.base.risk_head(h).squeeze(-1)
            y = torch.as_tensor([s['y'][end]], dtype=z.dtype, device=device)
            loss = F.binary_cross_entropy_with_logits(z, y)
            gradients = torch.autograd.grad(loss, tuple(fast.values()))
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('nonfinite historical risk gradient')
            for (k, _), g in zip(fast.items(), gradients):
                fisher[k].add_(g.detach().double().square(), alpha=weight)
            receipts.append(dict(shot=int(s['shot']), end=int(end), label=float(y.item()),
                                 is_disrupt=bool(s['is_disrupt']), weight=weight))
    return fisher, receipts


def importance_weights(fisher):
    if not fisher or any(not torch.isfinite(f).all() or (f < 0).any() for f in fisher.values()):
        raise ValueError('finite nonnegative empirical gradient squares required')
    mean = sum(f.double().sum() for f in fisher.values())/sum(f.numel() for f in fisher.values())
    if mean <= 0:
        raise ValueError('all empirical gradient squares are zero')
    weights = {k:(1/(1+f.double()/mean)).float() for k, f in fisher.items()}
    rng = torch.Generator(device='cpu').manual_seed(131)
    shuffled = {}
    for k, a in weights.items():
        permutation = torch.randperm(a.numel(), generator=rng).to(a.device)
        shuffled[k] = a.flatten()[permutation].reshape_as(a)
    return weights, shuffled


def matched_direction(gradients, weights, *, reference_norm=None):
    if not gradients or gradients.keys() != weights.keys():
        raise ValueError('matching gradient and weight keys required')
    if any(g.shape != weights[k].shape or not torch.isfinite(g).all() or
           not torch.isfinite(weights[k]).all() or (weights[k] < 0).any()
           for k, g in gradients.items()):
        raise ValueError('finite gradients and matching nonnegative weights required')
    raw_norm = sum(g.square().sum() for g in gradients.values()).sqrt()
    direction = {k:g*weights[k] for k, g in gradients.items()}
    norm = sum(g.square().sum() for g in direction.values()).sqrt()
    reference = raw_norm if reference_norm is None else torch.as_tensor(reference_norm, device=norm.device)
    if not torch.isfinite(reference) or reference < 0:
        raise ValueError('finite nonnegative reference gradient norm required')
    zero = bool(norm == 0 or reference == 0)
    if zero:
        # No target information gives no update; do not revert to an unrelated
        # ordinary gradient when the declared direction is zero.
        direction = {k:torch.zeros_like(v) for k, v in direction.items()}
    else:
        factor = reference.clamp_max(1.)/norm
        direction = {k:v*factor for k, v in direction.items()}
    return direction, dict(raw_norm=float(raw_norm), reference_norm=float(reference),
                           weighted_norm=float(norm), zero_direction=zero)


def target_loss(base, prediction, truth, available, *, omit):
    if truth.shape != prediction.shape or available.shape != truth.shape or available.dtype != torch.bool:
        raise ValueError('forecast targets and masks must match')
    valid = available & torch.isfinite(truth)
    clean = torch.where(valid, truth.detach(), base.center)
    target = torch.asinh((clean-base.center)/base.scale)
    loss = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~valid, 0.)
    if omit:
        keep = torch.ones(23, device=prediction.device, dtype=prediction.dtype)
        keep[list(OMITTED)] = 0.
        loss = loss*keep
    # The ORIGINAL observed count remains the denominator, including the three
    # omitted columns. Do not increase the other columns' loss coefficients.
    return loss.sum()/valid.sum().clamp_min(1)


def adapt(model, signal, count, fast, arm, weights, *, device, learning_rate=1e-3):
    if arm not in ARMS or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('declared arm and positive learning rate required')
    x, mask, target, tm, ends = forecast_examples(signal, count, model.cfg, device)
    observed = int((tm & torch.isfinite(target)).sum())
    receipt = dict(last_observation=count-1, first_affected_prediction=count,
                   forecast_origins=ends, observed_targets=observed, gradient_steps=0,
                   parameter_delta_norm=0.)
    if not observed:
        receipt['reason'] = 'no_observed_targets'
        return fast, receipt
    prediction = model.predict_fast(x, mask, fast)['forecast']
    full_loss = target_loss(model.base, prediction, target, tm, omit=False)
    full = torch.autograd.grad(full_loss, tuple(fast.values()), retain_graph=arm == 'omit_uncertain_targets')
    full = dict(zip(fast, full))
    reference = sum(g.square().sum() for g in full.values()).sqrt()
    loss = full_loss
    if arm == 'omit_uncertain_targets':
        loss = target_loss(model.base, prediction, target, tm, omit=True)
        grads = dict(zip(fast, torch.autograd.grad(loss, tuple(fast.values()))))
        weight = {k:torch.ones_like(v) for k, v in fast.items()}
    else:
        grads, weight = full, weights[arm]
    direction, info = matched_direction(grads, weight, reference_norm=reference)
    if not torch.isfinite(loss) or not torch.isfinite(full_loss):
        raise FloatingPointError('nonfinite arrived feedback')
    updated = {k:(v-learning_rate*direction[k]).detach().requires_grad_(True) for k, v in fast.items()}
    delta = sum((updated[k].detach()-v.detach()).double().square().sum() for k, v in fast.items()).sqrt()
    with torch.no_grad():
        after = target_loss(model.base, model.predict_fast(x, mask, updated)['forecast'], target, tm,
                            omit=arm == 'omit_uncertain_targets')
    receipt.update(info, gradient_steps=int(not info['zero_direction']), parameter_delta_norm=float(delta),
                   loss_before=float(loss.detach()), loss_after=float(after), full_loss_before=float(full_loss.detach()))
    return updated, receipt


def predict(model, signal, weights, *, device='cuda', interval=128):
    if set(signal) != {'values', 'available'} or interval != 128:
        raise ValueError('signal-only input and fixed128 schedule required')
    fast = {a:initial_encoder(model) for a in ARMS}
    predictions, updates = {a:[] for a in ARMS}, {a:[] for a in ARMS}
    n = len(signal['values'])
    for lo in range(0, n, interval):
        hi = min(lo+interval, n)
        for start in range(lo, hi, 32):
            windows = [simple_data.window(signal, t, model.cfg.window) for t in range(start, min(start+32, hi))]
            x, mask = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
            # Match the archived predictor's autograd execution path. no_grad
            # selects a different fused Transformer path (regression measured
            # a 1.2e-7 difference), breaking bitwise pre-update comparability.
            for arm in ARMS:
                h = model.predict_fast(x, mask, fast[arm], mode='risk_state')['risk_state']
                predictions[arm].append(model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy())
        if hi >= model.cfg.window+model.cfg.horizon and hi < n:
            for arm in ARMS:
                fast[arm], receipt = adapt(model, signal, hi, fast[arm], arm, weights, device=device)
                updates[arm].append(receipt)
    result = {k:np.concatenate(v) for k, v in predictions.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite inference')
    return result, updates
