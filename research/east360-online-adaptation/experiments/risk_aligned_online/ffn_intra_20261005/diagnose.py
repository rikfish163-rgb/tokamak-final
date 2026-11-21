"""Source TRAIN post-hoc decoder directions, never an online acceptance rule.

No data loading or parameter updates. The caller must verify TRAIN membership,
complete H25 maturity and later query ordering; tensors cannot certify provenance.
"""
import json
import math

import torch
from torch.nn import functional as F

from ..fast_update import normalized_step
from .loss import signal_components, weighted_signal_loss


def _gradient(value, delta, *, retain_graph=False):
    gradient = (torch.autograd.grad(value, delta, retain_graph=retain_graph, allow_unused=True)[0]
                if value.requires_grad else None)
    gradient = torch.zeros_like(delta) if gradient is None else gradient
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('nonfinite diagnostic gradient')
    return gradient


def _risk(model, qx, qa, delta, label, boundary):
    logits = model(qx, qa, delta)['logit']
    if logits.shape != (len(qx),) or not torch.isfinite(logits).all():
        raise FloatingPointError('one finite risk logit per later query required')
    peak = logits.max()
    return F.binary_cross_entropy_with_logits(peak - boundary, peak.new_tensor(label))


def _persistence(model, x, a, prediction, truth, observed):
    seen = a & torch.isfinite(x)
    positions = torch.arange(x.shape[1], device=x.device)[None, :, None]
    last = positions.masked_fill(~seen, -1).amax(1)
    anchor_valid = last >= 0
    raw = x.gather(1, last.clamp_min(0)[:, None, :]).squeeze(1)
    raw = torch.where(anchor_valid, raw, model.center)
    anchor = torch.asinh((raw - model.center) / model.scale)
    common = observed & torch.isfinite(truth) & anchor_valid[:, None]
    decoder = signal_components(prediction, truth, common, model.center, model.scale)
    baseline = signal_components(anchor[:, None].expand_as(truth), truth, common,
                                 model.center, model.scale)
    result = dict(anchor_valid=anchor_valid[0].tolist(), level_valid=decoder['valid'][0].tolist(),
                  difference_valid=decoder['difference_valid'][0].tolist())
    for name, parts in (('decoder', decoder), ('persistence', baseline)):
        for component, mask in (('level', 'valid'), ('difference', 'difference_valid')):
            result[f'{name}_{component}'] = [float(v) if valid else None for v, valid in
                                             zip(parts[component][0].detach(), parts[mask][0])]
    return result


@torch.enable_grad()
def diagnostic(model, x, a, truth, observed, qx, qa, label, boundary, step, radius):
    """JSON-safe zero-delta, one-event direction probe for a frozen eval model.

    Components are level[0:23], then difference[0:23], with exactly loss.py's
    normalization and .25 factor. Positive dot means local risk descent under
    negative signal-gradient motion. B is a box-weight first-order bound only;
    the actual finite step uses uniform weights, not risk-label-selected weights.
    Persistence errors use common mature masks AND a genuinely observed anchor;
    unavailable channel comparisons are null. No parameter .grad is populated.
    """
    label, boundary, step, radius = map(float, (label, boundary, step, radius))
    if label not in (0., 1.) or not all(map(math.isfinite, (boundary, step, radius))) or min(step, radius) <= 0:
        raise ValueError('binary label, finite boundary and positive finite step/radius required')
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('frozen eval model required; diagnostic never changes its mode')
    if (x.shape != (1, 128, 23) or a.shape != x.shape or a.dtype != torch.bool
            or truth.shape != (1, 25, 23) or observed.shape != truth.shape or observed.dtype != torch.bool
            or qx.ndim != 3 or qx.shape[1:] != x.shape[1:] or len(qx) == 0
            or qa.shape != qx.shape or qa.dtype != torch.bool):
        raise ValueError('one W128/H25 support and nonempty later query bag with boolean masks required')
    delta = model.initial_delta()
    if delta.ndim != 1 or not delta.requires_grad or not torch.isfinite(delta).all() or delta.detach().any():
        raise ValueError('initial_delta must be a finite, gradient-carrying zero vector')
    prediction = model(x, a, delta)['forecast']
    parts = signal_components(prediction, truth, observed, model.center, model.scale)
    valid = torch.cat((parts['valid'], parts['valid'] & parts['difference_valid']), 1)[0]
    signal = weighted_signal_loss(parts).sum()
    uniform = _gradient(signal, delta, retain_graph=True)
    risk = _risk(model, qx, qa, delta, label, boundary)
    r = _gradient(risk, delta).double()
    dots, norms, losses = [], [], []
    component_sum = torch.zeros_like(uniform, dtype=torch.float64)
    for index in range(46):
        gradient, value = torch.zeros_like(delta), delta.new_zeros(())
        if valid[index]:
            weights = prediction.new_zeros(1, 2, 23)
            weights.view(-1)[index] = 1.
            # Reuse the real loss: no separate denominator or difference formula.
            value = weighted_signal_loss(parts, weights).sum()
            gradient = _gradient(value, delta, retain_graph=True)
        gradient = gradient.double()
        component_sum += gradient
        dots.append(float(torch.dot(r, gradient)))
        norms.append(float(gradient.norm()))
        losses.append(float(value.detach()))
    updated = normalized_step(delta.detach()[None], torch.zeros_like(delta)[None],
                              uniform.detach()[None], step=step, radius=radius)[0].detach().requires_grad_()
    after = _risk(model, qx, qa, updated, label, boundary)
    result = dict(scope='SOURCE_POSTHOC_DIRECTION_NOT_ONLINE',
        provenance='caller must verify J-TEXT TRAIN membership, H25 maturity and later queries',
        component_order='level[0:23], difference[0:23]', component_valid=valid.tolist(),
        component_loss=losses, component_gradient_norm=norms, dotproducts=dots,
        boxmax_B=sum(2 * v if v >= 0 else .25 * v for v in dots),
        uniform_dot=float(torch.dot(r, uniform.double())),
        risk_gradient_norm=float(r.norm()), uniform_gradient_norm=float(uniform.double().norm()),
        decomposition_error_norm=float((component_sum - uniform.double()).norm()),
        uniform_signal_loss=float(signal.detach()), risk_before=float(risk.detach()),
        risk_after_uniform_step=float(after.detach()),
        risk_change_uniform_step=float((after - risk).detach()),
        delta_norm=float(updated.detach().double().norm()), step=step, radius=radius,
        persistence=_persistence(model, x, a, prediction.detach(), truth, observed))
    json.dumps(result, allow_nan=False)  # Reject nonfinite output, never serialize NaN/Infinity.
    return result
