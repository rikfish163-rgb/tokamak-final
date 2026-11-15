"""Differentiable signal-based update; no outcome labels accepted by adapt()."""
import torch
from torch.nn import functional as F


def signal_loss(output, target, available):
    prediction = output['forecast']
    if prediction.shape != target.shape or available.shape != target.shape or available.dtype != torch.bool:
        raise ValueError('forecast, observed target and boolean mask must agree')
    valid = available & torch.isfinite(target) & output['channel_valid'][:, None]
    observed = torch.where(valid, target.detach(), output['reference_last'][:, None])
    truth = (observed - output['reference_last'][:, None]) / output['reference_scale'][:, None]
    if output['forecast_space'] == 'asinh':
        # An invertible, sign-preserving coordinate, fixed at forecast origin.
        # Prevents near-constant channels followed by large jumps from making
        # the controller's loss/(1+loss) input almost always saturated.
        truth = torch.asinh(truth)
    element = F.smooth_l1_loss(prediction, truth, reduction='none')
    count = valid.sum((1, 2))
    loss = torch.where(valid, element, 0.).sum((1, 2)) / count.clamp_min(1)
    # Zero-delta is the last-observation-carried-forward baseline, expressed in
    # exactly the same origin-fixed coordinates as the learned forecast.
    persistence = torch.where(valid, F.smooth_l1_loss(torch.zeros_like(truth), truth, reduction='none'), 0.)
    persistence = persistence.sum((1, 2)) / count.clamp_min(1)
    return loss, count, persistence


def adapt(model, values, available, target, target_available, *, fast=None,
          create_graph=False, control='learned', level_drop=None):
    if control not in ('learned', 'fixed'):
        raise ValueError('control must be learned or fixed')
    if fast is None:
        fast = model.initial_fast(len(values)).clone()
    if not create_graph:
        fast = fast.detach().requires_grad_(True)
    output = model(values, available, fast=fast, level_drop=level_drop)
    loss, count, persistence = signal_loss(output, target, target_available)
    features = torch.stack([loss.detach() / (1. + loss.detach()),
                            count.to(loss.dtype) / (model.cfg.horizon * model.cfg.channels),
                            output['variation']], -1)
    gate = (model.update_controller(features).squeeze(-1).sigmoid()
            if control == 'learned' else torch.ones_like(loss))
    gate = gate * (count > 0).to(gate.dtype)
    gradient, = torch.autograd.grad(loss.sum(), fast, create_graph=create_graph)
    grad_norm = (gradient.square().sum(-1) + 1e-12).sqrt()
    gradient = gradient * (model.cfg.grad_cap / grad_norm).clamp_max(1.)[:, None]
    candidate = fast - model.cfg.fast_lr * gate[:, None] * gradient
    origin = model.initial_fast(len(values))
    delta = candidate - origin
    delta_norm = (delta.square().sum(-1) + 1e-12).sqrt()
    updated = origin + delta * (model.cfg.fast_radius / delta_norm).clamp_max(1.)[:, None]
    audit = dict(signal_loss=loss.detach(), persistence_loss=persistence.detach(),
                 valid_count=count.detach(), gate=gate.detach(), gradient_norm=grad_norm.detach(),
                 fast_change=(updated - fast).detach().norm(dim=-1))
    return (updated if create_graph else updated.detach()), audit


def risk_loss(logits, y, y125):
    """Explicit source label and common-horizon auxiliary task, not new truth."""
    primary = F.binary_cross_entropy_with_logits(logits[:, 0], y)
    common = F.binary_cross_entropy_with_logits(logits[:, 1], y125)
    return primary + 0.5 * common
