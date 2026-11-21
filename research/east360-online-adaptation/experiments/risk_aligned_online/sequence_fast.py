"""Differentiable causal multi-update episodes; labels only enter outer training.

Unlike independent one-step episodes, temporary state accumulates through the
sequence. Forecast targets mature at t+25; risk is evaluated at t+26. This is
an MT3-inspired temporal experiment, not a reproduction of image-domain MT3.
"""
import torch
from torch.nn import functional as F

from .fast_direction import episode
from .fast_update import normalized_step
from .simple_online import FAST_RADIUS, FAST_STEP
from .channel_loss import weighted_loss


def sequence(shot, origin, steps, cfg, device):
    if steps < 1:
        raise ValueError('positive unroll length required')
    return [episode(shot, origin+i*(cfg.horizon+1), cfg, device) for i in range(steps)]


def unroll(model, episodes, *, create_graph, channel_logits=None, loss_rule=None):
    if channel_logits is not None and loss_rule is not None:
        raise ValueError('choose one auxiliary weighting rule')
    if not episodes or model.cfg.fast_rule != 'normalized':
        raise ValueError('nonempty normalized-update sequence required')
    model.eval()
    initial = model.initial_fast(len(episodes[0]['x'])).detach()
    fast = initial.clone().requires_grad_(True)
    logits, signals, offsets = [], [], []
    for e in episodes:
        prediction = model(e['x'], e['a'], fast)['forecast']
        mask = e['mask'] & torch.isfinite(e['truth'])
        truth = torch.where(mask, e['truth'].detach(), model.center)
        target = torch.asinh((truth-model.center)/model.scale)
        losses = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~mask, 0.)
        per_episode = losses.sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
        if channel_logits is not None:
            per_episode = weighted_loss(losses, mask, channel_logits)
        if loss_rule is not None:
            per_episode = loss_rule(model, prediction, e['truth'], e['mask'])
        gradient, = torch.autograd.grad(per_episode.sum(), fast, create_graph=create_graph,
                                       retain_graph=create_graph)
        fast = normalized_step(fast, initial, gradient, step=FAST_STEP, radius=FAST_RADIUS)
        if not create_graph:
            fast = fast.detach().requires_grad_(True)
        logits.append(model(e['qx'], e['qa'], fast)['logit'])
        signals.append(per_episode.mean())
        offsets.append((fast-initial).norm(dim=-1))
    return dict(logits=torch.stack(logits), signal_loss=torch.stack(signals).mean(),
                offsets=torch.stack(offsets), fast=fast)
