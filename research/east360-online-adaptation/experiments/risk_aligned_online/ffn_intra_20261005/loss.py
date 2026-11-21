"""Signal-only learned loss weights; prototype, not evidence of online gain."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _observed_signals(prediction, truth, mask, center, scale):
    if prediction.ndim != 3 or prediction.shape[-1] != 23:
        raise ValueError('prediction must have shape [B, H, 23]')
    if truth.shape != prediction.shape or mask.shape != prediction.shape or mask.dtype != torch.bool:
        raise ValueError('truth and boolean mask must match prediction')
    if center.shape != (23,) or scale.shape != (23,):
        raise ValueError('center and scale must have shape [23]')
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('center and scale must be finite, with positive scale')
    observed = mask & torch.isfinite(truth)
    if (observed & ~torch.isfinite(prediction)).any():
        raise FloatingPointError('nonfinite prediction at an observed signal')
    # Sanitize before arithmetic: masking a NaN loss afterwards can poison gradients.
    target = torch.where(observed, truth.detach(), center)
    target = torch.asinh((target - center) / scale)
    return torch.where(observed, prediction, 0.), target, observed


def _masked_mean(values, mask):
    # Divide before summing so extreme finite values can cancel without overflow.
    count = mask.sum(1).clamp_min(1)
    return (values.masked_fill(~mask, 0.) / count[:, None]).sum(1)


def signal_components(prediction, truth, mask, center, scale):
    """Per-channel SmoothL1 in source-asinh coordinates, using finite observations.

    Forecasts are already normalized. Truth is detached; differences require
    both original adjacent samples. Empty channels retain a zero forecast graph.
    """
    prediction, target, observed = _observed_signals(prediction, truth, mask, center, scale)
    pairs = observed[:, 1:] & observed[:, :-1]
    level = F.smooth_l1_loss(prediction, target, reduction='none')
    difference = F.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1],
                                 target[:, 1:] - target[:, :-1], reduction='none')
    return dict(level=_masked_mean(level, observed), difference=_masked_mean(difference, pairs),
                valid=observed.any(1), difference_valid=pairs.any(1))


@torch.no_grad()
def context_features(parent_forecast, truth, mask, center, scale):
    """Detached [B, 92]: signed error, absolute error, truth change, availability.

    Each block contains 23 channels. Statistics are clipped to [-10, 10];
    float64 accumulation avoids float32 overflow before clipping/cancellation.
    Missing channels/pairs produce zero. No outcome or logit context is accepted.
    """
    prediction, target, observed = _observed_signals(
        parent_forecast.double(), truth.double(), mask, center.double(), scale.double())
    pairs = observed[:, 1:] & observed[:, :-1]
    residual = prediction - target
    context = torch.cat((_masked_mean(residual, observed),
                         _masked_mean(residual.abs(), observed),
                         _masked_mean((target[:, 1:] - target[:, :-1]).abs(), pairs),
                         observed.sum(1) / max(parent_forecast.shape[1], 1)), dim=-1)
    return context.clamp(-10., 10.).to(parent_forecast.dtype)


class SignalRule(nn.Module):
    """Bounded level/difference weights from the detached signal context."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(92, 32), nn.SiLU(), nn.Linear(32, 46))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(3. / 4.))

    def forward(self, context):
        logits = self.net(context).reshape(-1, 2, 23)
        # Equivalent to .25 + 1.75*sigmoid(logits), but starts at exactly 1 in float32.
        return 2. - 1.75 * torch.sigmoid(-logits)


def weighted_signal_loss(components, weights=None):
    """Per-example loss, normalized by level-valid channels, never weight sum."""
    valid = components['valid']
    level = components['level'].masked_fill(~valid, 0.)
    difference = components['difference'].masked_fill(~(valid & components['difference_valid']), 0.)
    if weights is not None:
        if weights.shape != (level.shape[0], 2, 23):
            raise ValueError('weights must have shape [B, 2, 23]')
        level = weights[:, 0] * level
        difference = weights[:, 1] * difference
    return (level + .25 * difference).sum(-1) / valid.sum(-1).clamp_min(1)
