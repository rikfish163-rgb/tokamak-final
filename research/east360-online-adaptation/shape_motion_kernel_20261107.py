"""Level and eight-point shape potentials on already-observed tensors."""

import math

import torch
from torch import nn

from motion_metric_kernel_20261107 import potential as level_potential


class ShapeMetric(nn.Module):
    def __init__(self, context_dim=69, channels=23):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 2 * channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.constant_(self.net[-1].bias[:channels], math.atanh(0.5))

    def forward(self, context):
        weights = 2 * torch.tanh(self.net(context.detach()))
        # Keep the open interval even when tanh rounds to exactly +/-1.
        limit = torch.nextafter(weights.new_tensor(2.0), weights.new_tensor(0.0))
        return weights.clamp(min=-limit, max=limit).reshape(-1, 2, self.channels)


def _lagged_motion(prediction, motion, mask):
    # Clear unobserved values before subtraction: NaNs cannot be masked away
    # safely after they have entered the differences and their gradients.
    prediction = torch.where(mask, prediction, 0.0)
    motion = torch.where(mask, motion, 0.0)
    pair_mask = mask[:, 8:] & mask[:, :-8]
    prediction_diff = torch.where(pair_mask, prediction[:, 8:] - prediction[:, :-8], 0.0)
    motion_diff = torch.where(pair_mask, motion[:, 8:] - motion[:, :-8], 0.0)
    return prediction_diff, motion_diff, pair_mask


def lagged_motion(prediction, motion, mask):
    """Return eight-point differences and their pair mask; invalid pairs are zero."""
    if prediction.ndim != 3 or prediction.shape[1:] != (25, 23):
        raise ValueError('prediction must have shape [B, 25, 23]')
    if motion.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError('motion and mask must have the prediction shape')
    if mask.dtype != torch.bool:
        raise TypeError('mask must be boolean')
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(motion[mask]).all():
        raise ValueError('observed prediction and motion must be finite')
    return _lagged_motion(prediction, motion, mask)


def potential(prediction, motion, mask, weights, shape_scale):
    """Return the level potential plus the per-row masked shape potential."""
    if weights.ndim != 3 or weights.shape[1:] != (2, 23):
        raise ValueError('weights must have shape [B, 2, 23]')
    if not isinstance(shape_scale, float) or not math.isfinite(shape_scale) or shape_scale <= 0:
        raise ValueError('shape_scale must be a positive finite float')

    # The existing level potential validates the shared input and level weights.
    level = level_potential(prediction, motion, mask, weights[:, 0])
    if not torch.isfinite(weights[:, 1]).all():
        raise ValueError('weights must be finite')
    prediction_diff, motion_diff, pair_mask = _lagged_motion(prediction, motion, mask)
    total = (prediction_diff * motion_diff * weights[:, 1, None, :]).sum(dim=(1, 2))
    count = pair_mask.sum(dim=(1, 2)).clamp_min(1)
    return level - shape_scale * (total / count)
