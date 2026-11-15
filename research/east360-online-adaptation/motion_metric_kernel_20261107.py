"""Differentiable motion-metric operations on already-observed tensors."""

import math

import torch
from torch import nn


class MotionMetric(nn.Module):
    def __init__(self, context_dim=69, channels=23):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.SiLU(),
            nn.Linear(32, channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        # At initialization the formula is 1 mathematically; tanh rounding
        # and later dtype conversion can change its last floating-point bits.
        nn.init.constant_(self.net[-1].bias, math.atanh(0.5))

    def forward(self, context):
        weights = 2 * torch.tanh(self.net(context.detach()))
        # Floating-point tanh can round to +/-1; keep the stated open bounds.
        limit = torch.nextafter(weights.new_tensor(2.0), weights.new_tensor(0.0))
        return weights.clamp(min=-limit, max=limit)


def potential(prediction, motion, mask, weights):
    """Return the negative, per-row observed mean of weighted motion alignment."""
    if prediction.ndim != 3 or prediction.shape[1:] != (25, 23):
        raise ValueError('prediction must have shape [B, 25, 23]')
    if motion.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError('motion and mask must have the prediction shape')
    if weights.shape != (prediction.shape[0], 23):
        raise ValueError('weights must have shape [B, 23]')
    if mask.dtype != torch.bool:
        raise TypeError('mask must be boolean')
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(motion[mask]).all():
        raise ValueError('observed prediction and motion must be finite')
    if not torch.isfinite(weights).all():
        raise ValueError('weights must be finite')

    # This local linear potential is not a nonnegative prediction MSE. For
    # weights=1, its prediction gradient equals that of
    # .5*MSE(prediction, hold+motion) - .5*MSE(prediction, hold), using the
    # same masked denominator in both terms; no independent p^2 term remains.
    observed_prediction = torch.where(mask, prediction, 0.0)
    observed_motion = torch.where(mask, motion, 0.0)
    total = (observed_prediction * observed_motion * weights[:, None, :]).sum(dim=(1, 2))
    count = mask.sum(dim=(1, 2)).clamp_min(1)
    return -total / count


def descent(gradient, step=0.35):
    """Make a per-row normalized descent update, with no zero-gradient direction."""
    if gradient.ndim != 3 or gradient.shape[1:] != (4, 64):
        raise ValueError('gradient must have shape [B, 4, 64]')
    norm = torch.linalg.vector_norm(gradient, dim=(1, 2), keepdim=True)
    denominator = norm.clamp_min(1e-12)
    # 1e-12 rounds to zero in float16; avoid 0/0 there as well. In other
    # dtypes, retain the clamp's derivative even when the gradient is zero.
    denominator = torch.where(denominator == 0, 1.0, denominator)
    return -step * (gradient / denominator)


def magnitude_descent(gradient, reference_norm, step=0.35):
    """Retain measured gradient magnitude below a fixed TRAIN-only scale."""
    if not math.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError('positive finite TRAIN gradient reference required')
    norm = torch.linalg.vector_norm(gradient, dim=(1, 2), keepdim=True)
    return -step * gradient / norm.clamp_min(reference_norm)


def match_state(state, reference):
    """Match each nonzero state's norm to its reference, preserving direction."""
    if state.ndim != 3 or state.shape[1:] != (4, 64) or reference.shape != state.shape:
        raise ValueError('state and reference must have matching shape [B, 4, 64]')
    state_norm = torch.linalg.vector_norm(state, dim=(1, 2), keepdim=True)
    reference_norm = torch.linalg.vector_norm(reference, dim=(1, 2), keepdim=True)
    denominator = torch.where(state_norm == 0, 1.0, state_norm)
    return (state / denominator) * reference_norm
