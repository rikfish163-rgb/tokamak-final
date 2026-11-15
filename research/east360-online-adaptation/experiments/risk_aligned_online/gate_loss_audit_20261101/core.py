"""Separate the temporal mean of a forecast's output gradient from its shape."""

import torch
from torch.nn import functional as F


def observed(prediction, target, mask):
    if (
        prediction.ndim != 3
        or prediction.shape != target.shape
        or mask.shape != prediction.shape
        or mask.dtype != torch.bool
        or prediction.device != target.device
        or mask.device != prediction.device
    ):
        raise ValueError("matching BxHxC forecast/target/mask required")
    if (
        not torch.isfinite(prediction[mask]).all()
        or not torch.isfinite(target[mask]).all()
    ):
        raise ValueError("finite observed predictions and targets required")
    return torch.where(mask, prediction, 0), torch.where(mask, target.detach(), 0)


def output_derivatives(prediction, target, mask):
    """Exact SmoothL1(beta=1) output derivative and linear DC/AC split.

    DC means a time-constant derivative within each channel, not a separable
    component of the nonlinear loss value. Unobserved channels contribute zero.
    """
    p, y = observed(prediction.detach(), target, mask)
    u = (p - y).clamp(-1, 1).masked_fill(~mask, 0) / mask.sum().clamp_min(1)
    dc = (u.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)) * mask
    return u, dc, u - dc


def centered_loss(prediction, target, mask):
    """Remove each forecast/target's observed temporal mean before SmoothL1."""
    p, y = observed(prediction, target, mask)
    count = mask.sum(1, keepdim=True).clamp_min(1)
    p = p - p.sum(1, keepdim=True) / count
    y = y - y.sum(1, keepdim=True) / count
    loss = F.smooth_l1_loss(p, y, reduction="none").masked_fill(~mask, 0)
    return loss.sum() / mask.sum().clamp_min(1)
