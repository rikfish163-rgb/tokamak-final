"""Exactly additive channel contributions with the original global denominator."""

import torch
from torch.nn import functional as F

from ..gate_loss_audit_20261101.core import observed


def channel_losses(model, prediction, truth, hold, mask):
    if truth.shape != prediction.shape or hold.shape != prediction.shape:
        raise ValueError("matching forecast, signal and hold targets required")
    center, scale = model.base.center, model.base.scale
    clean = torch.where(mask, truth.detach(), center)
    constant = torch.where(mask, hold.detach(), center)
    target = torch.asinh((clean - center) / scale)
    control = torch.asinh((constant - center) / scale)
    p, y = observed(prediction, target, mask)
    _, h = observed(prediction, control, mask)
    count = mask.sum(1, keepdim=True).clamp_min(1)
    p = p - p.sum(1, keepdim=True) / count
    y = y - y.sum(1, keepdim=True) / count
    h = h - h.sum(1, keepdim=True) / count
    difference = F.smooth_l1_loss(p, y, reduction="none") - F.smooth_l1_loss(
        p, h, reduction="none"
    )
    return difference.masked_fill(~mask, 0).sum((0, 1)) / mask.sum().clamp_min(1)
