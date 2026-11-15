"""Positive source-learned channel weights for a signal-only inner loss."""
import torch
from torch.nn import functional as F


def weighted_loss(losses, mask, logits):
    if logits.shape != (23,) or losses.shape != mask.shape or losses.shape[-1] != 23:
        raise ValueError('23 channel weights and matching forecast mask required')
    weights = 23*torch.softmax(logits, dim=0)
    observed_weights = mask.to(losses.dtype)*weights
    return (losses.masked_fill(~mask, 0.)*weights).sum((1, 2))/observed_weights.sum((1, 2)).clamp_min(1e-12)


def forecast_loss(model, prediction, truth, available, logits):
    mask = available & torch.isfinite(truth)
    truth = torch.where(mask, truth.detach(), model.center)
    target = torch.asinh((truth-model.center)/model.scale)
    losses = F.smooth_l1_loss(prediction, target, reduction='none')
    return weighted_loss(losses, mask, logits).mean()
