"""Observable short-time changes, not inferred equilibrium or disruption labels.

The decoder predicts relative changes in SIX recorded diagnostics. All 23
inputs still feed the parent. Each origin fixes its past-only reference and
scale; arriving target values cannot change those coordinates. This does not
calibrate instruments between devices or recover missing physical units.
"""
import torch

from .io23 import RAW23

DIAGNOSTICS = ('ip', 'dx_a', 'dy_a', 'Mir_avg_amp', 'sxr_mean', 'xuv_var')
INDICES = tuple(RAW23.index(name) for name in DIAGNOSTICS)


def coordinates(values, available, *, min_count=16):
    if values.ndim != 3 or values.shape[-1] != 23 or available.shape != values.shape:
        raise ValueError('past window and matching 23-channel mask required')
    if available.dtype != torch.bool or min_count < 2:
        raise ValueError('boolean availability and minimum count >=2 required')
    data = values[..., INDICES].detach().double()
    mask = available[..., INDICES] & torch.isfinite(data)
    count = mask.sum(1)
    clean = data.masked_fill(~mask, 0.)
    mean = clean.sum(1) / count.clamp_min(1)
    variance = (data-mean[:, None]).masked_fill(~mask, 0.).square().sum(1) / count.clamp_min(1)
    eligible = (count >= min_count) & (variance > 0) & mask[:, -1]
    # Invalid channels are masked out, not treated as normalized zero targets.
    # Unit scale here is only a finite placeholder for that excluded channel.
    scale = torch.where(eligible, variance.sqrt(), 1.)
    reference = data[:, -1].masked_fill(~eligible, 0.)
    return dict(reference=reference.to(values.dtype), scale=scale.to(values.dtype), eligible=eligible)


def trajectory_target(origin, truth, available):
    if truth.ndim != 3 or truth.shape[-1] != 23 or available.shape != truth.shape or available.dtype != torch.bool:
        raise ValueError('arrived full-window 23-channel truth and boolean mask required')
    data = truth[..., INDICES].detach()
    mask = available[..., INDICES] & torch.isfinite(data) & origin['eligible'][:, None]
    clean = torch.where(mask, data, origin['reference'][:, None])
    target = (clean-origin['reference'][:, None]) / origin['scale'][:, None]
    if not torch.isfinite(target).all():
        raise FloatingPointError('nonfinite relative diagnostic change')
    return target, mask


def trajectory_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('trajectory prediction, target and mask must match')
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none')
    return loss.masked_fill(~mask, 0.).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)


def linear_extrapolation(values, available, origin, *, horizon=25, length=16):
    """Least-squares past slope, a causal engineering baseline, not physics."""
    data = values[:, -length:, INDICES].detach().double()
    mask = available[:, -length:, INDICES] & torch.isfinite(data)
    count = mask.sum(1).clamp_min(1)
    time = torch.arange(-data.shape[1]+1, 1, device=data.device).double()[None, :, None]
    mean_time = (time*mask).sum(1)/count
    mean_value = data.masked_fill(~mask, 0.).sum(1)/count
    centered_time = (time-mean_time[:, None]).masked_fill(~mask, 0.)
    centered_value = (data-mean_value[:, None]).masked_fill(~mask, 0.)
    denominator = centered_time.square().sum(1)
    slope = (centered_time*centered_value).sum(1)/denominator.clamp_min(1.)
    normalized = slope.to(values.dtype)/origin['scale']
    future_time = torch.arange(1, horizon+1, device=values.device).to(values.dtype)
    return normalized[:, None]*future_time[None, :, None]
