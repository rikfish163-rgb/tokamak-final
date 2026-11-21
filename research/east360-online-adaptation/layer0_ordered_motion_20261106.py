"""Order-odd arrival targets and causal, paired source representation differences."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fit_layer0_arrival_motion_20261106 import past_arrays, source_mask
import layer0_arrival_motion_20261106 as motion


OFFSET, SLOTS, MIN_ORIGIN = 25, 10, motion.W - 1 + 2 * 25


def block_motion(level, valid):
    """Return the independent half of the reversal-odd lag-five differences."""
    motion._check_values(level, valid, motion.H)
    if not torch.isfinite(level).masked_fill(~valid, True).all():
        raise ValueError("observed levels must be finite")
    z = level.masked_fill(~valid, 0.0)
    delta = z[:, motion.LAG :] - z[:, : -motion.LAG]
    pairs = valid[:, motion.LAG :] & valid[:, : -motion.LAG]
    use = pairs[:, :SLOTS] & pairs.flip(1)[:, :SLOTS]
    odd = (delta[:, :SLOTS] + delta.flip(1)[:, :SLOTS]) * 0.5
    return odd.masked_fill(~use, 0.0), use


def target(raw, measured, end, past, center, scale):
    normalized, valid = motion.target_terms(raw, measured, end, center, scale)
    signal, keep = block_motion(normalized, valid)
    if (
        past.shape != keep.shape
        or past.dtype != torch.bool
        or past.device != keep.device
    ):
        raise ValueError("paired past support must match target shape, type and device")
    keep = keep & past
    return signal.masked_fill(~keep, 0.0), keep


def history(x, a, center, scale):
    """Observed raw history, persistence baseline, and past/endpoint availability."""
    motion._check_values(x, a, motion.W)
    end = a[:, -1]
    center, scale = motion._normalization(center, scale, x, motion.C)
    past = x[:, -2 * OFFSET :]
    use = a[:, -2 * OFFSET :] & end[:, None]
    normalized = motion._coordinates(past, use, center, scale)
    local = (normalized[:, :-1] - normalized[:, -1:]).masked_fill(~use[:, :-1], 0.0)
    # Use the same categorical-channel exclusion and endpoint criterion for both blocks.
    block, support = motion.target_terms(
        x[:, -OFFSET:], a[:, -OFFSET:], end, center, scale
    )
    persist, mask = block_motion(block, support)
    return local.transpose(1, 2), persist, mask, end


def pair_windows(values, mask, origins, *, source):
    if (
        origins.ndim != 1
        or origins.dtype.kind not in "iu"
        or (origins < MIN_ORIGIN).any()
    ):
        raise ValueError("three complete ordered causal windows required")
    # Explicit int64 avoids unsigned wraparound when subtracting past offsets.
    origins = origins.astype(np.int64, copy=False)
    ix = origins[:, None] - OFFSET * np.arange(3, dtype=np.int64)[None, :]
    flat = ix.reshape(-1)
    x, a = past_arrays(values, mask, flat)
    if source:
        a = source_mask(a, np.repeat(np.arange(len(origins)), 3))
    shape = (len(origins), 3, motion.W, motion.C)
    return x.reshape(shape), a.reshape(shape)


def context_pair(base, x, a, action=None):
    """Encode current and two prior windows; optionally use one differentiable action for all."""
    if x.ndim != 4 or x.shape[1:] != (3, motion.W, motion.C) or x.shape != a.shape:
        raise ValueError("three aligned windows required")
    b = len(x)
    if action is not None:
        if action.shape != (b, motion.WIDTH):
            raise ValueError("one first-layer action per origin required")
        action = action.repeat_interleave(3, dim=0)
    z, _ = motion.features(base, x.flatten(0, 1), a.flatten(0, 1), action=action)
    z = z.reshape(b, 3, motion.WIDTH)
    return z[:, 0] - z[:, 1], z[:, 1] - z[:, 2]


def masked_loss(pred, truth, support):
    """Per-row beta=1 Huber on the exact shared support and its count."""
    if (
        pred.ndim != 3
        or pred.shape != truth.shape
        or pred.shape != support.shape
        or support.dtype != torch.bool
        or not (pred.device == truth.device == support.device)
        or pred.dtype != truth.dtype
    ):
        raise ValueError("paired ordered forecasts and boolean support required")
    p, y = [v.masked_fill(~support, 0.0) for v in (pred, truth)]
    if not (torch.isfinite(p).all() and torch.isfinite(y).all()):
        raise ValueError("supported forecasts and targets must be finite")
    count = support.sum((1, 2))
    error = F.smooth_l1_loss(p, y, reduction="none", beta=1.0).sum((1, 2))
    return error / count.clamp_min(1), count


class OrderedHead(nn.Module):
    """Channelwise local baseline plus a bias-free, order-odd contextual residual."""

    def __init__(self, scale):
        super().__init__()
        scale = torch.as_tensor(scale)
        if not scale.is_floating_point():
            scale = scale.float()
        if (
            scale.shape != (motion.WIDTH,)
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("positive finite context RMS required")
        self.register_buffer("scale", scale.detach().clone())
        self.local_weight = nn.Parameter(
            scale.new_zeros(motion.C, 2 * OFFSET - 1, SLOTS)
        )
        self.context = nn.Linear(
            motion.WIDTH,
            SLOTS * motion.C,
            bias=False,
            dtype=scale.dtype,
            device=scale.device,
        )
        nn.init.zeros_(self.context.weight)

    def local(self, x):
        if x.ndim != 3 or x.shape[1:] != (motion.C, 2 * OFFSET - 1):
            raise ValueError("past relative channel history required")
        return torch.einsum("bck,ckj->bjc", x, self.local_weight)

    def context_motion(self, ctx):
        return self.context(ctx / self.scale).reshape(len(ctx), SLOTS, motion.C)

    def match_stale(self, actual, prior):
        if (
            actual.shape != prior.shape
            or actual.ndim != 2
            or actual.shape[1] != motion.WIDTH
        ):
            raise ValueError("paired current and stale context differences required")
        ref, old = [(v / self.scale).norm(dim=1, keepdim=True) for v in (actual, prior)]
        # Zero history contains no stale direction; multiplying it by a finite factor leaves zero.
        denom = torch.where(old > 0, old, 1.0)
        return prior * (ref / denom)
