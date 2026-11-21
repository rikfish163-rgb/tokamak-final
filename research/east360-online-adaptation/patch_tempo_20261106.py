"""Causal matching of observed feature time structure, not a plasma clock."""

import numpy as np


FIRST, STRIDE = 248, 32
INTERVALS = (8, 4, 16)
LAGS = (2, 4, 8, 16, 32)
CHANNELS = ("n=1 amplitude", "dx_a", "dy_a", "sxr_var", "xuv_var")


def gather(values, available, ends, interval):
    """Keep contiguous samples within patches and preserve the latest patch."""
    if interval not in INTERVALS:
        raise ValueError("prespecified patch interval required")
    ends = np.asarray(ends, dtype=np.int64)
    offsets = (-7 - np.arange(15, -1, -1)[:, None] * interval
               + np.arange(8)[None]).reshape(128)
    ix = ends[:, None] + offsets
    if np.any(ends < 0) or np.any(ends >= len(values)):
        raise ValueError("endpoints must already be observed")
    mask = available[ix.clip(0)] & (ix >= 0)[:, :, None]
    return np.where(mask, values[ix.clip(0)], 0), mask


def lag_profile(values, available):
    """Masked variogram divided by twice the observed within-window variance."""
    if values.ndim != 3 or values.shape[1] != 128 or values.shape != available.shape:
        raise ValueError("128-point windows and matching masks required")
    m = available & np.isfinite(values)
    x = np.where(m, values, 0).astype(np.float64)
    count = m.sum(1)
    mean = x.sum(1) / count.clip(1)
    centered = np.where(m, x - mean[:, None], 0)
    variance = np.square(centered).sum(1) / count.clip(1)
    result = np.full((len(x), len(LAGS), x.shape[-1]), np.nan)
    for j, lag in enumerate(LAGS):
        both = m[:, lag:] & m[:, :-lag]
        pairs = both.sum(1)
        numerator = np.where(both, np.square(x[:, lag:] - x[:, :-lag]), 0).sum(1)
        valid = (count >= 64) & (pairs >= 32) & (variance > 0)
        np.divide(numerator, pairs.clip(1) * 2 * variance,
                  out=result[:, j], where=valid)
    return result


def clock(values, available, reference):
    """Update shot-local EMA matching losses at releases; emit only future choices.

    Choice 0 is the original stride8, 1 stride4, and 2 stride16. Releases at t
    use only samples through t-1. Insufficient common support resets the EMA
    and selects the original window for both arms, rather than imputing a clock.
    """
    if values.shape != available.shape or reference.shape != (3, values.shape[-1]):
        raise ValueError("matching proxy channels and three-lag reference required")
    n = len(values)
    anchors = np.arange(FIRST, n, STRIDE)
    true = np.zeros(n, dtype=np.uint8)
    delayed = np.zeros_like(true)
    if not len(anchors):
        return true, delayed, np.zeros(0, dtype=np.int16)
    x, m = gather(values, available, anchors - 1, 8)
    profile = lag_profile(x, m)
    finite = np.isfinite(profile).all(1) & np.isfinite(reference).all(0)[None]
    support = finite.sum(1).astype(np.int16)
    ema, decisions = None, []
    for j, t in enumerate(anchors):
        if support[j] < 3:
            ema, decision, stale = None, 0, 0
        else:
            keep = finite[j]
            expected = np.sqrt(reference[:, keep])
            profiles = np.stack([profile[j, s:s+3][:, keep] for s in (1, 0, 2)])
            loss = np.square(np.sqrt(profiles) - expected).mean((1, 2))
            ema = loss if ema is None else 0.5 * (ema + loss)
            decision = int(np.argmin(ema))
            stale = decisions[j-4] if j >= 4 else decision
        decisions.append(decision)
        true[t:min(t+STRIDE, n)] = decision
        delayed[t:min(t+STRIDE, n)] = stale
    return true, delayed, support
