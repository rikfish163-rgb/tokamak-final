"""Build future diagnostic windows from one direct forecast, not autoregression."""

import numpy as np


def decode_forecast(prediction, center, scale, latest, available):
    p = np.asarray(prediction, np.float64)
    center, scale, latest = np.asarray(center), np.asarray(scale), np.asarray(latest)
    available = np.asarray(available)
    if (
        p.shape != (25, 23)
        or any(v.shape != (23,) for v in (center, scale, latest, available))
        or available.dtype != bool
    ):
        raise ValueError(
            "direct25x23 forecast and known current channel state required"
        )
    mask = np.repeat(available[None], 25, axis=0)
    mask[:, 2] = False
    continuous = mask.copy()
    continuous[:, 10] = False
    if (
        not np.isfinite(p[continuous]).all()
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("finite valid forecasts and positive source scales required")
    # Missing channels are never invented. Discrete mode_number_n is observed
    # carry-forward, not a continuous forecast or an overflow workaround.
    z = np.where(continuous, p, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = center + scale * np.sinh(z)
    raw[:, 10] = latest[10] if available[10] else 0.0
    raw = np.where(mask, raw, 0.0)
    if not np.isfinite(raw).all() or (abs(raw) > np.finfo(np.float32).max).any():
        raise FloatingPointError("forecast cannot be represented; no clipping")
    return raw.astype(np.float32), mask


def future_windows(past, past_mask, forecast, forecast_mask):
    x, m, p, a = map(np.asarray, (past, past_mask, forecast, forecast_mask))
    if (
        x.shape != (128, 23)
        or m.shape != x.shape
        or p.ndim != 2
        or p.shape[1] != 23
        or not 1 <= len(p) <= 128
        or a.shape != p.shape
        or m.dtype != bool
        or a.dtype != bool
    ):
        raise ValueError("one causal128x23 past and a direct future block required")
    values = np.stack(
        [np.concatenate((x[j:], p[:j]), axis=0) for j in range(1, len(p) + 1)]
    )
    masks = np.stack(
        [np.concatenate((m[j:], a[:j]), axis=0) for j in range(1, len(p) + 1)]
    )
    return values.astype(np.float32), masks
