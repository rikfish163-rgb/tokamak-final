"""Unit-invariant variograms and an exact complete-patch variance partition."""

import numpy as np

LAGS = (1, 2, 4, 8, 16, 32, 64)


def variogram(values, available):
    x, m = np.asarray(values, dtype=np.float64), np.asarray(available)
    if x.shape != (128,) or m.shape != x.shape or m.dtype != bool:
        raise ValueError("one full128-point channel and boolean observations required")
    m = m & np.isfinite(x)
    pairs = np.array([int((m[lag:] & m[:-lag]).sum()) for lag in LAGS], dtype=np.uint16)
    result = np.full(len(LAGS), np.nan)
    observed = x[m]
    if len(observed) < 64:
        return result, pairs
    variance = np.square(observed - observed.mean()).mean()
    if variance == 0:
        return result, pairs
    if not np.isfinite(variance):
        raise FloatingPointError("unrepresentable observed variance")
    for i, lag in enumerate(LAGS):
        if pairs[i] >= 32:
            mask = m[lag:] & m[:-lag]
            difference = x[lag:][mask] - x[:-lag][mask]
            result[i] = np.square(difference).mean() / (2 * variance)
    return result, pairs


def patch_retention(values, available):
    x, m = np.asarray(values, dtype=np.float64), np.asarray(available)
    if x.shape != (128,) or m.shape != x.shape or m.dtype != bool:
        raise ValueError("one full128-point channel and boolean observations required")
    valid = (m & np.isfinite(x)).reshape(16, 8).all(1)
    count = int(valid.sum())
    if count < 8:
        return None, count
    patches = x.reshape(16, 8)[valid]
    means = patches.mean(1)
    between = np.square(means - means.mean()).mean()
    within = np.square(patches - means[:, None]).mean()
    total = between + within
    return (float(between / total) if total > 0 else None), count
