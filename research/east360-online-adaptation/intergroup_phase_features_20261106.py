"""Physical-group timing from a single reached prefix, with exact paired controls."""

from itertools import combinations
from numbers import Integral

import numpy as np

from paired_diagnostic_features_20261106 import GROUPS


CHANNELS = tuple(c for _, c in GROUPS)
WINDOW = 64
MIN_COMMON = 49
LAGS = (4, 8, 16)
SHIFTS = (7, 11, 13)
EPS = 1e-12


def latest_longest(ok):
    boundaries = np.flatnonzero(np.diff(np.r_[False, ok, False]))
    lo, hi = boundaries[::2], boundaries[1::2]
    if not len(lo):
        return 0, 0
    length = hi - lo
    j = np.flatnonzero(length == length.max())[-1]
    return int(lo[j]), int(hi[j])


def prefix(values, available, center, scale, m):
    """Only inspect the last 64 samples ending at reached index m (inclusive)."""
    v, a = np.asarray(values), np.asarray(available)
    c, s = np.asarray(center), np.asarray(scale)
    if v.ndim != 2 or v.shape[1] != 23 or a.shape != v.shape or a.dtype != bool:
        raise ValueError("source values and Boolean mask must be [time,23]")
    if c.shape != (23,) or s.shape != (23,):
        raise ValueError("source normalizers must be 23-vectors")
    if not isinstance(m, Integral) or isinstance(m, (bool, np.bool_)) or not 63 <= m < len(v):
        raise ValueError("a reached 64-sample prefix is required")
    window = slice(m - WINDOW + 1, m + 1)
    signal, valid = [], []
    for g in CHANNELS:
        measured = a[window, g]
        x, mid, size = (np.asarray(q, dtype=np.float64) for q in (v[window, g], c[list(g)], s[list(g)]))
        if not np.isfinite(x[measured]).all() or not np.isfinite(mid).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("finite observed samples and positive source scales required")
        z = np.arcsinh((np.where(measured, x, mid) - mid) / size)
        observed = measured.all(axis=1)
        ok = observed[1:] & observed[:-1]
        activity = np.sqrt(np.mean(np.diff(z, axis=0)**2, axis=1))
        signal.append(np.where(ok, activity, 0.0))
        valid.append(ok)
    within = np.zeros((3, 3))
    for j, (group, mask) in enumerate(zip(signal, valid, strict=True)):
        z = group[mask]
        if len(z):
            within[j] = np.log1p(np.sqrt(np.mean(z*z))), np.log1p(z.std()), len(z) / (WINDOW-1)
    real, shifted = np.zeros(9), np.zeros(9)
    support = np.zeros(9, dtype=bool)
    lo, hi = latest_longest(np.logical_and.reduce(valid))
    n = hi-lo
    if n >= MIN_COMMON:
        data = np.stack(signal)[:, lo:hi]
        core = np.arange(16, n-16)
        for p, (i, j) in enumerate(combinations(range(len(CHANNELS)), 2)):
            x = data[i, core] - data[i, core].mean()
            nx = np.linalg.norm(x)
            for k, lag in enumerate(LAGS):
                left, right = data[j, core-lag], data[j, core+lag]
                y = np.stack((left-left.mean(), right-right.mean()))
                ny = np.linalg.norm(y, axis=1)
                if nx <= EPS or (ny <= EPS).any():
                    continue
                key = p * len(LAGS) + k
                r = (y @ x) / (nx * ny)
                q = (np.roll(y, SHIFTS[p], axis=1) @ x) / (nx * ny)
                real[key], shifted[key] = (r[1]-r[0])/2, (q[1]-q[0])/2
                support[key] = True
    return dict(within=within.ravel(), real=real, shifted=shifted, supported=support, common=n)
