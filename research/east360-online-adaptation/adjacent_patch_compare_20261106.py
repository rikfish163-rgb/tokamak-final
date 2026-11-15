"""Small NumPy controls for the matched adjacent-patch comparison."""

import numpy as np

from h25_direction_geometry_20261106 import basis


LIMIT = 0.10
GUARD_HEADROOM = 2e-5
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
RADIUS = 0.35


def common_spaces(gradients):
    g = np.asarray(gradients)
    if g.ndim != 3 or g.shape[0] != 4:
        raise ValueError("paired true/reverse banks in both representations required")
    q = tuple(basis(rows) for rows in g)
    r = min(x.shape[1] for x in q)
    return tuple(x[:, :r] for x in q), r


def equal_rms_factors(sizes):
    x = np.asarray(sizes, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("finite nonnegative embedding RMS for the same four arms")
    common = x.min(1)
    supported = common > 1e-7
    scale = np.zeros_like(x)
    np.divide(common[:, None], x, where=supported[:, None], out=scale)
    return scale, supported


def common_step(shifts, supported):
    x, ok = np.asarray(shifts), np.asarray(supported)
    if (
        x.ndim != 5
        or x.shape[1:] != (4, len(SCALES), 2, 3)
        or ok.shape != (len(x),)
        or ok.dtype != bool
        or not np.isfinite(x).all()
    ):
        raise ValueError("paired true/reversed risk shifts, both signs and three times")
    fits = np.max(np.abs(x), axis=(1, 3, 4)) <= LIMIT - GUARD_HEADROOM
    return np.where(ok & fits.any(1), fits.argmax(1), -1)


def future_blocks(maturity, count):
    m = np.asarray(maturity)
    if (
        m.ndim != 1
        or m.dtype.kind not in "iu"
        or count < 0
        or (m < 63).any()
        or (m >= count).any()
        or ((m - 63) % 32).any()
        or (np.diff(m) != 32).any()
    ):
        raise ValueError(
            "complete, ordered sequence of reached 32-point clocks required"
        )
    return [np.arange(t + 1, min(t + 33, count), dtype=np.int64) for t in m]
