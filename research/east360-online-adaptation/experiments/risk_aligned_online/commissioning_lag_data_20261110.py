"""Causal, phase-controlled cross-group lag signatures and paired feedback."""

import numpy as np


GROUPS = (0, 0, 0, 1, 1, 3, 2, 2, 2, 2)
LAGS = (4, 8)
PHASES = ((0, 0, 0, 0), (0, 6, 12, 18), (0, -6, -12, -18))

_WINDOW = 25
_MIN_NORM = 1e-4
_BATCH = 128


def _inputs(z, mask, ends, groups):
    z = np.asarray(z)
    mask = np.asarray(mask, dtype=bool)
    if z.ndim != 2 or mask.shape != z.shape:
        raise ValueError('z and mask must have the same two-dimensional shape')

    ends = np.asarray(ends)
    if ends.ndim != 1 or (ends.size and not np.issubdtype(ends.dtype, np.integer)):
        raise ValueError('ends must be a one-dimensional sequence of indices')
    ends = ends.astype(np.int64, copy=False)
    if np.any((ends >= _WINDOW) & (ends >= len(z))):
        raise IndexError('window end is outside the signal')

    groups = np.asarray(groups)
    if (groups.ndim != 1 or len(groups) != z.shape[1]
            or (groups.size and not np.issubdtype(groups.dtype, np.integer))
            or np.any((groups < 0) | (groups > 3))):
        raise ValueError('groups must assign each channel an integer group from 0 to 3')
    groups = groups.astype(np.intp, copy=False)
    left, right = np.triu_indices(z.shape[1], 1)
    cross_group = groups[left] != groups[right]
    return z, mask, ends, groups, left[cross_group], right[cross_group]


def _phases(phases, groups):
    phases = np.asarray(phases)
    needed = int(groups.max()) + 1 if groups.size else 0
    if (phases.ndim != 1 or len(phases) < needed
            or (phases.size and not np.issubdtype(phases.dtype, np.integer))):
        raise ValueError('phases must provide an integer shift for each group')
    return phases


def _units(z, mask, ends):
    """Return normalized 25-increment windows and channel availability."""
    channels = z.shape[1]
    units = np.zeros((len(ends), channels, _WINDOW), dtype=np.float64)
    valid = np.zeros((len(ends), channels), dtype=bool)
    rows = np.flatnonzero(ends >= _WINDOW)
    if not rows.size:
        return units, valid

    indices = ends[rows, None] + np.arange(-_WINDOW, 1)
    available = mask[indices].all(axis=1)
    windows = np.asarray(z[indices], dtype=np.float64)
    increments = np.diff(windows, axis=1).transpose(0, 2, 1)
    increments -= increments.mean(axis=2, keepdims=True)
    norm = np.linalg.norm(increments, axis=2)
    available &= norm >= _MIN_NORM
    np.divide(increments, norm[:, :, None], out=increments, where=available[:, :, None])
    increments[~available] = 0
    units[rows] = increments
    valid[rows] = available
    return units, valid


def _relations(units, groups, phases, left, right, edge_valid):
    edges = len(left)
    values = np.zeros((len(units), len(LAGS) * edges), dtype=np.float32)
    if not edges:
        return values

    rolled = units.copy()
    for group in np.unique(groups):
        shift = int(phases[group]) % _WINDOW
        if shift:
            columns = groups == group
            rolled[:, columns] = np.roll(units[:, columns], shift, axis=2)
    for lag_index, lag in enumerate(LAGS):
        later = np.roll(rolled, -lag, axis=2)
        products = np.matmul(rolled, later.transpose(0, 2, 1))
        direction = 0.5 * (products[:, left, right] - products[:, right, left])
        values[:, lag_index * edges:(lag_index + 1) * edges] = np.where(edge_valid, direction, 0)
    return values


def signatures(z, mask, ends, *, phases=PHASES[0], groups=GROUPS):
    """Return lag-major cross-group relations and their availability mask.

    Each end uses only the 26 samples ending there. Positive group phases
    follow NumPy's positive circular-roll direction on the 25 increments.
    """
    z, mask, ends, groups, left, right = _inputs(z, mask, ends, groups)
    phases = _phases(phases, groups)
    edges = len(left)
    values = np.zeros((len(ends), len(LAGS) * edges), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=bool)
    for start in range(0, len(ends), _BATCH):
        sl = slice(start, start + _BATCH)
        units, channel_valid = _units(z, mask, ends[sl])
        edge_valid = channel_valid[:, left] & channel_valid[:, right]
        values[sl] = _relations(units, groups, phases, left, right, edge_valid)
        for lag_index in range(len(LAGS)):
            valid[sl, lag_index * edges:(lag_index + 1) * edges] = edge_valid
    return values, valid


def feedback(z, mask, ends, *, groups=GROUPS):
    """Return jointly eligible, unit-norm real and phase-control deltas."""
    z, mask, ends, groups, left, right = _inputs(z, mask, ends, groups)
    edges = len(left)
    shape = (len(ends), len(LAGS) * edges)
    real_delta = np.zeros(shape, dtype=np.float32)
    null_delta = np.zeros(shape, dtype=np.float32)
    eligible = np.zeros(len(ends), dtype=bool)
    for start in range(0, len(ends), _BATCH):
        sl = slice(start, start + _BATCH)
        units, channel_valid = _units(z, mask, ends[sl])
        edge_valid = channel_valid[:, left] & channel_valid[:, right]
        r, p, n = (_relations(units, groups, phases, left, right, edge_valid) for phases in PHASES)
        real = r - (p + n) * 0.5
        null = p - n
        real_norm = np.linalg.norm(real, axis=1)
        null_norm = np.linalg.norm(null, axis=1)
        allowed = edge_valid.any(axis=1) & (real_norm >= _MIN_NORM) & (null_norm >= _MIN_NORM)
        np.divide(real, real_norm[:, None], out=real_delta[sl], where=allowed[:, None])
        np.divide(null, null_norm[:, None], out=null_delta[sl], where=allowed[:, None])
        eligible[sl] = allowed
    return real_delta, null_delta, eligible
