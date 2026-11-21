"""Paired, causal two-stage replay of per-shot forecast corrections.

The short and tail steps use the forecast frozen at the origin. Each supported
step retains 0.92 of the current state and takes a 0.45 preconditioned Huber
gradient step; this retention is applied separately for the two stages, only
to channels with observations. The diagonal preconditioner is 0.25 plus the
segment mean of mask * basis**2. Supported stages require a channel with at
least half the segment observed in each of at least two physical families.
Once this gate opens, every channel with any observation can update.
"""

import operator

import numpy as np

from .commissioning_weak_shape_data_20261111 import GROUPS


_BASIS = np.column_stack((
    np.ones(25, dtype=np.float32),
    np.arange(1, 26, dtype=np.float32) / np.float32(25),
))
_GROUPS = np.asarray(GROUPS)
_ARMS = ('split', 'delayed', 'reversed')


def _stage(issued, target, observed, basis, *, reverse=False, centered=False):
    """Return the fixed issued-snapshot step, or None if the gate is shut."""
    counts = observed.sum(axis=0)
    qualifying = counts >= (len(basis) + 1) // 2
    if np.unique(_GROUPS[qualifying]).size < 2:
        return None

    if centered:
        # Eliminate the unknown intercept before the robust derivative. A
        # constant diagnostic offset must not produce any temporal update.
        denom = np.maximum(counts, 1).astype(np.float32)
        error = np.where(observed, issued, 0) - np.where(observed, target, 0)
        residual = np.where(observed, error - error.sum(0) / denom, 0.)
        t = basis[:, 1:2]
        local = np.where(observed, t - (observed * t).sum(0) / denom, 0.)
        local *= np.float32(2. / np.ptp(t))
        slope = (np.clip(residual, -1., 1.) * local).sum(0) / denom
        gradient = np.column_stack((np.zeros_like(slope), -slope if reverse else slope))
        diagonal = np.column_stack((np.ones_like(slope), .25 + (local**2).sum(0) / denom))
        return counts >= 2, gradient, diagonal

    if reverse:
        target = target.copy()
        for channel in range(target.shape[1]):
            slots = np.flatnonzero(observed[:, channel])
            target[slots, channel] = target[slots[::-1], channel]

    error = np.where(observed, issued, 0) - np.where(observed, target, 0)
    huber_gradient = np.clip(error, -1, 1)
    gradient = np.einsum('hd,hc->dc', huber_gradient, basis)
    gradient /= np.maximum(counts, 1)[:, None]
    diagonal = np.float32(0.25) + np.einsum(
        'hd,hc->dc', observed.astype(np.float32), basis**2
    ) / np.float32(len(basis))
    return counts > 0, gradient, diagonal


def _apply(state, step):
    if step is None:
        return
    available, gradient, diagonal = step
    state[available] = np.clip(
        np.float32(0.92) * state[available]
        - np.float32(0.45) * gradient[available] / diagonal[available],
        -2, 2,
    )


def replay(forecast, target, observed, origins, length, *, centered=False):
    """Return channel-major (level, slope) states at each sample and audit.

    Inputs have shape (Q, 25, 10); horizon index 0 corresponds to q+1.
    Origins are in-range, strictly increasing integer indices at least 26
    apart. The short and tail updates become visible at q+9 and q+26. The
    delayed arm applies both frozen steps at q+26, in short-then-tail order.
    The reversed arm reverses target values only within each channel's
    observed slots and each segment; its release and gate are unchanged.
    In the centered experiment, the control instead flips only the signed
    slope gradient from the same issued reference; no mean residual enters.

    A support count is the number of real-data stages whose family gate passes
    and whose respective split release is visible in the returned sequence.
    Per-arm visible_updates counts supported steps applied at a visible index,
    including a zero-gradient step. A fresh zero state is used for each call.
    """
    forecast = np.asarray(forecast, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    observed = np.asarray(observed)
    origins = np.asarray(origins)
    try:
        length = operator.index(length)
    except TypeError as exc:
        raise ValueError('length must be a nonnegative integer') from exc
    if length < 0:
        raise ValueError('length must be a nonnegative integer')
    if forecast.ndim != 3 or forecast.shape[1:] != (25, 10):
        raise ValueError('forecast must have shape (Q, 25, 10)')
    if target.shape != forecast.shape or observed.shape != forecast.shape:
        raise ValueError('target and observed must match forecast shape')
    if observed.dtype != np.bool_:
        raise ValueError('observed must be boolean')
    if origins.ndim != 1 or len(origins) != len(forecast):
        raise ValueError('origins must be one-dimensional with Q entries')
    if origins.size and not np.issubdtype(origins.dtype, np.integer):
        raise ValueError('origins must contain integers')
    if (np.any(origins < 0) or np.any(origins >= length)
            or np.any(origins[1:] <= origins[:-1]) or np.any(np.diff(origins) < 26)):
        raise ValueError('origins must be in range and separated by at least 26')

    states = {arm: np.zeros((length, 20), dtype=np.float32) for arm in _ARMS}
    current = {arm: np.zeros((10, 2), dtype=np.float32) for arm in _ARMS}
    audit = {
        'visible_updates': {arm: 0 for arm in _ARMS},
        'short_supported': 0,
        'tail_supported': 0,
    }

    for i, origin in enumerate(origins):
        q = int(origin)
        stop = int(origins[i + 1]) if i + 1 < len(origins) else length
        issued = {
            arm: forecast[i] + np.einsum('hc,dc->hd', _BASIS, current[arm])
            for arm in _ARMS
        }
        for arm in _ARMS:
            states[arm][q:min(q + 9, stop)] = current[arm].reshape(20)
        if q + 9 >= length:
            continue

        control = issued['split'] if centered else issued['reversed']
        short = _stage(issued['split'][:8], target[i, :8], observed[i, :8], _BASIS[:8],
                       centered=centered)
        reverse_short = _stage(
            control[:8], target[i, :8], observed[i, :8], _BASIS[:8],
            reverse=True, centered=centered,
        )
        if short is not None:
            audit['short_supported'] += 1
            audit['visible_updates']['split'] += 1
            audit['visible_updates']['reversed'] += 1
        _apply(current['split'], short)
        _apply(current['reversed'], reverse_short)
        for arm in _ARMS:
            states[arm][q + 9:min(q + 26, stop)] = current[arm].reshape(20)
        if q + 26 >= length:
            continue

        tail = _stage(issued['split'][8:], target[i, 8:], observed[i, 8:], _BASIS[8:],
                      centered=centered)
        reverse_tail = _stage(
            control[8:], target[i, 8:], observed[i, 8:], _BASIS[8:],
            reverse=True, centered=centered,
        )
        _apply(current['delayed'], short)
        _apply(current['split'], tail)
        _apply(current['delayed'], tail)
        _apply(current['reversed'], reverse_tail)
        if short is not None:
            audit['visible_updates']['delayed'] += 1
        if tail is not None:
            audit['tail_supported'] += 1
            for arm in _ARMS:
                audit['visible_updates'][arm] += 1
        for arm in _ARMS:
            states[arm][q + 26:stop] = current[arm].reshape(20)

    return states, audit
