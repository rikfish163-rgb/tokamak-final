"""Select pre-positive ordinal endpoints and match their two-point contexts."""

import operator

import numpy as np


def endpoints(times, valid, disruptive, down, horizon_ms, *, first_effect=153):
    """Return valid ``[far, near]`` sample indices, using timestamps in seconds.

    Disruptive near endpoints are chosen independently in the three bins of
    lead rounded to nine decimal places in milliseconds, after excluding the
    unrounded official positive window. Safe near endpoints use an eight-sample
    grid plus the last valid sample. A far endpoint is the latest sample at
    least 32 ms earlier, within the 1 ns clock tolerance; if that sample is
    invalid, the near endpoint is ineligible.
    """
    times = np.asarray(times, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if times.ndim != 1 or valid.shape != times.shape:
        raise ValueError('times and valid must be one-dimensional and aligned')
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('times must be finite and nondecreasing')
    try:
        first_effect = operator.index(first_effect)
    except TypeError as exc:
        raise ValueError('first_effect must be a nonnegative index') from exc
    if first_effect < 0:
        raise ValueError('first_effect must be a nonnegative index')

    empty = np.empty((0, 2), dtype=np.int64)
    if not len(times):
        return empty

    if disruptive:
        try:
            down = float(down)
            horizon_ms = float(horizon_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError('disruptive endpoints require a finite down and horizon_ms') from exc
        if not np.isfinite(down) or not np.isfinite(horizon_ms) or horizon_ms < 0:
            raise ValueError('disruptive endpoints require a finite down and nonnegative horizon_ms')
        lead = (down - times) * 1000.
        rounded_lead = np.round(lead, 9)
        near = np.flatnonzero(valid & (np.arange(len(times)) >= first_effect)
                              & (lead > horizon_ms + 1e-6))
    else:
        near = np.arange(first_effect, len(times), 8, dtype=np.int64)
        near = near[valid[near]]
        last_valid = np.flatnonzero(valid)
        if len(last_valid) and last_valid[-1] >= first_effect:
            if not len(near) or near[-1] != last_valid[-1]:
                near = np.append(near, last_valid[-1])

    if not len(near):
        return empty
    far = np.searchsorted(times, times[near] - .032 + 1e-9, side='right') - 1
    eligible = (far >= first_effect) & (far < near)
    eligible[eligible] = valid[far[eligible]]
    near, far = near[eligible], far[eligible]
    if not len(near):
        return empty

    if disruptive:
        selected = []
        for low, high, target in ((30., 60., 45.), (60., 95., 75.), (95., 125., 105.)):
            in_bin = (rounded_lead[near] > low) & (rounded_lead[near] <= high)
            options = np.flatnonzero(in_bin)
            if len(options):
                selected.append(options[np.argmin(np.abs(rounded_lead[near[options]] - target))])
        if not selected:
            return empty
        near, far = near[selected], far[selected]

    return np.column_stack((far, near)).astype(np.int64, copy=False)


def match(context, disruptive, falling_tail, scale):
    """Match each disruptive row to a safe tail and a safe non-tail row.

    ``context`` has shape ``(N, 2, 7)`` in far/near order. ``scale`` holds
    either seven shared feature scales or fourteen endpoint-specific scales,
    fixed from TRAIN. Distances are RMS differences across all 14 coordinates.
    """
    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 3 or context.shape[1:] != (2, 7) or not np.isfinite(context).all():
        raise ValueError('context must be finite and have shape (N, 2, 7)')
    disruptive = np.asarray(disruptive, dtype=bool)
    falling_tail = np.asarray(falling_tail, dtype=bool)
    if disruptive.shape != (len(context),) or falling_tail.shape != (len(context),):
        raise ValueError('disruptive and falling_tail must have one value per context row')
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape == (14,):
        scale = scale.reshape(2, 7)
    if scale.shape not in ((7,), (2, 7)) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('scale must contain seven or fourteen finite positive TRAIN scales')

    anchors = np.flatnonzero(disruptive)
    pos = np.repeat(anchors, 2).astype(np.int64, copy=False)
    if not len(anchors):
        return pos, np.empty(0, np.int64), np.empty(0, bool), np.empty(0, np.float64)

    safe = ~disruptive
    pools = (np.flatnonzero(safe & falling_tail), np.flatnonzero(safe & ~falling_tail))
    if any(not len(pool) for pool in pools):
        raise ValueError('each safe stratum (falling tail and other) requires at least one row')

    donors = np.empty(len(pos), dtype=np.int64)
    distances = np.empty(len(pos), dtype=np.float64)
    for i, anchor in enumerate(anchors):
        for j, pool in enumerate(pools):
            squared = np.mean(((context[pool] - context[anchor]) / scale) ** 2, axis=(1, 2))
            best = np.argmin(squared)
            donors[2 * i + j] = pool[best]
            distances[2 * i + j] = np.sqrt(squared[best])
    return pos, donors, falling_tail[donors], distances
