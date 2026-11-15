"""Extend the lifecycle sampler with uniformly selected short histories."""

import numpy as np

from ..lifecycle_stream_20261025.protocol import sampled_endpoints


def sampled_with_prefix(times, y, valid, scores, *, down):
    """Sample a completed shot, preserving all original lifecycle endpoints.

    The caller must supply the legally revealed arrays from the completed shot;
    this function cannot enforce the surrounding seal/reveal ordering.
    """
    times, y, valid, scores = map(np.asarray, (times, y, valid, scores))
    if times.ndim != 1 or any(a.shape != times.shape for a in (y, valid, scores)):
        raise ValueError('matching per-endpoint training arrays required')
    if valid.dtype != np.dtype(bool):
        raise ValueError('valid must be a boolean array')
    try:
        finite = all(np.isrealobj(a) and np.isfinite(a).all()
                     for a in (times, y, scores))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError('times, y, and scores must be finite real arrays')
    if down is not None:
        down = np.asarray(down)
        try:
            valid_down = down.ndim == 0 and np.isrealobj(down) and np.isfinite(down)
        except (TypeError, ValueError):
            valid_down = False
        if not valid_down:
            raise ValueError('down must be a finite real scalar or None')

    old = sampled_endpoints(times, y, valid, scores, down=down)
    candidates = np.flatnonzero(valid[:127])
    if down is not None:
        candidates = candidates[np.round((down - times[candidates]) * 1000, 9) > 30]
    if not len(candidates):
        return old
    selected = candidates[np.linspace(0, len(candidates) - 1,
                                      min(16, len(candidates)), dtype=int)]
    return np.union1d(old, selected)
