"""Causal SXR moment descriptors and delayed, jointly observed targets.

These coordinates describe the recorded moments; they are not a physical-state
estimate. Missing or unsupported observations remain masked from learning.
"""

import numpy as np


# Canonical io23.RAW23 columns: sxr_mean, sxr_var, sxr_skew, sxr_kurt.
_SXR_COLUMNS = (15, 17, 16, 14)
_GAP_SLACK = 1e-5


def coordinates(values, available):
    """Return float64 [g, d, skew, log1p(gap)] and a joint row mask.

    ``values`` and boolean ``available`` have shape (time, 23). Unsupported
    rows are zero in the output and must be excluded using the returned mask.
    The gap is computed from the recorded fields as sxr_kurt + 2 - sxr_skew**2.
    """
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available)
    if values.ndim != 2 or values.shape[1] != 23:
        raise ValueError("values must have shape (time, 23) in RAW23 order")
    if available.shape != values.shape or available.dtype != np.bool_:
        raise ValueError("available must be a boolean array with the same shape as values")

    moments = values[:, _SXR_COLUMNS]
    mean, variance, skew, kurtosis = moments.T
    valid = (available[:, _SXR_COLUMNS].all(axis=1)
             & np.isfinite(moments).all(axis=1)
             & (mean > 0) & (variance > 0))
    indices = np.flatnonzero(valid)
    with np.errstate(over="ignore", invalid="ignore"):
        gap = kurtosis[indices] + 2.0 - skew[indices]**2
    supported = np.isfinite(gap) & (gap >= -_GAP_SLACK)
    valid[indices[~supported]] = False
    indices, gap = indices[supported], gap[supported]

    result = np.zeros((len(values), 4), dtype=np.float64)
    u = np.log(mean[indices])
    w = .5 * np.log(variance[indices])
    result[indices, 0] = (u + w) / np.sqrt(2.)
    result[indices, 1] = (w - u) / np.sqrt(2.)
    result[indices, 2] = skew[indices]
    result[indices, 3] = np.log1p(np.maximum(gap, 0.))
    return result, valid


def packet(values, available, q):
    """Build issue-time context and mature future targets for issue indices ``q``.

    ``q`` is a one-dimensional integer array indexing rows of the RAW23 input;
    output rows follow its order. The context uses the jointly valid samples in
    q-31..q-16 and q-15..q, with at least eight in each span. At the start of a
    series, nonexistent history is unavailable rather than padded.

    Five targets average q+1..q+25 in successive groups of five, each relative
    to the issue-time recent mean. A target is mature only when q+26 exists,
    the issue is valid, and all 25 target samples are jointly valid. The q+26
    sample is a release-clock boundary, not part of the target. For online use,
    pass only the input prefix that has arrived at the current clock: maturity
    is determined by the length of that prefix. Invalid contexts and immature
    targets are zero and must be excluded by their masks.
    """
    coords, observed = coordinates(values, available)
    q = np.asarray(q)
    if q.ndim != 1:
        raise ValueError("q must be a one-dimensional array of issue indices")
    if q.size and not np.issubdtype(q.dtype, np.integer):
        raise TypeError("q must contain integer issue indices")
    if np.any(q < 0) or np.any(q >= len(coords)):
        raise ValueError("q contains an index outside the input series")
    q = q.astype(np.intp)

    context = np.zeros((len(q), 8), dtype=np.float64)
    targets = np.zeros((len(q), 5, 4), dtype=np.float64)
    issue_valid = np.zeros(len(q), dtype=bool)
    mature_valid = np.zeros(len(q), dtype=bool)
    if not len(q):
        return context, targets, issue_valid, mature_valid

    history = q[:, None, None] + np.arange(-31, 1).reshape(1, 2, 16)
    safe_history = np.maximum(history, 0)
    present = (history >= 0) & observed[safe_history]
    counts = present.sum(axis=2)
    total = np.where(present[..., None], coords[safe_history], 0.).sum(axis=2)
    means = np.divide(total, counts[..., None], out=np.zeros_like(total),
                      where=counts[..., None] > 0)
    issue_valid = (counts >= 8).all(axis=1)
    context[issue_valid, :4] = means[issue_valid, 1]
    context[issue_valid, 4:] = means[issue_valid, 1] - means[issue_valid, 0]

    candidates = np.flatnonzero(issue_valid & (q + 26 < len(coords)))
    future = q[candidates, None] + np.arange(1, 26)
    complete = observed[future].all(axis=1)
    rows = candidates[complete]
    mature_valid[rows] = True
    levels = coords[future[complete]].reshape(-1, 5, 5, 4).mean(axis=2)
    targets[rows] = levels - context[rows, None, :4]
    return context, targets, issue_valid, mature_valid
