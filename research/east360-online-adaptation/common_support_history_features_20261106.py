"""Causal v1 summaries for three history arms on identical four-block support.

For an anchor, the short window is [anchor-127, anchor+1); older block j
is [anchor-127-(j+1)*block, anchor-127-j*block), for j=0..3. Each block
requires its full length and at least one observation on a v1 used channel.
Only rows with all four blocks supported receive any feature values. The
caller must use the original risk on all other rows; zeros mean unavailable,
not a prediction of zero risk.

Statistics and the whole-block scramble are the frozen v1 implementations.
No observation beyond an anchor is read when computing that row. Nonfinite
observed used-channel values are rejected only in computed, common-support
windows; masked and excluded-channel values do not contribute.
"""

import numpy as np

from deep_miss_history_features_20261106 import (
    FEAT_DIM,
    USED_CHANNELS,
    _permutation,
    _summarize,
)


def _validate(mask, anchors, block):
    mask = np.asarray(mask)
    anchors = np.asarray(anchors)
    if mask.ndim != 2 or mask.shape[1] != 23 or mask.dtype != np.bool_:
        raise ValueError("mask must be a boolean [T,23] array")
    if anchors.ndim != 1 or (anchors.size and anchors.dtype.kind not in "iu"):
        raise ValueError("anchors must be a one-dimensional integer array")
    if np.any(anchors < 127) or np.any(anchors >= len(mask)):
        raise ValueError("anchors must be in [127,T)")
    if isinstance(block, (bool, np.bool_)) or not isinstance(block, (int, np.integer)):
        raise ValueError("block must be an integer >= 16")
    if block < 16:
        raise ValueError("block must be an integer >= 16")
    return mask, anchors, int(block)


def _support(mask, anchors, block):
    result = np.zeros((len(anchors), 4), dtype=bool)
    for row, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        for slot in range(4):
            stop = anchor - 127 - slot * block
            start = stop - block
            if start >= 0:
                result[row, slot] = np.any(mask[start:stop, USED_CHANNELS])
    return result


def support(mask, anchors, block):
    """Return [N,4] boolean support for complete, observed older blocks.

    ``mask`` is boolean [T,23]; anchors may be unsorted or empty and must
    otherwise be integers in [127,T). ``block`` is an integer >= 16.
    """
    mask, anchors, block = _validate(mask, anchors, block)
    return _support(mask, anchors, block)


def build(z, mask, anchors, shot, block):
    """Build common-support short and real/stale/scrambled v1 summaries.

    ``z`` is a real numeric [T,23] array and ``mask`` is a matching boolean
    array. The result contains float64 short [N,146] and history arms
    [N,3,146], plus boolean support [N,4]. All features are zero outside the
    rows where every support flag is true. ``shot`` is passed to v1's stable
    shot-and-anchor permutation unchanged.
    """
    z = np.asarray(z)
    if z.ndim != 2 or z.shape[1] != 23 or z.dtype.kind not in "fiu":
        raise ValueError("z must be a real numeric [T,23] array")
    mask, anchors, block = _validate(mask, anchors, block)
    if mask.shape != z.shape:
        raise ValueError("mask must have the same shape as z")

    support_flags = _support(mask, anchors, block)
    short = np.zeros((len(anchors), FEAT_DIM), dtype=np.float64)
    real = np.zeros((len(anchors), 3, FEAT_DIM), dtype=np.float64)
    stale = np.zeros_like(real)
    scrambled = np.zeros_like(real)
    history_cache = {}
    for row in np.flatnonzero(support_flags.all(axis=1)):
        anchor = int(anchors[row])
        short[row], _ = _summarize(z, mask, anchor - 127, anchor + 1)
        for slot in range(4):
            stop = anchor - 127 - slot * block
            start = stop - block
            if stop not in history_cache:
                history_cache[stop] = _summarize(z, mask, start, stop)[0]
            features = history_cache[stop]
            if slot < 3:
                real[row, slot] = features
            if slot > 0:
                stale[row, slot - 1] = features
        permutation = _permutation(shot, anchor, 3)
        scrambled[row] = real[row, list(permutation)]

    return dict(short=short, real=real, stale=stale, scrambled=scrambled,
                support=support_flags)
