"""Simultaneous whole-feature past samples with explicit prefix behavior."""

import numpy as np

from ..simple_data import window


def span_windows(signal, ends, stride):
    if (
        set(signal) != {"values", "available"}
        or type(stride) is not int
        or stride not in (1, 2)
    ):
        raise ValueError("signals only and explicit stride1 or2 required")
    ends = np.asarray(ends)
    if (
        ends.ndim != 1
        or not len(ends)
        or not np.issubdtype(ends.dtype, np.integer)
        or (ends < 0).any()
        or (ends >= len(signal["values"])).any()
    ):
        raise ValueError("nonempty arrived endpoint indices required")
    xs, masks = [], []
    eligible = ends >= 127 * stride
    for end, enabled in zip(ends, eligible, strict=True):
        if stride == 1 or not enabled:
            x, m = window(signal, int(end), 128)
        else:
            indices = end - stride * np.arange(127, -1, -1)
            x, m = signal["values"][indices].copy(), signal["available"][indices].copy()
        xs.append(x)
        masks.append(m)
    return np.stack(xs), np.stack(masks), eligible
