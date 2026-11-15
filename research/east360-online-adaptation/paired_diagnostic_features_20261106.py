"""Causal, paired diagnostic block differences in fixed native coordinates."""

import numpy as np


BLOCK = 32
GROUPS = (
    ("mir", (3, 4, 5)),
    ("sxr", (14, 15, 16, 17)),
    ("xuv", (19, 20, 21, 22)),
)
CHANNELS = tuple(channel for _, channels in GROUPS for channel in channels)
GROUP_INDEX = {
    channel: group
    for group, (_, channels) in enumerate(GROUPS)
    for channel in channels
}


def schedule(values, available, center, scale):
    """Return block-pair diagnostics, indexed by their inclusive maturities.

    ``values`` and the Boolean ``available`` mask are [time, channel];
    ``center`` and ``scale`` are fixed [channel] vectors. Only CHANNELS are
    inspected. Every observed selected value must be finite, including values
    after the last maturity; masked values and excluded channels are ignored.

    At maturity t, each channel compares the true medians of the observed
    asinh-normalized samples in [t-63, t-31) and [t-31, t+1). A channel is
    valid with at least 16 observations in each block. A row is supported only
    when such channels occur in at least two groups. Unsupported rows retain
    their individual valid diagnostics and must not drive a caller update.
    """
    values = np.asarray(values)
    available = np.asarray(available)
    center = np.asarray(center)
    scale = np.asarray(scale)
    if values.ndim != 2 or values.shape[1] <= max(CHANNELS):
        raise ValueError("values must be [time, channel] with all selected channels")
    if available.shape != values.shape or available.dtype != np.bool_:
        raise ValueError("available must be a matching Boolean mask")
    if center.shape != (values.shape[1],) or scale.shape != center.shape:
        raise ValueError("center and scale must be [channel] vectors")

    selected_values = np.asarray(values[:, CHANNELS], dtype=np.float64)
    selected_available = available[:, CHANNELS]
    selected_center = np.asarray(center[list(CHANNELS)], dtype=np.float64)
    selected_scale = np.asarray(scale[list(CHANNELS)], dtype=np.float64)
    if not np.isfinite(selected_center).all():
        raise ValueError("selected center values must be finite")
    if not np.isfinite(selected_scale).all() or (selected_scale <= 0).any():
        raise ValueError("selected scale values must be finite and positive")
    if not np.isfinite(selected_values[selected_available]).all():
        raise ValueError("observed selected values must be finite")

    maturity = np.arange(2 * BLOCK - 1, len(values), BLOCK, dtype=np.int64)
    origin = maturity - BLOCK
    delta = np.zeros((len(maturity), len(CHANNELS)), dtype=np.float32)
    valid = np.zeros(delta.shape, dtype=bool)
    support = np.zeros(len(maturity), dtype=bool)
    for row, t in enumerate(maturity):
        first = slice(t - 2 * BLOCK + 1, t - BLOCK + 1)
        second = slice(t - BLOCK + 1, t + 1)
        groups = set()
        for col, channel in enumerate(CHANNELS):
            mask_a = selected_available[first, col]
            mask_b = selected_available[second, col]
            if mask_a.sum() < BLOCK // 2 or mask_b.sum() < BLOCK // 2:
                continue
            # Extended precision protects finite inputs from intermediate overflow.
            a = selected_values[first, col][mask_a].astype(np.longdouble)
            b = selected_values[second, col][mask_b].astype(np.longdouble)
            c = np.longdouble(selected_center[col])
            s = np.longdouble(selected_scale[col])
            median_a = np.median(np.arcsinh((a - c) / s))
            median_b = np.median(np.arcsinh((b - c) / s))
            delta[row, col] = median_b - median_a
            valid[row, col] = True
            groups.add(GROUP_INDEX[channel])
        support[row] = len(groups) >= 2

    return dict(origin=origin, maturity=maturity, delta=delta, valid=valid, support=support)
