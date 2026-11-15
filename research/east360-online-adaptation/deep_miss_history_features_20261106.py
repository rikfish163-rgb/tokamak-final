"""Causal, masked block summaries in the caller's frozen asinh coordinates.

For anchor t, the short window is [t-127, t]. Four older 256-sample
windows end at t-128-256*j, j=0..3. ``real`` uses j=0..2 and ``stale``
uses j=1..3; incomplete windows are entirely absent, not padded. Only the
20 RAW23 columns other than 1, 2, and 10 can contribute support.

FEATURE_NAMES defines the output order. In statistic-major order there are
20 each of masked mean, population standard deviation, second-half minus
first-half mean, observed fraction, RMS difference at lag 8, half-difference
validity, and lag-8 validity. A half-difference requires an observation in
each half; lag 8 requires a jointly observed pair. Three Pearson correlations
and then their validity flags follow. They use the Mir_VV, sxr_mean, and
xuv_ratio representative columns of the three disjoint RAW23 groups. A
correlation requires two joint observations and variation in both columns.
Undefined values are zero, with an explicit validity flag where needed.

``scrambled`` permutes whole supported real-history blocks in their supported
positions, never their contents. The nonidentity permutation depends only on
the shot, anchor, and fixed seed. A block-local numerical rescaling is undone
before returning the statistics; there is no shot-level normalization. Values
outside float64's representable range are saturated to finite float64 limits.
"""

import hashlib
import itertools

import numpy as np


# Kept local so importing this pure NumPy module never imports an HDF reader.
_RAW23 = (
    "Bt", "CIII", "MNM", "Mir_VV", "Mir_avg_amp", "Mir_avg_fre",
    "Z_proxy", "dx_a", "dy_a", "ip", "mode_number_n", "n=1 amplitude",
    "ne_nG", "qa_proxy", "sxr_kurt", "sxr_mean", "sxr_skew", "sxr_var",
    "v_loop", "xuv_kurt", "xuv_ratio", "xuv_skew", "xuv_var",
)
USED_CHANNELS = tuple(index for index in range(23) if index not in (1, 2, 10))
_USED_NAMES = tuple(_RAW23[index] for index in USED_CHANNELS)
_STATS = (
    "mean", "std", "half_delta", "observed_fraction", "rms_lag8",
    "half_delta_valid", "rms_lag8_valid",
)
PROXY_PAIRS = (("Mir_VV", "sxr_mean"), ("Mir_VV", "xuv_ratio"),
               ("sxr_mean", "xuv_ratio"))
FEATURE_NAMES = tuple(f"{stat}:{name}" for stat in _STATS for name in _USED_NAMES) + tuple(
    f"{stat}:{left}|{right}"
    for stat in ("corr", "corr_valid") for left, right in PROXY_PAIRS
)
FEAT_DIM = len(FEATURE_NAMES)
SCRAMBLE_SEED = 131
_PROXY_POSITIONS = tuple((_USED_NAMES.index(left), _USED_NAMES.index(right))
                         for left, right in PROXY_PAIRS)
_NONIDENTITY = {
    n: tuple(p for p in itertools.permutations(range(n)) if p != tuple(range(n)))
    for n in (2, 3)
}


def _rescale(values, scale):
    """Undo per-window arithmetic scaling, saturating unrepresentable results."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.nan_to_num(values * scale, nan=0.0,
                             posinf=np.finfo(np.float64).max,
                             neginf=-np.finfo(np.float64).max)


def _correlation(values, mask, left, right):
    joint = mask[:, left] & mask[:, right]
    if np.count_nonzero(joint) < 2:
        return 0.0, 0.0
    x, y = values[joint, left], values[joint, right]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 0.0
    # Subtract before scaling to retain small fluctuations on large offsets.
    # For opposite, near-limit values the subtraction may overflow, in which
    # case joint-only prescaling supplies a finite equivalent instead.
    with np.errstate(over="ignore"):
        x_offset, y_offset = x - x[0], y - y[0]
    if not np.isfinite(x_offset).all():
        scaled = x / np.max(np.abs(x))
        x_offset = scaled - scaled[0]
    if not np.isfinite(y_offset).all():
        scaled = y / np.max(np.abs(y))
        y_offset = scaled - scaled[0]
    x = x_offset / np.max(np.abs(x_offset))
    y = y_offset / np.max(np.abs(y_offset))
    x, y = x - np.mean(x), y - np.mean(y)
    x_scale, y_scale = np.max(np.abs(x)), np.max(np.abs(y))
    if x_scale == 0 or y_scale == 0:
        return 0.0, 0.0
    x, y = x / x_scale, y / y_scale
    corr = np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.clip(corr, -1.0, 1.0)), 1.0


def _summarize(z, available, start, stop):
    """Summarize exactly one complete half-open window; no other samples read."""
    values = np.asarray(z[start:stop, USED_CHANNELS], dtype=np.float64)
    mask = available[start:stop, USED_CHANNELS]
    if not np.isfinite(values[mask]).all():
        raise ValueError("observed used-channel coordinates must be finite")
    counts = mask.sum(axis=0)
    if not counts.any():
        return np.zeros(FEAT_DIM, dtype=np.float64), False
    clean = np.where(mask, values, 0.0)
    scale = np.max(np.abs(clean), axis=0)
    safe_scale = np.where(scale > 0, scale, 1.0)
    normalized = clean / safe_scale
    reference = clean[np.argmax(mask, axis=0), np.arange(len(USED_CHANNELS))]
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = (clean - reference) / safe_scale
    shifted = np.where(np.isfinite(shifted), shifted,
                       normalized - reference / safe_scale)
    shifted = np.where(mask, shifted, 0.0)
    divisor = np.maximum(counts, 1)
    mean = normalized.sum(axis=0) / divisor
    shifted_mean = shifted.sum(axis=0) / divisor
    centered = np.where(mask, shifted - shifted_mean, 0.0)
    std = np.sqrt(np.sum(centered * centered, axis=0) / divisor)

    middle = len(values) // 2
    first_count, second_count = mask[:middle].sum(axis=0), mask[middle:].sum(axis=0)
    first_mean = shifted[:middle].sum(axis=0) / np.maximum(first_count, 1)
    second_mean = shifted[middle:].sum(axis=0) / np.maximum(second_count, 1)
    half_valid = (first_count > 0) & (second_count > 0)
    half_delta = np.where(half_valid, second_mean - first_mean, 0.0)

    lag_mask = mask[8:] & mask[:-8]
    lag_count = lag_mask.sum(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        lag_delta = (clean[8:] - clean[:-8]) / safe_scale
    lag_delta = np.where(np.isfinite(lag_delta), lag_delta,
                         normalized[8:] - normalized[:-8])
    lag_rms = np.sqrt(np.sum(np.where(lag_mask, lag_delta * lag_delta, 0.0), axis=0)
                      / np.maximum(lag_count, 1))
    correlations = [_correlation(clean, mask, left, right)
                    for left, right in _PROXY_POSITIONS]
    result = np.concatenate((
        _rescale(mean, safe_scale), _rescale(std, safe_scale),
        _rescale(half_delta, safe_scale), counts / len(values),
        _rescale(lag_rms, safe_scale), half_valid.astype(np.float64),
        (lag_count > 0).astype(np.float64),
        np.array([v for v, _ in correlations]),
        np.array([valid for _, valid in correlations]),
    ))
    return result, True


def _permutation(shot, anchor, size):
    if size < 2:
        return tuple(range(size))
    key = f"{SCRAMBLE_SEED}\0{shot}\0{anchor}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    choices = _NONIDENTITY[size]
    return choices[int.from_bytes(digest, "little") % len(choices)]


def build(z, available, anchors, shot):
    """Return short, real/stale/scrambled history and history support masks.

    Inputs are a real numeric [T,23] array, its matching boolean availability
    mask, one-dimensional integer anchors in [127,T), and a stable shot ID.
    Unavailable values may be non-finite; observed values on used channels in
    complete requested windows must be finite. Outputs are float64 except for
    boolean support masks. Inputs and the global random generator are unchanged.
    """
    z, available, anchors = np.asarray(z), np.asarray(available), np.asarray(anchors)
    if z.ndim != 2 or z.shape[1] != 23 or z.dtype.kind not in "fiu":
        raise ValueError("z must be a real numeric [T,23] array")
    if available.shape != z.shape or available.dtype != np.bool_:
        raise ValueError("available must be a boolean [T,23] array")
    if anchors.ndim != 1 or (anchors.size and anchors.dtype.kind not in "iu"):
        raise ValueError("anchors must be a one-dimensional integer array")
    if np.any(anchors < 127) or np.any(anchors >= len(z)):
        raise ValueError("anchors must be in [127,T)")

    n = len(anchors)
    short = np.zeros((n, FEAT_DIM), dtype=np.float64)
    real = np.zeros((n, 3, FEAT_DIM), dtype=np.float64)
    stale = np.zeros_like(real)
    support_real = np.zeros((n, 3), dtype=bool)
    support_stale = np.zeros_like(support_real)
    # Reuse an identical historical window when anchors overlap by 256 samples.
    history_cache = {}
    for row, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        short[row], _ = _summarize(z, available, anchor - 127, anchor + 1)
        for j in range(4):
            end = anchor - 128 - 256 * j
            start = end - 255
            if start < 0:
                continue
            if end not in history_cache:
                history_cache[end] = _summarize(z, available, start, end + 1)
            features, supported = history_cache[end]
            if j < 3:
                real[row, j], support_real[row, j] = features, supported
            if j > 0:
                stale[row, j - 1], support_stale[row, j - 1] = features, supported

    scrambled = real.copy()
    for row, anchor_value in enumerate(anchors):
        positions = np.flatnonzero(support_real[row])
        if len(positions) >= 2:
            permutation = _permutation(shot, int(anchor_value), len(positions))
            scrambled[row, positions] = real[row, positions[list(permutation)]]
    return dict(short=short, real=real, stale=stale, scrambled=scrambled,
                support_real=support_real, support_stale=support_stale)
