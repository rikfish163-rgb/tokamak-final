"""Source-fitted, independent-input forecast of future AXUV-skew shape.

Fit only on J-TEXT TRAIN packets and use VAL only for held-out evaluation.
The caller is responsible for the split: these signal-only functions cannot
identify devices or data partitions.  EAST must not be passed to ``fit_weighted``.
An issued forecast can be compared with the measured direction only when
``mature_valid`` is true and the forecast was issued at the original clock.
On a full-shot array, the mask alone does not enforce the current online clock:
the caller must wait for q+26 or pass only the prefix available at that time.
"""

from collections.abc import Mapping

import numpy as np


HISTORY_POINTS = 126  # q-125 through q, inclusive
HORIZON = 25
BIN_SIZE = 5
RELEASE_OFFSET = 26
RIDGE = 0.10
# A fixed dimensionless float64 floor; near or below it exact affine
# invariance is not expected.  It also defines finite constant-window output.
SCALE_FLOOR = np.finfo(np.float64).eps

TARGET_INDEX = 21  # xuv_skew in the canonical mapped flat23 order
SIGNED_INPUTS = (("n=1 amplitude", 11), ("v_loop", 18), ("ip", 9))
POSITION_INPUTS = (("dx_a", 7), ("dy_a", 8))
FEATURE_NAMES = (
    ("intercept",)
    + tuple(
        f"{name}:w{window}_{kind}"
        for name, _ in SIGNED_INPUTS
        for window in range(3)
        for kind in ("start", "change")
    )
    + tuple(
        f"{name}:w{window}_abs_change"
        for name, _ in POSITION_INPUTS
        for window in range(3)
    )
)
D = len(FEATURE_NAMES)

__all__ = [
    "HISTORY_POINTS", "HORIZON", "BIN_SIZE", "RELEASE_OFFSET", "RIDGE",
    "SCALE_FLOOR", "TARGET_INDEX", "SIGNED_INPUTS", "POSITION_INPUTS",
    "FEATURE_NAMES", "D", "make_samples", "fit_weighted", "predict",
]


def _history(values, available, indices, channel):
    """Return past samples, masks, mean, population std, and scale validity."""
    past = np.asarray(values[indices, channel], dtype=np.float64)
    valid = available[indices, channel] & np.isfinite(past)
    count = valid.sum(axis=1)
    safe = np.where(valid, past, 0.0)
    mean = safe.sum(axis=1) / np.maximum(count, 1)
    centered = np.where(valid, past - mean[:, None], 0.0)
    scale = np.sqrt(np.square(centered).sum(axis=1) / np.maximum(count, 1))
    return safe, valid, mean, np.maximum(scale, SCALE_FLOOR), count >= 2


def _end_means(safe, valid, window):
    # The three consecutive 25-point windows cover q-74 through q.
    start = HISTORY_POINTS - 3 * HORIZON + window * HORIZON
    early = slice(start, start + BIN_SIZE)
    late = slice(start + HORIZON - BIN_SIZE, start + HORIZON)
    early_count = valid[:, early].sum(axis=1)
    late_count = valid[:, late].sum(axis=1)
    first = safe[:, early].sum(axis=1) / np.maximum(early_count, 1)
    last = safe[:, late].sum(axis=1) / np.maximum(late_count, 1)
    return first, last, (early_count > 0) & (late_count > 0)


def make_samples(signal, q):
    """Build issued inputs and, separately, strictly matured targets.

    ``signal`` has only ``values`` and ``available`` arrays in mapped flat23
    order. ``q`` is an increasing 1-D sequence of existing integer sample
    indices >=127, and may be empty.  Both arrays are read without mutation.

    For each independent channel, the scale and center are computed from its
    finite, available observations in q-125..q.  On each consecutive 25-point
    subwindow in q-74..q, start is the first-five mean minus that past center,
    and change is last-five mean minus first-five mean.  Both are divided by
    the channel's past population std and transformed by asinh.  The position
    proxies contribute only asinh(abs(change / std)), never signed position.
    Each past scale needs at least two observations, and each endpoint mean at
    least one.  The target channel's past scale likewise needs two observations
    but none of its values or derived statistics is included in the design.

    Each target uses all 25 finite, available xuv_skew points q+1..q+25.
    Their five five-point means are centered, divided by the target's past
    std, passed through asinh, and centered again.  It matures only when
    q+26 exists in the supplied array.  For a full-shot array the caller must
    also wait until q+26 has arrived.  Maturity depends on the target, not on
    independent-input validity, so callers must filter on both returned masks
    when fitting.

    Returns float64 design [N,D], bool issued_valid [N], float64 target [N,5],
    and bool mature_valid [N].  Invalid design/target rows contain NaNs; an
    empty query returns the corresponding empty shapes.
    """
    if not isinstance(signal, Mapping) or set(signal) != {"values", "available"}:
        raise ValueError("signal must contain only values and available")
    values = np.asarray(signal["values"])
    available = np.asarray(signal["available"])
    if (values.ndim != 2 or values.shape[1] != 23
            or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)):
        raise ValueError("values must be a real numeric [time, 23] array")
    if available.shape != values.shape or available.dtype != np.bool_:
        raise ValueError("available must be a matching boolean array")
    q = np.asarray(q)
    if q.ndim != 1 or (q.size and not np.issubdtype(q.dtype, np.integer)):
        raise ValueError("q must be a one-dimensional integer sequence")
    if q.size and (np.any(q < 127) or np.any(q >= len(values))):
        raise ValueError("q must contain existing indices >=127")
    q = q.astype(np.int64, copy=False)
    if np.any(np.diff(q) <= 0):
        raise ValueError("q must be strictly increasing")

    count = len(q)
    design = np.ones((count, D), dtype=np.float64)
    target = np.full((count, HORIZON // BIN_SIZE), np.nan, dtype=np.float64)
    mature = np.zeros(count, dtype=bool)
    if not count:
        return design, np.zeros(0, dtype=bool), target, mature

    past_indices = q[:, None] + np.arange(1 - HISTORY_POINTS, 1)
    _, _, _, target_scale, target_has_scale = _history(
        values, available, past_indices, TARGET_INDEX
    )
    issued = target_has_scale.copy()
    column = 1
    for name, channel in SIGNED_INPUTS + POSITION_INPUTS:
        past, observed, center, scale, has_scale = _history(
            values, available, past_indices, channel
        )
        issued &= has_scale
        for window in range(3):
            first, last, has_ends = _end_means(past, observed, window)
            issued &= has_ends
            if (name, channel) in SIGNED_INPUTS:
                design[:, column] = np.arcsinh((first - center) / scale)
                design[:, column + 1] = np.arcsinh((last - first) / scale)
                column += 2
            else:
                design[:, column] = np.arcsinh(np.abs((last - first) / scale))
                column += 1
    design[~issued] = np.nan

    candidates = np.flatnonzero((q + RELEASE_OFFSET < len(values)) & target_has_scale)
    if candidates.size:
        future_indices = q[candidates, None] + np.arange(1, HORIZON + 1)
        future = np.asarray(values[future_indices, TARGET_INDEX], dtype=np.float64)
        complete = (available[future_indices, TARGET_INDEX] & np.isfinite(future)).all(axis=1)
        rows = candidates[complete]
        if rows.size:
            means = future[complete].reshape(-1, HORIZON // BIN_SIZE, BIN_SIZE).mean(axis=2)
            centered = means - means.mean(axis=1, keepdims=True)
            shape = np.arcsinh(centered / target_scale[rows, None])
            target[rows] = shape - shape.mean(axis=1, keepdims=True)
            mature[rows] = True
    return design, issued, target, mature


def fit_weighted(packets):
    """Fit fixed 0.10 ridge to already-filtered, equally weighted shots.

    Every nonempty ``(design[M,D], target[M,5])`` packet has total weight
    one, regardless of M.  The normal equations are divided by the number of
    usable shots before regularization; every coefficient, including any
    intercept, is regularized.  Empty packets are skipped.  Only TRAIN packets
    belong here.  A row-centered float64 [D,5] matrix is returned.
    """
    gram = cross = None
    width = None
    shots = 0
    for design, target in packets:
        design = np.asarray(design, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if (design.ndim != 2 or not design.shape[1] or target.ndim != 2
                or target.shape != (len(design), 5)):
            raise ValueError("each packet must contain [M,D] and [M,5] arrays")
        if width is None:
            width = design.shape[1]
            gram = np.zeros((width, width), dtype=np.float64)
            cross = np.zeros((width, 5), dtype=np.float64)
        if design.shape[1] != width:
            raise ValueError("all packets must use the same feature dimension")
        if not np.isfinite(design).all() or not np.isfinite(target).all():
            raise ValueError("filter invalid and non-finite rows before fitting")
        if not len(design):
            continue
        gram += (design.T @ design) / len(design)
        cross += (design.T @ target) / len(design)
        shots += 1
    if not shots:
        raise ValueError("at least one nonempty training shot is required")
    coefficient = np.linalg.solve(gram / shots + RIDGE * np.eye(width), cross / shots)
    return coefficient - coefficient.mean(axis=1, keepdims=True)


def predict(design, coefficient):
    """Return centered [N,5] forecasts; NaN design rows remain NaN."""
    design = np.asarray(design, dtype=np.float64)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if (design.ndim != 2 or coefficient.ndim != 2
            or coefficient.shape != (design.shape[1], 5)):
        raise ValueError("design and coefficient must have shapes [N,D] and [D,5]")
    prediction = design @ coefficient
    return prediction - prediction.mean(axis=1, keepdims=True)
