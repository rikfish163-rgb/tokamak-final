"""Describe arrived patch levels and changes; neither quantity predicts risk.

The source bounds are per-channel quantiles of source patch means, supplied by
the caller.  A change is an observed difference, not a forecast error.  A
constant prefix describes exact equality on observed pairs, not safety.
"""

import math
import operator

import numpy as np


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have a real numeric dtype")
    return array


def patch_diagnostics(means, complete, lower, upper):
    """Return one diagnostic value per patch, using only current/past patches.

    ``means`` and boolean ``complete`` have shape (P, 23); ``lower`` and
    ``upper`` have shape (23,).  Unavailable means may be nonfinite, but all
    observed means and all bounds must be finite.  Source-bound equality is
    inside the range.  The per-patch RMS uses common observed channels for
    lags one through four and is not restricted to the interval [0, 1].

    Returned values are one-dimensional NumPy arrays.  A missing level or
    change is NaN, distinguished from a measured zero by its corresponding
    count.  Inputs are not modified.
    """
    means = _real_array(means, "means")
    complete = np.asarray(complete)
    lower = _real_array(lower, "lower")
    upper = _real_array(upper, "upper")
    if means.ndim != 2 or means.shape[1] != 23:
        raise ValueError("means must have shape (P, 23)")
    if complete.shape != means.shape:
        raise ValueError("complete must have the same shape as means")
    if complete.dtype.kind != "b":
        raise TypeError("complete must have boolean dtype")
    if lower.shape != (23,) or upper.shape != (23,):
        raise ValueError("lower and upper must each have shape (23,)")
    if not np.isfinite(means[complete]).all():
        raise ValueError("observed means must be finite")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("source bounds must be finite")

    # Zero unavailable entries before any subtraction, avoiding NaN * 0.
    # Extended precision, where available, avoids premature float32/64
    # overflow and loss of integer precision in the difference and square.
    safe = np.where(complete, means, 0).astype(np.longdouble)
    lower = lower.astype(np.longdouble)
    upper = upper.astype(np.longdouble)
    if np.any(lower > upper):
        raise ValueError("lower must not exceed upper")

    observed_count = complete.sum(axis=1, dtype=np.int64)
    outside_count = (((safe < lower) | (safe > upper)) & complete).sum(
        axis=1, dtype=np.int64
    )
    outside_fraction = np.full(means.shape[0], np.nan, dtype=np.float64)
    np.divide(outside_count, observed_count, out=outside_fraction,
              where=observed_count > 0)

    squared_change = np.zeros(means.shape[0], dtype=np.longdouble)
    feedback_pairs = np.zeros(means.shape[0], dtype=np.int64)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            for lag in range(1, min(4, means.shape[0] - 1) + 1):
                common = complete[lag:] & complete[:-lag]
                difference = np.zeros_like(safe[lag:])
                np.subtract(safe[lag:], safe[:-lag], out=difference, where=common)
                squared = difference * difference
                if np.any((difference != 0) & (squared == 0)):
                    raise ValueError("squared change is too small to represent")
                squared_change[lag:] += squared.sum(axis=1, dtype=np.longdouble)
                feedback_pairs[lag:] += common.sum(axis=1, dtype=np.int64)
            cumulative_squared_change = np.cumsum(squared_change, dtype=np.longdouble)
            cumulative_feedback_pairs = np.cumsum(feedback_pairs, dtype=np.int64)
            increment_rms = np.full(means.shape[0], np.nan, dtype=np.longdouble)
            # Taking the square root first avoids underflow in sum / pairs.
            np.sqrt(squared_change, out=increment_rms, where=feedback_pairs > 0)
            np.divide(increment_rms, np.sqrt(feedback_pairs.astype(np.longdouble)),
                      out=increment_rms,
                      where=feedback_pairs > 0)
    except FloatingPointError as exc:
        raise ValueError("squared change exceeds the numeric range") from exc

    return {
        "observed_count": observed_count,
        "outside_count": outside_count,
        "outside_fraction": outside_fraction,
        "increment_rms": increment_rms,
        "feedback_pairs": feedback_pairs,
        "cumulative_squared_change": cumulative_squared_change,
        "cumulative_feedback_pairs": cumulative_feedback_pairs,
        "prefix_exactly_constant": ((cumulative_feedback_pairs > 0)
                                    & (cumulative_squared_change == 0)),
    }


def _json_float(value, name):
    converted = float(value)
    if not math.isfinite(converted) or (converted == 0 and value != 0):
        raise ValueError(f"{name} is not faithfully representable as a JSON float")
    return converted


def _measured_or_none(value, count, name):
    if np.isnan(value) and count == 0:
        return None
    if count == 0:
        raise ValueError(f"{name} must be NaN when its count is zero")
    return _json_float(value, name)


def at_endpoint(diag, end):
    """Convert the last complete patch at sample endpoint ``end`` to JSON.

    The patch index is ``end // 8 - 1``; endpoints 0 through 7 have no complete
    patch, and a partially arrived subsequent patch is not included.  Negative
    endpoints and indices past the supplied diagnostics are errors.
    The absent-patch row has zero counts, no level/change, and an explicit
    false ``patch_available`` flag.  No-feedback is never evidence of exact
    constancy.  NaN is converted to None only when its count is zero.  Values
    that would overflow or underflow to zero as JSON floats raise ValueError.
    """
    if isinstance(end, (bool, np.bool_)):
        raise TypeError("end must be an integer, not a boolean")
    try:
        end = operator.index(end)
    except TypeError as exc:
        raise TypeError("end must be an integer") from exc
    if end < 0:
        raise ValueError("end must not be negative")
    index = end // 8 - 1
    if index < 0:
        return {
            "patch_available": False,
            "patch_index": None,
            "observed_count": 0,
            "outside_count": 0,
            "outside_fraction": None,
            "increment_rms": None,
            "feedback_pairs": 0,
            "cumulative_squared_change": 0.0,
            "cumulative_feedback_pairs": 0,
            "prefix_exactly_constant": False,
        }
    if index >= len(diag["observed_count"]):
        raise ValueError("endpoint is beyond the supplied patches")

    observed_count = int(diag["observed_count"][index])
    feedback_pairs = int(diag["feedback_pairs"][index])
    outside_fraction = _measured_or_none(diag["outside_fraction"][index],
                                         observed_count, "outside_fraction")
    if outside_fraction is not None and not 0 <= outside_fraction <= 1:
        raise ValueError("outside_fraction must be between zero and one")
    return {
        "patch_available": True,
        "patch_index": index,
        "observed_count": observed_count,
        "outside_count": int(diag["outside_count"][index]),
        "outside_fraction": outside_fraction,
        "increment_rms": _measured_or_none(diag["increment_rms"][index],
                                           feedback_pairs, "increment_rms"),
        "feedback_pairs": feedback_pairs,
        "cumulative_squared_change": _json_float(
            diag["cumulative_squared_change"][index], "cumulative_squared_change"
        ),
        "cumulative_feedback_pairs": int(diag["cumulative_feedback_pairs"][index]),
        "prefix_exactly_constant": bool(diag["prefix_exactly_constant"][index]),
    }
