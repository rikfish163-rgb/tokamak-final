"""Auxiliary H25 paired-logistic scores; neither difference is a calibrated Bayes factor.

All normalization and both independent models are fit only on the supplied
training rows. The caller owns the fold split and supplies actual normalized
asinh future-minus-current signals and a complete-channel availability mask.
Scoring consumes signal arrays only and never reads labels or outcomes.
"""

import numpy as np

from ..conditional_forecast_20261023.learning import shot_weights
from ..feedback_capacity_20261024.learning import logistic_fit


_HORIZON = 25
_CHANNELS = 23
_PREFIX = 64
_STATS = 4
_FLOOR = 1e-3
_L2 = 0.1


def summarize_arrival(target, valid):
    """Return channel-major mean, last-first, centered RMS, and last value."""
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid)
    if (
        target.ndim != 3
        or target.shape[1:] != (_HORIZON, _CHANNELS)
        or valid.shape != (len(target), _CHANNELS)
        or valid.dtype != np.bool_
    ):
        raise ValueError("target must be N-by-25-by-23 with an N-by-23 boolean mask")
    if not np.all(np.isfinite(target) | ~valid[:, None, :]):
        raise ValueError("valid target channels must be finite at every time point")
    safe = np.where(valid[:, None, :], target, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = safe.mean(axis=1)
        drift = safe[:, -1, :] - safe[:, 0, :]
        rms = np.sqrt(np.mean((safe - mean[:, None, :]) ** 2, axis=1))
        summary = np.stack((mean, drift, rms, safe[:, -1, :]), axis=2)
    if not np.isfinite(summary).all():
        raise ValueError("arrival summary must be finite")
    return summary.reshape(len(target), _CHANNELS * _STATS)


def _prefix_array(prefix, n):
    prefix = np.asarray(prefix, dtype=np.float64)
    if prefix.shape != (n, _PREFIX) or not np.isfinite(prefix).all():
        raise ValueError("prefix must be a finite N-by-64 matrix")
    return prefix


def _json_fit_ids(ids):
    """Keep shot identities intact while rejecting non-JSON or nonfinite IDs."""
    values = []
    for value in ids:
        value = value.item() if isinstance(value, np.generic) else value
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("shot IDs must be JSON-compatible scalar values")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("shot IDs must be finite")
        values.append(value)
    return values


def fit_pair(prefix, target, valid, ids, labels, *, device="cpu"):
    """Fit fold-only paired classifiers and return JSON-serializable parameters.

    Prefix features use the shot/class-weighted mean and standard deviation.
    Each arrival feature uses weighted RMS over training rows where its entire
    channel is valid, without centering. Both scales have a 1e-3 floor.
    """
    summary = summarize_arrival(target, valid)
    valid = np.asarray(valid)
    prefix = _prefix_array(prefix, len(summary))
    ids, labels = np.asarray(ids), np.asarray(labels)
    if ids.shape != (len(summary),) or labels.shape != (len(summary),):
        raise ValueError("IDs and labels must match the training window count")
    _json_fit_ids(ids)
    weights = shot_weights(ids, labels)
    weights = weights / weights.sum()
    with np.errstate(over="ignore", invalid="ignore"):
        center = np.einsum("n,nd->d", weights, prefix)
        scale = np.maximum(
            np.sqrt(np.einsum("n,nd->d", weights, (prefix - center) ** 2)),
            _FLOOR,
        )
        available = np.repeat(valid, _STATS, axis=1)
        denominator = np.einsum("n,nd->d", weights, available)
        numerator = np.einsum("n,nd->d", weights, summary**2)
        arrival_scale = np.maximum(
            np.sqrt(
                np.divide(
                    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
                )
            ),
            _FLOOR,
        )
    if not all(np.isfinite(v).all() for v in (center, scale, arrival_scale)):
        raise ValueError("training normalization must be finite")
    q0 = np.column_stack(((prefix - center) / scale, valid))
    q1 = np.column_stack((q0, summary / arrival_scale))
    coefficient0, trace0 = logistic_fit(q0, labels, weights, device=device, l2=_L2)
    coefficient1, trace1 = logistic_fit(q1, labels, weights, device=device, l2=_L2)
    return {
        "prefix_center": center.tolist(),
        "prefix_scale": scale.tolist(),
        "arrival_scale": arrival_scale.tolist(),
        "q0_coef": coefficient0.tolist(),
        "q1_coef": coefficient1.tolist(),
        "q0_trace": trace0,
        "q1_trace": trace1,
        "fit_ids": _json_fit_ids(np.unique(ids)),
        "label_shot_counts": {
            str(c): int(len(np.unique(ids[labels == c]))) for c in (0, 1)
        },
        "label_window_counts": {str(c): int(np.sum(labels == c)) for c in (0, 1)},
        "l2": _L2,
    }


def _parameter(fit, name, size, *, scale=False):
    value = np.asarray(fit[name], dtype=np.float64)
    if (
        value.shape != (size,)
        or not np.isfinite(value).all()
        or (scale and np.any(value < _FLOOR))
    ):
        raise ValueError(f"invalid fitted parameter: {name}")
    return value


def score_pair(fit, prefix, target, valid):
    """Score signal-only inputs; no fit, labels, or outcome access is performed.

    Independent models may disagree even on prefix and mask terms. Only the
    q1-minus-constant control cancels those same q1 terms exactly.
    """
    summary = summarize_arrival(target, valid)
    valid = np.asarray(valid)
    prefix = _prefix_array(prefix, len(summary))
    center = _parameter(fit, "prefix_center", _PREFIX)
    scale = _parameter(fit, "prefix_scale", _PREFIX, scale=True)
    arrival_scale = _parameter(fit, "arrival_scale", _CHANNELS * _STATS, scale=True)
    coefficient0 = _parameter(fit, "q0_coef", _PREFIX + _CHANNELS + 1)
    coefficient1 = _parameter(
        fit, "q1_coef", _PREFIX + _CHANNELS + _CHANNELS * _STATS + 1
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q0 = np.column_stack(((prefix - center) / scale, valid))
        q1 = np.column_stack((q0, summary / arrival_scale))
        constant = np.column_stack((q0, np.zeros_like(summary)))
        q0_logit = q0 @ coefficient0[:-1] + coefficient0[-1]
        q1_logit = q1 @ coefficient1[:-1] + coefficient1[-1]
        constant_logit = constant @ coefficient1[:-1] + coefficient1[-1]
        present = valid.any(axis=1)
        paired = np.where(present, q1_logit - q0_logit, 0.0)
        arrival = np.where(present, q1_logit - constant_logit, 0.0)
    if not all(
        np.isfinite(v).all() for v in (q0_logit, q1_logit, constant_logit, paired, arrival)
    ):
        raise ValueError("scored logits and increments must be finite")
    return {
        "q0_logit": q0_logit,
        "q1_logit": q1_logit,
        "constant_logit": constant_logit,
        "paired_increment": paired,
        "arrival_increment": arrival,
    }
