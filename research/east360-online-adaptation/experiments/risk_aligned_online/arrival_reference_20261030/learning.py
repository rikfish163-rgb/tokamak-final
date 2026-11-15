"""Shot-weighted arrival references and fixed-classifier feedback.

The caller supplies a training fold, not held-out rows.  Label 0 identifies
operational-negative 125 ms windows; it does not establish healthy plasma.
The frozen pair supplies all normalization and the q1 arrival coefficients.
Prediction and feedback consume only signal-derived inputs, never labels.
"""

import numpy as np

from ..arrival_increment_20261030.learning import _json_fit_ids, summarize_arrival
from ..conditional_forecast_20261023.learning import ridge_fit


_PREFIX = 64
_CHANNELS = 23
_STATS = 4
_HORIZON = 25
_L2 = 0.1


def _parameter(pair, name, shape, *, positive=False):
    try:
        value = np.asarray(pair[name], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid pair parameter: {name}") from exc
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid pair parameter: {name}")
    if positive and np.any(value <= 0):
        raise ValueError(f"pair parameter must be positive: {name}")
    return value


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return value


def _prefix(value, n):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n, _PREFIX) or not np.isfinite(value).all():
        raise ValueError("prefix must be a finite N-by-64 matrix")
    return value


def _groups(values):
    """JSON-compatible shot IDs, first-seen order, and group indices."""
    positions = {}
    unique = []
    inverse = []
    for value in values:
        if value not in positions:
            positions[value] = len(unique)
            unique.append(value)
        inverse.append(positions[value])
    return unique, np.asarray(inverse, dtype=int)


def fit_reference(pair, prefix, target, valid, ids, labels, *, device="cpu"):
    """Fit one ridge per channel, using only observed label-0 training rows.

    ``fit_ids`` records all supplied training-shot IDs. ``fit_row_indices``
    identifies label-0 rows in that supplied array, and each channel records
    its eligible row indices and IDs. Unsupported channels retain these counts
    for auditing but have no coefficients. Their mean is descriptive only;
    the zero mean for an empty cohort is a placeholder, not a prediction.
    """
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 3 or target.shape[1:] != (_HORIZON, _CHANNELS):
        raise ValueError("target must be N-by-25-by-23")
    n = len(target)
    valid = _mask(valid, (n, _CHANNELS), "valid")
    prefix = np.asarray(prefix, dtype=np.float64)
    if prefix.shape != (n, _PREFIX):
        raise ValueError("prefix must be an N-by-64 matrix")
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    if ids.shape != (n,) or labels.shape != (n,):
        raise ValueError("IDs and labels must match the training window count")
    json_ids = _json_fit_ids(ids)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    center = _parameter(pair, "prefix_center", (_PREFIX,))
    scale = _parameter(pair, "prefix_scale", (_PREFIX,), positive=True)
    arrival_scale = _parameter(
        pair, "arrival_scale", (_CHANNELS * _STATS,), positive=True
    ).reshape(_CHANNELS, _STATS)

    train_rows = np.flatnonzero(labels == 0)
    train_prefix = _prefix(prefix[train_rows], len(train_rows))
    train_valid = valid[train_rows]
    summary = summarize_arrival(target[train_rows], train_valid).reshape(
        len(train_rows), _CHANNELS, _STATS
    )
    train_ids = [json_ids[index] for index in train_rows]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = (train_prefix - center) / scale
        y = summary / arrival_scale
    if not np.isfinite(h).all() or not np.isfinite(y).all():
        raise ValueError("normalized training values must be finite")

    supported = []
    channels = []
    for channel in range(_CHANNELS):
        selected = np.flatnonzero(train_valid[:, channel])
        shot_ids, inverse = _groups([train_ids[index] for index in selected])
        nshots = len(shot_ids)
        nrows = len(selected)
        mean = np.zeros(_STATS, dtype=np.float64)
        coefficients = None
        reason = None
        if nrows:
            counts = np.bincount(inverse, minlength=nshots)
            weights = 1.0 / (nshots * counts[inverse])
            with np.errstate(over="ignore", invalid="ignore"):
                mean = np.sum(weights[:, None] * y[selected, channel], axis=0)
            if not np.isfinite(mean).all():
                raise ValueError("weighted target mean must be finite")
        if nshots < 2:
            reason = "no_complete_windows" if nshots == 0 else "fewer_than_two_shots"
        else:
            coefficients = ridge_fit(
                h[selected], y[selected, channel], weights, _L2, device=device
            )
            if (
                np.shape(coefficients) != (_PREFIX + 1, _STATS)
                or not np.isfinite(coefficients).all()
            ):
                raise ValueError("invalid fitted ridge coefficients")
            coefficients = np.asarray(coefficients).tolist()
        supported.append(reason is None)
        channels.append(
            {
                "coefficients": coefficients,
                "mean": mean.tolist(),
                "fit_ids": shot_ids,
                "nrows": nrows,
                "nshots": nshots,
                "fit_row_indices": train_rows[selected].tolist(),
                "unsupported_reason": reason,
            }
        )

    return {
        "supported": supported,
        "channels": channels,
        "fit_ids": _groups(json_ids)[0],
        "reference_fit_ids": _groups(train_ids)[0],
        "fit_row_indices": train_rows.tolist(),
        "l2": _L2,
    }


def predict_reference(ref, pair, prefix, valid, *, mode="context"):
    """Return standardized arrival predictions, zeroed if absent/unsupported.

    The ridge input is origin-known normalized prefix only; ``valid`` is used
    solely as an output mask. RMS is projected to [0, infinity), its convex
    feasible range. This deterministic range projection is not a fitted
    threshold; the three signed statistics remain unconstrained.
    """
    if mode not in ("context", "mean"):
        raise ValueError("mode must be context or mean")
    prefix = np.asarray(prefix, dtype=np.float64)
    if prefix.ndim != 2:
        raise ValueError("prefix must be a finite N-by-64 matrix")
    prefix = _prefix(prefix, len(prefix))
    valid = _mask(valid, (len(prefix), _CHANNELS), "valid")
    center = _parameter(pair, "prefix_center", (_PREFIX,))
    scale = _parameter(pair, "prefix_scale", (_PREFIX,), positive=True)
    _parameter(pair, "arrival_scale", (_CHANNELS * _STATS,), positive=True)
    try:
        supported = _mask(ref["supported"], (_CHANNELS,), "supported")
        channels = ref["channels"]
    except (KeyError, TypeError) as exc:
        raise ValueError("reference must contain supported and channels") from exc
    if not isinstance(channels, (list, tuple)) or len(channels) != _CHANNELS:
        raise ValueError("reference must contain 23 channels")
    if mode == "context":
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            h = (prefix - center) / scale
        if not np.isfinite(h).all():
            raise ValueError("normalized prefix must be finite")
        design = np.column_stack((h, np.ones(len(prefix))))
    result = np.zeros((len(prefix), _CHANNELS, _STATS), dtype=np.float64)
    for channel in np.flatnonzero(supported):
        key = "coefficients" if mode == "context" else "mean"
        shape = (_PREFIX + 1, _STATS) if mode == "context" else (_STATS,)
        try:
            parameter = np.asarray(channels[channel][key], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid reference {key} for channel {channel}") from exc
        if parameter.shape != shape or not np.isfinite(parameter).all():
            raise ValueError(f"invalid reference {key} for channel {channel}")
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = design @ parameter if mode == "context" else parameter
        if not np.isfinite(prediction).all():
            raise ValueError("reference predictions must be finite")
        result[:, channel] = np.where(valid[:, channel, None], prediction, 0.0)
    result[:, :, 2] = np.maximum(0.0, result[:, :, 2])
    return result


def feedback(pair, target, valid, reference, supported):
    """Return fixed-q1 weighted observed-minus-reference and channel counts.

    ``reference`` is already standardized in the pair's arrival-scale units;
    only channels both completely observed and explicitly supported contribute.
    """
    summary = summarize_arrival(target, valid).reshape(-1, _CHANNELS, _STATS)
    valid = _mask(valid, (len(summary), _CHANNELS), "valid")
    supported = _mask(supported, (_CHANNELS,), "supported")
    try:
        reference = np.asarray(reference, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference must be a finite N-by-23-by-4 array") from exc
    if reference.shape != summary.shape or not np.isfinite(reference).all():
        raise ValueError("reference must be a finite N-by-23-by-4 array")
    scale = _parameter(pair, "arrival_scale", (_CHANNELS * _STATS,), positive=True)
    coefficients = _parameter(
        pair, "q1_coef", (_PREFIX + _CHANNELS + _CHANNELS * _STATS + 1,)
    )[_PREFIX + _CHANNELS : -1].reshape(_CHANNELS, _STATS)
    mask = valid & supported[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        observed = np.divide(
            summary,
            scale.reshape(_CHANNELS, _STATS),
            out=np.zeros_like(summary),
            where=mask[:, :, None],
        )
        residual = np.subtract(
            observed, reference, out=np.zeros_like(summary), where=mask[:, :, None]
        )
        increment = np.einsum("ncs,cs->n", residual, coefficients)
    if not np.isfinite(increment).all():
        raise ValueError("feedback increment must be finite")
    return increment, mask.sum(axis=1)
