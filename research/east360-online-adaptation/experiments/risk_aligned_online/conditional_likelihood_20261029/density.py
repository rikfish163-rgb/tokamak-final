"""Per-channel, shot-weighted conditional Student-t forecast densities.

``fit_density`` consumes only the supplied training examples. Its returned
dictionary contains a ``channels`` list in input order; each entry has a
``supported`` flag and two ``classes`` entries with ``nrows``, ``nshots`` and
``shot_ids``. Supported classes additionally contain ``ridge``, ``t_scale``
and ``t_cholesky``. Array parameters are CPU tensors so the dictionary supports
``torch.load(..., weights_only=True)`` as well as explicit NumPy/NPZ export.

Scoring returns NumPy arrays and never accepts labels. A channel contributes
only when both fitted classes are supported and all 25 target points are
observed. The caller owns channel selection and the causality of its inputs.
The scale matrix is the specified fitted Student-t scale, not a claim about the
true residual covariance or calibrated predictive uncertainty.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from ..readout_identifiability_20261029.learning import predict, ridge


HORIZON = 25
CHANNELS = 23
DF = 4.0
_RIDGE_PENALTY = 0.1
_JITTER = 1e-6
_T_LOG_CONSTANT = (
    math.lgamma((DF + HORIZON) / 2.0)
    - math.lgamma(DF / 2.0)
    - HORIZON / 2.0 * math.log(DF * math.pi)
)


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _inputs(x, target, valid, *, feature_dim=None):
    x = np.asarray(_numpy(x), dtype=np.float64)
    target = np.asarray(_numpy(target), dtype=np.float64)
    valid = _numpy(valid)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("x must be a finite N-by-D matrix with D > 0")
    if feature_dim is not None and x.shape[1] != feature_dim:
        raise ValueError("feature dimension differs from the fitted model")
    if target.shape != (len(x), HORIZON, CHANNELS):
        raise ValueError("target must have shape N-by-25-by-23")
    if valid.shape != (len(x), CHANNELS) or valid.dtype != np.bool_:
        raise ValueError("valid must be an N-by-23 boolean mask")
    return x, target, valid


def _shot_weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def _torch_parameters(parameters):
    return {
        key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
        for key, value in parameters.items()
    }


def _numpy_parameters(parameters):
    return {key: _numpy(value) if isinstance(value, torch.Tensor) else value
            for key, value in parameters.items()}


def fit_density(x, target, valid, ids, labels, *, device="cpu"):
    """Fit two class-conditional H25 densities for each supported channel.

    A row contributes to a channel only when the caller's validity flag is true
    and all 25 values are finite. Each class must have at least two distinct
    such shot IDs. Each eligible shot has equal total weight within its class,
    both in the specified ridge fit and in the centered residual second moment.
    Labels are used here only, never by the scoring functions.
    """
    x, target, valid = _inputs(x, target, valid)
    ids, labels = _numpy(ids), _numpy(labels)
    if ids.shape != (len(x),) or labels.shape != (len(x),):
        raise ValueError("ids and labels must have one entry per row")
    if ids.dtype.kind not in "biufUS" or (
        ids.dtype.kind in "f" and not np.isfinite(ids).all()
    ):
        raise ValueError("shot IDs must be finite numeric values or strings")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("training labels must be binary 0 or 1")

    complete = valid & np.isfinite(target).all(axis=1)
    channels = []
    for channel in range(CHANNELS):
        masks = [complete[:, channel] & (labels == label) for label in (0, 1)]
        classes = []
        for mask in masks:
            unique = np.unique(ids[mask])
            classes.append({
                "nrows": int(mask.sum()),
                "nshots": len(unique),
                "shot_ids": [item.item() if isinstance(item, np.generic) else item
                             for item in unique],
                "ridge": None,
                "t_scale": None,
                "t_cholesky": None,
            })
        supported = all(entry["nshots"] >= 2 for entry in classes)
        if supported:
            for mask, entry in zip(masks, classes, strict=True):
                xc, yc, shot_ids = x[mask], target[mask, :, channel], ids[mask]
                regression = ridge(xc, yc, shot_ids, _RIDGE_PENALTY, device=device)
                residual = yc - predict(regression, xc)
                weights = _shot_weights(shot_ids)
                centered = residual - np.average(residual, axis=0, weights=weights)
                covariance = centered.T @ (weights[:, None] * centered)
                scale = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
                scale += _JITTER * np.eye(HORIZON)
                if not np.isfinite(scale).all():
                    raise FloatingPointError("nonfinite fitted Student-t scale")
                chol = np.linalg.cholesky(scale)
                entry.update(
                    ridge=_torch_parameters(regression),
                    t_scale=torch.from_numpy(scale.copy()),
                    t_cholesky=torch.from_numpy(chol.copy()),
                )
        channels.append({"supported": supported, "classes": classes})
    return {
        "version": 1,
        "feature_dim": int(x.shape[1]),
        "horizon": HORIZON,
        "n_channels": CHANNELS,
        "df": DF,
        "channels": channels,
    }


def _student_t_log_density(target, mean, chol):
    whitened = np.linalg.solve(chol, (target - mean).T).T
    if not np.isfinite(whitened).all():
        raise FloatingPointError("nonfinite whitened Student-t residual")
    # Form log(1 + ||whitened||^2 / nu) without squaring very large values.
    magnitude = np.max(np.abs(whitened), axis=1)
    log_ratio = np.full(len(magnitude), -np.inf)
    nonzero = magnitude > 0
    scaled = whitened[nonzero] / magnitude[nonzero, None]
    log_ratio[nonzero] = (
        2.0 * np.log(magnitude[nonzero])
        + np.log(np.square(scaled).sum(axis=1))
        - math.log(DF)
    )
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    return _T_LOG_CONSTANT - 0.5 * log_det - (DF + HORIZON) / 2.0 * np.logaddexp(
        0.0, log_ratio
    )


def channel_log_densities(fit, x, target, valid):
    """Return per-channel log densities (N,23,2) and shared used mask (N,23).

    Unused channel slots are exactly zero for both classes, so summing over
    channels produces the same factorized log densities as ``log_densities``.
    The used mask must be consulted to distinguish absence from zero evidence.
    """
    if (fit.get("version") != 1 or fit.get("horizon") != HORIZON
            or fit.get("n_channels") != CHANNELS or fit.get("df") != DF
            or len(fit.get("channels", ())) != CHANNELS):
        raise ValueError("incompatible conditional density model")
    x, target, valid = _inputs(x, target, valid, feature_dim=fit["feature_dim"])
    complete = valid & np.isfinite(target).all(axis=1)
    result = np.zeros((len(x), CHANNELS, 2), dtype=np.float64)
    used = np.zeros((len(x), CHANNELS), dtype=bool)
    for channel, channel_fit in enumerate(fit["channels"]):
        if not channel_fit["supported"]:
            continue
        mask = complete[:, channel]
        if not mask.any():
            continue
        used[:, channel] = mask
        for label, class_fit in enumerate(channel_fit["classes"]):
            mean = predict(_numpy_parameters(class_fit["ridge"]), x[mask])
            result[mask, channel, label] = _student_t_log_density(
                target[mask, :, channel], mean, _numpy(class_fit["t_cholesky"])
            )
    return result, used


def log_densities(fit, x, target, valid):
    """Return class log-density sums (N,2) and supported-channel counts (N,)."""
    by_channel, used = channel_log_densities(fit, x, target, valid)
    return by_channel.sum(axis=1), used.sum(axis=1)


def mixture_feedback(logit, logs):
    """Return per-row mixture NLL, class-1 posterior, and dNLL/dlogit.

    Density values are detached here even if the caller supplied tensors with
    gradients. The equal-density branch is exactly independent of the logit,
    including when both class densities are zero evidence or very large.
    """
    if (not isinstance(logit, torch.Tensor) or not isinstance(logs, torch.Tensor)
            or logit.ndim != 1 or logs.shape != (len(logit), 2)
            or not logit.is_floating_point() or not logs.is_floating_point()
            or logit.device != logs.device):
        raise ValueError("expected same-device floating logit (N,) and logs (N,2)")
    if not torch.isfinite(logit).all() or not torch.isfinite(logs).all():
        raise ValueError("logits and log densities must be finite")
    dtype = torch.promote_types(logit.dtype, logs.dtype)
    z, density = logit.to(dtype=dtype), logs.detach().to(dtype=dtype)
    center = density.max(dim=1).values
    terms = torch.stack((F.logsigmoid(-z), F.logsigmoid(z)), dim=1)
    relative = terms + (density - center[:, None])
    loss = -center - torch.logsumexp(relative, dim=1)
    probability = torch.sigmoid(z)
    posterior = torch.sigmoid(z + (density[:, 1] - density[:, 0]))
    equal = density[:, 0] == density[:, 1]
    loss = torch.where(equal, -density[:, 0] + z * 0, loss)
    posterior = torch.where(equal, probability, posterior)
    return loss, posterior, probability - posterior
