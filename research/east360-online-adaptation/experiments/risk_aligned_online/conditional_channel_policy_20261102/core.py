"""CPU-only conditional channel weights fitted to supplied first-order slopes.

The slopes decompose an existing, shared-cap step. Reweighting the underlying
loss can also change its gradient and cap, neither of which is modeled here.
This module does not perform a finite-step network or alarm evaluation.
"""

import numpy as np


_SCHEMA = "conditional_channel_policy_20261102/v1"
_CHANNELS = 23
_TRAIN_GROUPS = ("safe", "danger30_125")
_ALL_GROUPS = (*_TRAIN_GROUPS, "disruptive_early")
_STEPS = 256


def _finite_array(value, name):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def _features(value):
    x = _finite_array(value, "x")
    if x.ndim != 2 or x.shape[1] not in (0, 6):
        raise ValueError("x must have shape N x 6 or N x 0")
    if x.shape[1] == 6 and (
        (x < 0).any()
        or (x > 1).any()
        or not np.allclose(x.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    ):
        raise ValueError("x must contain six-class probability rows")
    return x


def _matrix(value, rows, name):
    result = _finite_array(value, name)
    if result.shape != (rows, _CHANNELS):
        raise ValueError(f"{name} must have shape N x 23")
    return result


def _group_weights(shot_ids, groups, rows, *, training):
    ids = np.asarray(shot_ids)
    labels = np.asarray(groups)
    if ids.shape != (rows,) or ids.dtype.kind not in "iu":
        raise ValueError("shot_ids must be an N-vector of integers")
    if labels.shape != (rows,) or not all(isinstance(g, str) for g in labels):
        raise ValueError("groups must be an N-vector of group names")
    present = set(labels.tolist())
    if training:
        if present != set(_TRAIN_GROUPS):
            raise ValueError("fit requires both safe and danger30_125 only")
    elif not present or not present.issubset(_ALL_GROUPS):
        raise ValueError("unknown or empty evaluation groups")

    safe_ids = set(ids[labels == "safe"].tolist())
    disruptive_ids = set(ids[labels != "safe"].tolist())
    if safe_ids & disruptive_ids:
        raise ValueError("a shot_id cannot be both safe and disruptive")

    result = {}
    for group in _ALL_GROUPS:
        if group not in present:
            continue
        mask = labels == group
        shots, inverse, counts = np.unique(
            ids[mask], return_inverse=True, return_counts=True
        )
        coefficients = np.zeros(rows, dtype=np.float64)
        coefficients[mask] = 1.0 / (len(shots) * counts[inverse])
        result[group] = coefficients
    return ids, result


def _weights(x, params):
    with np.errstate(over="ignore", invalid="ignore"):
        scores = x @ params[:-1] + params[-1]
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite channel policy scores")
    u = np.tanh(scores)
    return 1.0 + 0.25 * (u - u.mean(axis=1, keepdims=True)), u


def _changes(contributions, weights, coefficients):
    with np.errstate(over="ignore", invalid="ignore"):
        row_delta = ((weights - 1.0) * contributions).sum(axis=1)
        result = {group: float(c @ row_delta) for group, c in coefficients.items()}
    if not np.isfinite(row_delta).all() or not all(
        np.isfinite(v) for v in result.values()
    ):
        raise FloatingPointError("nonfinite first-order risk change")
    return result


def fit(x, contributions, shot_ids, groups, *, steps=256):
    """Fit exactly 256 full-batch Adam steps on the supplied training rows.

    Group labels and shot IDs affect the training objective and receipt only;
    they never enter the prediction function. History is evaluated immediately
    before each update, so the final model is the state after update 256.
    """
    if type(steps) is not int or steps != _STEPS:
        raise ValueError("the fixed training protocol requires 256 integer steps")
    x = _features(x)
    contributions = _matrix(contributions, len(x), "contributions")
    ids, coefficients = _group_weights(shot_ids, groups, len(x), training=True)
    params = np.zeros((x.shape[1] + 1, _CHANNELS), dtype=np.float64)
    first = np.zeros_like(params)
    second = np.zeros_like(params)
    history = []

    for step in range(1, steps + 1):
        weights, u = _weights(x, params)
        deltas = _changes(contributions, weights, coefficients)
        with np.errstate(over="ignore", invalid="ignore"):
            objective = max(deltas.values()) / 0.01 + 0.01 * np.mean(params**2)
        if not np.isfinite(objective):
            raise FloatingPointError("nonfinite fitting objective")
        history.append(dict(step=step, group_deltas=deltas, obj=float(objective)))

        safe, danger = (deltas[group] for group in _TRAIN_GROUPS)
        # torch.maximum uses the average of the two subgradients at an exact tie.
        safe_fraction = 0.5 if safe == danger else float(safe > danger)
        sample_weight = (
            safe_fraction * coefficients["safe"]
            + (1.0 - safe_fraction) * coefficients["danger30_125"]
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            centered = contributions - contributions.mean(axis=1, keepdims=True)
            dz = (0.25 / 0.01) * sample_weight[:, None] * centered * (1.0 - u**2)
            gradient = np.empty_like(params)
            gradient[:-1] = x.T @ dz
            gradient[-1] = dz.sum(axis=0)
            gradient += (0.02 / params.size) * params
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            corrected_first = first / (1.0 - 0.9**step)
            corrected_second = second / (1.0 - 0.999**step)
            params -= 0.01 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        if not all(np.isfinite(v).all() for v in (gradient, first, second, params)):
            raise FloatingPointError("nonfinite Adam state")

    return dict(
        schema=_SCHEMA,
        feature_dim=int(x.shape[1]),
        A=params[:-1].tolist(),
        b=params[-1].tolist(),
        fit_ids=sorted({int(shot) for shot in ids}),
        history=history,
        history_evaluation="before_each_update",
        steps=steps,
        optimizer=dict(name="Adam", learning_rate=0.01, betas=[0.9, 0.999], eps=1e-8),
        scope=(
            "FIRST_ORDER_SHARED_CAP_LINEAR_PROXY_ONLY; "
            "NOT_RECOMPUTED_GRADIENT_CAP_FINITE_STEP_OR_ALARM"
        ),
    )


def predict(x, fitted):
    """Return N x 23 weights using only x and a validated serialized model."""
    x = _features(x)
    if not isinstance(fitted, dict) or fitted.get("schema") != _SCHEMA:
        raise ValueError("unsupported fitted model schema")
    feature_dim = fitted.get("feature_dim")
    if type(feature_dim) is not int or feature_dim not in (0, 6):
        raise ValueError("invalid fitted feature dimension")
    if x.shape[1] != feature_dim:
        raise ValueError("x does not match the fitted feature dimension")
    try:
        a = _finite_array(fitted["A"], "A")
        b = _finite_array(fitted["b"], "b")
    except KeyError as exc:
        raise ValueError("fitted model is missing A or b") from exc
    if feature_dim == 0 and a.shape == (0,):
        a = a.reshape(0, _CHANNELS)
    if a.shape != (feature_dim, _CHANNELS) or b.shape != (_CHANNELS,):
        raise ValueError("fitted A or b has an invalid shape")
    return _weights(x, np.concatenate((a, b[None, :]), axis=0))[0]


def group_changes(contributions, weights, shot_ids, groups):
    """Return each present group's within-shot, then across-shot mean delta."""
    contributions = _finite_array(contributions, "contributions")
    if contributions.ndim != 2 or contributions.shape[1] != _CHANNELS:
        raise ValueError("contributions must have shape N x 23")
    weights = _matrix(weights, len(contributions), "weights")
    _, coefficients = _group_weights(
        shot_ids, groups, len(contributions), training=False
    )
    return _changes(contributions, weights, coefficients)
