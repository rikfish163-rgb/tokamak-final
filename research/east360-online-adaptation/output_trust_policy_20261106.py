"""Deterministic, single-seed offline action scoring using only NumPy.

``fit_heads`` accepts causal descriptors ``x`` of shape (N, F), integer shot
IDs of shape (N,), utility and guard roles of shape (N,), and ``raw``,
``joint``, ``harm``, and Boolean ``available`` arrays of shape (N, A).
Roles are -1 (unknown), 0 (utility safe / guard safe), or 1 (utility danger /
guard existing timely first). NaN targets are unknown. Targets belonging to
an unknown role or unavailable action are ignored, even if non-finite or out
of range. Otherwise utility targets must be finite and harm must be in [0, 1].

The returned dictionary contains only NumPy arrays, suitable for ``np.savez``
and loading without pickle. Its fields are:

* ``head_names``: (4,) Unicode array: raw, joint, harm_safe, harm_first.
* ``feature_mean``, ``feature_scale``, ``coef``: (4, A, F) float64 arrays.
* ``intercept``: (4, A) float64 array.
* ``shot_counts``: (4, A, 2) int64 array; last axis is role 0, role 1.
  For a guard head, the unused role's count is zero.
* ``eligible``: (A,) Boolean array. All four heads must meet their own
  six-distinct-shot-per-relevant-role requirement, regardless of guard use.
* ``ridge``: scalar float64 array recording the fitted penalty.

Unfitted heads have NaN parameters and NaN predictions. Fitted raw and joint
heads use equal role mass, then equal shot mass within each role, then equal
episode mass within each shot/role. Guard heads use equal shots and equal
episodes within their respective role. For each head/action, these weights
sum to its number of distinct eligible shots (a shot appearing in both utility
roles is counted once). Feature mean and population standard deviation are
computed with those weights, the standard deviation is floored at 0.05, and
standardized features are clipped to [-4, 4] during both fit and inference.
Ridge penalizes slopes but not the intercept; zero ridge is allowed.

``predictions`` returns a tuple of four (B, A) arrays in ``head_names`` order;
only guard predictions are clamped to [0, 1]. ``choose`` uses only this fitted
dictionary, descriptors, availability, objective, and guard flag: no labels,
shot identities, or outcomes are accepted at selection time. It finds the
lowest-index maximizing eligible available action, then returns -1 if its
objective is at most 0.01 or, when guarded, either predicted harm exceeds
0.10. A guard veto does not fall back to another action. It also returns -1
when no action is eligible and available. The fit has no random component.
"""

import numpy as np


_HEAD_NAMES = ("raw", "joint", "harm_safe", "harm_first")
_MIN_SHOTS = 6


def _float_array(value, name, ndim):
    try:
        if np.iscomplexobj(value):
            raise ValueError
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return array


def _features(x, n_features=None):
    x = _float_array(x, "x", 2)
    if n_features is not None and x.shape[1] != n_features:
        raise ValueError("x has the wrong number of features")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    return x


def _roles(value, name, n):
    roles = _float_array(value, name, 1)
    if roles.shape != (n,) or not np.all(np.isin(roles, (-1, 0, 1))):
        raise ValueError(f"{name} must have shape (N,) and values -1, 0, or 1")
    return roles.astype(np.int8)


def _availability(value, shape):
    available = np.asarray(value)
    if available.dtype != np.bool_ or available.shape != shape:
        raise ValueError("available must be Boolean with the expected shape")
    return available


def _ridge_value(ridge):
    ridge = _float_array(ridge, "ridge", 0)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite nonnegative scalar")
    return float(ridge)


def _fit_one(x, shot, roles, target, mask, relevant_roles, ridge):
    """Fit a head after its role-specific shot counts have been checked."""
    features = x[mask]
    target = target[mask]
    selected_shots = shot[mask]
    selected_roles = roles[mask]
    weights = np.empty(len(target), dtype=np.float64)
    role_mass = 1.0 / len(relevant_roles)
    for role in relevant_roles:
        within_role = selected_roles == role
        shots, inverse, counts = np.unique(
            selected_shots[within_role], return_inverse=True, return_counts=True
        )
        weights[within_role] = role_mass / (len(shots) * counts[inverse])
    weights *= len(np.unique(selected_shots))

    mean = np.average(features, axis=0, weights=weights)
    with np.errstate(over="ignore", invalid="ignore"):
        variance = np.average((features - mean) ** 2, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(variance), 0.05)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("feature normalization overflowed")
    standardized = np.clip((features - mean) / scale, -4.0, 4.0)
    # Center explicitly so the intercept is unpenalized. Subtracting a
    # reference value first preserves constant targets exactly at thresholds.
    z_mean = standardized[0] + np.average(
        standardized - standardized[0], axis=0, weights=weights
    )
    target_mean = target[0] + np.average(target - target[0], weights=weights)
    root_weight = np.sqrt(weights)
    weighted_design = (standardized - z_mean) * root_weight[:, None]
    weighted_target = (target - target_mean) * root_weight
    penalty = np.eye(features.shape[1]) * np.sqrt(ridge)
    design_with_penalty = np.concatenate((weighted_design, penalty), axis=0)
    target_with_penalty = np.concatenate((weighted_target, np.zeros(len(penalty))))
    solution = np.linalg.lstsq(design_with_penalty, target_with_penalty, rcond=None)[0]
    intercept = target_mean - z_mean @ solution
    if not np.all(np.isfinite(solution)) or not np.isfinite(intercept):
        raise ValueError("ridge fit produced non-finite parameters")
    return mean, scale, solution, intercept


def fit_heads(
    x, shot, utility_role, guard_role, raw, joint, harm, available, *, ridge=4.0
):
    """Fit action-specific expected rewards and two conditional harm heads.

    See the module docstring for the complete input, weighting, and return
    contracts. Known targets are validated separately for each role/action.
    """
    x = _features(x)
    n, n_features = x.shape
    shot = np.asarray(shot)
    if shot.shape != (n,) or shot.dtype.kind not in "iu":
        raise ValueError("shot must be an integer array with shape (N,)")
    utility_role = _roles(utility_role, "utility_role", n)
    guard_role = _roles(guard_role, "guard_role", n)
    raw = _float_array(raw, "raw", 2)
    if raw.shape[0] != n:
        raise ValueError("raw must have shape (N, A)")
    shape = raw.shape
    joint = _float_array(joint, "joint", 2)
    harm = _float_array(harm, "harm", 2)
    if joint.shape != shape or harm.shape != shape:
        raise ValueError("raw, joint, and harm must have the same shape")
    available = _availability(available, shape)
    ridge = _ridge_value(ridge)

    # Validate only known, available targets for the corresponding role.
    for name, values, roles in (
        ("raw", raw, utility_role),
        ("joint", joint, utility_role),
        ("harm", harm, guard_role),
    ):
        active = available & (roles[:, None] != -1) & ~np.isnan(values)
        if np.any(active & ~np.isfinite(values)):
            raise ValueError(f"known {name} targets must be finite")
        if name == "harm" and np.any(active & ((values < 0) | (values > 1))):
            raise ValueError("known harm targets must be in [0, 1]")

    n_actions = shape[1]
    means = np.full((4, n_actions, n_features), np.nan)
    scales = np.full_like(means, np.nan)
    coefficients = np.full_like(means, np.nan)
    intercepts = np.full((4, n_actions), np.nan)
    shot_counts = np.zeros((4, n_actions, 2), dtype=np.int64)
    specs = (
        (raw, utility_role, (0, 1)),
        (joint, utility_role, (0, 1)),
        (harm, guard_role, (0,)),
        (harm, guard_role, (1,)),
    )
    for head_index, (targets, roles, relevant_roles) in enumerate(specs):
        for action in range(n_actions):
            mask = (
                available[:, action]
                & np.isin(roles, relevant_roles)
                & ~np.isnan(targets[:, action])
            )
            for role in relevant_roles:
                shot_counts[head_index, action, role] = len(
                    np.unique(shot[mask & (roles == role)])
                )
            if any(
                shot_counts[head_index, action, role] < _MIN_SHOTS
                for role in relevant_roles
            ):
                continue
            mean, scale, coef, intercept = _fit_one(
                x, shot, roles, targets[:, action], mask, relevant_roles, ridge
            )
            means[head_index, action] = mean
            scales[head_index, action] = scale
            coefficients[head_index, action] = coef
            intercepts[head_index, action] = intercept

    eligible = (
        np.all(shot_counts[0] >= _MIN_SHOTS, axis=1)
        & np.all(shot_counts[1] >= _MIN_SHOTS, axis=1)
        & (shot_counts[2, :, 0] >= _MIN_SHOTS)
        & (shot_counts[3, :, 1] >= _MIN_SHOTS)
    )
    return {
        "head_names": np.asarray(_HEAD_NAMES),
        "feature_mean": means,
        "feature_scale": scales,
        "coef": coefficients,
        "intercept": intercepts,
        "shot_counts": shot_counts,
        "eligible": eligible,
        "ridge": np.asarray(ridge),
    }


def _read_heads(heads):
    if not isinstance(heads, dict):
        raise ValueError("heads must be a fitted dictionary")
    try:
        names = np.asarray(heads["head_names"])
        mean = _float_array(heads["feature_mean"], "feature_mean", 3)
        scale = _float_array(heads["feature_scale"], "feature_scale", 3)
        coef = _float_array(heads["coef"], "coef", 3)
        intercept = _float_array(heads["intercept"], "intercept", 2)
        counts = np.asarray(heads["shot_counts"])
        eligible = np.asarray(heads["eligible"])
        _ridge_value(heads["ridge"])
    except KeyError as exc:
        raise ValueError("heads is missing a required field") from exc
    if not np.array_equal(names, _HEAD_NAMES) or mean.shape[0] != 4:
        raise ValueError("heads has invalid head names or dimensions")
    n_actions = mean.shape[1]
    if (
        scale.shape != mean.shape
        or coef.shape != mean.shape
        or intercept.shape != (4, n_actions)
        or counts.shape != (4, n_actions, 2)
        or counts.dtype.kind not in "iu"
        or np.any(counts < 0)
        or eligible.shape != (n_actions,)
        or eligible.dtype != np.bool_
    ):
        raise ValueError("heads has invalid parameter, count, or eligibility shapes")
    fitted = np.stack(
        (
            np.all(counts[0] >= _MIN_SHOTS, axis=1),
            np.all(counts[1] >= _MIN_SHOTS, axis=1),
            counts[2, :, 0] >= _MIN_SHOTS,
            counts[3, :, 1] >= _MIN_SHOTS,
        )
    )
    if not np.array_equal(eligible, np.all(fitted, axis=0)):
        raise ValueError("heads eligibility disagrees with shot counts")
    if (
        not np.all(np.isfinite(intercept[fitted]))
        or not np.all(np.isfinite(mean[fitted]))
        or not np.all(np.isfinite(scale[fitted]))
        or not np.all(scale[fitted] > 0)
        or not np.all(np.isfinite(coef[fitted]))
    ):
        raise ValueError("fitted heads must have finite parameters and positive scales")
    return mean, scale, coef, intercept, eligible


def _predict(parameters, x):
    mean, scale, coef, intercept, _ = parameters
    output = []
    for head_index in range(4):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            z = np.clip((x[:, None, :] - mean[head_index]) / scale[head_index], -4, 4)
            value = np.einsum("baf,af->ba", z, coef[head_index]) + intercept[head_index]
        if head_index >= 2:
            value = np.clip(value, 0, 1)
        output.append(value)
    return tuple(output)


def predictions(heads, x):
    """Return (raw, joint, harm_safe, harm_first), each shaped (B, A)."""
    parameters = _read_heads(heads)
    x = _features(x, parameters[0].shape[2])
    return _predict(parameters, x)


def choose(heads, x, available, *, objective="joint", guard=True):
    """Choose an action per row without consuming labels or outcomes."""
    if objective not in ("raw", "joint"):
        raise ValueError("objective must be 'raw' or 'joint'")
    if not isinstance(guard, (bool, np.bool_)):
        raise ValueError("guard must be Boolean")
    parameters = _read_heads(heads)
    x = _features(x, parameters[0].shape[2])
    eligible = parameters[-1]
    available = _availability(available, (len(x), len(eligible)))
    if not len(eligible):
        return np.full(len(x), -1, dtype=np.int64)
    raw, joint, harm_safe, harm_first = _predict(parameters, x)
    reward = raw if objective == "raw" else joint
    valid = available & eligible[None, :] & np.isfinite(reward)
    scores = np.where(valid, reward, -np.inf)
    action = np.argmax(scores, axis=1).astype(np.int64)
    row = np.arange(len(x))
    accepted = np.any(valid, axis=1) & (reward[row, action] > 0.01)
    if guard:
        safe = harm_safe[row, action]
        first = harm_first[row, action]
        accepted &= (
            np.isfinite(safe) & np.isfinite(first) & (safe <= 0.10) & (first <= 0.10)
        )
    return np.where(accepted, action, -1)
