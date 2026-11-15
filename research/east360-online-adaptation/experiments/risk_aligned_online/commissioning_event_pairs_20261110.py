"""Pure-array same-family matching and family-balanced ridge directions.

Endpoints are zero-based sample indices. A safe row may be reused, so a
report should include both ``len({p['shot'] for p, _ in pairs})`` and
``len({s['shot'] for _, s in pairs})`` alongside the number of pairs.
"""

from numbers import Integral

import numpy as np


def _row_arrays(row):
    for name in ("shot", "family", "endpoint"):
        if not isinstance(row[name], Integral):
            raise ValueError(f"{name} must be an integer")
    if not 0 <= row["family"] <= 5:
        raise ValueError("family must be in 0..5")
    if row["endpoint"] < 0:
        raise ValueError("endpoint must be nonnegative")
    context = np.asarray(row["context"], dtype=np.float64)
    feature = np.asarray(row["feature"], dtype=np.float64)
    if context.shape != (3,) or not np.isfinite(context).all():
        raise ValueError("context must be a finite 3-vector")
    if feature.ndim != 1 or not np.isfinite(feature).all():
        raise ValueError("feature must be a finite 1D array")
    return row, context, feature


def match(positive, safe):
    """Match each positive to the nearest eligible safe row, with reuse.

    The context scale is 1.4826 times the coordinatewise MAD of all supplied
    safe contexts, floored at 0.05. Candidates must have the same family and
    lie within 50 inclusive endpoint indices. Their squared context distance
    is augmented by ``(endpoint_difference / 50) ** 2``. Ties use
    safe shot and endpoint, then context and feature for duplicate row IDs.
    Returned pairs are sorted by positive shot and family; unmatched positives
    are omitted. The input rows and arrays are not modified.
    """
    positive = [_row_arrays(row) for row in positive]
    safe = [_row_arrays(row) for row in safe]
    if not positive or not safe:
        return []

    contexts = np.stack([entry[1] for entry in safe])
    median = np.median(contexts, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(contexts - median), axis=0), 0.05)

    groups = {}
    for entry in safe:
        groups.setdefault(entry[0]["family"], []).append(entry)
    for family, entries in groups.items():
        entries.sort(key=lambda entry: entry[0]["endpoint"])
        groups[family] = (
            entries,
            np.array([entry[0]["endpoint"] for entry in entries]),
            np.stack([entry[1] for entry in entries]),
        )

    pairs = []
    for pos, pos_context, _ in sorted(positive, key=lambda entry: (entry[0]["shot"], entry[0]["family"])):
        group = groups.get(pos["family"])
        if group is None:
            continue
        entries, endpoints, safe_contexts = group
        left = np.searchsorted(endpoints, pos["endpoint"] - 50, side="left")
        right = np.searchsorted(endpoints, pos["endpoint"] + 50, side="right")
        if left == right:
            continue
        differences = (safe_contexts[left:right] - pos_context) / scale
        offsets = (endpoints[left:right] - pos["endpoint"]) / 50.0
        distance = np.sum(differences * differences, axis=1) + offsets * offsets
        tied = np.flatnonzero(distance == np.min(distance)) + left
        best = min(
            tied,
            key=lambda index: (
                entries[index][0]["shot"],
                entries[index][0]["endpoint"],
                tuple(entries[index][1]),
                tuple(entries[index][2]),
            ),
        )
        pairs.append((pos, entries[best][0]))
    return pairs


def _pair_arrays(pairs):
    pairs = list(pairs)
    if not pairs:
        raise ValueError("at least one pair is required")
    pairs.sort(key=lambda pair: (pair[0]["family"], pair[0]["shot"], pair[0]["endpoint"],
                                 pair[1]["shot"], pair[1]["endpoint"]))
    differences = []
    families = []
    shape = None
    for pos, safe in pairs:
        family = pos["family"]
        if not isinstance(family, Integral) or not 0 <= family <= 5 or family != safe["family"]:
            raise ValueError("each pair must have the same family in 0..5")
        pos_feature = np.asarray(pos["feature"], dtype=np.float64)
        safe_feature = np.asarray(safe["feature"], dtype=np.float64)
        if pos_feature.ndim != 1 or pos_feature.shape != safe_feature.shape:
            raise ValueError("pair features must be 1D arrays of the same shape")
        if shape is None:
            shape = pos_feature.shape
        if pos_feature.shape != shape or not (np.isfinite(pos_feature).all() and np.isfinite(safe_feature).all()):
            raise ValueError("all pair features must have the same shape and be finite")
        differences.append(pos_feature - safe_feature)
        families.append(family)
    return np.stack(differences), np.asarray(families)


def fit_axis(pairs, *, shuffled=False):
    """Fit the unit ridge direction from positive-minus-safe feature rows.

    The minimized objective is the family-mean squared prediction error plus
    the squared coefficient norm (ridge 1.0). Each family has equal mass and
    each pair within it has equal mass, so all pair weights sum to one. No
    feature scaling or intercept is used. An empty fit raises ValueError; if
    the fitted coefficient is zero, a float32 zero vector is returned.

    The shuffled control changes only targets: alternating +1/-1 signs are
    permuted within each family using seed 131. For an odd-sized family the
    extra sign is randomized, including for a singleton.
    """
    differences, families = _pair_arrays(pairs)
    observed, counts = np.unique(families, return_counts=True)
    weights = np.empty(len(families), dtype=np.float64)
    targets = np.ones(len(families), dtype=np.float64)
    rng = np.random.default_rng(131) if shuffled else None
    for family, count in zip(observed, counts):
        mask = families == family
        weights[mask] = 1.0 / (len(observed) * count)
        if shuffled:
            signs = np.ones(count, dtype=np.float64)
            signs[1::2] = -1.0
            if count % 2 and rng.integers(2) == 0:
                signs *= -1.0
            targets[mask] = rng.permutation(signs)
    covariance = differences.T @ (weights[:, None] * differences)
    rhs = differences.T @ (weights * targets)
    coefficient = np.linalg.solve(covariance + np.eye(differences.shape[1]), rhs)
    norm = np.linalg.norm(coefficient)
    if norm:
        coefficient = coefficient / norm
    return coefficient.astype(np.float32)


def pair_agreement(pairs, axis):
    """Return family-equally weighted directional agreement; ties score 0.5."""
    differences, families = _pair_arrays(pairs)
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (differences.shape[1],) or not np.isfinite(axis).all():
        raise ValueError("axis must be a finite vector matching the feature dimension")
    scores = differences @ axis
    correct = (scores > 0).astype(np.float64) + 0.5 * (scores == 0)
    return float(np.mean([np.mean(correct[families == family]) for family in np.unique(families)]))
