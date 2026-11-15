"""Deterministic, whole-shot-weighted ridge for synthetic action utilities.

The caller is responsible for choosing disjoint training/calibration shots.
Action 0 is abstention: it is never fitted, and its prediction is exactly zero.
This module performs no data loading, calibration, or file writes.
"""

from dataclasses import dataclass
import hashlib
from numbers import Integral, Real

import numpy as np


def _boolean_array(value, name, ndim):
    result = np.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if result.dtype.kind != "b":
        raise TypeError(f"{name} must contain booleans")
    return result


def _real_array(value, name, ndim, *, allow_nonfinite=False):
    result = np.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if result.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float64, copy=False)
    if not allow_nonfinite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _integer_vector(value, name):
    # Object conversion preserves arbitrary-size Python IDs and rejects floats
    # instead of silently truncating them to integers.
    result = np.asarray(value, dtype=object)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if any(not isinstance(item, Integral) or isinstance(item, (bool, np.bool_)) for item in result):
        raise TypeError(f"{name} must contain integers, not booleans")
    return [int(item) for item in result]


def _allowed_array(allowed):
    allowed = _boolean_array(allowed, "allowed", 2)
    if allowed.shape[1] == 0:
        raise ValueError("allowed must include action 0")
    if not allowed[:, 0].all():
        raise ValueError("action 0 must be allowed in every block")
    return allowed


def _nonnegative_real(value, name, *, allow_positive_infinity=False):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if np.isnan(value) or value < 0 or (np.isinf(value) and not allow_positive_infinity):
        raise ValueError(f"{name} must be nonnegative" + ("" if allow_positive_infinity else " and finite"))
    return value


def shot_folds(shot_ids, n=5):
    """Assign unique integer IDs by SHA-256(decimal ID), then round robin.

    Assignment uses no labels and is invariant to the order of the same set of
    IDs. A hash collision is resolved by the numeric ID. Changing the set of IDs
    can change existing assignments, so callers must split the intended set once.
    """
    ids = _integer_vector(shot_ids, "shot_ids")
    if not isinstance(n, Integral) or isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be a positive integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if len(set(ids)) != len(ids):
        raise ValueError("shot_ids must be distinct")
    order = sorted(
        range(len(ids)),
        key=lambda i: (hashlib.sha256(str(ids[i]).encode("ascii")).digest(), ids[i]),
    )
    result = np.empty(len(ids), dtype=np.int64)
    for rank, original_index in enumerate(order):
        result[original_index] = rank % int(n)
    return result


def shot_weights(block_shot, disruptive, train_shots, allowed):
    """Return [B,A] loss weights with class/shot/block/action balance.

    Only training blocks with at least one available positive action count.
    Each supported class receives equal total weight, then supported shots in
    that class, eligible blocks in each shot, and available positive actions in
    each block receive equal shares. All other entries have zero weight.
    """
    allowed = _allowed_array(allowed)
    disruptive = _boolean_array(disruptive, "disruptive", 1)
    train_shots = _boolean_array(train_shots, "train_shots", 1)
    if disruptive.shape != train_shots.shape:
        raise ValueError("disruptive and train_shots must have the same shape")
    block_ids = _integer_vector(block_shot, "block_shot")
    if len(block_ids) != allowed.shape[0]:
        raise ValueError("block_shot must have one entry per block")
    if any(shot < 0 or shot >= len(disruptive) for shot in block_ids):
        raise ValueError("block_shot contains an out-of-range shot position")
    block_ids = np.asarray(block_ids, dtype=np.intp)
    action_counts = allowed[:, 1:].sum(axis=1)
    eligible = train_shots[block_ids] & (action_counts > 0)
    block_indices = np.flatnonzero(eligible)
    if block_indices.size == 0:
        raise ValueError("no eligible positive-action training support")

    shot_block_counts = np.bincount(block_ids[block_indices], minlength=len(disruptive))
    supported_shots = np.flatnonzero(shot_block_counts)
    class_shot_counts = np.bincount(disruptive[supported_shots].astype(np.intp), minlength=2)
    supported_classes = np.count_nonzero(class_shot_counts)
    block_shots = block_ids[block_indices]
    class_ids = disruptive[block_shots].astype(np.intp)
    per_action = (
        1.0
        / supported_classes
        / class_shot_counts[class_ids]
        / shot_block_counts[block_shots]
        / action_counts[block_indices]
    )
    weights = np.zeros(allowed.shape, dtype=np.float64)
    weights[block_indices, 1:] = per_action[:, None] * allowed[block_indices, 1:]
    return weights


@dataclass(frozen=True, slots=True)
class Model:
    """Frozen coefficients and training-only feature normalization."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficient: tuple[float, ...]
    intercept: float

    def predict(self, x, allowed):
        """Score available positive actions; anchor 0 and mask unavailable ones."""
        x = _real_array(x, "x", 3)
        allowed = _allowed_array(allowed)
        if x.shape[:2] != allowed.shape or x.shape[2] != len(self.mean):
            raise ValueError("x and allowed must match each other and the fitted feature count")
        result = np.full(allowed.shape, -np.inf, dtype=np.float64)
        result[:, 0] = 0.0
        positive = allowed.copy()
        positive[:, 0] = False
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            score = ((x[positive] - np.asarray(self.mean)) / np.asarray(self.scale)) @ np.asarray(self.coefficient)
            score = score + self.intercept
        if not np.isfinite(score).all():
            raise ValueError("prediction is not representable as a finite float")
        result[positive] = score
        return result


def fit_model(x, y, allowed, block_shot, disruptive, train_shots, *, alpha=0.03):
    """Fit weighted standardized ridge with a free intercept and fixed alpha.

    The objective is sum_i(w_i * (y_i - intercept - z_i @ beta)**2)
    + alpha * ||beta||**2, with sum_i(w_i) = 1. Means and standard
    deviations are weighted using only selected positive-action train rows.
    A constant feature is centered to zero and assigned scale 1.
    """
    alpha = _nonnegative_real(alpha, "alpha")
    x = _real_array(x, "x", 3)
    y = _real_array(y, "y", 2)
    allowed = _allowed_array(allowed)
    if x.shape[:2] != allowed.shape or y.shape != allowed.shape or x.shape[2] == 0:
        raise ValueError("x, y, and allowed must have matching block/action dimensions and nonempty features")
    if not np.all(y[:, 0] == 0):
        raise ValueError("action 0 must have exactly zero target")
    weights = shot_weights(block_shot, disruptive, train_shots, allowed)
    selected = weights > 0
    train_x, train_y = x[selected], y[selected]
    w = weights[selected]
    w = w / w.sum()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = np.average(train_x, axis=0, weights=w)
        constant = np.min(train_x, axis=0) == np.max(train_x, axis=0)
        mean[constant] = train_x[0, constant]
        centered = train_x - mean
        amplitude = np.max(np.abs(centered), axis=0)
        amplitude[constant] = 1.0
        scaled_centered = centered / amplitude
        scale = amplitude * np.sqrt(np.average(scaled_centered * scaled_centered, axis=0, weights=w))
        scale[constant] = 1.0
        z = centered / scale
        target_mean = np.average(train_y, weights=w)
        weighted_z = np.sqrt(w)[:, None] * z
        weighted_y = np.sqrt(w) * (train_y - target_mean)
    if not all(np.isfinite(v).all() for v in (mean, scale, z, target_mean, weighted_z, weighted_y)):
        raise ValueError("training statistics are not representable as finite floats")

    # The augmented system implements the sum-of-weights-one loss above, not
    # a penalty that changes with the number of blocks or action rows.
    design = np.vstack((weighted_z, np.sqrt(alpha) * np.eye(x.shape[2])))
    target = np.concatenate((weighted_y, np.zeros(x.shape[2])))
    coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
    if not np.isfinite(coefficient).all():
        raise ValueError("ridge coefficients are not representable as finite floats")
    return Model(tuple(mean), tuple(scale), tuple(coefficient), float(target_mean))


def choices(prediction, allowed, margin):
    """Choose the first best finite available action strictly above margin.

    Non-finite scores cannot be selected. Ties are broken by the smallest
    actual action index, and action 0 is returned when nothing clears margin.
    """
    margin = _nonnegative_real(margin, "margin", allow_positive_infinity=True)
    prediction = _real_array(prediction, "prediction", 2, allow_nonfinite=True)
    allowed = _allowed_array(allowed)
    if prediction.shape != allowed.shape:
        raise ValueError("prediction and allowed must have the same shape")
    candidates = allowed & np.isfinite(prediction)
    candidates[:, 0] = False
    scores = np.where(candidates, prediction, -np.inf)
    best = np.argmax(scores, axis=1)
    best_score = scores[np.arange(scores.shape[0]), best]
    return np.where(np.isfinite(best_score) & (best_score > margin), best, 0).astype(np.int64)
