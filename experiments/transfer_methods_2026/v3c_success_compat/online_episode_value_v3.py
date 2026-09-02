#!/usr/bin/env python3
"""Causal episode confirmation and decision-aware active-query value.

The module is intentionally outcome-blind at prediction time.  Every class
first seals a decision from current/past observations, and accepts labels only
through an explicit selected-query update afterwards.  It contains no HDF5
access and cannot reveal an unqueried outcome by itself.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _unit(value: float) -> float:
    value = float(value)
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class _RegularizedOnlineLogistic:
    """Small deterministic classifier refit only from queried examples."""

    def __init__(
        self, *, feature_dim: int, minimum_history: int,
        minimum_per_class: int, l2: float = 0.10,
        capacity: int = 160,
    ) -> None:
        if feature_dim <= 0 or minimum_history <= 0 or minimum_per_class <= 0:
            raise ValueError("invalid online logistic geometry")
        if l2 <= 0 or capacity < minimum_history:
            raise ValueError("invalid online logistic regularization")
        self.feature_dim = int(feature_dim)
        self.minimum_history = int(minimum_history)
        self.minimum_per_class = int(minimum_per_class)
        self.l2 = float(l2)
        self.capacity = int(capacity)
        self._history: list[dict[str, Any]] = []
        self._center = np.zeros(self.feature_dim, dtype=np.float64)
        self._scale = np.ones(self.feature_dim, dtype=np.float64)
        self._weight = np.zeros(self.feature_dim, dtype=np.float64)
        self._bias = 0.0
        self._fitted = False

    def _vector(self, features: np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if len(vector) != self.feature_dim or not np.isfinite(vector).all():
            raise ValueError("online logistic feature vector is invalid")
        return np.clip(vector, -10.0, 10.0)

    def _class_counts(self) -> tuple[int, int]:
        negative = sum(int(item["target"] == 0) for item in self._history)
        return negative, len(self._history) - negative

    def confidence(self) -> float:
        negative, positive = self._class_counts()
        history = min(len(self._history) / self.minimum_history, 1.0)
        classes = min(
            negative / self.minimum_per_class,
            positive / self.minimum_per_class,
            1.0,
        )
        return float(max(0.0, history * classes))

    def _fit(self) -> None:
        negative, positive = self._class_counts()
        if (
            len(self._history) < self.minimum_history
            or min(negative, positive) < self.minimum_per_class
        ):
            self._fitted = False
            return
        matrix = np.stack([item["features"] for item in self._history])
        target = np.asarray([item["target"] for item in self._history], np.float64)
        self._center = matrix.mean(axis=0)
        self._scale = np.maximum(matrix.std(axis=0), 0.10)
        standardized = np.clip(
            (matrix - self._center) / self._scale, -8.0, 8.0,
        )
        counts = np.bincount(target.astype(np.int64), minlength=2)
        sample_weight = np.asarray([
            len(target) / (2.0 * max(int(counts[int(value)]), 1))
            for value in target
        ])
        weight = np.zeros(self.feature_dim, dtype=np.float64)
        prevalence = float(np.clip(target.mean(), 1e-4, 1.0 - 1e-4))
        bias = math.log(prevalence / (1.0 - prevalence))
        denominator = float(sample_weight.sum())
        for step in range(160):
            probability = np.asarray(_sigmoid(standardized @ weight + bias))
            residual = (probability - target) * sample_weight
            gradient = standardized.T @ residual / denominator + self.l2 * weight
            gradient_bias = float(residual.sum() / denominator)
            learning_rate = 0.25 / math.sqrt(1.0 + step / 20.0)
            weight -= learning_rate * gradient
            bias -= learning_rate * gradient_bias
        self._weight = np.clip(weight, -6.0, 6.0)
        self._bias = float(np.clip(bias, -6.0, 6.0))
        self._fitted = True

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        vector = self._vector(features)
        positive = sum(int(item["target"] == 1) for item in self._history)
        prior = (positive + 1.0) / (len(self._history) + 2.0)
        confidence = self.confidence()
        if not self._fitted:
            return float(prior), confidence
        standardized = np.clip((vector - self._center) / self._scale, -8.0, 8.0)
        fitted = float(_sigmoid(standardized @ self._weight + self._bias))
        probability = (1.0 - confidence) * prior + confidence * fitted
        return _unit(probability), confidence

    def geometry(
        self, features: np.ndarray, *, shot_id: int, radius: float = 1.5,
    ) -> dict[str, float]:
        vector = self._vector(features)
        if not self._history:
            return {"novelty": 1.0, "density": 0.0, "coverage_debt": 1.0}
        matrix = np.stack([item["features"] for item in self._history])
        scale = np.maximum(matrix.std(axis=0), 0.25)
        distance = np.linalg.norm((matrix - vector) / scale, axis=1) / math.sqrt(
            self.feature_dim
        )
        similarity = np.exp(-0.5 * (distance / float(radius)) ** 2)
        nearest = int(np.argmax(similarity))
        density = float(np.mean(np.sort(similarity)[-min(3, len(similarity)):]))
        novelty = 1.0 - float(similarity[nearest])
        age = max(0, int(shot_id) - int(self._history[nearest]["shot"]))
        coverage_debt = float(np.clip(age / 40.0, 0.0, 1.0))
        return {
            "novelty": _unit(novelty), "density": _unit(density),
            "coverage_debt": coverage_debt,
        }

    def update(self, *, shot_id: int, features: np.ndarray, target: int) -> None:
        if int(target) not in (0, 1):
            raise ValueError("online logistic target must be binary")
        vector = self._vector(features)
        if self._history and int(shot_id) <= int(self._history[-1]["shot"]):
            raise RuntimeError("queried classifier shots must be strictly increasing")
        self._history.append({
            "shot": int(shot_id), "features": vector.copy(),
            "target": int(target),
        })
        if len(self._history) > self.capacity:
            self._history = self._history[-self.capacity:]
        self._fit()

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "minimum_history": self.minimum_history,
            "minimum_per_class": self.minimum_per_class,
            "l2": self.l2, "capacity": self.capacity,
            "history": [
                {
                    "shot": int(item["shot"]),
                    "features": item["features"].tolist(),
                    "target": int(item["target"]),
                }
                for item in self._history
            ],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_RegularizedOnlineLogistic":
        model = cls(
            feature_dim=int(state["feature_dim"]),
            minimum_history=int(state["minimum_history"]),
            minimum_per_class=int(state["minimum_per_class"]),
            l2=float(state["l2"]), capacity=int(state["capacity"]),
        )
        model._history = [
            {
                "shot": int(item["shot"]),
                "features": np.asarray(item["features"], np.float64),
                "target": int(item["target"]),
            }
            for item in state["history"]
        ]
        model._fit()
        return model


class CausalEpisodeFPHead:
    """Causally reject learned FP-like episodes; never create a new alarm.

    A finite persistence counter is not enough for EAST: many false alarms stay
    above the base boundary for hundreds of samples.  Once queried history has
    fitted both classes, this head therefore keeps an FP-like episode blocked
    until either the causal classifier clears a later endpoint or an explicitly
    registered independent-evidence escape fires.  Before the classifier is
    reliable it is transparent and preserves the migrated baseline.
    """

    VALID_GROUPS = frozenset({"FP", "TP", "TARDY", "TN", "FN"})

    def __init__(
        self, *, feature_dim: int = 20, minimum_history: int = 12,
        minimum_per_class: int = 3, fp_probability_threshold: float = 0.65,
        minimum_confidence: float = 0.75, max_extra_points: int = 8,
    ) -> None:
        if not 0.5 <= float(fp_probability_threshold) < 1.0:
            raise ValueError("FP confirmation threshold must lie in [0.5, 1)")
        if max_extra_points <= 0:
            raise ValueError("maximum confirmation delay must be positive")
        if not 0.0 <= float(minimum_confidence) <= 1.0:
            raise ValueError("episode minimum confidence must lie in [0, 1]")
        self.feature_dim = int(feature_dim)
        self.fp_probability_threshold = float(fp_probability_threshold)
        self.minimum_confidence = float(minimum_confidence)
        self.max_extra_points = int(max_extra_points)
        self.model = _RegularizedOnlineLogistic(
            feature_dim=feature_dim, minimum_history=minimum_history,
            minimum_per_class=minimum_per_class,
        )
        self._pending: dict[int, dict[str, Any]] = {}
        self.version = 0

    def predict_and_seal(
        self, shot_id: int, *, candidate_mask: np.ndarray,
        feature_matrix: np.ndarray, route: np.ndarray,
        escape_mask: np.ndarray,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id in self._pending:
            raise RuntimeError("episode decision already sealed for shot")
        candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        features = np.asarray(feature_matrix, dtype=np.float64)
        routes = np.asarray(route, dtype=object).reshape(-1)
        escape = np.asarray(escape_mask, dtype=bool).reshape(-1)
        if (
            features.shape != (len(candidate), self.feature_dim)
            or routes.shape != candidate.shape or escape.shape != candidate.shape
            or not np.isfinite(features).all()
        ):
            raise ValueError("episode candidate inputs must align")
        positions = np.flatnonzero(candidate)
        if not len(positions):
            receipt = {
                "alarmed": False, "alarm_index": None, "route": None,
                "base_alarm_index": None, "fp_probability": 0.0,
                "fp_confidence": 0.0, "required_points": 1,
                "observed_points": 0, "filter_reason": "no_supported_candidate",
            }
            self._pending[shot_id] = {"features": None, "receipt": copy.deepcopy(receipt)}
            return receipt

        base_index = int(positions[0])
        base_features = features[base_index].copy()
        fp_probability, confidence = self.model.predict(base_features)
        selected_index: int | None = None
        selected_route: str | None = None
        selected_probability: float | None = None
        selected_confidence: float | None = None
        selection_reason: str | None = None
        observed = 0
        blocked = 0
        for index in range(len(candidate)):
            if not bool(candidate[index]):
                continue
            observed += 1
            local_probability, local_confidence = self.model.predict(features[index])
            if bool(escape[index]):
                selected_index = int(index)
                selected_route = None if routes[index] is None else str(routes[index])
                selected_probability = float(local_probability)
                selected_confidence = float(local_confidence)
                selection_reason = "independent_evidence_escape"
                break
            if local_confidence < self.minimum_confidence:
                selected_index = int(index)
                selected_route = None if routes[index] is None else str(routes[index])
                selected_probability = float(local_probability)
                selected_confidence = float(local_confidence)
                selection_reason = "episode_model_not_ready"
                break
            if local_probability < self.fp_probability_threshold:
                selected_index = int(index)
                selected_route = None if routes[index] is None else str(routes[index])
                selected_probability = float(local_probability)
                selected_confidence = float(local_confidence)
                selection_reason = "episode_model_cleared"
                break
            blocked += 1

        required = (
            self.max_extra_points + 1
            if confidence >= self.minimum_confidence
            and fp_probability >= self.fp_probability_threshold
            else 1
        )

        receipt = {
            "alarmed": selected_index is not None,
            "alarm_index": selected_index, "route": selected_route,
            "base_alarm_index": base_index,
            "fp_probability": float(fp_probability),
            "fp_confidence": float(confidence),
            "selected_fp_probability": selected_probability,
            "selected_fp_confidence": selected_confidence,
            "required_points": int(required),
            "observed_points": int(observed),
            "blocked_points": int(blocked),
            "filter_reason": (
                selection_reason if selected_index is not None
                else "learned_fp_episode_blocked"
            ),
        }
        self._pending[shot_id] = {
            "features": base_features, "receipt": copy.deepcopy(receipt),
        }
        return receipt

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, is_disrupt: bool,
        discrepancy_group: str,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("episode decision must be sealed before reveal")
        if not selected:
            raise PermissionError("episode supervision requires a selected query")
        if discrepancy_group not in self.VALID_GROUPS:
            raise ValueError("invalid episode-head discrepancy group")
        pending = self._pending.pop(shot_id)
        features = pending["features"]
        updated = features is not None
        if updated:
            self.model.update(
                shot_id=shot_id, features=features,
                target=int(not bool(is_disrupt)),
            )
            self.version += 1
        return {
            "event": "causal_episode_fp_update", "shot": shot_id,
            "updated": bool(updated), "group": discrepancy_group,
            "target_fp": int(not bool(is_disrupt)),
            "version_after": self.version,
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)

    def state_dict(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError("cannot checkpoint episode head with pending shots")
        return {
            "feature_dim": self.feature_dim,
            "fp_probability_threshold": self.fp_probability_threshold,
            "minimum_confidence": self.minimum_confidence,
            "max_extra_points": self.max_extra_points,
            "model": self.model.state_dict(), "version": self.version,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CausalEpisodeFPHead":
        model_state = state["model"]
        head = cls(
            feature_dim=int(state["feature_dim"]),
            minimum_history=int(model_state["minimum_history"]),
            minimum_per_class=int(model_state["minimum_per_class"]),
            fp_probability_threshold=float(state["fp_probability_threshold"]),
            minimum_confidence=float(state.get("minimum_confidence", 0.75)),
            max_extra_points=int(state["max_extra_points"]),
        )
        head.model = _RegularizedOnlineLogistic.from_state_dict(model_state)
        head.version = int(state.get("version", 0))
        return head


def causal_delta_sequence_matrix(
    feature_matrix: np.ndarray, *, short_width: int = 8,
    long_width: int = 32,
) -> np.ndarray:
    """Return current state, short-vs-long drift and long-window volatility.

    Row ``t`` is a function only of input rows ``[:t + 1]``.  This makes the
    representation suitable for a live endpoint gate and also gives the
    future-invariance property an explicit unit-testable boundary.
    """
    values = np.asarray(feature_matrix, dtype=np.float64)
    if (
        values.ndim != 2 or not len(values) or values.shape[1] <= 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("causal delta sequence input must be a finite matrix")
    if short_width <= 0 or long_width < short_width:
        raise ValueError("causal sequence widths must satisfy 0 < short <= long")

    def rolling_mean(matrix: np.ndarray, width: int) -> np.ndarray:
        cumulative = np.vstack((
            np.zeros((1, matrix.shape[1]), dtype=np.float64),
            np.cumsum(matrix, axis=0),
        ))
        index = np.arange(len(matrix), dtype=np.int64)
        lower = np.maximum(0, index - int(width) + 1)
        count = (index - lower + 1).astype(np.float64)[:, None]
        return (cumulative[index + 1] - cumulative[lower]) / count

    mean_short = rolling_mean(values, int(short_width))
    mean_long = rolling_mean(values, int(long_width))
    square_long = rolling_mean(values**2, int(long_width))
    std_long = np.sqrt(np.maximum(square_long - mean_long**2, 0.0))
    result = np.concatenate(
        (values, mean_short - mean_long, std_long), axis=1,
    )
    if result.shape != (len(values), 3 * values.shape[1]):
        raise RuntimeError("causal delta sequence representation changed shape")
    return result


class CausalProtectedSequenceFPHead:
    """Veto FP-like causal sequences behind a queried-history safety boundary.

    Unlike a globally fixed probability threshold, the boundary is rebuilt
    from the current classifier's scores on earlier selected safe and
    disruptive episodes.  A veto is enabled only when the upper disruptive
    score plus a safety margin remains below the lower safe-score support.
    Overlapping classes therefore fail open and preserve the migrated alarm.
    """

    VALID_GROUPS = CausalEpisodeFPHead.VALID_GROUPS

    def __init__(
        self, *, base_feature_dim: int = 20, minimum_history: int = 12,
        minimum_per_class: int = 3, l2: float = 0.30,
        safety_margin: float = 0.06, negative_quantile: float = 0.90,
        safe_quantile: float = 0.25, minimum_confidence: float = 0.75,
        max_extra_points: int = 8, capacity: int = 160,
    ) -> None:
        if base_feature_dim <= 0:
            raise ValueError("protected sequence base feature dimension is invalid")
        if l2 <= 0 or safety_margin < 0:
            raise ValueError("protected sequence regularization is invalid")
        if not 0.5 <= negative_quantile <= 1.0:
            raise ValueError("negative protection quantile must lie in [0.5, 1]")
        if not 0.0 <= safe_quantile <= 0.5:
            raise ValueError("safe support quantile must lie in [0, 0.5]")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("protected sequence confidence must lie in [0, 1]")
        if max_extra_points <= 0:
            raise ValueError("protected sequence audit delay must be positive")
        self.base_feature_dim = int(base_feature_dim)
        self.feature_dim = 3 * self.base_feature_dim
        self.l2 = float(l2)
        self.safety_margin = float(safety_margin)
        self.negative_quantile = float(negative_quantile)
        self.safe_quantile = float(safe_quantile)
        self.minimum_confidence = float(minimum_confidence)
        self.max_extra_points = int(max_extra_points)
        self.model = _RegularizedOnlineLogistic(
            feature_dim=self.feature_dim,
            minimum_history=int(minimum_history),
            minimum_per_class=int(minimum_per_class),
            l2=self.l2, capacity=int(capacity),
        )
        self._pending: dict[int, dict[str, Any]] = {}
        self.version = 0

    def _history_probabilities(self) -> tuple[np.ndarray, np.ndarray]:
        safe: list[float] = []
        disruptive: list[float] = []
        for item in self.model._history:
            probability, _confidence = self.model.predict(item["features"])
            (safe if int(item["target"]) == 1 else disruptive).append(
                float(probability)
            )
        return np.asarray(safe, np.float64), np.asarray(disruptive, np.float64)

    def _protection_boundary(self) -> dict[str, Any]:
        safe, disruptive = self._history_probabilities()
        confidence = self.model.confidence()
        minimum = self.model.minimum_per_class
        if (
            confidence < self.minimum_confidence
            or len(safe) < minimum or len(disruptive) < minimum
        ):
            return {
                "threshold": 1.01, "confidence": confidence,
                "safe_q": None, "disruptive_q": None,
                "safe_count": int(len(safe)),
                "disruptive_count": int(len(disruptive)),
                "separated": False, "ready": False,
            }
        safe_q = float(np.quantile(safe, self.safe_quantile))
        disruptive_q = float(np.quantile(
            disruptive, self.negative_quantile,
        ))
        candidate = max(0.50, disruptive_q + self.safety_margin)
        separated = bool(candidate < safe_q)
        return {
            "threshold": float(candidate if separated else 1.01),
            "confidence": confidence, "safe_q": safe_q,
            "disruptive_q": disruptive_q,
            "safe_count": int(len(safe)),
            "disruptive_count": int(len(disruptive)),
            "separated": separated, "ready": True,
        }

    def predict_and_seal(
        self, shot_id: int, *, candidate_mask: np.ndarray,
        feature_matrix: np.ndarray, route: np.ndarray,
        escape_mask: np.ndarray,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id in self._pending:
            raise RuntimeError("protected sequence decision already sealed for shot")
        candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        base = np.asarray(feature_matrix, dtype=np.float64)
        routes = np.asarray(route, dtype=object).reshape(-1)
        escape = np.asarray(escape_mask, dtype=bool).reshape(-1)
        if (
            base.shape != (len(candidate), self.base_feature_dim)
            or routes.shape != candidate.shape or escape.shape != candidate.shape
            or not np.isfinite(base).all()
        ):
            raise ValueError("protected sequence candidate inputs must align")
        positions = np.flatnonzero(candidate)
        boundary = self._protection_boundary()
        if not len(positions):
            receipt = {
                "alarmed": False, "alarm_index": None, "route": None,
                "base_alarm_index": None, "fp_probability": 0.0,
                "fp_confidence": float(boundary["confidence"]),
                "protection_threshold": float(boundary["threshold"]),
                "safe_support_count": int(boundary["safe_count"]),
                "disruptive_support_count": int(boundary["disruptive_count"]),
                "safe_support_quantile": boundary["safe_q"],
                "disruptive_protection_quantile": boundary["disruptive_q"],
                "required_points": 1, "observed_points": 0,
                "blocked_points": 0,
                "filter_reason": "no_supported_candidate",
            }
            self._pending[shot_id] = {
                "features": None, "receipt": copy.deepcopy(receipt),
            }
            return receipt

        sequence = causal_delta_sequence_matrix(base)
        base_index = int(positions[0])
        base_features = sequence[base_index].copy()
        base_probability, confidence = self.model.predict(base_features)
        selected_index: int | None = None
        selected_probability: float | None = None
        selected_confidence: float | None = None
        selection_reason: str | None = None
        observed = 0
        blocked = 0
        threshold = float(boundary["threshold"])
        for index in positions:
            observed += 1
            probability, local_confidence = self.model.predict(sequence[int(index)])
            if bool(escape[int(index)]):
                selected_index = int(index)
                selected_probability = float(probability)
                selected_confidence = float(local_confidence)
                selection_reason = "independent_evidence_escape"
                break
            if local_confidence < self.minimum_confidence:
                selected_index = int(index)
                selected_probability = float(probability)
                selected_confidence = float(local_confidence)
                selection_reason = "episode_model_not_ready"
                break
            if threshold > 1.0:
                selected_index = int(index)
                selected_probability = float(probability)
                selected_confidence = float(local_confidence)
                selection_reason = "protection_boundary_not_separated"
                break
            if probability < threshold:
                selected_index = int(index)
                selected_probability = float(probability)
                selected_confidence = float(local_confidence)
                selection_reason = "protected_sequence_cleared"
                break
            blocked += 1
        receipt = {
            "alarmed": selected_index is not None,
            "alarm_index": selected_index,
            "route": (
                None if selected_index is None or routes[selected_index] is None
                else str(routes[selected_index])
            ),
            "base_alarm_index": base_index,
            "fp_probability": float(base_probability),
            "fp_confidence": float(confidence),
            "selected_fp_probability": selected_probability,
            "selected_fp_confidence": selected_confidence,
            "protection_threshold": threshold,
            "safe_support_count": int(boundary["safe_count"]),
            "disruptive_support_count": int(boundary["disruptive_count"]),
            "safe_support_quantile": boundary["safe_q"],
            "disruptive_protection_quantile": boundary["disruptive_q"],
            "required_points": (
                self.max_extra_points + 1
                if threshold <= 1.0 and base_probability >= threshold else 1
            ),
            "observed_points": int(observed),
            "blocked_points": int(blocked),
            "filter_reason": (
                selection_reason if selected_index is not None
                else "protected_sequence_episode_blocked"
            ),
        }
        self._pending[shot_id] = {
            "features": base_features, "receipt": copy.deepcopy(receipt),
        }
        return receipt

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, is_disrupt: bool,
        discrepancy_group: str,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("protected sequence decision must be sealed before reveal")
        if not selected:
            raise PermissionError("protected sequence supervision requires selected query")
        if discrepancy_group not in self.VALID_GROUPS:
            raise ValueError("invalid protected sequence discrepancy group")
        pending = self._pending.pop(shot_id)
        features = pending["features"]
        updated = features is not None
        if updated:
            self.model.update(
                shot_id=shot_id, features=features,
                target=int(not bool(is_disrupt)),
            )
            self.version += 1
        return {
            "event": "causal_protected_sequence_fp_update",
            "shot": shot_id, "updated": bool(updated),
            "group": discrepancy_group,
            "target_fp": int(not bool(is_disrupt)),
            "version_after": self.version,
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)

    def state_dict(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError("cannot checkpoint protected sequence with pending shots")
        return {
            "base_feature_dim": self.base_feature_dim, "l2": self.l2,
            "safety_margin": self.safety_margin,
            "negative_quantile": self.negative_quantile,
            "safe_quantile": self.safe_quantile,
            "minimum_confidence": self.minimum_confidence,
            "max_extra_points": self.max_extra_points,
            "model": self.model.state_dict(), "version": self.version,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any],
    ) -> "CausalProtectedSequenceFPHead":
        model_state = state["model"]
        head = cls(
            base_feature_dim=int(state["base_feature_dim"]),
            minimum_history=int(model_state["minimum_history"]),
            minimum_per_class=int(model_state["minimum_per_class"]),
            l2=float(state["l2"]),
            safety_margin=float(state["safety_margin"]),
            negative_quantile=float(state["negative_quantile"]),
            safe_quantile=float(state["safe_quantile"]),
            minimum_confidence=float(state["minimum_confidence"]),
            max_extra_points=int(state["max_extra_points"]),
            capacity=int(model_state["capacity"]),
        )
        head.model = _RegularizedOnlineLogistic.from_state_dict(model_state)
        head.version = int(state.get("version", 0))
        return head


class CausalLocalRetrievalFPHead:
    """Veto locally repeated false-alarm episodes from queried history only.

    The existing logistic episode head asks for one global linear separation
    between safe and disruptive episodes.  EAST false alarms are multi-modal,
    so this head instead requires local support from earlier queried safe
    alarms and explicit separation from earlier queried disruptive alarms.
    It never creates an alarm, and the current outcome is accepted only after
    the decision has been sealed.
    """

    VALID_GROUPS = CausalEpisodeFPHead.VALID_GROUPS

    def __init__(
        self, *, feature_dim: int, minimum_per_class: int = 3,
        neighbor_count: int = 3, distance_ratio: float = 0.8,
        fp_probability_threshold: float = 0.65, capacity: int = 160,
    ) -> None:
        if feature_dim <= 0 or minimum_per_class <= 0:
            raise ValueError("invalid local retrieval geometry")
        if neighbor_count <= 0 or capacity < 2 * minimum_per_class:
            raise ValueError("invalid local retrieval history budget")
        if not 0.0 < float(distance_ratio) <= 1.5:
            raise ValueError("local retrieval distance ratio is invalid")
        if not 0.5 <= float(fp_probability_threshold) < 1.0:
            raise ValueError("local retrieval FP threshold must lie in [0.5, 1)")
        self.feature_dim = int(feature_dim)
        self.minimum_per_class = int(minimum_per_class)
        self.neighbor_count = int(neighbor_count)
        self.distance_ratio = float(distance_ratio)
        self.fp_probability_threshold = float(fp_probability_threshold)
        self.capacity = int(capacity)
        self._history: list[dict[str, Any]] = []
        self._pending: dict[int, dict[str, Any]] = {}
        self.version = 0

    def _vector(self, features: np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if len(vector) != self.feature_dim or not np.isfinite(vector).all():
            raise ValueError("local retrieval feature vector is invalid")
        return np.clip(vector, -10.0, 10.0)

    def _evidence(self, features: np.ndarray) -> dict[str, Any]:
        safe = sum(int(item["target_fp"] == 1) for item in self._history)
        disruptive = len(self._history) - safe
        if min(safe, disruptive) < self.minimum_per_class:
            return {
                "ready": False, "block": False, "fp_probability": 0.0,
                "safe_distance": None, "disruptive_distance": None,
                "safe_support_count": safe,
                "disruptive_support_count": disruptive,
            }
        matrix = np.stack([item["features"] for item in self._history])
        target = np.asarray(
            [item["target_fp"] for item in self._history], dtype=np.int64,
        )
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        scale = np.maximum(q75 - q25, 0.20)
        distance = np.linalg.norm(
            (matrix - self._vector(features)[None, :]) / scale, axis=1,
        ) / math.sqrt(self.feature_dim)
        count = self.neighbor_count
        safe_distance = np.sort(distance[target == 1])[:count]
        disruptive_distance = np.sort(distance[target == 0])[:count]
        safe_similarity = float(np.exp(-0.5 * safe_distance**2).sum())
        disruptive_similarity = float(
            np.exp(-0.5 * disruptive_distance**2).sum()
        )
        fp_probability = safe_similarity / (
            safe_similarity + disruptive_similarity + 1e-12
        )
        mean_safe = float(safe_distance.mean())
        mean_disruptive = float(disruptive_distance.mean())
        block = (
            fp_probability >= self.fp_probability_threshold
            and mean_safe <= self.distance_ratio * max(mean_disruptive, 1e-12)
        )
        return {
            "ready": True, "block": bool(block),
            "fp_probability": float(fp_probability),
            "safe_distance": mean_safe,
            "disruptive_distance": mean_disruptive,
            "safe_support_count": safe,
            "disruptive_support_count": disruptive,
        }

    def predict_and_seal(
        self, shot_id: int, *, candidate_mask: np.ndarray,
        feature_matrix: np.ndarray, route: np.ndarray,
        escape_mask: np.ndarray,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id in self._pending:
            raise RuntimeError("local retrieval decision already sealed for shot")
        candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        features = np.asarray(feature_matrix, dtype=np.float64)
        routes = np.asarray(route, dtype=object).reshape(-1)
        escape = np.asarray(escape_mask, dtype=bool).reshape(-1)
        if (
            features.shape != (len(candidate), self.feature_dim)
            or routes.shape != candidate.shape or escape.shape != candidate.shape
            or not np.isfinite(features).all()
        ):
            raise ValueError("local retrieval candidate inputs must align")
        positions = np.flatnonzero(candidate)
        if not len(positions):
            receipt = {
                "alarmed": False, "alarm_index": None, "route": None,
                "base_alarm_index": None, "fp_probability": 0.0,
                "filter_reason": "no_supported_candidate",
                "safe_support_count": sum(
                    int(item["target_fp"] == 1) for item in self._history
                ),
                "disruptive_support_count": sum(
                    int(item["target_fp"] == 0) for item in self._history
                ),
            }
            self._pending[shot_id] = {
                "features": None, "receipt": copy.deepcopy(receipt),
            }
            return receipt

        base_index = int(positions[0])
        base_features = self._vector(features[base_index])
        selected_index: int | None = None
        selected_evidence: dict[str, Any] | None = None
        blocked = 0
        reason = "local_queried_fp_episode_blocked"
        for position in positions:
            index = int(position)
            evidence = self._evidence(features[index])
            selected_evidence = evidence
            if bool(escape[index]):
                selected_index = index
                reason = "independent_evidence_escape"
                break
            if not bool(evidence["ready"]):
                selected_index = index
                reason = "local_retrieval_not_ready"
                break
            if not bool(evidence["block"]):
                selected_index = index
                reason = "local_retrieval_cleared"
                break
            blocked += 1
        assert selected_evidence is not None
        receipt = {
            "alarmed": selected_index is not None,
            "alarm_index": selected_index,
            "route": (
                None if selected_index is None or routes[selected_index] is None
                else str(routes[selected_index])
            ),
            "base_alarm_index": base_index,
            "fp_probability": float(selected_evidence["fp_probability"]),
            "safe_distance": selected_evidence["safe_distance"],
            "disruptive_distance": selected_evidence["disruptive_distance"],
            "safe_support_count": int(selected_evidence["safe_support_count"]),
            "disruptive_support_count": int(
                selected_evidence["disruptive_support_count"]
            ),
            "blocked_points": int(blocked), "filter_reason": reason,
        }
        self._pending[shot_id] = {
            "features": base_features.copy(),
            "receipt": copy.deepcopy(receipt),
        }
        return receipt

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, is_disrupt: bool,
        discrepancy_group: str,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("local retrieval decision must precede reveal")
        if not selected:
            raise PermissionError("local retrieval supervision requires a query")
        if discrepancy_group not in self.VALID_GROUPS:
            raise ValueError("invalid local retrieval discrepancy group")
        pending = self._pending.pop(shot_id)
        features = pending["features"]
        updated = features is not None
        if updated:
            if self._history and shot_id <= int(self._history[-1]["shot"]):
                raise RuntimeError("local retrieval reveals must be strictly ordered")
            self._history.append({
                "shot": shot_id, "features": features.copy(),
                "target_fp": int(not bool(is_disrupt)),
            })
            self._history = self._history[-self.capacity:]
            self.version += 1
        return {
            "event": "causal_local_retrieval_fp_update", "shot": shot_id,
            "updated": bool(updated), "group": discrepancy_group,
            "target_fp": int(not bool(is_disrupt)),
            "version_after": self.version,
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)


@dataclass(frozen=True)
class QueryExpertWeights:
    error_correction: float = 0.35
    information_gain: float = 0.20
    physics_conflict: float = 0.15
    gradient_actionability: float = 0.20
    coverage_exploration: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "error_correction": self.error_correction,
            "information_gain": self.information_gain,
            "physics_conflict": self.physics_conflict,
            "gradient_actionability": self.gradient_actionability,
            "coverage_exploration": self.coverage_exploration,
        }


class CausalDecisionQueryValueModel:
    """Multi-expert value of buying a label, not merely error likelihood."""

    REQUIRED = frozenset({
        "uncertainty", "physics_contradiction", "self_supervised_surprise",
        "teacher_disagreement", "private_innovation", "expected_gradient",
        "decision_changeability", "fp_probability", "fp_confidence",
        "physics_reliability", "stability_risk",
    })
    ERROR_GROUPS = frozenset({"FP", "FN", "TARDY"})
    VALID_GROUPS = ERROR_GROUPS | frozenset({"TP", "TN"})

    def __init__(
        self, *, feature_dim: int = 20, learning_rate: float = 0.15,
        initial_weights: QueryExpertWeights | None = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("query expert learning rate must be positive")
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        weights = (initial_weights or QueryExpertWeights()).as_dict()
        total = sum(weights.values())
        self._weights = {name: value / total for name, value in weights.items()}
        self.error_model = _RegularizedOnlineLogistic(
            feature_dim=feature_dim, minimum_history=12,
            minimum_per_class=3,
        )
        self._pending: dict[int, dict[str, Any]] = {}
        self.version = 0

    def weights_dict(self) -> dict[str, float]:
        return dict(self._weights)

    def score_and_seal(
        self, shot_id: int, features: Mapping[str, float], *, phenotype: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        shot_id = int(shot_id)
        if shot_id in self._pending:
            raise RuntimeError("decision query value already sealed for shot")
        if not self.REQUIRED.issubset(features):
            missing = sorted(self.REQUIRED - set(features))
            raise ValueError(f"decision query value missing {missing}")
        bounded = {name: _unit(features[name]) for name in self.REQUIRED}
        vector = np.asarray(phenotype, np.float64).reshape(-1)
        if len(vector) != self.feature_dim or not np.isfinite(vector).all():
            raise ValueError("decision query phenotype is invalid")
        model_probability, model_confidence = self.error_model.predict(vector)
        base_probability = (
            0.55 * bounded["uncertainty"]
            + 0.45 * bounded["physics_contradiction"]
        )
        error_probability = (
            (1.0 - model_confidence) * base_probability
            + model_confidence * model_probability
        )
        fp_confidence = bounded["fp_confidence"]
        error_probability = (
            (1.0 - fp_confidence) * error_probability
            + fp_confidence * max(error_probability, bounded["fp_probability"])
        )
        error_probability = _unit(error_probability)
        geometry = self.error_model.geometry(vector, shot_id=shot_id)
        novelty = geometry["novelty"]
        density = geometry["density"]
        coverage = geometry["coverage_debt"]
        changeability = bounded["decision_changeability"]
        gradient = bounded["expected_gradient"]
        entropy = _unit(4.0 * error_probability * (1.0 - error_probability))
        experts = {
            "error_correction": _unit(
                error_probability * changeability * (0.40 + 0.60 * gradient)
                * (0.50 + 0.50 * density)
            ),
            "information_gain": _unit(
                entropy * novelty * (0.50 + 0.50 * coverage)
            ),
            "physics_conflict": _unit(
                bounded["physics_contradiction"]
                * bounded["physics_reliability"]
            ),
            "gradient_actionability": _unit(
                gradient * (0.50 + 0.50 * bounded["private_innovation"])
                * (1.0 - 0.50 * bounded["stability_risk"])
            ),
            "coverage_exploration": _unit(
                coverage * novelty * (0.50 + 0.50 * bounded["uncertainty"])
            ),
        }
        protection_penalty = (
            0.20 * bounded["stability_risk"]
            * (1.0 - error_probability) * (1.0 - changeability)
        )
        score = _unit(
            0.05 + sum(self._weights[name] * experts[name] for name in experts)
            - protection_penalty
        )
        receipt = {
            "score": score, "error_probability": error_probability,
            "error_model_probability": model_probability,
            "error_model_confidence": model_confidence,
            "novelty": novelty, "density": density,
            "coverage_debt": coverage, "experts": dict(experts),
            "expert_weights": self.weights_dict(),
            "protection_penalty": float(protection_penalty),
        }
        self._pending[shot_id] = {
            "phenotype": vector.copy(), "experts": dict(experts),
            "receipt": copy.deepcopy(receipt),
        }
        return score, receipt

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, discrepancy_group: str,
        realized_utility: float,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("decision query value must be sealed before reveal")
        if not selected:
            raise PermissionError("decision query-value update requires a selected query")
        if discrepancy_group not in self.VALID_GROUPS:
            raise ValueError("invalid decision query discrepancy group")
        pending = self._pending.pop(shot_id)
        target_error = int(discrepancy_group in self.ERROR_GROUPS)
        self.error_model.update(
            shot_id=shot_id, features=pending["phenotype"], target=target_error,
        )
        target_utility = _unit(realized_utility)
        before = self.weights_dict()
        updated = {}
        for name, prediction in pending["experts"].items():
            loss = (float(prediction) - target_utility) ** 2
            updated[name] = max(
                0.03,
                self._weights[name] * math.exp(-self.learning_rate * loss),
            )
        total = sum(updated.values())
        self._weights = {name: value / total for name, value in updated.items()}
        self.version += 1
        return {
            "event": "causal_decision_query_value_update", "shot": shot_id,
            "updated": True, "discrepancy_group": discrepancy_group,
            "target_error": target_error,
            "realized_utility": target_utility,
            "weights_before": before, "weights_after": self.weights_dict(),
            "version_after": self.version,
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)

    def state_dict(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError("cannot checkpoint decision value with pending shots")
        return {
            "feature_dim": self.feature_dim,
            "learning_rate": self.learning_rate,
            "weights": self.weights_dict(),
            "error_model": self.error_model.state_dict(),
            "version": self.version,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any],
    ) -> "CausalDecisionQueryValueModel":
        value = cls(
            feature_dim=int(state["feature_dim"]),
            learning_rate=float(state["learning_rate"]),
        )
        weights = {
            str(name): float(weight)
            for name, weight in dict(state["weights"]).items()
        }
        if set(weights) != set(value._weights) or any(
            weight <= 0.0 or not np.isfinite(weight)
            for weight in weights.values()
        ):
            raise ValueError("invalid saved query expert weights")
        total = sum(weights.values())
        value._weights = {name: weight / total for name, weight in weights.items()}
        value.error_model = _RegularizedOnlineLogistic.from_state_dict(
            state["error_model"]
        )
        value.version = int(state.get("version", 0))
        return value


class StratifiedRollingTokenBudgetSelector:
    """Rolling label budget with an explicit score-independent exploration slot."""

    def __init__(
        self, *, query_rate: float = 0.30, capacity: float = 12.0,
        block_size: int = 20, quota: int = 6, history_size: int = 160,
        quantile: float = 0.70, initial_cutoff: float = 0.35,
        exploration_per_block: int = 1, exploration_offset: int = 3,
        exploration_stride: int = 7, query_cost: float = 0.0,
    ) -> None:
        if not 0 < query_rate <= 1 or capacity < 1:
            raise ValueError("invalid stratified rolling budget")
        if block_size <= 0 or quota <= 0 or quota > block_size:
            raise ValueError("invalid block query cap")
        if history_size <= 0 or not 0 < quantile < 1:
            raise ValueError("invalid score history")
        if exploration_per_block not in (0, 1):
            raise ValueError("v3 supports zero or one reserved exploration per block")
        if not 0.0 <= float(query_cost) <= 1.0:
            raise ValueError("query cost must lie in [0, 1]")
        self.query_rate = float(query_rate)
        self.capacity = float(capacity)
        self.block_size = int(block_size)
        self.quota = int(quota)
        self.history_size = int(history_size)
        self.quantile = float(quantile)
        self.initial_cutoff = _unit(initial_cutoff)
        self.exploration_per_block = int(exploration_per_block)
        self.exploration_offset = int(exploration_offset)
        self.exploration_stride = int(exploration_stride)
        self.query_cost = float(query_cost)
        self._tokens = 0.0
        self._history: list[float] = []
        self._next_index = 0
        self._block = -1
        self._used = 0
        self._exploration_done = False
        self._exploration_due = False
        self._exploration_position = 0
        self.total_selected = 0

    def _begin_block(self, block: int) -> None:
        self._block = int(block)
        self._used = 0
        self._exploration_done = self.exploration_per_block == 0
        self._exploration_due = False
        self._exploration_position = (
            self.exploration_offset + int(block) * self.exploration_stride
        ) % self.block_size

    def decide(self, *, score: float, stream_index: int) -> dict[str, Any]:
        stream_index = int(stream_index)
        if stream_index != self._next_index:
            raise RuntimeError("stratified query decisions must be contiguous")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("query score must be finite")
        value = _unit(value)
        block = stream_index // self.block_size
        position = stream_index % self.block_size
        if block != self._block:
            self._begin_block(block)
        self._tokens = min(self.capacity, self._tokens + self.query_rate)
        prior = self._history[-self.history_size:]
        cutoff = (
            float(np.quantile(prior, self.quantile))
            if len(prior) >= min(20, self.history_size)
            else self.initial_cutoff
        )
        if not self._exploration_done and position >= self._exploration_position:
            self._exploration_due = True
        exploration = (
            self._exploration_due and self._tokens >= 1.0
            and self._used < self.quota
        )
        reserved = int(not self._exploration_done)
        exploitation = (
            not exploration and value >= max(cutoff, self.query_cost)
            and self._tokens >= 1.0
            and self._used < self.quota - reserved
        )
        selected = bool(exploration or exploitation)
        route = None
        if selected:
            route = "exploration" if exploration else "error_likelihood"
            self._tokens -= 1.0
            self._used += 1
            self.total_selected += 1
            if exploration:
                self._exploration_done = True
                self._exploration_due = False
        receipt = {
            "selected": selected, "route": route, "score": value,
            "cutoff": cutoff, "block": block, "position": position,
            "query_cost": self.query_cost,
            "net_value": value - self.query_cost,
            "history_count": min(len(prior), self.history_size),
            "exploration_position": self._exploration_position,
            "tokens_after": float(self._tokens),
        }
        self._history.append(value)
        self._next_index += 1
        return receipt

    def state_dict(self) -> dict[str, Any]:
        return {
            "query_rate": self.query_rate, "capacity": self.capacity,
            "block_size": self.block_size, "quota": self.quota,
            "history_size": self.history_size, "quantile": self.quantile,
            "initial_cutoff": self.initial_cutoff,
            "query_cost": self.query_cost,
            "exploration_per_block": self.exploration_per_block,
            "exploration_offset": self.exploration_offset,
            "exploration_stride": self.exploration_stride,
            "tokens": self._tokens, "history": list(self._history),
            "next_index": self._next_index, "block": self._block,
            "used": self._used,
            "exploration_done": self._exploration_done,
            "exploration_due": self._exploration_due,
            "exploration_position": self._exploration_position,
            "total_selected": self.total_selected,
        }


__all__ = [
    "CausalEpisodeFPHead", "CausalProtectedSequenceFPHead",
    "causal_delta_sequence_matrix", "CausalDecisionQueryValueModel",
    "StratifiedRollingTokenBudgetSelector", "QueryExpertWeights",
]
