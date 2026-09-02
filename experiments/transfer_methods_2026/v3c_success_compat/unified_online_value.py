#!/usr/bin/env python3
"""Causal value, query, and projected alarm control for EAST online learning.

The classes in this module deliberately contain no HDF5 access.  They accept
only already-observed, label-free values until a caller has sealed a shot and
explicitly releases a queried outcome.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from dual_threshold_alarm import dual_threshold_alarm


@dataclass(frozen=True)
class UpdateValueWeights:
    reconstruction: float = 0.40
    teacher: float = 0.20
    physics: float = 0.25
    private: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.reconstruction, self.teacher, self.physics, self.private,
        )
        if any(value < 0 for value in values) or not math.isclose(sum(values), 1.0):
            raise ValueError("update-value weights must be non-negative and sum to one")


class CausalUpdateValue:
    """Map causal self-supervised evidence to a bounded optimizer scale.

    Inputs are normalized support-relative quantities in ``[0, 1]``.  A low
    availability fraction fails closed.  Redundant states retain a small
    teacher-anchor update, but receive less gradient than novel states.
    """

    def __init__(
        self, *, weights: UpdateValueWeights | None = None,
        minimum_value: float = 0.20,
        saturation_value: float = 1.0,
    ) -> None:
        if not 0 <= float(minimum_value) < float(saturation_value) <= 1:
            raise ValueError("require 0 <= minimum_value < saturation_value <= 1")
        self.weights = weights or UpdateValueWeights()
        self.minimum_value = float(minimum_value)
        self.saturation_value = float(saturation_value)

    @staticmethod
    def _unit(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    def score(
        self, *, reconstruction_surprise: float,
        teacher_disagreement: float, physics_mismatch: float,
        private_innovation: float, availability_fraction: float,
        redundancy: float,
    ) -> float:
        availability = self._unit(availability_fraction)
        if availability <= 0:
            return 0.0
        evidence = (
            self.weights.reconstruction * self._unit(reconstruction_surprise)
            + self.weights.teacher * self._unit(teacher_disagreement)
            + self.weights.physics * self._unit(physics_mismatch)
            + self.weights.private * self._unit(private_innovation)
        )
        novelty_multiplier = 1.0 - 0.5 * self._unit(redundancy)
        return self._unit(availability * novelty_multiplier * evidence)

    def learning_rate_scale(self, value: float) -> float:
        bounded = self._unit(value)
        if bounded <= self.minimum_value:
            return 0.0
        return float(np.clip(
            (bounded - self.minimum_value)
            / (self.saturation_value - self.minimum_value),
            0.0, 1.0,
        ))


class PrequentialWindowClock:
    """Fail closed unless an endpoint is predicted before it can train."""

    def __init__(self) -> None:
        self._last_sealed = -1
        self._last_updated = -1

    @property
    def last_sealed(self) -> int:
        return self._last_sealed

    def seal_prediction(self, endpoint_index: int) -> None:
        endpoint_index = int(endpoint_index)
        if endpoint_index != self._last_sealed + 1:
            raise RuntimeError("endpoint predictions must seal the next consecutive index")
        self._last_sealed = endpoint_index

    def permit_update(self, observed_endpoint_index: int) -> None:
        observed_endpoint_index = int(observed_endpoint_index)
        if observed_endpoint_index > self._last_sealed:
            raise RuntimeError("current endpoint prediction must be sealed before update")
        if observed_endpoint_index <= self._last_updated:
            raise RuntimeError("an endpoint can authorize at most one online update")
        self._last_updated = observed_endpoint_index


class CausalWindowUpdateScheduler:
    """Spend a causal rolling gradient budget on valuable observed windows.

    The scheduler never receives a shot length.  Decisions for a prefix are
    therefore invariant when more future endpoints are appended to the shot.
    Only lightweight candidate values are observed at the registered stride;
    expensive reconstruction losses are evaluated after a candidate is chosen.
    """

    def __init__(
        self, *, candidate_stride: int = 8, update_rate: float = 0.25,
        token_capacity: float = 4.0, history_size: int = 64,
        quantile: float = 0.80, initial_cutoff: float = 0.40,
        minimum_history: int = 8, minimum_gap: int = 8,
        max_updates: int = 64,
    ) -> None:
        if candidate_stride <= 0 or history_size <= 0 or minimum_history <= 0:
            raise ValueError("invalid causal window scheduler history")
        if not 0 < float(update_rate) <= 1 or float(token_capacity) < 1:
            raise ValueError("invalid causal window token budget")
        if not 0 < float(quantile) < 1 or not 0 <= float(initial_cutoff) <= 1:
            raise ValueError("invalid causal window cutoff")
        if minimum_gap < 0 or max_updates <= 0:
            raise ValueError("invalid causal window update limit")
        self.candidate_stride = int(candidate_stride)
        self.update_rate = float(update_rate)
        self.token_capacity = float(token_capacity)
        self.history_size = int(history_size)
        self.quantile = float(quantile)
        self.initial_cutoff = float(initial_cutoff)
        self.minimum_history = int(minimum_history)
        self.minimum_gap = int(minimum_gap)
        self.max_updates = int(max_updates)
        self._next_step = 0
        self._tokens = 0.0
        self._history: list[float] = []
        self._last_selected = -10**12
        self.total_selected = 0

    def consider(self, *, causal_step: int, value: float) -> dict[str, Any]:
        causal_step = int(causal_step)
        if causal_step != self._next_step:
            raise RuntimeError("causal window decisions must be contiguous and ordered")
        bounded = float(np.clip(value, 0.0, 1.0))
        if not np.isfinite(value):
            raise ValueError("causal window value must be finite")
        candidate = causal_step % self.candidate_stride == 0
        prior = self._history[-self.history_size:]
        cutoff = (
            float(np.quantile(prior, self.quantile))
            if len(prior) >= self.minimum_history else self.initial_cutoff
        )
        selected = False
        route = None
        tokens_before = self._tokens
        if candidate:
            self._tokens = min(
                self.token_capacity, self._tokens + self.update_rate,
            )
            selected = (
                self._tokens >= 1.0
                and bounded >= cutoff
                and causal_step - self._last_selected >= self.minimum_gap
                and self.total_selected < self.max_updates
            )
            if selected:
                self._tokens -= 1.0
                self._last_selected = causal_step
                self.total_selected += 1
                route = "past_quantile_value"
            self._history.append(bounded)
        self._next_step += 1
        return {
            "candidate": bool(candidate), "selected": bool(selected),
            "route": route, "causal_step": causal_step, "value": bounded,
            "cutoff": cutoff, "tokens_before": float(tokens_before),
            "tokens_after": float(self._tokens),
            "history_count": min(len(prior), self.history_size),
            "selected_count": self.total_selected,
        }


class TypedEndpointResidualMemory:
    """Causal endpoint-level FP/FN/TARDY memory with TP/TN protection.

    Keys and predictions for a shot are sealed endpoint by endpoint.  Nothing
    enters the persistent memory until the complete shot has been selected for
    a label query and its outcome is revealed.  Similarity is aggregated by
    distinct source shot so one long shot cannot manufacture support.
    """

    ERROR_GROUPS = frozenset({"FP", "FN", "TARDY"})
    PROTECTION_GROUPS = frozenset({"TP", "TN"})
    VALID_GROUPS = ERROR_GROUPS | PROTECTION_GROUPS

    def __init__(
        self, *, radius: float = 0.35, minimum_error_shots: int = 2,
        max_positive_residual: float = 0.35,
        max_negative_residual: float = 0.25,
        positive_physics_gate: float = 0.50,
        protection_ratio: float = 1.0,
        capacity: int = 512,
    ) -> None:
        if radius <= 0 or minimum_error_shots <= 0 or capacity <= 0:
            raise ValueError("invalid typed endpoint-memory geometry")
        if max_positive_residual < 0 or max_negative_residual < 0:
            raise ValueError("endpoint residual bounds must be non-negative")
        if not 0 <= positive_physics_gate <= 1 or protection_ratio < 0:
            raise ValueError("invalid endpoint-memory guard")
        self.radius = float(radius)
        self.minimum_error_shots = int(minimum_error_shots)
        self.max_positive_residual = float(max_positive_residual)
        self.max_negative_residual = float(max_negative_residual)
        self.positive_physics_gate = float(positive_physics_gate)
        self.protection_ratio = float(protection_ratio)
        self.capacity = int(capacity)
        self._key_size: int | None = None
        self._entries: list[dict[str, Any]] = []
        self._pending: dict[int, dict[str, Any]] = {}
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _key(self, key: np.ndarray) -> np.ndarray:
        vector = np.asarray(key, dtype=np.float64).reshape(-1)
        if not len(vector) or not np.isfinite(vector).all():
            raise ValueError("endpoint-memory key must be finite and non-empty")
        if self._key_size is None:
            self._key_size = len(vector)
        if len(vector) != self._key_size:
            raise ValueError("endpoint-memory key size changed")
        return np.clip(vector, -1.0, 1.0)

    def _evidence(self, key: np.ndarray) -> dict[str, Any]:
        by_group: dict[str, dict[int, float]] = {
            group: {} for group in self.VALID_GROUPS
        }
        if self._entries and self._cache is None:
            self._cache = (
                np.stack([entry["key"] for entry in self._entries]),
                np.asarray([entry["group"] for entry in self._entries], dtype=object),
                np.asarray([entry["shot"] for entry in self._entries], dtype=np.int64),
            )
        if self._cache is not None:
            matrix, groups, shots = self._cache
            distances = np.linalg.norm(matrix - key[None, :], axis=1) / math.sqrt(len(key))
            local = np.flatnonzero(distances <= self.radius)
        else:
            distances = np.empty(0, dtype=np.float64)
            groups = np.empty(0, dtype=object)
            shots = np.empty(0, dtype=np.int64)
            local = np.empty(0, dtype=np.int64)
        for index in local:
            distance = float(distances[index])
            similarity = math.exp(-0.5 * (distance / self.radius) ** 2)
            group = str(groups[index])
            shot = int(shots[index])
            by_group[group][shot] = max(
                by_group[group].get(shot, 0.0), similarity,
            )
        support = {
            group: float(sum(values.values()))
            for group, values in by_group.items()
        }
        positive_shots = set(by_group["FN"]) | set(by_group["TARDY"])
        return {
            **{f"{group.lower()}_support": support[group]
               for group in sorted(self.VALID_GROUPS)},
            "positive_error_shots": len(positive_shots),
            "negative_error_shots": len(by_group["FP"]),
            "tp_protection": support["TP"],
            "tn_protection": support["TN"],
            "positive_support": support["FN"] + 0.7 * support["TARDY"],
            "negative_support": support["FP"],
        }

    def begin_shot(self, shot_id: int) -> None:
        shot_id = int(shot_id)
        if shot_id in self._pending:
            raise RuntimeError("endpoint-memory shot already active")
        self._pending[shot_id] = {
            "sealed": False, "last_endpoint": -1, "endpoints": {},
        }

    def correct_and_seal(
        self, shot_id: int, *, endpoint_index: int, key: np.ndarray,
        base_logit: float, physics_support: float,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        endpoint_index = int(endpoint_index)
        pending = self._pending.get(shot_id)
        if pending is None or pending["sealed"]:
            raise RuntimeError("endpoint-memory shot must be active")
        if endpoint_index <= int(pending["last_endpoint"]):
            raise RuntimeError("endpoint-memory predictions must be strictly ordered")
        vector = self._key(key)
        evidence = self._evidence(vector)
        residual = 0.0
        positive = float(evidence["positive_support"])
        negative = float(evidence["negative_support"])
        if (
            int(evidence["positive_error_shots"]) >= self.minimum_error_shots
            and float(physics_support) >= self.positive_physics_gate
            and float(evidence["tn_protection"])
            < self.protection_ratio * max(positive, 1e-12)
        ):
            confidence = positive / (
                positive + float(evidence["tn_protection"]) + 1e-12
            )
            residual += self.max_positive_residual * confidence
        if (
            int(evidence["negative_error_shots"]) >= self.minimum_error_shots
            and float(evidence["tp_protection"])
            < self.protection_ratio * max(negative, 1e-12)
        ):
            confidence = negative / (
                negative + float(evidence["tp_protection"]) + 1e-12
            )
            residual -= self.max_negative_residual * confidence
        residual = float(np.clip(
            residual, -self.max_negative_residual, self.max_positive_residual,
        ))
        pending["last_endpoint"] = endpoint_index
        pending["endpoints"][endpoint_index] = {
            "key": vector.copy(), "base_logit": float(base_logit),
            "physics_support": float(np.clip(physics_support, 0.0, 1.0)),
        }
        return {
            "corrected_logit": float(base_logit) + residual,
            "residual": residual, **evidence,
        }

    def seal_shot(self, shot_id: int) -> None:
        pending = self._pending.get(int(shot_id))
        if pending is None:
            raise RuntimeError("endpoint-memory shot is not pending")
        pending["sealed"] = True

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, discrepancy_group: str,
        endpoint_indices: list[int] | np.ndarray,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        pending = self._pending.get(shot_id)
        if pending is None:
            raise RuntimeError("endpoint-memory shot is not pending")
        if not bool(pending["sealed"]):
            raise RuntimeError("endpoint-memory shot must be sealed before reveal")
        if not selected:
            raise PermissionError("endpoint memory requires a selected query")
        if discrepancy_group not in self.VALID_GROUPS:
            raise ValueError("invalid endpoint-memory discrepancy group")
        indices = [int(index) for index in endpoint_indices]
        if not indices:
            raise ValueError("endpoint memory requires at least one endpoint")
        available_indices = sorted(int(index) for index in pending["endpoints"])
        if not available_indices:
            raise RuntimeError("endpoint memory has no causally sealed endpoints")
        resolved_indices = [
            index if index in pending["endpoints"] else min(
                available_indices, key=lambda candidate: abs(candidate - index),
            )
            for index in indices
        ]
        stored = 0
        stored_indices: list[int] = []
        for endpoint_index in dict.fromkeys(resolved_indices):
            endpoint = pending["endpoints"][endpoint_index]
            self._entries.append({
                "shot": shot_id, "endpoint": endpoint_index,
                "group": discrepancy_group,
                "key": endpoint["key"].copy(),
                "base_logit": endpoint["base_logit"],
                "physics_support": endpoint["physics_support"],
            })
            stored += 1
            stored_indices.append(endpoint_index)
        self._pending.pop(shot_id)
        if len(self._entries) > self.capacity:
            self._entries = self._entries[-self.capacity:]
        self._cache = None
        return {
            "event": "typed_endpoint_memory_update", "shot": shot_id,
            "group": discrepancy_group, "stored_endpoints": stored,
            "entry_count": len(self._entries),
            "requested_endpoints": indices,
            "resolved_endpoints": stored_indices,
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)

    def entries(self) -> list[dict[str, Any]]:
        return [
            {
                "shot": int(entry["shot"]),
                "endpoint": int(entry["endpoint"]),
                "group": str(entry["group"]),
                "key": entry["key"].tolist(),
                "base_logit": float(entry["base_logit"]),
                "physics_support": float(entry["physics_support"]),
            }
            for entry in self._entries
        ]

    def state_dict(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError("cannot checkpoint endpoint memory with pending shots")
        return {
            "radius": self.radius,
            "minimum_error_shots": self.minimum_error_shots,
            "max_positive_residual": self.max_positive_residual,
            "max_negative_residual": self.max_negative_residual,
            "positive_physics_gate": self.positive_physics_gate,
            "protection_ratio": self.protection_ratio,
            "capacity": self.capacity, "key_size": self._key_size,
            "entries": self.entries(),
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any],
    ) -> "TypedEndpointResidualMemory":
        memory = cls(
            radius=float(state["radius"]),
            minimum_error_shots=int(state["minimum_error_shots"]),
            max_positive_residual=float(state["max_positive_residual"]),
            max_negative_residual=float(state["max_negative_residual"]),
            positive_physics_gate=float(state["positive_physics_gate"]),
            protection_ratio=float(state["protection_ratio"]),
            capacity=int(state["capacity"]),
        )
        memory._key_size = (
            int(state["key_size"]) if state.get("key_size") is not None else None
        )
        memory._entries = [
            {
                "shot": int(entry["shot"]),
                "endpoint": int(entry["endpoint"]),
                "group": str(entry["group"]),
                "key": np.asarray(entry["key"], dtype=np.float64),
                "base_logit": float(entry["base_logit"]),
                "physics_support": float(entry["physics_support"]),
            }
            for entry in state["entries"]
        ]
        memory._cache = None
        return memory


@contextmanager
def temporary_optimizer_scale(
    optimizer: torch.optim.Optimizer, scale: float,
) -> Iterator[None]:
    """Scale the actual optimizer learning rate, including for Adam-like rules."""
    if not np.isfinite(scale) or not 0 <= float(scale) <= 1:
        raise ValueError("optimizer scale must lie in [0, 1]")
    original = [float(group["lr"]) for group in optimizer.param_groups]
    try:
        for group, learning_rate in zip(optimizer.param_groups, original):
            group["lr"] = learning_rate * float(scale)
        yield
    finally:
        for group, learning_rate in zip(optimizer.param_groups, original):
            group["lr"] = learning_rate


class IncrementalDualThresholdAlarm:
    """Endpoint-by-endpoint implementation exactly matching the offline rule."""

    def __init__(
        self, *, high_threshold: float = 0.0, physics_weight: float = 8.0,
        trend_weight: float = 12.0, early_threshold: float = 7.9,
        fast_width: int = 5, slow_width: int = 100,
    ) -> None:
        if fast_width <= 0 or slow_width <= fast_width:
            raise ValueError("require 0 < fast_width < slow_width")
        self.high_threshold = float(high_threshold)
        self.physics_weight = float(physics_weight)
        self.trend_weight = float(trend_weight)
        self.early_threshold = float(early_threshold)
        self.fast_width = int(fast_width)
        self.slow_width = int(slow_width)
        self.margin: list[float] = []
        self.support: list[float] = []
        self.times: list[float] = []
        self._safe_cumsum = [0.0]
        self._valid_cumsum = [0]
        self.first_alarm_index: int | None = None
        self.first_route: str | None = None
        self.first_early_score: float | None = None

    def _mean(self, width: int) -> float:
        end = len(self.margin)
        start = max(0, end - int(width))
        total = self._safe_cumsum[end] - self._safe_cumsum[start]
        count = self._valid_cumsum[end] - self._valid_cumsum[start]
        return total / count if count else float("nan")

    def observe(
        self, margin: float, physics_support: float, time_s: float,
    ) -> dict[str, Any]:
        value = float(margin)
        physics = float(physics_support)
        finite_margin = bool(np.isfinite(value))
        finite_support = bool(np.isfinite(physics))
        bounded_support = float(np.clip(physics, 0.0, 1.0)) if finite_support else 0.0
        self.margin.append(value)
        self.support.append(physics)
        self.times.append(float(time_s))
        self._safe_cumsum.append(
            self._safe_cumsum[-1] + (value if finite_margin else 0.0)
        )
        self._valid_cumsum.append(self._valid_cumsum[-1] + int(finite_margin))
        fast = self._mean(self.fast_width)
        slow = self._mean(self.slow_width)
        trend = max(fast - slow, 0.0) if np.isfinite(fast) and np.isfinite(slow) else 0.0
        early_score = (
            value + self.physics_weight * bounded_support + self.trend_weight * trend
            if finite_margin else float("-inf")
        )
        high = finite_margin and value >= self.high_threshold
        early = finite_margin and finite_support and early_score >= self.early_threshold
        index = len(self.margin) - 1
        new_alarm = (high or early) and self.first_alarm_index is None
        if new_alarm:
            self.first_alarm_index = index
            self.first_route = "high" if high else "early"
            self.first_early_score = float(early_score)
        return {
            "index": index, "time_s": float(time_s), "high": bool(high),
            "early": bool(early), "alarmed": bool(high or early),
            "new_alarm": bool(new_alarm), "early_score": float(early_score),
            "trend": float(trend), "physics_support": bounded_support,
        }

    def final_receipt(self) -> dict[str, Any]:
        return {
            "alarmed": self.first_alarm_index is not None,
            "alarm_index": self.first_alarm_index,
            "route": self.first_route,
            "early_score": self.first_early_score,
        }


class ProjectedAlarmController:
    """Causal dual-threshold controller with post-query projected updates.

    Only continuous parameters that passed the balanced360 robustness audit are
    adaptive.  The causal moving-average widths remain fixed at 5/100 because
    changing a discrete history operator online cannot be achieved by a local
    gradient and would make prior alarm semantics unstable.
    """

    PARAMETER_BOUNDS = {
        "high_threshold": (-0.10, 0.10),
        "physics_weight": (6.0, 10.0),
        "trend_weight": (12.0, 15.0),
        "early_threshold": (7.1, 8.7),
    }

    def __init__(
        self, *, high_threshold: float = 0.0,
        physics_weight: float = 8.0, trend_weight: float = 12.0,
        early_threshold: float = 7.9, learning_rate: float = 0.02,
        max_guard_harms: int = 0,
    ) -> None:
        if learning_rate <= 0 or max_guard_harms < 0:
            raise ValueError("invalid alarm adaptation contract")
        self._parameters = {
            "high_threshold": float(high_threshold),
            "physics_weight": float(physics_weight),
            "trend_weight": float(trend_weight),
            "early_threshold": float(early_threshold),
        }
        self._project()
        self.learning_rate = float(learning_rate)
        self.max_guard_harms = int(max_guard_harms)
        self.version = 0
        self.active_shot: int | None = None
        self.pending_shot: int | None = None
        self._pending: dict[str, Any] | None = None
        self._seals: dict[int, dict[str, Any]] = {}
        self._guards: list[dict[str, Any]] = []
        self._incremental: IncrementalDualThresholdAlarm | None = None
        self.last_finalized_shot: int | None = None

    def _project(self) -> None:
        for name, (lower, upper) in self.PARAMETER_BOUNDS.items():
            self._parameters[name] = float(np.clip(
                self._parameters[name], lower, upper,
            ))
        if self._parameters["early_threshold"] >= 7.9:
            self._parameters["physics_weight"] = max(
                self._parameters["physics_weight"], 8.0,
            )

    def parameters_dict(self) -> dict[str, float]:
        return dict(self._parameters)

    def begin_shot(self, shot_id: int) -> None:
        shot_id = int(shot_id)
        if self.active_shot is not None or self.pending_shot is not None:
            raise RuntimeError("previous alarm shot must be finalized")
        if self.last_finalized_shot is not None and shot_id <= self.last_finalized_shot:
            raise RuntimeError("alarm shot ids must be strictly increasing")
        self.active_shot = shot_id
        self._incremental = IncrementalDualThresholdAlarm(
            **self._parameters, fast_width=5, slow_width=100,
        )

    def _alarm(
        self, margin: np.ndarray, support: np.ndarray,
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        values = self._parameters if parameters is None else parameters
        return dual_threshold_alarm(
            margin, support,
            high_threshold=float(values["high_threshold"]),
            physics_weight=float(values["physics_weight"]),
            trend_weight=float(values["trend_weight"]),
            early_threshold=float(values["early_threshold"]),
            fast_width=5, slow_width=100,
        )

    def predict_and_seal(
        self, shot_id: int, margin_trace: np.ndarray,
        physics_support: np.ndarray, times_s: np.ndarray,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if self.active_shot != shot_id:
            raise RuntimeError("begin_shot is required before alarm prediction")
        margin = np.asarray(margin_trace, dtype=np.float64).reshape(-1).copy()
        support = np.asarray(physics_support, dtype=np.float64).reshape(-1).copy()
        times = np.asarray(times_s, dtype=np.float64).reshape(-1).copy()
        if margin.shape != support.shape or margin.shape != times.shape or not len(margin):
            raise ValueError("alarm traces and times must be aligned and non-empty")
        for value, physics, time_s in zip(margin, support, times):
            self.observe_endpoint(shot_id, value, physics, time_s)
        return self.seal_observed_shot(shot_id)

    def observe_endpoint(
        self, shot_id: int, margin: float, physics_support: float,
        time_s: float,
    ) -> dict[str, Any]:
        if self.active_shot != int(shot_id) or self._incremental is None:
            raise RuntimeError("begin_shot is required before endpoint alarm prediction")
        return self._incremental.observe(margin, physics_support, time_s)

    def seal_observed_shot(
        self, shot_id: int, *,
        decision_filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if self.active_shot != shot_id or self._incremental is None:
            raise RuntimeError("cannot seal an inactive incremental alarm shot")
        if not self._incremental.margin:
            raise RuntimeError("cannot seal an alarm shot without endpoint predictions")
        margin = np.asarray(self._incremental.margin, dtype=np.float64)
        support = np.asarray(self._incremental.support, dtype=np.float64)
        times = np.asarray(self._incremental.times, dtype=np.float64)
        unfiltered = self._incremental.final_receipt()
        alarm = dict(unfiltered)
        filter_metadata: dict[str, Any] = {}
        if decision_filter is not None:
            required = {"alarmed", "alarm_index", "route", "early_score"}
            missing = required.difference(decision_filter)
            if missing:
                raise KeyError(f"decision filter missing {sorted(missing)}")
            filtered_alarmed = bool(decision_filter["alarmed"])
            filtered_index = decision_filter["alarm_index"]
            if filtered_alarmed:
                if not bool(unfiltered["alarmed"]):
                    raise ValueError("decision filter cannot advance or create an alarm")
                if filtered_index is None:
                    raise ValueError("filtered alarm requires an index")
                filtered_index = int(filtered_index)
                if (
                    filtered_index < int(unfiltered["alarm_index"])
                    or filtered_index >= len(margin)
                ):
                    raise ValueError("decision filter cannot advance or create an alarm")
            elif filtered_index is not None:
                raise ValueError("vetoed alarm cannot retain an index")
            alarm = {
                "alarmed": filtered_alarmed,
                "alarm_index": filtered_index,
                "route": decision_filter["route"],
                "early_score": decision_filter["early_score"],
            }
            filter_metadata = {
                name: copy.deepcopy(value)
                for name, value in decision_filter.items()
                if name not in required
            }
            filter_metadata["unfiltered_alarm"] = copy.deepcopy(unfiltered)
        receipt = {
            **alarm, **filter_metadata,
            "shot": shot_id,
            "parameter_version": self.version,
            "parameters": self.parameters_dict(),
        }
        self._seals[shot_id] = copy.deepcopy(receipt)
        self._pending = {
            "shot": shot_id, "margin": margin, "support": support,
            "times": times, "receipt": copy.deepcopy(receipt),
        }
        self.active_shot = None
        self.pending_shot = shot_id
        self._incremental = None
        return copy.deepcopy(receipt)

    @staticmethod
    def _group(
        receipt: Mapping[str, Any], times: np.ndarray, *, is_disrupt: bool,
        down_time_s: float | None,
    ) -> str:
        if not is_disrupt:
            return "FP" if bool(receipt["alarmed"]) else "TN"
        if down_time_s is None or not np.isfinite(down_time_s):
            raise ValueError("disruptive alarm update requires DownTime")
        if not bool(receipt["alarmed"]):
            return "FN"
        index = int(receipt["alarm_index"])
        lead_ms = (float(down_time_s) - float(times[index])) * 1000.0
        return "TP" if lead_ms >= 30.0 - 1e-9 else "TARDY"

    def _guard_harms(self, candidate: Mapping[str, float]) -> int:
        harms = 0
        for guard in self._guards:
            alarm = self._alarm(guard["margin"], guard["support"], candidate)
            group = self._group(
                alarm, guard["times"], is_disrupt=guard["is_disrupt"],
                down_time_s=guard["down_time_s"],
            )
            expected = "TP" if guard["is_disrupt"] else "TN"
            harms += int(group != expected)
        return harms

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, is_disrupt: bool,
        down_time_s: float | None, update_value: float,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if self.pending_shot != shot_id or self._pending is None:
            raise RuntimeError("a matching sealed alarm shot is required before reveal")
        if not selected:
            raise PermissionError("alarm parameter update requires a selected query")
        scale = float(np.clip(update_value, 0.0, 1.0))
        pending = self._pending
        group = self._group(
            pending["receipt"], pending["times"], is_disrupt=bool(is_disrupt),
            down_time_s=down_time_s,
        )
        before = self.parameters_dict()
        candidate = dict(before)
        step = self.learning_rate * scale
        route = pending["receipt"].get("route")
        if group == "FP":
            candidate["early_threshold"] += step
            candidate["physics_weight"] -= step
            candidate["trend_weight"] -= 0.5 * step
            if route == "high":
                candidate["high_threshold"] += 0.25 * step
        elif group in ("FN", "TARDY"):
            candidate["early_threshold"] -= step
            candidate["physics_weight"] += step
            candidate["trend_weight"] += 0.5 * step
            candidate["high_threshold"] -= 0.25 * step
        for name, (lower, upper) in self.PARAMETER_BOUNDS.items():
            candidate[name] = float(np.clip(candidate[name], lower, upper))
        if candidate["early_threshold"] >= 7.9:
            candidate["physics_weight"] = max(candidate["physics_weight"], 8.0)
        harms = self._guard_harms(candidate)
        finite = all(np.isfinite(value) for value in candidate.values())
        accepted = finite and harms <= self.max_guard_harms
        changed = accepted and any(
            not math.isclose(candidate[name], before[name]) for name in before
        )
        if accepted:
            self._parameters = candidate
            if changed:
                self.version += 1
        if group in ("TP", "TN"):
            self._guards.append({
                "margin": pending["margin"], "support": pending["support"],
                "times": pending["times"], "is_disrupt": bool(is_disrupt),
                "down_time_s": down_time_s,
            })
        receipt = {
            "event": "projected_alarm_update", "shot": shot_id,
            "discrepancy_group": group, "accepted": bool(accepted),
            "updated": bool(changed), "guard_harms": int(harms),
            "update_value": scale, "parameters_before": before,
            "parameters_after": self.parameters_dict(),
            "sealed_alarm": copy.deepcopy(pending["receipt"]),
            "parameter_version_after": self.version,
        }
        self.pending_shot = None
        self._pending = None
        self.last_finalized_shot = shot_id
        return receipt

    def discard_without_reveal(self, shot_id: int) -> dict[str, Any]:
        if self.pending_shot != int(shot_id):
            raise RuntimeError("a matching sealed alarm shot is required before discard")
        self.pending_shot = None
        self._pending = None
        self.last_finalized_shot = int(shot_id)
        return {"event": "alarm_discard_unqueried", "shot": int(shot_id)}

    def finalize_revealed_no_update(
        self, shot_id: int, *, discrepancy_group: str,
    ) -> dict[str, Any]:
        """Close a revealed shot while freezing the alarm state.

        This is reserved for selector-only compute controls in which query
        identity is the treatment and every non-model learner is held fixed.
        """
        if self.pending_shot != int(shot_id) or self._pending is None:
            raise RuntimeError("a matching sealed alarm shot is required")
        if discrepancy_group not in ("TP", "TN", "FP", "FN", "TARDY"):
            raise ValueError("invalid discrepancy group")
        receipt = {
            "event": "alarm_revealed_no_update", "shot": int(shot_id),
            "discrepancy_group": str(discrepancy_group),
            "updated": False, "parameter_version_after": self.version,
            "parameters_after": self.parameters_dict(),
        }
        self.pending_shot = None
        self._pending = None
        self.last_finalized_shot = int(shot_id)
        return receipt

    def sealed_receipt(self, shot_id: int) -> dict[str, Any]:
        return copy.deepcopy(self._seals[int(shot_id)])


class OnlineQueryValueModel:
    """Small projected utility model updated only from queried sealed shots."""

    DEFAULT_WEIGHTS = {
        "uncertainty": 0.30,
        "physics_contradiction": 0.22,
        "self_supervised_surprise": 0.16,
        "teacher_disagreement": 0.08,
        "private_innovation": 0.12,
        "expected_gradient": 0.12,
    }
    DEFAULT_BOUNDS = {
        "uncertainty": (0.15, 0.50),
        "physics_contradiction": (0.08, 0.40),
        "self_supervised_surprise": (0.05, 0.30),
        "teacher_disagreement": (0.02, 0.20),
        "private_innovation": (0.05, 0.30),
        "expected_gradient": (0.05, 0.30),
    }

    def __init__(self, *, learning_rate: float = 0.02) -> None:
        if learning_rate <= 0:
            raise ValueError("query-value learning rate must be positive")
        self.learning_rate = float(learning_rate)
        self._weights = dict(self.DEFAULT_WEIGHTS)
        self._bias = 0.0
        self._pending: dict[int, dict[str, Any]] = {}
        self.version = 0

    def weights_dict(self) -> dict[str, float]:
        return dict(self._weights)

    def weight_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self.DEFAULT_BOUNDS)

    def score_and_seal(
        self, shot_id: int, features: Mapping[str, float],
    ) -> float:
        if set(features) != set(self._weights):
            raise ValueError("query-value features do not match the registered schema")
        bounded = {
            name: float(np.clip(value, 0.0, 1.0))
            if np.isfinite(value) else 0.0
            for name, value in features.items()
        }
        logit = self._bias + sum(
            self._weights[name] * bounded[name] for name in self._weights
        )
        score = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -30, 30))))
        self._pending[int(shot_id)] = {
            "features": bounded, "score": float(score), "version": self.version,
        }
        return float(score)

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, utility_target: float,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("query value must be sealed before reveal")
        if not selected:
            raise PermissionError("query-value supervision requires a selected query")
        target = float(np.clip(utility_target, 0.0, 1.0))
        pending = self._pending.pop(shot_id)
        error = float(pending["score"] - target)
        before = self.weights_dict()
        for name, value in pending["features"].items():
            lower, upper = self.DEFAULT_BOUNDS[name]
            self._weights[name] = float(np.clip(
                self._weights[name] - self.learning_rate * error * float(value),
                lower, upper,
            ))
        self._bias = float(np.clip(
            self._bias - self.learning_rate * error, -1.0, 1.0,
        ))
        changed = before != self._weights
        if changed:
            self.version += 1
        return {
            "event": "query_value_update", "shot": shot_id,
            "updated": bool(changed), "utility_target": target,
            "prediction_before_reveal": pending["score"],
            "weights_before": before, "weights_after": self.weights_dict(),
            "version_after": self.version,
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)


class AdaptiveBlockBudgetSelector:
    """Immediate exact-budget querying with a past-only adaptive cutoff.

    The cutoff is frozen at the start of each block from the preceding score
    history.  One deterministic, score-independent exploration position is
    reserved per block; the remaining slots use the cutoff and a final causal
    forced fill.  No score from a later shot can alter an earlier decision.
    """

    def __init__(
        self, *, block_size: int = 20, quota: int = 4,
        history_size: int = 80, quantile: float = 0.80,
        exploration_offset: int = 3, exploration_stride: int = 7,
    ) -> None:
        if block_size <= 0 or quota <= 0 or quota > block_size:
            raise ValueError("require 0 < quota <= block_size")
        if history_size <= 0 or not 0 < float(quantile) < 1:
            raise ValueError("invalid adaptive cutoff history contract")
        self.block_size = int(block_size)
        self.quota = int(quota)
        self.history_size = int(history_size)
        self.quantile = float(quantile)
        self.exploration_offset = int(exploration_offset)
        self.exploration_stride = int(exploration_stride)
        self._history: list[float] = []
        self._block = -1
        self._used = 0
        self._exploration_used = False
        self._exploration_position = 0
        self._cutoff: float | None = None
        self._next_index = 0
        self.total_selected = 0

    def _begin_block(self, block: int) -> None:
        prior = self._history[-self.history_size:]
        self._cutoff = (
            float(np.quantile(prior, self.quantile)) if prior else None
        )
        self._block = int(block)
        self._used = 0
        self._exploration_used = False
        self._exploration_position = (
            self.exploration_offset + int(block) * self.exploration_stride
        ) % self.block_size

    def decide(self, *, score: float, stream_index: int) -> dict[str, Any]:
        if int(stream_index) != self._next_index:
            raise RuntimeError("adaptive query decisions must be contiguous and ordered")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("query score must be finite")
        block = int(stream_index) // self.block_size
        position = int(stream_index) % self.block_size
        if block != self._block:
            self._begin_block(block)
        remaining = self.block_size - position
        slots = self.quota - self._used
        reserve_exploration = int(
            not self._exploration_used and position < self._exploration_position
        )
        exploitation_slots = max(slots - reserve_exploration, 0)
        exploration = (
            slots > 0 and not self._exploration_used
            and position == self._exploration_position
        )
        exploitation = (
            slots > 0 and exploitation_slots > 0
            and self._cutoff is not None and value >= self._cutoff
        )
        forced_fill = slots > 0 and remaining <= slots
        selected = bool(exploration or exploitation or forced_fill)
        route = None
        if selected:
            route = (
                "exploration" if exploration else
                "exploitation" if exploitation else "forced_fill"
            )
            self._used += 1
            self.total_selected += 1
            if exploration:
                self._exploration_used = True
        receipt = {
            "selected": selected, "route": route,
            "score": value, "cutoff": self._cutoff,
            "block": block, "position": position,
            "history_count": min(len(self._history), self.history_size),
            "exploration_position": self._exploration_position,
        }
        self._history.append(value)
        self._next_index += 1
        if position == self.block_size - 1 and self._used != self.quota:
            raise RuntimeError("adaptive query selector failed exact block quota")
        return receipt


class CappedAdaptiveBudgetSelector:
    """Past-only query selector with a hard cap, never a forced quota.

    A low-value shot is not queried merely because a block is ending.  This is
    intentionally different from :class:`AdaptiveBlockBudgetSelector`, which
    remains available to reproduce the frozen exact-budget baseline.
    """

    def __init__(
        self, *, block_size: int = 20, quota: int = 4,
        history_size: int = 80, quantile: float = 0.80,
        initial_cutoff: float = 0.50, exploration_per_block: int = 1,
        exploration_offset: int = 3, exploration_stride: int = 7,
    ) -> None:
        if block_size <= 0 or quota <= 0 or quota > block_size:
            raise ValueError("require 0 < quota <= block_size")
        if history_size <= 0 or not 0 < float(quantile) < 1:
            raise ValueError("invalid adaptive cutoff history contract")
        if not 0 <= float(initial_cutoff) <= 1:
            raise ValueError("initial cutoff must lie in [0, 1]")
        if not 0 <= int(exploration_per_block) <= int(quota):
            raise ValueError("exploration count must fit inside the block cap")
        self.block_size = int(block_size)
        self.quota = int(quota)
        self.history_size = int(history_size)
        self.quantile = float(quantile)
        self.initial_cutoff = float(initial_cutoff)
        self.exploration_per_block = int(exploration_per_block)
        self.exploration_offset = int(exploration_offset)
        self.exploration_stride = int(exploration_stride)
        self._history: list[float] = []
        self._block = -1
        self._used = 0
        self._exploration_positions: set[int] = set()
        self._cutoff = self.initial_cutoff
        self._next_index = 0
        self.total_selected = 0

    def _begin_block(self, block: int) -> None:
        prior = self._history[-self.history_size:]
        self._cutoff = (
            float(np.quantile(prior, self.quantile))
            if prior else self.initial_cutoff
        )
        self._block = int(block)
        self._used = 0
        self._exploration_positions = {
            (
                self.exploration_offset
                + int(block) * self.exploration_stride
                + rank * max(1, self.block_size // max(self.exploration_per_block, 1))
            ) % self.block_size
            for rank in range(self.exploration_per_block)
        }

    def decide(self, *, score: float, stream_index: int) -> dict[str, Any]:
        if int(stream_index) != self._next_index:
            raise RuntimeError("capped query decisions must be contiguous and ordered")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("query score must be finite")
        block = int(stream_index) // self.block_size
        position = int(stream_index) % self.block_size
        if block != self._block:
            self._begin_block(block)
        exploration = position in self._exploration_positions
        exploitation = value >= self._cutoff
        selected = self._used < self.quota and (exploration or exploitation)
        route = None
        if selected:
            route = "exploration" if exploration else "error_likelihood"
            self._used += 1
            self.total_selected += 1
        receipt = {
            "selected": bool(selected), "route": route, "score": value,
            "cutoff": self._cutoff, "block": block, "position": position,
            "history_count": min(len(self._history), self.history_size),
            "exploration_position": sorted(self._exploration_positions),
        }
        self._history.append(value)
        self._next_index += 1
        return receipt


class RollingTokenBudgetSelector:
    """Carry unused label budget forward and spend it only on high-value shots."""

    def __init__(
        self, *, query_rate: float = 0.20, capacity: float = 8.0,
        history_size: int = 160, quantile: float = 0.85,
        initial_cutoff: float = 0.40, exploration_interval: int = 40,
        query_cost: float = 0.0,
    ) -> None:
        if not 0 < float(query_rate) <= 1 or float(capacity) < 1:
            raise ValueError("invalid rolling query budget")
        if history_size <= 0 or not 0 < float(quantile) < 1:
            raise ValueError("invalid rolling cutoff")
        if not 0 <= float(initial_cutoff) <= 1 or exploration_interval <= 0:
            raise ValueError("invalid rolling query exploration contract")
        if not 0.0 <= float(query_cost) <= 1.0:
            raise ValueError("query cost must lie in [0, 1]")
        self.query_rate = float(query_rate)
        self.capacity = float(capacity)
        self.history_size = int(history_size)
        self.quantile = float(quantile)
        self.initial_cutoff = float(initial_cutoff)
        self.exploration_interval = int(exploration_interval)
        self.query_cost = float(query_cost)
        self._tokens = 0.0
        self._history: list[float] = []
        self._next_index = 0
        self._last_selected = -1
        self.total_selected = 0

    def decide(self, *, score: float, stream_index: int) -> dict[str, Any]:
        if int(stream_index) != self._next_index:
            raise RuntimeError("rolling query decisions must be contiguous and ordered")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("query score must be finite")
        self._tokens = min(self.capacity, self._tokens + self.query_rate)
        prior = self._history[-self.history_size:]
        warmup = min(20, self.history_size)
        cutoff = (
            float(np.quantile(prior, self.quantile))
            if len(prior) >= warmup else self.initial_cutoff
        )
        exploration = (
            int(stream_index) + 1 >= self.exploration_interval
            and int(stream_index) - self._last_selected >= self.exploration_interval
        )
        exploitation = value >= max(cutoff, self.query_cost)
        selected = self._tokens >= 1.0 and (exploration or exploitation)
        route = None
        if selected:
            route = "exploration" if exploration else "error_likelihood"
            self._tokens -= 1.0
            self._last_selected = int(stream_index)
            self.total_selected += 1
        self._history.append(value)
        self._next_index += 1
        return {
            "selected": bool(selected), "route": route, "score": value,
            "cutoff": cutoff, "block": int(stream_index) // 20,
            "position": int(stream_index) % 20,
            "history_count": min(len(prior), self.history_size),
            "exploration_position": None,
            "query_cost": self.query_cost,
            "net_value": value - self.query_cost,
            "tokens_after": float(self._tokens),
        }


class CausalErrorMemory:
    """Retrieve past revealed mistakes without exposing future outcomes.

    The memory contains only shots that were first predicted and sealed, then
    selected for an outcome query.  FP/FN/TARDY entries provide error evidence;
    TP/TN entries are equally important counterexamples that protect regions
    where the current model was already correct.
    """

    ERROR_GROUPS = frozenset({"FP", "FN", "TARDY"})
    CORRECT_GROUPS = frozenset({"TP", "TN"})
    VALID_GROUPS = ERROR_GROUPS | CORRECT_GROUPS
    GROUP_WEIGHT = {"FP": 0.8, "FN": 1.0, "TARDY": 0.9, "TP": 1.0, "TN": 1.0}

    def __init__(
        self, *, radius: float = 0.35, minimum_support: int = 2,
        memory_weight: float = 0.60, capacity: int = 128,
    ) -> None:
        if radius <= 0 or minimum_support <= 0 or capacity <= 0:
            raise ValueError("invalid causal error-memory geometry")
        if not 0 <= memory_weight <= 1:
            raise ValueError("memory weight must lie in [0, 1]")
        self.radius = float(radius)
        self.minimum_support = int(minimum_support)
        self.memory_weight = float(memory_weight)
        self.capacity = int(capacity)
        self._schema: tuple[str, ...] | None = None
        self._entries: list[dict[str, Any]] = []
        self._pending: dict[int, dict[str, Any]] = {}

    def _vector(self, features: Mapping[str, float]) -> np.ndarray:
        if self._schema is None:
            self._schema = tuple(sorted(features))
        if tuple(sorted(features)) != self._schema:
            raise ValueError("error-memory phenotype schema changed")
        vector = np.asarray([features[name] for name in self._schema], np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("error-memory phenotype must be finite")
        return np.clip(vector, 0.0, 1.0)

    def _evidence(self, vector: np.ndarray) -> dict[str, float]:
        support = {group: 0.0 for group in sorted(self.VALID_GROUPS)}
        for entry in self._entries:
            distance = float(np.linalg.norm(vector - entry["vector"]) / math.sqrt(len(vector)))
            similarity = math.exp(-0.5 * (distance / self.radius) ** 2)
            support[entry["group"]] += similarity * self.GROUP_WEIGHT[entry["group"]]
        error_support = sum(support[group] for group in self.ERROR_GROUPS)
        correct_support = sum(support[group] for group in self.CORRECT_GROUPS)
        total = error_support + correct_support
        confidence = float(np.clip(total / self.minimum_support, 0.0, 1.0))
        raw_error_probability = error_support / total if total > 1e-12 else 0.0
        error_probability = confidence * raw_error_probability
        result = {
            "error_support": float(error_support),
            "correct_support": float(correct_support),
            "memory_confidence": confidence,
            "error_probability": float(error_probability),
            "conditional_error_probability": float(raw_error_probability),
        }
        result.update({f"{group.lower()}_support": float(value) for group, value in support.items()})
        return result

    def score_and_seal(
        self, shot_id: int, features: Mapping[str, float], *, base_score: float,
    ) -> tuple[float, dict[str, float]]:
        shot_id = int(shot_id)
        if shot_id in self._pending:
            raise RuntimeError("shot already sealed in error memory")
        vector = self._vector(features)
        evidence = self._evidence(vector)
        base = float(np.clip(base_score, 0.0, 1.0))
        confidence = evidence["memory_confidence"]
        local_target = (
            evidence["error_probability"] + (1.0 - confidence) * base
        )
        score = base + self.memory_weight * (local_target - base)
        score = float(np.clip(score, 0.0, 1.0))
        self._pending[shot_id] = {
            "vector": vector.copy(), "base_score": base,
            "memory_score": score, "evidence": dict(evidence),
        }
        return score, evidence

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, discrepancy_group: str,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("error-memory phenotype must be sealed before reveal")
        if not selected:
            raise PermissionError("error memory requires a selected query")
        if discrepancy_group not in self.VALID_GROUPS:
            raise ValueError("invalid discrepancy group")
        pending = self._pending.pop(shot_id)
        self._entries.append({
            "shot": shot_id, "group": discrepancy_group,
            "vector": pending["vector"].copy(),
        })
        if len(self._entries) > self.capacity:
            self._entries.pop(0)
        return {
            "event": "causal_error_memory_update", "shot": shot_id,
            "group": discrepancy_group, "stored": True,
            "entry_count": len(self._entries),
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)

    def entries(self) -> list[dict[str, Any]]:
        return [
            {"shot": item["shot"], "group": item["group"],
             "vector": item["vector"].tolist()}
            for item in self._entries
        ]

    def state_dict(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError("cannot checkpoint error memory with unfinalized shots")
        return {
            "radius": self.radius, "minimum_support": self.minimum_support,
            "memory_weight": self.memory_weight, "capacity": self.capacity,
            "schema": list(self._schema or ()), "entries": self.entries(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CausalErrorMemory":
        memory = cls(
            radius=float(state["radius"]),
            minimum_support=int(state["minimum_support"]),
            memory_weight=float(state["memory_weight"]),
            capacity=int(state["capacity"]),
        )
        memory._schema = tuple(str(name) for name in state["schema"]) or None
        memory._entries = [
            {"shot": int(item["shot"]), "group": str(item["group"]),
             "vector": np.asarray(item["vector"], np.float64)}
            for item in state["entries"]
        ]
        return memory


class OperationalQueryValueModel:
    """Bounded two-signal value model for querying operational mistakes.

    Reconstruction, teacher consistency and EAST-private innovation determine
    the intra-shot gradient value.  Outcome queries instead target the two
    signals that remained stable in the balanced360 development audit: alarm
    boundary uncertainty and registered physics contradiction.  The single
    free weight is projected into the audited ``[0.55, 0.75]`` region.
    """

    WEIGHT_BOUNDS = (0.55, 0.75)
    ERROR_GROUPS = frozenset({"FP", "FN", "TARDY"})
    VALID_GROUPS = frozenset({"TP", "TN", "FP", "FN", "TARDY"})

    def __init__(
        self, *, uncertainty_weight: float = 0.65,
        learning_rate: float = 0.05,
    ) -> None:
        lower, upper = self.WEIGHT_BOUNDS
        if not lower <= float(uncertainty_weight) <= upper or learning_rate <= 0:
            raise ValueError("invalid operational query-value contract")
        self.uncertainty_weight = float(uncertainty_weight)
        self.learning_rate = float(learning_rate)
        self._pending: dict[int, dict[str, float]] = {}
        self.version = 0

    def weights_dict(self) -> dict[str, float]:
        return {
            "uncertainty": self.uncertainty_weight,
            "physics_contradiction": 1.0 - self.uncertainty_weight,
        }

    def score_and_seal(
        self, shot_id: int, features: Mapping[str, float],
    ) -> float:
        required = {"uncertainty", "physics_contradiction"}
        if not required.issubset(features):
            raise ValueError("operational query value requires uncertainty and contradiction")
        uncertainty = float(np.clip(features["uncertainty"], 0.0, 1.0))
        contradiction = float(np.clip(
            features["physics_contradiction"], 0.0, 1.0,
        ))
        if not np.isfinite(uncertainty) or not np.isfinite(contradiction):
            raise ValueError("operational query features must be finite")
        score = (
            self.uncertainty_weight * uncertainty
            + (1.0 - self.uncertainty_weight) * contradiction
        )
        self._pending[int(shot_id)] = {
            "uncertainty": uncertainty,
            "physics_contradiction": contradiction,
            "score": float(score),
        }
        return float(score)

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, discrepancy_group: str,
    ) -> dict[str, Any]:
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("operational query value must be sealed before reveal")
        if not selected:
            raise PermissionError("operational query-value update requires a selected query")
        if discrepancy_group not in self.VALID_GROUPS:
            raise ValueError("invalid operational discrepancy group")
        pending = self._pending.pop(shot_id)
        target = float(discrepancy_group in self.ERROR_GROUPS)
        before = self.weights_dict()
        error = pending["score"] - target
        gradient = error * (
            pending["uncertainty"] - pending["physics_contradiction"]
        )
        lower, upper = self.WEIGHT_BOUNDS
        self.uncertainty_weight = float(np.clip(
            self.uncertainty_weight - self.learning_rate * gradient,
            lower, upper,
        ))
        after = self.weights_dict()
        changed = before != after
        if changed:
            self.version += 1
        return {
            "event": "operational_query_value_update", "shot": shot_id,
            "updated": changed, "discrepancy_group": discrepancy_group,
            "utility_target": target,
            "prediction_before_reveal": pending["score"],
            "weights_before": before, "weights_after": after,
            "version_after": self.version,
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        self._pending.pop(int(shot_id), None)

    def finalize_revealed_no_update(self, shot_id: int) -> dict[str, Any]:
        """Consume a sealed queried shot without learning from its outcome."""
        shot_id = int(shot_id)
        if shot_id not in self._pending:
            raise RuntimeError("operational query value must be sealed before reveal")
        self._pending.pop(shot_id)
        return {
            "event": "operational_query_value_revealed_no_update",
            "shot": shot_id, "updated": False,
            "version_after": self.version,
        }


__all__ = [
    "UpdateValueWeights", "CausalUpdateValue", "temporary_optimizer_scale",
    "PrequentialWindowClock", "CausalWindowUpdateScheduler",
    "TypedEndpointResidualMemory",
    "IncrementalDualThresholdAlarm", "ProjectedAlarmController",
    "OnlineQueryValueModel", "AdaptiveBlockBudgetSelector",
    "OperationalQueryValueModel", "CappedAdaptiveBudgetSelector",
    "RollingTokenBudgetSelector",
    "CausalErrorMemory",
]
