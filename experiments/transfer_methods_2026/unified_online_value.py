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


__all__ = [
    "UpdateValueWeights", "CausalUpdateValue", "temporary_optimizer_scale",
    "IncrementalDualThresholdAlarm", "ProjectedAlarmController",
    "OnlineQueryValueModel", "AdaptiveBlockBudgetSelector",
    "OperationalQueryValueModel",
]
