#!/usr/bin/env python3
"""Causal online adaptation of a frozen EAST transfer-expert mixture.

The three experts are independently trained instances of the registered
J-TEXT -> EAST support110 transfer model.  Their neural parameters and alarm
thresholds never change on balanced360.  Before the stream, EAST support110
selects one convex weight vector.  During the stream, the current shot is
predicted and sealed first; an outcome-free value function then decides
whether its HDF5 outcome may be opened.  Only already queried outcomes may
change the convex weights used by later shots.

This module deliberately contains no HDF5 reader.  Outcome access belongs to
the runner so the reveal boundary can be audited independently.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

import dual_threshold_alarm


DEFAULT_ALARM_PARAMETERS = {
    "high_threshold": 0.0,
    "physics_weight": 8.0,
    "trend_weight": 12.0,
    "early_threshold": 7.9,
}


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_weights(values: np.ndarray | Sequence[float], members: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        weights.shape != (int(members),)
        or not np.isfinite(weights).all()
        or np.any(weights < -1e-12)
        or not np.isclose(float(weights.sum()), 1.0, atol=1e-10)
    ):
        raise ValueError("expert weights must be a finite probability simplex")
    weights = np.maximum(weights, 0.0)
    return weights / weights.sum()


def simplex_grid(*, members: int = 3, step: float = 0.1) -> np.ndarray:
    """Enumerate a deterministic convex grid without floating-point drift."""
    if int(members) <= 0 or not 0.0 < float(step) <= 1.0:
        raise ValueError("invalid simplex-grid dimensions")
    denominator = int(round(1.0 / float(step)))
    if not np.isclose(denominator * float(step), 1.0, atol=1e-12):
        raise ValueError("simplex step must divide one exactly")

    rows: list[list[int]] = []

    def visit(prefix: list[int], remaining: int, width: int) -> None:
        if width == 1:
            rows.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value, width - 1)

    visit([], denominator, int(members))
    return np.asarray(rows, dtype=np.float64) / float(denominator)


def _logit(probability: np.ndarray | float) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def select_support_weights(
    receipts: Sequence[Mapping[str, Any]], *, candidates: np.ndarray,
) -> dict[str, Any]:
    """Choose frozen initial weights using aligned EAST support receipts only."""
    if not receipts:
        raise ValueError("support selection requires at least one expert receipt")
    member_count = len(receipts)
    grid = np.asarray(candidates, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != member_count or not len(grid):
        raise ValueError("support candidate grid does not match expert count")
    normalized_grid = np.stack([
        _as_weights(row, member_count) for row in grid
    ])

    reference_roster: list[int] | None = None
    reference_outcomes: np.ndarray | None = None
    margins: list[np.ndarray] = []
    receipt_summaries: list[dict[str, Any]] = []
    for receipt in receipts:
        threshold = float(receipt.get("chosen_threshold", float("nan")))
        predictions = receipt.get("predictions")
        if not 0.0 < threshold < 1.0 or not isinstance(predictions, list) or not predictions:
            raise ValueError("support threshold receipt is incomplete")
        roster = [int(row["shot"]) for row in predictions]
        outcomes = np.asarray([bool(row["outcome"]) for row in predictions])
        scores = np.asarray([float(row["score"]) for row in predictions])
        if len(roster) != len(set(roster)):
            raise RuntimeError("support receipt roster contains duplicate shots")
        if reference_roster is None:
            reference_roster = roster
            reference_outcomes = outcomes
        elif roster != reference_roster:
            raise RuntimeError("expert support receipt roster mismatch")
        elif not np.array_equal(outcomes, reference_outcomes):
            raise RuntimeError("expert support outcomes mismatch")
        margins.append(_logit(scores) - float(_logit(threshold)))
        receipt_summaries.append({
            "chosen_threshold": threshold,
            "roster_count": len(roster),
            "roster_sha256": _sha256_json(roster),
        })
    assert reference_outcomes is not None
    labels = reference_outcomes
    if not np.any(labels) or not np.any(~labels):
        raise RuntimeError("support weight selection requires both outcomes")
    matrix = np.stack(margins, axis=1)
    equal = np.full(member_count, 1.0 / member_count)
    rows: list[dict[str, Any]] = []
    for candidate in normalized_grid:
        alarmed = matrix @ candidate >= 0.0
        tpr = float(np.mean(alarmed[labels]))
        fpr = float(np.mean(alarmed[~labels]))
        rows.append({
            "weights": candidate.tolist(),
            "tpr": tpr,
            "fpr": fpr,
            "j": tpr - fpr,
            "distance_to_equal": float(np.linalg.norm(candidate - equal)),
        })
    chosen = min(
        rows,
        key=lambda row: (
            -float(row["j"]),
            float(row["distance_to_equal"]),
            tuple(float(value) for value in row["weights"]),
        ),
    )
    return {
        "weights": list(chosen["weights"]),
        "metrics": {
            "tpr": float(chosen["tpr"]),
            "fpr": float(chosen["fpr"]),
            "j": float(chosen["j"]),
        },
        "selection_rule": "max support TPR-FPR, tie nearest equal weights",
        "candidate_count": len(rows),
        "support_count": int(len(labels)),
        "support_disruptive": int(labels.sum()),
        "support_safe": int((~labels).sum()),
        "receipts": receipt_summaries,
    }


def mixture_alarm(
    expert_margin_traces: np.ndarray, physics_support: np.ndarray,
    weights: np.ndarray | Sequence[float],
    alarm_parameters: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the registered causal alarm to a convex expert-margin trace."""
    traces = np.asarray(expert_margin_traces, dtype=np.float64)
    physics = np.asarray(physics_support, dtype=np.float64).reshape(-1)
    if traces.ndim != 2 or not traces.shape[0] or traces.shape[1] != len(physics):
        raise ValueError("expert traces and physics support must align")
    if not np.isfinite(traces).all() or not np.isfinite(physics).all():
        raise ValueError("expert traces and physics support must be finite")
    normalized = _as_weights(weights, traces.shape[0])
    parameters = dict(DEFAULT_ALARM_PARAMETERS)
    if alarm_parameters is not None:
        parameters.update({key: float(value) for key, value in alarm_parameters.items()})
    mixed_margin = normalized @ traces
    decision = dual_threshold_alarm.dual_threshold_alarm(
        mixed_margin, physics, **parameters,
    )
    return {
        **decision,
        "weights": normalized.tolist(),
        "mixed_margin_max": float(np.max(mixed_margin)),
    }


def expert_alarm_evidence(
    expert_margin_traces: np.ndarray, physics_support: np.ndarray,
    alarm_parameters: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return each frozen expert's maximum causal alarm-boundary excess."""
    traces = np.asarray(expert_margin_traces, dtype=np.float64)
    physics = np.asarray(physics_support, dtype=np.float64).reshape(-1)
    if traces.ndim != 2 or traces.shape[1] != len(physics) or not len(physics):
        raise ValueError("expert evidence inputs must align")
    parameters = dict(DEFAULT_ALARM_PARAMETERS)
    if alarm_parameters is not None:
        parameters.update({key: float(value) for key, value in alarm_parameters.items()})
    evidence = []
    for margin in traces:
        trace = dual_threshold_alarm.dual_threshold_trace(
            margin, physics, **parameters,
        )
        high_excess = np.max(margin - float(parameters["high_threshold"]))
        early_excess = np.max(
            trace["early_score"] - float(parameters["early_threshold"])
        )
        evidence.append(float(max(high_excess, early_excess)))
    result = np.asarray(evidence, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("expert alarm evidence must be finite")
    return result


def active_query_value(
    expert_evidence: np.ndarray | Sequence[float], *,
    weights: np.ndarray | Sequence[float], uncertainty_weight: float = 0.55,
    disagreement_weight: float = 0.45, disagreement_scale: float = 0.25,
) -> dict[str, float]:
    """Outcome-free value: boundary uncertainty plus frozen-expert disagreement."""
    evidence = np.asarray(expert_evidence, dtype=np.float64).reshape(-1)
    normalized = _as_weights(weights, len(evidence))
    if not np.isfinite(evidence).all() or float(disagreement_scale) <= 0.0:
        raise ValueError("invalid expert query evidence")
    if (
        float(uncertainty_weight) < 0.0 or float(disagreement_weight) < 0.0
        or not np.isclose(
            float(uncertainty_weight) + float(disagreement_weight), 1.0,
            atol=1e-12,
        )
    ):
        raise ValueError("query-value weights must be non-negative and sum to one")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(evidence, -60.0, 60.0)))
    mixture_probability = float(normalized @ probabilities)
    uncertainty = float(4.0 * mixture_probability * (1.0 - mixture_probability))
    spread = float(np.std(probabilities))
    disagreement = spread / (spread + float(disagreement_scale))
    score = (
        float(uncertainty_weight) * uncertainty
        + float(disagreement_weight) * disagreement
    )
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
        "expert_disagreement": float(np.clip(disagreement, 0.0, 1.0)),
        "mixture_probability": float(np.clip(mixture_probability, 0.0, 1.0)),
    }


class CausalExpertWeightController:
    """Update only convex expert weights from already queried shot outcomes."""

    def __init__(
        self, *, initial_weights: np.ndarray | Sequence[float],
        candidates: np.ndarray, minimum_per_class: int = 16,
        minimum_lead_ms: float = 30.0,
        maximum_expert_weight: float | None = None,
        alarm_parameters: Mapping[str, float] | None = None,
        uncertainty_weight: float = 0.55,
        disagreement_weight: float = 0.45,
        disagreement_scale: float = 0.25,
    ) -> None:
        grid = np.asarray(candidates, dtype=np.float64)
        if grid.ndim != 2 or not len(grid):
            raise ValueError("controller requires a non-empty candidate grid")
        normalized_candidates = np.stack([
            _as_weights(row, grid.shape[1]) for row in grid
        ])
        self.initial_weights = _as_weights(initial_weights, grid.shape[1])
        self.maximum_expert_weight = (
            1.0 if maximum_expert_weight is None else float(maximum_expert_weight)
        )
        if (
            not 0.0 < self.maximum_expert_weight <= 1.0
            or np.max(self.initial_weights) > self.maximum_expert_weight + 1e-12
        ):
            raise ValueError(
                "maximum expert weight must be valid and preserve the support initialization"
            )
        allowed = np.max(normalized_candidates, axis=1) <= (
            self.maximum_expert_weight + 1e-12
        )
        if not np.any(allowed):
            raise ValueError("expert exposure cap removed every candidate")
        self.candidates = normalized_candidates[allowed]
        self.weights = self.initial_weights.copy()
        if int(minimum_per_class) <= 0 or float(minimum_lead_ms) <= 0.0:
            raise ValueError("online weight-update gates must be positive")
        self.minimum_per_class = int(minimum_per_class)
        self.minimum_lead_ms = float(minimum_lead_ms)
        self.alarm_parameters = dict(DEFAULT_ALARM_PARAMETERS)
        if alarm_parameters is not None:
            self.alarm_parameters.update({
                key: float(value) for key, value in alarm_parameters.items()
            })
        self.uncertainty_weight = float(uncertainty_weight)
        self.disagreement_weight = float(disagreement_weight)
        self.disagreement_scale = float(disagreement_scale)
        self._history: list[dict[str, Any]] = []
        self._pending: dict[int, dict[str, Any]] = {}
        self._last_shot: int | None = None
        self.version = 0

    def predict_and_seal(
        self, shot_id: int, *, expert_margin_traces: np.ndarray,
        physics_support: np.ndarray, endpoint_times_s: np.ndarray,
    ) -> dict[str, Any]:
        shot = int(shot_id)
        if self._pending:
            raise RuntimeError("previous expert-mixture prediction is still pending")
        if self._last_shot is not None and shot <= self._last_shot:
            raise RuntimeError("expert-mixture shots must be strictly increasing")
        traces = np.asarray(expert_margin_traces, dtype=np.float64)
        physics = np.asarray(physics_support, dtype=np.float64).reshape(-1)
        times = np.asarray(endpoint_times_s, dtype=np.float64).reshape(-1)
        if traces.ndim != 2 or traces.shape[1] != len(times) or physics.shape != times.shape:
            raise ValueError("expert traces, physics, and endpoint times must align")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            if len(times) > 1:
                raise ValueError("endpoint times must be finite and strictly increasing")
            if not np.isfinite(times).all():
                raise ValueError("endpoint times must be finite and strictly increasing")

        decision = mixture_alarm(
            traces, physics, self.weights, self.alarm_parameters,
        )
        candidate_decisions = [
            mixture_alarm(traces, physics, weights, self.alarm_parameters)
            for weights in self.candidates
        ]
        evidence = expert_alarm_evidence(
            traces, physics, self.alarm_parameters,
        )
        query = active_query_value(
            evidence, weights=self.weights,
            uncertainty_weight=self.uncertainty_weight,
            disagreement_weight=self.disagreement_weight,
            disagreement_scale=self.disagreement_scale,
        )
        alarm_index = decision["alarm_index"]
        alarm_time = None if alarm_index is None else float(times[int(alarm_index)])
        receipt = {
            "shot": shot,
            "alarmed": bool(decision["alarmed"]),
            "alarm_index": None if alarm_index is None else int(alarm_index),
            "alarm_time_s": alarm_time,
            "route": decision["route"],
            "weights_before_prediction": self.weights.tolist(),
            "weight_version": int(self.version),
            "expert_evidence": evidence.tolist(),
            "query_value": query,
        }
        receipt["prediction_sha256"] = _sha256_json(receipt)
        self._pending[shot] = {
            "receipt": copy.deepcopy(receipt),
            "candidate_decisions": candidate_decisions,
            "endpoint_times_s": times.copy(),
        }
        self._last_shot = shot
        return copy.deepcopy(receipt)

    def update_after_reveal(
        self, shot_id: int, *, selected: bool, is_disrupt: bool,
        down_time_s: float | None,
    ) -> dict[str, Any]:
        shot = int(shot_id)
        if not bool(selected):
            raise RuntimeError("outcome reveal is forbidden unless the shot was selected")
        if shot not in self._pending:
            raise RuntimeError("cannot update without a sealed pending prediction")
        disruptive = bool(is_disrupt)
        if disruptive and (
            down_time_s is None or not np.isfinite(float(down_time_s))
        ):
            raise ValueError("a queried disruptive shot requires finite down time")
        pending = self._pending.pop(shot)
        times = pending["endpoint_times_s"]
        timely: list[bool] = []
        false_alarm: list[bool] = []
        for decision in pending["candidate_decisions"]:
            alarmed = bool(decision["alarmed"])
            index = decision["alarm_index"]
            if disruptive and alarmed:
                lead_ms = (
                    float(down_time_s) - float(times[int(index)])
                ) * 1000.0
                timely.append(bool(lead_ms >= self.minimum_lead_ms - 1e-9))
            else:
                timely.append(False)
            false_alarm.append(bool((not disruptive) and alarmed))
        self._history.append({
            "shot": shot,
            "is_disrupt": disruptive,
            "timely": timely,
            "false_alarm": false_alarm,
        })
        before = self.weights.copy()
        positives = sum(int(row["is_disrupt"]) for row in self._history)
        negatives = len(self._history) - positives
        candidate_metrics: list[dict[str, float]] = []
        if positives >= self.minimum_per_class and negatives >= self.minimum_per_class:
            timely_matrix = np.asarray([row["timely"] for row in self._history], bool)
            fp_matrix = np.asarray([row["false_alarm"] for row in self._history], bool)
            labels = np.asarray([row["is_disrupt"] for row in self._history], bool)
            for index, candidate in enumerate(self.candidates):
                tpr = float(np.mean(timely_matrix[labels, index]))
                fpr = float(np.mean(fp_matrix[~labels, index]))
                candidate_metrics.append({"tpr": tpr, "fpr": fpr, "j": tpr - fpr})
            chosen = min(
                range(len(self.candidates)),
                key=lambda index: (
                    -candidate_metrics[index]["j"],
                    float(np.linalg.norm(
                        self.candidates[index] - self.initial_weights
                    )),
                    tuple(float(value) for value in self.candidates[index]),
                ),
            )
            self.weights = self.candidates[chosen].copy()
        changed = not np.allclose(before, self.weights, atol=1e-12)
        if changed:
            self.version += 1
        return {
            "shot": shot,
            "selected": True,
            "history_count": len(self._history),
            "disruptive_history_count": positives,
            "safe_history_count": negatives,
            "minimum_per_class": self.minimum_per_class,
            "weights_before": before.tolist(),
            "weights_after": self.weights.tolist(),
            "weight_changed": bool(changed),
            "weight_version": int(self.version),
            "candidate_metrics_available": bool(candidate_metrics),
        }

    def discard_without_reveal(self, shot_id: int) -> None:
        shot = int(shot_id)
        if shot not in self._pending:
            raise RuntimeError("cannot discard a shot without a sealed prediction")
        self._pending.pop(shot)

    def state_dict(self) -> dict[str, Any]:
        return {
            "initial_weights": self.initial_weights.tolist(),
            "weights": self.weights.tolist(),
            "candidates": self.candidates.tolist(),
            "minimum_per_class": self.minimum_per_class,
            "minimum_lead_ms": self.minimum_lead_ms,
            "maximum_expert_weight": self.maximum_expert_weight,
            "alarm_parameters": copy.deepcopy(self.alarm_parameters),
            "revealed_history": copy.deepcopy(self._history),
            "pending_shots": sorted(self._pending),
            "last_shot": self._last_shot,
            "weight_version": int(self.version),
        }


__all__ = [
    "CausalExpertWeightController", "DEFAULT_ALARM_PARAMETERS",
    "active_query_value", "expert_alarm_evidence", "mixture_alarm",
    "select_support_weights", "simplex_grid",
]
