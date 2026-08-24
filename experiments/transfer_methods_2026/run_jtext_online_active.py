#!/usr/bin/env python3
"""J-TEXT TEST220 two-timescale online-active diagnostic runner.

The static source prediction seal is completed before the outcome oracle is
constructed.  Online state is local to this runner and is never a transfer
parent: every cross-device run must continue to load the original locked
TRAIN1353/VAL160 source checkpoint.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jtext_online_adapter as online_adapter
import jtext_online_io as jtext_io


ARMS = (
    "static",
    "intra_only",
    "inter_random",
    "inter_active",
    "full_random",
    "full_active",
    "frontload_equal_labels",
    "equal_compute_random",
)
CLAIM_STATUS = "TEST_DRIVEN_EXPLORATION"
RESEARCH_CLAIM = "JTEXT_TEST220_ORDERED_PREQUENTIAL_DIAGNOSTIC"
RUNNER_RESUME_SCHEMA = "jtext-online-active-resume-v1"
STATIC_SEAL_SCHEMA = "jtext-online-static-seal-v1"
RUN_SCHEMA = "jtext-online-active-run-v1"
FULL_ARMS = frozenset({"full_random", "full_active", "equal_compute_random"})
ACTIVE_ARMS = frozenset({"inter_active", "full_active"})
RANDOM_ARMS = frozenset({"inter_random", "full_random", "equal_compute_random"})
INTRA_ARMS = frozenset({"intra_only", "full_random", "full_active", "equal_compute_random"})
INTER_ARMS = frozenset({
    "inter_random", "inter_active", "full_random", "full_active",
    "frontload_equal_labels", "equal_compute_random",
})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(Path(path), _canonical_json(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical_json(dict(row)) + b"\n" for row in rows)
    _atomic_write(Path(path), payload)


@dataclass(frozen=True)
class JTextQueryValue:
    """Frozen TEST220 query-value equation and TRAIN/VAL-selected scales."""

    w: float = 0.65
    eta: float = 0.10
    uncertainty_scale: float = 1.0
    contradiction_scale: float = 1.0
    novelty_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.w) <= 1.0:
            raise ValueError("J-TEXT query weight w must lie in [0, 1]")
        if not math.isfinite(float(self.eta)) or float(self.eta) < 0.0:
            raise ValueError("J-TEXT novelty coefficient eta must be non-negative")
        scales = (
            self.uncertainty_scale, self.contradiction_scale, self.novelty_scale,
        )
        if any(not math.isfinite(float(scale)) or float(scale) <= 0 for scale in scales):
            raise ValueError("J-TEXT query scales must be finite and positive")

    @property
    def weight(self) -> float:
        return float(self.w)

    @property
    def scales(self) -> tuple[float, float, float]:
        return (
            float(self.uncertainty_scale),
            float(self.contradiction_scale),
            float(self.novelty_scale),
        )

    def evaluate(
        self,
        *,
        uncertainty: float,
        contradiction: float,
        phenotype: Sequence[float] | np.ndarray,
        memory: "JTextRevealedPhenotypeMemory",
    ) -> dict[str, Any]:
        raw_uncertainty = float(uncertainty)
        raw_contradiction = float(contradiction)
        if not math.isfinite(raw_uncertainty) or not math.isfinite(raw_contradiction):
            raise ValueError("J-TEXT query components must be finite")
        scaled_uncertainty = float(np.clip(
            raw_uncertainty / self.uncertainty_scale, 0.0, 1.0,
        ))
        scaled_contradiction = float(np.clip(
            raw_contradiction / self.contradiction_scale, 0.0, 1.0,
        ))
        novelty = memory.novelty(phenotype, scale=self.novelty_scale)
        value = (
            self.w * scaled_uncertainty
            + (1.0 - self.w) * scaled_contradiction
            + self.eta * novelty
        )
        return {
            "value": float(value),
            "uncertainty": scaled_uncertainty,
            "contradiction": scaled_contradiction,
            "novelty": float(novelty),
            "w": float(self.w),
            "eta": float(self.eta),
            "scales": {
                "uncertainty": float(self.uncertainty_scale),
                "contradiction": float(self.contradiction_scale),
                "novelty": float(self.novelty_scale),
            },
            "revealed_memory_size": memory.size,
        }


class JTextRevealedPhenotypeMemory:
    """Phenotypes become visible to novelty scoring only after reveal."""

    def __init__(self) -> None:
        self._entries: list[tuple[int, np.ndarray]] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def shots(self) -> tuple[int, ...]:
        return tuple(shot for shot, _phenotype in self._entries)

    def novelty(
        self, phenotype: Sequence[float] | np.ndarray, *, scale: float,
    ) -> float:
        current = np.asarray(phenotype, dtype=np.float64).reshape(-1)
        if not len(current) or not np.all(np.isfinite(current)):
            raise ValueError("J-TEXT query phenotype must be finite and non-empty")
        if not self._entries:
            return 0.0
        distances = []
        for _shot, prior in self._entries:
            if prior.shape != current.shape:
                raise ValueError("J-TEXT query phenotype shape changed")
            distances.append(float(np.sqrt(np.mean(np.square(current - prior)))))
        return float(np.clip(min(distances) / float(scale), 0.0, 1.0))

    def add_after_reveal(
        self, shot_id: int, phenotype: Sequence[float] | np.ndarray,
    ) -> None:
        shot = int(shot_id)
        if self._entries and shot <= self._entries[-1][0]:
            raise RuntimeError("revealed J-TEXT phenotype shots must be strictly increasing")
        value = np.asarray(phenotype, dtype=np.float64).reshape(-1)
        if not len(value) or not np.all(np.isfinite(value)):
            raise ValueError("revealed J-TEXT phenotype must be finite and non-empty")
        value = np.ascontiguousarray(value)
        value.setflags(write=False)
        self._entries.append((shot, value))

    def state_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {"shot": shot, "phenotype": phenotype.tolist()}
                for shot, phenotype in self._entries
            ],
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._entries = []
        for row in payload.get("entries", []):
            self.add_after_reveal(int(row["shot"]), row["phenotype"])


def budget_quota(percent: int | float, *, block_size: int = 20) -> int:
    value = float(percent)
    if value not in (10.0, 20.0, 30.0):
        raise ValueError("J-TEXT budget sensitivity must be 10, 20, or 30 percent")
    quota = int(round(int(block_size) * value / 100.0))
    if quota <= 0 or not math.isclose(quota, int(block_size) * value / 100.0):
        raise ValueError("J-TEXT block size does not realize the requested budget exactly")
    return quota


class JTextExactBlockSelector:
    """Past-only score selection with an exact causal forced fill."""

    def __init__(self, *, cutoff: float, block_size: int, quota: int) -> None:
        if int(block_size) <= 0 or not 0 < int(quota) <= int(block_size):
            raise ValueError("invalid J-TEXT exact block budget")
        if not math.isfinite(float(cutoff)):
            raise ValueError("J-TEXT active cutoff must be finite")
        self.cutoff = float(cutoff)
        self.block_size = int(block_size)
        self.quota = int(quota)
        self._next_index = 0
        self._block = -1
        self._used = 0

    def decide(self, *, score: float, stream_index: int) -> dict[str, Any]:
        index = int(stream_index)
        value = float(score)
        if index != self._next_index:
            raise RuntimeError("J-TEXT active decisions must be contiguous")
        if not math.isfinite(value):
            raise ValueError("J-TEXT active query score must be finite")
        block = index // self.block_size
        position = index % self.block_size
        if block != self._block:
            self._block = block
            self._used = 0
        slots = self.quota - self._used
        remaining = self.block_size - position
        active = slots > 0 and value >= self.cutoff
        forced = slots > 0 and remaining <= slots
        selected = bool(active or forced)
        route = None
        if selected:
            route = "active" if active else "forced_fill"
            self._used += 1
        self._next_index += 1
        if position == self.block_size - 1 and self._used != self.quota:
            raise RuntimeError("J-TEXT exact block query budget was not filled")
        return {
            "selected": selected,
            "route": route,
            "block": block,
            "position": position,
            "score": value,
            "cutoff": self.cutoff,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff,
            "block_size": self.block_size,
            "quota": self.quota,
            "next_index": self._next_index,
            "block": self._block,
            "used": self._used,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if (
            float(payload.get("cutoff")) != self.cutoff
            or int(payload.get("block_size")) != self.block_size
            or int(payload.get("quota")) != self.quota
        ):
            raise ValueError("J-TEXT active selector resume contract mismatch")
        self._next_index = int(payload["next_index"])
        self._block = int(payload["block"])
        self._used = int(payload["used"])


def _validate_complete_blocks(shots: Sequence[int], block_size: int) -> tuple[int, ...]:
    roster = tuple(int(shot) for shot in shots)
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("J-TEXT online roster must be non-empty and unique")
    if len(roster) % int(block_size):
        raise ValueError("J-TEXT online roster must contain complete query blocks")
    return roster


def build_random_query_manifest(
    ordered_shots: Sequence[int], *, budget_percent: int, seed: int,
    block_size: int = 20, quota: int | None = None,
) -> list[dict[str, Any]]:
    roster = _validate_complete_blocks(ordered_shots, block_size)
    block_quota = budget_quota(budget_percent, block_size=block_size) if quota is None else int(quota)
    if not 0 < block_quota <= int(block_size):
        raise ValueError("invalid J-TEXT random block quota")
    generator = np.random.default_rng(int(seed))
    selected: set[int] = set()
    for begin in range(0, len(roster), int(block_size)):
        positions = generator.choice(int(block_size), size=block_quota, replace=False)
        selected.update(begin + int(position) for position in positions)
    return [
        {
            "stream_index": index,
            "shot": shot,
            "block": index // int(block_size),
            "selected": index in selected,
            "route": "random" if index in selected else None,
        }
        for index, shot in enumerate(roster)
    ]


def build_active_query_manifest(
    ordered_shots: Sequence[int], *, scores: Sequence[float],
    budget_percent: int, cutoff: float, block_size: int = 20,
    quota: int | None = None,
) -> list[dict[str, Any]]:
    roster = _validate_complete_blocks(ordered_shots, block_size)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(roster) or not np.all(np.isfinite(values)):
        raise ValueError("J-TEXT active scores must align with the roster")
    block_quota = budget_quota(budget_percent, block_size=block_size) if quota is None else int(quota)
    selector = JTextExactBlockSelector(
        cutoff=cutoff, block_size=block_size, quota=block_quota,
    )
    rows = []
    for index, (shot, score) in enumerate(zip(roster, values)):
        decision = selector.decide(score=float(score), stream_index=index)
        rows.append({
            "stream_index": index,
            "shot": shot,
            "block": decision["block"],
            "selected": decision["selected"],
            "route": decision["route"],
            "score": float(score),
        })
    return rows


@dataclass(frozen=True)
class JTextComputeCaps:
    current_windows: int = 8
    replay_windows: int = 8
    guard_windows: int = 8
    optimizer_attempts: int = 2

    def __post_init__(self) -> None:
        if any(int(value) <= 0 for value in (
            self.current_windows, self.replay_windows,
            self.guard_windows, self.optimizer_attempts,
        )):
            raise ValueError("J-TEXT matched compute caps must be positive")


def realize_compute_ledger(
    query_manifest: Sequence[Mapping[str, Any]], *, caps: JTextComputeCaps,
    online_parameter_count: int,
) -> list[dict[str, Any]]:
    if int(online_parameter_count) <= 0:
        raise ValueError("J-TEXT compute ledger requires online parameters")
    rows = []
    for ordinal, query in enumerate(row for row in query_manifest if row["selected"]):
        replay_windows = 0 if ordinal == 0 else int(caps.replay_windows)
        flop_inputs = {
            "online_parameter_count": int(online_parameter_count),
            "current_windows": int(caps.current_windows),
            "replay_windows": replay_windows,
            "guard_windows": int(caps.guard_windows),
            "optimizer_attempts": int(caps.optimizer_attempts),
        }
        rows.append({
            "attempt_index": ordinal,
            "stream_index": int(query["stream_index"]),
            "shot": int(query["shot"]),
            "optimizer_attempts": int(caps.optimizer_attempts),
            "current_windows": int(caps.current_windows),
            "replay_windows": replay_windows,
            "guard_windows": int(caps.guard_windows),
            "replay_size": ordinal,
            "flop_inputs": flop_inputs,
            "reported_flops": int(
                2 * int(online_parameter_count)
                * (int(caps.current_windows) + replay_windows + int(caps.guard_windows))
                * int(caps.optimizer_attempts)
            ),
        })
    return rows


def _queries_per_block(manifest: Sequence[Mapping[str, Any]]) -> list[int]:
    blocks = 1 + max(int(row["block"]) for row in manifest)
    return [
        sum(bool(row["selected"]) and int(row["block"]) == block for row in manifest)
        for block in range(blocks)
    ]


def audit_matched_controls(
    active_manifest: Sequence[Mapping[str, Any]],
    random_manifest: Sequence[Mapping[str, Any]],
    active_compute: Sequence[Mapping[str, Any]],
    random_compute: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    active_resources = [{
        name: row[name] for name in (
            "optimizer_attempts", "current_windows", "replay_windows",
            "guard_windows", "replay_size", "flop_inputs", "reported_flops",
        )
    } for row in active_compute]
    random_resources = [{
        name: row[name] for name in (
            "optimizer_attempts", "current_windows", "replay_windows",
            "guard_windows", "replay_size", "flop_inputs", "reported_flops",
        )
    } for row in random_compute]
    active_per_block = _queries_per_block(active_manifest)
    random_per_block = _queries_per_block(random_manifest)
    result = {
        "active_queries": sum(bool(row["selected"]) for row in active_manifest),
        "random_queries": sum(bool(row["selected"]) for row in random_manifest),
        "queries_per_block": {
            "active": active_per_block, "random": random_per_block,
        },
        "optimizer_attempts": {
            "active": sum(int(row["optimizer_attempts"]) for row in active_compute),
            "random": sum(int(row["optimizer_attempts"]) for row in random_compute),
        },
        "current_windows": {
            "active": sum(int(row["current_windows"]) for row in active_compute),
            "random": sum(int(row["current_windows"]) for row in random_compute),
        },
        "replay_windows": {
            "active": sum(int(row["replay_windows"]) for row in active_compute),
            "random": sum(int(row["replay_windows"]) for row in random_compute),
        },
        "guard_windows": {
            "active": sum(int(row["guard_windows"]) for row in active_compute),
            "random": sum(int(row["guard_windows"]) for row in random_compute),
        },
    }
    result["matched"] = bool(
        result["active_queries"] == result["random_queries"]
        and active_per_block == random_per_block
        and active_resources == random_resources
    )
    return result


def build_frontload_plan(
    ordered_shots: Sequence[int], *, budget_percent: int,
    block_size: int = 20,
) -> dict[str, Any]:
    roster = _validate_complete_blocks(ordered_shots, block_size)
    label_count = len(roster) // int(block_size) * budget_quota(
        budget_percent, block_size=block_size,
    )
    if not 0 < label_count < len(roster):
        raise ValueError("J-TEXT frontload requires a non-empty evaluation suffix")
    training = roster[:label_count]
    evaluation = roster[label_count:]
    mask = [index >= label_count for index in range(len(roster))]
    return {
        "selection_rule": "label_free_chronological_prefix",
        "training_shots": list(training),
        "suffix_start_index": label_count,
        "evaluation_shots": list(evaluation),
        "evaluation_mask": mask,
    }


def abnormal_run_gate(
    *, arm: str, operational_tpr: float, fpr: float,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown J-TEXT online arm")
    tpr = float(operational_tpr)
    false_positive_rate = float(fpr)
    if not math.isfinite(tpr) or not math.isfinite(false_positive_rate):
        raise ValueError("J-TEXT abnormal-run metrics must be finite")
    reasons = []
    if arm in FULL_ARMS and tpr < 0.60:
        reasons.append("TPR_BELOW_0.60")
    if arm in FULL_ARMS and false_positive_rate > 0.40:
        reasons.append("FPR_ABOVE_0.40")
    triggered = bool(reasons)
    return {
        "schema_version": "jtext-abnormal-run-audit-v1",
        "arm": arm,
        "triggered": triggered,
        "status": "STOPPED_ABNORMAL_RUN" if triggered else "PASS",
        "reasons": reasons,
        "operational_tpr": tpr,
        "fpr": false_positive_rate,
        "tuning_allowed": not triggered,
        "required_audits": [
            "label", "threshold", "alarm_index", "scaler", "order", "output",
        ] if triggered else [],
    }


class _JTextReplayMemory:
    def __init__(self, *, capacity_shots: int = 64) -> None:
        self.capacity_shots = int(capacity_shots)
        self._entries: list[tuple[int, dict[str, torch.Tensor]]] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, shot_id: int, batch: online_adapter.JTextInterBatch) -> None:
        shot = int(shot_id)
        if self._entries and shot <= self._entries[-1][0]:
            raise RuntimeError("J-TEXT replay shots must be strictly increasing")
        values = {
            name: torch.as_tensor(value).detach().cpu().clone()
            for name, value in vars(batch).items()
        }
        self._entries.append((shot, values))
        self._entries = self._entries[-self.capacity_shots:]

    def sample(self, *, count: int) -> online_adapter.JTextInterBatch | None:
        if not self._entries:
            return None
        joined = {
            name: torch.cat([batch[name] for _shot, batch in self._entries], dim=0)
            for name in ("rebuilt", "source_logit", "available", "label")
        }
        indices = torch.arange(int(count), dtype=torch.long) % len(joined["label"])
        return online_adapter.JTextInterBatch(**{
            name: value.index_select(0, indices) for name, value in joined.items()
        })

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_shots": self.capacity_shots,
            "entries": [
                {"shot": shot, "batch": copy.deepcopy(batch)}
                for shot, batch in self._entries
            ],
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if int(payload.get("capacity_shots")) != self.capacity_shots:
            raise ValueError("J-TEXT replay resume capacity mismatch")
        self._entries = []
        for row in payload.get("entries", []):
            self._entries.append((
                int(row["shot"]),
                {
                    name: torch.as_tensor(value).detach().cpu().clone()
                    for name, value in row["batch"].items()
                },
            ))


def _fixed_indices(length: int, count: int) -> np.ndarray:
    if int(length) <= 0 or int(count) <= 0:
        raise ValueError("J-TEXT sampled windows require positive sizes")
    return np.asarray(
        [int(index) % int(length) for index in range(int(count))],
        dtype=np.int64,
    )


def _writable_array(value: np.ndarray) -> np.ndarray:
    """Detach immutable sealed signal views before exposing them to Torch."""

    return np.array(value, copy=True, order="C")


def _inter_batch(
    signal: jtext_io.JTextSignalShot,
    source_logits: np.ndarray,
    target: np.ndarray,
    *,
    count: int,
) -> online_adapter.JTextInterBatch:
    indices = _fixed_indices(len(source_logits), count)
    return online_adapter.JTextInterBatch(
        rebuilt=torch.from_numpy(_writable_array(signal.values[indices])),
        source_logit=torch.from_numpy(_writable_array(source_logits[indices])),
        available=torch.from_numpy(_writable_array(signal.availability[indices])),
        label=torch.from_numpy(_writable_array(target[indices])),
    )


def _query_components(
    signal: jtext_io.JTextSignalShot, risk: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    probabilities = np.asarray(risk, dtype=np.float64).reshape(-1)
    uncertainty = float(np.mean(1.0 - np.abs(2.0 * probabilities - 1.0)))
    risk_change = float(abs(probabilities[-1] - probabilities[0]))
    feature_change = float(np.mean(np.abs(
        np.asarray(signal.values[-1], dtype=np.float64)
        - np.asarray(signal.values[0], dtype=np.float64)
    )))
    normalized_feature_change = feature_change / (1.0 + feature_change)
    contradiction = float(np.clip(
        abs(risk_change - normalized_feature_change), 0.0, 1.0,
    ))
    phenotype = np.asarray([
        float(np.mean(probabilities)),
        float(np.max(probabilities)),
        float(probabilities[-1] - probabilities[0]),
        normalized_feature_change,
    ], dtype=np.float64)
    return uncertainty, contradiction, phenotype


def _static_predictions(
    ordered_shots: Sequence[int],
    *,
    signal_loader: Callable[[int], jtext_io.JTextSignalShot],
    source_trace_builder: Callable[[jtext_io.JTextSignalShot], np.ndarray],
) -> tuple[dict[int, jtext_io.JTextSignalShot], dict[int, np.ndarray], list[dict[str, Any]]]:
    signals: dict[int, jtext_io.JTextSignalShot] = {}
    source_logits: dict[int, np.ndarray] = {}
    rows = []
    for stream_index, shot in enumerate(ordered_shots):
        signal = signal_loader(int(shot))
        if int(signal.shot_id) != int(shot):
            raise ValueError("J-TEXT signal loader returned the wrong shot")
        logits = np.asarray(source_trace_builder(signal), dtype=np.float32).reshape(-1)
        if logits.shape != (len(signal.values),) or not np.all(np.isfinite(logits)):
            raise ValueError("J-TEXT source trace must be finite and endpoint-aligned")
        risk = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        row = {
            "stream_index": stream_index,
            "shot": int(shot),
            "source_logit_trace_sha256": _sha256_array(logits),
            "risk_trace_sha256": _sha256_array(risk),
            "prediction_sha256": _sha256_json({
                "stream_index": stream_index,
                "shot": int(shot),
                "source_logit_trace_sha256": _sha256_array(logits),
                "risk_trace_sha256": _sha256_array(risk),
            }),
            "claim_status": CLAIM_STATUS,
        }
        signals[int(shot)] = signal
        source_logits[int(shot)] = logits
        rows.append(row)
    return signals, source_logits, rows


def _static_manifest(
    rows: Sequence[Mapping[str, Any]], *, parent_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": STATIC_SEAL_SCHEMA,
        "claim_status": CLAIM_STATUS,
        "research_claim": RESEARCH_CLAIM,
        "shot_count": len(rows),
        "ordered_shots_sha256": _sha256_json([int(row["shot"]) for row in rows]),
        "predictions_sha256": _sha256_json(list(rows)),
        "source_parent_sha256": parent_sha256,
        "outcome_fields_present": False,
    }
    return {**payload, "seal_sha256": _sha256_json(payload)}


def _arm_policy(arm: str) -> dict[str, bool | str]:
    if arm not in ARMS:
        raise ValueError(f"unknown J-TEXT online arm: {arm}")
    return {
        "intra": arm in INTRA_ARMS,
        "inter": arm in INTER_ARMS,
        "active": arm in ACTIVE_ARMS,
        "random": arm in RANDOM_ARMS,
        "frontload": arm == "frontload_equal_labels",
        "query": arm in ACTIVE_ARMS or arm in RANDOM_ARMS or arm == "frontload_equal_labels",
    }


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [copy.deepcopy(dict(row)) for row in rows]


def _write_run_artifacts(
    output: Path,
    *,
    static_rows: Sequence[Mapping[str, Any]],
    static_manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    endpoint_receipts: Sequence[Mapping[str, Any]],
    query_receipts: Sequence[Mapping[str, Any]],
    update_receipts: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "static_predictions": output / "static_predictions.jsonl",
        "static_seal": output / "static_seal_manifest.json",
        "events": output / "online_events.jsonl",
        "endpoints": output / "endpoint_receipts.jsonl",
        "queries": output / "query_ledger.jsonl",
        "updates": output / "update_ledger.jsonl",
        "summary": output / "summary.json",
    }
    _write_jsonl(paths["static_predictions"], static_rows)
    _write_json(paths["static_seal"], static_manifest)
    _write_jsonl(paths["events"], events)
    _write_jsonl(paths["endpoints"], endpoint_receipts)
    _write_jsonl(paths["queries"], query_receipts)
    _write_jsonl(paths["updates"], update_receipts)
    _write_json(paths["summary"], summary)
    return {name: str(path.resolve()) for name, path in paths.items()}


def _resume_binding(
    *,
    ordered_shots: Sequence[int],
    static_manifest: Mapping[str, Any],
    arm: str,
    budget_percent: int,
    seed: int,
    query_block_size: int,
    query_quota: int,
) -> dict[str, Any]:
    return {
        "ordered_shots_sha256": _sha256_json([int(shot) for shot in ordered_shots]),
        "static_seal_sha256": static_manifest["seal_sha256"],
        "arm": arm,
        "budget_percent": int(budget_percent),
        "seed": int(seed),
        "query_block_size": int(query_block_size),
        "query_quota": int(query_quota),
        "claim_status": CLAIM_STATUS,
    }


def save_runner_resume(path: Path, state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != RUNNER_RESUME_SCHEMA:
        raise ValueError("J-TEXT runner resume schema mismatch")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, target)


def load_runner_resume(path: Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != RUNNER_RESUME_SCHEMA:
        raise ValueError("J-TEXT runner resume schema mismatch")
    return payload


def run_jtext_online_stream(
    *,
    ordered_shots: Sequence[int],
    signal_loader: Callable[[int], jtext_io.JTextSignalShot],
    source_trace_builder: Callable[[jtext_io.JTextSignalShot], np.ndarray],
    oracle_factory: Callable[[tuple[int, ...]], Any],
    controller: online_adapter.JTextOnlineController,
    arm: str,
    budget_percent: int,
    seed: int,
    output: Path,
    query_block_size: int = 20,
    query_quota: int | None = None,
    active_cutoff: float = 0.5,
    query_value: JTextQueryValue | None = None,
    compute_caps: JTextComputeCaps | None = None,
    stop_after: int | None = None,
    resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a sealed static pass and an ordered causal online replay on CPU."""

    policy = _arm_policy(arm)
    roster = _validate_complete_blocks(ordered_shots, query_block_size)
    quota = (
        budget_quota(budget_percent, block_size=query_block_size)
        if query_quota is None else int(query_quota)
    )
    if not 0 < quota <= int(query_block_size):
        raise ValueError("invalid J-TEXT online query quota")
    if stop_after is not None and int(stop_after) <= 0:
        raise ValueError("J-TEXT stop_after must be positive")
    value_model = query_value or JTextQueryValue()
    caps = compute_caps or JTextComputeCaps()
    parent_sha256 = controller.persistent_hash()
    source_hashes = controller.source_hashes

    # Phase A is entirely label-free and complete before oracle construction.
    signals, source_logits, static_rows = _static_predictions(
        roster, signal_loader=signal_loader,
        source_trace_builder=source_trace_builder,
    )
    static_manifest = _static_manifest(static_rows, parent_sha256=parent_sha256)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "static_predictions.jsonl", static_rows)
    _write_json(output / "static_seal_manifest.json", static_manifest)
    phase_receipts = [
        {
            "event": "static_seal",
            "seal_sha256": static_manifest["seal_sha256"],
            "shot_count": len(roster),
            "claim_status": CLAIM_STATUS,
        },
        {
            "event": "online_reset",
            "parent_sha256": parent_sha256,
            "source_hashes": source_hashes,
            "claim_status": CLAIM_STATUS,
        },
    ]
    if controller.persistent_hash() != parent_sha256:
        raise RuntimeError("J-TEXT online reset did not use the exact static parent")

    random_manifest = (
        build_random_query_manifest(
            roster, budget_percent=budget_percent, seed=seed,
            block_size=query_block_size, quota=quota,
        )
        if bool(policy["random"]) else None
    )
    selector = (
        JTextExactBlockSelector(
            cutoff=active_cutoff, block_size=query_block_size, quota=quota,
        )
        if bool(policy["active"]) else None
    )
    frontload_count = (
        len(roster) // query_block_size * quota if bool(policy["frontload"]) else 0
    )
    replay = _JTextReplayMemory()
    phenotype_memory = JTextRevealedPhenotypeMemory()
    events: list[dict[str, Any]] = []
    endpoint_receipts: list[dict[str, Any]] = []
    query_receipts: list[dict[str, Any]] = []
    update_receipts: list[dict[str, Any]] = []
    shot_receipts: list[dict[str, Any]] = []
    cursor = 0

    binding = _resume_binding(
        ordered_shots=roster,
        static_manifest=static_manifest,
        arm=arm,
        budget_percent=budget_percent,
        seed=seed,
        query_block_size=query_block_size,
        query_quota=quota,
    )
    if resume_state is not None:
        if resume_state.get("schema_version") != RUNNER_RESUME_SCHEMA:
            raise ValueError("J-TEXT runner resume schema mismatch")
        if dict(resume_state.get("binding", {})) != binding:
            raise ValueError("J-TEXT runner resume binding mismatch")
        cursor = int(resume_state["cursor"])
        if not 0 < cursor <= len(roster):
            raise ValueError("J-TEXT runner resume cursor is invalid")
        controller.load_resume_payload(resume_state["controller"])
        replay.load_state_dict(resume_state["replay"])
        phenotype_memory.load_state_dict(resume_state["phenotype_memory"])
        if selector is not None:
            selector.load_state_dict(resume_state["selector"])
        events = _copy_rows(resume_state["events"])
        endpoint_receipts = _copy_rows(resume_state["endpoint_receipts"])
        query_receipts = _copy_rows(resume_state["query_receipts"])
        update_receipts = _copy_rows(resume_state["update_receipts"])
        shot_receipts = _copy_rows(resume_state["shot_receipts"])

    oracle = oracle_factory(tuple(roster[cursor:]))
    event_sequence = 0 if not events else 1 + max(int(row["sequence"]) for row in events)
    endpoint_sequence = (
        0 if not endpoint_receipts else
        1 + max(int(row["sequence"]) for row in endpoint_receipts)
    )
    processed_this_call = 0

    for stream_index in range(cursor, len(roster)):
        shot = int(roster[stream_index])
        signal = signals[shot]
        logits = source_logits[shot]
        persistent_before = controller.persistent_hash()
        controller.begin_shot(shot)
        adjusted_trace: list[float] = []

        for endpoint in range(len(logits)):
            rebuilt = torch.from_numpy(_writable_array(signal.values[endpoint:endpoint + 1]))
            base_logit = torch.as_tensor([float(logits[endpoint])], dtype=torch.float32)
            available = torch.from_numpy(_writable_array(
                signal.availability[endpoint: endpoint + 1],
            ))
            adjusted = float(controller.predict(rebuilt, base_logit, available)[0].cpu())
            risk = float(torch.sigmoid(torch.tensor(adjusted)).item())
            prefix_sha = _sha256_json({
                "values_sha256": _sha256_array(signal.values[: endpoint + 1]),
                "availability_sha256": _sha256_array(signal.availability[: endpoint + 1]),
                "source_logits_sha256": _sha256_array(logits[: endpoint + 1]),
            })
            prediction_payload = {
                "shot": shot,
                "endpoint": endpoint,
                "source_logit": float(logits[endpoint]),
                "adjusted_logit": adjusted,
                "risk": risk,
                "prefix_sha256": prefix_sha,
            }
            prediction_sha = _sha256_json(prediction_payload)
            endpoint_receipts.append({
                "sequence": endpoint_sequence,
                "event": "endpoint_predict",
                **prediction_payload,
                "prediction_sha256": prediction_sha,
                "claim_status": CLAIM_STATUS,
            })
            endpoint_sequence += 1
            endpoint_receipts.append({
                "sequence": endpoint_sequence,
                "event": "endpoint_seal",
                "shot": shot,
                "endpoint": endpoint,
                "prediction_sha256": prediction_sha,
                "claim_status": CLAIM_STATUS,
            })
            endpoint_sequence += 1
            adjusted_trace.append(adjusted)

            if bool(policy["intra"]) and endpoint > 0:
                intra_batch = online_adapter.JTextIntraBatch(
                    source_rebuilt=torch.from_numpy(_writable_array(
                        signal.values[endpoint - 1: endpoint],
                    )),
                    source_logit=torch.as_tensor(
                        [float(logits[endpoint - 1])], dtype=torch.float32,
                    ),
                    source_available=torch.from_numpy(_writable_array(
                        signal.availability[endpoint - 1: endpoint],
                    )),
                    observed_rebuilt=torch.from_numpy(_writable_array(
                        signal.values[endpoint: endpoint + 1],
                    )),
                    observed_source_logit=torch.as_tensor(
                        [float(logits[endpoint])], dtype=torch.float32,
                    ),
                    observed_available=torch.from_numpy(_writable_array(
                        signal.availability[endpoint: endpoint + 1],
                    )),
                )
                intra_receipt = controller.intra_update(
                    intra_batch,
                    source_endpoint=endpoint - 1,
                    observed_endpoint=endpoint,
                    steps=1,
                )
                endpoint_receipts.append({
                    **intra_receipt,
                    "sequence": endpoint_sequence,
                    "claim_status": CLAIM_STATUS,
                })
                endpoint_sequence += 1

        adjusted_array = np.asarray(adjusted_trace, dtype=np.float32)
        risk_trace = 1.0 / (1.0 + np.exp(-adjusted_array.astype(np.float64)))
        shot_prediction_sha = _sha256_json({
            "shot": shot,
            "adjusted_logit_trace_sha256": _sha256_array(adjusted_array),
            "risk_trace_sha256": _sha256_array(risk_trace),
        })
        events.append({
            "sequence": event_sequence,
            "event": "predict",
            "shot": shot,
            "stream_index": stream_index,
            "prediction_sha256": shot_prediction_sha,
            "risk_trace_sha256": _sha256_array(risk_trace),
            "claim_status": CLAIM_STATUS,
        })
        event_sequence += 1
        controller.seal_shot(shot)
        oracle.seal_current(shot, prediction_sha256=shot_prediction_sha)
        events.append({
            "sequence": event_sequence,
            "event": "seal",
            "shot": shot,
            "prediction_sha256": shot_prediction_sha,
            "claim_status": CLAIM_STATUS,
        })
        event_sequence += 1

        uncertainty, contradiction, phenotype = _query_components(signal, risk_trace)
        value_receipt = value_model.evaluate(
            uncertainty=uncertainty,
            contradiction=contradiction,
            phenotype=phenotype,
            memory=phenotype_memory,
        )
        if selector is not None:
            decision = selector.decide(
                score=value_receipt["value"], stream_index=stream_index,
            )
        elif random_manifest is not None:
            decision = dict(random_manifest[stream_index])
            decision["position"] = stream_index % query_block_size
        elif bool(policy["frontload"]):
            selected = stream_index < frontload_count
            decision = {
                "selected": selected,
                "route": "frontload_prefix" if selected else None,
                "block": stream_index // query_block_size,
                "position": stream_index % query_block_size,
            }
        else:
            decision = {
                "selected": False,
                "route": None,
                "block": stream_index // query_block_size,
                "position": stream_index % query_block_size,
            }
        query_row = {
            "stream_index": stream_index,
            "shot": shot,
            "selected": bool(decision["selected"]),
            "route": decision.get("route"),
            "block": int(decision["block"]),
            "position": int(decision["position"]),
            "prediction_sha256": shot_prediction_sha,
            **value_receipt,
            "claim_status": CLAIM_STATUS,
        }
        query_sha = _sha256_json(query_row)
        query_row["query_sha256"] = query_sha
        query_receipts.append(query_row)
        events.append({
            "sequence": event_sequence,
            "event": "query",
            "shot": shot,
            "selected": bool(decision["selected"]),
            "query_sha256": query_sha,
            "claim_status": CLAIM_STATUS,
        })
        event_sequence += 1

        if bool(decision["selected"]):
            revealed = oracle.reveal_for_query(shot, selected=True)
            reveal_payload = {
                "shot": shot,
                "is_disrupt": bool(revealed.is_disrupt),
                "down_time_s": revealed.down_time_s,
                "advanced_time_ms": revealed.advanced_time_ms,
                "positive_horizon_ms": revealed.positive_horizon_ms,
            }
            events.append({
                "sequence": event_sequence,
                "event": "reveal",
                "shot": shot,
                "reveal_sha256": _sha256_json(reveal_payload),
                "claim_status": CLAIM_STATUS,
            })
            event_sequence += 1
            target = jtext_io.build_postseal_jtext_target(revealed, signal.times_s)
            current_batch = _inter_batch(
                signal, logits, target, count=caps.current_windows,
            )
            replay_batch = replay.sample(count=caps.replay_windows)
            guard_batch = _inter_batch(
                signal, logits, np.zeros_like(target), count=caps.guard_windows,
            )
            attempt_index = len(update_receipts)
            compute = realize_compute_ledger(
                [{
                    "selected": True,
                    "stream_index": stream_index,
                    "shot": shot,
                    "block": stream_index // query_block_size,
                }],
                caps=caps,
                online_parameter_count=controller.online_parameter_count,
            )[0]
            # Realized replay resources depend only on the attempt ordinal.
            compute["attempt_index"] = attempt_index
            compute["replay_size"] = replay.size
            compute["replay_windows"] = 0 if replay_batch is None else caps.replay_windows
            compute["flop_inputs"]["replay_windows"] = compute["replay_windows"]
            compute["reported_flops"] = int(
                2 * controller.online_parameter_count
                * (caps.current_windows + compute["replay_windows"] + caps.guard_windows)
                * caps.optimizer_attempts
            )
            if bool(policy["inter"]):
                update = controller.inter_update(
                    shot,
                    current=current_batch,
                    replay=replay_batch,
                    guard_batch=guard_batch,
                    steps=caps.optimizer_attempts,
                )
            else:
                update = controller.finalize_revealed_without_update(shot)
                update.update({
                    "accepted": True,
                    "rollback_reason": None,
                    "steps_attempted": 0,
                    "steps_committed": 0,
                    "losses": {},
                })
            update_row = {
                **update,
                "stream_index": stream_index,
                "compute": compute,
                "claim_status": CLAIM_STATUS,
            }
            update_receipts.append(update_row)
            events.append({
                "sequence": event_sequence,
                "event": "update",
                "shot": shot,
                "accepted": bool(update_row.get("accepted", True)),
                "persistent_sha256": update_row["persistent_sha256"],
                "claim_status": CLAIM_STATUS,
            })
            event_sequence += 1
            replay.add(shot, current_batch)
            phenotype_memory.add_after_reveal(shot, phenotype)
        else:
            oracle.discard_unqueried(shot)
            discard = controller.discard_unqueried(shot)
            events.append({
                "sequence": event_sequence,
                "event": "discard",
                "shot": shot,
                "persistent_sha256": discard["persistent_sha256"],
                "claim_status": CLAIM_STATUS,
            })
            event_sequence += 1

        persistent_after = controller.persistent_hash()
        shot_receipts.append({
            "stream_index": stream_index,
            "shot": shot,
            "persistent_before_sha256": persistent_before,
            "persistent_after_sha256": persistent_after,
            "selected": bool(decision["selected"]),
            "claim_status": CLAIM_STATUS,
        })
        cursor = stream_index + 1
        processed_this_call += 1
        if stop_after is not None and processed_this_call >= int(stop_after):
            break

    controller.assert_source_frozen()
    completed = cursor == len(roster)
    evaluation_mask = (
        [index >= frontload_count for index in range(len(roster))]
        if bool(policy["frontload"]) else [True] * len(roster)
    )
    summary = {
        "schema_version": RUN_SCHEMA,
        "claim_status": CLAIM_STATUS,
        "research_claim": RESEARCH_CLAIM,
        "arm": arm,
        "budget_percent": int(budget_percent),
        "seed": int(seed),
        "shot_count": len(roster),
        "processed_shots": cursor,
        "completed": completed,
        "query_count": sum(bool(row["selected"]) for row in query_receipts),
        "update_attempts": sum(int(row["steps_attempted"]) for row in update_receipts),
        "accepted_updates": sum(bool(row.get("accepted")) for row in update_receipts),
        "rollbacks": sum(not bool(row.get("accepted")) for row in update_receipts),
        "online_parameter_count": controller.online_parameter_count,
        "static_parent_sha256": parent_sha256,
        "final_online_state_sha256": controller.persistent_hash(),
        "source_hashes": source_hashes,
        "static_seal_sha256": static_manifest["seal_sha256"],
        "east_transfer_parent": "raw_static_jtext_source_checkpoint_only",
        "jtext_online_state_transfer_allowed": False,
        "evaluation_mask_sha256": _sha256_json(evaluation_mask),
        "frontload_suffix_start_index": frontload_count if bool(policy["frontload"]) else None,
    }
    resume_payload = {
        "schema_version": RUNNER_RESUME_SCHEMA,
        "binding": binding,
        "cursor": cursor,
        "controller": controller.resume_payload(extra_state={"cursor": cursor}),
        "selector": selector.state_dict() if selector is not None else None,
        "replay": replay.state_dict(),
        "phenotype_memory": phenotype_memory.state_dict(),
        "events": _copy_rows(events),
        "endpoint_receipts": _copy_rows(endpoint_receipts),
        "query_receipts": _copy_rows(query_receipts),
        "update_receipts": _copy_rows(update_receipts),
        "shot_receipts": _copy_rows(shot_receipts),
    }
    artifact_paths = _write_run_artifacts(
        output,
        static_rows=static_rows,
        static_manifest=static_manifest,
        events=events,
        endpoint_receipts=endpoint_receipts,
        query_receipts=query_receipts,
        update_receipts=update_receipts,
        summary=summary,
    )
    save_runner_resume(output / "resume.pt", resume_payload)
    return {
        "phase_receipts": phase_receipts,
        "events": events,
        "endpoint_receipts": endpoint_receipts,
        "query_receipts": query_receipts,
        "update_receipts": update_receipts,
        "shot_receipts": shot_receipts,
        "summary": summary,
        "static_manifest": static_manifest,
        "resume_state": resume_payload,
        "artifact_paths": artifact_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sealed J-TEXT TEST220 two-timescale online diagnostics",
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--budget", type=int, choices=(10, 20, 30), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    return parser


def _selected_source_trace_builder(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
) -> Callable[[jtext_io.JTextSignalShot], np.ndarray]:
    def build(signal: jtext_io.JTextSignalShot) -> np.ndarray:
        rows = [signal.causal_window(endpoint, window_points=128) for endpoint in range(len(signal.values))]
        windows = torch.from_numpy(np.stack([row[0] for row in rows])).to("cpu")
        valid_lengths = torch.as_tensor([row[1] for row in rows], dtype=torch.long)
        encoder.eval()
        head.eval()
        with torch.no_grad():
            logits = head(encoder(windows, valid_lengths=valid_lengths))
        return logits.detach().cpu().numpy().astype(np.float32, copy=False)
    return build


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import selected_mainline_contract as selected

    encoder, head, source_payload = selected.load_selected_source_checkpoint(
        args.source_checkpoint, args.seed,
    )
    ordered = jtext_io.load_ordered_test220(ROOT)
    center = source_payload["scaler_center"].detach().cpu().numpy()
    scale = source_payload["scaler_scale"].detach().cpu().numpy()

    def load_scaled_signal(shot_id: int) -> jtext_io.JTextSignalShot:
        raw = jtext_io.load_jtext_signal_only(args.data_root, shot_id)
        values = np.clip((raw.values - center) / scale, -10.0, 10.0)
        values = np.where(raw.availability, values, 0.0).astype(np.float32)
        return jtext_io.JTextSignalShot(
            shot_id=raw.shot_id,
            values=values,
            availability=raw.availability,
            times_s=raw.times_s,
            sample_rate_hz=raw.sample_rate_hz,
            start_time_s=raw.start_time_s,
        )

    residual = online_adapter.JTextOnlineResidual(
        feature_count=len(jtext_io.REBUILD_FEATURE_NAMES),
        hidden_size=32,
        max_logit_shift=0.5,
    ).to("cpu")
    controller = online_adapter.JTextOnlineController(
        residual,
        source_encoder=encoder,
        source_head=head,
        intra_lr=1e-4,
        inter_lr=5e-4,
    )

    def oracle_factory(remaining: tuple[int, ...]):
        return jtext_io.JTextLazyOutcomeOracle(
            data_root=args.data_root,
            ordered_roster=remaining,
            advanced_time_path=selected.JTEXT_ADVANCED_TIME,
        )

    resume = load_runner_resume(args.resume) if args.resume is not None else None
    run_jtext_online_stream(
        ordered_shots=ordered.shots,
        signal_loader=load_scaled_signal,
        source_trace_builder=_selected_source_trace_builder(encoder, head),
        oracle_factory=oracle_factory,
        controller=controller,
        arm=args.arm,
        budget_percent=args.budget,
        seed=args.seed,
        output=args.output,
        stop_after=args.stop_after,
        resume_state=resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "CLAIM_STATUS",
    "RESEARCH_CLAIM",
    "JTextQueryValue",
    "JTextRevealedPhenotypeMemory",
    "JTextExactBlockSelector",
    "JTextComputeCaps",
    "budget_quota",
    "build_random_query_manifest",
    "build_active_query_manifest",
    "realize_compute_ledger",
    "audit_matched_controls",
    "build_frontload_plan",
    "abnormal_run_gate",
    "run_jtext_online_stream",
    "save_runner_resume",
    "load_runner_resume",
    "build_parser",
]
