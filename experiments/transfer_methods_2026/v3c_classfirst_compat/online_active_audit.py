#!/usr/bin/env python3
"""Causal A1 query audit primitives for the ordered EAST 360-shot stream.

This module intentionally cannot update the disruption predictor.  It computes
causal physics innovations, a shadow query score, and an exact block budget.
Outcome access is fail-closed until a query decision has already been sealed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np


RETAINED_NAMES = (
    "Bt", "ip_A", "Bt_over_Ip", "Ip_diff", "v_loop", "Z_proxy",
    "sxr_kurt", "sxr_skew", "sxr_var", "xuv_kurt", "xuv_skew", "xuv_var",
    "Mir_avg_fre", "n=1 amplitude", "mode_number_n", "dx_a", "dy_a",
)


class PhysicsInnovations(NamedTuple):
    metrics: dict[str, np.ndarray]
    available: dict[str, np.ndarray]


def _lagged_delta(values: np.ndarray, valid: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    previous = np.maximum(np.arange(n) - lag, 0)
    ok = valid & valid[previous]
    out = np.zeros(n, dtype=np.float64)
    out[ok] = values[ok] - values[previous[ok]]
    return out, ok


def causal_innovations(
    values: np.ndarray,
    available: np.ndarray,
    *,
    sample_rate_hz: float,
    lag: int = 8,
) -> PhysicsInnovations:
    """Build device-local innovations using only each row and its past.

    Values must use ``RETAINED_NAMES`` column order and remain in their rebuilt
    physical units.  The function never computes a whole-shot center or scale.
    """
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    if values.ndim != 2 or values.shape != available.shape:
        raise ValueError("values and available must be equal [time, channel] arrays")
    if values.shape[1] != len(RETAINED_NAMES):
        raise ValueError(f"expected {len(RETAINED_NAMES)} retained channels")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if int(lag) <= 0:
        raise ValueError("lag must be positive")
    lag = int(lag)
    idx = {name: i for i, name in enumerate(RETAINED_NAMES)}
    metrics: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    def add(metric: str, series: np.ndarray, valid: np.ndarray, *, per_second: bool = False) -> None:
        delta, ok = _lagged_delta(np.asarray(series, float), np.asarray(valid, bool), lag)
        if per_second:
            delta *= float(sample_rate_hz) / lag
        metrics[metric] = np.abs(delta)
        masks[metric] = ok

    ip = values[:, idx["ip_A"]]
    ip_ok = available[:, idx["ip_A"]] & np.isfinite(ip) & (np.abs(ip) > 1e-9)
    log_ip = np.zeros(len(values), float)
    log_ip[ip_ok] = np.log(np.abs(ip[ip_ok]) + 1e-9)
    add("electrical_dlog_ip", log_ip, ip_ok)
    add("electrical_dip_dt", ip, ip_ok, per_second=True)

    bt = values[:, idx["Bt"]]
    bt_ok = available[:, idx["Bt"]] & np.isfinite(bt)
    bt_ip = np.zeros(len(values), float)
    bt_ip[ip_ok & bt_ok] = bt[ip_ok & bt_ok] / ip[ip_ok & bt_ok]
    add("electrical_d_bt_over_ip", bt_ip, ip_ok & bt_ok)

    vloop = values[:, idx["v_loop"]]
    v_ok = available[:, idx["v_loop"]] & np.isfinite(vloop)
    add("electrical_d_vloop", vloop, v_ok)
    z = np.zeros(len(values), float)
    z[ip_ok & v_ok] = vloop[ip_ok & v_ok] / ip[ip_ok & v_ok]
    add("electrical_d_zproxy", z, ip_ok & v_ok)

    for name in ("dx_a", "dy_a"):
        x = values[:, idx[name]]
        ok = available[:, idx[name]] & np.isfinite(x)
        add(f"position_d_{name}", x, ok)

    for name in ("sxr_kurt", "sxr_skew", "sxr_var", "xuv_kurt", "xuv_skew", "xuv_var"):
        x = values[:, idx[name]]
        ok = available[:, idx[name]] & np.isfinite(x)
        add(f"radiation_d_{name}", x, ok)

    for name in ("Mir_avg_fre", "n=1 amplitude", "mode_number_n"):
        x = values[:, idx[name]]
        ok = available[:, idx[name]] & np.isfinite(x)
        # These channels use an all-zero trace as the missing-data sentinel in
        # the rebuilt EAST files.  Determine observability causally: a future
        # non-zero sample must not retroactively make an earlier zero valid.
        observed_nonzero = ok & ~np.isclose(x, 0.0)
        ok = ok & (np.cumsum(observed_nonzero.astype(np.int64)) > 0)
        add(f"mhd_d_{name}", x, ok)

    return PhysicsInnovations(metrics=metrics, available=masks)


def fit_innovation_scales(
    traces: Iterable[PhysicsInnovations], *, quantile: float = 0.90
) -> dict[str, float]:
    """Fit label-free innovation scales from a pre-registered support cohort."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie in (0, 1)")
    pools: dict[str, list[np.ndarray]] = {}
    for trace in traces:
        for name, values in trace.metrics.items():
            valid = trace.available[name] & np.isfinite(values)
            if np.any(valid):
                pools.setdefault(name, []).append(np.asarray(values[valid], float))
    scales = {}
    for name, chunks in pools.items():
        joined = np.concatenate(chunks)
        positive = joined[joined > 0]
        basis = positive if positive.size else joined
        scales[name] = max(float(np.quantile(basis, quantile)), 1e-12)
    return scales


def score_physics_families(
    trace: PhysicsInnovations, scales: Mapping[str, float]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Normalize metrics and average them within explicit physics families."""
    grouped: dict[str, list[np.ndarray]] = {}
    grouped_ok: dict[str, list[np.ndarray]] = {}
    for name, values in trace.metrics.items():
        family = name.split("_", 1)[0]
        scale = float(scales.get(name, 0.0))
        ok = trace.available[name] & np.isfinite(values) & (scale > 0)
        normalized = np.zeros_like(values, dtype=np.float64)
        normalized[ok] = np.clip(np.abs(values[ok]) / scale, 0.0, 1.0)
        grouped.setdefault(family, []).append(normalized)
        grouped_ok.setdefault(family, []).append(ok)
    family_scores: dict[str, np.ndarray] = {}
    family_available: dict[str, np.ndarray] = {}
    for family, parts in grouped.items():
        data = np.stack(parts)
        ok = np.stack(grouped_ok[family])
        count = ok.sum(axis=0)
        family_available[family] = count > 0
        family_scores[family] = np.divide(
            (data * ok).sum(axis=0), count, out=np.zeros(data.shape[1]), where=count > 0
        )
    return family_scores, family_available


def combine_family_scores(
    family_scores: Mapping[str, np.ndarray],
    family_available: Mapping[str, np.ndarray],
    *,
    activity_threshold: float = 0.5,
    min_families: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine energy and cross-family concurrence without treating missing as quiet."""
    families = sorted(family_scores)
    if not families:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, int)
    scores = np.stack([np.asarray(family_scores[f], float) for f in families])
    avail = np.stack([np.asarray(family_available[f], bool) for f in families])
    count = avail.sum(axis=0)
    energy = np.divide((scores * avail).sum(axis=0), count, out=np.zeros(scores.shape[1]), where=count > 0)
    active = ((scores >= activity_threshold) & avail).sum(axis=0)
    concurrence = np.divide(active, count, out=np.zeros(scores.shape[1]), where=count > 0)
    known = count >= int(min_families)
    support = np.where(known, energy * concurrence, 0.0)
    return support, known, count


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1 - 1e-6))
    return math.log(value / (1 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def query_components(
    *,
    risk: np.ndarray,
    physics_support: np.ndarray,
    physics_known: np.ndarray,
    threshold: float,
    persistence_samples: int = 128,
) -> dict[str, float]:
    """Compute shadow FP/FN contradictions; never returns an alarm override."""
    risk = np.asarray(risk, float).reshape(-1)
    support = np.asarray(physics_support, float).reshape(-1)
    known = np.asarray(physics_known, bool).reshape(-1)
    if not len(risk) or len(support) != len(risk) or len(known) != len(risk):
        raise ValueError("risk, physics_support, and physics_known must be equal non-empty traces")
    peak = float(np.max(risk))
    highness = 1.0 / (1.0 + math.exp(-(_logit(peak) - _logit(threshold))))
    high = risk >= threshold
    persistence = min(_longest_run(high) / max(int(persistence_samples), 1), 1.0)
    supported_high = high & known
    if np.any(supported_high):
        support_at_risk = float(np.median(support[supported_high]))
    elif known[int(np.argmax(risk))]:
        support_at_risk = float(support[int(np.argmax(risk))])
    else:
        support_at_risk = 0.0
    physics_max = float(np.max(support[known])) if np.any(known) else 0.0
    fp = highness * persistence * (1.0 - support_at_risk) if np.any(known) else 0.0
    fn = (1.0 - highness) * physics_max if np.any(known) else 0.0
    uncertainty = math.exp(-abs(_logit(peak) - _logit(threshold)))
    return {
        "max_risk": peak,
        "highness": highness,
        "persistence": persistence,
        "physics_at_risk": support_at_risk,
        "physics_max": physics_max,
        "physics_known_fraction": float(np.mean(known)),
        "fp_contradiction": fp,
        "fn_contradiction": fn,
        "uncertainty": uncertainty,
        "query_score": max(fp, fn, uncertainty),
    }


class BlockBudgetSelector:
    """Immediate causal decisions with an exact per-block quota via forced fill."""

    def __init__(self, *, cutoff: float, block_size: int = 20, quota: int = 4):
        if block_size <= 0 or quota < 0 or quota > block_size:
            raise ValueError("require 0 <= quota <= block_size")
        self.cutoff = float(cutoff)
        self.block_size = int(block_size)
        self.quota = int(quota)
        self._block = -1
        self._used = 0
        self.total_selected = 0

    def decide(self, *, score: float, stream_index: int) -> bool:
        if stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        block = int(stream_index) // self.block_size
        position = int(stream_index) % self.block_size
        if block != self._block:
            self._block, self._used = block, 0
        remaining = self.block_size - position
        slots = self.quota - self._used
        selected = slots > 0 and (float(score) >= self.cutoff or remaining <= slots)
        if selected:
            self._used += 1
            self.total_selected += 1
        return bool(selected)


class CausalLedgerQuerySelector:
    """Consume an unlabeled per-shot score ledger strictly in stream order.

    The ledger may be precomputed from sealed causal traces, but the selector
    exposes only the current row.  It refuses label-bearing columns and cannot
    skip ahead, so a runner cannot use the final query roster as an input.
    """

    _FORBIDDEN_KEYS = frozenset({
        "is_disrupt", "label", "outcome", "revealed_is_disrupt",
    })

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        cutoff: float,
        block_size: int = 20,
        quota: int = 4,
    ):
        self._rows = [dict(row) for row in rows]
        if not self._rows:
            raise ValueError("causal query ledger is empty")
        for expected_index, row in enumerate(self._rows):
            forbidden = self._FORBIDDEN_KEYS.intersection(row)
            if forbidden:
                raise ValueError(f"label-bearing query ledger columns are forbidden: {sorted(forbidden)}")
            if int(row.get("stream_index", -1)) != expected_index:
                raise ValueError("query ledger stream indices must be contiguous")
            if "shot" not in row or "query_score" not in row:
                raise ValueError("query ledger requires shot and query_score")
            if not np.isfinite(float(row["query_score"])):
                raise ValueError("query ledger score must be finite")
        shots = [int(row["shot"]) for row in self._rows]
        if shots != sorted(shots) or len(shots) != len(set(shots)):
            raise ValueError("query ledger shots must be unique and increasing")
        self._selector = BlockBudgetSelector(
            cutoff=cutoff, block_size=block_size, quota=quota
        )
        self._cursor = 0
        self.selected_shots: list[int] = []

    @property
    def consumed(self) -> int:
        return self._cursor

    @property
    def total_selected(self) -> int:
        return self._selector.total_selected

    def decide(self, *, shot: int, stream_index: int) -> bool:
        if self._cursor >= len(self._rows):
            raise RuntimeError("causal query ledger is exhausted")
        row = self._rows[self._cursor]
        if int(row["stream_index"]) != int(stream_index) or int(row["shot"]) != int(shot):
            raise RuntimeError("request does not match the next causal ledger row")
        selected = self._selector.decide(
            score=float(row["query_score"]), stream_index=int(stream_index)
        )
        self._cursor += 1
        if selected:
            self.selected_shots.append(int(shot))
        return selected


class StratifiedBlockBudgetSelector:
    """Causal exact block budget with protected base and auxiliary slots."""

    def __init__(
        self, *, block_size: int, base_quota: int, auxiliary_quota: int,
        base_cutoff: float, auxiliary_cutoff: float,
    ):
        if block_size <= 0 or min(base_quota, auxiliary_quota) < 0:
            raise ValueError("invalid stratified budget")
        if base_quota + auxiliary_quota > block_size:
            raise ValueError("combined quota exceeds block size")
        self.block_size = int(block_size)
        self.base_quota = int(base_quota)
        self.auxiliary_quota = int(auxiliary_quota)
        self.base_cutoff = float(base_cutoff)
        self.auxiliary_cutoff = float(auxiliary_cutoff)
        self._block = -1
        self._base_used = 0
        self._auxiliary_used = 0
        self.total_selected = 0

    def decide(
        self, *, base_score: float, auxiliary_score: float, stream_index: int
    ) -> str | None:
        if stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        block = int(stream_index) // self.block_size
        position = int(stream_index) % self.block_size
        if block != self._block:
            self._block = block
            self._base_used = 0
            self._auxiliary_used = 0
        base_slots = self.base_quota - self._base_used
        auxiliary_slots = self.auxiliary_quota - self._auxiliary_used
        reason = None
        if base_slots > 0 and float(base_score) >= self.base_cutoff:
            reason = "base"
        elif auxiliary_slots > 0 and float(auxiliary_score) >= self.auxiliary_cutoff:
            reason = "auxiliary"
        else:
            remaining = self.block_size - position
            if remaining <= base_slots + auxiliary_slots:
                reason = "base_forced" if base_slots > 0 else "auxiliary_forced"
        if reason is not None:
            if reason.startswith("base"):
                self._base_used += 1
            else:
                self._auxiliary_used += 1
            self.total_selected += 1
        return reason


class OutcomeVault:
    """Fail-closed outcome access for a simulated prospective stream."""

    def __init__(self, labels: Mapping[int, bool]):
        self._labels = {int(k): bool(v) for k, v in labels.items()}
        self._evaluation_open = False

    def reveal_for_query(self, shot: int, *, selected: bool) -> bool:
        if not selected:
            raise PermissionError("outcome is unavailable before a positive query decision")
        return self._labels[int(shot)]

    def open_final_evaluation(self) -> None:
        self._evaluation_open = True

    def all_labels(self) -> dict[int, bool]:
        if not self._evaluation_open:
            raise PermissionError("full labels are sealed until final evaluation")
        return dict(self._labels)


class ShotOutcomeVault:
    """Fail-closed joint storage for outcome and disruption time metadata."""

    def __init__(self, outcomes: Mapping[int, Mapping[str, Any]]):
        self._outcomes: dict[int, dict[str, Any]] = {}
        for shot, row in outcomes.items():
            if "is_disrupt" not in row or "down_time_s" not in row:
                raise ValueError("each sealed outcome requires is_disrupt and down_time_s")
            is_disrupt = bool(row["is_disrupt"])
            down_time = row["down_time_s"]
            if is_disrupt and (down_time is None or not np.isfinite(float(down_time))):
                raise ValueError("a disruptive shot requires a finite down_time_s")
            self._outcomes[int(shot)] = {
                "is_disrupt": is_disrupt,
                "down_time_s": None if down_time is None else float(down_time),
            }
        self._evaluation_open = False

    def reveal_for_query(self, shot: int, *, selected: bool) -> dict[str, Any]:
        if not selected:
            raise PermissionError("outcome metadata is unavailable before a positive query decision")
        return dict(self._outcomes[int(shot)])

    def open_final_evaluation(self) -> None:
        self._evaluation_open = True

    def all_outcomes(self) -> dict[int, dict[str, Any]]:
        if not self._evaluation_open:
            raise PermissionError("full outcome metadata is sealed until final evaluation")
        return {shot: dict(row) for shot, row in self._outcomes.items()}


class LazyShotOutcomeOracle:
    """Read outcome metadata lazily only after the current shot is sealed."""

    def __init__(
        self, shots: Sequence[int], reader: Callable[[int], Mapping[str, Any]],
        *, allow_nonmonotonic: bool = False,
    ):
        self._shots = [int(shot) for shot in shots]
        if len(self._shots) != len(set(self._shots)):
            raise ValueError("oracle roster must be unique")
        if not allow_nonmonotonic and self._shots != sorted(self._shots):
            raise ValueError("oracle roster must be unique and increasing")
        self.allow_nonmonotonic = bool(allow_nonmonotonic)
        self._reader = reader
        self._cursor = 0
        self._pending: int | None = None
        self._evaluation_open = False

    def seal_current(self, shot: int) -> None:
        if self._pending is not None:
            raise RuntimeError("previous sealed shot has not been finalized")
        if self._cursor >= len(self._shots) or int(shot) != self._shots[self._cursor]:
            raise RuntimeError("shot does not match the next oracle roster entry")
        self._pending = int(shot)

    def _finalize(self, shot: int) -> None:
        if self._pending != int(shot):
            raise RuntimeError("outcome request does not match the sealed current shot")
        self._pending = None
        self._cursor += 1

    def reveal_for_query(self, shot: int, *, selected: bool) -> dict[str, Any]:
        if not selected:
            raise PermissionError("lazy outcome metadata requires a positive query decision")
        if self._pending != int(shot):
            raise RuntimeError("outcome request does not match the sealed current shot")
        row = dict(self._reader(int(shot)))
        if "is_disrupt" not in row or "down_time_s" not in row:
            raise ValueError("oracle reader must return is_disrupt and down_time_s")
        result = {
            "is_disrupt": bool(row["is_disrupt"]),
            "down_time_s": None if row["down_time_s"] is None else float(row["down_time_s"]),
        }
        self._finalize(shot)
        return result

    def discard_unqueried(self, shot: int) -> None:
        self._finalize(shot)

    def open_final_evaluation(self) -> None:
        if self._pending is not None or self._cursor != len(self._shots):
            raise RuntimeError("the complete stream must finish before final evaluation")
        self._evaluation_open = True

    def all_outcomes(self) -> dict[int, dict[str, Any]]:
        if not self._evaluation_open:
            raise PermissionError("full outcomes are unavailable before final evaluation")
        return {
            shot: {
                "is_disrupt": bool((row := dict(self._reader(shot)))["is_disrupt"]),
                "down_time_s": (
                    None if row["down_time_s"] is None else float(row["down_time_s"])
                ),
            }
            for shot in self._shots
        }


class SafeLocalResidual:
    """Bounded negative-only residual supported by independent confirmed FPs.

    Contexts are expected to have been transformed using support-only
    statistics.  A local region is active only when two distinct FP shots are
    within ``radius`` and at least one entered through an exploration query.
    """

    def __init__(self, *, threshold: float, radius: float, max_logit_drop: float):
        if not 0 < threshold < 1 or radius <= 0 or max_logit_drop <= 0:
            raise ValueError("invalid residual configuration")
        self.threshold = float(threshold)
        self.radius = float(radius)
        self.max_logit_drop = float(max_logit_drop)
        self._prototypes: list[dict[str, Any]] = []

    @property
    def prototype_shots(self) -> list[int]:
        return [int(item["shot"]) for item in self._prototypes]

    def local_evidence(self, context: np.ndarray) -> dict[str, Any]:
        context = np.asarray(context, dtype=np.float64).reshape(-1)
        local = []
        for item in self._prototypes:
            prototype = np.asarray(item["context"], dtype=np.float64)
            distance = float(np.linalg.norm(context - prototype))
            if distance <= self.radius:
                local.append((int(item["shot"]), distance, bool(item["exploration"])))
        return {
            "local_prototype_count": len(local),
            "local_has_exploration": any(value[2] for value in local),
            "local_prototype_shots": [value[0] for value in local],
            "local_min_distance": min((value[1] for value in local), default=None),
        }

    def _correct_with(self, base_score: float, context: np.ndarray) -> float:
        base_score = float(base_score)
        if base_score < self.threshold or len(self._prototypes) < 2:
            return base_score
        context = np.asarray(context, dtype=np.float64).reshape(-1)
        candidates = []
        for item in self._prototypes:
            prototype = np.asarray(item["context"], dtype=np.float64)
            if prototype.shape != context.shape:
                raise ValueError("context shape mismatch")
            distance = float(np.linalg.norm(context - prototype))
            if distance <= self.radius:
                candidates.append((item, math.exp(-0.5 * (distance / self.radius) ** 2)))
        if len(candidates) < 2 or not any(bool(item["exploration"]) for item, _ in candidates):
            return base_score
        weight = sum(value for _, value in candidates)
        drop = sum(float(item["logit_drop"]) * value for item, value in candidates) / weight
        drop = float(np.clip(drop, 0.0, self.max_logit_drop))
        return _sigmoid(_logit(base_score) - drop)

    def correct(self, base_score: float, context: np.ndarray) -> float:
        return self._correct_with(base_score, context)

    def add_confirmed_fp(
        self,
        shot: int,
        context: np.ndarray,
        base_score: float,
        *,
        exploration: bool,
        protected_tp: Sequence[tuple[float, np.ndarray]],
    ) -> bool:
        if int(shot) in self.prototype_shots:
            raise ValueError(f"duplicate prototype shot {shot}")
        target = self.threshold * 0.95
        desired_drop = max(0.0, _logit(float(base_score)) - _logit(target))
        candidate = {
            "shot": int(shot),
            "context": np.asarray(context, dtype=np.float64).reshape(-1).copy(),
            "logit_drop": min(desired_drop, self.max_logit_drop),
            "exploration": bool(exploration),
        }
        self._prototypes.append(candidate)
        for score, anchor_context in protected_tp:
            if float(score) >= self.threshold and self.correct(float(score), anchor_context) < self.threshold:
                self._prototypes.pop()
                return False
        return True

    def rollback_active(self, context: np.ndarray) -> list[int]:
        """Remove prototypes whose local region covered a newly revealed TP."""
        context = np.asarray(context, dtype=np.float64).reshape(-1)
        removed = [
            int(item["shot"]) for item in self._prototypes
            if np.linalg.norm(context - np.asarray(item["context"], dtype=np.float64)) <= self.radius
        ]
        self._prototypes = [item for item in self._prototypes if int(item["shot"]) not in removed]
        return removed


def run_prequential_audit(
    *,
    shots: Sequence[Mapping[str, Any]],
    predict: Callable[[Mapping[str, Any]], np.ndarray],
    summarize: Callable[[Mapping[str, Any], np.ndarray], Mapping[str, float]],
    selector: BlockBudgetSelector,
    outcomes: OutcomeVault | LazyShotOutcomeOracle,
    on_seal: Callable[[dict[str, Any]], None] | None = None,
    on_reveal: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run A1 in shot order; this function has no model-update callback by design."""
    ordered = sorted(shots, key=lambda item: int(item["shot"]))
    if len({int(item["shot"]) for item in ordered}) != len(ordered):
        raise ValueError("duplicate shot IDs in prequential stream")
    records = []
    for index, shot in enumerate(ordered):
        shot_id = int(shot["shot"])
        risk = np.asarray(predict(shot), dtype=np.float32).reshape(-1)
        if not len(risk) or not np.all(np.isfinite(risk)):
            raise ValueError(f"invalid risk trace for shot {shot_id}")
        summary = dict(summarize(shot, risk))
        if "query_score" not in summary:
            raise ValueError("summarize must return query_score")
        record: dict[str, Any] = {
            "stream_index": index,
            "shot": shot_id,
            "base_max_risk": float(np.max(risk)),
            **summary,
            "queried": False,
        }
        if on_seal is not None:
            on_seal(dict(record))
        if isinstance(outcomes, LazyShotOutcomeOracle):
            outcomes.seal_current(shot_id)
        selected = selector.decide(score=float(summary["query_score"]), stream_index=index)
        record["queried"] = selected
        if selected:
            revealed = outcomes.reveal_for_query(shot_id, selected=True)
            record["revealed_is_disrupt"] = (
                bool(revealed["is_disrupt"])
                if isinstance(revealed, Mapping) else bool(revealed)
            )
            if on_reveal is not None:
                on_reveal(dict(record))
        elif isinstance(outcomes, LazyShotOutcomeOracle):
            outcomes.discard_unqueried(shot_id)
        records.append(record)
    return records


def confusion_and_query_capture(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[int, bool],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate the immutable base alarm and how strongly A1 queried errors."""
    groups = {name: 0 for name in ("TP", "FP", "FN", "TN")}
    queried = {name: 0 for name in groups}
    for record in records:
        shot = int(record["shot"])
        actual = bool(labels[shot])
        predicted = float(record["base_max_risk"]) >= float(threshold)
        group = "TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"
        groups[group] += 1
        if bool(record.get("queried", False)):
            queried[group] += 1
    n = sum(groups.values())
    n_query = sum(queried.values())
    n_error = groups["FP"] + groups["FN"]
    queried_error = queried["FP"] + queried["FN"]
    stream_error_rate = n_error / n if n else 0.0
    query_error_precision = queried_error / n_query if n_query else 0.0
    enrichment = query_error_precision / stream_error_rate if stream_error_rate else 0.0
    return {
        "confusion": groups,
        "queried_by_group": queried,
        "query_count": n_query,
        "query_error_precision": float(query_error_precision),
        "stream_error_rate": float(stream_error_rate),
        "error_enrichment_vs_stream": float(enrichment),
    }


__all__ = [
    "RETAINED_NAMES", "PhysicsInnovations", "causal_innovations",
    "fit_innovation_scales", "score_physics_families", "combine_family_scores",
    "query_components", "BlockBudgetSelector", "CausalLedgerQuerySelector",
    "OutcomeVault", "ShotOutcomeVault", "LazyShotOutcomeOracle",
    "run_prequential_audit",
    "confusion_and_query_capture",
    "SafeLocalResidual",
    "StratifiedBlockBudgetSelector",
]
