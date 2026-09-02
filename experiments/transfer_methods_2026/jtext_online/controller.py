"""Delayed-feedback active-Hedge controller and J-TEXT metrics.

Three frozen source checkpoints act as experts.  During a TEST220 replay the
controller may change only their convex mixture weights.  The prediction
trace, query score and selected-shot update are all separated so the active
gain cannot be mistaken for a change to the Transformer representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .io import OrderedOutcomeOracle, RevealedOutcome, SignalShot


SEEDS = (131, 133, 137)
CLAIM_STATUS = "TEST_DRIVEN_EXPLORATION"


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return math.log(value / (1.0 - value))


def _softmax_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def mixture_trace(
    margins: np.ndarray, persistent_log_weights: np.ndarray, *, intra_eta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix expert margins with a label-free optional fast weighting.

    The caller supplies normalized persistent probabilities despite the legacy
    parameter name.  Converting them to log space here is important: passing
    probabilities directly to a second softmax changes the registered replay
    after the first feedback block.
    """

    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(SEEDS) or not np.isfinite(values).all():
        raise ValueError("expert margins must be finite with shape [3, time]")
    persistent = np.asarray(persistent_log_weights, dtype=np.float64)
    if persistent.shape != (len(SEEDS),) or not np.isfinite(persistent).all():
        raise ValueError("persistent expert weights have an invalid shape")
    if np.any(persistent <= 0.0):
        raise ValueError("persistent expert weights must be positive")
    fast_log = np.log(persistent)
    output = np.empty(values.shape[1], dtype=np.float64)
    previous = values[:, 0].copy()
    for endpoint in range(values.shape[1]):
        current = values[:, endpoint]
        if endpoint and float(intra_eta) != 0.0:
            median = float(np.median(current))
            unsupported_positive = np.maximum(current - median, 0.0)
            disagreement = np.abs(current - median)
            abrupt = np.abs(current - previous)
            fast_loss = unsupported_positive + 0.10 * disagreement + 0.05 * abrupt
            fast_log -= float(intra_eta) * fast_loss
        output[endpoint] = float(_softmax_log_weights(fast_log) @ current)
        previous = current
    return output, _softmax_log_weights(fast_log)


def query_score(margins: np.ndarray, mixture: np.ndarray) -> tuple[float, dict[str, float]]:
    """Causal query value using only expert disagreement and trace geometry."""

    values = np.asarray(margins, dtype=np.float64)
    trace = np.asarray(mixture, dtype=np.float64)
    if values.ndim != 2 or trace.ndim != 1 or values.shape[1] != len(trace):
        raise ValueError("query inputs are not aligned")
    expert_scores = np.max(values, axis=1)
    mixed_score = float(np.max(trace))
    uncertainty = float(math.exp(-abs(mixed_score)))
    disagreement = float(np.std(expert_scores) / (1.0 + np.std(expert_scores)))
    recent = trace[-min(20, len(trace)) :]
    temporal_conflict = float(np.max(trace) - np.median(recent))
    score = 0.45 * uncertainty + 0.40 * disagreement + 0.15 * np.clip(
        temporal_conflict, 0.0, 1.0,
    )
    return float(score), {
        "uncertainty": uncertainty,
        "expert_disagreement": disagreement,
        "temporal_conflict": temporal_conflict,
    }


def feedback_loss(margins: np.ndarray, *, disruptive: bool) -> np.ndarray:
    """Per-expert delayed outcome loss used by the Hedge update."""

    values = np.asarray(margins, dtype=np.float64)
    top_count = min(10, values.shape[1])
    top = np.partition(values, -top_count, axis=1)[:, -top_count:]
    aggregate = 0.5 * np.max(values, axis=1) + 0.5 * np.mean(top, axis=1)
    target = 1.0 if bool(disruptive) else 0.0
    return np.logaddexp(0.0, aggregate) - target * aggregate


def select_candidates(
    candidates: Sequence[tuple[float, int, np.ndarray]], *, quota: int,
    policy: str, rng: np.random.Generator,
) -> list[tuple[float, int, np.ndarray]]:
    if quota <= 0 or quota > len(candidates):
        raise ValueError("query quota is outside the candidate block")
    if policy == "active":
        return sorted(candidates, key=lambda row: (-row[0], row[1]))[:quota]
    if policy == "random":
        indices = rng.choice(len(candidates), size=quota, replace=False)
        selected = {int(index) for index in indices}
        return [row for index, row in enumerate(candidates) if index in selected]
    raise ValueError("query policy must be active or random")


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError("AUC inputs are invalid")
    positive = int(y.sum())
    negative = int((~y).sum())
    if not positive or not negative:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    cursor = 0
    while cursor < len(p):
        end = cursor + 1
        while end < len(p) and p[order[end]] == p[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _lead_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median_ms": None, "q25_ms": None, "q75_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(array), "median_ms": float(np.median(array)),
        "q25_ms": float(np.quantile(array, 0.25)),
        "q75_ms": float(np.quantile(array, 0.75)),
    }


def rows_from_risks(
    roster: Sequence[int], risks: Mapping[int, np.ndarray],
    outcomes: Mapping[int, RevealedOutcome], signals: Mapping[int, SignalShot],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shot in roster:
        values = np.asarray(risks[int(shot)], dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"invalid J-TEXT risk trace for shot {shot}")
        passing = np.flatnonzero(values >= 0.5)
        alarm_index = int(passing[0]) if len(passing) else None
        signal = signals[int(shot)]
        alarm_time = (
            float(signal.times_s[alarm_index]) if alarm_index is not None else None
        )
        outcome = outcomes[int(shot)]
        rows.append({
            "shot": int(shot), "is_disrupt": bool(outcome.is_disrupt),
            "down_time_s": outcome.down_time_s,
            "score": float(np.max(values)), "alarmed": alarm_index is not None,
            "alarm_trace_index": alarm_index, "alarm_time_s": alarm_time,
        })
    return rows


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("metrics require non-empty rows")
    disruptions = [row for row in rows if bool(row["is_disrupt"])]
    safe = [row for row in rows if not bool(row["is_disrupt"])]
    if not disruptions or not safe:
        raise ValueError("metrics require both disruptive and safe shots")
    operational_tp = 0
    raw_tp = 0
    operational_leads: list[float] = []
    raw_leads: list[float] = []
    for row in disruptions:
        if not bool(row["alarmed"]):
            continue
        raw_tp += 1
        lead = (float(row["down_time_s"]) - float(row["alarm_time_s"])) * 1000.0
        raw_leads.append(lead)
        if lead > 10.0:
            operational_tp += 1
            operational_leads.append(lead)
    fp = sum(bool(row["alarmed"]) for row in safe)
    counts = {
        "disruptions": len(disruptions), "non_disruptions": len(safe),
        "raw_tp": raw_tp, "raw_fn": len(disruptions) - raw_tp,
        "tardy": raw_tp - operational_tp, "operational_tp": operational_tp,
        "operational_fn": len(disruptions) - operational_tp,
        "fp": fp, "tn": len(safe) - fp,
    }
    return {
        "n_shots": len(rows), "counts": counts,
        "raw_tpr": raw_tp / len(disruptions),
        "operational_tpr": operational_tp / len(disruptions),
        "fpr": fp / len(safe),
        "j_operational": operational_tp / len(disruptions) - fp / len(safe),
        "auc": _auc(
            [bool(row["is_disrupt"]) for row in rows],
            [float(row["score"]) for row in rows],
        ),
        "minimum_lead_rule": "lead_ms > 10",
        "raw_alarm_lead_ms": _lead_summary(raw_leads),
        "operational_tp_lead_ms": _lead_summary(operational_leads),
        "very_early_alarm": {
            "count": sum(lead > 3000.0 for lead in raw_leads),
            "denominator": len(disruptions), "rule": "lead_ms > 3000",
        },
    }


def run_ordered_stream(
    *, roster: Sequence[int], expert_traces: Mapping[int, Mapping[int, np.ndarray]],
    thresholds: Mapping[int, float], signals: Mapping[int, SignalShot],
    oracle: OrderedOutcomeOracle, budget_percent: int = 20, block_size: int = 20,
    inter_eta: float = 0.50, intra_eta: float = 0.0,
    query_policy: str = "active", query_seed: int = 131,
) -> dict[str, Any]:
    roster = tuple(int(shot) for shot in roster)
    if tuple(sorted(roster)) != roster or len(roster) != 220:
        raise ValueError("J-TEXT TEST220 must be sorted and contain 220 shots")
    if block_size <= 0 or len(roster) % block_size:
        raise ValueError("TEST220 must divide into complete query blocks")
    quota_float = block_size * float(budget_percent) / 100.0
    if not quota_float.is_integer() or not 0 < quota_float <= block_size:
        raise ValueError("budget must produce an integer non-zero block quota")
    quota = int(quota_float)
    for seed in SEEDS:
        if seed not in expert_traces or seed not in thresholds:
            raise ValueError(f"missing frozen J-TEXT expert seed {seed}")
    rng = np.random.default_rng(int(query_seed))
    threshold_logits = np.asarray([_logit(thresholds[seed]) for seed in SEEDS])
    persistent_log = np.zeros(len(SEEDS), dtype=np.float64)
    static_risk: dict[int, np.ndarray] = {}
    online_risk: dict[int, np.ndarray] = {}
    query_ledger: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []

    for start in range(0, len(roster), block_size):
        block = roster[start : start + block_size]
        oracle.begin_block(block)
        persistent = _softmax_log_weights(persistent_log)
        candidates: list[tuple[float, int, np.ndarray]] = []
        block_rows: list[dict[str, Any]] = []
        for shot in block:
            raw = np.stack([
                np.asarray(expert_traces[seed][shot], dtype=np.float64)
                for seed in SEEDS
            ])
            if len({len(row) for row in raw}) != 1:
                raise ValueError(f"expert trace lengths differ for J-TEXT shot {shot}")
            margins = raw - threshold_logits[:, None]
            static_margin = np.mean(margins, axis=0)
            online_margin, fast_weights = mixture_trace(
                margins, persistent, intra_eta=intra_eta,
            )
            static = 1.0 / (1.0 + np.exp(-static_margin))
            online = 1.0 / (1.0 + np.exp(-online_margin))
            static_risk[shot] = static.astype(np.float64)
            online_risk[shot] = online.astype(np.float64)
            value, components = query_score(margins, online_margin)
            candidates.append((value, shot, margins))
            seal = sha256_array(online.astype(np.float32))
            oracle.seal_prediction(shot, seal)
            block_rows.append({
                "stream_index": start + block.index(shot), "shot": shot,
                "block": start // block_size, "selected": False,
                "query_value": value, **components,
                "persistent_weights": persistent.tolist(),
                "fast_weights_final": fast_weights.tolist(),
                "outcome_accessed_for_score": False,
                "prediction_seal_sha256": seal,
            })
        oracle.finish_prediction_block()
        selected = select_candidates(
            candidates, quota=quota, policy=query_policy, rng=rng,
        )
        selected_shots = {shot for _value, shot, _margins in selected}
        feedback = []
        for _value, shot, margins in selected:
            outcome = oracle.reveal_for_query(shot, selected=True)
            feedback.append(feedback_loss(margins, disruptive=outcome.is_disrupt))
        mean_loss = np.mean(np.stack(feedback), axis=0)
        persistent_log -= float(inter_eta) * mean_loss
        for row in block_rows:
            row["selected"] = row["shot"] in selected_shots
        query_ledger.extend(block_rows)
        blocks.append({
            "block": start // block_size, "shot_start": block[0],
            "shot_end": block[-1], "selected_shots": sorted(selected_shots),
            "queries_in_block": quota, "weights_before": persistent.tolist(),
            "feedback_loss": mean_loss.tolist(),
            "weights_after": _softmax_log_weights(persistent_log).tolist(),
            "labels_accessed_after_prediction_seal": True,
        })
        oracle.complete_block(selected_shots)

    # Capture the causal-time read count before the separate post-stream
    # evaluation pass opens the remaining outcomes.
    stream_outcome_reads = oracle.outcome_reads
    outcomes = oracle.read_all_after_stream()
    static_rows = rows_from_risks(roster, static_risk, outcomes, signals)
    online_rows = rows_from_risks(roster, online_risk, outcomes, signals)
    return {
        "schema_version": "jtext-adv25-online-active-v1",
        "claim_status": CLAIM_STATUS,
        "method": "delayed-feedback-active-hedge-inter-shot",
        "representation_updated": False,
        "seeds": list(SEEDS), "shot_count": len(roster),
        "budget_percent": int(budget_percent), "block_size": int(block_size),
        "queries_per_block": quota,
        "query_count": sum(int(row["selected"]) for row in query_ledger),
        "query_policy": query_policy, "query_seed": int(query_seed),
        "intra_eta": float(intra_eta), "inter_eta": float(inter_eta),
        "static": compute_metrics(static_rows),
        "online": compute_metrics(online_rows),
        "query_outcome_access": "selected shots only after complete block seal",
        "outcome_reads_during_stream": stream_outcome_reads,
        "blocks": blocks,
        "predictions": {"static": static_rows, "online": online_rows},
        "query_ledger": query_ledger,
    }
