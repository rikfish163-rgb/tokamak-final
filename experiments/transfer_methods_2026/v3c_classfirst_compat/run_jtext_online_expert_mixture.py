#!/usr/bin/env python3
"""Delayed-feedback active Hedge mixture of frozen J-TEXT Transformer experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import online_evidence as evidence  # noqa: E402


SEEDS = (131, 133, 137)
CLAIM_STATUS = "TEST_DRIVEN_EXPLORATION"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return math.log(value / (1.0 - value))


def _softmax_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = np.asarray(log_weights, dtype=np.float64) - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def load_expert_traces(
    cache_paths: Mapping[int, Path], checkpoints: Mapping[int, Path],
) -> tuple[list[int], dict[int, dict[int, np.ndarray]], dict[int, float], dict[str, str]]:
    traces: dict[int, dict[int, np.ndarray]] = {}
    thresholds: dict[int, float] = {}
    hashes: dict[str, str] = {}
    roster: list[int] | None = None
    for seed in SEEDS:
        records = evidence.read_jsonl(cache_paths[seed] / "endpoint_receipts.jsonl")
        grouped: dict[int, list[float]] = {}
        for row in records:
            if row.get("event") == "endpoint_predict":
                grouped.setdefault(int(row["shot"]), []).append(float(row["source_logit"]))
        current = list(grouped)
        if roster is None:
            roster = current
        if current != roster:
            raise RuntimeError("J-TEXT expert cache rosters differ")
        traces[seed] = {
            shot: np.asarray(values, dtype=np.float64) for shot, values in grouped.items()
        }
        thresholds[seed] = float(evidence.load_threshold(
            checkpoints[seed], seed,
        )["value"])
        hashes[f"cache_seed{seed}"] = _sha256(cache_paths[seed] / "endpoint_receipts.jsonl")
        hashes[f"checkpoint_seed{seed}"] = _sha256(checkpoints[seed])
    assert roster is not None
    for shot in roster:
        lengths = {len(traces[seed][shot]) for seed in SEEDS}
        if len(lengths) != 1:
            raise RuntimeError(f"J-TEXT expert endpoint counts differ: {shot}")
    return roster, traces, thresholds, hashes


def _mixture_trace(
    margins: np.ndarray, persistent: np.ndarray, *, intra_eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    fast_log = np.log(np.asarray(persistent, dtype=np.float64))
    output = np.empty(margins.shape[1], dtype=np.float64)
    previous = margins[:, 0].copy()
    for endpoint in range(margins.shape[1]):
        current = margins[:, endpoint]
        if endpoint:
            median = float(np.median(current))
            unsupported_positive = np.maximum(current - median, 0.0)
            disagreement = np.abs(current - median)
            abrupt = np.abs(current - previous)
            loss = unsupported_positive + 0.10 * disagreement + 0.05 * abrupt
            fast_log -= float(intra_eta) * loss
        weights = _softmax_log_weights(fast_log)
        output[endpoint] = float(weights @ current)
        previous = current
    return output, _softmax_log_weights(fast_log)


def _query_score(margins: np.ndarray, mixture: np.ndarray) -> tuple[float, dict[str, float]]:
    expert_scores = np.max(margins, axis=1)
    mixed_score = float(np.max(mixture))
    uncertainty = float(math.exp(-abs(mixed_score)))
    disagreement = float(np.std(expert_scores) / (1.0 + np.std(expert_scores)))
    temporal_conflict = float(np.max(mixture) - np.median(mixture[-min(20, len(mixture)):]))
    score = 0.45 * uncertainty + 0.40 * disagreement + 0.15 * np.clip(
        temporal_conflict, 0.0, 1.0,
    )
    return float(score), {
        "uncertainty": uncertainty,
        "expert_disagreement": disagreement,
        "temporal_conflict": temporal_conflict,
    }


def _expert_feedback_loss(
    margins: np.ndarray, *, disruptive: bool,
) -> np.ndarray:
    # The per-shot maximum is outcome supervision.  The upper-tail mean keeps
    # the update sensitive to temporal support rather than one isolated point.
    top_count = min(10, margins.shape[1])
    top = np.partition(margins, -top_count, axis=1)[:, -top_count:]
    score = 0.5 * np.max(margins, axis=1) + 0.5 * np.mean(top, axis=1)
    target = 1.0 if disruptive else 0.0
    return np.logaddexp(0.0, score) - target * score


def _select_candidates(
    candidates: Sequence[tuple[float, int, np.ndarray]], *, quota: int,
    policy: str, rng: np.random.Generator,
) -> list[tuple[float, int, np.ndarray]]:
    if policy == "active":
        return sorted(candidates, key=lambda row: (-row[0], row[1]))[:quota]
    if policy == "random":
        chosen = set(int(index) for index in rng.choice(
            len(candidates), size=quota, replace=False,
        ))
        return [row for index, row in enumerate(candidates) if index in chosen]
    raise ValueError("query policy must be active or random")


def _rows_from_traces(
    roster: Sequence[int], risk: Mapping[int, np.ndarray], *,
    truth: Mapping[int, Mapping[str, Any]], timing: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for shot in roster:
        values = np.asarray(risk[int(shot)], dtype=np.float64)
        passing = np.flatnonzero(values >= 0.5)
        endpoint = int(passing[0]) if len(passing) else None
        rows.append({
            "shot": int(shot), **dict(truth[int(shot)]),
            "score": float(np.max(values)), "alarmed": endpoint is not None,
            "alarm_time_s": (
                None if endpoint is None else
                float(timing[int(shot)]["start_time_s"])
                + endpoint / float(timing[int(shot)]["sample_rate_hz"])
            ),
        })
    return rows


def run(
    *, cache_paths: Mapping[int, Path], checkpoints: Mapping[int, Path],
    data_root: Path, output: Path, budget_percent: int = 20,
    block_size: int = 20, intra_eta: float = 0.05, inter_eta: float = 0.50,
    query_policy: str = "active", query_seed: int = 131,
) -> dict[str, Any]:
    roster, traces, thresholds, hashes = load_expert_traces(cache_paths, checkpoints)
    quota = int(block_size * budget_percent / 100)
    if len(roster) % block_size or quota <= 0:
        raise ValueError("expert-mixture budget must be exact in complete blocks")
    if query_policy not in {"active", "random"}:
        raise ValueError("query_policy must be active or random")
    rng = np.random.default_rng(query_seed)
    threshold_logits = np.asarray([_logit(thresholds[seed]) for seed in SEEDS])
    store = evidence.TruthStore(data_root)
    # Timing comes from signal attributes and carries no outcome.  HDF5 truth is
    # deliberately absent until a complete block has been predicted and sealed.
    timing = {int(shot): store.timing("JTEXT", int(shot)) for shot in roster}
    static_risk: dict[int, np.ndarray] = {}
    online_risk: dict[int, np.ndarray] = {}
    persistent_log = np.zeros(len(SEEDS), dtype=np.float64)
    queries: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for start in range(0, len(roster), block_size):
        block = roster[start:start + block_size]
        persistent = _softmax_log_weights(persistent_log)
        candidates: list[tuple[float, int, np.ndarray]] = []
        weights_before = persistent.tolist()
        for shot in block:
            raw = np.stack([traces[seed][shot] for seed in SEEDS])
            margins = raw - threshold_logits[:, None]
            static_margin = np.mean(margins, axis=0)
            online_margin, fast_final = _mixture_trace(
                margins, persistent, intra_eta=intra_eta,
            )
            static_risk[shot] = 1.0 / (1.0 + np.exp(-static_margin))
            online_risk[shot] = 1.0 / (1.0 + np.exp(-online_margin))
            value, components = _query_score(margins, online_margin)
            candidates.append((value, shot, margins))
            queries.append({
                "stream_index": roster.index(shot), "shot": shot,
                "block": start // block_size, "selected": False,
                "value": value, **components,
                "persistent_weights": persistent.tolist(),
                "fast_weights_final": fast_final.tolist(),
                "outcome_accessed_for_score": False,
            })
        selected = _select_candidates(
            candidates, quota=quota, policy=query_policy, rng=rng,
        )
        selected_shots = {shot for _value, shot, _margins in selected}
        feedback_losses = []
        for _value, shot, margins in selected:
            # This is the first label access for the complete already-sealed block.
            revealed = store.truth("JTEXT", shot)
            feedback_losses.append(_expert_feedback_loss(
                margins, disruptive=bool(revealed["is_disrupt"]),
            ))
        mean_loss = np.mean(np.stack(feedback_losses), axis=0)
        persistent_log -= float(inter_eta) * mean_loss
        for row in queries[-block_size:]:
            row["selected"] = int(row["shot"]) in selected_shots
        blocks.append({
            "block": start // block_size, "shot_start": int(block[0]),
            "shot_end": int(block[-1]), "selected_shots": sorted(selected_shots),
            "weights_before": weights_before,
            "feedback_loss": mean_loss.tolist(),
            "weights_after": _softmax_log_weights(persistent_log).tolist(),
            "labels_accessed_after_block_seal": True,
        })
    # Full-cohort truth is opened only after every online prediction is frozen.
    truth = {int(shot): store.truth("JTEXT", int(shot)) for shot in roster}
    static_rows = _rows_from_traces(
        roster, static_risk, truth=truth, timing=timing,
    )
    online_rows = _rows_from_traces(
        roster, online_risk, truth=truth, timing=timing,
    )
    result = {
        "schema_version": "jtext-online-expert-mixture-v1",
        "claim_status": CLAIM_STATUS,
        "method": "delayed-feedback-active-hedge-with-intra-shot-fast-weights",
        "seeds": list(SEEDS), "shot_count": len(roster),
        "budget_percent": budget_percent, "query_count": sum(row["selected"] for row in queries),
        "block_size": block_size, "queries_per_block": quota,
        "intra_eta": intra_eta, "inter_eta": inter_eta,
        "query_policy": query_policy, "query_seed": query_seed,
        "static": evidence.compute_metrics(static_rows, device="JTEXT"),
        "online": evidence.compute_metrics(online_rows, device="JTEXT"),
        "delta": evidence.compute_delta(online_rows, static_rows, device="JTEXT"),
        "thresholds": thresholds, "hashes": hashes,
        "query_outcome_access": "only selected shots after complete block prediction seal",
        "jtext_online_state_transfer_allowed": False,
        "east_transfer_parent": "raw_static_jtext_source_checkpoint_only",
        "blocks": blocks,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "query_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in queries),
        encoding="utf-8",
    )
    (output / "predictions.json").write_text(
        json.dumps({"static": static_rows, "online": online_rows}, indent=2,
                   sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for seed in SEEDS:
        parser.add_argument(f"--cache-{seed}", type=Path, required=True)
        parser.add_argument(f"--checkpoint-{seed}", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--intra-eta", type=float, default=0.05)
    parser.add_argument("--inter-eta", type=float, default=0.50)
    parser.add_argument("--query-policy", choices=("active", "random"), default="active")
    parser.add_argument("--query-seed", type=int, default=131)
    args = parser.parse_args()
    result = run(
        cache_paths={seed: getattr(args, f"cache_{seed}") for seed in SEEDS},
        checkpoints={seed: getattr(args, f"checkpoint_{seed}") for seed in SEEDS},
        data_root=args.data_root, output=args.output, budget_percent=args.budget,
        intra_eta=args.intra_eta, inter_eta=args.inter_eta,
        query_policy=args.query_policy, query_seed=args.query_seed,
    )
    print(json.dumps({
        "static": result["static"], "online": result["online"],
        "delta": result["delta"], "query_count": result["query_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
