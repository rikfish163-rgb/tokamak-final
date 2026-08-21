#!/usr/bin/env python3
"""Audit and summarize the fixed-compute selector-roster treatment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import h5py
import numpy as np


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    project = next(
        (parent for parent in (resolved, *resolved.parents)
         if (parent / ".git").exists()),
        None,
    )
    if project is None:
        raise RuntimeError(f"cannot locate project root for {resolved}")
    return f"<PROJECT_ROOT>/{resolved.relative_to(project).as_posix()}"


def utility(metrics: Mapping[str, Any]) -> float:
    return float(metrics["tpr_at_30ms"]) - float(metrics["fpr"])


def roster_screen_adjudication(deltas: list[float]) -> dict[str, Any]:
    passed = all(float(value) > 0.0 for value in deltas)
    if passed:
        text = (
            "The archived active roster passes the post-development, declared "
            "all-seed screening rule under this fixed-compute assimilation "
            "operator. This does not estimate the adaptive selector-policy "
            "effect and is not confirmatory evidence."
        )
    else:
        text = (
            "The archived active roster fails the post-development, declared "
            "all-seed screening rule under this fixed-compute assimilation "
            "operator. Archived-roster superiority is not supported under this "
            "operator; this does not estimate the adaptive selector-policy "
            "effect and does not establish equivalence or a universal null."
        )
    return {"passed": passed, "adjudication": text}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_east_truth(data_root: Path, shots: list[int]) -> dict[int, dict[str, Any]]:
    truth: dict[int, dict[str, Any]] = {}
    for shot in sorted(set(int(value) for value in shots)):
        path = (
            data_root / "EAST/unified_hdf5" / str(shot // 100 * 100)
            / f"{shot}.hdf5"
        )
        with h5py.File(path, "r") as handle:
            disruptive = bool(
                np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0]
            )
            down = float(
                np.asarray(handle["meta/DownTime"][()]).reshape(-1)[0]
            )
        truth[shot] = {
            "is_disrupt": disruptive,
            "down_time_s": down if disruptive else None,
        }
    return truth


def recompute_alarm_metrics(
    records: list[dict[str, Any]], truth: Mapping[int, Mapping[str, Any]],
    *, minimum_lead_ms: float = 30.0,
) -> dict[str, Any]:
    if len(records) != 360 or len({int(row["shot"]) for row in records}) != 360:
        raise RuntimeError("equal-compute prediction ledger must contain 360 unique shots")
    disruptive = [row for row in records if truth[int(row["shot"])]["is_disrupt"]]
    safe = [row for row in records if not truth[int(row["shot"])]["is_disrupt"]]

    def lead(row: Mapping[str, Any]) -> float | None:
        if not bool(row["alarmed"]) or row.get("alarm_time_s") is None:
            return None
        down_time = truth[int(row["shot"])]["down_time_s"]
        return (float(down_time) - float(row["alarm_time_s"])) * 1000.0

    raw_hits = [bool(row["alarmed"]) for row in disruptive]
    false_alarms = [bool(row["alarmed"]) for row in safe]
    leads = [value for row in disruptive if (value := lead(row)) is not None]
    timely = []
    for row in disruptive:
        lead_ms = lead(row)
        timely.append(
            bool(row["alarmed"])
            and lead_ms is not None
            and lead_ms >= float(minimum_lead_ms) - 1e-9
        )
    return {
        "TP": int(sum(raw_hits)), "FN": int(len(raw_hits) - sum(raw_hits)),
        "FP": int(sum(false_alarms)),
        "TN": int(len(false_alarms) - sum(false_alarms)),
        "tpr": float(np.mean(raw_hits)),
        "tpr_at_30ms": float(np.mean(timely)),
        "fpr": float(np.mean(false_alarms)),
        "median_lead_ms": float(np.median(leads)),
        "over_3000ms_count": int(sum(value > 3000.0 for value in leads)),
    }


def verify_recomputed_metrics(
    path: Path, summary_metrics: Mapping[str, Any], *, data_root: Path,
    predictions_name: str,
) -> dict[str, Any]:
    predictions_path = path / predictions_name
    rows = load_jsonl(predictions_path)
    truth = load_east_truth(data_root, [int(row["shot"]) for row in rows])
    recomputed = recompute_alarm_metrics(rows, truth)
    mismatches = {}
    for name, observed in recomputed.items():
        expected = summary_metrics[name]
        equal = (
            math.isclose(float(observed), float(expected), abs_tol=1e-9)
            if isinstance(observed, float)
            else int(observed) == int(expected)
        )
        if not equal:
            mismatches[name] = {"recomputed": observed, "summary": expected}
    if mismatches:
        raise RuntimeError(f"{path}: HDF5 metric recomputation mismatch {mismatches}")
    truth_rows = [
        [shot, bool(truth[shot]["is_disrupt"]), truth[shot]["down_time_s"]]
        for shot in sorted(truth)
    ]
    return {
        "status": "PASS",
        "predictions_file": predictions_name,
        "predictions_sha256": sha256_file(predictions_path),
        "hdf5_truth_tuple_sha256": canonical_hash(truth_rows),
        "truth_contract": "HDF5 /meta/IsDisrupt and /meta/DownTime",
        "minimum_lead_ms": 30.0,
        "metrics": recomputed,
    }


def signature_core(signature: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: signature[name] for name in (
            "current_windows", "replay_windows", "guard_windows", "steps",
            "objective", "current_batch", "replay_batch", "guard_batch",
            "trainable_scope", "optimizer_steps_attempted",
        )
    }


def compact_run(path: Path, *, data_root: Path) -> dict[str, Any]:
    summary = load(path / "summary.json")
    control = summary.get("selector_compute_control", {})
    if not control.get("enabled") or not all(
        control.get("realized_contract", {}).values()
    ):
        raise RuntimeError(f"{path}: fixed-compute contract is not satisfied")
    receipts = [
        row for row in load_jsonl(path / "updates.jsonl")
        if row.get("event") == "unified_revealed_finalize"
    ]
    if len(receipts) != int(summary["query_count"]):
        raise RuntimeError(f"{path}: one reveal receipt per query is required")
    shape_stream: list[dict[str, Any]] = []
    causal_effect_stream: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    shared_committed = private_committed = 0
    for ordinal, row in enumerate(receipts, start=1):
        shared = row["shared"]["fixed_compute_signature"]
        private = row["private"]["fixed_compute_signature"]
        fixed = row.get("fixed_compute")
        if shared is None or private is None:
            raise RuntimeError(f"{path}: missing fixed-compute receipt")
        if not isinstance(fixed, Mapping):
            raise RuntimeError(f"{path}: missing fixed-compute causal receipt")
        indices = [int(value) for value in fixed["current_endpoint_indices"]]
        if int(fixed["query_ordinal_one_based"]) != ordinal:
            raise RuntimeError(f"{path}: non-monotone fixed-compute query ordinal")
        if len(indices) != len(set(indices)) or len(indices) != int(
            fixed["current_window_count"]
        ):
            raise RuntimeError(f"{path}: invalid fixed-compute endpoint indices")
        if not bool(fixed["current_shot_prediction_precedes_update"]):
            raise RuntimeError(f"{path}: current prediction/update order is unsafe")
        shape_stream.append({
            "shared": signature_core(shared),
            "private": signature_core(private),
        })
        causal_effect_stream.append({
            "shot": int(row["shot"]),
            "query_ordinal_one_based": ordinal,
            "current_window_count": len(indices),
            "support_replay_window_count": int(
                fixed["support_replay_window_count"]
            ),
            "first_effect_stream_index_zero_based": fixed[
                "first_effect_stream_index_zero_based"
            ],
            "first_effect_shot": fixed["first_effect_shot"],
        })
        group = str(row["group"])
        group_counts[group] = group_counts.get(group, 0) + 1
        shared_committed += int(shared["optimizer_steps_committed"])
        private_committed += int(private["optimizer_steps_committed"])
    if len({canonical_hash(row) for row in shape_stream}) != 1:
        raise RuntimeError(f"{path}: per-query compute shapes are not constant")
    metrics = dict(summary["results"]["online_unified"])
    static_metrics = dict(summary["results"]["static_unified"])
    metric_recomputation = {
        "online": verify_recomputed_metrics(
            path, metrics, data_root=data_root,
            predictions_name="predictions.jsonl",
        ),
        "static": verify_recomputed_metrics(
            path, static_metrics, data_root=data_root,
            predictions_name="static_predictions.jsonl",
        ),
    }
    return {
        "path": portable_path(path),
        "metrics": {**metrics, "utility_tpr30_minus_fpr": utility(metrics)},
        "static_metrics": {
            **static_metrics,
            "utility_tpr30_minus_fpr": utility(static_metrics),
        },
        "metric_recomputation": metric_recomputation,
        "query_count": int(summary["query_count"]),
        "counts": dict(summary["counts"]),
        "group_counts": group_counts,
        "compute_shape_stream_sha256": canonical_hash(shape_stream),
        "compute_signature_sha256": canonical_hash(shape_stream[0]),
        "shared_steps_committed": shared_committed,
        "private_steps_committed": private_committed,
        "causal_effect_stream_sha256": canonical_hash(causal_effect_stream),
        "runtime": dict(summary.get("runtime", {})),
        "parent_hashes": {
            name: summary["hashes"][name] for name in (
                "base_before_after", "phase2_initial_shared",
                "phase2_initial_private", "static_predictions",
            )
        },
    }


def build(
    root: Path, *, seeds: list[int], replicates: int, data_root: Path,
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    deltas: list[float] = []
    active_static_deltas: list[float] = []
    for seed in seeds:
        active = compact_run(
            root / f"seed{seed}/active_fixed_compute", data_root=data_root,
        )
        randoms = [
            compact_run(
                root / f"seed{seed}/random_fixed_compute_r{replicate}",
                data_root=data_root,
            )
            for replicate in range(replicates)
        ]
        for random in randoms:
            if random["query_count"] != active["query_count"]:
                raise RuntimeError(f"seed {seed}: query budget mismatch")
            if random["parent_hashes"] != active["parent_hashes"]:
                raise RuntimeError(f"seed {seed}: parent/static hash mismatch")
            if (
                random["compute_shape_stream_sha256"]
                != active["compute_shape_stream_sha256"]
            ):
                raise RuntimeError(f"seed {seed}: realized compute shape mismatch")
        random_mean = {
            name: mean(float(row["metrics"][name]) for row in randoms)
            for name in ("tpr_at_30ms", "fpr", "utility_tpr30_minus_fpr")
        }
        delta = (
            float(active["metrics"]["utility_tpr30_minus_fpr"])
            - random_mean["utility_tpr30_minus_fpr"]
        )
        deltas.append(delta)
        active_static_delta = (
            float(active["metrics"]["utility_tpr30_minus_fpr"])
            - float(active["static_metrics"]["utility_tpr30_minus_fpr"])
        )
        active_static_deltas.append(active_static_delta)
        per_seed[str(seed)] = {
            "active_archived_roster": active,
            "random_rosters": randoms,
            "random_mean": random_mean,
            "active_minus_random_mean_utility": delta,
            "active_minus_static_utility": active_static_delta,
            "equal_query_budget": True,
            "equal_parent_and_static": True,
            "equal_realized_compute_shape": True,
        }
    roster_screen = roster_screen_adjudication(deltas)
    return {
        "schema_version": "equal-compute-selector-report-v2",
        "claim_status": "BALANCED360_POST_DEVELOPMENT_MECHANISM_CONTROL",
        "estimand": (
            "archived active query roster versus block-count-matched random "
            "rosters under a new fixed-compute assimilation operator"
        ),
        "seeds": seeds, "replicates_per_seed": int(replicates),
        "per_seed": per_seed,
        "aggregate": {
            "active_minus_random_utility_equal_seed_mean": mean(deltas),
            "active_minus_static_utility_equal_seed_mean": mean(
                active_static_deltas
            ),
            "seeds_active_better": sum(value > 0 for value in deltas),
            "seeds_tied": sum(value == 0 for value in deltas),
            "screening_gate_requires_active_better_in_all_seeds": True,
            "screening_gate_passed": roster_screen["passed"],
            "archived_roster_superiority_supported": roster_screen["passed"],
            "adaptive_selector_policy_effect_estimated": False,
            "equivalence_or_null_proven": False,
            "confirmatory_claim": False,
        },
        "adjudication": roster_screen["adjudication"],
        "limitations": [
            "balanced360 informed architecture and policy development",
            "archived active identities are a roster treatment, not a rerun of the adaptive selector policy",
            "random block counts condition on the archived active trajectory",
            "random schedules quantify algorithmic randomization on one reused stream, not independent cohorts",
            "a new sealed chronological EAST stream remains required",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[131, 133, 137])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("FUSION_DATA_ROOT", "data")),
    )
    args = parser.parse_args()
    report = build(
        args.root.expanduser().resolve(),
        seeds=[int(seed) for seed in args.seeds],
        replicates=int(args.replicates),
        data_root=args.data_root.expanduser().resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
