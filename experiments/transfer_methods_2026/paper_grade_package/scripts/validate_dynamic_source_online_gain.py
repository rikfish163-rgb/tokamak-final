#!/usr/bin/env python3
"""Fail-closed validator for the dynamic-source balanced360 online replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCRIPT = Path(__file__).resolve()
PACKAGE = SCRIPT.parents[1]
REPO = PACKAGE.parents[2]
DEFAULT_CONFIG = PACKAGE / "configs/dynamic_source_online_alarm_search_v1.json"
FORBIDDEN_QUERY_LABEL_KEYS = {
    "is_disrupt", "down_time_s", "revealed_is_disrupt",
    "revealed_down_time_s", "outcome", "label",
}
HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def recompute_metrics(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    disruptive = safe = tp = fp = timely = 0
    leads: list[float] = []
    for row in rows:
        shot = int(row["shot"])
        outcome = outcomes[shot]
        alarmed = bool(row["alarmed"])
        if bool(outcome["is_disrupt"]):
            disruptive += 1
            if alarmed:
                if row.get("alarm_time_s") is None or outcome.get("down_time_s") is None:
                    raise RuntimeError(f"shot {shot}: disruptive alarm lacks a time")
                tp += 1
                lead = (
                    float(outcome["down_time_s"]) - float(row["alarm_time_s"])
                ) * 1000.0
                leads.append(lead)
                timely += int(lead >= 30.0 - 1e-9)
        else:
            safe += 1
            fp += int(alarmed)
    return {
        "TP": tp,
        "FP": fp,
        "FN": disruptive - tp,
        "TN": safe - fp,
        "tpr": tp / max(disruptive, 1),
        "fpr": fp / max(safe, 1),
        "tpr_at_30ms": timely / max(disruptive, 1),
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "over_3000ms_count": int(sum(lead > 3000.0 for lead in leads)),
    }


def assert_metric_equal(
    recomputed: Mapping[str, Any], recorded: Mapping[str, Any], *, name: str,
) -> None:
    required = (
        "TP", "FP", "FN", "TN", "tpr", "fpr", "tpr_at_30ms",
        "median_lead_ms", "over_3000ms_count",
    )
    for key in required:
        if key not in recomputed or key not in recorded:
            raise RuntimeError(f"{name} is missing metric {key}")
        left, right = recomputed[key], recorded[key]
        if left is None or right is None:
            if left is not right:
                raise RuntimeError(f"{name} {key} differs: {left!r} != {right!r}")
        elif isinstance(left, (int, np.integer)) and isinstance(right, int):
            if int(left) != int(right):
                raise RuntimeError(f"{name} {key} differs: {left!r} != {right!r}")
        elif not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{name} {key} differs: {left!r} != {right!r}")


def validate_temporal_ledgers(
    static: Sequence[Mapping[str, Any]],
    online: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    reveals: Sequence[Mapping[str, Any]],
    *, expected_count: int = 360,
) -> dict[str, int]:
    if not (len(static) == len(online) == len(queries) == expected_count):
        raise RuntimeError("static, online and query ledgers must cover the full stream")
    static_shots = [int(row["shot"]) for row in static]
    online_shots = [int(row["shot"]) for row in online]
    query_shots = [int(row["shot"]) for row in queries]
    if static_shots != online_shots or static_shots != query_shots:
        raise RuntimeError("static, online and query shot order differs")
    if static_shots != sorted(static_shots) or len(set(static_shots)) != expected_count:
        raise RuntimeError("balanced360 roster is not unique and chronological")
    expected_indices = list(range(expected_count))
    if any(
        [int(row["stream_index"]) for row in ledger] != expected_indices
        for ledger in (static, online, queries)
    ):
        raise RuntimeError("stream indices are not contiguous and aligned")
    if [int(row["state_shot"]) for row in online] != list(
        range(1, expected_count + 1)
    ):
        raise RuntimeError("online state-shot order is not strictly causal")
    for row in queries:
        leaked = FORBIDDEN_QUERY_LABEL_KEYS.intersection(row)
        if leaked:
            raise RuntimeError(
                "query decision contains outcome field(s): " + ", ".join(sorted(leaked))
            )
    selected = [
        (int(row["shot"]), int(row["stream_index"]))
        for row in queries if bool(row["selected"])
    ]
    predicted_selected = [
        (int(row["shot"]), int(row["stream_index"]))
        for row in online if bool(row["queried"])
    ]
    revealed = [
        (int(row["shot"]), int(row["stream_index"])) for row in reveals
    ]
    if selected != predicted_selected or selected != revealed:
        raise RuntimeError("selected queries, sealed prediction flags and reveals differ")
    if len(set(revealed)) != len(revealed):
        raise RuntimeError("a queried shot was revealed more than once")
    return {
        "shots": expected_count,
        "queries": len(selected),
        "reveals": len(revealed),
    }


def read_hdf5_outcomes(data_root: Path, shots: Sequence[int]) -> dict[int, dict[str, Any]]:
    outcomes: dict[int, dict[str, Any]] = {}
    for shot in shots:
        path = (
            data_root / "EAST/unified_hdf5" / str(shot // 100 * 100)
            / f"{shot}.hdf5"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            is_disrupt = bool(
                np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0]
            )
            down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
        outcomes[shot] = {
            "is_disrupt": is_disrupt,
            "down_time_s": float(down[0]) if is_disrupt and down.size else None,
        }
    return outcomes


def operational_group(
    row: Mapping[str, Any], outcome: Mapping[str, Any],
) -> str:
    if not bool(outcome["is_disrupt"]):
        return "FP" if bool(row["alarmed"]) else "TN"
    if not bool(row["alarmed"]):
        return "FN"
    lead = (
        float(outcome["down_time_s"]) - float(row["alarm_time_s"])
    ) * 1000.0
    return "TP" if lead >= 30.0 - 1e-9 else "TARDY"


def transition_receipt(
    static: Sequence[Mapping[str, Any]],
    online: Sequence[Mapping[str, Any]],
    outcomes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    rescues: list[int] = []
    harms: list[int] = []
    for static_row, online_row in zip(static, online):
        shot = int(static_row["shot"])
        left = operational_group(static_row, outcomes[shot])
        right = operational_group(online_row, outcomes[shot])
        transitions[f"{left}->{right}"] += 1
        if (left == "FP" and right == "TN") or (
            left in {"FN", "TARDY"} and right == "TP"
        ):
            rescues.append(shot)
        if (left == "TN" and right == "FP") or (
            left == "TP" and right in {"FN", "TARDY"}
        ):
            harms.append(shot)
    return {
        "transitions": dict(sorted(transitions.items())),
        "rescue_shots": rescues,
        "harm_shots": harms,
        "gross_rescues": len(rescues),
        "gross_harms": len(harms),
        "net_rescues": len(rescues) - len(harms),
    }


def validate_static_manifest(run_dir: Path, static_shots: Sequence[int]) -> dict[str, Any]:
    path = run_dir / "static_seal_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    predictions = run_dir / str(manifest["prediction_path"])
    if sha256_file(predictions) != manifest["prediction_sha256"]:
        raise RuntimeError("static prediction seal SHA256 mismatch")
    traces = manifest.get("risk_traces")
    if not isinstance(traces, list) or len(traces) != len(static_shots):
        raise RuntimeError("static risk trace manifest is incomplete")
    if [int(row["shot"]) for row in traces] != list(static_shots):
        raise RuntimeError("static risk trace order differs from predictions")
    for row in traces:
        trace = (run_dir / str(row["path"])).resolve()
        if run_dir.resolve() not in trace.parents or not trace.is_file():
            raise RuntimeError("static risk trace path escapes or is missing")
        if sha256_file(trace) != row["sha256"]:
            raise RuntimeError(f"static risk trace SHA256 mismatch: {trace.name}")
    return {
        "manifest_sha256": sha256_file(path),
        "prediction_sha256": manifest["prediction_sha256"],
        "risk_trace_count": len(traces),
    }


def validate_updates(path: Path, *, expected_shots: int, expected_queries: int) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    finalize_state_shots: list[int] = []
    revealed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            event = str(row.get("event"))
            event_counts[event] += 1
            if event == "unified_intra_update" and not bool(
                row.get("strict_prediction_sealed_before_update")
            ):
                raise RuntimeError(
                    f"intra update lacks causal seal at {path}:{line_number}"
                )
            if event in {"unified_revealed_finalize", "unified_unqueried_finalize"}:
                finalize_state_shots.append(int(row["state_shot"]))
                revealed += int(event == "unified_revealed_finalize")
    if finalize_state_shots != list(range(1, expected_shots + 1)):
        raise RuntimeError("shot finalization ledger is not strictly chronological")
    if revealed != expected_queries:
        raise RuntimeError("revealed finalize count differs from query count")
    return {"event_counts": dict(sorted(event_counts.items()))}


def _assert_hash(path: Path, expected: str, name: str) -> None:
    if not HEX64.fullmatch(str(expected)) or sha256_file(path) != expected:
        raise RuntimeError(f"{name} SHA256 mismatch")


def _controller_candidate(candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(name).removeprefix("initial_"): float(value)
        for name, value in candidate.items()
    }


def validate(
    report_path: Path, *, data_root: Path, run_root: Path | None,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["true_replay_gate"]["seeds"]]
    if report.get("claim_status") != "TEST_DRIVEN_EXPLORATION":
        raise RuntimeError("report overclaims balanced360 development evidence")
    if report.get("independent_future_confirmation") is not False:
        raise RuntimeError("report incorrectly claims independent future confirmation")
    if report.get("selected_candidate") != config["selected_candidate"]:
        raise RuntimeError("report candidate differs from registered search config")
    seed_rows = report.get("seed_results")
    if not isinstance(seed_rows, list) or [int(row["seed"]) for row in seed_rows] != seeds:
        raise RuntimeError("report does not contain the registered three seeds")

    expected_initial = _controller_candidate(config["selected_candidate"])
    all_receipts: list[dict[str, Any]] = []
    common_roster: list[int] | None = None
    total_queries = total_rescues = total_harms = 0
    query_disruptive = query_safe = 0
    recomputed_seed_metrics: list[dict[str, Any]] = []
    for report_row in seed_rows:
        seed = int(report_row["seed"])
        summary_path = Path(str(report_row["summary"]))
        if run_root is not None:
            summary_path = run_root / f"seed{seed}/east-online/summary.json"
        run_dir = summary_path.parent
        _assert_hash(summary_path, str(report_row["summary_sha256"]), f"seed{seed} summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary["seed"]) != seed or summary["status"] != "ONLINE_DIAGNOSTIC":
            raise RuntimeError(f"seed{seed}: summary status/identity mismatch")
        if summary["stage"] != "STATIC360_THEN_RESET_ONLINE360_DEVELOPMENT_SEARCH":
            raise RuntimeError(f"seed{seed}: non-development stage recorded")
        protocol = summary["execution_protocol"]
        if not (
            bool(protocol["phase1_static_complete_before_phase2"])
            and bool(protocol["phase2_reset_to_same_migrated_initial_sha"])
            and bool(protocol["selection_on_360_in_this_run"])
        ):
            raise RuntimeError(f"seed{seed}: static/reset/development protocol failed")
        order = summary["order_contract"]
        if not (
            bool(order["chronological_main_result"])
            and not bool(order["is_reordered"])
            and not bool(order["query_selection_label_oracle_used"])
        ):
            raise RuntimeError(f"seed{seed}: chronology or query-oracle contract failed")
        if summary["timescale_contract"] != {
            "enable_intra": True,
            "enable_inter": True,
            "enable_query": True,
            "enable_reveal": True,
        }:
            raise RuntimeError(f"seed{seed}: full online timescale method was not used")
        if summary["alarm_contract"].get("parameters_initial") != expected_initial:
            raise RuntimeError(f"seed{seed}: alarm initialization mismatch")
        if summary["alarm_contract"].get("minimum_lead_ms") != 30.0:
            raise RuntimeError(f"seed{seed}: EAST minimum lead is not 30 ms")
        if not bool(summary["alarm_contract"].get("over_3000ms_counts_as_tpr")):
            raise RuntimeError(f"seed{seed}: >3000 ms alarm scoring contract changed")

        static_path = run_dir / "static_predictions.jsonl"
        online_path = run_dir / "predictions.jsonl"
        query_path = run_dir / "query_decisions.jsonl"
        reveal_path = run_dir / "reveals.jsonl"
        update_path = run_dir / "updates.jsonl"
        hashes = summary["hashes"]
        for name, path, key in (
            ("static predictions", static_path, "static_predictions"),
            ("online predictions", online_path, "predictions"),
            ("query decisions", query_path, "queries"),
            ("reveals", reveal_path, "reveals"),
            ("updates", update_path, "updates"),
        ):
            _assert_hash(path, str(hashes[key]), f"seed{seed} {name}")
        if not HEX64.fullmatch(str(hashes.get("base_before_after", ""))):
            raise RuntimeError(f"seed{seed}: frozen base before/after hash missing")
        static = read_jsonl(static_path)
        online = read_jsonl(online_path)
        queries = read_jsonl(query_path)
        reveals = read_jsonl(reveal_path)
        ledger = validate_temporal_ledgers(static, online, queries, reveals)
        if ledger["queries"] != int(summary["query_count"]):
            raise RuntimeError(f"seed{seed}: query count differs from summary")
        roster = [int(row["shot"]) for row in static]
        if common_roster is None:
            common_roster = roster
        elif roster != common_roster:
            raise RuntimeError("the three seeds did not use the same balanced360 roster")
        outcomes = read_hdf5_outcomes(data_root, roster)
        class_counts = Counter(bool(row["is_disrupt"]) for row in outcomes.values())
        if class_counts != Counter({True: 180, False: 180}):
            raise RuntimeError(f"seed{seed}: HDF5 balanced360 is not 180/180")
        static_metric = recompute_metrics(static, outcomes)
        online_metric = recompute_metrics(online, outcomes)
        assert_metric_equal(
            static_metric, summary["results"]["static_unified"],
            name=f"seed{seed} static",
        )
        assert_metric_equal(
            online_metric, summary["results"]["online_unified"],
            name=f"seed{seed} online",
        )
        transition = transition_receipt(static, online, outcomes)
        recorded_transition = summary["transition_report"]
        for key in ("transitions", "rescue_shots", "harm_shots", "gross_rescues", "gross_harms", "net_rescues"):
            if transition[key] != recorded_transition[key]:
                raise RuntimeError(f"seed{seed}: transition receipt differs at {key}")
        static_j = static_metric["tpr_at_30ms"] - static_metric["fpr"]
        online_j = online_metric["tpr_at_30ms"] - online_metric["fpr"]
        delta_j = online_j - static_j
        if delta_j <= 1e-12:
            raise RuntimeError(f"seed{seed}: true online replay lacks strict positive delta-J")
        static_manifest = validate_static_manifest(run_dir, roster)
        updates = validate_updates(
            update_path, expected_shots=360, expected_queries=ledger["queries"],
        )
        for reveal in reveals:
            query_disruptive += int(bool(reveal["revealed_is_disrupt"]))
            query_safe += int(not bool(reveal["revealed_is_disrupt"]))
        total_queries += ledger["queries"]
        total_rescues += transition["gross_rescues"]
        total_harms += transition["gross_harms"]
        recomputed_seed_metrics.append({
            "seed": seed,
            "static": static_metric,
            "online": online_metric,
            "delta_j": delta_j,
        })
        all_receipts.append({
            "seed": seed,
            "run_dir": f"${{FUSION_RUN_ROOT}}/seed{seed}/east-online",
            "summary_sha256": sha256_file(summary_path),
            "source_checkpoint_sha256": hashes["source_checkpoint"],
            "sealed_parent_sha256": hashes["sealed_parent_state"],
            "frozen_base_before_after_sha256": hashes["base_before_after"],
            "ledger": ledger,
            "static_seal": static_manifest,
            "updates": updates,
            "transition": transition,
            "metrics": recomputed_seed_metrics[-1],
        })

    mean_static_tpr = float(np.mean([
        row["static"]["tpr_at_30ms"] for row in recomputed_seed_metrics
    ]))
    mean_static_fpr = float(np.mean([
        row["static"]["fpr"] for row in recomputed_seed_metrics
    ]))
    mean_online_tpr = float(np.mean([
        row["online"]["tpr_at_30ms"] for row in recomputed_seed_metrics
    ]))
    mean_online_fpr = float(np.mean([
        row["online"]["fpr"] for row in recomputed_seed_metrics
    ]))
    mean_delta_j = float(np.mean([row["delta_j"] for row in recomputed_seed_metrics]))
    gate = config["true_replay_gate"]
    passed = (
        mean_online_tpr >= float(gate["minimum_mean_tpr30"])
        and mean_online_fpr <= float(gate["maximum_mean_fpr"])
        and all(row["delta_j"] > 1e-12 for row in recomputed_seed_metrics)
    )
    if not passed:
        raise RuntimeError("registered true replay gate failed")
    aggregate = {
        "mean_static": {
            "tpr_at_30ms": mean_static_tpr,
            "fpr": mean_static_fpr,
            "j": mean_static_tpr - mean_static_fpr,
        },
        "mean_online": {
            "tpr_at_30ms": mean_online_tpr,
            "fpr": mean_online_fpr,
            "j": mean_online_tpr - mean_online_fpr,
        },
        "mean_delta_j": mean_delta_j,
        "total_queries": total_queries,
        "query_outcomes": {
            "disruptive": query_disruptive,
            "safe": query_safe,
        },
        "gross_rescues": total_rescues,
        "gross_harms": total_harms,
        "net_rescues": total_rescues - total_harms,
        "net_rescues_per_query": (
            (total_rescues - total_harms) / max(total_queries, 1)
        ),
        "gate_passed": True,
    }
    validation = {
        "schema_version": "dynamic-source-online-gain-validation-v1",
        "status": "PASS",
        "claim_status": "TEST_DRIVEN_EXPLORATION",
        "independent_future_confirmation": False,
        "source_report": "${FUSION_RUN_ROOT}/report.json",
        "source_report_sha256": sha256_file(report_path),
        "config": str(config_path.resolve().relative_to(REPO)),
        "config_sha256": sha256_file(config_path),
        "data_root": "${FUSION_DATA_ROOT}",
        "checks": {
            "hdf5_outcomes_authoritative": True,
            "static_sealed_before_online": True,
            "same_parent_reset": True,
            "chronological_360": True,
            "query_decisions_label_free": True,
            "reveals_selected_only": True,
            "intra_updates_prediction_sealed": True,
            "frozen_source_hash_recorded": True,
            "all_file_hashes_match": True,
            "metrics_independently_recomputed": True,
        },
        "aggregate": aggregate,
        "seeds": all_receipts,
    }
    archived = dict(report)
    archived["validation"] = validation
    archived["artifact_root"] = "${FUSION_RUN_ROOT}"
    return validation, archived


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--completion", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if data_root is None:
        raw = os.environ.get("FUSION_DATA_ROOT")
        if not raw:
            raise RuntimeError("--data-root or FUSION_DATA_ROOT is required")
        data_root = Path(raw)
    validation, archived = validate(
        args.report, data_root=data_root,
        run_root=args.run_root, config_path=args.config,
    )
    if args.archive is not None:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.archive.write_text(
            json.dumps(archived, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.completion is not None:
        args.completion.parent.mkdir(parents=True, exist_ok=True)
        args.completion.write_text(
            json.dumps({
                "status": "complete",
                "mission": "dynamic-source EAST online gain",
                "validator_status": validation["status"],
                "claim_status": validation["claim_status"],
                "aggregate": validation["aggregate"],
                "archive": str(args.archive) if args.archive is not None else None,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(validation["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
