#!/usr/bin/env python3
"""Run the 2x2 AdvancedTime positive-horizon ablation on the current chain.

The experiment changes only the upper boundary of positive training windows.
HDF5 ``IsDisrupt`` and ``DownTime`` remain authoritative.  Two PADFIX source
parents are trained (fixed125 and J-TEXT max(25, AdvancedTime)), each is adapted
to EAST support110 with both fixed125 and EAST max(125, AdvancedTime), and all
four frozen target models predict the same balanced360 development stream.

Balanced360 outcomes are opened only after a complete outcome-free prediction
ledger has been sealed for an arm.  The support-selected threshold is the
primary non-test-selected comparison.  A paper131-style test-selected TPR=0.8
operation point is also reported and explicitly marked non-deployable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PADFIX = HERE / "paper_grade_package/scripts/train_source_padfix.py"
JTEXT_ADVANCED_TIME = ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"
EAST_ADVANCED_TIME = ROOT / "shot_list/EAST/AdvancedTime_EAST.json"
WINDOW = 128
OUTCOME_KEYS = frozenset({
    "label", "truth", "outcome", "is_disrupt", "down", "down_time_s",
    "lead_ms", "timely", "tp", "fp", "tn", "fn",
})


def comparison_arms() -> tuple[tuple[str, str, str], ...]:
    """Return a full factorial matrix for separate source/target attribution."""

    return (
        ("source_fixed__east_fixed", "fixed125", "fixed125"),
        ("source_dynamic__east_fixed", "advanced_floor", "fixed125"),
        ("source_fixed__east_dynamic", "fixed125", "advanced_floor"),
        ("source_dynamic__east_dynamic", "advanced_floor", "advanced_floor"),
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(values: Sequence[int]) -> str:
    return hashlib.sha256(canonical_json([int(value) for value in values]).encode()).hexdigest()


def model_sha256(*modules: Any) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{module_index}:{name}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    delta = positive[:, None] - negative[None, :]
    return float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))


def select_tpr_first_threshold(
    scores: Sequence[float], labels: Sequence[bool], *, target_tpr: float,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    if values.ndim != 1 or truth.shape != values.shape or not len(values):
        raise ValueError("scores and labels must be aligned non-empty vectors")
    if not np.isfinite(values).all() or not truth.any() or truth.all():
        raise ValueError("threshold selection requires finite scores and both classes")
    best = None
    for threshold in sorted(set(values.tolist()), reverse=True) + [float("-inf")]:
        tpr = float(np.mean(values[truth] >= threshold))
        fpr = float(np.mean(values[~truth] >= threshold))
        if tpr + 1e-12 < float(target_tpr):
            continue
        rank = (fpr, -float(threshold), -tpr)
        if best is None or rank < best[0]:
            best = (rank, float(threshold), tpr, fpr)
    if best is None:
        raise RuntimeError(f"cannot reach target TPR={target_tpr}")
    return {"threshold": best[1], "tpr": best[2], "fpr": best[3]}


def evaluate_threshold(
    rows: Sequence[Mapping[str, Any]], traces: Mapping[int, np.ndarray],
    truth: Mapping[int, Mapping[str, Any]], *, threshold: float,
    minimum_lead_ms: float = 30.0,
) -> dict[str, Any]:
    """Evaluate first crossing; alarms earlier than 3000 ms remain true hits."""

    roster = [int(row["shot"]) for row in rows]
    if len(roster) != len(set(roster)) or set(roster) != set(traces) or set(roster) != set(truth):
        raise ValueError("prediction, trace, and truth rosters do not match")
    for row in rows:
        forbidden = OUTCOME_KEYS.intersection(row)
        if forbidden:
            raise ValueError(f"prediction seal contains outcome field: {sorted(forbidden)[0]}")
    labels = np.asarray([bool(truth[shot]["is_disrupt"]) for shot in roster], dtype=bool)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    raw_hits: list[bool] = []
    timely_hits: list[bool] = []
    false_alarms: list[bool] = []
    timely_leads: list[float] = []
    over_3000: list[int] = []
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        shot = int(row["shot"])
        risk = np.asarray(traces[shot], dtype=np.float32)
        passing = np.flatnonzero(risk >= float(threshold))
        fired = bool(len(passing))
        trace_index = int(passing[0]) if fired else None
        endpoint = (
            int(row.get("first_endpoint", WINDOW - 1)) + trace_index
            if trace_index is not None else None
        )
        alarm_time_s = (
            float(row["start_time_s"]) + endpoint / float(row["sample_rate_hz"])
            if endpoint is not None else None
        )
        target = truth[shot]
        is_disrupt = bool(target["is_disrupt"])
        lead_ms = None
        timely = False
        if is_disrupt:
            raw_hits.append(fired)
            if fired:
                down = target.get("down_time_s")
                if down is None or not math.isfinite(float(down)):
                    raise ValueError(f"disruptive shot {shot} has no finite DownTime")
                lead_ms = (float(down) - float(alarm_time_s)) * 1000.0
                timely = lead_ms + 1e-9 >= float(minimum_lead_ms)
                if timely:
                    timely_leads.append(float(lead_ms))
                    if lead_ms > 3000.0:
                        over_3000.append(shot)
            timely_hits.append(timely)
        else:
            false_alarms.append(fired)
        evaluated.append({
            "shot": shot,
            "is_disrupt": is_disrupt,
            "score": float(row["score"]),
            "alarm": fired,
            "alarm_endpoint": endpoint,
            "alarm_time_s": alarm_time_s,
            "lead_ms": lead_ms,
            "timely": timely if is_disrupt else None,
        })
    raw_tpr = float(np.mean(raw_hits))
    operational_tpr = float(np.mean(timely_hits))
    fpr = float(np.mean(false_alarms))
    return {
        "shots": len(roster),
        "disruptions": int(labels.sum()),
        "safe": int((~labels).sum()),
        "threshold": float(threshold),
        "minimum_run_points": 1,
        "minimum_lead_ms": float(minimum_lead_ms),
        "raw_tpr": raw_tpr,
        "operational_tpr": operational_tpr,
        "fpr": fpr,
        "auc": _auc(labels, scores),
        "utility_operational_tpr_minus_fpr": operational_tpr - fpr,
        "median_lead_ms": float(np.median(timely_leads)) if timely_leads else None,
        "over_3000ms_count": len(over_3000),
        "over_3000ms_shots": over_3000,
        "evaluated_rows": evaluated,
    }


def operational_selection_scores(
    rows: Sequence[Mapping[str, Any]], traces: Mapping[int, np.ndarray],
    truth: Mapping[int, Mapping[str, Any]], *, minimum_lead_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scores whose positive-class meaning already enforces lead time."""

    scores: list[float] = []
    labels: list[bool] = []
    for row in rows:
        shot = int(row["shot"])
        risk = np.asarray(traces[shot], dtype=np.float32)
        target = truth[shot]
        is_disrupt = bool(target["is_disrupt"])
        labels.append(is_disrupt)
        if not is_disrupt:
            scores.append(float(risk.max()))
            continue
        down = target.get("down_time_s")
        if down is None or not math.isfinite(float(down)):
            raise ValueError(f"disruptive shot {shot} has no finite DownTime")
        endpoints = int(row.get("first_endpoint", WINDOW - 1)) + np.arange(len(risk))
        times = float(row["start_time_s"]) + endpoints / float(row["sample_rate_hz"])
        lead_ms = (float(down) - times) * 1000.0
        eligible = lead_ms + 1e-9 >= float(minimum_lead_ms)
        scores.append(float(risk[eligible].max()) if bool(eligible.any()) else float("-inf"))
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _padfix_command(
    *, seed: int, policy: str, output: Path, args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable, str(PADFIX),
        "--seed", str(seed),
        "--epochs", str(args.source_epochs),
        "--minimum-epochs", str(args.source_minimum_epochs),
        "--patience", str(args.source_patience),
        "--steps-per-epoch", str(args.source_steps_per_epoch),
        "--batch-size", str(args.source_batch_size),
        "--learning-rate", str(args.source_learning_rate),
        "--target-tpr", str(args.source_target_tpr),
        "--context-view-weight", "0.5",
        "--consistency-weight", "0.2",
        "--representation-consistency-weight", "0.5",
        "--pad-adversary-weight", "0.005",
        "--device", args.device,
        "--label-policy", policy,
        "--advanced-time-path", str(args.jtext_advanced_time),
        "--evaluate-test",
        "--output", str(output),
    ]


def _start_time_s(data_root: Path, shot: int) -> float:
    import h5py

    path = data_root / "EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    with h5py.File(path, "r") as handle:
        signal = handle["data/ip"]
        return float(np.asarray(signal.attrs.get("StartTime", 0.0)).reshape(-1)[0])


def _run_target_arm(
    *, name: str, source_policy: str, east_policy: str,
    source_checkpoint: Path, source_test_metrics: Path, seed: int,
    data_root: Path, output: Path, args: argparse.Namespace,
) -> dict[str, Any]:
    import common
    import grid_search as grid
    import run_online_active_a1 as online
    import shot_alarm as alarm

    output.mkdir(parents=True)
    enc, head, support, center, scale, target_label_audit = alarm.train_support_only(
        seed, source_checkpoint,
        target_label_policy=east_policy,
        advanced_time_path=args.east_advanced_time,
        adapt_epochs=args.target_epochs,
        adapt_batch_size=args.target_batch_size,
        return_label_audit=True,
    )
    support_traces = [
        alarm.full_risk(enc, head, shot, bs=args.inference_batch_size)
        for shot in support
    ]
    support_scores = np.asarray([float(trace.max()) for trace in support_traces])
    support_labels = np.asarray([bool(shot["is_disrupt"]) for shot in support])
    support_point = select_tpr_first_threshold(
        support_scores, support_labels, target_tpr=args.target_tpr,
    )
    support_rows = [
        {"shot": int(shot["shot"]), "score": float(score), "is_disrupt": bool(label)}
        for shot, score, label in zip(support, support_scores, support_labels)
    ]
    _write_jsonl(output / "support_scores.jsonl", support_rows)

    full_test_ids = sorted(int(value) for value in common.DataSpec().east["test_locked"])
    test_ids = full_test_ids[: int(args.max_test_shots)]
    if len(test_ids) < 2:
        raise ValueError("test smoke subset must contain at least two shots")
    test = online.load_test_features_only(test_ids, center, scale)
    sealed_rows: list[dict[str, Any]] = []
    traces: dict[int, np.ndarray] = {}
    for stream_index, shot in enumerate(test):
        shot_id = int(shot["shot"])
        risk = alarm.full_risk(
            enc, head, shot, bs=args.inference_batch_size,
        )
        traces[shot_id] = np.asarray(risk, dtype=np.float32)
        sealed_rows.append({
            "shot": shot_id,
            "stream_index": int(stream_index),
            "score": float(risk.max()),
            "risk_length": int(len(risk)),
            "first_endpoint": WINDOW - 1,
            "start_time_s": _start_time_s(data_root, shot_id),
            "sample_rate_hz": float(shot["rate"]),
            "sealed": True,
        })
        if (stream_index + 1) % 40 == 0:
            print(f"[{name}] sealed {stream_index + 1}/{len(test_ids)}", flush=True)
    _write_jsonl(output / "balanced360_prediction_seal.jsonl", sealed_rows)
    risk_path = output / "balanced360_risk_traces.npz"
    np.savez_compressed(risk_path, **{f"shot_{shot}": trace for shot, trace in traces.items()})
    prediction_sha = sha256_file(output / "balanced360_prediction_seal.jsonl")

    # First balanced360 outcome access for this arm occurs only after the full seal.
    truth = {shot: online.read_outcome_after_seal(shot) for shot in test_ids}
    counts = {
        "disruptions": sum(bool(value["is_disrupt"]) for value in truth.values()),
        "safe": sum(not bool(value["is_disrupt"]) for value in truth.values()),
    }
    if len(test_ids) == 360 and counts != {"disruptions": 180, "safe": 180}:
        raise ValueError(f"balanced360 HDF5 counts changed: {counts}")
    if not counts["disruptions"] or not counts["safe"]:
        raise ValueError(f"test subset needs both outcomes: {counts}")
    support_metrics = evaluate_threshold(
        sealed_rows, traces, truth,
        threshold=float(support_point["threshold"]), minimum_lead_ms=30.0,
    )
    test_labels = [bool(truth[int(row["shot"])] ["is_disrupt"]) for row in sealed_rows]
    test_point = select_tpr_first_threshold(
        [float(row["score"]) for row in sealed_rows], test_labels, target_tpr=0.8,
    )
    test_selected_metrics = evaluate_threshold(
        sealed_rows, traces, truth,
        threshold=float(test_point["threshold"]), minimum_lead_ms=30.0,
    )
    operational_scores, operational_labels = operational_selection_scores(
        sealed_rows, traces, truth, minimum_lead_ms=30.0,
    )
    operational_test_point = select_tpr_first_threshold(
        operational_scores, operational_labels, target_tpr=0.8,
    )
    operational_test_selected_metrics = evaluate_threshold(
        sealed_rows, traces, truth,
        threshold=float(operational_test_point["threshold"]), minimum_lead_ms=30.0,
    )
    _write_jsonl(
        output / "balanced360_support_threshold_evaluated.jsonl",
        support_metrics.pop("evaluated_rows"),
    )
    _write_jsonl(
        output / "balanced360_test_selected_evaluated.jsonl",
        test_selected_metrics.pop("evaluated_rows"),
    )
    _write_jsonl(
        output / "balanced360_operational_test_selected_evaluated.jsonl",
        operational_test_selected_metrics.pop("evaluated_rows"),
    )
    source_result = json.loads(source_test_metrics.read_text())
    result = {
        "schema_version": "advanced-time-label-ablation-arm-v1",
        "status": (
            "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION"
            if len(test_ids) == 360 else "SMOKE_ONLY"
        ),
        "arm": name,
        "seed": int(seed),
        "source_label_policy": source_policy,
        "east_label_policy": east_policy,
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_test220": source_result,
        "east_support_label_audit": target_label_audit,
        "east_support_threshold": {
            **support_point,
            "source": "EAST support110 HDF5 outcomes only",
            "balanced360_used": False,
        },
        "balanced360_support_selected": support_metrics,
        "balanced360_paper131_test_selected_tpr80": {
            "deployable": False,
            "selection_source": "balanced360 outcomes after prediction seal",
            "operation_point": test_point,
            "metrics": test_selected_metrics,
        },
        "balanced360_operational_test_selected_tpr80": {
            "deployable": False,
            "selection_source": "balanced360 outcomes after prediction seal",
            "selection_metric": "TPR with lead >=30 ms",
            "operation_point": operational_test_point,
            "metrics": operational_test_selected_metrics,
        },
        "model_sha256_after_east_adaptation": model_sha256(enc, head),
        "trainable_parameters_after_adaptation": int(sum(
            parameter.numel() for module in (enc, head)
            for parameter in module.parameters() if parameter.requires_grad
        )),
        "prediction_seal_sha256": prediction_sha,
        "risk_traces_sha256": sha256_file(risk_path),
        "prediction_order_sha256": sha256_ids(test_ids),
        "balanced360_outcomes_read_after_complete_prediction_seal": True,
        "balanced360_used_for_training_or_support_threshold": False,
    }
    abnormal = (
        float(support_metrics["operational_tpr"]) < 0.60
        or float(support_metrics["fpr"]) > 0.40
    )
    result["normality"] = {
        "abnormal": bool(abnormal),
        "rule": "operational_tpr<0.60 or fpr>0.40",
    }
    _write_json(output / "summary.json", result)
    return result


def _effect_decomposition(results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    ff = results["source_fixed__east_fixed"]["balanced360_support_selected"]
    df = results["source_dynamic__east_fixed"]["balanced360_support_selected"]
    fd = results["source_fixed__east_dynamic"]["balanced360_support_selected"]
    dd = results["source_dynamic__east_dynamic"]["balanced360_support_selected"]
    output: dict[str, dict[str, float]] = {}
    for metric in ("operational_tpr", "raw_tpr", "fpr", "auc", "utility_operational_tpr_minus_fpr"):
        values = [float(row[metric]) for row in (ff, df, fd, dd)]
        output[metric] = {
            "jtext_effect_when_east_fixed": values[1] - values[0],
            "jtext_effect_when_east_dynamic": values[3] - values[2],
            "east_effect_when_jtext_fixed": values[2] - values[0],
            "east_effect_when_jtext_dynamic": values[3] - values[1],
            "factorial_interaction": values[3] - values[1] - values[2] + values[0],
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    output.mkdir(parents=True)
    source_root = output / "sources"
    target_root = output / "targets"
    source_root.mkdir(); target_root.mkdir()
    data_root = Path(args.data_root).expanduser().resolve()
    os.environ["FUSION_PROJECT_ROOT"] = str(ROOT)
    os.environ["FUSION_DATA_ROOT"] = str(data_root)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the requested rl_env run")
    if Path(sys.executable).resolve() != Path(args.expected_python).expanduser().resolve():
        raise RuntimeError(
            f"wrong interpreter: {sys.executable}; expected {args.expected_python}"
        )
    started = time.time()
    commands: dict[str, list[str]] = {}
    source_paths: dict[str, Path] = {}
    source_dirs: dict[str, Path] = {}
    environment = dict(os.environ)
    reuse_root = (
        Path(args.reuse_source_root).expanduser().resolve()
        if args.reuse_source_root is not None else None
    )
    for policy in ("fixed125", "advanced_floor"):
        tag = "fixed" if policy == "fixed125" else "dynamic"
        source_dir = (reuse_root / tag) if reuse_root is not None else (source_root / tag)
        if reuse_root is None:
            command = _padfix_command(
                seed=args.seed, policy=policy, output=source_dir, args=args,
            )
            commands[f"source_{tag}"] = command
            print(f"[source:{tag}] {' '.join(command)}", flush=True)
            subprocess.run(command, check=True, cwd=ROOT, env=environment)
        else:
            commands[f"source_{tag}"] = ["REUSED_VERBATIM", str(source_dir)]
        checkpoint_path = source_dir / "source_checkpoint.pt"
        metrics_path = source_dir / "test220_metrics.json"
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"incomplete source artifacts: {source_dir}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        actual_policy = checkpoint.get("training_label", {}).get("policy")
        if actual_policy != policy or int(checkpoint.get("seed", -1)) != int(args.seed):
            raise ValueError(
                f"reused source contract mismatch: {source_dir} "
                f"policy={actual_policy!r} seed={checkpoint.get('seed')!r}"
            )
        source_paths[policy] = checkpoint_path
        source_dirs[policy] = source_dir

    contract = {
        "schema_version": "advanced-time-label-ablation-v1",
        "status": (
            "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION"
            if int(args.max_test_shots) == 360 else "SMOKE_ONLY"
        ),
        "seed": int(args.seed),
        "matrix": comparison_arms(),
        "label_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
        "jtext_dynamic_formula": "max(25 ms, AdvancedTime_ms); missing->25 ms",
        "east_dynamic_formula": "max(125 ms, AdvancedTime_ms); missing->125 ms",
        "advanced_time_unit": "milliseconds as stored; no x1000 conversion",
        "jtext_advanced_time_sha256": sha256_file(args.jtext_advanced_time),
        "east_advanced_time_sha256": sha256_file(args.east_advanced_time),
        "balanced360_role": "development stream; not independent confirmation",
        "source_training_commands": commands,
        "source_artifacts_reused": bool(reuse_root is not None),
        "inference_batch_size": int(args.inference_batch_size),
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    _write_json(output / "experiment_contract.json", contract)

    results: dict[str, dict[str, Any]] = {}
    for name, source_policy, east_policy in comparison_arms():
        source_dir = source_dirs[source_policy]
        results[name] = _run_target_arm(
            name=name, source_policy=source_policy, east_policy=east_policy,
            source_checkpoint=source_paths[source_policy],
            source_test_metrics=source_dir / "test220_metrics.json",
            seed=args.seed, data_root=data_root, output=target_root / name, args=args,
        )
        torch.cuda.empty_cache()
    summary = {
        "schema_version": "advanced-time-label-ablation-result-v1",
        "status": (
            "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION"
            if int(args.max_test_shots) == 360 else "SMOKE_ONLY"
        ),
        "seed": int(args.seed),
        "arms": results,
        "support_selected_factorial_effects": _effect_decomposition(results),
        "elapsed_seconds": time.time() - started,
        "next_gate": (
            "Run seeds 133/137 only if seed131 remains normal and the dynamic source "
            "effect is directionally useful."
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--expected-python", type=Path,
        default=Path("/home/hetaisheng/miniconda3/envs/rl_env/bin/python"),
    )
    parser.add_argument("--jtext-advanced-time", type=Path, default=JTEXT_ADVANCED_TIME)
    parser.add_argument("--east-advanced-time", type=Path, default=EAST_ADVANCED_TIME)
    parser.add_argument("--source-epochs", type=int, default=32)
    parser.add_argument("--source-minimum-epochs", type=int, default=32)
    parser.add_argument("--source-patience", type=int, default=32)
    parser.add_argument("--source-steps-per-epoch", type=int, default=128)
    parser.add_argument("--source-batch-size", type=int, default=128)
    parser.add_argument("--source-learning-rate", type=float, default=3e-4)
    parser.add_argument("--source-target-tpr", type=float, default=0.90)
    parser.add_argument("--target-epochs", type=int, default=12)
    parser.add_argument("--target-batch-size", type=int, default=128)
    parser.add_argument("--target-tpr", type=float, default=0.80)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument(
        "--reuse-source-root", type=Path,
        help="reuse completed fixed/dynamic source directories after an interrupted target run",
    )
    parser.add_argument(
        "--max-test-shots", type=int, choices=range(2, 361), default=360,
        help="smoke only when below 360; full evidence always uses 360",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    compact = {
        "status": result["status"],
        "seed": result["seed"],
        "effects": result["support_selected_factorial_effects"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
