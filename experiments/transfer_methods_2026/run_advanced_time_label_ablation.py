#!/usr/bin/env python3
"""Run a label-only 2x2 ablation through the complete frozen static chain.

The fixed/fixed arm must reproduce the archived seed-specific source model,
support-only target parent, threshold, and every balanced360 alarm decision.
Only after that fail-closed control passes may the other three arms be reported.

The sole interventions are the positive-window upper boundaries:

* J-TEXT: fixed 125 ms versus max(25 ms, AdvancedTime), missing -> 25 ms;
* EAST: fixed 125 ms versus max(125 ms, AdvancedTime), missing -> 125 ms.

HDF5 meta/IsDisrupt and meta/DownTime remain the only outcome authorities.
All target arms retain the historical shot-first sampler, top-block adaptation,
shared hazard adapter, EAST-private raw8 residual, support110 threshold, and
dual-threshold physics/trend alarm state machine used by the archived mainline.
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
PACKAGE = HERE / "paper_grade_package"
SOURCE_TRAINER = HERE / "train_jtext_source_transformer.py"
JTEXT_ADVANCED_TIME = ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"
EAST_ADVANCED_TIME = ROOT / "shot_list/EAST/AdvancedTime_EAST.json"
FULL_STATIC_PIPELINE = (
    "encoder-head+shared-hazard+east-private-raw8+dual-threshold-v1"
)
FIXED_CONTROL_BY_SEED = {
    131: {"tpr_at_30ms": 148 / 180, "fpr": 28 / 180},
    133: {"tpr_at_30ms": 151 / 180, "fpr": 26 / 180},
    137: {"tpr_at_30ms": 155 / 180, "fpr": 36 / 180},
}
SOURCE_CONFIG = {
    "epochs": 30,
    "minimum_epochs": 12,
    "patience": 7,
    "steps_per_epoch": 128,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "target_tpr": 0.90,
}


def comparison_arms() -> tuple[tuple[str, str, str], ...]:
    return (
        ("source_fixed__east_fixed", "fixed125", "fixed125"),
        ("source_dynamic__east_fixed", "advanced_floor", "fixed125"),
        ("source_fixed__east_dynamic", "fixed125", "advanced_floor"),
        ("source_dynamic__east_dynamic", "advanced_floor", "advanced_floor"),
    )


def assert_fixed_control(metrics: Mapping[str, Any], *, seed: int) -> None:
    """Fail when the control is not the archived full static operating point."""

    expected = FIXED_CONTROL_BY_SEED[int(seed)]
    actual_tpr = float(metrics["tpr_at_30ms"])
    actual_fpr = float(metrics["fpr"])
    if not (
        math.isclose(
            actual_tpr, float(expected["tpr_at_30ms"]),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            actual_fpr, float(expected["fpr"]), rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            f"seed {seed} fixed/fixed control does not reproduce the archived "
            "full static chain: "
            f"TPR@30={actual_tpr:.10f}, FPR={actual_fpr:.10f}; "
            f"expected {float(expected['tpr_at_30ms']):.10f}/"
            f"{float(expected['fpr']):.10f}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source_command(
    *, seed: int, policy: str, output: Path, advanced_time: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SOURCE_TRAINER),
        "--seed", str(seed),
        "--output", str(output),
        "--device", "cuda",
        "--epochs", str(SOURCE_CONFIG["epochs"]),
        "--minimum-epochs", str(SOURCE_CONFIG["minimum_epochs"]),
        "--patience", str(SOURCE_CONFIG["patience"]),
        "--steps-per-epoch", str(SOURCE_CONFIG["steps_per_epoch"]),
        "--batch-size", str(SOURCE_CONFIG["batch_size"]),
        "--learning-rate", str(SOURCE_CONFIG["learning_rate"]),
        "--target-tpr", str(SOURCE_CONFIG["target_tpr"]),
        "--label-policy", policy,
        "--advanced-time-path", str(advanced_time),
    ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prediction_control_mismatches(
    records: Sequence[Mapping[str, Any]], archived: Sequence[Mapping[str, Any]],
) -> list[str]:
    if len(records) != len(archived):
        return [f"row count {len(records)} != {len(archived)}"]
    failures: list[str] = []
    for current, expected in zip(records, archived):
        decision = current["primary"]
        shot = int(current["shot"])
        for key in ("shot", "stream_index"):
            if current[key] != expected[key]:
                failures.append(f"shot {shot}: {key} mismatch")
        for key in ("alarmed", "alarm_index", "route", "early_score", "alarm_time_s"):
            left = decision.get(key)
            right = expected.get(key)
            if isinstance(left, float) and isinstance(right, float):
                equal = math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            else:
                equal = left == right
            if not equal:
                failures.append(f"shot {shot}: {key} mismatch")
        if failures:
            break
    return failures


def _factorial_effects(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ff = results["source_fixed__east_fixed"]["metrics"]
    df = results["source_dynamic__east_fixed"]["metrics"]
    fd = results["source_fixed__east_dynamic"]["metrics"]
    dd = results["source_dynamic__east_dynamic"]["metrics"]
    output: dict[str, Any] = {}
    for metric in ("tpr_at_30ms", "fpr", "tpr", "auc", "utility_tpr30_minus_fpr"):
        values = [float(row[metric]) for row in (ff, df, fd, dd)]
        output[metric] = {
            "jtext_effect_when_east_fixed": values[1] - values[0],
            "jtext_effect_when_east_dynamic": values[3] - values[2],
            "east_effect_when_jtext_fixed": values[2] - values[0],
            "east_effect_when_jtext_dynamic": values[3] - values[1],
            "interaction": values[3] - values[1] - values[2] + values[0],
        }
    return output


def add_operational_confusion_counts(
    metrics: dict[str, Any], *, disruptions: int,
) -> None:
    """Annotate raw-alarm counts with the lead-qualified EAST counts."""

    disruptions = int(disruptions)
    timely_true_positive = int(round(float(metrics["tpr_at_30ms"]) * disruptions))
    if not 0 <= timely_true_positive <= disruptions:
        raise ValueError("lead-qualified true-positive count is outside the cohort")
    metrics["raw_alarm_TP"] = int(metrics["TP"])
    metrics["raw_alarm_FN"] = int(metrics["FN"])
    metrics["timely_TP_at_30ms"] = timely_true_positive
    metrics["timely_FN_at_30ms"] = disruptions - timely_true_positive
    metrics["confusion_count_semantics"] = (
        "TP/FN and raw_alarm_TP/FN count any alarm occurrence; "
        "timely_TP_at_30ms/timely_FN_at_30ms enforce lead>=30ms"
    )


def _run_target_arm(
    *, name: str, source_policy: str, east_policy: str,
    source_checkpoint: Path, seed: int, output: Path,
    east_advanced_time: Path, modules: Mapping[str, Any],
) -> dict[str, Any]:
    ATL = modules["ATL"]
    grid = modules["grid"]
    frontload = modules["frontload"]
    a1 = modules["a1"]
    auxiliary = modules["auxiliary"]
    unified = modules["unified"]
    common = modules["common"]
    torch = modules["torch"]

    output.mkdir(parents=True)
    support_raw, unexpected_test = grid.load_east(include_test=False)
    if unexpected_test:
        raise RuntimeError("support-only loader unexpectedly returned test shots")
    advanced = (
        ATL.load_advanced_time_ms(east_advanced_time)
        if east_policy == "advanced_floor" else None
    )
    support_raw, label_audit = ATL.attach_positive_horizons(
        support_raw, device="EAST", policy=east_policy,
        advanced_time_ms=advanced,
    )
    _enc, _head, source_contract = grid.load_source_checkpoint(
        source_checkpoint, channels=len(grid.RH.REBUILD),
        expected_seed=seed, device="cpu",
    )
    center = source_contract["scaler_center"].cpu().numpy()
    scale = source_contract["scaler_scale"].cpu().numpy()
    del _enc, _head

    model = frontload.train_frontloaded_target(
        seed=seed,
        source_checkpoint=source_checkpoint,
        support_raw=support_raw,
        extra_raw=[],
        scaler_center=center,
        scaler_scale=scale,
    )
    model_hashes = {
        "encoder_head": a1.model_sha256(model["encoder"], model["head"]),
        "shared_adapter": auxiliary.module_sha256(model["shared"]),
        "private_adapter": auxiliary.module_sha256(model["private"]),
    }
    threshold = float(model["support_selected_threshold"])
    fidelity = {
        **model_hashes,
        "threshold": threshold,
    }
    if name == "source_fixed__east_fixed":
        parent_fidelity = json.loads((
            PACKAGE
            / f"evidence/equal_label_budget_frontload_v2/seed{seed}"
            / "support_only/parent_fidelity.json"
        ).read_text())
        if fidelity != parent_fidelity["expected"]:
            raise RuntimeError(
                "fixed/fixed target parent fidelity failed: "
                + canonical_json({"expected": parent_fidelity["expected"], "actual": fidelity})
            )

    roster = sorted(int(value) for value in common.DataSpec().east["test_locked"])
    if len(roster) != 360:
        raise RuntimeError(f"expected balanced360, got {len(roster)} shots")
    records: list[dict[str, Any]] = []
    writer = a1.JsonlSealWriter(output / "balanced360_prediction_seal.jsonl")
    try:
        for index, shot in enumerate(roster):
            record = {
                "stream_index": index,
                **frontload.score_frontloaded_shot(
                    shot_id=shot,
                    model=model,
                    center=center,
                    scale=scale,
                    thresholds={"primary": threshold},
                ),
            }
            writer.append(record)
            records.append(record)
            if (index + 1) % 40 == 0:
                print(f"[{name}] sealed {index + 1}/360", flush=True)
    finally:
        writer.close()

    # The first balanced360 outcome read occurs only after all 360 predictions.
    outcomes = {shot: a1.read_outcome_after_seal(shot) for shot in roster}
    counts = {
        "disruptions": sum(bool(row["is_disrupt"]) for row in outcomes.values()),
        "safe": sum(not bool(row["is_disrupt"]) for row in outcomes.values()),
    }
    if counts != {"disruptions": 180, "safe": 180}:
        raise RuntimeError(f"balanced360 HDF5 class counts changed: {counts}")
    metric_rows = frontload.rows_for_metric(records, "primary")
    metrics = unified.alarm_metrics(metric_rows, outcomes)
    add_operational_confusion_counts(metrics, disruptions=counts["disruptions"])
    labels = np.asarray([bool(outcomes[shot]["is_disrupt"]) for shot in roster])
    scores = np.asarray([float(row["combined_max_probability"]) for row in records])
    metrics["auc"] = float(grid.auc(labels.astype(int), scores))
    metrics["utility_tpr30_minus_fpr"] = (
        float(metrics["tpr_at_30ms"]) - float(metrics["fpr"])
    )

    prediction_fidelity = None
    if name == "source_fixed__east_fixed":
        assert_fixed_control(metrics, seed=seed)
        archived = _load_jsonl(
            PACKAGE / f"evidence/primary_online_artifacts/seed{seed}/static_predictions.jsonl"
        )
        mismatches = _prediction_control_mismatches(records, archived)
        if mismatches:
            raise RuntimeError("fixed/fixed prediction fidelity failed: " + mismatches[0])
        prediction_fidelity = {
            "passed": True,
            "rows_compared": len(records),
            "fields": [
                "shot", "stream_index", "alarmed", "alarm_index", "route",
                "early_score", "alarm_time_s",
            ],
        }

    torch.save({
        "schema_version": "advanced-time-full-static-target-v2",
        "seed": seed,
        "arm": name,
        "source_policy": source_policy,
        "east_policy": east_policy,
        "encoder": model["encoder"].state_dict(),
        "head": model["head"].state_dict(),
        "shared": model["shared"].state_dict(),
        "private": model["private"].state_dict(),
        "threshold": threshold,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
    }, output / "target_state.pt")
    result = {
        "schema_version": "advanced-time-full-static-arm-v2",
        "status": "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION",
        "arm": name,
        "seed": seed,
        "pipeline": FULL_STATIC_PIPELINE,
        "source_label_policy": source_policy,
        "east_label_policy": east_policy,
        "east_label_audit": label_audit,
        "threshold": {
            "value": threshold,
            "source": "EAST support110 HDF5 outcomes only",
            "target_tpr": 0.80,
            "balanced360_used": False,
        },
        "metrics": metrics,
        "model_hashes": model_hashes,
        "target_parent_fidelity": fidelity,
        "prediction_fidelity": prediction_fidelity,
        "prediction_seal_sha256": sha256_file(
            output / "balanced360_prediction_seal.jsonl"
        ),
        "target_state_sha256": sha256_file(output / "target_state.pt"),
        "balanced360_outcomes_read_after_complete_prediction_seal": True,
        "balanced360_used_for_training_or_threshold": False,
        "normality": {
            "abnormal": bool(
                float(metrics["tpr_at_30ms"]) < 0.60
                or float(metrics["fpr"]) > 0.40
            ),
            "rule": "TPR@30ms<0.60 or FPR>0.40",
        },
    }
    _write_json(output / "summary.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    (output / "targets").mkdir()

    os.environ["FUSION_PROJECT_ROOT"] = str(ROOT)
    os.environ["FUSION_DATA_ROOT"] = str(data_root)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import torch
    import advanced_time_labels as ATL
    import common
    import grid_search as grid
    import online_auxiliary as auxiliary
    import run_equal_label_budget_ablation as frontload
    import run_online_active_a1 as a1
    import run_unified_online_active as unified

    if not torch.cuda.is_available() or str(args.device) != "cuda":
        raise RuntimeError("registered ablation requires rl_env CUDA")
    if Path(sys.executable).resolve() != Path(args.expected_python).expanduser().resolve():
        raise RuntimeError(
            f"wrong interpreter {sys.executable}; expected {args.expected_python}"
        )
    torch.set_float32_matmul_precision("highest")
    started = time.time()
    environment = dict(os.environ)

    fixed_source = (
        PACKAGE / f"evidence/source_lineage/seed{args.seed}/source_checkpoint.pt"
    )
    if not fixed_source.is_file():
        raise FileNotFoundError(fixed_source)
    dynamic_source_dir = (
        Path(args.dynamic_source_dir).expanduser().resolve()
        if args.dynamic_source_dir is not None
        else output / "sources/dynamic"
    )
    source_command = None
    if args.dynamic_source_dir is None:
        source_command = _source_command(
            seed=args.seed,
            policy="advanced_floor",
            output=dynamic_source_dir,
            advanced_time=Path(args.jtext_advanced_time).expanduser().resolve(),
        )
        subprocess.run(source_command, check=True, cwd=ROOT, env=environment)
    dynamic_source = dynamic_source_dir / "source_checkpoint.pt"
    if not dynamic_source.is_file():
        raise FileNotFoundError(dynamic_source)

    fixed_enc, fixed_head, fixed_contract = grid.load_source_checkpoint(
        fixed_source, channels=len(grid.RH.REBUILD),
        expected_seed=args.seed, device="cpu",
    )
    dynamic_enc, dynamic_head, dynamic_contract = grid.load_source_checkpoint(
        dynamic_source, channels=len(grid.RH.REBUILD),
        expected_seed=args.seed, device="cpu",
    )
    fixed_policy = fixed_contract.get("training_label", {}).get("policy", "fixed125")
    dynamic_policy = dynamic_contract.get("training_label", {}).get("policy")
    if fixed_policy != "fixed125" or dynamic_policy != "advanced_floor":
        raise RuntimeError(
            f"source label contract mismatch: fixed={fixed_policy}, dynamic={dynamic_policy}"
        )
    source_hashes = {
        "fixed": a1.model_sha256(fixed_enc, fixed_head),
        "dynamic": a1.model_sha256(dynamic_enc, dynamic_head),
    }
    del fixed_enc, fixed_head, dynamic_enc, dynamic_head

    modules = {
        "torch": torch,
        "ATL": ATL,
        "common": common,
        "grid": grid,
        "auxiliary": auxiliary,
        "frontload": frontload,
        "a1": a1,
        "unified": unified,
    }
    source_paths = {"fixed125": fixed_source, "advanced_floor": dynamic_source}
    results: dict[str, dict[str, Any]] = {}
    for name, source_policy, east_policy in comparison_arms():
        results[name] = _run_target_arm(
            name=name,
            source_policy=source_policy,
            east_policy=east_policy,
            source_checkpoint=source_paths[source_policy],
            seed=args.seed,
            output=output / "targets" / name,
            east_advanced_time=Path(args.east_advanced_time).expanduser().resolve(),
            modules=modules,
        )

    summary = {
        "schema_version": "advanced-time-label-ablation-result-v2",
        "status": "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION",
        "seed": args.seed,
        "pipeline": FULL_STATIC_PIPELINE,
        "correction": {
            "supersedes": "advanced-time-label-ablation-result-v1",
            "reason": (
                "v1 evaluated a bare encoder/head with a single threshold and did not "
                "run the archived shared/private/dual-threshold static chain"
            ),
        },
        "label_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
        "source_config": SOURCE_CONFIG,
        "source_command": source_command,
        "source_checkpoints": {
            "fixed": str(fixed_source.relative_to(ROOT)),
            "dynamic": (
                str(dynamic_source.relative_to(ROOT))
                if dynamic_source.is_relative_to(ROOT) else str(dynamic_source)
            ),
        },
        "source_model_hashes": source_hashes,
        "jtext_advanced_time_sha256": sha256_file(
            Path(args.jtext_advanced_time).expanduser().resolve()
        ),
        "east_advanced_time_sha256": sha256_file(
            Path(args.east_advanced_time).expanduser().resolve()
        ),
        "arms": results,
        "factorial_effects": _factorial_effects(results),
        "fixed_control_passed": True,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "balanced360 is a development comparator and not independent future evidence"
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(131, 133, 137), default=131)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--expected-python", type=Path,
        default=Path("/home/hetaisheng/miniconda3/envs/rl_env/bin/python"),
    )
    parser.add_argument("--jtext-advanced-time", type=Path, default=JTEXT_ADVANCED_TIME)
    parser.add_argument("--east-advanced-time", type=Path, default=EAST_ADVANCED_TIME)
    parser.add_argument(
        "--dynamic-source-dir", type=Path,
        help="reuse a completed current-mainline dynamic source directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps({
        "status": result["status"],
        "seed": result["seed"],
        "pipeline": result["pipeline"],
        "metrics": {
            name: arm["metrics"] for name, arm in result["arms"].items()
        },
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
