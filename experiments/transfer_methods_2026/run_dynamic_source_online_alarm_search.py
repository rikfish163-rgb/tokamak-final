#!/usr/bin/env python3
"""Replay the selected dynamic-source EAST online method with one alarm seed.

This runner deliberately reuses the already sealed J-TEXT source checkpoint and
EAST support110 parent.  It does not retrain either stage.  The only permitted
method change is the four-dimensional initial state of the existing projected
alarm controller.  Every invocation still executes the complete static seal
before the chronological online balanced360 stream.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "paper_grade_package/configs"
DEFAULT_SEARCH_CONFIG = CONFIG_DIR / "dynamic_source_online_alarm_search_v1.json"
DEFAULT_PARENT_ROOT = Path("/home/hetaisheng/Fuison-runs/corrected_mainline_screen")
DEFAULT_OUTPUT_ROOT = Path(
    "/media/hetaisheng/044A81D94A81C83E/Fuison-dynamic-source-online-search/"
    "alarm_state_v1"
)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_selected_full_mainline as selected_runner  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_search_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dynamic-source-online-alarm-search-v1":
        raise ValueError("dynamic source alarm search schema mismatch")
    if payload.get("claim_status") != "TEST_DRIVEN_EXPLORATION":
        raise ValueError("balanced360 alarm selection must remain development-only")
    candidate = payload.get("selected_candidate")
    expected = {
        "initial_high_threshold",
        "initial_physics_weight",
        "initial_trend_weight",
        "initial_early_threshold",
    }
    if not isinstance(candidate, dict) or set(candidate) != expected:
        raise ValueError("selected alarm candidate is incomplete")
    return payload


def controller_parameters(candidate: dict[str, Any]) -> dict[str, float]:
    return {
        name.removeprefix("initial_"): float(value)
        for name, value in candidate.items()
    }


def resolve_seed_inputs(parent_root: Path, seed: int) -> dict[str, Path]:
    seed_root = parent_root / f"seed{int(seed)}"
    paths = {
        "source_checkpoint": seed_root / "source-train/source_checkpoint.pt",
        "parent_state": seed_root / "east-parent/east_fixed_parent.pt",
        "run_contract": seed_root / "run_contract.json",
        "roster_manifest": seed_root / "east-parent/balanced360_roster.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing sealed replay input: " + ", ".join(missing))
    return paths


def build_replay_command(
    *, seed: int, parent_root: Path, output: Path, python: Path,
    search_config_path: Path = DEFAULT_SEARCH_CONFIG,
) -> tuple[list[str], dict[str, Any]]:
    search = load_search_config(search_config_path)
    base_path = search_config_path.parent / str(search["base_config"])
    config = copy.deepcopy(selected_runner.load_selected_config(base_path))
    config["online"].update(search["selected_candidate"])
    paths = resolve_seed_inputs(parent_root, seed)
    command = selected_runner.unified_stage_command(
        stage="east-online", seed=seed, output=output,
        source_checkpoint=paths["source_checkpoint"],
        parent_state=paths["parent_state"],
        run_contract=paths["run_contract"],
        roster_manifest=paths["roster_manifest"],
        config=config, python=python,
    )
    receipt = {
        "seed": int(seed),
        "claim_status": search["claim_status"],
        "selection_stream": search["selection_stream"],
        "selected_candidate": search["selected_candidate"],
        "command": command,
        "reused_parent_artifacts": True,
        "source_training_repeated": False,
        "east_support_training_repeated": False,
        "input_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
    }
    return command, receipt


def metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary["results"]
    static = results["static_unified"]
    online = results["online_unified"]

    def compact(metric: dict[str, Any]) -> dict[str, Any]:
        tpr30 = float(metric["tpr_at_30ms"])
        fpr = float(metric["fpr"])
        return {
            "tpr_at_30ms": tpr30,
            "fpr": fpr,
            "j": tpr30 - fpr,
            "median_lead_ms": float(metric["median_lead_ms"]),
            "confusion": {
                name: int(metric[name]) for name in ("TP", "FN", "FP", "TN")
            },
        }

    static_view = compact(static)
    online_view = compact(online)
    return {
        "static": static_view,
        "online": online_view,
        "paired_delta": {
            "tpr_at_30ms": (
                online_view["tpr_at_30ms"] - static_view["tpr_at_30ms"]
            ),
            "fpr": online_view["fpr"] - static_view["fpr"],
            "j": online_view["j"] - static_view["j"],
        },
    }


def aggregate(seed_results: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    if not seed_results:
        raise ValueError("cannot aggregate an empty replay")
    count = len(seed_results)

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in seed_results:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / count

    deltas = [float(row["metrics"]["paired_delta"]["j"]) for row in seed_results]
    mean_online_tpr = mean(("metrics", "online", "tpr_at_30ms"))
    mean_online_fpr = mean(("metrics", "online", "fpr"))
    positive = sum(delta > 1e-12 for delta in deltas)
    nonnegative = all(
        delta >= float(gate["minimum_paired_delta_j_each_seed"]) - 1e-12
        for delta in deltas
    )
    passed = (
        count == len(gate["seeds"])
        and mean_online_tpr >= float(gate["minimum_mean_tpr30"])
        and mean_online_fpr <= float(gate["maximum_mean_fpr"])
        and nonnegative
        and positive >= int(gate["minimum_strictly_positive_seeds"])
    )
    return {
        "seed_count": count,
        "mean_static": {
            "tpr_at_30ms": mean(("metrics", "static", "tpr_at_30ms")),
            "fpr": mean(("metrics", "static", "fpr")),
            "j": mean(("metrics", "static", "j")),
        },
        "mean_online": {
            "tpr_at_30ms": mean_online_tpr,
            "fpr": mean_online_fpr,
            "j": mean(("metrics", "online", "j")),
        },
        "mean_paired_delta_j": sum(deltas) / count,
        "paired_delta_j_by_seed": deltas,
        "strictly_positive_seed_count": positive,
        "all_seeds_nonnegative": nonnegative,
        "gate_passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[131, 133, 137])
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-complete", action="store_true",
        help="reuse an existing seed only after validating its complete summary",
    )
    parser.add_argument(
        "--exec-single", action="store_true",
        help="replace this process with one unified replay to isolate CUDA lifetime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search = load_search_config(args.search_config)
    seeds = [int(seed) for seed in args.seeds]
    if any(seed not in search["true_replay_gate"]["seeds"] for seed in seeds):
        raise ValueError("requested seed is outside the registered replay gate")
    if args.exec_single and len(seeds) != 1:
        raise ValueError("--exec-single requires exactly one seed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    command_dir = args.output_root / "commands"
    command_dir.mkdir(exist_ok=True)
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        output = args.output_root / f"seed{seed}" / "east-online"
        command, receipt = build_replay_command(
            seed=seed, parent_root=args.parent_root, output=output,
            python=Path(sys.executable), search_config_path=args.search_config,
        )
        command_path = command_dir / f"seed{seed}.json"
        command_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        if args.exec_single:
            if output.exists():
                raise FileExistsError(
                    f"refusing to overwrite single replay output: {output}"
                )
            environment = dict(os.environ)
            if not environment.get("FUSION_DATA_ROOT"):
                raise RuntimeError("FUSION_DATA_ROOT must point to the frozen data root")
            os.execvpe(command[0], command, environment)
        summary_path = output / "summary.json"
        if output.exists():
            if not args.reuse_complete or not summary_path.is_file():
                raise FileExistsError(
                    f"refusing to overwrite or trust incomplete replay output: {output}"
                )
        else:
            environment = dict(os.environ)
            if not environment.get("FUSION_DATA_ROOT"):
                raise RuntimeError("FUSION_DATA_ROOT must point to the frozen data root")
            subprocess.run(command, check=True, cwd=HERE.parent.parent, env=environment)
        if not summary_path.is_file():
            raise RuntimeError(f"online replay did not produce {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["alarm_contract"].get("parameters_initial") != (
            controller_parameters(search["selected_candidate"])
        ):
            raise RuntimeError("executed alarm initialization differs from config")
        seed_results.append({
            "seed": seed,
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "metrics": metric_view(summary),
            "query_count": int(summary["query_count"]),
            "parameters_final": summary["alarm_contract"]["parameters_final"],
        })
        partial = {
            "schema_version": "dynamic-source-online-alarm-replay-v1",
            "claim_status": search["claim_status"],
            "selected_candidate": search["selected_candidate"],
            "seed_results": seed_results,
            "aggregate": aggregate(seed_results, {
                **search["true_replay_gate"], "seeds": seeds,
                "minimum_strictly_positive_seeds": len(seeds),
            }),
        }
        (args.output_root / "report.partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.dry_run:
        return
    report = {
        "schema_version": "dynamic-source-online-alarm-replay-v1",
        "claim_status": search["claim_status"],
        "independent_future_confirmation": False,
        "selected_candidate": search["selected_candidate"],
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results, search["true_replay_gate"]),
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
