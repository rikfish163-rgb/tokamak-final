from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from reproducibility.data_contract import audit_data
from reproducibility.paths import data_root, project_root
from reproducibility.verify_results import compare_multiseed_results


SEEDS = (131, 133, 137)


def build_seed_commands(
    *, seed: int, output_root: Path, python_executable: str = sys.executable,
) -> list[list[str]]:
    root = project_root()
    methods = root / "experiments/transfer_methods_2026"
    seed_root = Path(output_root) / f"seed{seed}"
    a1 = seed_root / "a1"
    adapter = seed_root / "adapter"
    final = seed_root / "final"
    return [
        [
            python_executable, str(methods / "run_online_active_a1.py"),
            "--seed", str(seed), "--out", str(a1),
        ],
        [
            python_executable, str(methods / "run_dynamic_hazard_adapter.py"),
            "--seed", str(seed), "--a1", str(a1), "--out", str(adapter),
        ],
        [
            python_executable, str(methods / "run_unified_online_active.py"),
            "--locked-best-v8", "--seed", str(seed),
            "--a1-dir", str(a1), "--adapter-dir", str(adapter),
            "--out", str(final),
        ],
    ]


def _aggregate(output_root: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    rows = []
    summaries = []
    for seed in seeds:
        summary_path = output_root / f"seed{seed}/final/summary.json"
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        static = summary["results"]["static_unified"]
        online = summary["results"]["online_unified"]
        rows.append({
            "seed": seed,
            "query_count": int(summary["query_count"]),
            "status": summary["status"],
            "execution_protocol": summary["execution_protocol"],
            "static": static,
            "online": online,
            "delta": {
                "tp": int(online["TP"] - static["TP"]),
                "fp": int(online["FP"] - static["FP"]),
                "tpr": float(online["tpr"] - static["tpr"]),
                "fpr": float(online["fpr"] - static["fpr"]),
                "tpr_at_30ms": float(
                    online["tpr_at_30ms"] - static["tpr_at_30ms"]
                ),
                "median_lead_ms": float(
                    online["median_lead_ms"] - static["median_lead_ms"]
                ),
            },
        })
    return {
        "protocol": "static360_then_reset_online360_locked_best_v8",
        "seeds": rows,
        "all_exact_query_budget": all(row["query_count"] == 72 for row in rows),
        "all_phase_separated": all(
            row["execution_protocol"]["phase1_static_complete_before_phase2"]
            and row["execution_protocol"]["phase2_reset_to_same_migrated_initial_sha"]
            and row["execution_protocol"]["hyperparameters_locked_before_phase1"]
            and not row["execution_protocol"]["selection_on_360_in_this_run"]
            for row in rows
        ),
        "all_no_fpr_regression": all(
            row["online"]["fpr"] <= row["static"]["fpr"] for row in rows
        ),
        "all_online_normal": all(
            row["online"]["tpr"] >= 0.60 and row["online"]["fpr"] <= 0.40
            for row in rows
        ),
    }


def run_reproduction(
    *, output_root: Path, seeds: tuple[int, ...] = SEEDS,
    selected_data_root: Path | None = None, dry_run: bool = False,
) -> dict[str, Any]:
    root = project_root()
    selected_data = (selected_data_root or data_root()).resolve()
    audit = audit_data(selected_data, full=not dry_run)
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"reproduction output must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    commands = [
        command
        for seed in seeds
        for command in build_seed_commands(seed=seed, output_root=output_root)
    ]
    manifest = {
        "project_root": str(root),
        "data_audit": audit,
        "seeds": list(seeds),
        "commands": commands,
        "dry_run": bool(dry_run),
    }
    (output_root / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if dry_run:
        return manifest

    environment = dict(os.environ)
    environment["FUSION_PROJECT_ROOT"] = str(root)
    environment["FUSION_DATA_ROOT"] = str(selected_data)
    for seed in seeds:
        seed_root = output_root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        for stage_index, command in enumerate(
            build_seed_commands(seed=seed, output_root=output_root), start=1
        ):
            log_path = seed_root / f"stage{stage_index}.log"
            with log_path.open("x", encoding="utf-8") as log:
                subprocess.run(
                    command, cwd=root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )

    aggregate = _aggregate(output_root, seeds)
    aggregate_path = output_root / "multiseed_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    expected = json.loads((root / "results/reference/multiseed_summary.json").read_text())
    verification = compare_multiseed_results(aggregate, expected)
    (output_root / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    if not verification["passed"]:
        raise RuntimeError("reproduced metrics differ from committed reference")
    return {**manifest, "aggregate": aggregate, "verification": verification}
