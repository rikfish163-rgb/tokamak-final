#!/usr/bin/env python3
"""Run block-budget-matched causal random-query online controls.

The random schedules are derived only from the active arm's sealed query
decision positions.  They match the active query count inside every 20-shot
block and obey the same rolling token budget.  Outcomes are never read while
the schedules are constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
CONFIG = PACKAGE / "configs/primary_chain_v3c.json"
PRIMARY = PACKAGE / "evidence/primary_online_artifacts"
SEALED_PARENTS = PACKAGE / "evidence/equal_label_budget_frontload_v2"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ablation = _load_module("online_timescale_ablation", HERE / "online_timescale_ablation.py")
reproduce = _load_module("paper_reproduce_primary_chain", Path(__file__).with_name("reproduce_primary_chain.py"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_manifests(*, seed: int, output: Path, replicates: int) -> list[Path]:
    contract = json.loads(CONFIG.read_text())
    online = contract["online"]
    source = PRIMARY / f"seed{seed}"
    static_path = source / "static_predictions.jsonl"
    decisions_path = source / "query_decisions.jsonl"
    roster = [int(row["shot"]) for row in read_jsonl(static_path)]
    reference = ablation.fixed_query_manifest_from_decisions(
        read_jsonl(decisions_path), roster=roster,
    )
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for replicate in range(int(replicates)):
        manifest = ablation.build_causal_random_query_manifest(
            roster=roster,
            reference_manifest=reference,
            block_size=int(online["query_block_size"]),
            query_rate=float(online["query_quota"]) / float(online["query_block_size"]),
            token_capacity=float(online["query_token_capacity"]),
            seed=int(seed),
            replicate=replicate,
        )
        manifest["source"] = {
            "active_query_decisions_sha256": sha256_file(decisions_path),
            "static_predictions_sha256": sha256_file(static_path),
            "active_query_count": len(reference["shots"]),
            "selection_inputs": "stream positions and active per-block counts only",
        }
        path = output / f"seed{seed}_random_query_r{replicate}.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise FileExistsError(f"refusing to overwrite a different manifest: {path}")
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths


def run_command(command: list[str], *, environment: dict[str, str]) -> None:
    print("[RANDOM-QUERY] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[131, 133, 137])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--stage", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.replicates <= 0:
        raise ValueError("replicates must be positive")
    environment = dict(os.environ)
    environment["FUSION_PROJECT_ROOT"] = str(ROOT)
    ledger: dict[str, Any] = {
        "schema_version": "causal-random-query-online-baseline-v1",
        "claim_status": "BALANCED360_METHOD_DEVELOPMENT_DIAGNOSTIC",
        "seeds": [int(seed) for seed in args.seeds],
        "replicates": int(args.replicates),
        "commands": [],
    }
    for seed in args.seeds:
        seed_root = output_root / f"seed{seed}"
        manifests = build_manifests(
            seed=int(seed), output=output_root / "manifests",
            replicates=int(args.replicates),
        )
        plan = reproduce.command_plan(
            seed=int(seed), output_root=output_root, retrain_source=False,
        )
        target_commands: list[list[str]] = []
        sealed_parent = (
            SEALED_PARENTS / f"seed{seed}/support_only/frontloaded_target_state.pt"
        )
        fidelity = sealed_parent.parent / "parent_fidelity.json"
        if not sealed_parent.is_file() or not fidelity.is_file():
            raise FileNotFoundError(
                f"seed{seed}: archived fidelity-verified parent is missing"
            )
        online_commands: list[list[str]] = []
        for replicate, manifest in enumerate(manifests):
            command = list(plan["online"])
            out_index = command.index("--out") + 1
            command[out_index] = str(seed_root / f"causal_random_query_r{replicate}")
            expected_parent = str(sealed_parent)
            expected_roster = str(
                PRIMARY / f"seed{seed}/static_predictions.jsonl"
            )
            command[command.index("--sealed-parent-state") + 1] = expected_parent
            command[command.index("--roster-manifest") + 1] = expected_roster
            command.extend(("--fixed-query-manifest", str(manifest)))
            online_commands.append(command)
        ledger["commands"].append({
            "seed": int(seed),
            "target": target_commands,
            "online": online_commands,
        })
        if args.stage == "run":
            for command in target_commands:
                out_path = Path(command[command.index("--out") + 1])
                if not (out_path / "summary.json").is_file():
                    run_command(command, environment=environment)
            for command in online_commands:
                out_path = Path(command[command.index("--out") + 1])
                if not (out_path / "summary.json").is_file():
                    run_command(command, environment=environment)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_plan.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({
        "output_root": str(output_root),
        "stage": args.stage,
        "planned_online_runs": len(args.seeds) * int(args.replicates),
    }, indent=2))


if __name__ == "__main__":
    main()
