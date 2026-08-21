#!/usr/bin/env python3
"""Plan or execute the frozen three-stage source/target/online chain.

The wrapper contains no metric selection.  It translates the registered JSON
configuration into the exact source, support-adaptation, target-adapter, and
chronological v3c commands.  Existing output directories fail closed.
"""

from __future__ import annotations

import argparse
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
SEALED_PARENTS = PACKAGE / "evidence/equal_label_budget_frontload_v2"
PRIMARY = PACKAGE / "evidence/primary_online_artifacts"


def option_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def options(payload: dict[str, Any]) -> list[str]:
    arguments: list[str] = []
    for name, value in payload.items():
        if isinstance(value, bool):
            if value:
                arguments.append(option_name(name))
        else:
            arguments.extend((option_name(name), str(value)))
    return arguments


def command_plan(
    *, seed: int, output_root: Path, retrain_source: bool,
    rebuild_target: bool = False,
) -> dict[str, list[str]]:
    contract = json.loads(CONFIG.read_text())
    if seed not in [int(value) for value in contract["seeds"]]:
        raise ValueError(f"unregistered seed: {seed}")
    run_root = output_root / f"seed{seed}"
    source_dir = run_root / "source"
    bundled_source = PACKAGE / f"evidence/source_lineage/seed{seed}/source_checkpoint.pt"
    source_checkpoint = source_dir / "source_checkpoint.pt" if retrain_source else bundled_source
    a1_dir = run_root / "a1"
    adapter_dir = run_root / "dynamic_hazard_adapter"
    online_dir = run_root / "final_episode_v3"

    source = [
        sys.executable, str(HERE / "train_jtext_source_transformer.py"),
        "--seed", str(seed), "--output", str(source_dir), "--device", "cuda",
        *options(contract["source"]),
    ]
    a1 = [
        sys.executable, str(HERE / "run_online_active_a1.py"),
        "--seed", str(seed), "--out", str(a1_dir),
        "--source-checkpoint", str(source_checkpoint),
        *options(contract["a1"]),
    ]
    adapter = [
        sys.executable, str(HERE / "run_dynamic_hazard_adapter.py"),
        "--seed", str(seed), "--a1", str(a1_dir), "--out", str(adapter_dir),
        "--source-checkpoint", str(source_checkpoint),
    ]
    online = [
        sys.executable, str(HERE / "run_unified_online_active.py"),
        "--seed", str(seed), "--out", str(online_dir),
        "--source-checkpoint", str(source_checkpoint),
        "--a1-dir", str(a1_dir), "--adapter-dir", str(adapter_dir),
        *options(contract["online"]),
    ]
    if not rebuild_target:
        a1 = []
        adapter = []
        online.extend((
            "--sealed-parent-state",
            str(
                SEALED_PARENTS
                / f"seed{seed}/support_only/frontloaded_target_state.pt"
            ),
            "--roster-manifest",
            str(PRIMARY / f"seed{seed}/static_predictions.jsonl"),
        ))
    return {"source": source, "a1": a1, "adapter": adapter, "online": online}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(131, 133, 137), required=True)
    parser.add_argument(
        "--stage", choices=("plan", "source", "target", "online", "all"),
        default="plan",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--retrain-source", action="store_true",
        help="train source from HDF5 instead of reusing the hash-bound source checkpoint",
    )
    parser.add_argument(
        "--rebuild-target-current-sampler", action="store_true",
        help=(
            "diagnostic only: rebuild A1/adapter with the current sampler "
            "instead of loading the fidelity-verified archived target parent"
        ),
    )
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    plan = command_plan(
        seed=args.seed, output_root=output_root,
        retrain_source=bool(args.retrain_source),
        rebuild_target=bool(args.rebuild_target_current_sampler),
    )
    print(json.dumps(plan, indent=2))
    if args.stage == "plan":
        return

    stages: list[str] = []
    if args.stage in ("source", "all") and args.retrain_source:
        stages.append("source")
    if args.stage in ("target", "all") and args.rebuild_target_current_sampler:
        stages.extend(("a1", "adapter"))
    if args.stage in ("online", "all"):
        stages.append("online")
    if args.stage == "source" and not args.retrain_source:
        raise ValueError("--stage source requires --retrain-source")
    if args.stage == "target" and not args.rebuild_target_current_sampler:
        raise ValueError(
            "the primary target parent is already archived; pass "
            "--rebuild-target-current-sampler only for a diagnostic rebuild"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["FUSION_PROJECT_ROOT"] = str(ROOT)
    for stage in stages:
        subprocess.run(plan[stage], check=True, cwd=ROOT, env=environment)


if __name__ == "__main__":
    main()
