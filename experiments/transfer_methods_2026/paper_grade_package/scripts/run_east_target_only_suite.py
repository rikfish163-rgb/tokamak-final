#!/usr/bin/env python3
"""Plan or execute the frozen two-arm EAST target-only baseline suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "experiments/transfer_methods_2026/run_east_target_only_baseline.py"
ARMS = ("target_only_local17", "target_only_matched17")
SEEDS = (131, 133, 137)


def command(*, arm: str, seed: int, data_root: Path, output_root: Path) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--arm", str(arm),
        "--seed", str(int(seed)),
        "--data-root", str(data_root),
        "--out", str(output_root / arm / f"seed{int(seed)}"),
        "--device", "cuda",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")),
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--stage", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    commands = [
        command(
            arm=str(arm), seed=int(seed), data_root=data_root,
            output_root=output_root,
        )
        for arm in args.arms for seed in args.seeds
    ]
    serialized_commands: list[list[str]] = []
    for current in commands:
        serialized = list(current)
        serialized[serialized.index("--data-root") + 1] = "<external-data-root>"
        serialized_commands.append(serialized)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "east-target-only-suite-plan-v1",
        "claim_boundary": "TEST_DRIVEN_EXPLORATION",
        "data_root": "<external-data-root>",
        "data_root_serialized": False,
        "commands": serialized_commands,
    }
    (output_root / "run_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8",
    )
    if args.stage == "run":
        environment = dict(os.environ)
        environment["FUSION_DATA_ROOT"] = str(data_root)
        for current in commands:
            out = Path(current[current.index("--out") + 1])
            if (out / "summary.json").is_file():
                continue
            subprocess.run(current, cwd=ROOT, env=environment, check=True)
    print(json.dumps({
        "stage": args.stage,
        "planned_runs": len(commands),
        "output_root": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
