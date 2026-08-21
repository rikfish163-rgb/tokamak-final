#!/usr/bin/env python3
"""Validate and archive a regenerated fixed-shape selector-control bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile


SEEDS = (131, 133, 137)
ARMS = (
    "active_fixed_compute",
    "random_fixed_compute_r0",
    "random_fixed_compute_r1",
    "random_fixed_compute_r2",
)
RUN_FILES = (
    "final_online_state.pt",
    "predictions.jsonl",
    "query_decisions.jsonl",
    "reveals.jsonl",
    "summary.json",
    "updates.jsonl",
)


def required_relative_paths() -> tuple[Path, ...]:
    paths = [Path("run_plan.json")]
    for seed in SEEDS:
        paths.append(Path("manifests") / f"seed{seed}_archived_active.json")
        paths.extend(
            Path("manifests") / f"seed{seed}_random_query_r{replicate}.json"
            for replicate in range(3)
        )
        for arm in ARMS:
            paths.extend(Path(f"seed{seed}") / arm / name for name in RUN_FILES)
    return tuple(paths)


def validate_source(source: Path) -> None:
    missing = [str(path) for path in required_relative_paths()
               if not (source / path).is_file()]
    if missing:
        raise FileNotFoundError(
            "incomplete equal-compute run bundle; missing: " + ", ".join(missing)
        )
    plan = json.loads((source / "run_plan.json").read_text())
    if plan.get("seeds") != list(SEEDS) or int(plan.get("replicates", -1)) != 3:
        raise ValueError("run plan is not the registered 3-seed x 3-replicate design")
    if len(plan.get("commands", [])) != len(SEEDS):
        raise ValueError("run plan does not contain all seed command groups")
    for seed in SEEDS:
        for arm in ARMS:
            run = source / f"seed{seed}" / arm
            summary = json.loads((run / "summary.json").read_text())
            predictions = [line for line in (run / "predictions.jsonl").read_text().splitlines()
                           if line.strip()]
            decisions = [line for line in (run / "query_decisions.jsonl").read_text().splitlines()
                         if line.strip()]
            if len(predictions) != 360 or len(decisions) != 360:
                raise ValueError(f"{seed}/{arm} does not contain 360 prediction/decision rows")
            if int(summary.get("shot_count", 360)) != 360:
                raise ValueError(f"{seed}/{arm} summary shot count is not 360")


def archive(source: Path, destination: Path, *, replace: bool) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    validate_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-archive-", dir=destination.parent,
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.mkdir()
        for relative in required_relative_paths():
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        validate_source(staged)
        if destination.exists():
            if not replace:
                raise FileExistsError(
                    f"destination exists; pass --replace after reviewing: {destination}"
                )
            backup = destination.with_name(destination.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"refusing to overwrite recovery backup: {backup}")
            destination.rename(backup)
            try:
                shutil.copytree(staged, destination)
            except Exception:
                if destination.exists():
                    shutil.rmtree(destination)
                backup.rename(destination)
                raise
            shutil.rmtree(backup)
        else:
            shutil.copytree(staged, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    archive(args.source_root, args.destination_root, replace=args.replace)
    print(json.dumps({
        "source": str(args.source_root.resolve()),
        "destination": str(args.destination_root.resolve()),
        "archived_files": len(required_relative_paths()),
    }, indent=2))


if __name__ == "__main__":
    main()
