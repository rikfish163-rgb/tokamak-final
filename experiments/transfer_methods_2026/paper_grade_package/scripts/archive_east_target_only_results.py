#!/usr/bin/env python3
"""Create a compact, hash-checked archive of EAST target-only baselines."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ARMS = ("target_only_local17", "target_only_matched17")
SEEDS = (131, 133, 137)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def compress(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "decompressed_sha256": sha256_file(source),
        "decompressed_bytes": source.stat().st_size,
    }


def archive(source_root: Path, output: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    artifacts: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics[arm] = {}
        for seed in SEEDS:
            run = source_root / arm / f"seed{seed}"
            summary = json.loads((run / "summary.json").read_text())
            lineage = json.loads((run / "lineage.json").read_text())
            if summary["arm"] != arm or int(summary["seed"]) != seed:
                raise RuntimeError(f"{arm}/seed{seed}: summary identity mismatch")
            if lineage["arm"] != arm or int(lineage["seed"]) != seed:
                raise RuntimeError(f"{arm}/seed{seed}: lineage identity mismatch")
            if summary["lineage_sha256"] != sha256_file(run / "lineage.json"):
                raise RuntimeError(f"{arm}/seed{seed}: lineage hash mismatch")
            if summary["prediction_ledger_sha256"] != sha256_file(
                run / "balanced360_predictions.jsonl"
            ):
                raise RuntimeError(f"{arm}/seed{seed}: prediction hash mismatch")
            relative = Path(arm) / f"seed{seed}"
            for filename in (
                "summary.json", "metrics.json", "lineage.json",
                "support_threshold.json", "training_ledger_summary.json",
                "balanced360_predictions.jsonl", "target_only_state.pt",
            ):
                row = copy(run / filename, output / relative / filename)
                row["path"] = str(relative / filename)
                artifacts.append(row)
            row = compress(
                run / "training_ledger.jsonl",
                output / relative / "training_ledger.jsonl.gz",
            )
            row["path"] = str(relative / "training_ledger.jsonl.gz")
            artifacts.append(row)
            metrics[arm][str(seed)] = summary["metrics"]
    payload = {
        "schema_version": "east-target-only-compact-archive-v1",
        "claim_boundary": "TEST_DRIVEN_EXPLORATION",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "metrics": metrics,
        "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        "omitted": [
            "risk_traces (regenerable and high volume)",
            "support_scores (regenerable from the sealed state and support roster)",
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = archive(args.source_root, args.output)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "runs": sum(len(rows) for rows in payload["metrics"].values()),
        "artifacts": len(payload["artifacts"]),
    }, indent=2))


if __name__ == "__main__":
    main()
