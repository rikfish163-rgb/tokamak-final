#!/usr/bin/env python3
"""Create a compact, hash-verified archive of causal random-query controls."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


PACKAGE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE.parents[2]
PRIMARY = PACKAGE / "evidence/primary_online_artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def portable_value(value: Any) -> Any:
    """Replace machine-local paths while preserving artifact identity fields."""
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value
    path = Path(value)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        if "manifests" in path.parts:
            return (Path("manifests") / path.name).as_posix()
        return (Path("<external-run-artifact>") / path.name).as_posix()


def write_portable_json(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(portable_value(load(source)), indent=2, sort_keys=True) + "\n"
    )
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def copy_file(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def gzip_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "decompressed_bytes": source.stat().st_size,
        "decompressed_sha256": sha256_file(source),
    }


def archive(
    *, source_root: Path, output: Path, seeds: list[int], replicates: int,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    artifacts: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    manifest_root = source_root / "manifests"
    for seed in seeds:
        active = load(PRIMARY / f"seed{seed}/summary.json")
        seed_rows = []
        for replicate in range(replicates):
            run = source_root / f"seed{seed}/causal_random_query_r{replicate}"
            summary = load(run / "summary.json")
            for name in (
                "source_checkpoint", "base_before_after",
                "phase2_initial_private", "phase2_initial_shared",
                "static_predictions",
            ):
                if summary["hashes"][name] != active["hashes"][name]:
                    raise RuntimeError(
                        f"seed{seed} r{replicate}: lineage mismatch for {name}"
                    )
            if int(summary["query_count"]) != int(active["query_count"]):
                raise RuntimeError(f"seed{seed} r{replicate}: budget mismatch")
            relative = Path(f"seed{seed}/r{replicate}")
            expected_hashes = {
                "predictions.jsonl": summary["hashes"]["predictions"],
                "query_decisions.jsonl": summary["hashes"]["queries"],
                "reveals.jsonl": summary["hashes"]["reveals"],
            }
            for filename, expected in expected_hashes.items():
                source = run / filename
                if sha256_file(source) != expected:
                    raise RuntimeError(
                        f"seed{seed} r{replicate}: {filename} hash mismatch"
                    )
                row = copy_file(source, output / relative / filename)
                row["path"] = str(relative / filename)
                artifacts.append(row)
            row = write_portable_json(
                run / "summary.json", output / relative / "summary.json"
            )
            row["path"] = str(relative / "summary.json")
            artifacts.append(row)
            for filename in ("final_online_state.pt",):
                row = copy_file(run / filename, output / relative / filename)
                row["path"] = str(relative / filename)
                artifacts.append(row)
            update_hash = sha256_file(run / "updates.jsonl")
            if update_hash != summary["hashes"]["updates"]:
                raise RuntimeError(
                    f"seed{seed} r{replicate}: updates hash mismatch"
                )
            compressed = gzip_file(
                run / "updates.jsonl", output / relative / "updates.jsonl.gz",
            )
            compressed["path"] = str(relative / "updates.jsonl.gz")
            artifacts.append(compressed)
            source_manifest = (
                manifest_root / f"seed{seed}_random_query_r{replicate}.json"
            )
            manifest_relative = Path("manifests") / source_manifest.name
            row = copy_file(source_manifest, output / manifest_relative)
            row["path"] = str(manifest_relative)
            artifacts.append(row)
            seed_rows.append({
                "replicate": replicate,
                "query_count": int(summary["query_count"]),
                "online": summary["results"]["online_unified"],
                "static_predictions_sha256": summary["hashes"][
                    "static_predictions"
                ],
            })
        runs[str(seed)] = seed_rows
    payload = {
        "schema_version": "causal-random-query-compact-archive-v1",
        "claim_status": "BALANCED360_METHOD_DEVELOPMENT_DIAGNOSTIC",
        "source_root_serialized": False,
        "seeds": seeds,
        "replicates_per_seed": int(replicates),
        "runs": runs,
        "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        "omitted": [
            "risk_traces (regenerable and high volume)",
            "duplicate static_predictions.jsonl (referenced by exact SHA256)",
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[131, 133, 137])
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()
    report = archive(
        source_root=args.source_root, output=args.output,
        seeds=[int(seed) for seed in args.seeds],
        replicates=int(args.replicates),
    )
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "run_count": sum(len(rows) for rows in report["runs"].values()),
        "artifact_count": len(report["artifacts"]),
    }, indent=2))


if __name__ == "__main__":
    main()
