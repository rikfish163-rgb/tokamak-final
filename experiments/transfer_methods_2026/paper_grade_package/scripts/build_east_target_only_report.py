#!/usr/bin/env python3
"""Aggregate EAST-only controls against the archived transfer-static arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


ARMS = ("target_only_local17", "target_only_matched17")
SEEDS = (131, 133, 137)


def compact(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "operational_tpr": float(metrics["operational_tpr"]),
        "fpr": float(metrics["fpr"]),
        "utility_tpr30_minus_fpr": (
            float(metrics["operational_tpr"]) - float(metrics["fpr"])
        ),
        "operational_median_lead_ms": float(
            metrics["operational_median_lead_ms"]
        ),
    }


def aggregate(rows: list[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: mean(float(row[key]) for row in rows)
        for key in (
            "operational_tpr", "fpr", "utility_tpr30_minus_fpr",
            "operational_median_lead_ms",
        )
    }


def build(*, source_root: Path, package: Path) -> dict[str, Any]:
    evidence = json.loads(
        (package / "evidence/three_stage_results.json").read_text()
    )
    transfer_rows = [
        compact(evidence["official_results"]["east_static360"]["seeds"][str(seed)])
        for seed in SEEDS
    ]
    transfer = aggregate(transfer_rows)
    arms: dict[str, Any] = {}
    for arm in ARMS:
        per_seed = {}
        rows = []
        for seed in SEEDS:
            summary = json.loads(
                (source_root / arm / f"seed{seed}/summary.json").read_text()
            )
            if summary["arm"] != arm or int(summary["seed"]) != seed:
                raise RuntimeError(f"{arm}/seed{seed}: identity mismatch")
            row = compact(summary["metrics"])
            rows.append(row)
            per_seed[str(seed)] = row
        arm_mean = aggregate(rows)
        arms[arm] = {
            "per_seed": per_seed,
            "equal_seed_mean": arm_mean,
            "transfer_static_minus_target_only": {
                key: transfer[key] - arm_mean[key]
                for key in arm_mean
            },
        }
    return {
        "schema_version": "east-target-only-report-v1",
        "claim_boundary": "TEST_DRIVEN_EXPLORATION",
        "comparison": (
            "same support110 label/window/optimizer-step budget, full target-side "
            "shared/private pipeline and fixed dual alarm; not equal FLOPs and not "
            "an independent balanced360 test"
        ),
        "transfer_static": {
            "per_seed": {str(seed): row for seed, row in zip(SEEDS, transfer_rows)},
            "equal_seed_mean": transfer,
        },
        "target_only_arms": arms,
        "interpretation_limits": [
            "balanced360 informed method development",
            "support110 is reused for fitting and threshold selection",
            "target-only updates the full encoder while transfer-static updates a restricted target path, so equal optimizer steps are not equal FLOPs",
            "matched17 consumes a source-derived scaler artifact and is not pure EAST-only",
            "local17 is the strict EAST-only arm",
        ],
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# EAST-only versus J-TEXT-initialized transfer-static baseline", "",
        "All values are balanced360 development-stream means over seeds 131/133/137.", "",
        "| Arm | TPR@30 | FPR | TPR@30-FPR | Median lead (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = {"J-TEXT initialized transfer-static": report["transfer_static"]["equal_seed_mean"]}
    rows.update({
        arm: value["equal_seed_mean"]
        for arm, value in report["target_only_arms"].items()
    })
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['operational_tpr']:.4f} | {row['fpr']:.4f} | "
            f"{row['utility_tpr30_minus_fpr']:.4f} | "
            f"{row['operational_median_lead_ms']:.1f} |"
        )
    lines += [
        "", "This is a target-data/initialization control on the reused development "
        "stream, not an independent generalization estimate.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        source_root=args.source_root.expanduser().resolve(),
        package=args.package.expanduser().resolve(),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.write_text(markdown(report))


if __name__ == "__main__":
    main()
