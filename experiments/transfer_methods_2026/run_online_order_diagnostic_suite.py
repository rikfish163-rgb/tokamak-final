#!/usr/bin/env python3
"""Run the registered balanced360 inter-shot order upper-bound controls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
TIMESCALE_SUITE = ROOT / "experiments/transfer_methods_2026/run_online_timescale_ablation_suite.py"


@dataclass(frozen=True)
class OrderDiagnosticSpec:
    name: str
    order_manifest: Path
    query_manifest: Path
    label_oracle_used: bool
    interpretation: str


def order_diagnostic_specs(manifest_dir: Path) -> list[OrderDiagnosticSpec]:
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    chronological_queries = manifest_dir / "chronological_queries.json"
    return [
        OrderDiagnosticSpec(
            name="random_fixed_q",
            order_manifest=manifest_dir / "random_fixed_query_order.json",
            query_manifest=chronological_queries,
            label_oracle_used=False,
            interpretation="Order sensitivity with the same queried shots and slots.",
        ),
        OrderDiagnosticSpec(
            name="chronological_oracle_q",
            order_manifest=manifest_dir / "chronological_order.json",
            query_manifest=(
                manifest_dir / "chronological_oracle_error_queries.json"
            ),
            label_oracle_used=True,
            interpretation=(
                "Hindsight query-selection ceiling with chronological order unchanged."
            ),
        ),
        OrderDiagnosticSpec(
            name="oracle_order_fixed_q",
            order_manifest=manifest_dir / "oracle_fixed_query_order.json",
            query_manifest=chronological_queries,
            label_oracle_used=True,
            interpretation=(
                "Hindsight order ceiling with the chronological query set and slots fixed."
            ),
        ),
        OrderDiagnosticSpec(
            name="oracle_order_plus_q",
            order_manifest=manifest_dir / "oracle_error_curriculum_order.json",
            query_manifest=manifest_dir / "oracle_error_queries.json",
            label_oracle_used=True,
            interpretation="Hindsight ceiling that optimizes both error queries and order.",
        ),
    ]


def build_command(
    spec: OrderDiagnosticSpec, *, seed: int, max_shots: int,
    source_checkpoint: Path, a1_dir: Path, adapter_dir: Path,
    out_root: Path, query_cost: float, query_quota: int,
) -> list[str]:
    return [
        sys.executable, str(TIMESCALE_SUITE),
        "--seed", str(int(seed)),
        "--max-shots", str(int(max_shots)),
        "--source-checkpoint", str(Path(source_checkpoint).resolve()),
        "--a1-dir", str(Path(a1_dir).resolve()),
        "--adapter-dir", str(Path(adapter_dir).resolve()),
        "--out-root", str((Path(out_root) / spec.name).resolve()),
        "--query-cost", str(float(query_cost)),
        "--query-quota", str(int(query_quota)),
        "--modes", "inter_only",
        "--order-manifest", str(spec.order_manifest),
        "--fixed-query-manifest", str(spec.query_manifest),
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max-shots", type=int, default=360)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--query-cost", type=float, default=0.35)
    parser.add_argument("--query-quota", type=int, default=2)
    args = parser.parse_args(argv)

    specs = order_diagnostic_specs(args.manifest_dir)
    missing = [
        str(path)
        for spec in specs
        for path in (spec.order_manifest, spec.query_manifest)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"order diagnostic manifests are missing: {missing}")
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    commands = [
        build_command(
            spec, seed=args.seed, max_shots=args.max_shots,
            source_checkpoint=args.source_checkpoint, a1_dir=args.a1_dir,
            adapter_dir=args.adapter_dir, out_root=out_root,
            query_cost=args.query_cost, query_quota=args.query_quota,
        )
        for spec in specs
    ]
    (out_root / "order_suite_commands.json").write_text(json.dumps({
        "schema_version": "online-order-diagnostic-v1",
        "status": "ORDER_ORACLE_DIAGNOSTIC",
        "chronological_result_is_main": True,
        "specs": [
            {
                "name": spec.name,
                "label_oracle_used": spec.label_oracle_used,
                "interpretation": spec.interpretation,
                "command": command,
            }
            for spec, command in zip(specs, commands)
        ],
    }, indent=2) + "\n")
    environment = os.environ.copy()
    environment.setdefault("FUSION_PROJECT_ROOT", str(ROOT))
    for spec, command in zip(specs, commands):
        summary = out_root / spec.name / f"seed{args.seed}_inter_only" / "summary.json"
        if summary.exists():
            print(f"[ORDER] already complete: {spec.name}", flush=True)
            continue
        print(f"[ORDER] {spec.name}: {spec.interpretation}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
