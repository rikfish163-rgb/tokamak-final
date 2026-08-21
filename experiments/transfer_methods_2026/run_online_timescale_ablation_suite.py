#!/usr/bin/env python3
"""Run the registered intra/inter timescale ablation with one command."""

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
RUNNER = ROOT / "experiments/transfer_methods_2026/run_unified_online_active.py"


@dataclass(frozen=True)
class SuitePaths:
    source_checkpoint: Path
    a1_dir: Path
    adapter_dir: Path
    out_root: Path
    sealed_parent_state: Path | None = None


def build_runner_command(
    *, seed: int, max_shots: int, mode: str, paths: SuitePaths,
    query_cost: float, query_quota: int,
    query_score_mode: str = "current", query_quantile: float = 0.70,
    error_memory_weight: float = 0.0,
    adaptation_events: int = 64,
    intra_steps: int = 1,
    intra_lr: float = 1e-4,
    private_intra_lr: float = 1e-4,
    intra_physics_weight: float = 0.10,
    intra_observed_physics_weight: float = 0.0,
    intra_parameter_scope: str = "legacy",
    intra_temporal_risk_weight: float = 0.0,
    intra_temporal_target_max_delta: float = 0.5,
    intra_temporal_target_gain: float = 1.0,
    episode_physics_escape: float = 0.85,
    novel_early_min_physics_support: float = 0.0,
    novel_early_min_trend: float = 0.0,
    order_manifest: Path | None = None,
    fixed_query_manifest: Path | None = None,
) -> list[str]:
    if mode not in {"intra_only", "inter_only", "intra_inter"}:
        raise ValueError("invalid timescale suite mode")
    effective_cost = 0.0 if mode == "intra_only" else float(query_cost)
    out = Path(paths.out_root) / f"seed{int(seed)}_{mode}"
    command = [
        sys.executable, str(RUNNER),
        "--seed", str(int(seed)), "--max-shots", str(int(max_shots)),
        "--out", str(out),
        "--source-checkpoint", str(Path(paths.source_checkpoint).resolve()),
        "--a1-dir", str(Path(paths.a1_dir).resolve()),
        "--adapter-dir", str(Path(paths.adapter_dir).resolve()),
        "--private-support-epochs", "8",
        "--adaptation-events", str(int(adaptation_events)),
        "--intra-steps", str(int(intra_steps)),
        "--inter-steps", "2", "--intra-lr", str(float(intra_lr)),
        "--private-intra-lr", str(float(private_intra_lr)),
        "--intra-physics-weight", str(float(intra_physics_weight)),
        "--intra-observed-physics-weight",
        str(float(intra_observed_physics_weight)),
        "--intra-parameter-scope", str(intra_parameter_scope),
        "--intra-temporal-risk-weight", str(float(intra_temporal_risk_weight)),
        "--intra-temporal-target-max-delta",
        str(float(intra_temporal_target_max_delta)),
        "--intra-temporal-target-gain", str(float(intra_temporal_target_gain)),
        "--inter-lr", "0.0005", "--private-inter-lr", "0.0002",
        "--alarm-lr", "0.02", "--query-block-size", "20",
        "--query-quota", str(int(query_quota)),
        "--query-history-size", "160", "--query-quantile", str(float(query_quantile)),
        "--query-uncertainty-weight", "0.55", "--query-value-lr", "0.05",
        "--causal-error-memory", "--error-memory-radius", "0.35",
        "--error-memory-min-support", "2", "--error-memory-weight",
        str(float(error_memory_weight)),
        "--query-initial-cutoff", "0.35", "--query-token-capacity", "12",
        "--strict-window-active", "--window-candidate-stride", "8",
        "--window-update-rate", "0.25", "--window-token-capacity", "4",
        "--window-history-size", "64", "--window-value-quantile", "0.80",
        "--window-initial-cutoff", "0.40", "--window-minimum-history", "8",
        "--window-minimum-gap", "8", "--typed-endpoint-memory",
        "--endpoint-memory-radius", "0.35", "--endpoint-memory-min-shots", "2",
        "--endpoint-memory-positive", "0.35", "--endpoint-memory-negative", "0.25",
        "--endpoint-memory-physics-gate", "0.50",
        "--endpoint-memory-protection-ratio", "1.0",
        "--endpoint-memory-stride", "16", "--episode-value-v3",
        "--episode-protected-sequence", "--episode-isolate-learning-feedback",
        "--episode-query-mode", "shadow", "--episode-gradient-mode", "legacy",
        "--episode-sequence-l2", "0.30", "--episode-protection-margin", "0.06",
        "--episode-negative-quantile", "0.90", "--episode-safe-quantile", "0.25",
        "--episode-minimum-confidence", "0.75", "--episode-min-history", "12",
        "--episode-min-per-class", "3", "--episode-high-escape-margin", "99",
        "--episode-physics-escape", str(float(episode_physics_escape)),
        "--novel-early-min-physics-support",
        str(float(novel_early_min_physics_support)),
        "--novel-early-min-trend", str(float(novel_early_min_trend)),
        "--episode-max-extra-points", "8",
        "--episode-maximum-points", "64", "--episode-shoulder-margin", "0.50",
        "--replay-max-distance", "0.35", "--query-exploration-per-block", "1",
        "--timescale-mode", mode, "--query-cost", str(effective_cost),
        "--query-score-mode", str(query_score_mode),
        "--development-search",
    ]
    if paths.sealed_parent_state is not None:
        command.extend([
            "--sealed-parent-state",
            str(Path(paths.sealed_parent_state).resolve()),
        ])
    if order_manifest is not None:
        command.extend([
            "--order-manifest", str(Path(order_manifest).resolve()),
            "--diagnostic-reorder",
        ])
    if fixed_query_manifest is not None:
        if mode == "intra_only":
            raise ValueError("intra-only cannot use a fixed query manifest")
        command.extend([
            "--fixed-query-manifest", str(Path(fixed_query_manifest).resolve()),
        ])
    return command


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max-shots", type=int, default=360)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--sealed-parent-state", type=Path, default=None)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--query-cost", type=float, default=0.35)
    parser.add_argument("--query-quota", type=int, default=2)
    parser.add_argument(
        "--query-score-mode", choices=("current", "gradient_blend"),
        default="current",
    )
    parser.add_argument("--query-quantile", type=float, default=0.70)
    parser.add_argument("--error-memory-weight", type=float, default=0.0)
    parser.add_argument("--adaptation-events", type=int, default=64)
    parser.add_argument("--intra-steps", type=int, default=1)
    parser.add_argument("--intra-lr", type=float, default=1e-4)
    parser.add_argument("--private-intra-lr", type=float, default=1e-4)
    parser.add_argument("--intra-physics-weight", type=float, default=0.10)
    parser.add_argument(
        "--intra-observed-physics-weight", type=float, default=0.0,
    )
    parser.add_argument(
        "--intra-parameter-scope",
        choices=("legacy", "representation_only"), default="legacy",
    )
    parser.add_argument("--intra-temporal-risk-weight", type=float, default=0.0)
    parser.add_argument(
        "--intra-temporal-target-max-delta", type=float, default=0.5,
    )
    parser.add_argument("--intra-temporal-target-gain", type=float, default=1.0)
    parser.add_argument("--episode-physics-escape", type=float, default=0.85)
    parser.add_argument(
        "--novel-early-min-physics-support", type=float, default=0.0,
    )
    parser.add_argument("--novel-early-min-trend", type=float, default=0.0)
    parser.add_argument(
        "--modes", nargs="+", choices=("intra_only", "inter_only", "intra_inter"),
        default=("intra_only", "inter_only", "intra_inter"),
    )
    parser.add_argument("--order-manifest", type=Path, default=None)
    parser.add_argument("--fixed-query-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = SuitePaths(
        source_checkpoint=args.source_checkpoint,
        a1_dir=args.a1_dir,
        adapter_dir=args.adapter_dir,
        out_root=args.out_root,
        sealed_parent_state=args.sealed_parent_state,
    )
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    commands = []
    for mode in args.modes:
        command = build_runner_command(
            seed=args.seed, max_shots=args.max_shots, mode=mode, paths=paths,
            query_cost=args.query_cost, query_quota=args.query_quota,
            query_score_mode=args.query_score_mode,
            query_quantile=args.query_quantile,
            error_memory_weight=args.error_memory_weight,
            adaptation_events=args.adaptation_events,
            intra_steps=args.intra_steps,
            intra_lr=args.intra_lr,
            private_intra_lr=args.private_intra_lr,
            intra_physics_weight=args.intra_physics_weight,
            intra_observed_physics_weight=(
                args.intra_observed_physics_weight
            ),
            intra_parameter_scope=args.intra_parameter_scope,
            intra_temporal_risk_weight=args.intra_temporal_risk_weight,
            intra_temporal_target_max_delta=(
                args.intra_temporal_target_max_delta
            ),
            intra_temporal_target_gain=args.intra_temporal_target_gain,
            episode_physics_escape=args.episode_physics_escape,
            novel_early_min_physics_support=(
                args.novel_early_min_physics_support
            ),
            novel_early_min_trend=args.novel_early_min_trend,
            order_manifest=args.order_manifest,
            fixed_query_manifest=args.fixed_query_manifest,
        )
        commands.append(command)
    (out_root / "suite_commands.json").write_text(json.dumps({
        "schema_version": "online-timescale-ablation-v1",
        "seed": args.seed, "max_shots": args.max_shots,
        "query_cost": args.query_cost, "query_quota": args.query_quota,
        "query_score_mode": args.query_score_mode,
        "query_quantile": args.query_quantile,
        "error_memory_weight": args.error_memory_weight,
        "adaptation_events": args.adaptation_events,
        "intra_steps": args.intra_steps,
        "intra_lr": args.intra_lr,
        "private_intra_lr": args.private_intra_lr,
        "intra_physics_weight": args.intra_physics_weight,
        "intra_observed_physics_weight": (
            args.intra_observed_physics_weight
        ),
        "intra_parameter_scope": args.intra_parameter_scope,
        "intra_temporal_risk_weight": args.intra_temporal_risk_weight,
        "intra_temporal_target_max_delta": (
            args.intra_temporal_target_max_delta
        ),
        "intra_temporal_target_gain": args.intra_temporal_target_gain,
        "episode_physics_escape": args.episode_physics_escape,
        "novel_early_min_physics_support": args.novel_early_min_physics_support,
        "novel_early_min_trend": args.novel_early_min_trend,
        "commands": commands,
    }, indent=2) + "\n")
    environment = os.environ.copy()
    environment.setdefault("FUSION_PROJECT_ROOT", str(ROOT))
    for command in commands:
        output = Path(command[command.index("--out") + 1])
        if (output / "summary.json").exists():
            print(f"[SUITE] already complete: {output}", flush=True)
            continue
        print("[SUITE] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
