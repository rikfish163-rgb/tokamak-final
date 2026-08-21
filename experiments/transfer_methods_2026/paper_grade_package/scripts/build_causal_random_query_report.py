#!/usr/bin/env python3
"""Aggregate active versus causal random-query online controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def utility(metrics: Mapping[str, Any]) -> float:
    return float(metrics["tpr_at_30ms"]) - float(metrics["fpr"])


def compact(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(summary["results"]["online_unified"])
    return {
        **metrics,
        "utility_tpr30_minus_fpr": utility(metrics),
        "query_count": int(summary["query_count"]),
        "intra_attempts": int(summary["counts"]["intra_attempts"]),
        "shared_inter_attempts": int(
            summary["counts"]["shared_inter_accepted"]
            + summary["counts"]["shared_inter_rollback"]
        ),
        "private_inter_attempts": int(
            summary["counts"]["private_inter_accepted"]
            + summary["counts"]["private_inter_rollback"]
        ),
        "median_lead_field_status": (
            "runner diagnostic; not promoted as the operational lead-qualified "
            "median in this treatment comparison"
        ),
    }


def build_report(
    *, package: Path, random_root: Path, seeds: list[int], replicates: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    deltas: list[float] = []
    active_utilities: list[float] = []
    random_utilities: list[float] = []
    active_inter_attempts: list[float] = []
    random_inter_attempts: list[float] = []
    for seed in seeds:
        active = load(
            package / "evidence/primary_online_artifacts"
            / f"seed{seed}/summary.json"
        )
        random_summaries = [
            load(random_root / f"seed{seed}/causal_random_query_r{rep}/summary.json")
            for rep in range(replicates)
        ]
        active_row = compact(active)
        random_rows = [compact(item) for item in random_summaries]
        active_hashes = active["hashes"]
        for replicate, random_summary in enumerate(random_summaries):
            random_hashes = random_summary["hashes"]
            for name in (
                "source_checkpoint", "base_before_after",
                "phase2_initial_private", "phase2_initial_shared",
                "static_predictions",
            ):
                if random_hashes[name] != active_hashes[name]:
                    raise RuntimeError(
                        f"seed {seed} replicate {replicate}: parent/static hash "
                        f"mismatch for {name}"
                    )
            if int(random_summary["query_count"]) != int(active["query_count"]):
                raise RuntimeError(
                    f"seed {seed} replicate {replicate}: query count mismatch"
                )
        random_mean = {
            key: mean(float(row[key]) for row in random_rows)
            for key in (
                "tpr_at_30ms", "fpr",
                "utility_tpr30_minus_fpr", "query_count", "intra_attempts",
                "shared_inter_attempts", "private_inter_attempts",
            )
        }
        delta = (
            float(active_row["utility_tpr30_minus_fpr"])
            - float(random_mean["utility_tpr30_minus_fpr"])
        )
        deltas.append(delta)
        active_utilities.append(float(active_row["utility_tpr30_minus_fpr"]))
        random_utilities.append(float(random_mean["utility_tpr30_minus_fpr"]))
        active_inter_attempts.append(float(active_row["shared_inter_attempts"]))
        random_inter_attempts.append(float(random_mean["shared_inter_attempts"]))
        rows[str(seed)] = {
            "active": active_row,
            "random_replicates": random_rows,
            "random_mean_within_seed": random_mean,
            "active_minus_random_mean_utility": delta,
            "parent_and_static_hashes_match": True,
            "label_budget_match": True,
            "configured_optimizer_budget_match": True,
            "realized_compute_exact_match": False,
        }
    return {
        "schema_version": "causal-random-query-online-report-v1",
        "claim_status": "BALANCED360_METHOD_DEVELOPMENT_DIAGNOSTIC",
        "comparison": (
            "same parent, runner, chronological stream, configured update-step "
            "budget and active per-block label count; random query identities "
            "use no outcomes"
        ),
        "seeds": seeds,
        "replicates_per_seed": int(replicates),
        "per_seed": rows,
        "aggregate": {
            "active_utility_equal_seed_mean": mean(active_utilities),
            "random_utility_equal_seed_mean": mean(random_utilities),
            "active_minus_random_utility_mean": mean(deltas),
            "seeds_active_better": sum(value > 0 for value in deltas),
            "seeds_active_not_worse": sum(value >= 0 for value in deltas),
            "seed_count": len(deltas),
            "confirmatory_significance_claim": False,
            "exact_one_sided_sign_test_p_if_all_three_positive": (
                0.125 if all(value > 0 for value in deltas) else None
            ),
            "active_shared_inter_attempts_equal_seed_mean": mean(
                active_inter_attempts
            ),
            "random_shared_inter_attempts_equal_seed_mean": mean(
                random_inter_attempts
            ),
            "realized_inter_attempt_ratio_active_over_random": (
                mean(active_inter_attempts) / mean(random_inter_attempts)
                if mean(random_inter_attempts) > 0 else None
            ),
        },
        "limitations": [
            "balanced360 was used for method development",
            "the active arm's realized per-block query counts were used post hoc to construct the matched random schedules",
            "query labels and configured optimizer-step budgets match, but realized FLOPs and adaptive replay paths are not guaranteed byte-for-byte equal",
            "model inter-shot updates are error-triggered; selecting more errors is part of the active mechanism but also increases realized training compute, so this comparison is not an equal-FLOP isolation of selector quality",
            "this closes the missing causal random-query baseline only within the development stream",
            "a new frozen chronological EAST stream remains required for confirmation",
        ],
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Causal active versus random-query online baseline", "",
        "Primary utility is `TPR@30ms - FPR`.", "",
        "| Seed | Active TPR@30 | Active FPR | Active J | Random mean J | Delta J |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        row = report["per_seed"][str(seed)]
        active = row["active"]
        random = row["random_mean_within_seed"]
        lines.append(
            f"| {seed} | {active['tpr_at_30ms']:.4f} | {active['fpr']:.4f} | "
            f"{active['utility_tpr30_minus_fpr']:.4f} | "
            f"{random['utility_tpr30_minus_fpr']:.4f} | "
            f"{row['active_minus_random_mean_utility']:+.4f} |"
        )
    aggregate = report["aggregate"]
    lines += [
        "", f"Mean active-minus-random J: "
        f"`{aggregate['active_minus_random_utility_mean']:+.6f}`.", "",
        "Realized shared inter-shot update attempts (equal-seed mean): "
        f"active `{aggregate['active_shared_inter_attempts_equal_seed_mean']:.2f}`, "
        f"random `{aggregate['random_shared_inter_attempts_equal_seed_mean']:.2f}`. "
        "The selector comparison is query-budget matched but not equal-FLOP.", "",
        "This remains development-stream evidence, not prospective confirmation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[131, 133, 137])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        package=args.package.expanduser().resolve(),
        random_root=args.random_root.expanduser().resolve(),
        seeds=[int(value) for value in args.seeds],
        replicates=int(args.replicates),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.write_text(markdown(report))


if __name__ == "__main__":
    main()
