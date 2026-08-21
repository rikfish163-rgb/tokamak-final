#!/usr/bin/env python3
"""Build a query-budget curve without mixing incomparable complements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _utility(metric: Mapping[str, Any]) -> float:
    return float(metric["tpr_at_30ms"]) - float(metric["fpr"])


def _with_utility(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(metric), "utility_tpr30_minus_fpr": _utility(metric)}


def build_report(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not points:
        raise ValueError("at least one budget point is required")
    ordered = sorted(points, key=lambda point: int(point["online"]["query_count"]))
    baseline = _with_utility(ordered[0]["online"]["results"]["static_unified"])
    curve: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    previous_j = baseline["utility_tpr30_minus_fpr"]
    previous_queries = 0
    for point in ordered:
        quota = int(point["quota"])
        online_summary = point["online"]
        query_count = int(online_summary["query_count"])
        metric = _with_utility(online_summary["results"]["online_unified"])
        delta_queries = query_count - previous_queries
        delta_j = metric["utility_tpr30_minus_fpr"] - previous_j
        curve.append({
            "quota_per_20_shots": quota,
            "query_count": query_count,
            "query_rate": query_count / 360.0,
            **metric,
            "delta_utility_from_previous_budget": delta_j,
            "marginal_utility_per_added_query": (
                delta_j / delta_queries if delta_queries > 0 else None
            ),
        })
        previous_j = metric["utility_tpr30_minus_fpr"]
        previous_queries = query_count

        frontload = point["frontload"]
        fm = frontload.get(
            "metrics", frontload.get("metrics_on_label_disjoint_complement")
        )
        if fm is None:
            raise KeyError("frontload summary has no complement metrics")
        online_comp = _with_utility(fm["original_online_same_complement"])
        static_comp = _with_utility(fm["original_static_same_complement"])
        direct_frozen = _with_utility(fm["frontload_frozen_parent_threshold"])
        direct_recal = _with_utility(
            fm["frontload_support110_recalibrated_threshold"]
        )
        best_direct_j = max(
            direct_frozen["utility_tpr30_minus_fpr"],
            direct_recal["utility_tpr30_minus_fpr"],
        )
        online_j = online_comp["utility_tpr30_minus_fpr"]
        evaluation_count = sum(
            int(online_comp[name]) for name in ("TP", "FP", "FN", "TN")
        )
        paired.append({
            "quota_per_20_shots": quota,
            "query_count": query_count,
            "evaluation_complement_count": evaluation_count,
            "static": static_comp,
            "online": online_comp,
            "direct_frontload_frozen_threshold": direct_frozen,
            "direct_frontload_support110_recalibrated": direct_recal,
            "online_minus_best_direct_utility": online_j - best_direct_j,
            "online_beats_best_direct": online_j > best_direct_j,
        })
    return {
        "schema_version": "query-budget-curve-v1",
        "claim_status": "BALANCED360_METHOD_DEVELOPMENT_DIAGNOSTIC",
        "metric_contract": {
            "primary": "TPR_at_lead_ge_30ms_minus_FPR",
            "raw_tpr_reported_separately": True,
            "over_3000ms_alarm_counts_as_tpr": True,
        },
        "static_full360": baseline,
        "full360_online_curve": curve,
        "paired_label_disjoint_complements": paired,
        "cross_budget_complement_comparison_allowed": False,
        "warnings": [
            "Full360 online rows are comparable across budgets.",
            "Each direct-frontload comparison excludes that budget's queried training shots; complement composition changes with budget.",
            "Label access is matched within a budget, but optimizer steps, replay and FLOPs are not exactly matched.",
            "The balanced360 stream was used for method development and is not an untouched future confirmation set.",
        ],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    base = report["static_full360"]
    lines = [
        "# Query-budget curve (seed 131)", "",
        "Primary utility: `TPR@30ms - FPR`.", "",
        "## Full balanced360 online curve", "",
        "| Queries | Rate | TPR@30ms | FPR | Raw TPR | Lead (ms) | Utility | Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 0 | 0.00% | {base['tpr_at_30ms']:.4f} | {base['fpr']:.4f} | "
        f"{base['tpr']:.4f} | {base['median_lead_ms']:.1f} | "
        f"{base['utility_tpr30_minus_fpr']:.4f} | — |",
    ]
    for row in report["full360_online_curve"]:
        lines.append(
            f"| {row['query_count']} | {100*row['query_rate']:.2f}% | "
            f"{row['tpr_at_30ms']:.4f} | {row['fpr']:.4f} | {row['tpr']:.4f} | "
            f"{row['median_lead_ms']:.1f} | {row['utility_tpr30_minus_fpr']:.4f} | "
            f"{row['delta_utility_from_previous_budget']:+.4f} |"
        )
    lines += [
        "", "## Paired equal-label-budget comparisons", "",
        "Complements differ between rows and must not be compared across budgets.", "",
        "| Queries | Complement | Online J | Best direct J | Online-direct | Online wins |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["paired_label_disjoint_complements"]:
        online_j = row["online"]["utility_tpr30_minus_fpr"]
        best_direct = max(
            row["direct_frontload_frozen_threshold"]["utility_tpr30_minus_fpr"],
            row["direct_frontload_support110_recalibrated"]["utility_tpr30_minus_fpr"],
        )
        lines.append(
            f"| {row['query_count']} | {row['evaluation_complement_count']} | "
            f"{online_j:.4f} | {best_direct:.4f} | "
            f"{row['online_minus_best_direct_utility']:+.4f} | "
            f"{'yes' if row['online_beats_best_direct'] else 'no'} |"
        )
    lines += ["", "## Claim boundary", ""]
    lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point", nargs=3, action="append", required=True,
        metavar=("QUOTA", "ONLINE_SUMMARY", "FRONTLOAD_SUMMARY"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    points = [
        {
            "quota": int(quota), "online": _load(Path(online)),
            "frontload": _load(Path(frontload)),
        }
        for quota, online, frontload in args.point
    ]
    report = build_report(points)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.write_text(_markdown(report))


if __name__ == "__main__":
    main()
