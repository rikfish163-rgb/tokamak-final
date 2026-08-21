#!/usr/bin/env python3
"""Build the causal timescale and equal-label-budget ablation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _utility(metrics: Mapping[str, Any]) -> float:
    return float(metrics["tpr_at_30ms"]) - float(metrics["fpr"])


def build_report(
    *, intra: Mapping[str, Any], inter: Mapping[str, Any],
    joint: Mapping[str, Any], frontload: Mapping[str, Any],
) -> dict[str, Any]:
    static = intra["results"]["static_unified"]
    arms = {
        "static": static,
        "intra_only": intra["results"]["online_unified"],
        "inter_only": inter["results"]["online_unified"],
        "intra_inter": joint["results"]["online_unified"],
    }
    rows = {
        name: {**dict(metrics), "utility_tpr30_minus_fpr": _utility(metrics)}
        for name, metrics in arms.items()
    }
    j0 = rows["static"]["utility_tpr30_minus_fpr"]
    ji = rows["intra_only"]["utility_tpr30_minus_fpr"]
    je = rows["inter_only"]["utility_tpr30_minus_fpr"]
    jj = rows["intra_inter"]["utility_tpr30_minus_fpr"]

    fm = frontload.get(
        "metrics", frontload.get("metrics_on_label_disjoint_complement")
    )
    if fm is None:
        raise KeyError("frontload summary has no complement metrics")
    complement = {
        "static": fm["original_static_same_complement"],
        "online_intra_inter": fm["original_online_same_complement"],
        "direct_frontload_frozen_threshold": (
            fm["frontload_frozen_parent_threshold"]
        ),
        "direct_frontload_support110_recalibrated": (
            fm["frontload_support110_recalibrated_threshold"]
        ),
    }
    complement_rows = {
        name: {**dict(metrics), "utility_tpr30_minus_fpr": _utility(metrics)}
        for name, metrics in complement.items()
    }
    best_direct = max(
        complement_rows["direct_frontload_frozen_threshold"][
            "utility_tpr30_minus_fpr"
        ],
        complement_rows["direct_frontload_support110_recalibrated"][
            "utility_tpr30_minus_fpr"
        ],
    )
    online_j = complement_rows["online_intra_inter"][
        "utility_tpr30_minus_fpr"
    ]
    return {
        "schema_version": "causal-timescale-label-budget-ablation-v1",
        "claim_status": "BALANCED360_METHOD_DEVELOPMENT_DIAGNOSTIC",
        "metric_contract": {
            "primary": "TPR_at_lead_ge_30ms_minus_FPR",
            "raw_tpr_reported_separately": True,
            "over_3000ms_alarm_counts_as_tpr": True,
        },
        "full360_timescale_ablation": rows,
        "effect_decomposition": {
            "intra_main": ji - j0,
            "inter_main": je - j0,
            "joint_total": jj - j0,
            "interaction": jj - j0 - (ji - j0) - (je - j0),
        },
        "query_contract": {
            "query_count": int(joint["query_count"]),
            "query_identity_semantics": (
                "selected causally by a label-free shot-level query-value "
                "policy, then frozen as one immutable manifest for the "
                "inter-only, intra+inter and direct-frontload controls"
            ),
            "not_an_arbitrary_fixed_cohort": True,
            "same_fixed_manifest": (
                inter["fixed_query_manifest"] == joint["fixed_query_manifest"]
            ),
            "intra_window_value_is_a_separate_selector": True,
            "query_selection_label_oracle_used": False,
            "predict_seal_query_reveal_update": True,
            "provenance_boundary": (
                "The recovered immutable Q30 manifest preserves shot identities, "
                "stream positions and the SHA256 of the original 360-row decision "
                "ledger. The original per-shot decision ledger was lost after the "
                "tmpfs reboot, so historical cutoff, score and route values cannot "
                "be independently recomputed from the compact package."
            ),
        },
        "same_label_disjoint_330_complement": complement_rows,
        "equal_label_budget_verdict": {
            "online_minus_best_direct_frontload_utility": online_j - best_direct,
            "online_minus_static_utility": online_j - complement_rows["static"][
                "utility_tpr30_minus_fpr"
            ],
            "online_beats_direct_frontload": online_j > best_direct,
            "compute_matched": False,
            "warning": (
                "Label access is matched at 30 queried shots; optimizer steps, "
                "replay and FLOPs are not exactly matched. The 330-shot "
                "complement excludes queried training shots."
            ),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _markdown(report: Mapping[str, Any]) -> str:
    rows = report["full360_timescale_ablation"]
    lines = [
        "# Causal online ablation (seed 131)", "",
        "Primary utility is `TPR@30ms - FPR`; raw TPR is reported separately.", "",
        "| Arm | Raw TPR | TPR@30ms | FPR | Median lead (ms) | Utility |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("static", "intra_only", "inter_only", "intra_inter"):
        row = rows[name]
        lines.append(
            f"| {name} | {row['tpr']:.4f} | {row['tpr_at_30ms']:.4f} | "
            f"{row['fpr']:.4f} | {row['median_lead_ms']:.1f} | "
            f"{row['utility_tpr30_minus_fpr']:.4f} |"
        )
    effects = report["effect_decomposition"]
    verdict = report["equal_label_budget_verdict"]
    query = report["query_contract"]
    lines += [
        "", "## Effects", "",
        f"- Intra main effect: `{effects['intra_main']:+.6f}`",
        f"- Inter main effect: `{effects['inter_main']:+.6f}`",
        f"- Joint total effect: `{effects['joint_total']:+.6f}`",
        f"- Interaction: `{effects['interaction']:+.6f}`",
        "", "## Equal-label-budget verdict", "",
        f"Online minus best direct-frontload utility: "
        f"`{verdict['online_minus_best_direct_frontload_utility']:+.6f}`.",
        "This is label-budget matched, not exact compute matched. The balanced360 "
        "stream was used for method development and is not an untouched future test.",
        "", "## Query identity and provenance", "",
        query["query_identity_semantics"] + ".",
        query["provenance_boundary"],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intra-summary", type=Path, required=True)
    parser.add_argument("--inter-summary", type=Path, required=True)
    parser.add_argument("--joint-summary", type=Path, required=True)
    parser.add_argument("--frontload-summary", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        intra=_load(args.intra_summary), inter=_load(args.inter_summary),
        joint=_load(args.joint_summary), frontload=_load(args.frontload_summary),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.write_text(_markdown(report))


if __name__ == "__main__":
    main()
