#!/usr/bin/env python3
"""Generate supplementary LaTeX tables from the locked machine evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
PACKAGE = ROOT / "experiments/transfer_methods_2026/paper_grade_package"
EVIDENCE = PACKAGE / "evidence/three_stage_results.json"
EQUAL_BUDGET = (
    PACKAGE
    / "evidence/equal_label_budget_frontload_v2/common_mask_report/summary.json"
)
CAUSAL_RANDOM = PACKAGE / "evidence/causal_random_query_online_v1/comparison.json"
EAST_TARGET_ONLY = PACKAGE / "evidence/east_target_only_v1/comparison.json"
OUT = PACKAGE / "manuscript/tables"
SEEDS = (131, 133, 137)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.rstrip() + "\n")


def primary_table(data: dict) -> str:
    source = data["official_results"]["source_test220"]["seeds"]
    static = data["official_results"]["east_static360"]["seeds"]
    online = data["official_results"]["east_online360"]["seeds"]
    aggregates = data["official_results"]
    query_counts = {
        str(seed): data["primary_lineage"]["east_online360"]["query_analysis"][str(seed)]["query_count"]
        for seed in SEEDS
    }
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Stage & Seed & TP/FN & Operational TPR & FP/TN & FPR & AUC & Lead (ms) & Queries \\",
        r"\midrule",
    ]
    stages = (
        ("J-TEXT source", "source_test220", source),
        ("EAST static", "east_static360", static),
        ("EAST online", "east_online360", online),
    )
    for stage_index, (label, key, rows) in enumerate(stages):
        for seed in SEEDS:
            row = rows[str(seed)]
            auc = fmt(row["auc"]) if row.get("auc") is not None else "--"
            queries = query_counts[str(seed)] if key == "east_online360" else (0 if key == "east_static360" else "--")
            lines.append(
                f"{label} & {seed} & {row['operational_TP']}/{row['operational_FN']} & "
                f"{fmt(row['operational_tpr'])} & {row['FP']}/{row['TN']} & "
                f"{fmt(row['fpr'])} & {auc} & "
                f"{fmt(row['operational_median_lead_ms'], 1)} & {queries} \\\\"
            )
        aggregate = aggregates[key]["aggregate"]
        auc_mean = (
            fmt(aggregate["auc"]["mean"])
            if "auc" in aggregate else "--"
        )
        query_mean = (
            fmt(sum(query_counts.values()) / len(query_counts), 1)
            if key == "east_online360" else ("0" if key == "east_static360" else "--")
        )
        lines.append(
            f"{label} & \\textbf{{mean}} & -- & "
            f"\\textbf{{{fmt(aggregate['operational_tpr']['mean'])}}} & -- & "
            f"\\textbf{{{fmt(aggregate['fpr']['mean'])}}} & "
            f"\\textbf{{{auc_mean}}} & "
            f"\\textbf{{{fmt(aggregate['operational_median_lead_ms']['mean'], 1)}}} & {query_mean} \\\\"
        )
        if stage_index != len(stages) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def block_table(data: dict) -> str:
    rows = data["time_blocks"]
    lines = [
        r"\begin{tabular}{rrlrrrrr}",
        r"\toprule",
        r"Seed & Block & Phase & D/N & TP$_{30}$/FP & Operational TPR & FPR & Lead$_{30}$ (ms) \\",
        r"\midrule",
    ]
    for seed in SEEDS:
        for block in range(1, 5):
            for phase in ("static", "online"):
                row = next(
                    item for item in rows
                    if item["seed"] == seed and item["block"] == block
                    and item["phase"] == phase
                )
                lines.append(
                    f"{seed} & {block} & {phase} & "
                    f"{row['n_disruptions']}/{row['n_non_disruptions']} & "
                    f"{row['operational_TP']}/{row['FP']} & "
                    f"{fmt(row['operational_tpr'])} & {fmt(row['fpr'])} & "
                    f"{fmt(row['operational_median_lead_ms'], 1)} \\\\"
                )
        if seed != SEEDS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def shortcut_table(data: dict) -> str:
    old = data["primary_lineage"]["source_test220"]
    pad = data["padfix_ablation"]
    lines = [
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"Seed & Parent & Probe AUC & $|\Delta r|$ & Flip & Operational TPR$_{10}$ & TEST FPR & TEST AUC \\",
        r"\midrule",
    ]
    for seed in SEEDS:
        audit = old["shortcut_audit"]["seeds"][str(seed)]["audit"]
        metric = old["seeds"][str(seed)]
        lines.append(
            f"{seed} & primary & {fmt(audit['paired_context_probe_macro_auc'])} & "
            f"{fmt(audit['paired_context_mean_abs_risk_difference'])} & "
            f"{fmt(audit['paired_context_alarm_flip_rate'])} & "
            f"{fmt(metric['timely_tpr'])} & {fmt(metric['fpr'])} & {fmt(metric['auc'])} \\\\"
        )
        audit = pad["validation_shortcut_audit"][str(seed)]
        metric = pad["source_test220"]["seeds"][str(seed)]["metrics"]
        lines.append(
            f"{seed} & PADFIX & {fmt(audit['paired_context_probe_macro_auc'])} & "
            f"{fmt(audit['paired_context_mean_abs_risk_difference'])} & "
            f"{fmt(audit['paired_context_alarm_flip_rate'])} & "
            f"{fmt(metric['timely_tpr'])} & {fmt(metric['fpr'])} & {fmt(metric['auc'])} \\\\"
        )
        if seed != SEEDS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def padfix_transfer_table(data: dict) -> str:
    pad = data["padfix_ablation"]
    lines = [
        r"\begin{tabular}{rlrrr}",
        r"\toprule",
        r"Seed & Phase & TP$_{30}$/FN$_{30}$ & Operational TPR & FPR \\",
        r"\midrule",
    ]
    for seed in SEEDS:
        for label, key in (("static", "east_static360"), ("online", "east_online360")):
            row = pad[key]["seeds"][str(seed)]
            n_disruptions = int(row.get("n_disruptions", int(row["TP"]) + int(row["FN"])))
            operational_tp = int(round(float(row["tpr_at_30ms"]) * n_disruptions))
            operational_fn = n_disruptions - operational_tp
            lines.append(
                f"{seed} & {label} & {operational_tp}/{operational_fn} & "
                f"{fmt(row['tpr_at_30ms'])} & {fmt(row['fpr'])} \\\\"
            )
        if seed != SEEDS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def controls_table(data: dict) -> str:
    rows = data["negative_controls"]["results"]
    order = (
        "no_labels", "feature_lag", "label_lag", "label_flip",
        "temporal_zero", "raw8_zero", "real_final_v3c",
    )
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Arm & Total FP reduction & Operational TP loss \\",
        r"\midrule",
    ]
    for name in order:
        row = rows[name]
        latex_name = name.replace("_", r"\_")
        lines.append(
            f"{latex_name} & {row['total_fp_reduction']} & "
            f"{row['total_operational_tp_loss']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def query_table(data: dict) -> str:
    rows = data["primary_lineage"]["east_online360"]["query_analysis"]
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Seed & Queries & Budget & Error all & Error queried & Enrichment & MW $p$ \\",
        r"\midrule",
    ]
    for seed in SEEDS:
        row = rows[str(seed)]
        lines.append(
            f"{seed} & {row['query_count']} & {fmt(row['budget_fraction'], 3)} & "
            f"{fmt(row['operational_error_rate_all'], 3)} & "
            f"{fmt(row['operational_error_rate_queried'], 3)} & "
            f"{fmt(row['error_enrichment'], 2)}$\\times$ & "
            f"{row['decision_value_one_sided_mannwhitney_p']:.2e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def hashes_table(data: dict) -> str:
    source = data["primary_lineage"]["source_test220"]["shortcut_audit"]["seeds"]
    artifacts = data["primary_lineage"]["artifact_hashes"]
    lines = [
        r"\begin{tabular}{rll}",
        r"\toprule",
        r"Seed & Source checkpoint SHA-256 & Online prediction SHA-256 \\",
        r"\midrule",
    ]
    for seed in SEEDS:
        src = source[str(seed)]["checkpoint_sha256"]
        pred = artifacts[str(seed)]["predictions.jsonl"]
        lines.append(
            f"{seed} & \\texttt{{{src[:16]}...{src[-8:]}}} & "
            f"\\texttt{{{pred[:16]}...{pred[-8:]}}} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def equal_label_budget_table(data: dict) -> str:
    direct = data["direct_selected_complement"]
    lines = [
        r"\begin{tabular}{rlllrrrr}",
        r"\toprule",
        r"Seed & Query D/N & Eval. D/N & Arm & TPR$_{30}$ & FPR & TPR$_{30}$--FPR & Lead (ms) \\",
        r"\midrule",
    ]
    arms = (
        ("static", "static"),
        ("front-load", "selected_frontload_frozen_threshold"),
        ("front-load, recal.", "selected_frontload_support110_recalibrated"),
        ("causal online", "online"),
    )
    for comparison in direct["comparisons"]:
        seed = int(comparison["seed"])
        query = comparison["query_label_counts"]
        evaluation = comparison["evaluation_class_counts"]
        query_counts = f"{query['disruptive']}/{query['safe']}"
        evaluation_counts = f"{evaluation['disruptive']}/{evaluation['safe']}"
        for index, (label, key) in enumerate(arms):
            row = comparison["arms"][key]
            lines.append(
                f"{seed if index == 0 else ''} & "
                f"{query_counts if index == 0 else ''} & "
                f"{evaluation_counts if index == 0 else ''} & "
                f"{label} & {fmt(row['tpr_at_30ms'])} & {fmt(row['fpr'])} & "
                f"{fmt(row['tpr30_minus_fpr'])} & {fmt(row['median_lead_ms'], 1)} \\\\"
            )
        if seed != SEEDS[-1]:
            lines.append(r"\addlinespace")
    lines.append(r"\midrule")
    for label, key in arms:
        row = direct["aggregate_equal_seed"][key]
        lines.append(
            f"mean & -- & -- & {label} & {fmt(row['tpr_at_30ms'])} & "
            f"{fmt(row['fpr'])} & {fmt(row['tpr30_minus_fpr'])} & "
            f"{fmt(row['median_lead_ms'], 1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def causal_random_table(data: dict) -> str:
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Seed & Queries & Active J & Random J mean & $\Delta J$ & Inter attempts A/R \\",
        r"\midrule",
    ]
    for seed in SEEDS:
        row = data["per_seed"][str(seed)]
        active = row["active"]
        random = row["random_mean_within_seed"]
        lines.append(
            f"{seed} & {active['query_count']} & "
            f"{fmt(active['utility_tpr30_minus_fpr'])} & "
            f"{fmt(random['utility_tpr30_minus_fpr'])} & "
            f"{fmt(row['active_minus_random_mean_utility'])} & "
            f"{fmt(active['shared_inter_attempts'], 1)}/{fmt(random['shared_inter_attempts'], 1)} \\\\"
        )
    aggregate = data["aggregate"]
    lines.append(r"\midrule")
    lines.append(
        f"mean & -- & {fmt(aggregate['active_utility_equal_seed_mean'])} & "
        f"{fmt(aggregate['random_utility_equal_seed_mean'])} & "
        f"{fmt(aggregate['active_minus_random_utility_mean'])} & "
        f"{fmt(aggregate['active_shared_inter_attempts_equal_seed_mean'], 1)}/"
        f"{fmt(aggregate['random_shared_inter_attempts_equal_seed_mean'], 1)} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def target_only_table(data: dict) -> str:
    arms = (
        ("target-only, source scaler", data["target_only_arms"]["target_only_matched17"]),
        ("strict EAST-only", data["target_only_arms"]["target_only_local17"]),
        (r"J-TEXT $\rightarrow$ EAST", data["transfer_static"]),
    )
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Arm & Seed & TPR$_{30}$ & FPR & TPR$_{30}$--FPR & Lead (ms) \\",
        r"\midrule",
    ]
    for arm_index, (label, arm) in enumerate(arms):
        for seed in SEEDS:
            row = arm["per_seed"][str(seed)]
            lines.append(
                f"{label} & {seed} & {fmt(row['operational_tpr'])} & "
                f"{fmt(row['fpr'])} & {fmt(row['utility_tpr30_minus_fpr'])} & "
                f"{fmt(row['operational_median_lead_ms'], 1)} \\\\"
            )
        mean = arm["equal_seed_mean"]
        lines.append(
            f"{label} & mean & {fmt(mean['operational_tpr'])} & "
            f"{fmt(mean['fpr'])} & {fmt(mean['utility_tpr30_minus_fpr'])} & "
            f"{fmt(mean['operational_median_lead_ms'], 1)} \\\\"
        )
        if arm_index != len(arms) - 1:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    data = json.loads(EVIDENCE.read_text())
    equal_budget = json.loads(EQUAL_BUDGET.read_text())
    causal_random = json.loads(CAUSAL_RANDOM.read_text())
    east_target_only = json.loads(EAST_TARGET_ONLY.read_text())
    builders = {
        "primary_per_seed.tex": primary_table,
        "time_blocks.tex": block_table,
        "shortcut_padfix.tex": shortcut_table,
        "padfix_transfer.tex": padfix_transfer_table,
        "negative_controls.tex": controls_table,
        "query_value.tex": query_table,
        "artifact_hashes.tex": hashes_table,
    }
    for name, builder in builders.items():
        write(name, builder(data))
    write("equal_label_budget.tex", equal_label_budget_table(equal_budget))
    write("causal_random_query.tex", causal_random_table(causal_random))
    write("east_target_only.tex", target_only_table(east_target_only))
    table_names = [
        *builders, "equal_label_budget.tex", "causal_random_query.tex",
        "east_target_only.tex",
    ]
    manifest = {
        "schema_version": "paper-latex-table-manifest-v2",
        "inputs": {
            str(EVIDENCE.relative_to(ROOT)): hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
            str(EQUAL_BUDGET.relative_to(ROOT)): hashlib.sha256(EQUAL_BUDGET.read_bytes()).hexdigest(),
            str(CAUSAL_RANDOM.relative_to(ROOT)): hashlib.sha256(CAUSAL_RANDOM.read_bytes()).hexdigest(),
            str(EAST_TARGET_ONLY.relative_to(ROOT)): hashlib.sha256(EAST_TARGET_ONLY.read_bytes()).hexdigest(),
        },
        "tables": {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in table_names
        },
    }
    write("table_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"generated": len(table_names), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
