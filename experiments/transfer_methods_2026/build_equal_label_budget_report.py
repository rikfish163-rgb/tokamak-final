#!/usr/bin/env python3
"""Build common-mask evidence for equal-label-budget front-load controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
sys.path.insert(0, str(HERE))

import grid_search as grid  # noqa: E402
import run_equal_label_budget_ablation as ablation  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_unified_online_active as unified  # noqa: E402


def objective(metrics: Mapping[str, Any]) -> float:
    return float(metrics["tpr_at_30ms"]) - float(metrics["fpr"])


def metric_bundle(
    *, records: Sequence[Mapping[str, Any]], key: str,
    outcomes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = unified.alarm_metrics(ablation.rows_for_metric(records, key), dict(outcomes))
    metrics["tpr30_minus_fpr"] = objective(metrics)
    labels = np.asarray([
        int(bool(outcomes[int(row["shot"])]["is_disrupt"])) for row in records
    ])
    scores = np.asarray([float(row["combined_max_probability"]) for row in records])
    metrics["auc"] = float(grid.auc(labels, scores))
    return metrics


def baseline_bundle(
    *, rows: Sequence[Mapping[str, Any]], outcomes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = unified.alarm_metrics(list(rows), dict(outcomes))
    metrics["tpr30_minus_fpr"] = objective(metrics)
    return metrics


def mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = ("tpr", "tpr_at_30ms", "fpr", "median_lead_ms", "tpr30_minus_fpr")
    result = {}
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = float(np.mean(values)) if values else float("nan")
    return result


def mean_within_seed_then_across(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Give every model seed equal weight regardless of random replicates."""

    per_seed = [
        mean_metrics([row for row in rows if int(row["seed"]) == seed])
        for seed in sorted({int(row["seed"]) for row in rows})
    ]
    return mean_metrics(per_seed)


def mean_delta_within_seed_then_across(
    comparisons: Sequence[Mapping[str, Any]], name: str,
) -> float:
    per_seed = []
    for seed in sorted({int(row["seed"]) for row in comparisons}):
        values = [
            float(row["primary_deltas"][name])
            for row in comparisons if int(row["seed"]) == seed
        ]
        per_seed.append(float(np.mean(values)))
    return float(np.mean(per_seed))


def build_adjudication(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit scientific verdict for the label-access ablation."""

    direct = report["direct_selected_complement"]
    direct_metrics = direct["aggregate_equal_seed"]
    direct_deltas = direct["aggregate_deltas"]
    random_metrics = report[
        "aggregate_equal_seed_after_averaging_random_replicates"
    ]
    fidelity = report["parent_fidelity"]
    return {
        "schema_version": "equal-label-budget-adjudication-v1",
        "status": report["status"],
        "question": (
            "Does front-loading the same labels queried by the causal online learner "
            "explain the online improvement?"
        ),
        "parent_fidelity_passed": bool(fidelity) and all(
            bool(row["passed"]) for row in fidelity.values()
        ),
        "primary_comparison": {
            "evaluation_rule": (
                "for each seed, remove that seed's queried shots from scoring; compare "
                "static, selected-query front-load, and causal online on the same "
                "query-label-disjoint complement"
            ),
            "metrics": direct_metrics,
            "deltas": direct_deltas,
            "all_three_seeds_frontload_j_worse_than_static": bool(
                direct["all_seed_selected_j_worse_than_static"]
            ),
            "all_three_seeds_online_j_better_than_frontload": bool(
                direct["all_seed_online_j_better_than_selected_frontload"]
            ),
        },
        "random_label_access_control": {
            "design": (
                "three label-agnostic, per-20-shot count-matched random front-load "
                "replicates per seed; replicate metrics are averaged within seed first"
            ),
            "metrics": {
                key: random_metrics[key]
                for key in (
                    "static", "selected_frontload_frozen_threshold",
                    "random_frontload_frozen_threshold", "online",
                )
            },
            "deltas": report["aggregate_primary_deltas_equal_seed"],
        },
        "verdict": (
            "The queried labels are informative, but ordinary front-loaded target "
            "refitting does not explain the online gain: it raises lead-qualified TPR "
            "while increasing FPR enough to reduce TPR30-FPR in every seed. The causal "
            "online path preserves a higher TPR30-FPR than the selected-label front-load "
            "in every seed."
        ),
        "verdict_zh": (
            "查询标签确实包含信息，但普通前置目标重训不能解释在线增益："
            "它提高了满足提前量要求的召回，同时把 FPR 推高到足以使每个 seed 的 "
            "TPR30-FPR 都下降；因果在线路径在每个 seed 上都保持了高于同查询标签"
            "前置重训的 TPR30-FPR。"
        ),
        "mechanistic_interpretation": (
            "The evidence is consistent with value from delayed selective updates, "
            "bounded step sizes, support replay, rollback, and alarm-state guards, rather "
            "than value from label count alone."
        ),
        "limitations": [
            "This is a post-selection analysis on balanced360, which informed method development.",
            "Label access is matched, but historical online optimizer steps, sampled windows, FLOPs, and RNG states were not archived; exact equal-gradient replay is unavailable.",
            "The random control is random front-loaded retraining, not a causal random-query online learner.",
            "With three model seeds, all-sign consistency is descriptive; the exact one-sided sign-test probability is 0.125 and is not confirmatory significance.",
        ],
        "limitations_zh": [
            "这是 balanced360 上的事后开发分析，该流参与过方法设计。",
            "标签访问量相同，但历史在线运行没有保存精确 optimizer steps、采样窗口、FLOPs 和 RNG 状态，无法重放完全等梯度预算的对照。",
            "随机对照是 random front-load，而不是因果 random-query 在线学习器。",
            "只有三个模型 seed；三次同向只构成描述性一致，单侧精确 sign-test 概率为 0.125，不能作为确认性显著性。",
        ],
    }


def adjudication_markdown(adjudication: Mapping[str, Any]) -> str:
    primary = adjudication["primary_comparison"]
    metrics = primary["metrics"]
    deltas = primary["deltas"]
    rows = []
    for label, key in (
        ("静态迁移", "static"),
        ("同查询标签前置重训", "selected_frontload_frozen_threshold"),
        ("因果在线主动", "online"),
    ):
        row = metrics[key]
        rows.append(
            f"| {label} | {row['tpr_at_30ms']:.4f} | {row['fpr']:.4f} | "
            f"{row['tpr30_minus_fpr']:.4f} | {row['median_lead_ms']:.1f} |"
        )
    limitations = "\n".join(
        f"- {item}" for item in adjudication["limitations_zh"]
    )
    return f"""# 同标签预算前置训练消融裁决

状态：`{adjudication['status']}`

## 研究问题

在线主动学习获得了一批 EAST 炮标签。若把完全相同的已查询炮在流开始前直接并入迁移训练，是否足以复现在线增益？

## 公平比较合同

- 三个 seed 的 support-only 重训均通过父模型位级哈希、阈值和静态预测复现门。
- 每个 seed 将该 seed 实际查询的炮从评分集合剔除；静态、前置重训和在线模型只在同一个标签不相交补集上比较。
- 主结果沿用原迁移阈值；用 support110 重新校准阈值的结果单独保存，且未挽救前置重训。
- 另做每 20 炮查询数匹配、完全不读取标签的随机前置对照，每 seed 三个重复，并先在 seed 内平均。

## 三 seed 等权均值

| 方法 | TPR@30 ms | FPR | TPR@30−FPR | 中位提前量 (ms) |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

同查询标签前置重训相对静态：`ΔTPR@30={deltas['selected_frontload_minus_static_tpr30']:+.4f}`，`ΔFPR={deltas['selected_frontload_minus_static_fpr']:+.4f}`，`Δ(TPR@30−FPR)={deltas['selected_frontload_minus_static_j']:+.4f}`。在线相对前置重训的 `Δ(TPR@30−FPR)={deltas['online_minus_selected_frontload_j']:+.4f}`。

## 裁决

{adjudication['verdict_zh']}

这说明查询标签本身有信息量——它提高了 TPR@30、AUC 和提前量——但“把更多标签一次性加到迁移训练”会把整体决策面推向破裂侧，误报增幅远大于召回增幅。当前证据更支持在线链路中的延迟选择、受限梯度、support replay、rollback 和报警状态保护共同贡献，而不是标签数量本身解释全部增益。

## 不能越界的解释

{limitations}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PACKAGE / "evidence/equal_label_budget_frontload_v2",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    out = (args.out or runs_root / "common_mask_report").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    per_comparison: list[dict[str, Any]] = []
    direct_selected_comparisons: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    direct_flat_rows: list[dict[str, Any]] = []
    parent_fidelity: dict[str, Any] = {}
    missing = []
    for seed in (131, 133, 137):
        support_dir = runs_root / f"seed{seed}/support_only"
        selected_dir = runs_root / f"seed{seed}/selected"
        random_dirs = sorted((runs_root / f"seed{seed}").glob("random_uniform_blocked_r*"))
        if (
            not (support_dir / "summary.json").is_file()
            or not (selected_dir / "summary.json").is_file()
            or not random_dirs
        ):
            missing.append(seed)
            continue
        support_records_all = ablation.load_jsonl(support_dir / "predictions.jsonl")
        selected_records_all = ablation.load_jsonl(selected_dir / "predictions.jsonl")
        fidelity = json.loads((support_dir / "parent_fidelity.json").read_text())
        if not bool(fidelity["passed"]):
            raise RuntimeError(f"seed{seed}: support-only parent fidelity failed")
        parent_fidelity[str(seed)] = fidelity
        online_dir = PACKAGE / f"evidence/primary_online_artifacts/seed{seed}"
        static_all = ablation.load_jsonl(online_dir / "static_predictions.jsonl")
        online_all = ablation.load_jsonl(online_dir / "predictions.jsonl")

        selected_summary = json.loads((selected_dir / "summary.json").read_text())
        selected_ids = [
            int(value) for value in selected_summary["partition"]["evaluation_ids"]
        ]
        if [int(row["shot"]) for row in selected_records_all] != selected_ids:
            raise RuntimeError("selected prediction roster differs from its complement")
        selected_outcomes = {
            shot: a1.read_outcome_after_seal(shot) for shot in selected_ids
        }
        direct_arms = {
            "static": baseline_bundle(
                rows=ablation.restrict_rows(static_all, selected_ids),
                outcomes=selected_outcomes,
            ),
            "online": baseline_bundle(
                rows=ablation.restrict_rows(online_all, selected_ids),
                outcomes=selected_outcomes,
            ),
            "support_only_refit_frozen_threshold": metric_bundle(
                records=ablation.restrict_rows(support_records_all, selected_ids),
                key="frozen_parent_threshold",
                outcomes=selected_outcomes,
            ),
            "selected_frontload_frozen_threshold": metric_bundle(
                records=selected_records_all,
                key="frozen_parent_threshold",
                outcomes=selected_outcomes,
            ),
            "selected_frontload_support110_recalibrated": metric_bundle(
                records=selected_records_all,
                key="support110_recalibrated_threshold",
                outcomes=selected_outcomes,
            ),
        }
        direct_deltas = {
            "selected_frontload_minus_static_j": (
                direct_arms["selected_frontload_frozen_threshold"]["tpr30_minus_fpr"]
                - direct_arms["static"]["tpr30_minus_fpr"]
            ),
            "selected_frontload_minus_static_tpr30": (
                direct_arms["selected_frontload_frozen_threshold"]["tpr_at_30ms"]
                - direct_arms["static"]["tpr_at_30ms"]
            ),
            "selected_frontload_minus_static_fpr": (
                direct_arms["selected_frontload_frozen_threshold"]["fpr"]
                - direct_arms["static"]["fpr"]
            ),
            "online_minus_static_j": (
                direct_arms["online"]["tpr30_minus_fpr"]
                - direct_arms["static"]["tpr30_minus_fpr"]
            ),
            "online_minus_selected_frontload_j": (
                direct_arms["online"]["tpr30_minus_fpr"]
                - direct_arms["selected_frontload_frozen_threshold"]["tpr30_minus_fpr"]
            ),
        }
        direct_selected_comparisons.append({
            "seed": seed,
            "evaluation_ids": selected_ids,
            "evaluation_ids_sha256": ablation.sha256_ids(selected_ids),
            "evaluation_class_counts": ablation.class_counts(
                selected_ids, selected_outcomes,
            ),
            "query_label_counts": selected_summary["label_budget"][
                "frontload_class_counts"
            ],
            "arms": direct_arms,
            "deltas": direct_deltas,
        })
        for arm_name, metrics in direct_arms.items():
            direct_flat_rows.append({
                "seed": seed,
                "common_shots": len(selected_ids),
                "arm": arm_name,
                **metrics,
            })

        for random_dir in random_dirs:
            random_summary = json.loads((random_dir / "summary.json").read_text())
            replicate = int(random_summary["random_replicate"])
            common_ids = [int(value) for value in random_summary["partition"]["evaluation_ids"]]
            outcomes = {shot: a1.read_outcome_after_seal(shot) for shot in common_ids}
            support_records = ablation.restrict_rows(support_records_all, common_ids)
            selected_records = ablation.restrict_rows(selected_records_all, common_ids)
            random_records = ablation.load_jsonl(random_dir / "predictions.jsonl")
            if [int(row["shot"]) for row in random_records] != common_ids:
                raise RuntimeError("random prediction roster differs from registered common mask")
            arms = {
                "static": baseline_bundle(
                    rows=ablation.restrict_rows(static_all, common_ids), outcomes=outcomes,
                ),
                "online": baseline_bundle(
                    rows=ablation.restrict_rows(online_all, common_ids), outcomes=outcomes,
                ),
                "support_only_refit_frozen_threshold": metric_bundle(
                    records=support_records, key="frozen_parent_threshold", outcomes=outcomes,
                ),
                "support_only_refit_support110_recalibrated": metric_bundle(
                    records=support_records,
                    key="support110_recalibrated_threshold",
                    outcomes=outcomes,
                ),
                "selected_frontload_frozen_threshold": metric_bundle(
                    records=selected_records, key="frozen_parent_threshold", outcomes=outcomes,
                ),
                "selected_frontload_support110_recalibrated": metric_bundle(
                    records=selected_records,
                    key="support110_recalibrated_threshold",
                    outcomes=outcomes,
                ),
                "random_frontload_frozen_threshold": metric_bundle(
                    records=random_records, key="frozen_parent_threshold", outcomes=outcomes,
                ),
                "random_frontload_support110_recalibrated": metric_bundle(
                    records=random_records,
                    key="support110_recalibrated_threshold",
                    outcomes=outcomes,
                ),
            }
            comparison = {
                "seed": seed,
                "random_replicate": replicate,
                "common_evaluation_ids": common_ids,
                "common_evaluation_ids_sha256": ablation.sha256_ids(common_ids),
                "common_class_counts": ablation.class_counts(common_ids, outcomes),
                "arms": arms,
                "primary_deltas": {
                    "selected_frontload_minus_static": (
                        arms["selected_frontload_frozen_threshold"]["tpr30_minus_fpr"]
                        - arms["static"]["tpr30_minus_fpr"]
                    ),
                    "selected_frontload_minus_support_only_refit": (
                        arms["selected_frontload_frozen_threshold"]["tpr30_minus_fpr"]
                        - arms["support_only_refit_frozen_threshold"]["tpr30_minus_fpr"]
                    ),
                    "random_frontload_minus_support_only_refit": (
                        arms["random_frontload_frozen_threshold"]["tpr30_minus_fpr"]
                        - arms["support_only_refit_frozen_threshold"]["tpr30_minus_fpr"]
                    ),
                    "online_minus_static": (
                        arms["online"]["tpr30_minus_fpr"]
                        - arms["static"]["tpr30_minus_fpr"]
                    ),
                    "selected_minus_random_frontload": (
                        arms["selected_frontload_frozen_threshold"]["tpr30_minus_fpr"]
                        - arms["random_frontload_frozen_threshold"]["tpr30_minus_fpr"]
                    ),
                    "online_minus_selected_frontload": (
                        arms["online"]["tpr30_minus_fpr"]
                        - arms["selected_frontload_frozen_threshold"]["tpr30_minus_fpr"]
                    ),
                },
            }
            per_comparison.append(comparison)
            for arm_name, metrics in arms.items():
                flat_rows.append({
                    "seed": seed,
                    "random_replicate": replicate,
                    "common_shots": len(common_ids),
                    "arm": arm_name,
                    **metrics,
                })

    if missing and not args.allow_partial:
        raise RuntimeError(f"missing completed selected/random arms for seeds {missing}")
    if not per_comparison:
        raise RuntimeError("no complete selected/random comparison is available")

    arm_names = sorted({row["arm"] for row in flat_rows})
    aggregate = {
        arm: mean_metrics([row for row in flat_rows if row["arm"] == arm])
        for arm in arm_names
    }
    aggregate_equal_seed = {
        arm: mean_within_seed_then_across(
            [row for row in flat_rows if row["arm"] == arm]
        )
        for arm in arm_names
    }
    delta_names = tuple(per_comparison[0]["primary_deltas"])
    aggregate_deltas = {
        name: float(np.mean([
            comparison["primary_deltas"][name] for comparison in per_comparison
        ]))
        for name in delta_names
    }
    aggregate_deltas_equal_seed = {
        name: mean_delta_within_seed_then_across(per_comparison, name)
        for name in delta_names
    }
    direct_arm_names = sorted({row["arm"] for row in direct_flat_rows})
    direct_aggregate = {
        arm: mean_metrics([row for row in direct_flat_rows if row["arm"] == arm])
        for arm in direct_arm_names
    }
    direct_delta_names = tuple(direct_selected_comparisons[0]["deltas"])
    direct_aggregate_deltas = {
        name: float(np.mean([
            row["deltas"][name] for row in direct_selected_comparisons
        ]))
        for name in direct_delta_names
    }
    report = {
        "schema_version": "equal-label-budget-common-mask-report-v2",
        "status": "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION",
        "completed_seeds": sorted({row["seed"] for row in per_comparison}),
        "missing_seeds": missing,
        "comparisons": per_comparison,
        "direct_selected_complement": {
            "comparisons": direct_selected_comparisons,
            "aggregate_equal_seed": direct_aggregate,
            "aggregate_deltas": direct_aggregate_deltas,
            "all_seed_selected_j_worse_than_static": all(
                row["deltas"]["selected_frontload_minus_static_j"] < 0
                for row in direct_selected_comparisons
            ),
            "all_seed_online_j_better_than_selected_frontload": all(
                row["deltas"]["online_minus_selected_frontload_j"] > 0
                for row in direct_selected_comparisons
            ),
        },
        "parent_fidelity": parent_fidelity,
        "aggregate_equal_weight_per_seed_replicate": aggregate,
        "aggregate_equal_seed_after_averaging_random_replicates": aggregate_equal_seed,
        "aggregate_primary_deltas": aggregate_deltas,
        "aggregate_primary_deltas_equal_seed": aggregate_deltas_equal_seed,
        "primary_threshold": "exact original fixed110 parent threshold",
        "claim_boundary": (
            "common-mask post-selection diagnostic; standard target refit is label-access "
            "matched but not optimizer-step/FLOP matched to historical online updates"
        ),
    }
    adjudication = build_adjudication(report)
    report["adjudication"] = adjudication
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "adjudication.json").write_text(
        json.dumps(adjudication, indent=2, sort_keys=True) + "\n"
    )
    (out / "ADJUDICATION_zh.md").write_text(
        adjudication_markdown(adjudication)
    )
    columns = [
        "seed", "random_replicate", "common_shots", "arm", "TP", "FP", "FN", "TN",
        "tpr", "tpr_at_30ms", "fpr", "median_lead_ms", "over_3000ms_count",
        "tpr30_minus_fpr", "auc",
    ]
    with (out / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    with (out / "direct_selected_complement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(direct_flat_rows)
    print(json.dumps({
        "completed_seeds": report["completed_seeds"],
        "missing_seeds": missing,
        "aggregate": aggregate,
        "deltas": aggregate_deltas,
        "direct_selected": report["direct_selected_complement"],
    }, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
