#!/usr/bin/env python3
"""Build the reproducible, cross-lineage J-TEXT -> EAST result receipt.

The project has deliberately retained several model families and several
online mechanisms.  This script does not recompute metrics and it never joins
rows from incompatible checkpoint lineages.  It only reads already sealed
JSON receipts, checks their minimum schemas, records their SHA-256 digests,
and emits one machine-readable index plus a concise Chinese hand-off report.

The generated report is diagnostic: TEST220 and balanced360 were used for
method development.  It is therefore not a claim of independent future
confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


SEEDS = (131, 133, 137)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def finite_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite: {value!r}")
    return float(value)


def metric_row(
    payload: Mapping[str, Any], *, source: str, stage: str,
    tpr_key: str, fpr_key: str, auc_key: str = "auc",
    lead_key: str = "median_lead_ms",
    counts: bool = True,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source": source,
        "stage": stage,
        "tpr": finite_number(payload[tpr_key], label=f"{source}.{stage}.tpr"),
        "fpr": finite_number(payload[fpr_key], label=f"{source}.{stage}.fpr"),
        "auc": (
            finite_number(payload[auc_key], label=f"{source}.{stage}.auc")
            if auc_key in payload else None
        ),
        "median_lead_ms": (
            finite_number(payload[lead_key], label=f"{source}.{stage}.median_lead_ms")
            if lead_key in payload else None
        ),
    }
    values["J"] = values["tpr"] - values["fpr"]
    if counts:
        for name in ("TP", "FP", "FN", "TN"):
            if name in payload:
                values[name] = int(payload[name])
        if "early_over_3000_ms_count" in payload:
            values["early_over_3000_ms_count"] = int(
                payload["early_over_3000_ms_count"]
            )
        elif "over_3000ms_count" in payload:
            values["early_over_3000_ms_count"] = int(
                payload["over_3000ms_count"]
            )
    return values


def delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("tpr", "fpr", "auc", "median_lead_ms", "J"):
        if candidate.get(key) is None or baseline.get(key) is None:
            continue
        result[key] = float(candidate[key]) - float(baseline[key])
    return result


def operational_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the two historical lead-field spellings used by receipts."""
    row = dict(payload)
    if "median_lead_ms" not in row:
        lead = row.get("operational_tp_lead_ms")
        if not isinstance(lead, Mapping) or "median_ms" not in lead:
            raise ValueError("operational receipt lacks a median lead")
        row["median_lead_ms"] = lead["median_ms"]
    return row


def arithmetic_mean(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("tpr", "fpr", "auc", "median_lead_ms", "J"):
        if all(row.get(key) is not None for row in rows):
            result[key] = mean(float(row[key]) for row in rows)
    return result


def source_metric_path(root: Path, seed: int) -> Path:
    candidates = (
        root / "fuison_ext4_runs" / "v3c_adv25_exact_20260830"
        / f"seed{seed}" / "source" / "test220_metrics.json",
        root / "fuison_ext4_runs" / "jtext_adv25_sources_20260830"
        / f"seed{seed}" / "test220_metrics.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no J-TEXT TEST220 source receipt for seed {seed}: "
        + ", ".join(str(path) for path in candidates)
    )


def build_payload(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    artifacts = root / "fuison_ext4_runs"
    input_paths: dict[str, Path] = {}

    source_rows: list[dict[str, Any]] = []
    source_checkpoints: dict[str, str] = {}
    for seed in SEEDS:
        path = source_metric_path(root, seed)
        input_paths[f"jtext_source_seed{seed}"] = path
        receipt = load_json(path)
        metrics = receipt.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"source receipt lacks metrics: {path}")
        row = metric_row(
            metrics, source=f"jtext_source_seed{seed}", stage="TEST220_static",
            tpr_key="timely_tpr", fpr_key="fpr", counts=False,
        )
        row["seed"] = seed
        row["checkpoint_sha256"] = str(receipt.get("checkpoint_sha256", ""))
        row["label_authority"] = receipt.get("label_authority")
        row["threshold_selection_split"] = receipt.get("threshold_selection_split")
        source_rows.append(row)
        source_checkpoints[str(seed)] = row["checkpoint_sha256"]

    canonical_jtext_path = (
        artifacts / "jtext_adv25_active_canonical_replay_20260831"
        / "budget20" / "summary.json"
    )
    input_paths["jtext_active_budget20"] = canonical_jtext_path
    canonical_jtext = load_json(canonical_jtext_path)
    jtext_static = metric_row(
        operational_row(canonical_jtext["static"]), source="jtext_active_canonical", stage="static",
        tpr_key="operational_tpr", fpr_key="fpr",
    )
    jtext_online = metric_row(
        operational_row(canonical_jtext["online"]), source="jtext_active_canonical", stage="online",
        tpr_key="operational_tpr", fpr_key="fpr",
    )
    jtext_active = {
        "budget_percent": int(canonical_jtext.get("budget_percent", 20)),
        "query_count": int(canonical_jtext["query_count"]),
        "n_shots": int(canonical_jtext["shot_count"]),
        "static": jtext_static,
        "online": jtext_online,
        "delta": delta(jtext_online, jtext_static),
        "representation_updated": bool(canonical_jtext.get("representation_updated", False)),
        "parent": canonical_jtext.get("east_transfer_parent"),
        "minimum_lead_rule": canonical_jtext["static"].get("minimum_lead_rule"),
        "claim_status": canonical_jtext.get("claim_status"),
    }

    jtext_neural_path = (
        artifacts / "jtext_episodic_online_20260831"
        / "seed131_b20_guarded" / "summary.json"
    )
    input_paths["jtext_neural_diagnostic"] = jtext_neural_path
    jtext_neural = load_json(jtext_neural_path)
    jtext_neural_static = metric_row(
        jtext_neural["static_metrics"], source="jtext_neural_episode", stage="static",
        tpr_key="operational_tpr", fpr_key="fpr",
    )
    jtext_neural_online = metric_row(
        jtext_neural["online_metrics"], source="jtext_neural_episode", stage="online",
        tpr_key="operational_tpr", fpr_key="fpr",
    )
    jtext_neural_active = {
        "seed": int(jtext_neural["seed"]),
        "query_count": int(jtext_neural["query_contract"]["selected"]),
        "static": jtext_neural_static,
        "online": jtext_neural_online,
        "delta": delta(jtext_neural_online, jtext_neural_static),
        "source_parameters_updated": bool(
            jtext_neural["frozen_source_integrity"]["source_parameters_updated"]
        ),
        "representation_path_trainable": bool(
            jtext_neural["adapter_contract"]["representation_path_trainable"]
        ),
        "claim_boundary": jtext_neural.get("claim_boundary"),
    }

    east_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = (
            artifacts / "adv25_v3c_success_shotfirst_replay_20260831"
            / f"seed{seed}" / "final_episode_v3" / "summary.json"
        )
        input_paths[f"east_v3c_seed{seed}"] = path
        receipt = load_json(path)
        results = receipt.get("results")
        if not isinstance(results, dict):
            raise ValueError(f"EAST receipt lacks results: {path}")
        static = metric_row(
            results["static_unified"], source=f"east_v3c_seed{seed}", stage="static360",
            tpr_key="tpr_at_30ms", fpr_key="fpr", counts=True,
        )
        online = metric_row(
            results["online_unified"], source=f"east_v3c_seed{seed}", stage="online360",
            tpr_key="tpr_at_30ms", fpr_key="fpr", counts=True,
        )
        east_rows.append({
            "seed": seed,
            "static": static,
            "online": online,
            "delta": delta(online, static),
            "query_count": int(receipt.get("query_count", -1)),
            "source_checkpoint_sha256": receipt.get("hashes", {}).get("source_checkpoint"),
            "support_parent_sha256": receipt.get("hashes", {}).get("base_before_after"),
            "claim_status": receipt.get("claim_status"),
        })

    east_static = [row["static"] for row in east_rows]
    east_online = [row["online"] for row in east_rows]
    east_static_mean = arithmetic_mean(east_static)
    east_online_mean = arithmetic_mean(east_online)
    east_summary = {
        "per_seed": east_rows,
        "arithmetic_mean": {
            "static": east_static_mean,
            "online": east_online_mean,
            "delta": {
                key: east_online_mean[key] - east_static_mean[key]
                for key in east_online_mean
                if key in east_static_mean
            },
        },
        "metric_contract": {
            "tpr": "TPR@30ms; lead >= 30 ms; lead > 3000 ms remains counted and is separately reported",
            "fpr": "safe-shot alarm fraction",
            "J": "TPR@30ms - FPR",
        },
    }

    attribution_path = (
        artifacts / "east_v3c_timescale_ablation_20260831"
        / "seed131" / "aggregate.json"
    )
    input_paths["east_seed131_attribution"] = attribution_path
    attribution = load_json(attribution_path)
    if attribution.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("unexpected attribution status")

    optional_budget_rows: list[dict[str, Any]] = []
    for budget_name in ("budget10_current", "budget30_current"):
        path = artifacts / "jtext_adv25_active_canonical_replay_20260831" / budget_name / "summary.json"
        if not path.is_file():
            continue
        input_paths[f"jtext_active_{budget_name}"] = path
        receipt = load_json(path)
        optional_budget_rows.append({
            "name": budget_name,
            "budget_percent": int(receipt.get("budget_percent", 0)),
            "query_count": int(receipt.get("query_count", -1)),
            "static": metric_row(operational_row(receipt["static"]), source=budget_name, stage="static", tpr_key="operational_tpr", fpr_key="fpr"),
            "online": metric_row(operational_row(receipt["online"]), source=budget_name, stage="online", tpr_key="operational_tpr", fpr_key="fpr"),
        })

    payload: dict[str, Any] = {
        "schema_version": "jtext-east-mainline-receipt-v2",
        "material_passport": {
            "id": "JTEXT-EAST-MAINLINE-20260831",
            "type": "experiment_result_receipt",
            "status": "ANALYZED",
            "scope": "sealed JSON result receipts only; no metric recomputation from raw HDF5",
        },
        "claim_status": "DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION",
        "claim_boundary": (
            "TEST220 and balanced360 participated in method or parameter development; "
            "these results are not independent future-stream confirmation"
        ),
        "label_contract": {
            "jtext_positive_window": "max(25 ms, AdvancedTime_ms); missing AdvancedTime -> 25 ms",
            "jtext_outcome_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
            "east_outcome_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
            "jtext_operational_lead": "> 10 ms",
            "east_operational_lead": ">= 30 ms",
            "over_3000_ms_rule": "counted as TPR and separately reported",
        },
        "lineage_contract": {
            "jtext_online_exported_to_east": False,
            "east_parent": "raw static J-TEXT checkpoint -> EAST support110 adaptation",
            "jtext_source_checkpoint_sha256": source_checkpoints,
            "east_source_profile": "east_success (separately registered per seed)",
            "source_profile_warning": "jtext_exact and east_success are not silently merged",
        },
        "jtext_source_test220": {
            "per_seed": source_rows,
            "arithmetic_mean": arithmetic_mean(source_rows),
        },
        "jtext_online_decision_level": jtext_active,
        "jtext_online_neural_diagnostic": jtext_neural_active,
        "jtext_online_optional_budgets": optional_budget_rows,
        "east_v3c_mainline": east_summary,
        "east_seed131_component_attribution": attribution,
        "input_receipts": {},
    }
    for name, path in sorted(input_paths.items()):
        payload["input_receipts"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return payload, input_paths


def markdown(payload: Mapping[str, Any]) -> str:
    source = payload["jtext_source_test220"]
    east = payload["east_v3c_mainline"]
    jtext = payload["jtext_online_decision_level"]
    neural = payload["jtext_online_neural_diagnostic"]
    attribution = payload["east_seed131_component_attribution"]

    def fmt(value: Any, digits: int = 6) -> str:
        return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else str(value)

    lines = [
        "# J-TEXT → EAST 主线结果回执（2026-08-31）",
        "",
        "## Material Passport",
        "",
        "- **ID：** `JTEXT-EAST-MAINLINE-20260831`",
        "- **类型：** sealed experiment-result receipt",
        "- **状态：** `ANALYZED`；总体声明状态为 `DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`",
        "- **数据边界：** TEST220 与 balanced360 已参与方法或参数开发，不能写成独立未来确认集。",
        "- **来源：** 本文由已封存 JSON 回执生成；每个输入文件的 SHA-256 均记录在同目录 `mainline_results.json`。",
        "",
        "## 1. 主线与不可混用边界",
        "",
        "```text",
        "J-TEXT TRAIN1353 (Adv25 label)",
        "  ├─ jtext_exact static source → J-TEXT TEST220 static / online-active (独立分支)",
        "  └─ east_success static source → EAST support110 transfer → EAST balanced360 V3C online",
        "```",
        "",
        "J-TEXT 在线状态、查询记录和权重没有导出到 EAST。EAST 迁移只从登记的静态",
        "J-TEXT checkpoint 开始；`jtext_exact` 与 `east_success` 是不同的 checkpoint profile，",
        "不能把两条分支的指标拼成同一个 SHA 的端到端结果。",
        "",
        "J-TEXT 破裂炮的正样本窗为 "
        "`H=max(25 ms, AdvancedTime_ms)`；缺失值取 25 ms。两台装置的结局均以 HDF5",
        "`meta/IsDisrupt`、`meta/DownTime` 为准。",
        "",
        "## 2. 三阶段主结果",
        "",
        "### 2.1 J-TEXT 静态源模型（TEST220）",
        "",
        "| seed | timely TPR (lead >10 ms) | FPR | AUC | median lead (ms) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in source["per_seed"]:
        lines.append(
            f"| {row['seed']} | {fmt(row['tpr'])} | {fmt(row['fpr'])} | "
            f"{fmt(row['auc'])} | {fmt(row['median_lead_ms'], 3)} |"
        )
    avg = source["arithmetic_mean"]
    lines += [
        f"| **算术均值** | **{fmt(avg['tpr'])}** | **{fmt(avg['fpr'])}** | "
        f"**{fmt(avg['auc'])}** | **{fmt(avg['median_lead_ms'], 3)}** |",
        "",
        "该表是三颗静态源 checkpoint 的 J-TEXT TEST220 结果；它不是 EAST 迁移后的",
        "目标域结果。TEST220 已用于在线方法开发，因此这里只作当前源域基线记录。",
        "",
        "### 2.2 J-TEXT 独立在线主动分支（决策层 canonical）",
        "",
        "| 视图 | 查询炮数 | TPR (lead >10 ms) | FPR | AUC | J=TPR−FPR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| static | 0 | {fmt(jtext['static']['tpr'])} | {fmt(jtext['static']['fpr'])} | {fmt(jtext['static']['auc'])} | {fmt(jtext['static']['J'])} |",
        f"| online active (20%) | {jtext['query_count']}/{jtext['n_shots']} | {fmt(jtext['online']['tpr'])} | {fmt(jtext['online']['fpr'])} | {fmt(jtext['online']['auc'])} | {fmt(jtext['online']['J'])} |",
        f"| online − static | {jtext['query_count']} | {fmt(jtext['delta']['tpr'])} | {fmt(jtext['delta']['fpr'])} | {fmt(jtext['delta']['auc'])} | {fmt(jtext['delta']['J'])} |",
        "",
        "这一分支的已观测增益来自查询揭示后的专家权重/决策层反馈；回执明确",
        f"`representation_updated={jtext['representation_updated']}`，所以不能称为 Transformer 表示在线学习。",
        "它不进入 EAST 迁移。",
        "",
        "### 2.3 EAST 静态迁移 → V3C 在线主动（balanced360）",
        "",
        "| seed | static TPR@30/FPR | online TPR@30/FPR | ΔTPR | ΔFPR | ΔJ | queries |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in east["per_seed"]:
        lines.append(
            f"| {row['seed']} | {fmt(row['static']['tpr'])}/{fmt(row['static']['fpr'])} | "
            f"{fmt(row['online']['tpr'])}/{fmt(row['online']['fpr'])} | "
            f"{fmt(row['delta']['tpr'])} | {fmt(row['delta']['fpr'])} | "
            f"{fmt(row['delta']['J'])} | {row['query_count']} |"
        )
    ea = east["arithmetic_mean"]
    lines += [
        f"| **算术均值** | **{fmt(ea['static']['tpr'])}/{fmt(ea['static']['fpr'])}** | "
        f"**{fmt(ea['online']['tpr'])}/{fmt(ea['online']['fpr'])}** | "
        f"**{fmt(ea['delta']['tpr'])}** | **{fmt(ea['delta']['fpr'])}** | "
        f"**{fmt(ea['delta']['J'])}** | **{fmt(mean(row['query_count'] for row in east['per_seed']), 2)}** |",
        "",
        "EAST 的操作口径为 `lead >=30 ms`；提前量大于 3000 ms 仍计入 TPR，",
        "但在原始回执中单独记录。balanced360 先完成静态封存，再从同一迁移父模型",
        "重置后按炮号运行在线流；结果是完整 V3C 系统的开发诊断。",
        "",
        "## 3. EAST seed131 同一 parent 归因筛查",
        "",
        "该筛查固定同一 support110 parent、阈值、炮序和 94 炮 query manifest。",
        "它回答的是在线变化来自哪一层，不是重新选择最佳参数。",
        "",
        "| 臂 | 炮内神经 | 炮间神经 | 决策/记忆侧 | online TPR@30 | online FPR | ΔJ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in attribution["runs"]:
        online = row["online"]
        lines.append(
            f"| {row['name']} | {'是' if row.get('neural_intra') else '否'} | "
            f"{'是' if row.get('neural_inter') else '否'} | "
            f"{'是' if row.get('decision_side_updates') else '否'} | "
            f"{fmt(online['tpr_at_30ms'])} | {fmt(online['fpr'])} | "
            f"{fmt(row['delta']['J'])} |"
        )
    lines += [
        "",
        "seed131 的关键解释：炮内 fast-state 虽执行反向传播，但本轮没有改变炮级",
        "报警边界；固定计算量的 pure-neural 对照也没有改变边界。禁用炮间神经更新、",
        "保留查询后的报警/episode/endpoint-memory 侧更新，仍复现一个 FP 的下降。",
        "因此目前最稳妥的结论是：旧 V3C 的可见 FPR 增益主要由决策/记忆侧路径贡献；",
        "现有证据不足以声称 Transformer 已学习新的跨装置破裂物理。",
        "",
        "## 4. J-TEXT 神经 episode 诊断",
        "",
        f"guarded local-neural episode：static TPR={fmt(neural['static']['tpr'])}, "
        f"FPR={fmt(neural['static']['fpr'])}; online TPR={fmt(neural['online']['tpr'])}, "
        f"FPR={fmt(neural['online']['fpr'])}; ΔJ={fmt(neural['delta']['J'])}。",
        "该适配器的表示路径可训练，但源 encoder/head 被锁定；本轮没有产生炮级边界",
        "增益，故不进入 canonical 在线主结果，更不进入 EAST parent。",
        "",
        "## 5. 论文可表述范围",
        "",
        "可以表述：",
        "",
        "- 在 Adv25 标签合同和 HDF5 结局权威下，J-TEXT 源模型、EAST 少样本迁移和",
        "  EAST 按炮序 V3C 在线主动流均有可复现的封存回执。",
        "- V3C 是包含炮内无标签适应、炮间监督更新和查询后决策保护的系统；在开发流",
        "  上三 seed 算术均值的 TPR@30 从 "
        f"{fmt(ea['static']['tpr'])} 提升到 {fmt(ea['online']['tpr'])}，FPR 从 "
        f"{fmt(ea['static']['fpr'])} 降到 {fmt(ea['online']['fpr'])}。",
        "- 同一 parent 的 seed131 归因表明，系统级增益不能单独归因于神经表示更新。",
        "",
        "不能表述：",
        "",
        "- 不能把 TEST220/balanced360 结果写成完全未见的未来确认；",
        "- 不能把 J-TEXT 决策层专家权重增益写成 Transformer 表示学习；",
        "- 不能声称模型已经识别所有破裂因果机理；",
        "- 不能把不同 source profile、不同 support sampler 或不同报警器结果拼接。",
        "",
        "## 6. 下一步唯一合理动作",
        "",
        "锁定 source checkpoint SHA、support parent SHA、阈值、报警器、query manifest 和",
        "代码版本，在完全未参与方法选择的 EAST chronological future stream 上一次性运行；",
        "同时保留同一 parent 的 static、side-only、neural-only 和 full 配对臂。若没有独立",
        "未来流，本文结果应继续标记为诊断，不升级为 `T0_ADMITTED`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
        help="default: fuison_ext4_runs/mainline_receipt_20260831",
    )
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "fuison_ext4_runs" / "mainline_receipt_20260831"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty receipt directory: {output_dir}")
    payload, _ = build_payload(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mainline_results.json"
    md_path = output_dir / "MAINLINE_RESULTS_CN.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(payload), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
