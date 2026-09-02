#!/usr/bin/env python3
"""Build a fail-closed attribution audit for the Adv25 online experiments.

The audit deliberately keeps three claims separate:

1. a complete online decision system improved an operational metric;
2. neural intra/inter-shot updates caused that improvement;
3. registered physics information caused the neural improvement.

The first claim does not imply the second, and the second does not imply the
third.  This script reads already sealed artifacts; it never trains a model or
selects a new operating threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


MODES = ("static", "intra_only", "inter_only", "intra_inter")
DEFAULT_FROZEN_EVIDENCE = (
    Path(__file__).resolve().parent
    / "online_architecture_search/candidate_table.json"
)
OUTCOME_FIELDS = {
    "is_disrupt", "outcome", "label", "revealed_is_disrupt",
    "down_time_s", "revealed_down_time_s",
}
ERROR_GROUPS = {"FP", "FN", "TARDY"}
EFFECT_EPS = 1e-9


def read_json(path: Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"missing or symlinked JSON artifact: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {candidate}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"missing or symlinked JSONL artifact: {candidate}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{candidate}:{number} is not a JSON object")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(target)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _forbidden_keys(value: Any, forbidden: set[str], *, path: str = "") -> list[str]:
    """Find outcome-bearing keys at any nesting depth in a ledger row."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}" if path else key_name
            if key_name in forbidden:
                found.append(child_path)
            found.extend(_forbidden_keys(child, forbidden, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden_keys(child, forbidden, path=f"{path}[{index}]"))
    return found


def _finite_number(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _metric_table(
    results: Mapping[str, Mapping[str, Any]], *, tpr_key: str,
) -> dict[str, dict[str, float]]:
    if set(results) != set(MODES):
        missing = sorted(set(MODES).difference(results))
        extra = sorted(set(results).difference(MODES))
        raise ValueError(f"factorial modes differ; missing={missing}, extra={extra}")
    table: dict[str, dict[str, float]] = {}
    for mode in MODES:
        row = results[mode]
        tpr = _finite_number(row[tpr_key], name=f"{mode}.{tpr_key}")
        fpr = _finite_number(row["fpr"], name=f"{mode}.fpr")
        auc = _finite_number(row["auc"], name=f"{mode}.auc")
        table[mode] = {
            "tpr": tpr,
            "fpr": fpr,
            "auc": auc,
            "objective_j": tpr - fpr,
        }
    return table


def factorial_effects(
    results: Mapping[str, Mapping[str, Any]], *, tpr_key: str,
) -> dict[str, dict[str, float]]:
    """Return the 2x2 main effects and interaction under one fixed contract."""

    table = _metric_table(results, tpr_key=tpr_key)
    output: dict[str, dict[str, float]] = {}
    for metric in ("tpr", "fpr", "auc", "objective_j"):
        static = table["static"][metric]
        intra = table["intra_only"][metric]
        inter = table["inter_only"][metric]
        joint = table["intra_inter"][metric]
        output[metric] = {
            "static": static,
            "intra_only": intra,
            "inter_only": inter,
            "intra_inter": joint,
            "intra_main": intra - static,
            "inter_main": inter - static,
            "joint_total": joint - static,
            "interaction": joint - intra - inter + static,
        }
    return output


def query_signature(rows: Iterable[Mapping[str, Any]]) -> tuple[tuple[int, int, bool], ...]:
    """Return an ordered, outcome-free query signature.

    A fixed-manifest ablation is invalid if its decision ledger embeds outcome
    fields, changes shot order, or omits a stream position.
    """

    signature: list[tuple[int, int, bool]] = []
    for expected, row in enumerate(rows):
        leaked = sorted(_forbidden_keys(row, OUTCOME_FIELDS))
        if leaked:
            raise ValueError(f"query ledger contains an outcome field: {leaked}")
        index = int(row["stream_index"])
        if index != expected:
            raise ValueError("query ledger is not strictly ordered from stream index zero")
        signature.append((index, int(row["shot"]), bool(row["selected"])))
    if len({shot for _index, shot, _selected in signature}) != len(signature):
        raise ValueError("query ledger contains duplicate shots")
    return tuple(signature)


def physics_ablation_verdict(
    physics_on: Mapping[str, Mapping[str, Any]],
    physics_off: Mapping[str, Mapping[str, Any]],
    *,
    tpr_key: str,
    minimum_j_advantage: float = 0.005,
    minimum_auc_advantage: float = 0.005,
) -> dict[str, Any]:
    """Judge whether registered physics terms show a measurable advantage.

    This is only a screen.  A publication claim additionally requires the
    advantage to repeat across seeds and beat time/channel-shuffle controls.
    """

    on = _metric_table(physics_on, tpr_key=tpr_key)
    off = _metric_table(physics_off, tpr_key=tpr_key)
    joint_delta_j = on["intra_inter"]["objective_j"] - off["intra_inter"]["objective_j"]
    joint_delta_auc = on["intra_inter"]["auc"] - off["intra_inter"]["auc"]
    supported = bool(
        joint_delta_j >= float(minimum_j_advantage)
        or joint_delta_auc >= float(minimum_auc_advantage)
    )
    return {
        "supported": supported,
        "joint_delta_j": joint_delta_j,
        "joint_delta_auc": joint_delta_auc,
        "minimum_j_advantage": float(minimum_j_advantage),
        "minimum_auc_advantage": float(minimum_auc_advantage),
        "per_mode": {
            mode: {
                "delta_tpr": on[mode]["tpr"] - off[mode]["tpr"],
                "delta_fpr": on[mode]["fpr"] - off[mode]["fpr"],
                "delta_auc": on[mode]["auc"] - off[mode]["auc"],
                "delta_j": on[mode]["objective_j"] - off[mode]["objective_j"],
            }
            for mode in MODES
        },
        "claim_boundary": (
            "zeroing registered online physics losses is not a sufficient causal "
            "mechanism test; real physics must also beat time reversal, temporal "
            "shuffle, family permutation, and equal-budget controls across seeds"
        ),
    }


def query_targeting(
    groups: Mapping[str, int], *, stream_error_count: int, stream_count: int,
) -> dict[str, Any]:
    counts = {str(name): int(value) for name, value in groups.items()}
    if any(value < 0 for value in counts.values()):
        raise ValueError("query group counts must be non-negative")
    queries = sum(counts.values())
    errors = sum(counts.get(group, 0) for group in ERROR_GROUPS)
    if queries <= 0 or stream_count <= 0 or not 0 <= stream_error_count <= stream_count:
        raise ValueError("invalid query/stream counts")
    query_rate = errors / queries
    stream_rate = int(stream_error_count) / int(stream_count)
    enrichment = query_rate / stream_rate if stream_rate > 0 else None
    return {
        "groups": counts,
        "queries": queries,
        "queried_errors": errors,
        "query_error_rate": query_rate,
        "stream_errors": int(stream_error_count),
        "stream_count": int(stream_count),
        "stream_error_rate": stream_rate,
        "enrichment": enrichment,
    }


def neural_effect_evidence(
    east_effects: Mapping[str, Mapping[str, Any]],
    jtext_effects: Mapping[str, Mapping[str, Any]],
    *,
    east_shots: int = 180,
    jtext_shots: int = 110,
    factorial_seed_count: int = 1,
) -> dict[str, Any]:
    """Separate an observed lattice-point effect from formal replication.

    Shot-level rates move on a discrete 1/N grid.  The first field answers
    ``did this single replay change at least one shot-equivalent of J?``.  The
    second field is intentionally stricter: it can only be true when the same
    factorial contract has been repeated across at least three seeds.  This
    prevents a historical headline run, which mixes mechanisms, from being
    used as replication evidence for a clean intra/inter ablation.
    """

    east_j = east_effects["objective_j"]
    jtext_j = jtext_effects["objective_j"]
    observed = {
        "intra": bool(
            east_j["intra_main"] >= 1.0 / east_shots - EFFECT_EPS
            or jtext_j["intra_main"] >= 1.0 / jtext_shots - EFFECT_EPS
        ),
        "inter": bool(
            east_j["inter_main"] >= 1.0 / east_shots - EFFECT_EPS
            or jtext_j["inter_main"] >= 1.0 / jtext_shots - EFFECT_EPS
        ),
    }
    repeated = bool(int(factorial_seed_count) >= 3)
    return {
        "single_seed_effect_observed": observed,
        "formal_supported_across_seeds": {
            "intra": bool(repeated and observed["intra"]),
            "inter": bool(repeated and observed["inter"]),
        },
        "factorial_seed_count": int(factorial_seed_count),
        "minimum_shot_equivalent": {
            "east": 1.0 / east_shots,
            "jtext": 1.0 / jtext_shots,
        },
    }


def _group_counts_from_updates(
    rows: Iterable[Mapping[str, Any]], *, family: str,
) -> Counter[str]:
    groups: Counter[str] = Counter()
    for row in rows:
        if family == "jtext":
            if row.get("mode") != "inter_only" or row.get("kind") != "inter":
                continue
            group = row.get("group")
        elif family == "east_two_scale":
            if row.get("mode") != "inter_only":
                continue
            group = row.get("error_group") or row.get("group")
        elif family == "v3c":
            if row.get("event") != "unified_revealed_finalize":
                continue
            group = row.get("learning_group") or row.get("group")
        else:
            raise ValueError(f"unknown update family: {family}")
        if group in {"TP", "TN", "FP", "FN", "TARDY"}:
            groups[str(group)] += 1
    return groups


def _operational_error_count(metrics: Mapping[str, Any], *, tpr_key: str) -> int:
    if "n_disruptions" in metrics:
        disruptions = int(metrics["n_disruptions"])
        safe_errors = int(metrics.get("FP", 0))
    else:
        confusion = metrics.get("confusion", {})
        disruptions = int(confusion.get("TP", 0)) + int(confusion.get("FN", 0))
        safe_errors = int(confusion.get("FP", 0))
    timely = _finite_number(metrics[tpr_key], name=tpr_key)
    disruption_errors = int(round(disruptions * (1.0 - timely)))
    return safe_errors + disruption_errors


def _v3c_effect(summary: Mapping[str, Any]) -> dict[str, Any]:
    static = summary["results"]["static_unified"]
    online = summary["results"]["online_unified"]
    static_j = float(static["tpr_at_30ms"]) - float(static["fpr"])
    online_j = float(online["tpr_at_30ms"]) - float(online["fpr"])
    return {
        "static": static,
        "online": online,
        "delta_tpr_at_30ms": float(online["tpr_at_30ms"]) - float(static["tpr_at_30ms"]),
        "delta_fpr": float(online["fpr"]) - float(static["fpr"]),
        "delta_j": online_j - static_j,
        "fp_removed": int(static["FP"]) - int(online["FP"]),
        "raw_tp_change": int(online["TP"]) - int(static["TP"]),
        "early_over_3000ms_change": (
            int(online["over_3000ms_count"]) - int(static["over_3000ms_count"])
        ),
    }


def _historical_v3c(evidence: Mapping[str, Any]) -> dict[str, Any]:
    static = evidence["official_results"]["east_static360"]["aggregate"]
    online = evidence["official_results"]["east_online360"]["aggregate"]
    static_tpr = float(static["operational_tpr"]["mean"])
    static_fpr = float(static["fpr"]["mean"])
    online_tpr = float(online["operational_tpr"]["mean"])
    online_fpr = float(online["fpr"]["mean"])
    return {
        "seeds": [131, 133, 137],
        "static_mean_tpr_at_30ms": static_tpr,
        "static_mean_fpr": static_fpr,
        "online_mean_tpr_at_30ms": online_tpr,
        "online_mean_fpr": online_fpr,
        "delta_tpr_at_30ms": online_tpr - static_tpr,
        "delta_fpr": online_fpr - static_fpr,
        "delta_j": (online_tpr - online_fpr) - (static_tpr - static_fpr),
        "negative_controls": evidence.get("negative_controls"),
        "claim_boundary": evidence.get("claim_boundary"),
    }


def _frozen_architecture_evidence(path: Path | None) -> dict[str, Any] | None:
    """Load the sealed frozen-v2.1 aggregate without mixing it into Adv25."""
    if path is None:
        return None
    candidate = path.expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        return {
            "available": False,
            "path": str(candidate),
            "reason": "frozen candidate table is unavailable",
        }
    payload = read_json(candidate)
    aggregate = payload.get("aggregate", {}).get("frozen_winner")
    if not isinstance(aggregate, Mapping):
        raise ValueError("frozen candidate table has no frozen_winner aggregate")
    return {
        "available": True,
        "path": str(candidate),
        "sha256": sha256_file(candidate),
        "claim_boundary": payload.get("claim_boundary"),
        "aggregate": aggregate,
        "winner": payload.get("winner"),
    }


def _validate_lineage(
    jtext: Mapping[str, Any], east_on: Mapping[str, Any],
    east_off: Mapping[str, Any], v3c: Mapping[str, Any],
) -> dict[str, Any]:
    # Do not stringify missing values before validation: the literal string
    # ``"None"`` can otherwise make a malformed artifact look well formed.
    source_values = [
        jtext.get("source_checkpoint_sha256"),
        east_on.get("source_checkpoint_sha256"),
        east_off.get("source_checkpoint_sha256"),
        v3c.get("hashes", {}).get("source_checkpoint"),
    ]
    if any(not isinstance(value, str) or not value.strip() for value in source_values):
        raise ValueError("Adv25 attribution artifacts contain a missing source checkpoint hash")
    source_hashes = {value.strip() for value in source_values}
    if len(source_hashes) != 1:
        raise ValueError(f"Adv25 attribution artifacts use different source hashes: {source_hashes}")
    seeds = {int(jtext["seed"]), int(east_on["seed"]), int(east_off["seed"]), int(v3c["seed"])}
    if len(seeds) != 1:
        raise ValueError(f"Adv25 attribution artifacts use different seeds: {seeds}")
    return {"seed": seeds.pop(), "source_checkpoint_sha256": source_hashes.pop()}


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    jtext_path = args.jtext_summary.expanduser().resolve()
    east_on_path = args.east_physics_on.expanduser().resolve()
    east_off_path = args.east_physics_off.expanduser().resolve()
    v3c_path = args.adv25_v3c.expanduser().resolve()
    jtext = read_json(jtext_path)
    east_on = read_json(east_on_path)
    east_off = read_json(east_off_path)
    v3c = read_json(v3c_path)
    lineage = _validate_lineage(jtext, east_on, east_off, v3c)

    east_query_on = query_signature(read_jsonl(east_on_path.parent / "query_decisions.jsonl"))
    east_query_off = query_signature(read_jsonl(east_off_path.parent / "query_decisions.jsonl"))
    if east_query_on != east_query_off:
        raise ValueError("physics-on/off runs do not use the same EAST query manifest")

    jtext_effects = factorial_effects(jtext["results"], tpr_key="timely_tpr")
    east_effects = factorial_effects(east_on["results"], tpr_key="tpr_at_30ms")
    physics = physics_ablation_verdict(
        east_on["results"], east_off["results"], tpr_key="tpr_at_30ms",
    )

    jtext_groups = _group_counts_from_updates(
        read_jsonl(jtext_path.parent / "updates.jsonl"), family="jtext",
    )
    east_groups = _group_counts_from_updates(
        read_jsonl(east_on_path.parent / "updates.jsonl"), family="east_two_scale",
    )
    v3c_groups = _group_counts_from_updates(
        read_jsonl(v3c_path.parent / "updates.jsonl"), family="v3c",
    )
    jtext_static = jtext["results"]["static"]
    east_static = east_on["results"]["static"]
    v3c_static = v3c["results"]["static_unified"]

    current_v3c = _v3c_effect(v3c)
    historical = (
        _historical_v3c(read_json(args.historical_evidence))
        if args.historical_evidence is not None else None
    )
    frozen = _frozen_architecture_evidence(args.frozen_evidence)
    jtext_query = query_targeting(
        jtext_groups,
        stream_error_count=_operational_error_count(jtext_static, tpr_key="timely_tpr"),
        stream_count=int(jtext_static["n_shots"]),
    )
    east_query = query_targeting(
        east_groups,
        stream_error_count=_operational_error_count(east_static, tpr_key="tpr_at_30ms"),
        stream_count=int(east_on["stream_shots"]),
    )
    v3c_query = query_targeting(
        v3c_groups,
        stream_error_count=(
            int(v3c_static["FP"])
            + int(round(180 * (1.0 - float(v3c_static["tpr_at_30ms"]))))
        ),
        stream_count=int(v3c["stream_shots"]),
    )

    neural_evidence = neural_effect_evidence(
        east_effects, jtext_effects,
        east_shots=int(east_on["stream_shots"]),
        jtext_shots=int(jtext_static["n_shots"]),
        factorial_seed_count=1,
    )
    formal_supported_across_seeds = neural_evidence["formal_supported_across_seeds"]
    neural_intra_supported = formal_supported_across_seeds["intra"]
    neural_inter_supported = formal_supported_across_seeds["inter"]
    query_intelligence_supported = bool(
        east_query["enrichment"] is not None
        and float(east_query["enrichment"]) >= 1.25
        and east_query["queried_errors"] > 0
    )
    decision_system_supported = bool(
        current_v3c["delta_j"] > 0.0 and current_v3c["fp_removed"] > 0
    )

    # These are protocol requirements, not claims that the current
    # diagnostic artifacts have already met them.  Keeping the checklist in
    # the JSON makes a later report auditable without relying on prose alone.
    formal_protocol = {
        "factorial_seed_minimum": 3,
        "future_stream_required": True,
        "same_contract_required": [
            "parent_checkpoint",
            "fold_local_scaler",
            "threshold",
            "alarm_state_machine",
            "query_manifest",
            "stream_order",
            "update_budget",
        ],
        "negative_controls_required": [
            "physics_zero",
            "physics_time_reversal",
            "within_shot_temporal_shuffle",
            "physics_family_permutation",
            "raw8_zero",
            "label_shift",
        ],
        "query_controls_required": [
            "random_equal_budget",
            "uncertainty_equal_budget",
            "physics_equal_budget",
            "expected_future_utility_equal_budget",
        ],
        "abnormal_stop": {
            "tpr_below": 0.60,
            "fpr_above": 0.40,
        },
        "current_pass": False,
        "current_fail_reasons": [
            "clean_factorial_has_one_seed",
            "balanced360_participated_in_method_development",
            "physics_off_is_not_the_full_negative_control_suite",
            "current_v3c_combines_neural_and_decision_layers",
        ],
    }

    audit = {
        "schema_version": "adv25-online-physics-attribution-audit-v4",
        "status": "DIAGNOSTIC_ONLY",
        "lineage": lineage,
        "artifacts": {
            "jtext_summary": {"path": str(jtext_path), "sha256": sha256_file(jtext_path)},
            "east_physics_on": {"path": str(east_on_path), "sha256": sha256_file(east_on_path)},
            "east_physics_off": {"path": str(east_off_path), "sha256": sha256_file(east_off_path)},
            "adv25_v3c": {"path": str(v3c_path), "sha256": sha256_file(v3c_path)},
        },
        "factorial": {
            "formula": {
                "intra_main": "M10-M00",
                "inter_main": "M01-M00",
                "joint_total": "M11-M00",
                "interaction": "M11-M10-M01+M00",
            },
            "jtext_test220": jtext_effects,
            "east_neural_path360": east_effects,
        },
        "physics_ablation": {
            **physics,
            "same_query_manifest": True,
            "query_count": sum(int(selected) for _index, _shot, selected in east_query_on),
            "static_prediction_hash_equal": (
                east_on["results"]["static"]["predictions_sha256"]
                == east_off["results"]["static"]["predictions_sha256"]
            ),
        },
        "query_targeting": {
            "jtext": jtext_query,
            "east_neural_path": east_query,
            "east_v3c": v3c_query,
        },
        "complete_v3c_system": {
            "adv25_seed131": current_v3c,
            "historical_three_seed": historical,
        },
        "lineage_comparison": {
            "frozen_v2_1": frozen,
            "comparison_rule": "report separately; never pool deltas across parent/alarm/label contracts",
        },
        "verdicts": {
            "complete_decision_system_gain_observed": decision_system_supported,
            "intra_shot_neural_gain_supported": neural_intra_supported,
            "inter_shot_neural_gain_supported": neural_inter_supported,
            "registered_online_physics_gain_supported": bool(physics["supported"]),
            "active_query_error_enrichment_supported": query_intelligence_supported,
            "physical_law_learning_supported": False,
            "independent_future_stream_confirmation": False,
        },
        "evidence_levels": {
            **neural_evidence,
            "formal_support_reason": (
                "The clean factorial artifacts contain only seed 131; historical "
                "three-seed V3C results are not a factorial neural attribution."
            ),
        },
        "formal_protocol": formal_protocol,
        "reasoning_boundary": {
            "complete_system": (
                "The Adv25 seed131 V3C development replay reduces false alarms, "
                "but it combines neural updates, endpoint memory, alarm-parameter "
                "updates, and an episode veto/delay head."
            ),
            "neural": (
                "The fixed-query 2x2 neural audit is the only valid source for "
                "intra/inter attribution; V3C headline deltas cannot be assigned "
                "to the Transformer or residual path."
            ),
            "physics": (
                "Physics-on and physics-off online losses do not change lead-qualified "
                "TPR or FPR in the current seed131 audit.  Physics-law learning is "
                "therefore not demonstrated."
            ),
            "evaluation": (
                "J-TEXT TEST220 and EAST balanced360 have participated in method "
                "development; all findings are diagnostic until a frozen future stream."
            ),
        },
    }
    return audit


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_markdown(audit: Mapping[str, Any]) -> str:
    jt = audit["factorial"]["jtext_test220"]
    east = audit["factorial"]["east_neural_path360"]
    v3c = audit["complete_v3c_system"]["adv25_seed131"]
    frozen = audit["lineage_comparison"]["frozen_v2_1"]
    physics = audit["physics_ablation"]
    queries = audit["query_targeting"]
    verdicts = audit["verdicts"]
    evidence = audit["evidence_levels"]
    protocol = audit["formal_protocol"]
    lines = [
        "# Adv25 在线主动学习物理归因与双时间尺度审查",
        "",
        "## 结论",
        "",
        "当前方法的因果访问顺序基本合理，但证据必须拆成三层：完整系统、神经更新、物理归因。",
        "Adv25 seed131 的完整 EAST V3C 在 balanced360 开发流上降低了 FPR；然而固定查询的神经四臂和物理开关实验没有把这一收益归因到炮内神经更新或物理损失。",
        "因此当前最严格的表述是：**受约束的在线决策系统有诊断增益；神经表示学习和物理规律学习尚未得到证明。**",
        "",
        "| 审查命题 | 裁决 |",
        "|---|---:|",
        f"| 完整 V3C 决策系统有开发流增益 | {_fmt(verdicts['complete_decision_system_gain_observed'])} |",
        f"| 炮内神经增益 | {_fmt(verdicts['intra_shot_neural_gain_supported'])} |",
        f"| 炮间神经增益 | {_fmt(verdicts['inter_shot_neural_gain_supported'])} |",
        f"| 在线物理损失带来可测增益 | {_fmt(verdicts['registered_online_physics_gain_supported'])} |",
        f"| 查询器具有错误富集能力 | {_fmt(verdicts['active_query_error_enrichment_supported'])} |",
        f"| 已证明学到物理规律 | {_fmt(verdicts['physical_law_learning_supported'])} |",
        f"| 已完成独立未来流确认 | {_fmt(verdicts['independent_future_stream_confirmation'])} |",
        "",
        f"单 seed 固定查询审计中，炮内是否观察到至少一个炮级 J 变化：{_fmt(evidence['single_seed_effect_observed']['intra'])}；炮间：{_fmt(evidence['single_seed_effect_observed']['inter'])}。这只是诊断证据，不是跨 seed 支持。当前干净 2×2 只有 {evidence['factorial_seed_count']} 个 seed，因此正式跨 seed 裁决仍为未通过。",
        "",
        "## 一、同父模型 2×2 炮内/炮间消融",
        "",
        "令 M00、M10、M01、M11 分别为 static、intra-only、inter-only、intra+inter。",
        "主效应为 ΔI=M10−M00、ΔR=M01−M00，交互项为 ΔIR=M11−M10−M01+M00。",
        "所有四臂必须共享 parent、阈值、查询清单、报警器和炮序。",
        "",
        "### J-TEXT TEST220（Adv25，J=TPR@10−FPR）",
        "",
        "| 因子 | ΔJ | ΔTPR@10 | ΔFPR |",
        "|---|---:|---:|---:|",
        f"| 炮内主效应 | {_fmt(jt['objective_j']['intra_main'], 6)} | {_fmt(jt['tpr']['intra_main'], 6)} | {_fmt(jt['fpr']['intra_main'], 6)} |",
        f"| 炮间主效应 | {_fmt(jt['objective_j']['inter_main'], 6)} | {_fmt(jt['tpr']['inter_main'], 6)} | {_fmt(jt['fpr']['inter_main'], 6)} |",
        f"| 联合总效应 | {_fmt(jt['objective_j']['joint_total'], 6)} | {_fmt(jt['tpr']['joint_total'], 6)} | {_fmt(jt['fpr']['joint_total'], 6)} |",
        f"| 交互项 | {_fmt(jt['objective_j']['interaction'], 6)} | {_fmt(jt['tpr']['interaction'], 6)} | {_fmt(jt['fpr']['interaction'], 6)} |",
        "",
        "J-TEXT 四臂的炮级指标相同；8295 次炮内尝试均被变化上限回退，炮内臂在功能上等同于静态臂。",
        "",
        "### EAST balanced360 神经残差路径（J=TPR@30−FPR）",
        "",
        "| 因子 | ΔJ | ΔTPR@30 | ΔFPR | ΔAUC |",
        "|---|---:|---:|---:|---:|",
        f"| 炮内主效应 | {_fmt(east['objective_j']['intra_main'], 6)} | {_fmt(east['tpr']['intra_main'], 6)} | {_fmt(east['fpr']['intra_main'], 6)} | {_fmt(east['auc']['intra_main'], 6)} |",
        f"| 炮间主效应 | {_fmt(east['objective_j']['inter_main'], 6)} | {_fmt(east['tpr']['inter_main'], 6)} | {_fmt(east['fpr']['inter_main'], 6)} | {_fmt(east['auc']['inter_main'], 6)} |",
        f"| 联合总效应 | {_fmt(east['objective_j']['joint_total'], 6)} | {_fmt(east['tpr']['joint_total'], 6)} | {_fmt(east['fpr']['joint_total'], 6)} | {_fmt(east['auc']['joint_total'], 6)} |",
        f"| 交互项 | {_fmt(east['objective_j']['interaction'], 6)} | {_fmt(east['tpr']['interaction'], 6)} | {_fmt(east['fpr']['interaction'], 6)} | {_fmt(east['auc']['interaction'], 6)} |",
        "",
        "炮间路径在单 seed replay 中只把一个破裂炮从 TARDY 推到 lead≥30 ms；原始 TP/FP/FN/TN 不变。该单炮变化在 physics-off 中同样出现，不能解释成物理损失收益；它也不足以构成跨 seed 神经增益。",
        "",
        "## 二、完整 V3C 与神经路径不能混称",
        "",
        "| 指标 | 静态 | 完整在线 | 差值 |",
        "|---|---:|---:|---:|",
        f"| TPR@30 | {_fmt(v3c['static']['tpr_at_30ms'])} | {_fmt(v3c['online']['tpr_at_30ms'])} | {_fmt(v3c['delta_tpr_at_30ms'])} |",
        f"| FPR | {_fmt(v3c['static']['fpr'])} | {_fmt(v3c['online']['fpr'])} | {_fmt(v3c['delta_fpr'])} |",
        f"| J | {_fmt(float(v3c['static']['tpr_at_30ms'])-float(v3c['static']['fpr']))} | {_fmt(float(v3c['online']['tpr_at_30ms'])-float(v3c['online']['fpr']))} | {_fmt(v3c['delta_j'])} |",
        f"| >3000 ms 过早报警数 | {v3c['static']['over_3000ms_count']} | {v3c['online']['over_3000ms_count']} | {v3c['early_over_3000ms_change']:+d} |",
        "",
        "完整 V3C 同时包含 fast/slow 神经更新、EAST-private raw8、端点记忆、四个报警参数和 delta60 延迟/否决头。",
        "故 7 个 FP 的减少是系统级收益，不是 Transformer 或物理损失的独立收益。",
        "",
        "### 与此前 frozen v2.1 主线的严格对照",
        "",
        "此前 frozen v2.1 的三 seed 汇总必须单独保留：static TPR@30=0.84074、FPR=0.16667；online TPR@30=0.84259、FPR=0.16667，ΔJ=+0.00185。它与 Adv25 clean/V3C 使用的 parent、报警器和实验合同不同，不能作为本轮神经 2×2 的重复。",
        f"frozen 证据表是否成功绑定：{_fmt(None if frozen is None else frozen.get('available'))}；其 SHA256 和路径已经写入 JSON lineage_comparison。",
        "",
        "## 三、物理负对照",
        "",
        f"physics-on 与 physics-off 使用同一查询清单：{physics['same_query_manifest']}，查询数 {physics['query_count']}。",
        f"联合臂 physics-on 相对 physics-off 的 ΔJ={_fmt(physics['joint_delta_j'], 6)}、ΔAUC={_fmt(physics['joint_delta_auc'], 6)}。",
        "当前结果未通过物理归因门。physics-off 只是第一层消融，不能单独证明物理信息的因果作用。下一轮必须在完全相同的优化预算下增加：物理时间反转、炮内时间打乱、物理族置换、raw8 置零和标签错位；真实物理臂需跨三个 seed 稳定优于最强负对照。",
        "",
        "## 四、查询是否具有智能性",
        "",
        "| 分支 | 查询数 | 查询错误率 | 全流错误率 | 富集倍数 |",
        "|---|---:|---:|---:|---:|",
        f"| J-TEXT | {queries['jtext']['queries']} | {_fmt(queries['jtext']['query_error_rate'])} | {_fmt(queries['jtext']['stream_error_rate'])} | {_fmt(queries['jtext']['enrichment'])} |",
        f"| EAST 神经四臂 | {queries['east_neural_path']['queries']} | {_fmt(queries['east_neural_path']['query_error_rate'])} | {_fmt(queries['east_neural_path']['stream_error_rate'])} | {_fmt(queries['east_neural_path']['enrichment'])} |",
        f"| EAST V3C | {queries['east_v3c']['queries']} | {_fmt(queries['east_v3c']['query_error_rate'])} | {_fmt(queries['east_v3c']['stream_error_rate'])} | {_fmt(queries['east_v3c']['enrichment'])} |",
        "",
        "EAST 神经路径有一定错误富集，但这只说明查询优于均匀抽样的方向，不等于查询具有未来纠错价值。",
        "真正的主动性必须用同预算 random、uncertainty-only、physics-only 和预期未来收益选择器比较，并统计每次查询对后续相似炮的净收益。",
        "",
        "## 五、正式门槛是否已经满足",
        "",
        f"当前协议状态：{_fmt(protocol['current_pass'])}。要求至少 {protocol['factorial_seed_minimum']} 个 seed 的 clean factorial、完整负对照套件和独立未来流；当前失败原因：{', '.join(protocol['current_fail_reasons'])}。异常停止规则为 TPR<{protocol['abnormal_stop']['tpr_below']:.2f} 或 FPR>{protocol['abnormal_stop']['fpr_above']:.2f}。",
        "",
        "只有在相同 parent、scaler、阈值、报警器、query manifest、炮序和更新预算下，才允许计算炮内/炮间主效应。",
        "",
        "## 六、下一轮唯一合理的正式矩阵",
        "",
        "1. 固定 Adv25 静态源 checkpoint、EAST110 parent、阈值、报警器、炮序和查询 manifest。",
        "2. 运行 static、intra-only、inter-only、intra+inter，先证明神经路径本身有收益。",
        "3. 在同一神经轨迹上正交比较 alarm-only、neural-only、neural+alarm，禁止把决策保护算作表示学习。",
        "4. 对 neural-only 重复 real-physics、zero、time-reverse、within-shot-shuffle、family-permutation。",
        "5. 对查询重复 active、equal-budget random、uncertainty-only、physics-only；每臂更新次数、窗口数和 FLOPs 相同。",
        "6. seeds 131/133/137 仅用于开发流稳定性；配置冻结后必须在新 EAST chronological stream 一次性确认。",
        "",
        "## 七、论文表述边界",
        "",
        "可以写：在 balanced360 开发流上，受约束 V3C 系统降低了误报，并保持 lead-qualified TPR。",
        "不能写：在线模型已经学习到新的破裂物理规律，或该结果已经得到独立未来流验证。",
        "",
        "机器可读证据见同目录 `attribution_audit.json`。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jtext-summary", type=Path, required=True)
    parser.add_argument("--east-physics-on", type=Path, required=True)
    parser.add_argument("--east-physics-off", type=Path, required=True)
    parser.add_argument("--adv25-v3c", type=Path, required=True)
    parser.add_argument("--historical-evidence", type=Path, default=None)
    parser.add_argument("--frozen-evidence", type=Path, default=DEFAULT_FROZEN_EVIDENCE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite attribution audit: {output}")
    output.mkdir(parents=True)
    audit = build_audit(args)
    atomic_json(output / "attribution_audit.json", audit)
    atomic_text(output / "ATTRIBUTION_AUDIT_CN.md", render_markdown(audit))
    print(json.dumps(audit["verdicts"], indent=2, sort_keys=True))
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
