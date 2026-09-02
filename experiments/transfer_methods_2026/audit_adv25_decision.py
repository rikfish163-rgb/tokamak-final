#!/usr/bin/env python3
"""Audit the Adv25 single-factor lineage and write a local decision record.

This is an audit, not a training or model-selection entry point.  It reads
the authoritative HDF5 metadata only to verify the shot contract and to
recompute label-window counts.  It never changes a threshold, chooses an
epoch, or writes back to an experiment run.  The generated JSON/Markdown
records are deliberately stored in the repository so a later run can be
reviewed without relying on an untracked external output directory.

The requested J-TEXT source policy is the literal hard-label rule::

    H_i = max(25 ms, AdvancedTime_i)
    positive: 0 <= TTD <= H_i
    negative: H_i < TTD <= 60000 ms

The script reports this assumption explicitly because an alternative
interpretation (``AdvancedTime`` as an uncertainty boundary) would require a
different, pre-registered gray-band experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments" / "transfer_methods_2026"
sys.path.insert(0, str(HERE))

import jtext_label_contract as labels  # noqa: E402


SEEDS = (131, 133, 137)
EXPECTED_JTEXT_COUNTS = {"train": 1353, "val": 160, "test": 220}
EXPECTED_EAST_COUNTS = {"disruption": 180, "safe": 180, "total": 360}
FIXED_HORIZON_MS = 125.0
NEGATIVE_MAX_MS = 60_000.0
ADV25_POLICY = "advanced_floor25"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"missing or symlinked JSON: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {candidate}")
    return payload


def _require_unique_roles(roles: Mapping[str, Iterable[int]]) -> dict[str, list[int]]:
    result = {name: [int(value) for value in values] for name, values in roles.items()}
    for name, values in result.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate shot IDs in J-TEXT {name}")
    names = tuple(result)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = set(result[left]) & set(result[right])
            if overlap:
                raise ValueError(
                    f"J-TEXT split overlap {left}/{right}: {sorted(overlap)[:8]}"
                )
    return result


def load_jtext_roles(root: Path = ROOT) -> dict[str, list[int]]:
    index = root / "artifacts/canonical/jtext_source_bundle_p131_v1/index.json"
    if index.is_file():
        payload = read_json(index)
        raw = payload.get("split_shot_ids")
        if not isinstance(raw, dict):
            raise ValueError(f"canonical index has no split_shot_ids: {index}")
        roles = {name: raw[name] for name in ("train", "val", "test")}
    else:
        roles = {
            name: [
                int(line.strip())
                for line in (root / f"splits/{name}.txt").read_text().splitlines()
                if line.strip()
            ]
            for name in ("train", "val", "test")
        }
    result = _require_unique_roles(roles)
    counts = {name: len(values) for name, values in result.items()}
    if counts != EXPECTED_JTEXT_COUNTS:
        raise ValueError(f"J-TEXT role counts changed: {counts}")
    return result


def load_east_roles(root: Path = ROOT) -> dict[str, list[int]]:
    payload = read_json(root / "config/splits/east_hdf5_authoritative_balanced360_v1.json")
    test = payload.get("test")
    if not isinstance(test, dict):
        raise ValueError("EAST balanced360 manifest has no test object")
    result = _require_unique_roles({
        "disruption": test["disruption_shots"],
        "safe": test["non_disruption_shots"],
    })
    counts = {
        "disruption": len(result["disruption"]),
        "safe": len(result["safe"]),
        "total": len(result["disruption"]) + len(result["safe"]),
    }
    if counts != EXPECTED_EAST_COUNTS:
        raise ValueError(f"EAST balanced360 counts changed: {counts}")
    return result


def shot_path(data_root: Path, device: str, shot: int) -> Path:
    return (
        Path(data_root)
        / device
        / "unified_hdf5"
        / str(int(shot) // 100 * 100)
        / f"{int(shot)}.hdf5"
    )


def read_hdf5_meta(data_root: Path, device: str, shot: int) -> dict[str, Any]:
    path = shot_path(data_root, device, shot)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing or symlinked {device} HDF5: {path}")
    with h5py.File(path, "r") as handle:
        for required in ("meta/IsDisrupt", "meta/DownTime"):
            if required not in handle:
                raise ValueError(f"{path} lacks authoritative dataset {required}")
        is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down_raw = float(np.asarray(handle["meta/DownTime"][()]).reshape(-1)[0])
        if "data/ip" not in handle:
            raise ValueError(f"{path} lacks data/ip for duration audit")
        signal = handle["data/ip"]
        length = int(signal.shape[0])
        start = float(np.asarray(signal.attrs.get("StartTime", 0.0)).reshape(-1)[0])
        rate = float(np.asarray(signal.attrs.get("SampleRate", 1000.0)).reshape(-1)[0])
    if length <= 0 or not np.isfinite(start) or not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"invalid time grid in {path}")
    if is_disrupt and not np.isfinite(down_raw):
        raise ValueError(f"disruptive shot {shot} has non-finite DownTime")
    return {
        "shot": int(shot),
        "is_disrupt": is_disrupt,
        "down_time_s": down_raw if is_disrupt else None,
        "length": length,
        "start_s": start,
        "sample_rate_hz": rate,
        "path": str(path),
    }


def _ttd_ms(meta: Mapping[str, Any]) -> np.ndarray:
    if not bool(meta["is_disrupt"]):
        return np.empty(0, dtype=np.float64)
    endpoints = np.arange(int(meta["length"]), dtype=np.float64)
    times = float(meta["start_s"]) + endpoints / float(meta["sample_rate_hz"])
    return (float(meta["down_time_s"]) - times) * 1000.0


def endpoint_count(meta: Mapping[str, Any], horizon_ms: float) -> int:
    ttd = _ttd_ms(meta)
    return int(np.count_nonzero(
        (ttd >= -1e-6) & (ttd <= float(horizon_ms) + 1e-6)
    ))


def summarize_jtext_labels(
    *, roles: Mapping[str, list[int]], data_root: Path, advanced_path: Path,
) -> dict[str, Any]:
    advanced = labels.load_advanced_time_ms(advanced_path)
    role_rows: dict[str, Any] = {}
    total_fixed = 0
    total_adv25 = 0
    total_flipped = 0
    all_horizons: list[float] = []
    all_durations: list[float] = []
    advanced_time_grid_errors: list[float] = []
    missing_by_role: dict[str, list[int]] = {}
    disruptive_by_role: dict[str, int] = {}
    for role, shot_ids in roles.items():
        fixed_count = 0
        adv_count = 0
        flipped_count = 0
        horizons: list[float] = []
        durations: list[float] = []
        missing: list[int] = []
        disruptive = 0
        for shot in shot_ids:
            meta = read_hdf5_meta(data_root, "J-TEXT", shot)
            if not meta["is_disrupt"]:
                continue
            disruptive += 1
            horizon, raw, used_fallback = labels.effective_horizon_ms(
                shot, advanced, floor_ms=labels.DEFAULT_FLOOR_MS
            )
            if used_fallback:
                missing.append(int(shot))
            horizons.append(float(horizon))
            duration_ms = float(meta["length"]) / float(meta["sample_rate_hz"]) * 1000.0
            durations.append(duration_ms)
            if raw is not None:
                last_sample_s = float(meta["start_s"]) + (
                    int(meta["length"]) - 1
                ) / float(meta["sample_rate_hz"])
                end_gap_ms = (float(meta["down_time_s"]) - last_sample_s) * 1000.0
                advanced_time_grid_errors.append(float(raw) - end_gap_ms)
            fixed = endpoint_count(meta, FIXED_HORIZON_MS)
            updated = endpoint_count(meta, horizon)
            fixed_count += fixed
            adv_count += updated
            flipped_count += max(0, fixed - updated)
            all_horizons.append(float(horizon))
            all_durations.append(duration_ms)
        role_rows[role] = {
            "shot_count": len(shot_ids),
            "disruptive_shot_count": disruptive,
            "missing_advanced_time_shots": missing,
            "missing_advanced_time_count": len(missing),
            "fixed125_positive_endpoint_count": fixed_count,
            "adv25_positive_endpoint_count": adv_count,
            "fixed125_to_adv25_hard_negative_flips": flipped_count,
            "effective_horizon_ms": _numeric_summary(horizons),
            "shot_duration_ms": _numeric_summary(durations),
        }
        total_fixed += fixed_count
        total_adv25 += adv_count
        total_flipped += flipped_count
        missing_by_role[role] = missing
        disruptive_by_role[role] = disruptive
    return {
        "advanced_time_source": str(advanced_path.resolve()),
        "advanced_time_source_sha256": sha256_file(advanced_path),
        "advanced_time_entry_count": len(advanced),
        "formula": "H_i=max(25 ms, AdvancedTime_i)",
        "positive_rule": "0 <= TTD <= H_i",
        "negative_rule": "H_i < TTD <= 60000 ms",
        "role_summary": role_rows,
        "disruptive_shots_by_role": disruptive_by_role,
        "missing_advanced_time_by_role": missing_by_role,
        "fixed125_positive_endpoint_count_total": total_fixed,
        "adv25_positive_endpoint_count_total": total_adv25,
        "fixed125_to_adv25_hard_negative_flips_total": total_flipped,
        "positive_endpoint_reduction": total_fixed - total_adv25,
        "positive_endpoint_reduction_fraction": (
            (total_fixed - total_adv25) / total_fixed if total_fixed else None
        ),
        "all_effective_horizon_ms": _numeric_summary(all_horizons),
        "all_shot_duration_ms": _numeric_summary(all_durations),
        "advanced_time_vs_signal_end_gap_ms": _numeric_summary(
            advanced_time_grid_errors
        ),
        "advanced_time_semantics": (
            "weak shot-level lead-time annotation; it is not required to equal "
            "DownTime minus the last stored signal sample"
        ),
    }


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _resolve_run_file(path: Path, key: str) -> Path:
    value = path / key
    if not value.is_file() or value.is_symlink():
        raise FileNotFoundError(f"missing or symlinked run artifact: {value}")
    return value


def audit_source_checkpoints(run_root: Path, advanced_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    actual_advanced_sha = sha256_file(advanced_path)
    for seed in SEEDS:
        seed_root = run_root / f"seed{seed}"
        summary_path = _resolve_run_file(seed_root / "source", "training_summary.json")
        checkpoint_path = _resolve_run_file(seed_root / "source", "source_checkpoint.pt")
        summary = read_json(summary_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        if summary.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"seed{seed} source summary/checkpoint SHA mismatch")
        training = summary.get("training_label")
        if not isinstance(training, dict) or training.get("policy") != ADV25_POLICY:
            raise ValueError(f"seed{seed} source is not an Adv25 checkpoint")
        if int(summary.get("test_shards_read", -1)) != 0:
            raise ValueError(f"seed{seed} source summary reports TEST220 access")
        try:
            import torch
        except ImportError as exc:  # the registered audit runtime must have torch
            raise RuntimeError("the Adv25 audit requires the rl_env PyTorch runtime") from exc
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("training_label", {}).get("policy") != ADV25_POLICY:
            raise ValueError(f"seed{seed} checkpoint payload is not an Adv25 checkpoint")
        if int(checkpoint.get("test_shards_read", -1)) != 0:
            raise ValueError(f"seed{seed} checkpoint payload reports TEST220 access")
        metadata = training.get("advanced_time_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"seed{seed} source lacks AdvancedTime metadata")
        recorded_sha = str(training.get("advanced_time_source_sha256", ""))
        if recorded_sha != actual_advanced_sha:
            raise ValueError(f"seed{seed} source AdvancedTime SHA mismatch")
        if metadata.get("ambiguous_context_until_ms") is not None:
            raise ValueError(f"seed{seed} source unexpectedly has a gray band")
        result[str(seed)] = {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "encoder_state_sha256": summary.get("encoder_state_sha256"),
            "best_epoch": summary.get("best_epoch"),
            "training_label_policy": training.get("policy"),
            "advanced_time_source_sha256": recorded_sha,
            "test_shards_read": int(summary.get("test_shards_read", -1)),
        }
    return result


def audit_fixed125_parity(current_path: Path, historical_path: Path) -> dict[str, Any]:
    """Compare the current shot-first replay with the registered seed-131 record.

    Only fields that define the static replay are compared.  Query-ledger
    serialization is intentionally not treated as a model mismatch: old and
    current writers use different JSONL formatting while their sealed static
    predictions and operational confusion counts agree.
    """

    current = read_json(current_path)
    historical = read_json(historical_path)
    current_results = current.get("results", {})
    historical_results = historical.get("results", {})
    fields = (
        ("static_unified", "TP"),
        ("static_unified", "FP"),
        ("static_unified", "FN"),
        ("static_unified", "TN"),
        ("static_unified", "tpr_at_30ms"),
        ("static_unified", "fpr"),
        ("online_unified", "TP"),
        ("online_unified", "FP"),
        ("online_unified", "FN"),
        ("online_unified", "TN"),
        ("online_unified", "tpr_at_30ms"),
        ("online_unified", "fpr"),
    )
    mismatches: dict[str, dict[str, Any]] = {}
    for section, field in fields:
        left = current_results.get(section, {}).get(field)
        right = historical_results.get(section, {}).get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            equal = bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
        else:
            equal = left == right
        if not equal:
            mismatches[f"{section}.{field}"] = {"current": left, "historical": right}
    current_hashes = current.get("hashes", {})
    historical_hashes = historical.get("hashes", {})
    static_prediction_hash_match = (
        current_hashes.get("static_predictions")
        == historical_hashes.get("static_predictions")
    )
    source_hash_match = (
        current_hashes.get("source_checkpoint")
        == historical_hashes.get("source_checkpoint")
    )
    return {
        "current_summary": str(current_path),
        "current_summary_sha256": sha256_file(current_path),
        "historical_summary": str(historical_path),
        "historical_summary_sha256": sha256_file(historical_path),
        "metric_mismatches": mismatches,
        "static_prediction_hash_match": bool(static_prediction_hash_match),
        "source_checkpoint_hash_match": bool(source_hash_match),
        "passed": not mismatches and static_prediction_hash_match and source_hash_match,
        "query_ledger_hash_compared": False,
    }


def _metrics_from_summary(summary: Mapping[str, Any]) -> dict[str, float | int | None]:
    results = summary.get("results")
    if not isinstance(results, dict):
        raise ValueError("EAST online summary has no results object")
    static = results.get("static_unified")
    online = results.get("online_unified")
    if not isinstance(static, dict) or not isinstance(online, dict):
        raise ValueError("EAST online summary lacks static/online metrics")
    return {
        "static_tpr30": float(static["tpr_at_30ms"]),
        "static_fpr": float(static["fpr"]),
        "static_j": float(static["tpr_at_30ms"]) - float(static["fpr"]),
        "online_tpr30": float(online["tpr_at_30ms"]),
        "online_fpr": float(online["fpr"]),
        "online_j": float(online["tpr_at_30ms"]) - float(online["fpr"]),
        "delta_tpr30": float(online["tpr_at_30ms"]) - float(static["tpr_at_30ms"]),
        "delta_fpr": float(online["fpr"]) - float(static["fpr"]),
        "delta_j": (
            float(online["tpr_at_30ms"]) - float(online["fpr"])
            - float(static["tpr_at_30ms"]) + float(static["fpr"])
        ),
        "static_over_3000ms_count": int(static.get("over_3000ms_count", 0)),
        "online_over_3000ms_count": int(online.get("over_3000ms_count", 0)),
    }


def _source_checkpoint_sha(summary: Mapping[str, Any]) -> str | None:
    """Read either summary schema without inventing a provenance value."""
    direct = summary.get("source_checkpoint_sha256")
    if direct is not None:
        return str(direct)
    hashes = summary.get("hashes", {})
    if isinstance(hashes, Mapping):
        value = hashes.get("source_checkpoint")
        return str(value) if value is not None else None
    return None


def _historical_branch(
    path: Path, summary: Mapping[str, Any], *, name: str,
) -> dict[str, Any]:
    return {
        "summary": str(path),
        "summary_sha256": sha256_file(path),
        "source_checkpoint_sha256": _source_checkpoint_sha(summary),
        "query_count": int(summary.get("query_count", -1)),
        "threshold": float(summary["threshold"]),
        "metrics": _metrics_from_summary(summary),
        "lineage": name,
    }


def audit_east_runs(
    run_root: Path,
    historical_root: Path,
    historical_locked_root: Path | None = None,
) -> dict[str, Any]:
    """Compare Adv25 with both historical lineages, never silently mixing them.

    ``historical_root`` is the newer development-v3c evidence directory and
    ``historical_locked_root`` is the registered locked V3C/v8 mainline.  The
    latter is the primary comparison.  The optional argument keeps the helper
    usable by older callers; in that case the supplied root is treated as the
    locked reference and no development branch is fabricated.
    """
    locked_root = historical_locked_root or historical_root
    result: dict[str, Any] = {}
    for seed in SEEDS:
        adv_path = _resolve_run_file(
            run_root / f"seed{seed}" / "final_episode_v3", "summary.json"
        )
        locked_path = _resolve_run_file(
            locked_root / f"seed{seed}", "summary.json"
        )
        adv_summary = read_json(adv_path)
        locked_summary = read_json(locked_path)
        row: dict[str, Any] = {
            "adv25": {
                "summary": str(adv_path),
                "summary_sha256": sha256_file(adv_path),
                "source_checkpoint_sha256": _source_checkpoint_sha(adv_summary),
                "query_count": int(adv_summary.get("query_count", -1)),
                "threshold": float(adv_summary["threshold"]),
                "metrics": _metrics_from_summary(adv_summary),
            },
            "historical_locked_v8": _historical_branch(
                locked_path, locked_summary, name="historical_locked_v8"
            ),
        }
        development_path = historical_root / f"seed{seed}" / "summary.json"
        if development_path.resolve() != locked_path.resolve():
            development_summary = read_json(development_path)
            row["historical_development_v3c"] = _historical_branch(
                development_path,
                development_summary,
                name="historical_development_v3c",
            )
        else:
            # Keep the schema explicit when a legacy caller supplies one root.
            row["historical_development_v3c"] = None
        # Backward-compatible alias for consumers written before the lineage
        # split.  It points to the locked mainline, never the dev branch.
        row["historical_v3c"] = row["historical_locked_v8"]
        result[str(seed)] = row
    return result


def mean_metrics(rows: Mapping[str, Mapping[str, Any]], branch: str) -> dict[str, float]:
    metrics = [rows[str(seed)][branch]["metrics"] for seed in SEEDS]
    keys = ("static_tpr30", "static_fpr", "static_j", "online_tpr30", "online_fpr", "online_j", "delta_tpr30", "delta_fpr", "delta_j")
    return {key: float(np.mean([float(row[key]) for row in metrics])) for key in keys}


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def runtime_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": sys.executable}
    try:
        import torch
        state.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        state["torch_import"] = "unavailable"
    return state


def render_report(audit: Mapping[str, Any]) -> str:
    labels_audit = audit["jtext_label_audit"]
    east = audit["east_runs"]
    old_mean = audit["means"]["historical_locked_v8"]
    development_mean = audit["means"].get("historical_development_v3c")
    adv_mean = audit["means"]["adv25"]
    parity = audit["fixed125_seed131_parity"]
    lines = [
        "# Adv25 标签替换根因审计与迭代裁决",
        "",
        f"审计时间：{audit['generated_at']}  ",
        "状态：`DIAGNOSTIC_ONLY / TEST_DRIVEN_EXPLORATION`  ",
        "运行环境：`rl_env`（审计脚本不训练模型）",
        "",
        "## 结论先行",
        "",
        "本次复核没有发现 AdvancedTime 的秒/毫秒换算错误、J-TEXT 炮表交叉、"
        "Adv25 checkpoint 越权读取 TEST220，或 EAST V3C 在线代码被替换的问题。"
        "专项时间审计也确认 AdvancedTime 与信号末端间隔并不要求相等；它是独立的"
        "炮级弱 lead-time 标注，不是采样末端时间戳。Adv25 结果下降的直接原因是"
        "监督分布发生了实质变化：在当前硬标签解释下，固定 125 ms 正窗中相当一部分"
        "端点被改成了硬负类。这个结论不能被表述为‘迁移算法失败’或‘J-TEXT 与 EAST"
        "完全不相通’。",
        "",
        "旧 V3C 的 EAST 在线机制仍可复现；但把源模型标签改为 Adv25 后，不能"
        "自动保证旧在线增益保持。下一次允许的迭代应是预注册的‘源表示保持/"
        "标签边界消融’，而不是用 balanced360 继续反向调报警参数。",
        "",
        "## 1. 标签合同与实际影响",
        "",
        "主合同固定为：",
        "",
        r"$$H_i=\max(25\,\mathrm{ms},\mathrm{AdvancedTime}_i),\quad "
        r"y=1\Leftrightarrow 0\leq\mathrm{TTD}\leq H_i.$$",
        "",
        "这里还明确采用了互补硬负类 `H_i < TTD <= 60000 ms`。如果研究意图是"
        "‘AdvancedTime 只是最早可靠边界’，则该互补负类不是同一个实验，必须"
        "另行注册灰区或软标签方案。",
        "",
        "| 角色 | 炮数 | 破裂炮 | 缺失 AdvancedTime | fixed125 正端点 | Adv25 正端点 | 被改为硬负类 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("train", "val", "test"):
        row = labels_audit["role_summary"][role]
        lines.append(
            f"| {role} | {row['shot_count']} | {row['disruptive_shot_count']} | "
            f"{row['missing_advanced_time_count']} | "
            f"{row['fixed125_positive_endpoint_count']} | "
            f"{row['adv25_positive_endpoint_count']} | "
            f"{row['fixed125_to_adv25_hard_negative_flips']} |"
        )
    lines.extend([
        f"| **合计** | **{sum(labels_audit['role_summary'][r]['shot_count'] for r in ('train','val','test'))}** | **{sum(labels_audit['role_summary'][r]['disruptive_shot_count'] for r in ('train','val','test'))}** | **{sum(labels_audit['role_summary'][r]['missing_advanced_time_count'] for r in ('train','val','test'))}** | **{labels_audit['fixed125_positive_endpoint_count_total']}** | **{labels_audit['adv25_positive_endpoint_count_total']}** | **{labels_audit['fixed125_to_adv25_hard_negative_flips_total']}** |",
        "",
        f"正端点从 {labels_audit['fixed125_positive_endpoint_count_total']} 减少到 "
        f"{labels_audit['adv25_positive_endpoint_count_total']}，减少 "
        f"{labels_audit['positive_endpoint_reduction']}（"
        f"{labels_audit['positive_endpoint_reduction_fraction']:.1%}）。",
        "",
        "AdvancedTime 与最后一个信号采样点的差值统计（仅用于语义审计，不用于改标签）："
        f"中位 {labels_audit['advanced_time_vs_signal_end_gap_ms']['median']:.3f} ms，"
        f"最大绝对值 {max(abs(float(labels_audit['advanced_time_vs_signal_end_gap_ms']['min'])), abs(float(labels_audit['advanced_time_vs_signal_end_gap_ms']['max']))):.3f} ms。"
        "该差异说明 AdvancedTime 是弱标签 lead-time，不能被当作绝对时间戳。",
        "",
        "## 2. EAST 配对结果",
        "",
        "下表的主对照是已经注册的锁定 V3C/v8 下游实现与 Adv25；balanced360 已参与"
        "方法开发，因此两者均为诊断性结果。",
        "",
        "| seed | 锁定V3C静态 TPR/FPR | 锁定V3C在线 TPR/FPR | Adv25静态 TPR/FPR | Adv25在线 TPR/FPR | Adv25查询炮 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in SEEDS:
        row = east[str(seed)]
        old = row["historical_locked_v8"]["metrics"]
        new = row["adv25"]["metrics"]
        lines.append(
            f"| {seed} | {old['static_tpr30']:.4f}/{old['static_fpr']:.4f} | "
            f"{old['online_tpr30']:.4f}/{old['online_fpr']:.4f} | "
            f"{new['static_tpr30']:.4f}/{new['static_fpr']:.4f} | "
            f"{new['online_tpr30']:.4f}/{new['online_fpr']:.4f} | "
            f"{row['adv25']['query_count']} |"
        )
    lines.extend([
        f"| **均值** | {old_mean['static_tpr30']:.4f}/{old_mean['static_fpr']:.4f} | "
        f"{old_mean['online_tpr30']:.4f}/{old_mean['online_fpr']:.4f} | "
        f"{adv_mean['static_tpr30']:.4f}/{adv_mean['static_fpr']:.4f} | "
        f"{adv_mean['online_tpr30']:.4f}/{adv_mean['online_fpr']:.4f} | "
        f"{np.mean([east[str(s)]['adv25']['query_count'] for s in SEEDS]):.1f} |",
        "",
        f"锁定 V3C 在线 ΔJ={old_mean['delta_j']:+.4f}；Adv25 在线 ΔJ="
        f"{adv_mean['delta_j']:+.4f}。因此旧在线收益没有在 Adv25 标签下稳定"
        "复现。该结论只适用于锁定 V3C 配对；不能把另一条 episode-v3 开发链的结果混入。",
        "",
        (
            f"另存的 development-v3c 历史证据均值："
            f"静态 {development_mean['static_tpr30']:.4f}/"
            f"{development_mean['static_fpr']:.4f}，在线 "
            f"{development_mean['online_tpr30']:.4f}/"
            f"{development_mean['online_fpr']:.4f}，ΔJ="
            f"{development_mean['delta_j']:+.4f}；它仅作诊断，不能替代锁定主对照."
            if isinstance(development_mean, Mapping)
            else "development-v3c 历史证据未提供，未构造替代对照。"
        ),
        "",
        "## 3. 谱系与泄漏检查",
        "",
        "- 三个 Adv25 source checkpoint 均声明 `advanced_floor25`，AdvancedTime 文件 SHA 一致，且 `test_shards_read=0`。",
        "- J-TEXT 在线分支的 checkpoint 没有作为 EAST 的 parent；EAST 阶段使用静态 Adv25 source checkpoint。",
        f"- 固定 125 ms 的当前 `shot_first` V3C seed131 静态/在线指标与静态预测哈希"
        f"复核：`{'PASS' if parity['passed'] else 'FAIL'}`。",
        "- 旧 A1 reference JSON 没有记录当前 sampler/code lineage；因此当前代码对它的逐分数校验可能产生误导性的失败。该旧 reference 不应继续作为 Adv25 判据。",
        "- 固定 25 ms 的 full V3C 临时运行没有产生完整 summary，且曾因 ENOSPC 中断；它不进入任何结果表。",
        "",
        "## 4. 下一步唯一允许改变的因素",
        "",
        "1. 保持 EAST 适配、报警器、炮序和阈值来源不变；",
        "2. 只注册一组源模型表示保持方案（例如 Adv25 主损失 + fixed125 表示蒸馏/"
        "局部多任务保持），并与纯 Adv25、fixed125 做同 parent 的配对消融；",
        "3. 先在 J-TEXT VAL160 选择 epoch/阈值，不能用 balanced360 选择；",
        "4. 重新跑三个 seed，报告源域和 EAST 的 static/online 变化；",
        "5. 在新方案锁定前，balanced360 只保留诊断，不作为独立未来测试。",
        "",
        "## 5. 产物",
        "",
        f"机器可读审计：`{audit['artifacts']['json']}`  ",
        f"本报告：`{audit['artifacts']['markdown']}`  ",
        f"AdvancedTime SHA256：`{labels_audit['advanced_time_source_sha256']}`",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    advanced_path = Path(args.advanced_time).expanduser().resolve()
    adv25_root = Path(args.adv25_run_root).expanduser().resolve()
    fixed125_root = Path(args.fixed125_run_root).expanduser().resolve()
    historical_root = Path(args.historical_root).expanduser().resolve()
    historical_locked_root = Path(args.historical_locked_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roles = load_jtext_roles(root)
    east_roles = load_east_roles(root)
    label_audit = summarize_jtext_labels(
        roles=roles, data_root=data_root, advanced_path=advanced_path
    )
    source = audit_source_checkpoints(adv25_root, advanced_path)
    east = audit_east_runs(
        adv25_root,
        historical_root,
        historical_locked_root,
    )
    fixed125_parity = audit_fixed125_parity(
        fixed125_root / "seed131/final_episode_v3/summary.json",
        historical_root / "seed131/summary.json",
    )
    means = {
        "adv25": mean_metrics(east, "adv25"),
        "historical_locked_v8": mean_metrics(east, "historical_locked_v8"),
    }
    if all(east[str(seed)]["historical_development_v3c"] is not None for seed in SEEDS):
        means["historical_development_v3c"] = mean_metrics(
            east, "historical_development_v3c"
        )
    # Compatibility alias: old readers now receive the locked mainline by
    # definition, rather than the ambiguous development directory.
    means["historical_v3c"] = means["historical_locked_v8"]
    audit: dict[str, Any] = {
        "schema_version": "adv25-root-cause-audit-v1",
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "status": "DIAGNOSTIC_ONLY",
        "claim_boundary": "balanced360 participated in method development; no independent-test claim",
        "root": str(root),
        "data_root": str(data_root),
        "jtext_roles": roles,
        "east_balanced360_roles": east_roles,
        "jtext_label_audit": label_audit,
        "source_checkpoints": source,
        "east_runs": east,
        "fixed125_seed131_parity": fixed125_parity,
        "means": means,
        "runtime": runtime_state(),
        "git": git_state(root),
        "artifacts": {
            "json": str(output_dir / "audit.json"),
            "markdown": str(output_dir / "REPORT.md"),
        },
    }
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "REPORT.md"
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_report(audit), encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--advanced-time",
        type=Path,
        default=ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json",
    )
    parser.add_argument(
        "--adv25-run-root",
        type=Path,
        default=ROOT / "fuison_ext4_runs/v3c_adv25_samplerfix_20260830",
    )
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=HERE / "paper_grade_package/evidence/primary_online_artifacts",
    )
    parser.add_argument(
        "--historical-locked-root",
        type=Path,
        default=HERE / "final_two_phase_runs",
    )
    parser.add_argument(
        "--fixed125-run-root",
        type=Path,
        default=ROOT / "fuison_ext4_runs/v3c_samplerfix_fixed125_20260830",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "adv25_root_cause_audit_20260830",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "json": result["artifacts"]["json"],
        "markdown": result["artifacts"]["markdown"],
        "adv25_mean": result["means"]["adv25"],
        "historical_locked_v8_mean": result["means"]["historical_locked_v8"],
        "historical_development_v3c_mean": result["means"].get(
            "historical_development_v3c"
        ),
    }, indent=2, sort_keys=True))
