#!/usr/bin/env python3
"""Build traceable tables and statistics for the three-stage paper package.

All primary numbers are recomputed from immutable per-shot predictions plus
HDF5 metadata.  The balanced360 stream is explicitly treated as a development
stream; statistics quantify sampling uncertainty but do not manufacture an
independent confirmation cohort.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.stats import binomtest, mannwhitneyu, rankdata, spearmanr


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
EVIDENCE = PACKAGE / "evidence"
SEEDS = (131, 133, 137)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_artifact(path: Path) -> Path:
    """Resolve a ledger stored either verbatim or as a gzip archive."""
    if path.is_file():
        return path
    compressed = Path(f"{path}.gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(path)


def read_jsonl(path: Path) -> list[dict]:
    resolved = resolve_artifact(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [json.loads(line) for line in resolved.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: Path) -> str:
    """Hash the original ledger bytes, transparently decompressing gzip."""
    resolved = resolve_artifact(path)
    if resolved.suffix != ".gz":
        return sha256_file(resolved)
    digest = hashlib.sha256()
    with gzip.open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_run(seed: int) -> Path:
    bundled = EVIDENCE / f"primary_online_artifacts/seed{int(seed)}"
    if bundled.is_dir():
        return bundled
    return HERE / f"online_episode_value_v3_runs/final_seed{int(seed)}_episode_v3"


def source_run(seed: int) -> Path:
    bundled = EVIDENCE / f"source_lineage/seed{int(seed)}"
    if bundled.is_dir():
        return bundled
    return HERE / f"source_transformer_runs/seed{int(seed)}"


def bundled_or_fallback(bundled: Path, fallback: Path) -> Path:
    return bundled if bundled.is_file() else fallback


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def normalize_local_paths(value: Any) -> Any:
    """Make aggregate evidence portable without altering scientific values."""
    if isinstance(value, dict):
        return {key: normalize_local_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [normalize_local_paths(child) for child in value]
    if not isinstance(value, str) or not value.startswith(("/home/", "/media/")):
        return value
    candidates = []
    for marker in (
        "experiments/", "config/", "splits/", "artifacts/", "results/", "data/",
    ):
        position = value.find(f"/{marker}")
        if position >= 0:
            candidates.append((position, value[position + 1 :]))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return Path(value).name


def load_east_truth(shots: Iterable[int]) -> dict[int, dict[str, Any]]:
    truth: dict[int, dict[str, Any]] = {}
    for shot in sorted({int(value) for value in shots}):
        path = DATA / "EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        with h5py.File(path, "r") as handle:
            is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
            down = float(np.asarray(handle["meta/DownTime"][()]).reshape(-1)[0])
            signal = handle["data/ip"]
            if "StartTime" in handle["meta"]:
                start = float(np.asarray(handle["meta/StartTime"][()]).reshape(-1)[0])
            else:
                start = float(np.asarray(signal.attrs.get("StartTime", 0.0)).reshape(-1)[0])
            if "length" in handle["meta"] and "time_interval" in handle["meta"]:
                length = int(np.asarray(handle["meta/length"][()]).reshape(-1)[0])
                interval = float(np.asarray(handle["meta/time_interval"][()]).reshape(-1)[0])
            else:
                length = int(signal.shape[0])
                rate = float(np.asarray(signal.attrs.get("SampleRate", 1000.0)).reshape(-1)[0])
                interval = 1.0 / rate
        truth[shot] = {
            "is_disrupt": is_disrupt,
            "down_time_s": down if is_disrupt else None,
            "start_time_s": start,
            "duration_s": length * interval,
            "hdf5": str(path),
        }
    return truth


def load_jtext_truth(shots: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Read J-TEXT outcomes from the same HDF5 authority used for training."""
    truth: dict[int, dict[str, Any]] = {}
    for shot in sorted({int(value) for value in shots}):
        path = DATA / "J-TEXT/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        with h5py.File(path, "r") as handle:
            is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
            down = float(np.asarray(handle["meta/DownTime"][()]).reshape(-1)[0])
        truth[shot] = {
            "is_disrupt": is_disrupt,
            "down_time_s": down if is_disrupt else None,
            "hdf5": str(path),
        }
    return truth


def read_shot_ids(path: Path) -> list[int]:
    values = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate shot IDs in {path}")
    return values


def jtext_split_audit() -> dict[str, Any]:
    """Describe the executed source split without implying temporal isolation."""
    roles = {
        name: read_shot_ids(ROOT / "splits" / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    if any(set(roles[left]) & set(roles[right])
           for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("J-TEXT split roles overlap")
    truth = load_jtext_truth(value for values in roles.values() for value in values)
    role_rows: dict[str, Any] = {}
    for name, shots in roles.items():
        disruptive = int(sum(truth[shot]["is_disrupt"] for shot in shots))
        role_rows[name] = {
            "shots": len(shots),
            "disruptions": disruptive,
            "safe": len(shots) - disruptive,
            "shot_min": min(shots),
            "shot_max": max(shots),
            "sorted_for_serialization": shots == sorted(shots),
            "roster_sha256": sha256_file(ROOT / "splits" / f"{name}.txt"),
        }
    strict_role_chronology = (
        max(roles["train"]) < min(roles["val"])
        and max(roles["val"]) < min(roles["test"])
    )
    return {
        "roles": role_rows,
        "role_overlap": 0,
        "selection_procedure": "fixed paper131-count-aligned comparator cohort",
        "strict_chronological_role_order": strict_role_chronology,
        "mimics_operational_prevalence": False,
        "claim_status": "WITHIN_RUN_HELD_OUT_BALANCED_COMPARATOR",
    }


def alarm_lead_ms(record: dict, truth: dict[int, dict[str, Any]]) -> float | None:
    shot = int(record["shot"])
    if not truth[shot]["is_disrupt"] or not bool(record["alarmed"]):
        return None
    alarm_time = record.get("alarm_time_s")
    if alarm_time is None:
        return None
    return (float(truth[shot]["down_time_s"]) - float(alarm_time)) * 1000.0


def alarm_metrics(
    records: list[dict],
    truth: dict[int, dict[str, Any]],
    *,
    minimum_lead_ms: float = 30.0,
) -> dict[str, Any]:
    disruptive = [row for row in records if truth[int(row["shot"])]["is_disrupt"]]
    safe = [row for row in records if not truth[int(row["shot"])]["is_disrupt"]]
    raw_hit = [bool(row["alarmed"]) for row in disruptive]
    false_alarm = [bool(row["alarmed"]) for row in safe]
    leads = [
        alarm_lead_ms(row, truth) for row in disruptive if bool(row["alarmed"])
    ]
    leads = [float(value) for value in leads if value is not None]
    timely = [
        bool(row["alarmed"])
        and (alarm_lead_ms(row, truth) or float("-inf"))
        >= float(minimum_lead_ms) - 1e-9
        for row in disruptive
    ]
    timely_leads = [
        value for value in leads
        if value >= float(minimum_lead_ms) - 1e-9
    ]
    operational_tp = int(sum(timely))
    result = {
        "n_shots": len(records),
        "n_disruptions": len(disruptive),
        "n_non_disruptions": len(safe),
        "TP": int(sum(raw_hit)),
        "FN": int(len(raw_hit) - sum(raw_hit)),
        "FP": int(sum(false_alarm)),
        "TN": int(len(false_alarm) - sum(false_alarm)),
        "tpr": float(np.mean(raw_hit)) if raw_hit else None,
        "tpr_at_30ms": float(np.mean(timely)) if timely else None,
        "fpr": float(np.mean(false_alarm)) if false_alarm else None,
        "median_lead_ms": float(np.median(leads)) if leads else None,
        # These are the formal deployment-facing quantities.  The legacy TP/FN,
        # tpr, and median_lead_ms fields above remain diagnostic-only so frozen
        # historical summaries can still be reproduced byte-for-byte.
        "operational_TP": operational_tp,
        "operational_FN": int(len(timely) - operational_tp),
        "operational_tpr": float(np.mean(timely)) if timely else None,
        "operational_median_lead_ms": (
            float(np.median(timely_leads)) if timely_leads else None
        ),
        "over_3000ms_count": int(sum(value > 3000.0 for value in leads)),
        "over_3000ms_shots": [
            int(row["shot"])
            for row in disruptive
            if bool(row["alarmed"])
            and (alarm_lead_ms(row, truth) or float("-inf")) > 3000.0
        ],
        "minimum_lead_ms": float(minimum_lead_ms),
    }
    return result


def exact_paired_change(before_correct: Iterable[bool], after_correct: Iterable[bool]) -> dict:
    before = np.asarray(list(before_correct), dtype=bool)
    after = np.asarray(list(after_correct), dtype=bool)
    if before.shape != after.shape:
        raise ValueError("paired correctness arrays differ")
    rescued = int(np.sum((~before) & after))
    harmed = int(np.sum(before & (~after)))
    discordant = rescued + harmed
    p = float(binomtest(rescued, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "rescued": rescued,
        "harmed": harmed,
        "discordant": discordant,
        "two_sided_exact_p": p,
    }


def aggregate_metric_rows(rows: dict[str, dict], names: Iterable[str]) -> dict:
    result = {}
    for name in names:
        values = np.asarray([rows[str(seed)][name] for seed in SEEDS], dtype=np.float64)
        result[name] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def _phase_arrays(
    records_by_seed: dict[str, list[dict]],
    truth: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    shots = [int(row["shot"]) for row in records_by_seed[str(SEEDS[0])]]
    if any([int(row["shot"]) for row in records_by_seed[str(seed)]] != shots for seed in SEEDS):
        raise RuntimeError("seed prediction rosters differ")
    labels = np.asarray([truth[shot]["is_disrupt"] for shot in shots], dtype=bool)
    alarms = np.asarray([
        [bool(row["alarmed"]) for row in records_by_seed[str(seed)]] for seed in SEEDS
    ], dtype=bool)
    timely = np.zeros_like(alarms)
    leads = np.full(alarms.shape, np.nan, dtype=np.float64)
    for seed_index, seed in enumerate(SEEDS):
        for index, row in enumerate(records_by_seed[str(seed)]):
            lead = alarm_lead_ms(row, truth)
            if lead is not None:
                leads[seed_index, index] = lead
                # Match alarm_metrics exactly: a nominal 30 ms warning remains
                # timely despite sub-picosecond floating-point subtraction error.
                timely[seed_index, index] = lead >= 30.0 - 1e-9
    operational_leads = np.where(timely, leads, np.nan)
    return {"shots": np.asarray(shots), "labels": labels, "alarms": alarms,
            "timely": timely, "leads": leads,
            "operational_leads": operational_leads}


def stratified_bootstrap(
    static: dict[str, np.ndarray],
    online: dict[str, np.ndarray],
    *,
    iterations: int = 5000,
    seed: int = 20260820,
) -> dict[str, Any]:
    labels = static["labels"]
    if not np.array_equal(labels, online["labels"]):
        raise RuntimeError("static and online labels differ")
    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in (
        "static_tpr", "online_tpr", "delta_tpr",
        "static_tpr30", "online_tpr30", "delta_tpr30",
        "static_fpr", "online_fpr", "delta_fpr",
        "static_median_lead", "online_median_lead", "delta_median_lead",
        "static_operational_median_lead", "online_operational_median_lead",
        "delta_operational_median_lead",
    )}
    for _ in range(int(iterations)):
        sampled_pos = pos[rng.integers(0, len(pos), len(pos))]
        sampled_neg = neg[rng.integers(0, len(neg), len(neg))]
        st_tpr = float(static["alarms"][:, sampled_pos].mean(axis=1).mean())
        on_tpr = float(online["alarms"][:, sampled_pos].mean(axis=1).mean())
        st_tpr30 = float(static["timely"][:, sampled_pos].mean(axis=1).mean())
        on_tpr30 = float(online["timely"][:, sampled_pos].mean(axis=1).mean())
        st_fpr = float(static["alarms"][:, sampled_neg].mean(axis=1).mean())
        on_fpr = float(online["alarms"][:, sampled_neg].mean(axis=1).mean())
        st_lead = float(np.nanmean([
            np.nanmedian(static["leads"][row, sampled_pos]) for row in range(len(SEEDS))
        ]))
        on_lead = float(np.nanmean([
            np.nanmedian(online["leads"][row, sampled_pos]) for row in range(len(SEEDS))
        ]))
        st_operational_lead = float(np.nanmean([
            np.nanmedian(static["operational_leads"][row, sampled_pos])
            for row in range(len(SEEDS))
        ]))
        on_operational_lead = float(np.nanmean([
            np.nanmedian(online["operational_leads"][row, sampled_pos])
            for row in range(len(SEEDS))
        ]))
        for name, value in (
            ("static_tpr", st_tpr), ("online_tpr", on_tpr),
            ("delta_tpr", on_tpr - st_tpr),
            ("static_tpr30", st_tpr30), ("online_tpr30", on_tpr30),
            ("delta_tpr30", on_tpr30 - st_tpr30),
            ("static_fpr", st_fpr), ("online_fpr", on_fpr),
            ("delta_fpr", on_fpr - st_fpr),
            ("static_median_lead", st_lead), ("online_median_lead", on_lead),
            ("delta_median_lead", on_lead - st_lead),
            ("static_operational_median_lead", st_operational_lead),
            ("online_operational_median_lead", on_operational_lead),
            ("delta_operational_median_lead", on_operational_lead - st_operational_lead),
        ):
            samples[name].append(value)
    result: dict[str, Any] = {
        "iterations": int(iterations),
        "resampling": "same stratified shot resample applied to all seeds and both phases",
    }
    result.update({
        name: {
            "median": float(np.median(values)),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        }
        for name, values in samples.items()
    })
    return result


def _circular_moving_block_indices(
    rng: np.random.Generator,
    *,
    n_observations: int,
    block_size: int,
) -> np.ndarray:
    """Sample circular contiguous blocks while retaining the original local order."""
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if block_size <= 0 or block_size > n_observations:
        raise ValueError("block_size must be in [1, n_observations]")
    block_count = int(math.ceil(n_observations / block_size))
    starts = rng.integers(0, n_observations, size=block_count)
    offsets = np.arange(block_size, dtype=np.int64)
    blocks = (starts[:, None] + offsets[None, :]) % n_observations
    return blocks.reshape(-1)[:n_observations]


def _mean_seed_median(values: np.ndarray, indices: np.ndarray) -> float:
    medians: list[float] = []
    for row in range(values.shape[0]):
        finite_values = values[row, indices]
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            medians.append(float(np.median(finite_values)))
    return float(np.mean(medians)) if medians else float("nan")


def moving_block_bootstrap(
    static: dict[str, np.ndarray],
    online: dict[str, np.ndarray],
    *,
    block_size: int,
    iterations: int = 5000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Paired circular moving-block resampling of the observed ordered stream.

    This preserves short-range order in the sealed ledger.  It does not rerun the
    adaptive learner on a resampled stream and therefore remains descriptive,
    especially after development-set model selection.
    """
    labels = static["labels"]
    if not np.array_equal(labels, online["labels"]):
        raise RuntimeError("static and online labels differ")
    if not np.array_equal(static["shots"], online["shots"]):
        raise RuntimeError("static and online shot order differs")
    rng = np.random.default_rng(seed + int(block_size))
    names = (
        "static_tpr30", "online_tpr30", "delta_tpr30",
        "static_fpr", "online_fpr", "delta_fpr",
        "static_operational_median_lead", "online_operational_median_lead",
        "delta_operational_median_lead",
    )
    samples = {name: [] for name in names}
    for _ in range(int(iterations)):
        indices = _circular_moving_block_indices(
            rng, n_observations=len(labels), block_size=int(block_size),
        )
        sampled_labels = labels[indices]
        pos = indices[sampled_labels]
        neg = indices[~sampled_labels]
        if not len(pos) or not len(neg):
            continue
        st_tpr = float(static["timely"][:, pos].mean(axis=1).mean())
        on_tpr = float(online["timely"][:, pos].mean(axis=1).mean())
        st_fpr = float(static["alarms"][:, neg].mean(axis=1).mean())
        on_fpr = float(online["alarms"][:, neg].mean(axis=1).mean())
        st_lead = _mean_seed_median(static["operational_leads"], pos)
        on_lead = _mean_seed_median(online["operational_leads"], pos)
        for name, value in (
            ("static_tpr30", st_tpr), ("online_tpr30", on_tpr),
            ("delta_tpr30", on_tpr - st_tpr),
            ("static_fpr", st_fpr), ("online_fpr", on_fpr),
            ("delta_fpr", on_fpr - st_fpr),
            ("static_operational_median_lead", st_lead),
            ("online_operational_median_lead", on_lead),
            ("delta_operational_median_lead", on_lead - st_lead),
        ):
            samples[name].append(value)
    if any(not values for values in samples.values()):
        raise RuntimeError("moving-block bootstrap produced no valid replicates")
    result: dict[str, Any] = {
        "iterations": min(len(values) for values in samples.values()),
        "block_size": int(block_size),
        "resampling": (
            "paired circular moving blocks on the observed chronological ledger; "
            "same indices for all seeds and both phases"
        ),
        "inference_boundary": (
            "descriptive sensitivity only; learner was not rerun and the stream was used "
            "for model selection"
        ),
    }
    result.update({
        name: {
            "median": float(np.median(values)),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        }
        for name, values in samples.items()
    })
    return result


def _nested_events(value: Any, event_name: str) -> list[dict[str, Any]]:
    """Collect named receipts from the append-only nested update ledger."""
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("event") == event_name:
            found.append(value)
        for child in value.values():
            found.extend(_nested_events(child, event_name))
    elif isinstance(value, list):
        for child in value:
            found.extend(_nested_events(child, event_name))
    return found


def alarm_state_machine_audit(run: Path, predictions: list[dict]) -> dict[str, Any]:
    """Reconstruct actual online alarm-parameter evolution from sealed ledgers."""
    events: list[dict[str, Any]] = []
    for row in read_jsonl(run / "updates.jsonl"):
        events.extend(_nested_events(row, "projected_alarm_update"))
    parameter_rows = [row["parameters"] for row in predictions if row.get("parameters")]
    if not parameter_rows:
        raise RuntimeError("online predictions do not contain alarm parameters")
    versions = sorted({
        int(row["parameter_version"])
        for row in predictions if row.get("parameter_version") is not None
    })
    groups = {
        name: int(sum(event.get("discrepancy_group") == name for event in events))
        for name in ("TP", "TN", "FP", "FN", "TARDY")
    }
    return {
        "prediction_count": len(predictions),
        "update_event_count": len(events),
        "accepted_update_count": int(sum(bool(event.get("accepted")) for event in events)),
        "changed_update_count": int(sum(bool(event.get("updated")) for event in events)),
        "parameter_version_count": len(versions),
        "first_parameter_version": min(versions),
        "last_parameter_version": max(versions),
        "initial_parameters": parameter_rows[0],
        "final_parameters": parameter_rows[-1],
        "discrepancy_groups": groups,
    }


def source_operational_row(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    truth: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Recompute source metrics from alarm endpoints and HDF5 truth."""
    if len(records) != len({int(item["shot"]) for item in records}):
        raise RuntimeError("duplicate source prediction shots")
    labels: list[bool] = []
    scores: list[float] = []
    disruptive: list[dict[str, Any]] = []
    safe: list[dict[str, Any]] = []
    recomputed_lead_by_shot: dict[int, float] = {}
    alarm_count = 0
    for item in records:
        shot = int(item["shot"])
        authoritative = bool(truth[shot]["is_disrupt"])
        if bool(item["is_disrupt"]) != authoritative:
            raise RuntimeError(f"source prediction label disagrees with HDF5 for shot {shot}")
        labels.append(authoritative)
        scores.append(float(item["score"]))
        (disruptive if authoritative else safe).append(item)
        if bool(item["alarm"]):
            alarm_count += 1
            if item.get("alarm_time_s") is None:
                raise RuntimeError(
                    f"source alarm endpoint is missing for shot {shot}; lead cannot be "
                    "independently recomputed"
                )
            if authoritative:
                hdf5_lead = (
                    float(truth[shot]["down_time_s"]) - float(item["alarm_time_s"])
                ) * 1000.0
                archived_lead = item.get("lead_ms")
                if archived_lead is None or not math.isclose(
                    hdf5_lead, float(archived_lead), rel_tol=0.0, abs_tol=1e-6,
                ):
                    raise RuntimeError(f"source lead reproduction failed for shot {shot}")
                recomputed_lead_by_shot[shot] = hdf5_lead
    minimum_lead_ms = float(row["minimum_lead_ms"])
    timely_leads = [
        recomputed_lead_by_shot[int(item["shot"])]
        for item in disruptive
        if bool(item["alarm"])
        and recomputed_lead_by_shot[int(item["shot"])] > minimum_lead_ms
    ]
    operational_tp = len(timely_leads)
    false_positive = int(sum(bool(item["alarm"]) for item in safe))
    labels_array = np.asarray(labels, dtype=bool)
    ranks = rankdata(np.asarray(scores, dtype=np.float64), method="average")
    n_disruptions = int(labels_array.sum())
    n_safe = int((~labels_array).sum())
    auc = float(
        (ranks[labels_array].sum() - n_disruptions * (n_disruptions + 1) / 2)
        / (n_disruptions * n_safe)
    )
    return {
        "n_shots": len(records),
        "n_disruptions": n_disruptions,
        "n_non_disruptions": n_safe,
        "operational_TP": operational_tp,
        "operational_FN": n_disruptions - operational_tp,
        "FP": false_positive,
        "TN": n_safe - false_positive,
        "operational_tpr": operational_tp / n_disruptions,
        "fpr": false_positive / n_safe,
        "auc": auc,
        "operational_median_lead_ms": float(np.median(timely_leads)),
        "minimum_lead_ms": minimum_lead_ms,
        "threshold": float(row["threshold"]),
        "threshold_origin": row["threshold_origin"],
        "over_3000ms_count": int(sum(value > 3000.0 for value in timely_leads)),
        "metric_source": (
            "per-shot alarm_time_s ledger; lead independently recomputed from HDF5 "
            "/meta/DownTime"
        ),
        "lead_recomputation": {
            "status": "PASS",
            "alarm_endpoint_coverage": alarm_count,
            "alarm_endpoint_missing": 0,
            "archived_vs_recomputed_tolerance_ms": 1e-6,
        },
    }


def east_operational_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "n_shots", "n_disruptions", "n_non_disruptions",
            "operational_TP", "operational_FN", "FP", "TN",
            "operational_tpr", "fpr", "operational_median_lead_ms",
            "minimum_lead_ms", "over_3000ms_count",
        )
    }


def majority_records(records_by_seed: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for aligned in zip(*(records_by_seed[str(seed)] for seed in SEEDS), strict=True):
        shot_ids = {int(row["shot"]) for row in aligned}
        if len(shot_ids) != 1:
            raise RuntimeError("committee shot mismatch")
        alarms = [bool(row["alarmed"]) for row in aligned]
        times = [float(row["alarm_time_s"]) for row in aligned if row.get("alarm_time_s") is not None]
        alarmed = sum(alarms) >= 2
        rows.append({
            "shot": int(aligned[0]["shot"]),
            "alarmed": alarmed,
            "alarm_time_s": float(np.median(times)) if alarmed and times else None,
            "score": float(np.median([
                float(row.get("score", float(row["alarmed"]))) for row in aligned
            ])),
        })
    return rows


def time_blocks(records: list[dict], truth: dict[int, dict[str, Any]]) -> list[dict]:
    if len(records) != 360:
        raise ValueError("balanced360 time blocks require exactly 360 shots")
    result = []
    for block in range(4):
        subset = records[block * 90:(block + 1) * 90]
        row = alarm_metrics(subset, truth)
        row.update({
            "block": block + 1,
            "stream_index_start": block * 90,
            "stream_index_end": (block + 1) * 90 - 1,
            "shot_min": min(int(item["shot"]) for item in subset),
            "shot_max": max(int(item["shot"]) for item in subset),
        })
        result.append(row)
    return result


def correlation(value: Iterable[float], covariate: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(value), dtype=np.float64)
    y = np.asarray(list(covariate), dtype=np.float64)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "p": None, "n": int(len(x))}
    statistic = spearmanr(x, y)
    return {"rho": finite(statistic.statistic), "p": finite(statistic.pvalue), "n": int(len(x))}


def shortcut_correlations(records: list[dict], truth: dict[int, dict[str, Any]]) -> dict:
    result = {}
    for group, wanted in (("disruptive", True), ("safe", False)):
        rows = [row for row in records if truth[int(row["shot"])]["is_disrupt"] == wanted]
        score = [float(row.get("score", float(row["alarmed"]))) for row in rows]
        alarm = [float(bool(row["alarmed"])) for row in rows]
        shot = [int(row["shot"]) for row in rows]
        duration = [truth[int(row["shot"])]["duration_s"] for row in rows]
        result[group] = {
            "score_vs_shot": correlation(score, shot),
            "score_vs_duration": correlation(score, duration),
            "alarm_vs_shot": correlation(alarm, shot),
            "alarm_vs_duration": correlation(alarm, duration),
        }
    return result


def query_analysis(
    predictions: list[dict], query_rows: list[dict], truth: dict[int, dict[str, Any]],
) -> dict:
    by_shot = {int(row["shot"]): row for row in predictions}
    selected = [row for row in query_rows if bool(row["selected"])]
    if {int(row["shot"]) for row in selected} != {
        int(row["shot"]) for row in predictions if bool(row.get("queried"))
    }:
        raise RuntimeError("query ledger and prediction seals disagree")

    def is_operational_error(shot: int) -> bool:
        row = by_shot[shot]
        actual = truth[shot]["is_disrupt"]
        if not actual:
            return bool(row["alarmed"])
        lead = alarm_lead_ms(row, truth)
        return not bool(row["alarmed"]) or lead is None or lead < 30.0 - 1e-9

    all_errors = np.asarray([is_operational_error(int(row["shot"])) for row in predictions])
    selected_errors = np.asarray([is_operational_error(int(row["shot"])) for row in selected])
    chosen_values = [float(row["decision_value_score"]) for row in selected]
    other_values = [
        float(row["decision_value_score"]) for row in query_rows if not bool(row["selected"])
    ]
    test = mannwhitneyu(chosen_values, other_values, alternative="greater")
    per_block = []
    for block in range(4):
        lo, hi = block * 90, (block + 1) * 90
        block_rows = [row for row in selected if lo <= int(row["stream_index"]) < hi]
        per_block.append({
            "block": block + 1,
            "queries": len(block_rows),
            "operational_errors": int(sum(
                is_operational_error(int(row["shot"])) for row in block_rows
            )),
        })
    return {
        "query_count": len(selected),
        "budget_fraction": len(selected) / len(predictions),
        "operational_error_rate_all": float(all_errors.mean()),
        "operational_error_rate_queried": float(selected_errors.mean()),
        "error_enrichment": float(selected_errors.mean() / all_errors.mean()),
        "decision_value_selected_mean": float(np.mean(chosen_values)),
        "decision_value_nonselected_mean": float(np.mean(other_values)),
        "decision_value_one_sided_mannwhitney_p": float(test.pvalue),
        "time_blocks": per_block,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    jtext_audit = jtext_split_audit()
    jtext_test_shots = read_shot_ids(ROOT / "splits/test.txt")
    jtext_test_truth = load_jtext_truth(jtext_test_shots)
    manifest_path = ROOT / "config/splits/east_hdf5_authoritative_balanced360_v1.json"
    manifest = read_json(manifest_path)
    disruption = [int(value) for value in manifest["test"]["disruption_shots"]]
    safe = [int(value) for value in manifest["test"]["non_disruption_shots"]]
    roster = sorted(disruption + safe)
    truth = load_east_truth(roster)
    if sum(item["is_disrupt"] for item in truth.values()) != 180:
        raise RuntimeError("balanced360 HDF5 truth is not 180/180")

    support_manifest = read_json(ROOT / "config/splits/east_paper131_split_all_in.json")
    # The legacy manifest categories are roster provenance only.  Several shots
    # were historically placed under the wrong outcome heading, so all outcome
    # counts must be recomputed from the same HDF5 authority used for evaluation.
    legacy_support_disruption = [
        int(value) for value in support_manifest["train"]["disruption_shots"]
    ]
    legacy_support_safe = [
        int(value) for value in support_manifest["train"]["non_disruption_shots"]
    ]
    support = set(legacy_support_disruption + legacy_support_safe)
    support_truth = load_east_truth(support)
    support_disruption = [
        shot for shot in sorted(support) if support_truth[shot]["is_disrupt"]
    ]
    support_safe = [
        shot for shot in sorted(support) if not support_truth[shot]["is_disrupt"]
    ]
    if len(support) != 110 or len(support_disruption) != 15 or len(support_safe) != 95:
        raise RuntimeError("EAST110 HDF5 truth is not the registered 15D/95N contract")
    if support & set(roster):
        raise RuntimeError("EAST110 support overlaps balanced360")

    static_records: dict[str, list[dict]] = {}
    online_records: dict[str, list[dict]] = {}
    static_metrics: dict[str, dict] = {}
    online_metrics: dict[str, dict] = {}
    query_results: dict[str, dict] = {}
    time_block_rows: list[dict] = []
    source_results: dict[str, dict] = {}
    source_records: dict[str, list[dict]] = {}
    source_training_summaries: dict[str, dict] = {}
    primary_hashes: dict[str, dict] = {}
    source_hashes: dict[str, dict] = {}
    alarm_audits: dict[str, dict] = {}
    for seed in SEEDS:
        run = primary_run(seed)
        static_records[str(seed)] = read_jsonl(run / "static_predictions.jsonl")
        online_records[str(seed)] = read_jsonl(run / "predictions.jsonl")
        summary = read_json(run / "summary.json")
        static_metrics[str(seed)] = alarm_metrics(static_records[str(seed)], truth)
        online_metrics[str(seed)] = alarm_metrics(online_records[str(seed)], truth)
        alarm_audits[str(seed)] = alarm_state_machine_audit(
            run, online_records[str(seed)],
        )
        for phase, computed in (
            ("static_unified", static_metrics[str(seed)]),
            ("online_unified", online_metrics[str(seed)]),
        ):
            registered = summary["results"][phase]
            for key in ("TP", "FP", "FN", "TN"):
                if int(computed[key]) != int(registered[key]):
                    raise RuntimeError(f"seed {seed} {phase} {key} reproduction failed")
            for key in ("tpr", "fpr", "tpr_at_30ms", "median_lead_ms"):
                if not math.isclose(float(computed[key]), float(registered[key]), abs_tol=1e-12):
                    raise RuntimeError(f"seed {seed} {phase} {key} reproduction failed")
        query_results[str(seed)] = query_analysis(
            online_records[str(seed)], read_jsonl(run / "query_decisions.jsonl"), truth,
        )
        for phase, records in (("static", static_records[str(seed)]),
                               ("online", online_records[str(seed)])):
            for row in time_blocks(records, truth):
                time_block_rows.append({"seed": seed, "phase": phase, **row})
        primary_hashes[str(seed)] = {
            name: sha256_artifact(run / name)
            for name in ("summary.json", "predictions.jsonl", "static_predictions.jsonl",
                         "query_decisions.jsonl", "reveals.jsonl", "updates.jsonl")
        }
        source_directory = source_run(seed)
        source_records[str(seed)] = read_jsonl(
            source_directory / "test220_predictions.jsonl"
        )
        if [int(item["shot"]) for item in source_records[str(seed)]] != jtext_test_shots:
            raise RuntimeError(f"seed {seed} source prediction roster/order mismatch")
        source_results[str(seed)] = read_json(
            source_directory / "test220_metrics.json"
        )["metrics"]
        source_training_summaries[str(seed)] = read_json(
            source_directory / "training_summary.json"
        )
        source_hashes[str(seed)] = {
            name: sha256_file(source_directory / name)
            for name in (
                "source_checkpoint.pt", "training_summary.json",
                "test220_metrics.json", "test220_predictions.jsonl",
            )
        }

    metrics_to_aggregate = ("tpr", "fpr", "tpr_at_30ms", "median_lead_ms")
    static_arrays = _phase_arrays(static_records, truth)
    online_arrays = _phase_arrays(online_records, truth)
    static_committee = majority_records(static_records)
    online_committee = majority_records(online_records)
    committee_static_metrics = alarm_metrics(static_committee, truth)
    committee_online_metrics = alarm_metrics(online_committee, truth)

    labels = static_arrays["labels"]
    committee_static_alarm = np.asarray([row["alarmed"] for row in static_committee])
    committee_online_alarm = np.asarray([row["alarmed"] for row in online_committee])
    committee_static_timely = np.asarray([
        (alarm_lead_ms(row, truth) or float("-inf")) >= 30.0 - 1e-9
        for row in static_committee
    ])
    committee_online_timely = np.asarray([
        (alarm_lead_ms(row, truth) or float("-inf")) >= 30.0 - 1e-9
        for row in online_committee
    ])
    paired = {
        "safe_false_alarm": exact_paired_change(
            ~committee_static_alarm[~labels], ~committee_online_alarm[~labels]
        ),
        "disruptive_raw_detection": exact_paired_change(
            committee_static_alarm[labels], committee_online_alarm[labels]
        ),
        "disruptive_timely_detection": exact_paired_change(
            committee_static_timely[labels], committee_online_timely[labels]
        ),
    }

    shortcut = read_json(EVIDENCE / "source_context_shortcut_audit.json")
    padfix_source = read_json(
        PACKAGE / "source_padfix_confirm_e32/test220_three_seed_summary.json"
    )
    padfix_transfer = {
        str(seed): read_json(EVIDENCE / f"raw_padfix_transfer/seed{seed}_summary.json")["results"]
        for seed in SEEDS
    }
    padfix_static = {
        str(seed): padfix_transfer[str(seed)]["static_unified"] for seed in SEEDS
    }
    padfix_online = {
        str(seed): padfix_transfer[str(seed)]["online_unified"] for seed in SEEDS
    }
    padfix_shortcut = {
        str(seed): read_json(
            PACKAGE / f"source_padfix_confirm_e32/seed{seed}/training_summary.json"
        )["validation_context_audit"]
        for seed in SEEDS
    }

    screen = read_json(bundled_or_fallback(
        EVIDENCE / "sequence_episode_v3c_screen.json",
        HERE / "online_episode_value_v3_runs/sequence_episode_v3c_screen.json",
    ))
    controls = {
        name: {
            "total_fp_reduction": value["total_fp_reduction"],
            "total_tp_loss": value["total_tp_loss"],
            "total_operational_tp_loss": int(sum(
                round(float(screen["baselines"][str(seed)]["tpr_at_30ms"]) * 180)
                - round(float(value["results"][str(seed)]["tpr_at_30ms"]) * 180)
                for seed in SEEDS
            )),
        }
        for name, value in screen["negative_controls"].items()
    }
    actual_fp_reduction = int(sum(
        static_metrics[str(seed)]["FP"] - online_metrics[str(seed)]["FP"] for seed in SEEDS
    ))
    actual_tp_change = int(sum(
        online_metrics[str(seed)]["TP"] - static_metrics[str(seed)]["TP"] for seed in SEEDS
    ))
    controls["real_final_v3c"] = {
        "total_fp_reduction": actual_fp_reduction,
        "total_tp_loss": -actual_tp_change,
        "total_operational_tp_loss": int(sum(
            round(float(screen["baselines"][str(seed)]["tpr_at_30ms"]) * 180)
            - int(online_metrics[str(seed)]["operational_TP"])
            for seed in SEEDS
        )),
    }
    robustness = read_json(bundled_or_fallback(
        EVIDENCE / "dual_threshold_alarm_robustness_summary.json",
        HERE / "dual_threshold_alarm_robustness_v1/summary.json",
    ))
    causal_random_query = read_json(
        EVIDENCE / "causal_random_query_online_v1/comparison.json"
    )
    east_target_only = read_json(
        EVIDENCE / "east_target_only_v1/comparison.json"
    )

    source_official = {
        str(seed): source_operational_row(
            source_results[str(seed)], source_records[str(seed)], jtext_test_truth,
        )
        for seed in SEEDS
    }
    static_official = {
        str(seed): east_operational_row(static_metrics[str(seed)]) for seed in SEEDS
    }
    online_official = {
        str(seed): east_operational_row(online_metrics[str(seed)]) for seed in SEEDS
    }
    iid_bootstrap = stratified_bootstrap(static_arrays, online_arrays)
    block_bootstraps = {
        str(block_size): moving_block_bootstrap(
            static_arrays, online_arrays, block_size=block_size,
        )
        for block_size in (10, 20, 30, 45)
    }

    def operational_bootstrap_view(bootstrap: dict[str, Any]) -> dict[str, Any]:
        view = {
        "iterations": bootstrap["iterations"],
        "resampling": bootstrap["resampling"],
        "static_tpr": bootstrap["static_tpr30"],
        "online_tpr": bootstrap["online_tpr30"],
        "delta_tpr": bootstrap["delta_tpr30"],
        "static_fpr": bootstrap["static_fpr"],
        "online_fpr": bootstrap["online_fpr"],
        "delta_fpr": bootstrap["delta_fpr"],
        "static_median_lead_ms": bootstrap["static_operational_median_lead"],
        "online_median_lead_ms": bootstrap["online_operational_median_lead"],
        "delta_median_lead_ms": bootstrap["delta_operational_median_lead"],
        }
        for optional in ("block_size", "inference_boundary"):
            if optional in bootstrap:
                view[optional] = bootstrap[optional]
        return view

    operational_bootstrap = operational_bootstrap_view(block_bootstraps["20"])
    operational_block_sensitivity = {
        size: operational_bootstrap_view(value)
        for size, value in block_bootstraps.items()
    }
    operational_iid_secondary = operational_bootstrap_view(iid_bootstrap)

    payload = {
        "schema_version": "jtext-east-three-stage-paper-evidence-v3",
        "status": "DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION",
        "claim_boundary": (
            "J-TEXT VAL/TEST are count-aligned balanced comparators and balanced360 was "
            "used for architecture and 972-candidate parameter development; no stage is "
            "an independent deployment confirmation"
        ),
        "data_contract": {
            "jtext": jtext_audit,
            "east_support": {
                "shots": len(support), "disruptions": len(support_disruption),
                "safe": len(support_safe),
                "outcome_authority": "HDF5 /meta/IsDisrupt",
                "used_for_target_fitting": True,
                "used_for_target_threshold": True,
                "independent_target_validation": False,
                "threshold_nominal_support_tpr": 0.8,
                "legacy_manifest_categories": {
                    "disruptions": len(legacy_support_disruption),
                    "safe": len(legacy_support_safe),
                    "role": "roster provenance only; not label authority",
                },
            },
            "east_balanced360": {"shots": 360, "disruptions": 180, "safe": 180},
            "support_test_overlap": 0,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "label_authority": "HDF5 /meta/IsDisrupt and /meta/DownTime",
            "demonstrably_pristine_local_future_stream": False,
        },
        "computational_reporting": {
            "environment": "rl_env",
            "device": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "device_memory_mib": 8151,
            "source_training_elapsed_seconds": {
                str(seed): float(source_training_summaries[str(seed)]["elapsed_seconds"])
                for seed in SEEDS
            },
            "target_online_wall_time": None,
            "target_online_flops": None,
            "energy": None,
            "missing_cost_boundary": (
                "historical target/online runner did not archive these values; they are not "
                "reconstructed from file timestamps"
            ),
        },
        "official_metric_contract": {
            "primary_metric": "lead-qualified operational TPR",
            "jtext_minimum_lead": ">10 ms",
            "east_minimum_lead": ">=30 ms",
            "east_operational_tp": "alarmed and lead >=30 ms",
            "east_tardy": "alarmed with lead <30 ms; formal operational error",
            "raw_alarm_detection": "diagnostic_only",
            "median_lead": "median over lead-qualified operational true positives only",
            "alarm_persistence": "first accepted causal supported crossing; no ten-point requirement",
        },
        "official_results": {
            "source_test220": {
                "seeds": source_official,
                "aggregate": aggregate_metric_rows(
                    source_official,
                    ("operational_tpr", "fpr", "auc", "operational_median_lead_ms"),
                ),
            },
            "east_static360": {
                "seeds": static_official,
                "aggregate": aggregate_metric_rows(
                    static_official,
                    ("operational_tpr", "fpr", "operational_median_lead_ms"),
                ),
            },
            "east_online360": {
                "seeds": online_official,
                "aggregate": aggregate_metric_rows(
                    online_official,
                    ("operational_tpr", "fpr", "operational_median_lead_ms"),
                ),
            },
            "bootstrap": operational_bootstrap,
            "moving_block_sensitivity": operational_block_sensitivity,
            "iid_stratified_bootstrap_secondary": operational_iid_secondary,
        },
        "primary_lineage": {
            "source_test220": {
                "metric_scope": "legacy evaluator fields; manuscript uses official_results",
                "seeds": source_results,
                "aggregate": aggregate_metric_rows(
                    source_results, ("auc", "raw_tpr", "timely_tpr", "fpr", "median_lead_ms")
                ),
                "shortcut_audit": shortcut,
                "admission": "FAILED_CONTEXT_SHORTCUT_GATE_AND_NUMERIC_S0_GATE",
            },
            "east_static360": {
                "metric_scope": "legacy raw-alarm diagnostics; manuscript uses official_results",
                "seeds": static_metrics,
                "aggregate": aggregate_metric_rows(static_metrics, metrics_to_aggregate),
            },
            "east_online360": {
                "metric_scope": "legacy raw-alarm diagnostics; manuscript uses official_results",
                "seeds": online_metrics,
                "aggregate": aggregate_metric_rows(online_metrics, metrics_to_aggregate),
                "query_analysis": query_results,
            },
            "committee_posthoc_diagnostic": {
                "static": committee_static_metrics,
                "online": committee_online_metrics,
                "paired_exact": paired,
                "warning": "majority committee was not preregistered and is descriptive only",
            },
            "bootstrap": {
                "primary_descriptive": block_bootstraps["20"],
                "moving_block_sensitivity": block_bootstraps,
                "iid_stratified_secondary": iid_bootstrap,
                "post_selection_warning": (
                    "all intervals are unadjusted descriptive summaries after 972-candidate "
                    "development-stream selection; they are not confirmatory inference"
                ),
            },
            "alarm_state_machine": {
                "formula": (
                    "alarm_t = (margin_t >= high_threshold) OR "
                    "(margin_t + physics_weight*physics_t + trend_weight*"
                    "max(MA5(margin)_t-MA100(margin)_t,0) >= early_threshold)"
                ),
                "adaptive_parameters": [
                    "high_threshold", "physics_weight", "trend_weight", "early_threshold"
                ],
                "parameter_bounds": {
                    "high_threshold": [-0.10, 0.10],
                    "physics_weight": [6.0, 10.0],
                    "trend_weight": [12.0, 15.0],
                    "early_threshold": [7.1, 8.7],
                },
                "fixed_causal_widths": {"fast": 5, "slow": 100},
                "update_rule": {
                    "FP": "stricter thresholds and weaker early evidence",
                    "FN_or_TARDY": "earlier thresholds and stronger early evidence",
                    "TP_or_TN": "add protected guard trajectory",
                },
                "episode_gate": "may delay or veto; cannot create or advance an alarm",
                "seeds": alarm_audits,
            },
            "online_score_shortcut_correlations": {
                str(seed): shortcut_correlations(online_records[str(seed)], truth)
                for seed in SEEDS
            },
            "artifact_hashes": primary_hashes,
            "source_artifact_hashes": source_hashes,
        },
        "padfix_ablation": {
            "source_test220": padfix_source,
            "validation_shortcut_audit": padfix_shortcut,
            "east_static360": {
                "seeds": padfix_static,
                "aggregate": aggregate_metric_rows(padfix_static, metrics_to_aggregate),
            },
            "east_online360": {
                "seeds": padfix_online,
                "aggregate": aggregate_metric_rows(padfix_online, metrics_to_aggregate),
            },
            "decision": (
                "context gates passed, but J-TEXT operational sensitivity fell and frozen v3c "
                "increased EAST false positives in all three seeds; negative transfer"
            ),
        },
        "east_target_only_baselines": east_target_only,
        "negative_controls": {
            "screen_candidate_count": screen["candidate_count"],
            "strict_solution_count": screen["strict_solution_count"],
            "selected_parameters": screen["selected_balanced_candidate"],
            "results": controls,
            "warning": "972 candidates were screened on balanced360; multiplicity is not erased",
            "equal_budget_random_query_available": True,
            "causal_random_query_online": causal_random_query,
        },
        "alarm_robustness_prior_stage": {
            "status": robustness["status"],
            "matrix_size": robustness["matrix_size"],
            "pass_fraction": robustness["neighborhood_pass_fraction"],
            "registered_metrics": robustness["registered_metrics"],
            "bootstrap": robustness["bootstrap_registered"],
            "claim_boundary": robustness["claim_boundary"],
        },
        "time_blocks": time_block_rows,
    }
    payload = normalize_local_paths(payload)
    output = EVIDENCE / "three_stage_results.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    table_rows = []
    for stage, rows in (
        ("J-TEXT source TEST220", source_official),
        ("EAST static balanced360", static_official),
        ("EAST online balanced360", online_official),
    ):
        for seed in SEEDS:
            row = rows[str(seed)]
            table_rows.append({
                "stage": stage, "seed": seed,
                "operational_tp": row["operational_TP"],
                "operational_fn": row["operational_FN"],
                "operational_tpr": row["operational_tpr"],
                "fpr": row["fpr"], "auc": row.get("auc"),
                "operational_median_lead_ms": row["operational_median_lead_ms"],
                "claim_status": (
                    "held-out count-aligned balanced comparator" if stage.startswith("J-TEXT")
                    else "balanced360 development"
                ),
            })
    write_csv(EVIDENCE / "table_three_stage.csv", table_rows)
    write_csv(EVIDENCE / "table_time_blocks.csv", time_block_rows)
    write_csv(EVIDENCE / "table_negative_controls.csv", [
        {"control": name, **values} for name, values in controls.items()
    ])
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "primary_static": payload["official_results"]["east_static360"]["aggregate"],
        "primary_online": payload["official_results"]["east_online360"]["aggregate"],
        "padfix_decision": payload["padfix_ablation"]["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
