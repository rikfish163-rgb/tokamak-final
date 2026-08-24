#!/usr/bin/env python3
"""Assemble portable dynamic-mainline metrics from sealed records and HDF5 truth."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


DEFAULT_EAST_ROOT = Path(
    "/home/hetaisheng/Fuison-runs/selected_full_mainline/prefix_causal_v2c"
)
DEFAULT_JTEXT_ONLINE_ROOT = Path("/home/hetaisheng/Fuison-runs/jtext_online_v2")
DEFAULT_DATA_ROOT = Path("/media/hetaisheng/044A81D94A81C83E/Fuison/data")
DEFAULT_OUTPUT = Path("/home/hetaisheng/Fuison-runs/dynamic_evidence/report.json")
SEEDS = (131, 133, 137)
CLAIM = "TEST_DRIVEN_EXPLORATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"empty JSONL artifact: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"non-object JSONL row: {path}")
    return rows


def _safe_relative_file(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"artifact path escapes root: {relative}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _first_scalar(value: Any) -> Any:
    array = np.asarray(value).reshape(-1)
    if not len(array):
        raise RuntimeError("empty HDF5 scalar field")
    return array[0]


class TruthStore:
    def __init__(self, data_root: Path) -> None:
        self.root = Path(data_root).expanduser().resolve()
        self._truth: dict[tuple[str, int], dict[str, Any]] = {}
        self._timing: dict[tuple[str, int], dict[str, float]] = {}

    def path(self, device: str, shot: int) -> Path:
        directory = int(shot) // 100 * 100
        name = "J-TEXT" if device == "JTEXT" else "EAST"
        return self.root / name / "unified_hdf5" / str(directory) / f"{shot}.hdf5"

    def truth(self, device: str, shot: int) -> dict[str, Any]:
        key = (device, int(shot))
        if key not in self._truth:
            path = self.path(device, shot)
            with h5py.File(path, "r") as handle:
                disruptive = bool(_first_scalar(handle["meta/IsDisrupt"][()]))
                down = float(_first_scalar(handle["meta/DownTime"][()]))
            if disruptive and not math.isfinite(down):
                raise RuntimeError(f"non-finite disruptive DownTime: {device} {shot}")
            self._truth[key] = {
                "is_disrupt": disruptive,
                "down_time_s": down if disruptive else None,
            }
        return dict(self._truth[key])

    def timing(self, device: str, shot: int) -> dict[str, float]:
        key = (device, int(shot))
        if key not in self._timing:
            path = self.path(device, shot)
            with h5py.File(path, "r") as handle:
                signal = handle["data/ip"]
                start = float(signal.attrs["StartTime"])
                rate = float(signal.attrs["SampleRate"])
            if not math.isfinite(start) or not math.isfinite(rate) or rate <= 0:
                raise RuntimeError(f"invalid HDF5 timebase: {device} {shot}")
            self._timing[key] = {
                "start_time_s": start,
                "sample_rate_hz": rate,
            }
        return dict(self._timing[key])


def rebuild_jtext_endpoint_rows(
    records: Sequence[Mapping[str, Any]], *, threshold: float,
    truth_by_shot: Mapping[int, Mapping[str, Any]],
    timing_by_shot: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("source checkpoint VAL threshold must lie in (0, 1)")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("event") != "endpoint_predict":
            continue
        shot = int(record["shot"])
        grouped.setdefault(shot, []).append(record)
    expected = set(map(int, truth_by_shot))
    if set(grouped) != expected or set(map(int, timing_by_shot)) != expected:
        raise RuntimeError("J-TEXT endpoint/truth/timebase shot sets differ")
    rows: list[dict[str, Any]] = []
    for shot, shot_records in grouped.items():
        ordered = list(shot_records)
        endpoints = [int(row["endpoint"]) for row in ordered]
        if (
            any(endpoint < 0 for endpoint in endpoints)
            or any(right <= left for left, right in zip(endpoints, endpoints[1:]))
        ):
            raise RuntimeError(f"J-TEXT endpoints are not strictly increasing and unique: {shot}")
        risks = np.asarray([float(row["risk"]) for row in ordered], dtype=np.float64)
        if not np.isfinite(risks).all():
            raise RuntimeError(f"non-finite J-TEXT risk: {shot}")
        passing = np.flatnonzero(risks >= float(threshold))
        alarm_endpoint = endpoints[int(passing[0])] if len(passing) else None
        timing = timing_by_shot[shot]
        alarm_time = (
            float(timing["start_time_s"])
            + alarm_endpoint / float(timing["sample_rate_hz"])
            if alarm_endpoint is not None else None
        )
        rows.append({
            "shot": shot,
            **dict(truth_by_shot[shot]),
            "score": float(np.max(risks)),
            "alarmed": alarm_endpoint is not None,
            "alarm_endpoint": alarm_endpoint,
            "alarm_time_s": alarm_time,
        })
    return rows


def rebuild_east_rows(
    predictions: Sequence[Mapping[str, Any]], *,
    truth_by_shot: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    shots = [int(row["shot"]) for row in predictions]
    if len(shots) != len(set(shots)) or set(shots) != set(map(int, truth_by_shot)):
        raise RuntimeError("EAST prediction/truth shot sets differ")
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        shot = int(prediction["shot"])
        alarmed = bool(prediction["alarmed"])
        raw_time = prediction.get("alarm_time_s")
        if alarmed and (raw_time is None or not math.isfinite(float(raw_time))):
            raise RuntimeError(f"alarmed EAST shot lacks alarm_time_s: {shot}")
        if not alarmed and raw_time is not None:
            raise RuntimeError(f"non-alarmed EAST shot has alarm_time_s: {shot}")
        score = prediction.get("score")
        rows.append({
            "shot": shot,
            **dict(truth_by_shot[shot]),
            "score": float(score) if score is not None else None,
            "alarmed": alarmed,
            "alarm_time_s": float(raw_time) if raw_time is not None else None,
        })
    return rows


def _auc(labels: Sequence[bool], scores: Sequence[float | None]) -> float | None:
    if any(score is None or not math.isfinite(float(score)) for score in scores):
        return None
    y = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _lead_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median_ms": None, "q25_ms": None, "q75_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "median_ms": float(np.median(array)),
        "q25_ms": float(np.quantile(array, 0.25)),
        "q75_ms": float(np.quantile(array, 0.75)),
    }


def _row_operational(row: Mapping[str, Any], device: str) -> tuple[bool, float | None]:
    if not bool(row["is_disrupt"]) or not bool(row["alarmed"]):
        return False, None
    lead = (float(row["down_time_s"]) - float(row["alarm_time_s"])) * 1000.0
    timely = lead > 10.0 if device == "JTEXT" else lead >= 30.0
    return bool(timely), float(lead)


def compute_metrics(rows: Sequence[Mapping[str, Any]], *, device: str) -> dict[str, Any]:
    if device not in {"JTEXT", "EAST"} or not rows:
        raise ValueError("metrics require non-empty JTEXT or EAST rows")
    disruptions = [row for row in rows if bool(row["is_disrupt"])]
    safe = [row for row in rows if not bool(row["is_disrupt"])]
    if not disruptions or not safe:
        raise RuntimeError("metrics require disruptive and non-disruptive shots")
    raw_tp = sum(bool(row["alarmed"]) for row in disruptions)
    fp = sum(bool(row["alarmed"]) for row in safe)
    raw_leads: list[float] = []
    operational_leads: list[float] = []
    operational_tp = 0
    for row in disruptions:
        timely, lead = _row_operational(row, device)
        if lead is not None:
            raw_leads.append(lead)
        if timely:
            operational_tp += 1
            operational_leads.append(float(lead))
    counts = {
        "disruptions": len(disruptions),
        "non_disruptions": len(safe),
        "raw_tp": raw_tp,
        "raw_fn": len(disruptions) - raw_tp,
        "tardy": raw_tp - operational_tp,
        "operational_tp": operational_tp,
        "operational_fn": len(disruptions) - operational_tp,
        "fp": fp,
        "tn": len(safe) - fp,
    }
    raw_tpr = raw_tp / len(disruptions)
    operational_tpr = operational_tp / len(disruptions)
    fpr = fp / len(safe)
    very_early = sum(lead > 3000.0 for lead in raw_leads)
    return {
        "n_shots": len(rows),
        "counts": counts,
        "raw_tpr": raw_tpr,
        "operational_tpr": operational_tpr,
        "fpr": fpr,
        "j_operational": operational_tpr - fpr,
        "auc": _auc(
            [bool(row["is_disrupt"]) for row in rows],
            [row.get("score") for row in rows],
        ),
        "minimum_lead_rule": "lead_ms > 10" if device == "JTEXT" else "lead_ms >= 30",
        "raw_alarm_lead_ms": _lead_summary(raw_leads),
        "operational_tp_lead_ms": _lead_summary(operational_leads),
        "very_early_alarm": {
            "count": very_early,
            "denominator": len(disruptions),
            "rate": very_early / len(disruptions),
            "rule": "lead_ms > 3000",
        },
    }


def _delta(
    candidate_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]],
    *, device: str,
) -> dict[str, Any]:
    if [int(row["shot"]) for row in candidate_rows] != [int(row["shot"]) for row in baseline_rows]:
        raise RuntimeError("delta requires identical ordered shots")
    candidate = compute_metrics(candidate_rows, device=device)
    baseline = compute_metrics(baseline_rows, device=device)
    rate_keys = ("raw_tpr", "operational_tpr", "fpr", "j_operational", "auc")
    rates = {
        key: (None if candidate[key] is None or baseline[key] is None else candidate[key] - baseline[key])
        for key in rate_keys
    }
    count_keys = ("raw_tp", "raw_fn", "tardy", "operational_tp", "operational_fn", "fp", "tn")
    return {
        "candidate_minus_baseline": rates,
        "count_delta": {
            key: candidate["counts"][key] - baseline["counts"][key]
            for key in count_keys
        },
    }


def _absolute_strings(value: Any, location: str = "report_body") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _absolute_strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _absolute_strings(child, f"{location}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        yield location


def validate_portable_report(report: Mapping[str, Any]) -> None:
    body = report.get("report_body")
    provenance = report.get("nonportable_provenance")
    if not isinstance(body, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("report requires report_body and nonportable_provenance")
    leaks = list(_absolute_strings(body))
    if leaks:
        raise ValueError(f"absolute path in portable report body: {leaks[0]}")


def _load_threshold(checkpoint: Path, seed: int) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"source checkpoint seed mismatch: {seed}")
    threshold = float(payload["validation_threshold"])
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(f"invalid source checkpoint VAL threshold: {seed}")
    receipt = payload.get("validation_threshold_receipt")
    return {
        "value": threshold,
        "origin": "source_checkpoint.validation_threshold",
        "selection_split": "JTEXT_VAL160",
        "test_threshold_search_performed": False,
        "receipt_sha256": sha256_json(receipt) if receipt is not None else None,
    }


class ArtifactIndex:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}

    def add(self, root_id: str, root: Path, path: Path, role: str) -> dict[str, Any]:
        root = Path(root).resolve()
        path = Path(path).resolve()
        relative = path.relative_to(root).as_posix()
        key = (root_id, relative)
        entry = {
            "root_id": root_id,
            "path": relative,
            "role": role,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        prior = self._entries.get(key)
        if prior is not None and prior["sha256"] != entry["sha256"]:
            raise RuntimeError(f"artifact changed during assembly: {relative}")
        self._entries[key] = entry
        return entry

    def rows(self) -> list[dict[str, Any]]:
        return [self._entries[key] for key in sorted(self._entries)]


def _truth_maps(
    store: TruthStore, device: str, shots: Sequence[int], *, timing: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, float]]]:
    truth = {int(shot): store.truth(device, int(shot)) for shot in shots}
    times = {int(shot): store.timing(device, int(shot)) for shot in shots} if timing else {}
    return truth, times


def _load_jtext_static_rows(
    stage: Path, threshold: float, store: TruthStore,
) -> tuple[list[dict[str, Any]], str]:
    records = read_jsonl(stage / "static_predictions.jsonl")
    shots = [int(row["shot"]) for row in records]
    if len(shots) != 220 or len(set(shots)) != 220:
        raise RuntimeError("J-TEXT static seal must contain 220 unique shots")
    truth, timing = _truth_maps(store, "JTEXT", shots, timing=True)
    trace_manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        shot = int(record["shot"])
        trace_path = _safe_relative_file(stage, str(record["risk_trace_path"]))
        actual_sha = sha256_file(trace_path)
        if actual_sha != str(record["risk_trace_sha256"]):
            raise RuntimeError(f"J-TEXT risk trace SHA mismatch: {shot}")
        with np.load(trace_path, allow_pickle=False) as payload:
            trace_shot = int(np.asarray(payload["shot_id"]).reshape(-1)[0])
            risk = np.asarray(payload["risk_trace"], dtype=np.float64)
        if trace_shot != shot or risk.ndim != 1 or not len(risk) or not np.isfinite(risk).all():
            raise RuntimeError(f"invalid J-TEXT sealed risk trace: {shot}")
        passing = np.flatnonzero(risk >= threshold)
        endpoint = int(passing[0]) if len(passing) else None
        timebase = timing[shot]
        if not math.isclose(float(record["start_time_s"]), timebase["start_time_s"], abs_tol=1e-12):
            raise RuntimeError(f"J-TEXT StartTime mismatch: {shot}")
        if not math.isclose(float(record["sample_rate_hz"]), timebase["sample_rate_hz"], abs_tol=1e-12):
            raise RuntimeError(f"J-TEXT SampleRate mismatch: {shot}")
        rows.append({
            "shot": shot,
            **truth[shot],
            "score": float(np.max(risk)),
            "alarmed": endpoint is not None,
            "alarm_endpoint": endpoint,
            "alarm_time_s": (
                timebase["start_time_s"] + endpoint / timebase["sample_rate_hz"]
                if endpoint is not None else None
            ),
        })
        trace_manifest.append({"shot": shot, "sha256": actual_sha})
    return rows, sha256_json(trace_manifest)


def _load_east_prediction_rows(
    path: Path, store: TruthStore,
) -> list[dict[str, Any]]:
    predictions = read_jsonl(path)
    shots = [int(row["shot"]) for row in predictions]
    if len(shots) != 360:
        raise RuntimeError("EAST prediction ledger must contain 360 rows")
    truth, _ = _truth_maps(store, "EAST", shots)
    return rebuild_east_rows(predictions, truth_by_shot=truth)


def _query_resource_audit(
    active_queries: Sequence[Mapping[str, Any]], random_queries: Sequence[Mapping[str, Any]],
    active_updates: Sequence[Mapping[str, Any]], random_updates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def per_block(rows: Sequence[Mapping[str, Any]]) -> list[int]:
        counts: dict[int, int] = defaultdict(int)
        for row in rows:
            if bool(row["selected"]):
                counts[int(row["block"])] += 1
        max_block = max(int(row["block"]) for row in rows)
        return [counts[index] for index in range(max_block + 1)]

    resource_names = (
        "optimizer_attempts", "current_windows", "replay_windows",
        "guard_windows", "replay_size", "flop_inputs", "reported_flops",
    )
    def resources(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{name: row["compute"][name] for name in resource_names} for row in rows]

    active_per_block = per_block(active_queries)
    random_per_block = per_block(random_queries)
    active_resources = resources(active_updates)
    random_resources = resources(random_updates)
    totals = {}
    for name in ("optimizer_attempts", "current_windows", "replay_windows", "guard_windows", "reported_flops"):
        totals[name] = {
            "active": sum(int(row[name]) for row in active_resources),
            "random": sum(int(row[name]) for row in random_resources),
        }
    active_count = sum(bool(row["selected"]) for row in active_queries)
    random_count = sum(bool(row["selected"]) for row in random_queries)
    return {
        "active_queries": active_count,
        "random_queries": random_count,
        "queries_per_block": {"active": active_per_block, "random": random_per_block},
        "resource_totals": totals,
        "per_attempt_resources_match": active_resources == random_resources,
        "matched": bool(
            active_count == random_count
            and active_per_block == random_per_block
            and active_resources == random_resources
        ),
    }


def assemble_report(
    *, east_root: Path, jtext_online_root: Path, data_root: Path,
) -> dict[str, Any]:
    east_root = Path(east_root).expanduser().resolve()
    jtext_online_root = Path(jtext_online_root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    for root in (east_root, jtext_online_root, data_root):
        if not root.is_dir():
            raise FileNotFoundError(root)
    artifacts = ArtifactIndex()
    truth_store = TruthStore(data_root)
    thresholds: dict[int, dict[str, Any]] = {}
    jtext_static_rows: dict[int, list[dict[str, Any]]] = {}
    jtext_static: dict[str, Any] = {}
    east_results: dict[str, Any] = {}

    for seed in SEEDS:
        seed_root = east_root / f"seed{seed}"
        checkpoint = seed_root / "source-train/source_checkpoint.pt"
        parent = seed_root / "east-parent/east_fixed_parent.pt"
        threshold = _load_threshold(checkpoint, seed)
        thresholds[seed] = threshold
        artifacts.add("formal_mainline", east_root, checkpoint, "source_checkpoint")
        artifacts.add("formal_mainline", east_root, parent, "east_parent")

        j_stage = seed_root / "jtext-static"
        j_rows, trace_set_sha = _load_jtext_static_rows(
            j_stage, float(threshold["value"]), truth_store,
        )
        jtext_static_rows[seed] = j_rows
        j_metrics = compute_metrics(j_rows, device="JTEXT")
        j_metrics.update({"threshold": threshold["value"], "trace_set_sha256": trace_set_sha})
        jtext_static[str(seed)] = j_metrics
        artifacts.add("formal_mainline", east_root, j_stage / "static_predictions.jsonl", "jtext_static_predictions")
        artifacts.add("formal_mainline", east_root, j_stage / "static_seal_manifest.json", "jtext_static_prediction_manifest")

        static_path = seed_root / "east-static/static_predictions.jsonl"
        online_path = seed_root / "east-online/predictions.jsonl"
        static_rows = _load_east_prediction_rows(static_path, truth_store)
        online_rows = _load_east_prediction_rows(online_path, truth_store)
        if [row["shot"] for row in static_rows] != [row["shot"] for row in online_rows]:
            raise RuntimeError(f"EAST static/online roster mismatch: {seed}")
        east_results[str(seed)] = {
            "static": compute_metrics(static_rows, device="EAST"),
            "online": compute_metrics(online_rows, device="EAST"),
            "delta": _delta(online_rows, static_rows, device="EAST"),
        }
        artifacts.add("formal_mainline", east_root, static_path, "east_static_predictions")
        artifacts.add("formal_mainline", east_root, seed_root / "east-static/static_seal_manifest.json", "east_static_prediction_manifest")
        artifacts.add("formal_mainline", east_root, online_path, "east_online_predictions")
        for filename, role in (
            ("query_decisions.jsonl", "east_query_ledger"),
            ("reveals.jsonl", "east_reveal_ledger"),
            ("updates.jsonl", "east_update_ledger"),
        ):
            artifacts.add("formal_mainline", east_root, seed_root / "east-online" / filename, role)

    jrun_cache: dict[tuple[int, str], list[dict[str, Any]]] = {}
    def load_jrun(seed: int, directory: str) -> list[dict[str, Any]]:
        key = (seed, directory)
        if key in jrun_cache:
            return jrun_cache[key]
        run = jtext_online_root / f"seed{seed}" / directory
        endpoint_path = run / "endpoint_receipts.jsonl"
        endpoint_rows = read_jsonl(endpoint_path)
        shots = list(dict.fromkeys(
            int(row["shot"]) for row in endpoint_rows
            if row.get("event") == "endpoint_predict"
        ))
        if len(shots) != 220:
            raise RuntimeError(f"J-TEXT online run lacks 220 shots: {seed} {directory}")
        truth, timing = _truth_maps(truth_store, "JTEXT", shots, timing=True)
        rebuilt = rebuild_jtext_endpoint_rows(
            endpoint_rows, threshold=float(thresholds[seed]["value"]),
            truth_by_shot=truth, timing_by_shot=timing,
        )
        if [row["shot"] for row in rebuilt] != [row["shot"] for row in jtext_static_rows[seed]]:
            raise RuntimeError(f"J-TEXT static/online roster mismatch: {seed} {directory}")
        artifacts.add("jtext_online", jtext_online_root, endpoint_path, "jtext_endpoint_predictions")
        for filename, role in (
            ("query_ledger.jsonl", "jtext_query_ledger"),
            ("update_ledger.jsonl", "jtext_update_ledger"),
            ("online_events.jsonl", "jtext_online_event_ledger"),
            ("static_seal_manifest.json", "jtext_static_prediction_manifest"),
        ):
            artifacts.add("jtext_online", jtext_online_root, run / filename, role)
        jrun_cache[key] = rebuilt
        return rebuilt

    jtext_full_active: dict[str, Any] = {}
    for seed in SEEDS:
        rows = load_jrun(seed, "full_active_b20")
        jtext_full_active[str(seed)] = {
            "metrics": compute_metrics(rows, device="JTEXT"),
            "delta_vs_static": _delta(rows, jtext_static_rows[seed], device="JTEXT"),
        }

    budgets = {}
    for budget in (10, 20, 30):
        rows = load_jrun(131, f"full_active_b{budget}")
        budgets[f"b{budget}"] = {
            "metrics": compute_metrics(rows, device="JTEXT"),
            "delta_vs_static": _delta(rows, jtext_static_rows[131], device="JTEXT"),
        }

    arm_directories = {
        "static": "static_b20",
        "intra_only": "intra_only_b20",
        "inter_active": "inter_active_b20",
        "full_active": "full_active_b20",
        "full_random": "full_random_b20",
    }
    arms = {}
    for arm, directory in arm_directories.items():
        rows = load_jrun(131, directory)
        arms[arm] = {
            "metrics": compute_metrics(rows, device="JTEXT"),
            "delta_vs_static_source": _delta(rows, jtext_static_rows[131], device="JTEXT"),
        }

    active_run = jtext_online_root / "seed131/full_active_b20"
    random_run = jtext_online_root / "seed131/full_random_b20"
    query_resource = _query_resource_audit(
        read_jsonl(active_run / "query_ledger.jsonl"),
        read_jsonl(random_run / "query_ledger.jsonl"),
        read_jsonl(active_run / "update_ledger.jsonl"),
        read_jsonl(random_run / "update_ledger.jsonl"),
    )

    front_dir = "frontload_equal_labels_b20"
    front_rows = load_jrun(131, front_dir)
    front_query_path = jtext_online_root / "seed131" / front_dir / "query_ledger.jsonl"
    front_queries = read_jsonl(front_query_path)
    selected_indices = [int(row["stream_index"]) for row in front_queries if bool(row["selected"])]
    prefix_count = len(selected_indices)
    if prefix_count != 44 or selected_indices != list(range(prefix_count)):
        raise RuntimeError("frontload selected shots are not the frozen 44-shot prefix")
    mask = [index >= prefix_count for index in range(220)]
    static_suffix = [row for index, row in enumerate(jtext_static_rows[131]) if mask[index]]
    trained_suffix = [row for index, row in enumerate(front_rows) if mask[index]]
    mask_sha = sha256_json(mask)
    frontload = {
        "prefix_training_shots": prefix_count,
        "evaluation_shots": len(static_suffix),
        "evaluation_mask_sha256": mask_sha,
        "same_mask_verified": [row["shot"] for row in static_suffix] == [row["shot"] for row in trained_suffix],
        "static": compute_metrics(static_suffix, device="JTEXT"),
        "trained": compute_metrics(trained_suffix, device="JTEXT"),
        "delta_trained_minus_static": _delta(trained_suffix, static_suffix, device="JTEXT"),
    }
    if not frontload["same_mask_verified"]:
        raise RuntimeError("frontload suffix static/trained masks differ")

    body = {
        "schema_version": "dynamic-results-portable-v1",
        "claim": CLAIM,
        "metric_recomputation": {
            "truth_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
            "jtext_static": "sealed per-shot risk traces; first source-VAL-threshold crossing",
            "jtext_online": "endpoint_predict rows; first source-VAL-threshold crossing",
            "east": "sealed alarmed and alarm_time_s fields",
            "test_threshold_search_performed": False,
            "summary_metrics_used": False,
        },
        "thresholds": {str(seed): thresholds[seed] for seed in SEEDS},
        "results": {
            "jtext_static_three_seed": jtext_static,
            "east_static_online_three_seed": east_results,
            "jtext_full_active_three_seed": jtext_full_active,
            "jtext_seed131_budget": budgets,
            "jtext_seed131_arm_matrix": arms,
            "jtext_seed131_active_random_resource_match": query_resource,
            "jtext_seed131_frontload_common_176": frontload,
        },
        "admission": {
            "S0": {
                "admitted": False,
                "reason": "The source arm is a post-adjudication user-selected development candidate; its historical preregistered per-seed source gate failed.",
            },
            "T0": {
                "admitted": False,
                "reason": "EAST balanced360 was already used for development and arm inspection; it is not an untouched prospective target cohort.",
            },
            "ONLINE": {
                "admitted": False,
                "reason": "JTEXT TEST220 and EAST balanced360 online replays are ordered prequential diagnostics on inspected cohorts, not a new future chronological stream.",
            },
        },
        "artifact_hashes": artifacts.rows(),
    }
    report = {
        "report_body": body,
        "nonportable_provenance": {
            "east_root": str(east_root),
            "jtext_online_root": str(jtext_online_root),
            "data_root": str(data_root),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    validate_portable_report(report)
    return report


def _write_report(report: dict[str, Any], output: Path) -> None:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report["nonportable_provenance"]["output"] = str(output)
    validate_portable_report(report)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--east-root", type=Path, default=DEFAULT_EAST_ROOT)
    parser.add_argument("--jtext-online-root", type=Path, default=DEFAULT_JTEXT_ONLINE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = assemble_report(
        east_root=args.east_root,
        jtext_online_root=args.jtext_online_root,
        data_root=args.data_root,
    )
    _write_report(report, args.output)
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "claim": report["report_body"]["claim"],
        "artifact_count": len(report["report_body"]["artifact_hashes"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
