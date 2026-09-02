"""Current online-evidence primitives shared by J-TEXT replay and audit.

This module intentionally contains no historical report assembler or experiment
entry point.  It provides only HDF5 truth access, operational metrics, and
source-threshold loading required by the retained online-active branch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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


def _first_scalar(value: Any) -> Any:
    array = np.asarray(value).reshape(-1)
    if not len(array):
        raise RuntimeError("empty HDF5 scalar field")
    return array[0]


class TruthStore:
    """Read HDF5-authoritative outcomes and causal signal time bases."""

    def __init__(self, data_root: Path) -> None:
        self.root = Path(data_root).expanduser().resolve()
        self._truth: dict[tuple[str, int], dict[str, Any]] = {}
        self._timing: dict[tuple[str, int], dict[str, float]] = {}

    def path(self, device: str, shot: int) -> Path:
        if device not in {"JTEXT", "EAST"}:
            raise ValueError(f"unsupported device: {device}")
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
    return float(
        (ranks[y].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


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


def _row_operational(
    row: Mapping[str, Any], device: str,
) -> tuple[bool, float | None]:
    if not bool(row["is_disrupt"]) or not bool(row["alarmed"]):
        return False, None
    lead = (float(row["down_time_s"]) - float(row["alarm_time_s"])) * 1000.0
    timely = lead > 10.0 if device == "JTEXT" else lead >= 30.0
    return bool(timely), float(lead)


def compute_metrics(
    rows: Sequence[Mapping[str, Any]], *, device: str,
) -> dict[str, Any]:
    """Compute shot-level operational metrics under the locked lead rule."""

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
        "minimum_lead_rule": (
            "lead_ms > 10" if device == "JTEXT" else "lead_ms >= 30"
        ),
        "raw_alarm_lead_ms": _lead_summary(raw_leads),
        "operational_tp_lead_ms": _lead_summary(operational_leads),
        "very_early_alarm": {
            "count": very_early,
            "denominator": len(disruptions),
            "rate": very_early / len(disruptions),
            "rule": "lead_ms > 3000",
        },
    }


def compute_delta(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> dict[str, Any]:
    """Compare two sealed prediction sets over identical ordered shots."""

    if [int(row["shot"]) for row in candidate_rows] != [
        int(row["shot"]) for row in baseline_rows
    ]:
        raise RuntimeError("delta requires identical ordered shots")
    candidate = compute_metrics(candidate_rows, device=device)
    baseline = compute_metrics(baseline_rows, device=device)
    rate_keys = ("raw_tpr", "operational_tpr", "fpr", "j_operational", "auc")
    rates = {
        key: (
            None
            if candidate[key] is None or baseline[key] is None
            else candidate[key] - baseline[key]
        )
        for key in rate_keys
    }
    count_keys = (
        "raw_tp",
        "raw_fn",
        "tardy",
        "operational_tp",
        "operational_fn",
        "fp",
        "tn",
    )
    return {
        "candidate_minus_baseline": rates,
        "count_delta": {
            key: candidate["counts"][key] - baseline["counts"][key]
            for key in count_keys
        },
    }


def load_threshold(checkpoint: Path, seed: int) -> dict[str, Any]:
    """Load the threshold selected on J-TEXT VAL160 from a source checkpoint."""

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
