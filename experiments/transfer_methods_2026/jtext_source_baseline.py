#!/usr/bin/env python3
"""Train and evaluate the rebuilt17 J-TEXT source parent before EAST adaptation.

The split and metric contract is deliberately separate from EAST:

* train on canonical J-TEXT TRAIN1353 only;
* choose the alarm threshold on J-TEXT VAL160 only;
* evaluate once on balanced J-TEXT TEST220;
* count a disruptive shot as a TP only when its first causal threshold crossing
  is strictly more than 10 ms before HDF5 ``meta/DownTime``.

HDF5 ``meta/IsDisrupt`` and ``meta/DownTime`` are the only label authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
import rebuilt_harness as RH  # noqa: E402


PROTOCOL = "jtext-rebuilt17-source-test220-lead-gt10-padfix-v2"
EXPECTED_COUNTS = {"train": 1353, "val": 160, "test": 220}
EXPECTED_DISRUPTIVE = {"train": 188, "val": 80, "test": 110}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_contract(root: Path = ROOT) -> dict[str, list[int]]:
    roles: dict[str, list[int]] = {}
    for role, expected in EXPECTED_COUNTS.items():
        path = root / "splits" / f"{role}.txt"
        values = [int(value) for value in path.read_text().split()]
        if len(values) != expected or len(values) != len(set(values)):
            raise ValueError(
                f"J-TEXT {role} must contain {expected} unique shots; got {len(values)}"
            )
        roles[role] = values
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(roles[left]) & set(roles[right])
        if overlap:
            raise ValueError(
                f"J-TEXT split overlap {left}/{right}: {sorted(overlap)[:10]}"
            )
    return roles


def _load_role(data_root: Path, shot_ids: Sequence[int]) -> list[dict[str, Any]]:
    shots: list[dict[str, Any]] = []
    missing: list[int] = []
    for shot_id in shot_ids:
        path = (
            data_root / "J-TEXT" / "unified_hdf5"
            / str(int(shot_id) // 100 * 100) / f"{int(shot_id)}.hdf5"
        )
        try:
            arr, rate, start, down, is_disrupt = RH.load_raw(path, "J-TEXT")
        except (FileNotFoundError, OSError, KeyError, ValueError):
            missing.append(int(shot_id))
            continue
        values, available = RH.rebuild_features(arr, "J-TEXT")
        shots.append({
            "shot": int(shot_id),
            "v": values.astype(np.float32),
            "a": available,
            "is_disrupt": bool(is_disrupt),
            "down": float(down) if is_disrupt and down is not None else None,
            "start": float(start),
            "rate": float(rate),
        })
    if missing:
        raise FileNotFoundError(
            f"missing or invalid J-TEXT HDF5 shots ({len(missing)}): {missing[:10]}"
        )
    return shots


def load_jtext_roles(
    data_root: Path, root: Path = ROOT,
) -> tuple[dict[str, list[int]], dict[str, list[dict[str, Any]]]]:
    roles = load_split_contract(root)
    shots = {name: _load_role(data_root, ids) for name, ids in roles.items()}
    for name, rows in shots.items():
        disruptive = sum(bool(row["is_disrupt"]) for row in rows)
        expected = EXPECTED_DISRUPTIVE[name]
        if disruptive != expected:
            raise ValueError(
                f"J-TEXT {name} HDF5 labels contain {disruptive} disruptions; "
                f"canonical contract requires {expected}"
            )
    return roles, shots


def predict_trace(
    enc: torch.nn.Module, head: torch.nn.Module, shot: Mapping[str, Any],
    *, batch_size: int = 512,
) -> dict[str, Any]:
    values = np.asarray(shot["v"], dtype=np.float32)
    endpoints = np.arange(len(values), dtype=np.int64)
    risk = np.empty(len(endpoints), dtype=np.float32)
    enc.eval()
    head.eval()
    for offset in range(0, len(endpoints), int(batch_size)):
        selected = endpoints[offset:offset + int(batch_size)]
        windows = np.stack([
            GS.causal_window(values, endpoint=end, width=GS.W) for end in selected
        ])
        with torch.no_grad():
            probability = torch.sigmoid(
                head(enc(torch.from_numpy(windows).to(GS.DEV)))
            )
        risk[offset:offset + len(selected)] = probability.cpu().numpy()
    times = float(shot["start"]) + endpoints.astype(np.float64) / float(shot["rate"])
    return {
        "shot": int(shot["shot"]),
        "risk": risk,
        "times": times,
        "is_disrupt": bool(shot["is_disrupt"]),
        "down_time_s": shot["down"],
    }


def predict_role(
    enc: torch.nn.Module, head: torch.nn.Module,
    shots: Sequence[Mapping[str, Any]], *, batch_size: int = 512,
) -> list[dict[str, Any]]:
    return [
        predict_trace(enc, head, shot, batch_size=batch_size) for shot in shots
    ]


def _available_score(row: Mapping[str, Any], minimum_lead_ms: float) -> float:
    risk = np.asarray(row["risk"], dtype=np.float64)
    times = np.asarray(row["times"], dtype=np.float64)
    if len(risk) != len(times):
        raise ValueError("risk trace and endpoint times must be aligned")
    if not len(risk):
        return float("-inf")
    if not bool(row["is_disrupt"]):
        return float(np.max(risk))
    down = row.get("down_time_s")
    if down is None or not np.isfinite(float(down)):
        raise ValueError("disruptive J-TEXT shot requires finite HDF5 DownTime")
    lead_ms = (float(down) - times) * 1000.0
    legal = lead_ms > float(minimum_lead_ms)
    return float(np.max(risk[legal])) if np.any(legal) else float("-inf")


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def evaluate_threshold(
    rows: Sequence[Mapping[str, Any]], *, threshold: float,
    minimum_lead_ms: float = 10.0,
) -> dict[str, Any]:
    if not np.isfinite(float(threshold)):
        raise ValueError("alarm threshold must be finite")
    if float(minimum_lead_ms) < 0:
        raise ValueError("minimum lead must be non-negative")
    tp = fp = tn = fn = tardy = 0
    timely_leads: list[float] = []
    records: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    disruptive_alarm_count = 0
    for row in rows:
        risk = np.asarray(row["risk"], dtype=np.float64)
        times = np.asarray(row["times"], dtype=np.float64)
        if risk.ndim != 1 or times.ndim != 1 or len(risk) != len(times):
            raise ValueError("each risk trace must be a one-dimensional aligned series")
        crossings = np.flatnonzero(risk >= float(threshold))
        alarm_index = int(crossings[0]) if len(crossings) else None
        alarm_time = float(times[alarm_index]) if alarm_index is not None else None
        is_disrupt = bool(row["is_disrupt"])
        classification: str
        lead_ms: float | None = None
        if is_disrupt:
            labels.append(1)
            scores.append(_available_score(row, minimum_lead_ms))
            if alarm_index is None:
                fn += 1
                classification = "FN"
            else:
                disruptive_alarm_count += 1
                down = row.get("down_time_s")
                if down is None or not np.isfinite(float(down)):
                    raise ValueError("disruptive J-TEXT shot requires finite HDF5 DownTime")
                lead_ms = float(round((float(down) - alarm_time) * 1000.0, 12))
                if lead_ms > float(minimum_lead_ms):
                    tp += 1
                    timely_leads.append(lead_ms)
                    classification = "TP"
                else:
                    fn += 1
                    tardy += 1
                    classification = "TARDY"
        else:
            labels.append(0)
            scores.append(_available_score(row, minimum_lead_ms))
            if alarm_index is None:
                tn += 1
                classification = "TN"
            else:
                fp += 1
                classification = "FP"
        records.append({
            "shot": int(row["shot"]),
            "is_disrupt": is_disrupt,
            "classification": classification,
            "alarmed": alarm_index is not None,
            "alarm_index": alarm_index,
            "alarm_time_s": alarm_time,
            "lead_ms": lead_ms,
            "max_risk": float(np.max(risk)) if len(risk) else None,
            "timely_available_score": (
                float(scores[-1]) if np.isfinite(scores[-1]) else None
            ),
        })
    disruptive = tp + fn
    safe = fp + tn
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "tardy_alarm_count": tardy,
        "disruptive_alarm_count": disruptive_alarm_count,
        "tpr": tp / max(disruptive, 1),
        "tpr_at_10ms": tp / max(disruptive, 1),
        "fpr": fp / max(safe, 1),
        "median_lead_ms": (
            float(np.median(timely_leads)) if timely_leads else None
        ),
        "timely_auc": _auc(labels_array, scores_array),
        "minimum_lead_ms": float(minimum_lead_ms),
        "lead_comparison": "strictly_greater_than",
        "threshold": float(threshold),
        "shots": records,
    }


def select_validation_threshold(
    rows: Sequence[Mapping[str, Any]], *, target_tpr: float = 0.90,
    minimum_lead_ms: float = 10.0,
) -> dict[str, Any]:
    if not 0 < float(target_tpr) <= 1:
        raise ValueError("target validation TPR must lie in (0, 1]")
    labels = np.asarray([int(bool(row["is_disrupt"])) for row in rows], dtype=np.int8)
    scores = np.asarray([
        _available_score(row, minimum_lead_ms) for row in rows
    ], dtype=np.float64)
    positive_scores = scores[labels == 1]
    maximum_structural_tpr = float(np.mean(np.isfinite(positive_scores)))
    if maximum_structural_tpr <= 0:
        raise ValueError(
            "validation has no disruptive shot with a legal causal alarm endpoint"
        )
    nominal_target_reached = maximum_structural_tpr >= float(target_tpr) - 1e-12
    effective_target_tpr = min(float(target_tpr), maximum_structural_tpr)
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        raise ValueError("validation contains no legal causal alarm endpoint")
    candidates = sorted(set(float(value) for value in finite), reverse=True)
    candidates.append(float(np.nextafter(np.min(finite), -np.inf)))
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        positive = positive_scores
        negative = scores[labels == 0]
        tpr = float(np.mean(positive >= threshold))
        fpr = float(np.mean(negative >= threshold))
        if tpr < effective_target_tpr - 1e-12:
            continue
        if (
            best is None
            or fpr < best[1]
            or (abs(fpr - best[1]) <= 1e-12 and threshold > best[0])
        ):
            best = (float(threshold), fpr, tpr)
    if best is None:
        raise RuntimeError("effective validation target TPR is unreachable")
    metrics = evaluate_threshold(
        rows, threshold=best[0], minimum_lead_ms=minimum_lead_ms,
    )
    return {
        "threshold": best[0],
        "target_tpr": float(target_tpr),
        "effective_target_tpr": effective_target_tpr,
        "maximum_structural_tpr": maximum_structural_tpr,
        "nominal_target_reached": nominal_target_reached,
        "selection_rule": (
            "VAL160 nominal target TPR, capped only by the no-context structural "
            "ceiling; then minimum FPR and highest threshold"
        ),
        "metrics": metrics,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")


def run(
    *, seed: int, output: Path, data_root: Path, source_epochs: int = 96,
    batch_size: int = 128, target_tpr: float = 0.90,
    minimum_lead_ms: float = 10.0,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"source baseline output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    roles, raw = load_jtext_roles(Path(data_root).resolve())
    transformed = {
        name: GS.transform(rows, "base") for name, rows in raw.items()
    }
    center, scale = GS.fit_scale(transformed["train"])
    for rows in transformed.values():
        GS.apply_scale(rows, center, scale)

    sampler = GS.build_sampler(
        transformed["train"], None,
        GS.POS["ea125"]["jt"], GS.GRAY["gray0"]["jt"], seed,
        minimum_pos_ms=10.0, allow_left_padding=True,
        context_lengths=(32, 64, 96, 128),
    )
    channels = transformed["train"][0]["v"].shape[1]
    enc, head = GS.train_source(
        sampler, channels, epochs=source_epochs, bs=batch_size, seed=seed,
    )
    validation_rows = predict_role(enc, head, transformed["val"])
    selected = select_validation_threshold(
        validation_rows, target_tpr=target_tpr,
        minimum_lead_ms=minimum_lead_ms,
    )
    test_rows = predict_role(enc, head, transformed["test"])
    test_metrics = evaluate_threshold(
        test_rows, threshold=selected["threshold"],
        minimum_lead_ms=minimum_lead_ms,
    )

    checkpoint_path = output / "source_parent.pt"
    torch.save({
        "protocol": PROTOCOL,
        "seed": int(seed),
        "features": list(RH.REBUILD),
        "enc": enc.state_dict(),
        "head": head.state_dict(),
        "center": center,
        "scale": scale,
        "threshold": float(selected["threshold"]),
        "minimum_lead_ms": float(minimum_lead_ms),
    }, checkpoint_path)
    np.savez_compressed(output / "scaler.npz", center=center, scale=scale)
    _write_jsonl(output / "validation_predictions.jsonl", selected["metrics"]["shots"])
    _write_jsonl(output / "test_predictions.jsonl", test_metrics["shots"])

    split_hashes = {
        name: _sha256(ROOT / "splits" / f"{name}.txt") for name in roles
    }
    summary = {
        "protocol": PROTOCOL,
        "status": "SOURCE_TEST220",
        "seed": int(seed),
        "device": GS.DEV,
        "data_root": str(Path(data_root).resolve()),
        "label_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
        "features": list(RH.REBUILD),
        "feature_count": int(channels),
        "training": {
            "source_epochs": int(source_epochs),
            "batches_per_epoch": 8,
            "batch_size": int(batch_size),
            "positive_horizon_ms": 125.0,
            "minimum_positive_lead_ms": 10.0,
            "gray_ms": 0.0,
            "left_padding": "repeat earliest observed causal value",
            "context_augmentation_points": [32, 64, 96, 128],
            "scaler_fit_role": "TRAIN1353 only",
        },
        "splits": {
            name: {
                "shots": len(roles[name]),
                "disruptive": EXPECTED_DISRUPTIVE[name],
                "safe": len(roles[name]) - EXPECTED_DISRUPTIVE[name],
                "sha256": split_hashes[name],
            }
            for name in roles
        },
        "threshold_selection": {
            key: value for key, value in selected.items() if key != "metrics"
        },
        "validation": {
            key: value for key, value in selected["metrics"].items()
            if key != "shots"
        },
        "test": {
            key: value for key, value in test_metrics.items() if key != "shots"
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "east_shots_read": 0,
            "test_used_for_training_or_threshold": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")),
    )
    parser.add_argument("--source-epochs", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-tpr", type=float, default=0.90)
    parser.add_argument("--minimum-lead-ms", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_float32_matmul_precision("high")
    summary = run(
        seed=args.seed, output=args.output, data_root=args.data_root,
        source_epochs=args.source_epochs, batch_size=args.batch_size,
        target_tpr=args.target_tpr, minimum_lead_ms=args.minimum_lead_ms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
