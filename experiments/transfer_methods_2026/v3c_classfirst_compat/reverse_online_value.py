#!/usr/bin/env python3
"""Retrospective value-policy design for the ordered EAST 360-shot stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
FEATURES = (
    "fp_contradiction", "fn_contradiction", "uncertainty",
    "aux_novelty", "teacher_disagreement", "risk_spikiness",
)


def robust_transform(
    rows: Sequence[Mapping[str, object]], feature_names: Sequence[str]
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    names = tuple(feature_names)
    values = np.asarray([[float(row[name]) for name in names] for row in rows], float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("reverse value features must be finite and non-empty")
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    transformed = np.clip((values - center) / scale, -5.0, 5.0)
    contract = {
        name: {"center": float(center[index]), "scale": float(scale[index])}
        for index, name in enumerate(names)
    }
    return transformed, contract


def apply_feature_contract(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str],
    contract: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    output = []
    for row in rows:
        output.append([
            np.clip(
                (float(row[name]) - float(contract[name]["center"]))
                / max(float(contract[name]["scale"]), 1e-6),
                -5.0, 5.0,
            )
            for name in feature_names
        ])
    return np.asarray(output, dtype=np.float64)


def score_value_row(
    row: Mapping[str, object], config: Mapping[str, object]
) -> tuple[float, dict[str, float]]:
    names = tuple(config["feature_names"])
    contract = config["feature_contract"]
    weights = config["reference_weights"]
    transformed = {
        name: float(np.clip(
            (float(row[name]) - float(contract[name]["center"]))
            / max(float(contract[name]["scale"]), 1e-6),
            -5.0, 5.0,
        ))
        for name in names
    }
    score = sum(float(weights[name]) * transformed[name] for name in names)
    return float(score), transformed


def runtime_value_features(
    *, query_components: Mapping[str, object],
    auxiliary_row: Mapping[str, object], risk: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(risk, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("risk trace must be finite and non-empty")
    top = np.sort(values)[-min(5, len(values)):]
    static_error = float(auxiliary_row["static_error"])
    online_error = float(auxiliary_row["online_error"])
    return {
        "fp_contradiction": float(query_components["fp_contradiction"]),
        "fn_contradiction": float(query_components["fn_contradiction"]),
        "uncertainty": float(query_components["uncertainty"]),
        "aux_novelty": float(auxiliary_row["aux_novelty"]),
        "teacher_disagreement": float(
            abs(static_error - online_error) / max(static_error, 1e-6)
        ),
        "risk_spikiness": float(
            max(float(values.max()) - float(top.mean()), 0.0)
            / max(float(values.max()), 1e-6)
        ),
    }


def causal_block_select(
    scores: np.ndarray, *, cutoff: float, block_size: int = 20, quota: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if block_size <= 0 or quota < 0 or quota > block_size:
        raise ValueError("invalid causal block budget")
    selected = np.zeros(len(scores), dtype=bool)
    forced = np.zeros(len(scores), dtype=bool)
    used = 0
    for index, score in enumerate(scores):
        position = index % block_size
        if position == 0:
            used = 0
        remaining = block_size - position
        slots = quota - used
        is_forced = slots > 0 and remaining <= slots
        choose = slots > 0 and (float(score) >= float(cutoff) or is_forced)
        if choose:
            selected[index] = True
            forced[index] = is_forced and float(score) < float(cutoff)
            used += 1
    return selected, forced


def _policy_metrics(
    rows: Sequence[Mapping[str, object]], selected: np.ndarray, *, fold_size: int
) -> dict[str, float]:
    error = np.asarray([bool(row["is_error"]) for row in rows])
    fp = np.asarray([bool(row["is_fp"]) for row in rows])
    fn = np.asarray([bool(row["is_fn"]) for row in rows])
    fold_enrichment = []
    for left in range(0, len(rows), int(fold_size)):
        right = min(left + int(fold_size), len(rows))
        stream_rate = float(np.mean(error[left:right]))
        queried = selected[left:right]
        query_rate = float(np.mean(error[left:right][queried])) if queried.any() else 0.0
        fold_enrichment.append(query_rate / stream_rate if stream_rate else 1.0)
    query_error_rate = float(np.mean(error[selected])) if selected.any() else 0.0
    stream_error_rate = float(np.mean(error))
    return {
        "worst_fold_error_enrichment": float(min(fold_enrichment)),
        "mean_fold_error_enrichment": float(np.mean(fold_enrichment)),
        "overall_error_enrichment": (
            query_error_rate / stream_error_rate if stream_error_rate else 1.0
        ),
        "fp_capture": float(np.sum(selected & fp) / max(np.sum(fp), 1)),
        "fn_capture": float(np.sum(selected & fn) / max(np.sum(fn), 1)),
        "query_error_precision": query_error_rate,
    }


def fit_value_policy(
    rows: Sequence[Mapping[str, object]],
    *,
    feature_names: Sequence[str],
    candidate_weights: np.ndarray,
    cutoff_quantiles: Sequence[float],
    fold_size: int = 60,
) -> dict[str, object]:
    names = tuple(feature_names)
    values, contract = robust_transform(rows, names)
    candidates = []
    for raw_weight in np.asarray(candidate_weights, dtype=np.float64):
        if raw_weight.shape != (len(names),) or np.any(raw_weight < 0):
            raise ValueError("candidate weights must be non-negative aligned vectors")
        total = float(raw_weight.sum())
        if total <= 0:
            continue
        weight = raw_weight / total
        scores = values @ weight
        for quantile in cutoff_quantiles:
            if not 0 < float(quantile) < 1:
                raise ValueError("cutoff quantiles must lie in (0,1)")
            cutoff = float(np.quantile(scores, float(quantile)))
            selected, forced = causal_block_select(scores, cutoff=cutoff)
            metrics = _policy_metrics(rows, selected, fold_size=fold_size)
            candidates.append({
                "weights": weight,
                "cutoff": cutoff,
                "cutoff_quantile": float(quantile),
                "query_count": int(selected.sum()),
                "forced_fraction": float(np.mean(forced[selected])) if selected.any() else 0.0,
                **metrics,
            })
    if not candidates:
        raise ValueError("no valid value-policy candidates")
    candidates.sort(key=lambda row: (
        row["worst_fold_error_enrichment"],
        min(row["fp_capture"], row["fn_capture"]),
        row["overall_error_enrichment"],
        -row["forced_fraction"],
    ), reverse=True)
    best = candidates[0]
    admitted = [
        row for row in candidates
        if row["worst_fold_error_enrichment"] >= 0.95 * best["worst_fold_error_enrichment"]
        and row["overall_error_enrichment"] >= 0.95 * best["overall_error_enrichment"]
    ][:32]
    weights = np.stack([row["weights"] for row in admitted])
    return {
        "reference_weights": {
            name: float(best["weights"][index]) for index, name in enumerate(names)
        },
        "weight_ranges": {
            name: [float(weights[:, index].min()), float(weights[:, index].max())]
            for index, name in enumerate(names)
        },
        "cutoff": float(best["cutoff"]),
        "cutoff_quantile": float(best["cutoff_quantile"]),
        "cutoff_quantile_range": [
            float(min(row["cutoff_quantile"] for row in admitted)),
            float(max(row["cutoff_quantile"] for row in admitted)),
        ],
        "query_count": int(best["query_count"]),
        "forced_fraction": float(best["forced_fraction"]),
        "worst_fold_error_enrichment": float(best["worst_fold_error_enrichment"]),
        "mean_fold_error_enrichment": float(best["mean_fold_error_enrichment"]),
        "overall_error_enrichment": float(best["overall_error_enrichment"]),
        "query_error_precision": float(best["query_error_precision"]),
        "fp_capture": float(best["fp_capture"]),
        "fn_capture": float(best["fn_capture"]),
        "feature_contract": contract,
        "admitted_candidate_count": len(admitted),
    }


def build_frozen_config(
    *, fit: Mapping[str, object], feature_contract: Mapping[str, object],
    physics_ranges: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "east-online-retrospective-value-v1",
        "claim_boundary": "RETROSPECTIVE_DESIGNED_PREQUENTIAL_360",
        "feature_names": list(fit["reference_weights"]),
        "reference_weights": dict(fit["reference_weights"]),
        "weight_ranges": dict(fit["weight_ranges"]),
        "feature_contract": dict(feature_contract),
        "cutoff": float(fit["cutoff"]),
        "cutoff_quantile": float(fit["cutoff_quantile"]),
        "physics_parameter_ranges": dict(physics_ranges),
        "online_budget": {"block_size": 20, "quota": 4, "total": 72},
        "forbidden_runtime_inputs": [
            "identifier_feature", "precomputed_query_roster",
            "current_outcome_before_seal", "future_outcomes",
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _outcome(shot: int) -> tuple[bool, float | None]:
    path = DATA / "EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    with h5py.File(path, "r") as handle:
        disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    return disrupt, float(down[0]) if disrupt and down.size else None


def _physics_ranges(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups = {name: [row for row in rows if bool(row[f"is_{name.lower()}"])] for name in ("FP", "FN")}
    tp = [row for row in rows if row["group"] == "TP"]
    tn = [row for row in rows if row["group"] == "TN"]
    def q(items, key, values=(0.25, 0.5, 0.75)):
        data = np.asarray([float(row[key]) for row in items], float)
        return [float(value) for value in np.quantile(data, values)] if len(data) else []
    return {
        "activity_threshold": [0.35, 0.65],
        "minimum_known_families": [2, 3],
        "physics_at_risk": {
            "FP_q25_q50_q75": q(groups["FP"], "physics_at_risk"),
            "TP_q25_q50_q75": q(tp, "physics_at_risk"),
        },
        "physics_max": {
            "FN_q25_q50_q75": q(groups["FN"], "physics_max"),
            "TN_q25_q50_q75": q(tn, "physics_max"),
        },
        "constraint_weights": [0.05, 0.10, 0.20],
        "unknown_policy": "disable physics loss when fewer than minimum_known_families are available",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=HERE / "reverse_online_value_seed131")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)

    a1_dir = HERE / f"online_active_a1_seed{args.seed}"
    a3_dir = HERE / f"online_active_a3_seed{args.seed}_intrareset"
    static_dir = HERE / f"two_timescale_online_seed{args.seed}_v10_opposite_bag"
    a1_rows = _jsonl(a1_dir / "seals.jsonl")
    a3_rows = {int(row["shot"]): row for row in _jsonl(a3_dir / "shot_seals.jsonl")}
    static_rows = {
        int(row["shot"]): row
        for row in _jsonl(static_dir / "predictions_static.jsonl")
    }
    rows = []
    for a1 in a1_rows:
        shot = int(a1["shot"])
        a3 = a3_rows[shot]
        risk = np.load(a1_dir / "risk_traces" / f"{shot}.npz")["risk"].astype(float)
        disrupt, down_time = _outcome(shot)
        static = static_rows[shot]
        alarm = bool(float(static["score"]) >= float(static["threshold"]))
        group = "TP" if disrupt and alarm else "FN" if disrupt else "FP" if alarm else "TN"
        value_features = runtime_value_features(
            query_components=a1, auxiliary_row=a3, risk=risk
        )
        rows.append({
            "stream_index": int(a1["stream_index"]), "shot": shot,
            "is_disrupt": disrupt, "down_time_s": down_time,
            "base_alarm": alarm, "group": group,
            "is_error": group in ("FP", "FN"), "is_fp": group == "FP", "is_fn": group == "FN",
            **value_features,
            "base_max_risk": float(static["score"]),
            "base_threshold": float(static["threshold"]),
            "persistence": float(a1["persistence"]),
            "physics_at_risk": float(a1["physics_at_risk"]),
            "physics_max": float(a1["physics_max"]),
            "physics_known_fraction": float(a1["physics_known_fraction"]),
        })
    if [row["stream_index"] for row in rows] != list(range(360)):
        raise RuntimeError("reverse table is not the complete ordered 360-shot stream")

    rng = np.random.default_rng(args.seed + 26000)
    weights = np.concatenate([
        np.eye(len(FEATURES)),
        np.ones((1, len(FEATURES))),
        rng.dirichlet(np.ones(len(FEATURES)), size=int(args.samples)),
    ])
    fit = fit_value_policy(
        rows, feature_names=FEATURES, candidate_weights=weights,
        cutoff_quantiles=(0.65, 0.75, 0.80, 0.85, 0.90), fold_size=60,
    )
    physics = _physics_ranges(rows)
    config = build_frozen_config(
        fit=fit, feature_contract=fit["feature_contract"], physics_ranges=physics
    )
    config["reverse_source"] = {
        "shots": 360, "seed": args.seed,
        "a1_seals_sha256": _sha256(a1_dir / "seals.jsonl"),
        "a3_seals_sha256": _sha256(a3_dir / "shot_seals.jsonl"),
        "static_predictions_sha256": _sha256(static_dir / "predictions_static.jsonl"),
    }
    table = args.out / "reverse_table.jsonl"
    table.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    config["reverse_table_sha256"] = _sha256(table)
    (args.out / "value_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.out / "fit_summary.json").write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: fit[key] for key in (
        "reference_weights", "weight_ranges", "cutoff_quantile",
        "query_error_precision", "overall_error_enrichment",
        "worst_fold_error_enrichment", "fp_capture", "fn_capture",
    )}, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "robust_transform", "apply_feature_contract", "causal_block_select",
    "score_value_row", "runtime_value_features", "fit_value_policy", "build_frozen_config",
]
