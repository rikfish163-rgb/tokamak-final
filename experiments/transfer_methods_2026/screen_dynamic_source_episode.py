#!/usr/bin/env python3
"""Dynamic-source development screen for causal multiscale EAST alarm episodes.

The script deliberately replays the already sealed v2.1 risk traces.  At each
shot it may train only on labels selected by the original chronological query
ledger.  Full HDF5 outcomes are consumed after the candidate decision solely
to score the development stream.  A successful screen therefore still needs
a fresh end-to-end online replay; it is not independent test evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import online_episode_value_v3 as episode_value  # noqa: E402
import run_two_timescale_online as two_runner  # noqa: E402
import run_unified_online_active as unified  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(bounded / (1.0 - bounded))


def _window(matrix: np.ndarray, index: int, width: int) -> np.ndarray:
    lower = max(0, int(index) - int(width) + 1)
    return np.asarray(matrix[lower:int(index) + 1], np.float64)


def sequence_vector(
    matrix: np.ndarray, *, index: int, candidate_mask: np.ndarray,
    representation: str,
) -> np.ndarray:
    """Summarize only the causal prefix ending at ``index``."""
    values = np.asarray(matrix, np.float64)
    candidate = np.asarray(candidate_mask, bool).reshape(-1)
    index = int(index)
    if (
        values.ndim != 2 or values.shape[1] != len(unified.EPISODE_FEATURE_NAMES)
        or len(values) != len(candidate) or index < 0 or index >= len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid causal sequence representation input")
    current = values[index]
    if representation == "endpoint20":
        return current.copy()

    recent = _window(values, index, 8)
    medium = _window(values, index, 32)
    mean8 = recent.mean(axis=0)
    mean32 = medium.mean(axis=0)
    std32 = medium.std(axis=0)
    if representation == "multiscale80":
        return np.concatenate((current, mean8, mean32, std32))
    if representation == "delta60":
        return np.concatenate((current, mean8 - mean32, std32))

    if representation != "mechanism52":
        raise ValueError(f"unknown sequence representation {representation}")
    # Use temporal derivatives only for continuous mechanism channels.  Route
    # and branch flags remain in the current endpoint block and are not
    # duplicated as pseudo-dynamics.
    temporal_indices = np.asarray([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        14, 15, 16, 17, 18, 19,
    ], dtype=np.int64)
    selected_delta = (mean8 - mean32)[temporal_indices]
    selected_std = std32[temporal_indices]
    candidate_recent = candidate[max(0, index - 63):index + 1]
    persistence = float(candidate_recent.mean())
    run = 0
    for active in candidate_recent[::-1]:
        if not bool(active):
            break
        run += 1
    scalar = np.asarray([
        persistence,
        min(run / 64.0, 1.0),
        min((index + 1) / 4096.0, 1.0),
        float(np.mean(recent[:, 0] * recent[:, 1])),
    ], np.float64)
    return np.concatenate((current, selected_delta, selected_std, scalar))


@dataclass
class ShotCache:
    shot: int
    selected_query: bool
    positions: np.ndarray
    vectors: dict[str, np.ndarray]
    high_route: np.ndarray
    high_excess: np.ndarray
    physics: np.ndarray
    outcome: dict[str, Any]
    start: float
    rate: float


def load_seed_cache(
    seed: int, run_dir: Path, *, median8: np.ndarray, scale8: np.ndarray,
    representations: tuple[str, ...],
) -> tuple[list[ShotCache], dict[str, Any]]:
    parent_manifest = run_dir.parent / "east-parent" / "parent_manifest.json"
    parent_contract = json.loads(parent_manifest.read_text())
    threshold = float(parent_contract["east_threshold"])
    threshold_logit = math.log(threshold / (1.0 - threshold))
    predictions = unified.load_jsonl(run_dir / "predictions.jsonl")
    query_rows = unified.load_jsonl(run_dir / "query_decisions.jsonl")
    queried = {int(row["shot"]): bool(row["selected"]) for row in query_rows}
    if [int(row["shot"]) for row in predictions] != list(queried):
        raise RuntimeError(f"seed {seed}: prediction/query rosters differ")

    cache: list[ShotCache] = []
    for row in predictions:
        shot_id = int(row["shot"])
        with np.load(run_dir / "risk_traces" / f"{shot_id}.npz") as payload:
            online_logit = _logit(payload["online_risk"])
            static_logit = _logit(payload["static_risk"])
            shared_logit = _logit(payload["online_shared_risk"])
            endpoint_residual = np.asarray(
                payload["endpoint_memory_logit_residual"], np.float64,
            )
            physics = np.asarray(payload["physics_support"], np.float64)
        trace = unified.supported_fusion_trace(
            online_logit=online_logit, static_logit=static_logit,
            physics_support=physics, threshold_logit=threshold_logit,
            alarm_parameters=row["parameters"],
            novel_early_min_physics_support=0.75,
            novel_early_min_trend=0.40,
        )
        raw, available = unified.causal_raw8_signal_only(shot_id)
        raw, available = unified.scale_raw8(raw, available, median8, scale8)
        matrix = unified.causal_episode_feature_matrix(
            online_logit=online_logit, static_logit=static_logit,
            shared_logit=shared_logit,
            endpoint_memory_residual=endpoint_residual,
            physics_support=physics, alarm_trace=trace["actual"],
            threshold_logit=threshold_logit,
            alarm_parameters=row["parameters"], raw=raw,
            raw_available=available,
            model_branch_mask=trace["model_branch"]["alarm_mask"],
            alarm_parameter_branch_mask=(
                trace["alarm_parameter_branch"]["alarm_mask"]
            ),
        )
        positions = np.flatnonzero(trace["supported_mask"])
        vectors = {
            name: np.stack([
                sequence_vector(
                    matrix, index=int(index),
                    candidate_mask=trace["supported_mask"],
                    representation=name,
                )
                for index in positions
            ]) if len(positions) else np.empty(
                (0, len(sequence_vector(
                    matrix, index=0,
                    candidate_mask=trace["supported_mask"],
                    representation=name,
                )))
            )
            for name in representations
        }
        actual = trace["actual"]
        high_excess = (
            online_logit - threshold_logit
            - float(row["parameters"]["high_threshold"])
        )
        signal = two_runner.load_east_signal_only(
            shot_id, np.zeros(17, np.float32), np.ones(17, np.float32),
        )
        cache.append(ShotCache(
            shot=shot_id, selected_query=queried[shot_id],
            positions=positions, vectors=vectors,
            high_route=np.asarray(actual["high_mask"], bool)[positions],
            high_excess=high_excess[positions], physics=physics[positions],
            outcome=two_runner.read_hdf5_outcome(shot_id),
            start=float(signal["start"]), rate=float(signal["rate"]),
        ))
    summary = json.loads((run_dir / "summary.json").read_text())
    return cache, summary["results"]["online_unified"]


class CausalCalibratedClassifier:
    """Queried-only classifier with a past-label protection boundary."""

    def __init__(
        self, *, feature_dim: int, l2: float, minimum_history: int = 12,
        minimum_per_class: int = 3,
    ) -> None:
        self.model = episode_value._RegularizedOnlineLogistic(
            feature_dim=feature_dim, minimum_history=minimum_history,
            minimum_per_class=minimum_per_class, l2=l2, capacity=160,
        )

    def probabilities(self, matrix: np.ndarray) -> tuple[np.ndarray, float]:
        values = np.asarray(matrix, np.float64)
        confidence = self.model.confidence()
        positives = sum(
            int(item["target"] == 1) for item in self.model._history
        )
        prior = (positives + 1.0) / (len(self.model._history) + 2.0)
        if not len(values):
            return np.empty(0, np.float64), confidence
        if not self.model._fitted:
            return np.full(len(values), prior, np.float64), confidence
        standardized = np.clip(
            (values - self.model._center) / self.model._scale, -8.0, 8.0,
        )
        fitted = 1.0 / (
            1.0 + np.exp(-np.clip(
                standardized @ self.model._weight + self.model._bias,
                -30.0, 30.0,
            ))
        )
        return (
            (1.0 - confidence) * prior + confidence * fitted
        ), confidence

    def historical_scores(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.model._history:
            return np.empty(0), np.empty(0)
        matrix = np.stack([
            item["features"] for item in self.model._history
        ])
        scores, _confidence = self.probabilities(matrix)
        target = np.asarray([
            item["target"] for item in self.model._history
        ], np.int64)
        return scores[target == 1], scores[target == 0]

    def threshold(
        self, *, mode: str, fixed: float, safety_margin: float,
        negative_quantile: float,
    ) -> float:
        if mode == "fixed":
            return float(fixed)
        safe, disruptive = self.historical_scores()
        if len(safe) < 3 or len(disruptive) < 3:
            return 1.01
        protected = float(np.quantile(disruptive, negative_quantile))
        safe_support = float(np.quantile(safe, 0.25))
        candidate = max(0.50, protected + float(safety_margin))
        # Fail open when past selected classes do not provide a separating
        # interval.  This prevents threshold pressure from fabricating a veto.
        return candidate if candidate < safe_support else 1.01

    def update(self, *, shot: int, vector: np.ndarray, safe: bool) -> None:
        self.model.update(shot_id=shot, features=vector, target=int(safe))


@dataclass
class ScoredShot:
    shot: int
    positions: np.ndarray
    probability: np.ndarray
    confidence: float
    safe_q25: float | None
    disruptive_q90: float | None
    disruptive_max: float | None
    high_route: np.ndarray
    high_excess: np.ndarray
    physics: np.ndarray
    outcome: dict[str, Any]
    start: float
    rate: float


def score_trajectory(
    cache: list[ShotCache], *, representation: str, l2: float,
    control: str = "real",
) -> list[ScoredShot]:
    dimension = next(
        row.vectors[representation].shape[1]
        for row in cache if len(row.positions)
    )
    classifier = CausalCalibratedClassifier(feature_dim=dimension, l2=l2)
    scored: list[ScoredShot] = []
    previous_selected_label: bool | None = None
    previous_selected_vector: np.ndarray | None = None
    for row in cache:
        values = row.vectors[representation].copy()
        if control == "temporal_zero":
            values[:, 20:] = 0.0
        elif control == "raw8_zero":
            raw_indices = np.asarray([14, 15, 16, 17, 18, 19])
            for offset in range(0, values.shape[1], 20):
                indices = offset + raw_indices
                indices = indices[indices < values.shape[1]]
                values[:, indices] = 0.0
        elif control not in {
            "real", "no_labels", "label_lag", "label_flip", "feature_lag",
        }:
            raise ValueError(f"unknown sequence control {control}")
        probability, confidence = classifier.probabilities(values)
        historical_safe, historical_disruptive = (
            classifier.historical_scores()
        )
        enough = (
            len(historical_safe) >= 3 and len(historical_disruptive) >= 3
        )
        scored.append(ScoredShot(
            shot=row.shot, positions=row.positions,
            probability=probability, confidence=confidence,
            safe_q25=(
                float(np.quantile(historical_safe, 0.25)) if enough else None
            ),
            disruptive_q90=(
                float(np.quantile(historical_disruptive, 0.90))
                if enough else None
            ),
            disruptive_max=(
                float(np.max(historical_disruptive)) if enough else None
            ),
            high_route=row.high_route, high_excess=row.high_excess,
            physics=row.physics, outcome=row.outcome,
            start=row.start, rate=row.rate,
        ))
        if row.selected_query and len(row.positions):
            safe = not bool(row.outcome["is_disrupt"])
            if control == "real" or control in {"temporal_zero", "raw8_zero"}:
                classifier.update(shot=row.shot, vector=values[0], safe=safe)
            elif control == "label_flip":
                classifier.update(shot=row.shot, vector=values[0], safe=not safe)
            elif control == "label_lag" and previous_selected_label is not None:
                classifier.update(
                    shot=row.shot, vector=values[0],
                    safe=previous_selected_label,
                )
            elif control == "feature_lag" and previous_selected_vector is not None:
                classifier.update(
                    shot=row.shot, vector=previous_selected_vector,
                    safe=safe,
                )
            previous_selected_label = safe
            previous_selected_vector = values[0].copy()
    return scored


def evaluate_candidate(
    trajectory: list[ScoredShot], *, threshold_mode: str,
    fixed_threshold: float, safety_margin: float,
    negative_quantile: float, high_escape_margin: float,
    physics_escape: float,
) -> dict[str, Any]:
    disruptive = safe = tp = fp = timely = 0
    blocked_disruptive = blocked_safe = delayed = 0
    leads: list[float] = []
    threshold_history: list[float] = []
    blocked_safe_shots: list[int] = []
    blocked_disruptive_shots: list[int] = []
    delayed_shots: list[dict[str, Any]] = []
    for row in trajectory:
        alarm_index = None
        if len(row.positions):
            if threshold_mode == "fixed":
                threshold = float(fixed_threshold)
            elif (
                row.safe_q25 is None or row.disruptive_q90 is None
                or row.disruptive_max is None
            ):
                threshold = 1.01
            else:
                protected = (
                    row.disruptive_q90
                    if math.isclose(float(negative_quantile), 0.90)
                    else row.disruptive_max
                )
                candidate = max(0.50, protected + float(safety_margin))
                threshold = candidate if candidate < row.safe_q25 else 1.01
            threshold_history.append(threshold)
            escape = (
                (row.high_route & (
                    row.high_excess >= float(high_escape_margin)
                ))
                | (row.physics >= float(physics_escape))
            )
            passes = (
                (row.confidence < 0.75)
                | (row.probability < threshold) | escape
            )
            selected = np.flatnonzero(passes)
            if len(selected):
                alarm_index = int(row.positions[int(selected[0])])
                if alarm_index != int(row.positions[0]):
                    delayed += 1
                    delayed_shots.append({
                        "shot": int(row.shot),
                        "is_disrupt": bool(row.outcome["is_disrupt"]),
                        "base_alarm_index": int(row.positions[0]),
                        "alarm_index": int(alarm_index),
                    })
        outcome = row.outcome
        if bool(outcome["is_disrupt"]):
            disruptive += 1
            if alarm_index is None:
                blocked_disruptive += int(len(row.positions) > 0)
                if len(row.positions):
                    blocked_disruptive_shots.append(int(row.shot))
            else:
                tp += 1
                alarm_time = row.start + (
                    unified.alarm.W - 1 + alarm_index
                ) / row.rate
                lead = (
                    float(outcome["down_time_s"]) - alarm_time
                ) * 1000.0
                leads.append(lead)
                timely += int(lead >= 30.0 - 1e-9)
        else:
            safe += 1
            fp += int(alarm_index is not None)
            blocked_safe += int(len(row.positions) > 0 and alarm_index is None)
            if len(row.positions) > 0 and alarm_index is None:
                blocked_safe_shots.append(int(row.shot))
    finite_thresholds = np.asarray([
        value for value in threshold_history if value <= 1.0
    ], np.float64)
    return {
        "TP": tp, "FP": fp, "FN": disruptive - tp, "TN": safe - fp,
        "tpr": tp / max(disruptive, 1), "fpr": fp / max(safe, 1),
        "tpr_at_30ms": timely / max(disruptive, 1),
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "blocked_safe": blocked_safe,
        "blocked_disruptive": blocked_disruptive,
        "delayed": delayed,
        "blocked_safe_shots": blocked_safe_shots,
        "blocked_disruptive_shots": blocked_disruptive_shots,
        "delayed_shots": delayed_shots,
        "calibrated_threshold_count": int(len(finite_thresholds)),
        "calibrated_threshold_median": (
            float(np.median(finite_thresholds))
            if len(finite_thresholds) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("/home/hetaisheng/Fuison-runs/corrected_mainline_screen"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    representations = (
        "endpoint20", "delta60", "mechanism52", "multiscale80",
    )
    parent = torch.load(
        args.runs_root / "seed131/east-parent/east_fixed_parent.pt",
        map_location="cpu", weights_only=True,
    )
    raw8_scaler = parent["raw8_scaler"]
    median8 = np.asarray(raw8_scaler["center"], np.float64)
    scale8 = np.asarray(raw8_scaler["scale"], np.float64)
    caches = {}
    baselines = {}
    hashes = {}
    for seed in (131, 133, 137):
        run = args.runs_root / f"seed{seed}" / "east-online"
        caches[seed], baselines[seed] = load_seed_cache(
            seed, run, median8=median8, scale8=scale8,
            representations=representations,
        )
        for name in ("predictions.jsonl", "query_decisions.jsonl", "summary.json"):
            hashes[f"seed{seed}/{name}"] = sha256_file(run / name)
        print(f"[SEQUENCE EPISODE] cached seed={seed}", flush=True)

    trajectories = {}
    for representation in representations:
        for l2 in (0.10, 0.30, 1.00):
            for seed in (131, 133, 137):
                trajectories[(seed, representation, l2)] = score_trajectory(
                    caches[seed], representation=representation, l2=l2,
                )
            print(
                f"[SEQUENCE EPISODE] scored {representation} l2={l2}",
                flush=True,
            )

    candidates = []
    grid = []
    for representation in representations:
        for l2 in (0.10, 0.30, 1.00):
            for fixed in (0.65, 0.75, 0.85):
                grid.append((representation, l2, "fixed", fixed, 0.0, 1.0))
            for margin in (0.00, 0.03, 0.06):
                for quantile in (0.90, 1.00):
                    grid.append((
                        representation, l2, "protected", 0.85,
                        margin, quantile,
                    ))
    for (
        representation, l2, mode, fixed, margin, quantile,
    ) in grid:
        for high_escape in (0.75, 1.50, 99.0):
            for physics_escape in (0.85, 0.95, 1.01):
                results = {
                    seed: evaluate_candidate(
                        trajectories[(seed, representation, l2)],
                        threshold_mode=mode, fixed_threshold=fixed,
                        safety_margin=margin, negative_quantile=quantile,
                        high_escape_margin=high_escape,
                        physics_escape=physics_escape,
                    )
                    for seed in (131, 133, 137)
                }
                nondegrading = all(
                    results[seed]["TP"] >= baselines[seed]["TP"]
                    and results[seed]["tpr_at_30ms"]
                    >= baselines[seed]["tpr_at_30ms"] - 1e-12
                    for seed in results
                )
                strict = nondegrading and all(
                    results[seed]["FP"] < baselines[seed]["FP"]
                    for seed in results
                )
                candidates.append({
                    "parameters": {
                        "representation": representation, "l2": l2,
                        "threshold_mode": mode,
                        "fixed_threshold": fixed,
                        "safety_margin": margin,
                        "negative_quantile": quantile,
                        "high_escape_margin": high_escape,
                        "physics_escape": physics_escape,
                    },
                    "strict_all_seed_improvement": strict,
                    "all_seed_non_degradation": nondegrading,
                    "total_fp_reduction": int(sum(
                        baselines[seed]["FP"] - results[seed]["FP"]
                        for seed in results
                    )),
                    "total_tp_loss": int(sum(
                        baselines[seed]["TP"] - results[seed]["TP"]
                        for seed in results
                    )),
                    "total_tpr30_count_loss": int(round(sum(
                        180.0 * (
                            baselines[seed]["tpr_at_30ms"]
                            - results[seed]["tpr_at_30ms"]
                        ) for seed in results
                    ))),
                    "results": {str(seed): results[seed] for seed in results},
                })
    ranked = sorted(candidates, key=lambda row: (
        not row["strict_all_seed_improvement"],
        not row["all_seed_non_degradation"],
        row["total_tp_loss"], row["total_tpr30_count_loss"],
        -row["total_fp_reduction"],
        row["parameters"]["representation"],
    ))
    selected_parameters = dict(ranked[0]["parameters"])
    negative_controls = {}
    for control in (
        "no_labels", "label_lag", "label_flip", "feature_lag",
        "temporal_zero", "raw8_zero",
    ):
        control_results = {}
        for seed in (131, 133, 137):
            trajectory = score_trajectory(
                caches[seed],
                representation=str(selected_parameters["representation"]),
                l2=float(selected_parameters["l2"]),
                control=control,
            )
            control_results[str(seed)] = evaluate_candidate(
                trajectory,
                threshold_mode=str(selected_parameters["threshold_mode"]),
                fixed_threshold=float(selected_parameters["fixed_threshold"]),
                safety_margin=float(selected_parameters["safety_margin"]),
                negative_quantile=float(selected_parameters["negative_quantile"]),
                high_escape_margin=float(selected_parameters["high_escape_margin"]),
                physics_escape=float(selected_parameters["physics_escape"]),
            )
        negative_controls[control] = {
            "results": control_results,
            "total_fp_reduction": int(sum(
                baselines[seed]["FP"] - control_results[str(seed)]["FP"]
                for seed in (131, 133, 137)
            )),
            "total_tp_loss": int(sum(
                baselines[seed]["TP"] - control_results[str(seed)]["TP"]
                for seed in (131, 133, 137)
            )),
        }
    payload = {
        "status": "TEST_DRIVEN_DIAGNOSTIC",
        "method": (
            "dynamic-JTEXT-source queried-only causal multiscale episode "
            "and protected calibration screen"
        ),
        "claim_boundary": (
            "balanced360 selects this dynamic-source candidate; every decision is "
            "causal but these metrics are not independent future-stream evidence "
            "and cannot replace a full gradient replay"
        ),
        "baselines": baselines, "input_hashes": hashes,
        "candidate_count": len(candidates),
        "strict_solution_count": sum(
            int(row["strict_all_seed_improvement"]) for row in candidates
        ),
        "nondegrading_count": sum(
            int(row["all_seed_non_degradation"]) for row in candidates
        ),
        "selected_balanced_candidate": selected_parameters,
        "negative_controls": negative_controls,
        "ranked_candidates": ranked[:100],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "strict_solution_count": payload["strict_solution_count"],
        "nondegrading_count": payload["nondegrading_count"],
        "best": ranked[0],
    }, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
