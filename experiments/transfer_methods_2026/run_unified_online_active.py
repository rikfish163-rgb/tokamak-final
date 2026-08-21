#!/usr/bin/env python3
"""Locked two-phase shared17 + EAST-private raw8 evaluation and online replay.

Phase 1 seals static predictions for all 360 shots.  Phase 2 starts from the
same migrated parent weights and follows endpoint prediction/update -> shot
seal -> query -> HDF5 reveal -> protected slow update.  The balanced360 stream
remains a method-development replay, not an untouched deployment test.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import dual_threshold_alarm as dual_alarm  # noqa: E402
import dynamic_hazard_adapter as dynamic  # noqa: E402
import east_private23_sidecar as private23  # noqa: E402
import east_private_continual as private_continual  # noqa: E402
import evaluate_dynamic_positive_gate as positive_gate  # noqa: E402
import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
import online_episode_value_v3 as episode_value  # noqa: E402
import online_timescale_ablation as timescale_ablation  # noqa: E402
import run_dynamic_hazard_adapter as dha  # noqa: E402
import run_east_private23_online as old_private  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_online_active_a3 as a3  # noqa: E402
import run_online_protected_residual as old_online  # noqa: E402
import run_two_timescale_online as two_runner  # noqa: E402
import shot_alarm as alarm  # noqa: E402
import two_timescale_online_adapter as two_scale  # noqa: E402
import unified_online_value as value_module  # noqa: E402


INITIAL_ALARM_PARAMETERS = {
    "high_threshold": 0.0,
    "physics_weight": 8.0,
    "trend_weight": 12.0,
    "early_threshold": 7.9,
}

LOCKED_BEST_V8 = {
    "private_support_epochs": 8,
    "adaptation_events": 64,
    "intra_steps": 1,
    "inter_steps": 2,
    "intra_lr": 1e-4,
    "private_intra_lr": 1e-4,
    "intra_physics_weight": 0.10,
    "inter_lr": 5e-4,
    "private_inter_lr": 2e-4,
    "alarm_lr": 0.02,
    "query_block_size": 20,
    "query_quota": 4,
    "query_history_size": 80,
    "query_quantile": 0.80,
    "query_uncertainty_weight": 0.65,
    "query_value_lr": 0.05,
    "novel_early_min_physics_support": 0.0,
}

FROZEN_PARAMETER_NAMES = tuple(LOCKED_BEST_V8)
FROZEN_V2_PARAMETER_NAMES = FROZEN_PARAMETER_NAMES + (
    "novel_early_min_trend",
    "max_shots",
    "causal_error_memory",
    "error_memory_radius",
    "error_memory_min_support",
    "error_memory_weight",
    "query_initial_cutoff",
    "query_token_capacity",
    "strict_window_active",
    "window_candidate_stride",
    "window_update_rate",
    "window_token_capacity",
    "window_history_size",
    "window_value_quantile",
    "window_initial_cutoff",
    "window_minimum_history",
    "window_minimum_gap",
    "typed_endpoint_memory",
    "endpoint_memory_radius",
    "endpoint_memory_min_shots",
    "endpoint_memory_positive",
    "endpoint_memory_negative",
    "endpoint_memory_physics_gate",
    "endpoint_memory_protection_ratio",
    "endpoint_memory_stride",
)


def resolve_parent_directories(
    *, seed: int, a1_dir: Path | None = None, adapter_dir: Path | None = None,
) -> tuple[Path, Path]:
    resolved_a1 = Path(
        a1_dir or HERE / f"online_active_a1_seed{int(seed)}"
    ).expanduser().resolve()
    resolved_adapter = Path(
        adapter_dir or HERE / f"dynamic_hazard_adapter_seed{int(seed)}"
    ).expanduser().resolve()
    return resolved_a1, resolved_adapter


def validate_sealed_parent_contract(
    payload: dict[str, Any], fidelity: dict[str, Any], *, seed: int,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless an archived support-only parent has exact lineage."""
    if payload.get("schema_version") != "equal-label-budget-frontload-v2":
        raise RuntimeError("sealed parent schema mismatch")
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError("sealed parent seed mismatch")
    if payload.get("arm") != "support_only":
        raise RuntimeError("sealed parent must come from the support_only fidelity arm")
    if payload.get("source_checkpoint_sha256") != str(source_checkpoint_sha256):
        raise RuntimeError("sealed parent source checkpoint mismatch")
    if not bool(fidelity.get("passed")):
        raise RuntimeError("sealed parent fidelity gate did not pass")
    expected = fidelity.get("expected")
    actual = fidelity.get("actual")
    required_hashes = ("encoder_head", "shared_adapter", "private_adapter")
    if (
        not isinstance(expected, dict) or not isinstance(actual, dict)
        or expected != actual
        or any(not isinstance(expected.get(name), str) for name in required_hashes)
    ):
        raise RuntimeError("sealed parent fidelity record is inconsistent")
    thresholds = payload.get("thresholds")
    threshold = (
        thresholds.get("frozen_parent_threshold")
        if isinstance(thresholds, dict) else None
    )
    if (
        threshold is None or not math.isfinite(float(threshold))
        or not 0.0 < float(threshold) < 1.0
        or not math.isclose(
            float(threshold), float(expected.get("threshold", float("nan"))),
            rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise RuntimeError("sealed parent fidelity threshold mismatch")
    for name in ("encoder", "head", "shared", "private"):
        if not isinstance(payload.get(name), dict) or not payload[name]:
            raise RuntimeError(f"sealed parent is missing {name} state")
    return {
        "threshold": float(threshold),
        "expected_base_sha256": expected["encoder_head"],
        "expected_shared_sha256": expected["shared_adapter"],
        "expected_private_sha256": expected["private_adapter"],
    }


def validate_locked_best_v8_arguments(args: argparse.Namespace) -> None:
    """Fail closed before any 360-shot artifact is created or read."""
    if not bool(args.locked_best_v8):
        return
    mismatches: list[str] = []
    if int(args.seed) not in (131, 133, 137):
        mismatches.append(f"seed={args.seed!r}")
    if int(args.max_shots) != 360:
        mismatches.append(f"max_shots={args.max_shots!r}")
    for name, expected in LOCKED_BEST_V8.items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = int(actual) == expected
        if not matches:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    if not math.isclose(
        float(args.novel_early_min_trend), 0.0, rel_tol=0.0, abs_tol=1e-12,
    ):
        mismatches.append(
            f"novel_early_min_trend={args.novel_early_min_trend!r} (expected 0.0)"
        )
    if mismatches:
        raise ValueError("locked best-v8 configuration mismatch: " + ", ".join(mismatches))


def validate_frozen_online_arguments(
    args: argparse.Namespace, contract: dict[str, Any],
) -> None:
    """Fail closed when a frozen architecture is invoked with parameter drift."""
    if int(args.seed) not in [int(value) for value in contract["allowed_seeds"]]:
        raise ValueError("frozen online configuration rejects seed")
    expected = contract["parameters"]
    if "locked_parameter_names" in contract:
        names = tuple(str(name) for name in contract["locked_parameter_names"])
        if names != FROZEN_V2_PARAMETER_NAMES:
            raise ValueError(
                "frozen online configuration has an incomplete v2 parameter lock"
            )
    else:
        # Backward-compatible validation for the already sealed v1 artifact.
        names = FROZEN_PARAMETER_NAMES
    mismatches = []
    for name in names:
        if name not in expected:
            mismatches.append(f"missing {name}")
            continue
        actual = getattr(args, name)
        wanted = expected[name]
        if isinstance(wanted, float):
            equal = math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = int(actual) == int(wanted)
        if not equal:
            mismatches.append(f"{name}={actual!r} expected {wanted!r}")
    if mismatches:
        raise ValueError("frozen online configuration mismatch: " + ", ".join(mismatches))


def validate_episode_v3_arguments(args: argparse.Namespace) -> None:
    """Fail closed around the explicitly test-driven v3 architecture."""
    isolate_feedback = bool(getattr(
        args, "episode_isolate_learning_feedback", False,
    ))
    if bool(getattr(args, "episode_protected_sequence", False)) and not bool(
        args.episode_value_v3
    ):
        raise ValueError("protected sequence requires --episode-value-v3")
    if isolate_feedback and not bool(args.episode_value_v3):
        raise ValueError("feedback isolation requires --episode-value-v3")
    if not bool(args.episode_value_v3):
        return
    if not bool(args.development_search):
        raise ValueError("episode v3 requires --development-search")
    if bool(args.locked_best_v8) or args.frozen_online_config is not None:
        raise ValueError("episode v3 cannot run in locked or frozen mode")
    if float(args.query_token_capacity) <= 0.0:
        raise ValueError("episode v3 requires a positive query token capacity")
    if not 0.5 <= float(args.episode_fp_threshold) < 1.0:
        raise ValueError("episode FP threshold must lie in [0.5, 1)")
    if not 0.0 <= float(args.episode_minimum_confidence) <= 1.0:
        raise ValueError("episode confidence must lie in [0, 1]")
    if int(args.episode_max_extra_points) <= 0:
        raise ValueError("episode maximum blocked-point audit must be positive")
    if int(args.episode_min_history) <= 0 or int(args.episode_min_per_class) <= 0:
        raise ValueError("episode history geometry must be positive")
    if float(args.episode_high_escape_margin) < 0.0:
        raise ValueError("episode high escape margin must be non-negative")
    if not 0.0 <= float(args.episode_physics_escape) <= 1.0:
        raise ValueError("episode physics escape must lie in [0, 1]")
    if float(args.replay_max_distance) <= 0.0:
        raise ValueError("episode replay distance must be positive")
    if int(args.episode_maximum_points) < 16:
        raise ValueError("episode supervision must retain at least 16 points")
    if float(args.episode_shoulder_margin) < 0.0:
        raise ValueError("episode shoulder margin must be non-negative")
    if int(args.query_exploration_per_block) not in (0, 1):
        raise ValueError("episode v3 permits zero or one exploration slot")
    if str(args.episode_query_mode) not in ("shadow", "active"):
        raise ValueError("episode query mode must be shadow or active")
    if str(args.episode_gradient_mode) not in ("legacy", "episode"):
        raise ValueError("episode gradient mode must be legacy or episode")
    if isolate_feedback:
        if not bool(getattr(args, "episode_protected_sequence", False)):
            raise ValueError("feedback isolation requires protected sequence")
        if str(args.episode_query_mode) != "shadow":
            raise ValueError("feedback isolation requires shadow query mode")
        if str(args.episode_gradient_mode) != "legacy":
            raise ValueError("feedback isolation requires legacy gradient mode")
    if bool(getattr(args, "episode_protected_sequence", False)):
        if float(args.episode_sequence_l2) <= 0.0:
            raise ValueError("protected sequence regularization must be positive")
        if float(args.episode_protection_margin) < 0.0:
            raise ValueError("protected sequence safety margin must be non-negative")
        if not 0.5 <= float(args.episode_negative_quantile) <= 1.0:
            raise ValueError("protected sequence negative quantile must lie in [0.5, 1]")
        if not 0.0 <= float(args.episode_safe_quantile) <= 0.5:
            raise ValueError("protected sequence safe quantile must lie in [0, 0.5]")


def validate_timescale_ablation_arguments(args: argparse.Namespace) -> None:
    """Fail closed around causal ablations and order-oracle diagnostics."""
    contract = timescale_ablation.timescale_contract(args.timescale_mode)
    if not 0.0 <= float(args.query_cost) <= 1.0:
        raise ValueError("query cost must lie in [0, 1]")
    if (
        str(args.query_score_mode) == "gradient_blend"
        and not bool(args.episode_value_v3)
    ):
        raise ValueError("gradient-blend query score requires --episode-value-v3")
    if not contract.enable_query and args.fixed_query_manifest is not None:
        raise ValueError("intra-only mode cannot accept a fixed query manifest")
    if not contract.enable_query and float(args.query_cost) != 0.0:
        raise ValueError("intra-only mode has no query operation to price")
    if (
        float(args.query_cost) > 0.0
        and float(args.query_token_capacity) <= 0.0
        and args.fixed_query_manifest is None
    ):
        raise ValueError("explicit query cost requires token-budget query control")
    experimental = (
        str(args.timescale_mode) != "intra_inter"
        or float(args.query_cost) != 0.0
        or args.fixed_query_manifest is not None
        or args.order_manifest is not None
        or bool(args.diagnostic_reorder)
    )
    if experimental and not bool(args.development_search):
        raise ValueError("timescale/order experiments require --development-search")
    if bool(getattr(args, "selector_compute_control", False)):
        if args.fixed_query_manifest is None:
            raise ValueError("selector compute control requires a fixed query manifest")
        if str(args.timescale_mode) != "inter_only":
            raise ValueError("selector compute control requires inter_only mode")
        forbidden = {
            "episode_value_v3": bool(args.episode_value_v3),
            "causal_error_memory": bool(args.causal_error_memory),
            "typed_endpoint_memory": bool(args.typed_endpoint_memory),
            "strict_window_active": bool(args.strict_window_active),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "selector compute control forbids adaptive side learners: "
                + ", ".join(enabled)
            )
        if (
            int(args.inter_steps) != 2
            or int(args.fixed_compute_current_windows) != 16
            or int(args.fixed_compute_replay_windows) != 16
        ):
            raise ValueError(
                "fixed-shape selector compute control requires 16+16 windows and 2 optimizer attempts"
            )
    if bool(args.diagnostic_reorder) and args.order_manifest is None:
        raise ValueError("diagnostic reorder requires --order-manifest")
    temporal_risk_weight = float(
        getattr(args, "intra_temporal_risk_weight", 0.0)
    )
    observed_physics_weight = float(
        getattr(args, "intra_observed_physics_weight", 0.0)
    )
    temporal_target_max_delta = float(
        getattr(args, "intra_temporal_target_max_delta", 0.5)
    )
    temporal_target_gain = float(
        getattr(args, "intra_temporal_target_gain", 1.0)
    )
    intra_parameter_scope = str(
        getattr(args, "intra_parameter_scope", "legacy")
    )
    if not math.isfinite(temporal_risk_weight):
        raise ValueError("intra temporal risk weight must be finite")
    if not math.isfinite(observed_physics_weight):
        raise ValueError("intra observed physics weight must be finite")
    if temporal_risk_weight < 0.0 or observed_physics_weight < 0.0:
        raise ValueError("intra unlabeled loss weights must be non-negative")
    if (
        not math.isfinite(temporal_target_max_delta)
        or temporal_target_max_delta <= 0.0
    ):
        raise ValueError("intra temporal target max delta must be positive")
    if not math.isfinite(temporal_target_gain) or temporal_target_gain <= 0.0:
        raise ValueError("intra temporal target gain must be positive")
    if intra_parameter_scope not in ("legacy", "representation_only"):
        raise ValueError("invalid intra parameter scope")
    if (
        (
            temporal_risk_weight > 0.0
            or observed_physics_weight > 0.0
            or intra_parameter_scope == "representation_only"
        )
        and not bool(args.strict_window_active)
    ):
        raise ValueError(
            "intra observed-window learning requires strict prediction-before-update"
        )


def expected_query_count(
    shot_count: int, *, block_size: int, quota: int,
) -> int:
    """Return the exact causal query budget for complete chronological blocks."""
    if block_size <= 0 or quota <= 0 or quota > block_size:
        raise ValueError("invalid exact query budget")
    if shot_count <= 0 or shot_count % block_size:
        raise ValueError("shot roster must contain complete query blocks")
    return int(shot_count // block_size * quota)


def inter_update_control(steps: int, update_scale: float) -> tuple[int, float]:
    """Map an ablation with zero steps to a state-finalizing zero-gradient call."""
    if int(steps) < 0:
        raise ValueError("inter steps cannot be negative")
    if not 0.0 <= float(update_scale) <= 1.0:
        raise ValueError("inter update scale must lie in [0, 1]")
    if int(steps) == 0:
        return 1, 0.0
    return int(steps), float(update_scale)


def fixed_compute_supervised_indices(
    *, risk: np.ndarray, positive: np.ndarray,
    legal_start: int, legal_stop: int, count: int = 16,
) -> np.ndarray:
    """Select a fixed-size post-reveal batch with a branch-free objective.

    Safe shots contribute their hardest negatives.  Disruptive shots contribute
    equal positive/negative halves.  The outcome is used only after the shot's
    prediction and query decision have been sealed.  Repetition is deterministic
    when a pool is smaller than its quota, preserving an identical tensor shape.
    """
    values = np.asarray(risk, dtype=np.float64).reshape(-1)
    labels = np.asarray(positive, dtype=bool).reshape(-1)
    if values.shape != labels.shape or not np.isfinite(values).all():
        raise ValueError("risk and positive labels must be aligned and finite")
    if count <= 0 or count % 2:
        raise ValueError("fixed-compute count must be a positive even integer")
    start, stop = int(legal_start), int(legal_stop)
    if start < 0 or stop > len(values) or start >= stop:
        raise ValueError("fixed-compute legal interval is empty or invalid")
    legal = np.arange(start, stop, dtype=np.int64)

    def take(pool: np.ndarray, difficulty: np.ndarray, quota: int) -> np.ndarray:
        if not len(pool):
            raise ValueError("fixed-compute supervision pool is empty")
        if len(pool) < int(quota):
            raise ValueError("fixed-compute supervision pool has fewer than quota rows")
        order = np.lexsort((pool, -difficulty))
        return pool[order[: int(quota)]].astype(np.int64, copy=False)

    positive_pool = legal[labels[legal]]
    if not len(positive_pool):
        return np.sort(take(legal, values[legal], count))
    negative_pool = legal[~labels[legal]]
    if not len(negative_pool):
        raise ValueError("disruptive fixed-compute batch requires negatives")
    half = count // 2
    chosen_positive = take(
        positive_pool, 1.0 - values[positive_pool], half,
    )
    chosen_negative = take(negative_pool, values[negative_pool], half)
    return np.sort(np.concatenate((chosen_positive, chosen_negative)))


def seal_static_phase_trace(
    path: Path, *, shot_id: int, static_logit: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        shot_id=np.asarray([int(shot_id)], dtype=np.int64),
        static_logit=np.asarray(static_logit, dtype=np.float32),
    )


def load_static_phase_trace(path: Path, *, expected_shot_id: int) -> np.ndarray:
    with np.load(path) as payload:
        sealed_shot = int(np.asarray(payload["shot_id"]).reshape(-1)[0])
        if sealed_shot != int(expected_shot_id):
            raise RuntimeError(
                f"static phase shot identity mismatch: {sealed_shot} != {expected_shot_id}"
            )
        result = np.asarray(payload["static_logit"], dtype=np.float32)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RuntimeError(f"shot {expected_shot_id}: invalid sealed static trace")
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def base_risk_for_parent(
    enc: torch.nn.Module, head: torch.nn.Module, shot: dict[str, Any],
    *, cached_path: Path | None,
) -> np.ndarray:
    """Recompute a sealed parent's risk unless an exact matching cache is supplied."""
    computed = alarm.full_risk(enc, head, shot).astype(np.float32)
    if cached_path is None:
        return computed
    with np.load(cached_path) as payload:
        cached = np.asarray(payload["risk"], dtype=np.float32)
    if cached.shape != computed.shape or np.max(np.abs(cached - computed)) > 1e-5:
        raise RuntimeError("cached A1 risk trace differs from the active parent")
    return cached


def causal_raw8_signal_only(shot_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Read raw8 without allowing future non-zero samples to mark the past valid."""
    path = DATA / "EAST/unified_hdf5" / str(shot_id // 100 * 100) / f"{shot_id}.hdf5"
    columns = []
    with h5py.File(path, "r") as handle:
        for name in private23.DROPPED_RAW8:
            columns.append(np.asarray(handle[f"data/{name}"][:], dtype=np.float32))
    values = np.stack(columns, axis=-1)
    available = np.isfinite(values)
    ciii = private23.DROPPED_RAW8.index("CIII")
    available[:, ciii] &= ~np.isclose(values[:, ciii], -10.0, atol=1e-2)
    for name in ("MNM", "Mir_VV", "Mir_avg_amp"):
        index = private23.DROPPED_RAW8.index(name)
        nonzero = available[:, index] & ~np.isclose(values[:, index], 0.0)
        seen = np.cumsum(nonzero.astype(np.int64)) > 0
        available[:, index] &= seen
    return values, available


def fit_causal_raw8_scaler(shot_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pools: list[list[np.ndarray]] = [[] for _ in private23.DROPPED_RAW8]
    for shot_id in shot_ids:
        values, available = causal_raw8_signal_only(int(shot_id))
        positions = np.linspace(0, len(values) - 1, min(len(values), 4096), dtype=np.int64)
        for channel in range(values.shape[1]):
            valid = available[positions, channel]
            if valid.any():
                pools[channel].append(values[positions[valid], channel])
    median = np.zeros(len(pools), dtype=np.float32)
    scale = np.ones(len(pools), dtype=np.float32)
    for channel, chunks in enumerate(pools):
        if not chunks:
            continue
        joined = np.concatenate(chunks)
        median[channel] = float(np.median(joined))
        q25, q75 = np.quantile(joined, [0.25, 0.75])
        scale[channel] = max(float(q75 - q25), 1e-6)
    return median, scale


def scale_raw8(
    values: np.ndarray, available: np.ndarray,
    median: np.ndarray, scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = np.clip((values - median) / scale, -10, 10)
    return np.where(available, scaled, 0.0).astype(np.float32), available.astype(bool)


def private_windows(
    values: np.ndarray, available: np.ndarray, indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    endpoints = alarm.W - 1 + indices
    raw = np.stack([values[end - alarm.W + 1:end + 1] for end in endpoints])
    mask = np.stack([available[end - alarm.W + 1:end + 1] for end in endpoints])
    return raw.astype(np.float32), mask.astype(bool)


def make_private_batch(
    values: np.ndarray, available: np.ndarray, indices: np.ndarray,
    base_logit: np.ndarray, *, target: np.ndarray | None = None,
    sealed_logit: np.ndarray | None = None, labels: np.ndarray | None = None,
    alarm_boundary: np.ndarray | None = None,
    next_indices: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    indices = np.asarray(indices, dtype=np.int64)
    raw, mask = private_windows(values, available, indices)
    batch = {
        "raw": torch.from_numpy(raw), "available": torch.from_numpy(mask),
        "base_logit": torch.from_numpy(np.asarray(base_logit, np.float32)[indices]),
    }
    if next_indices is not None:
        next_raw, next_mask = private_windows(values, available, next_indices)
        batch["next_features"] = private23.causal_private_features(
            torch.from_numpy(next_raw), torch.from_numpy(next_mask)
        ).detach()
    if target is not None:
        batch["target"] = torch.from_numpy(np.asarray(target, np.float32)[indices])
    if sealed_logit is not None:
        batch["sealed_logit"] = torch.from_numpy(
            np.asarray(sealed_logit, np.float32)[indices]
        )
    if labels is not None:
        batch["label"] = torch.from_numpy(np.asarray(labels, np.float32)[indices])
    if alarm_boundary is not None:
        batch["alarm_boundary"] = torch.from_numpy(
            np.asarray(alarm_boundary, np.float32)[indices]
        )
    return batch


def concatenate(*batches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = tuple(batches[0])
    if any(tuple(batch) != keys for batch in batches[1:]):
        raise ValueError("online batches do not share a schema")
    return {name: torch.cat([batch[name] for batch in batches]) for name in keys}


def sample_batch(
    batch: dict[str, torch.Tensor], *, count: int, seed: int,
) -> dict[str, torch.Tensor]:
    size = len(next(iter(batch.values())))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randint(size, (int(count),), generator=generator)
    return {name: value.index_select(0, indices) for name, value in batch.items()}


def build_private_support_batches(
    enc: torch.nn.Module, head: torch.nn.Module,
    shared_adapter: dynamic.DynamicHazardAdapter, transfer: list[dict],
    scales: dict[str, float], median8: np.ndarray, scale8: np.ndarray,
    *, minimum_positive: float, base_risk_atol: float = 1e-5,
) -> list[dict[str, torch.Tensor]]:
    support_batches: list[dict[str, torch.Tensor]] = []
    for shot in transfer:
        shot_id = int(shot["shot"])
        base_risk = alarm.full_risk(enc, head, shot)
        shared_risk, _delta = dha.score_shot(
            shared_adapter, enc, head, shot, scales, base_risk,
            minimum_positive_residual=minimum_positive,
            base_risk_atol=base_risk_atol,
        )
        binary, soft = old_private.temporal_targets(shot, len(shared_risk))
        shared_logit = np.log(
            np.clip(shared_risk, 1e-6, 1 - 1e-6)
            / np.clip(1 - shared_risk, 1e-6, 1 - 1e-6)
        ).astype(np.float32)
        legal = np.arange(max(len(shared_risk) - dha.LAG, 1), dtype=np.int64)
        selected_local = old_online.select_revealed_windows(
            shared_logit[legal], binary[legal], count=16,
        )
        selected = legal[selected_local]
        raw, available = causal_raw8_signal_only(shot_id)
        raw, available = scale_raw8(raw, available, median8, scale8)
        batch = make_private_batch(
            raw, available, selected, shared_logit, target=soft,
            sealed_logit=shared_logit, labels=binary,
            next_indices=np.minimum(selected + dha.LAG, len(shared_risk) - 1),
        )
        support_batches.append(batch)
    return support_batches


def private_reconstruction_contract(
    model: private_continual.EastPrivateDynamicsResidual,
    support_batches: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        errors = []
        for batch in support_batches:
            raw = batch["raw"].to(alarm.DEV)
            available = batch["available"].to(alarm.DEV)
            base = batch["base_logit"].to(alarm.DEV)
            _adjusted, _residual, latent, _features = model(raw, available, base)
            prediction = model.predict_next(latent)
            target = batch["next_features"].to(alarm.DEV)
            errors.extend(F.smooth_l1_loss(
                prediction, target, reduction="none",
            ).mean(dim=1).cpu().tolist())
    return {
        "private_reconstruction_q80": max(
            float(np.quantile(errors, 0.80)), 1e-6,
        ),
    }


def fit_private_model(
    enc: torch.nn.Module, head: torch.nn.Module,
    shared_adapter: dynamic.DynamicHazardAdapter, transfer: list[dict],
    scales: dict[str, float], median8: np.ndarray, scale8: np.ndarray,
    *, threshold: float, minimum_positive: float, seed: int, epochs: int,
    base_risk_atol: float = 1e-5,
) -> tuple[
    private_continual.EastPrivateDynamicsResidual,
    list[dict[str, torch.Tensor]], dict[str, float],
]:
    support_batches = build_private_support_batches(
        enc, head, shared_adapter, transfer, scales, median8, scale8,
        minimum_positive=minimum_positive, base_risk_atol=base_risk_atol,
    )
    support = concatenate(*support_batches)
    torch.manual_seed(int(seed) + 31000)
    model = private_continual.EastPrivateDynamicsResidual(
        channel_count=8, hidden=16, maximum_logit_residual=0.50,
    ).to(alarm.DEV)
    if sum(parameter.numel() for parameter in model.parameters()) >= 5000:
        raise RuntimeError("EAST-private continual parameter budget exceeded")
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    positive = torch.nonzero(support["target"] > 0, as_tuple=False).flatten()
    negative = torch.nonzero(support["target"] <= 0, as_tuple=False).flatten()
    if not len(positive) or not len(negative):
        raise RuntimeError("private support requires positive and negative temporal states")
    generator = torch.Generator().manual_seed(int(seed) + 31001)
    history = []
    for epoch in range(int(epochs)):
        losses = []
        model.train()
        for _ in range(max(1, len(support["target"]) // 64)):
            pos = positive[torch.randint(len(positive), (32,), generator=generator)]
            neg = negative[torch.randint(len(negative), (32,), generator=generator)]
            index = torch.cat([pos, neg])
            raw = support["raw"].index_select(0, index).to(alarm.DEV)
            available = support["available"].index_select(0, index).to(alarm.DEV)
            base = support["base_logit"].index_select(0, index).to(alarm.DEV)
            adjusted, residual, latent, _features = model(raw, available, base)
            hazard = private23.sealed_temporal_residual_loss(
                adjusted, support["sealed_logit"].index_select(0, index).to(alarm.DEV),
                support["target"].index_select(0, index).to(alarm.DEV),
            )
            next_target = support["next_features"].index_select(0, index).to(alarm.DEV)
            reconstruction = F.smooth_l1_loss(model.predict_next(latent), next_target)
            loss = hazard + 0.20 * reconstruction + 0.02 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return model, support_batches, {
        **private_reconstruction_contract(model, support_batches),
        "history_final_loss": history[-1]["loss"],
    }


def shared_reconstruction_scale(
    adapter: dynamic.DynamicHazardAdapter, support_cache: list[dict],
) -> float:
    errors = []
    adapter.eval()
    with torch.no_grad():
        for row in support_cache:
            frozen = torch.from_numpy(row["frozen"]).to(alarm.DEV)
            physics = torch.from_numpy(row["physics"]).to(alarm.DEV)
            available = torch.from_numpy(row["available"]).to(alarm.DEV)
            target = torch.from_numpy(row["next_physics"]).to(alarm.DEV)
            target_available = torch.from_numpy(row["next_available"]).to(alarm.DEV)
            latent = adapter.encode(frozen, physics, available)
            element = F.smooth_l1_loss(
                adapter.predict_next(latent), target, reduction="none",
            )
            valid = target_available & torch.isfinite(target)
            for index in range(len(element)):
                errors.append(float(element[index][valid[index]].mean().cpu()) if bool(valid[index].any()) else 0.0)
    return max(float(np.quantile(errors, 0.80)), 1e-6)


def static_combined_logits(
    shared: dynamic.DynamicHazardAdapter,
    private: private_continual.EastPrivateDynamicsResidual,
    *, frozen: np.ndarray, base_logit: np.ndarray,
    physics: np.ndarray, physics_available: np.ndarray,
    raw: np.ndarray, raw_available: np.ndarray,
    minimum_positive: float,
    risk_offset: int = 0,
) -> np.ndarray:
    output = np.empty(len(base_logit), dtype=np.float32)
    shared.eval()
    private.eval()
    for start in range(0, len(base_logit), 512):
        index = np.arange(start, min(start + 512, len(base_logit)))
        frozen_t = torch.from_numpy(frozen[index]).to(alarm.DEV)
        base_t = torch.from_numpy(base_logit[index]).to(alarm.DEV)
        physics_t = torch.from_numpy(physics[index]).to(alarm.DEV)
        available_t = torch.from_numpy(physics_available[index]).to(alarm.DEV)
        with torch.no_grad():
            _adjusted, delta, _gate, _latent = shared(
                frozen_t, physics_t, available_t, base_t,
            )
            shared_logit = base_t + dynamic.apply_positive_residual_gate(
                delta, minimum_positive=minimum_positive,
            )
        full_shared = np.zeros(int(risk_offset) + len(base_logit), dtype=np.float32)
        full_shared[index + int(risk_offset)] = shared_logit.cpu().numpy()
        private_batch = make_private_batch(
            raw, raw_available, index + int(risk_offset), full_shared,
        )
        with torch.no_grad():
            combined, _residual, _latent, _features = private(
                private_batch["raw"].to(alarm.DEV),
                private_batch["available"].to(alarm.DEV),
                private_batch["base_logit"].to(alarm.DEV),
            )
        output[index] = combined.cpu().numpy()
    return output


def _masked_shared_error(
    controller: two_scale.TwoTimescaleAdapterController,
    batch: dict[str, torch.Tensor],
) -> tuple[float, float]:
    prepared = {name: value.to(controller.device) for name, value in batch.items()}
    assert controller.fast is not None
    controller.fast.eval()
    controller.persistent.eval()
    with torch.no_grad():
        fast_logit, _delta, _gate, latent = controller._forward(controller.fast, prepared)
        slow_logit, _delta, _gate, _latent = controller._forward(controller.persistent, prepared)
        prediction = controller.fast.predict_next(latent)
        target = prepared["next_physics"]
        valid = prepared["next_available"] & torch.isfinite(target)
        error = F.smooth_l1_loss(prediction[valid], target[valid]) if bool(valid.any()) else prediction.sum() * 0.0
        disagreement = torch.abs(torch.sigmoid(fast_logit) - torch.sigmoid(slow_logit)).mean()
    return float(error.cpu()), float(disagreement.cpu())


def _private_error(
    controller: private_continual.PrivateTwoTimescaleController,
    batch: dict[str, torch.Tensor],
) -> tuple[float, float]:
    prepared = {name: value.to(controller.device) for name, value in batch.items()}
    assert controller.fast is not None
    controller.fast.eval()
    controller.persistent.eval()
    with torch.no_grad():
        fast_logit, _r, latent, _f = controller._forward(controller.fast, prepared)
        slow_logit, _r, _l, _f = controller._forward(controller.persistent, prepared)
        prediction = controller.fast.predict_next(latent)
        target = prepared["next_features"]
        error = F.smooth_l1_loss(prediction, target)
        disagreement = torch.abs(torch.sigmoid(fast_logit) - torch.sigmoid(slow_logit)).mean()
    return float(error.cpu()), float(disagreement.cpu())


def redundancy_score(vector: np.ndarray, history: list[np.ndarray]) -> float:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not history or norm <= 1e-12:
        return 0.0
    similarities = []
    for previous in history[-16:]:
        denominator = norm * float(np.linalg.norm(previous))
        if denominator > 1e-12:
            similarities.append(float(np.dot(vector, previous) / denominator))
    return float(np.clip((max(similarities) + 1.0) / 2.0, 0.0, 1.0)) if similarities else 0.0


EPISODE_FEATURE_NAMES = (
    "risk_probability", "physics_current", "physics_mean20",
    "physics_mean50", "risk_trend", "early_boundary_excess",
    "high_boundary_excess", "private_logit_residual",
    "endpoint_memory_residual", "risk_delta20", "risk_std20",
    "high_route", "model_branch_support", "alarm_parameter_branch_support",
    "ne_nG_std", "ne_nG_recent_delta", "ne_nG_full_delta",
    "sxr_mean_std", "CIII_last", "xuv_ratio_last",
)


def _causal_rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if width <= 0 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("causal rolling mean requires finite values and width")
    cumulative = np.cumsum(np.r_[0.0, vector])
    index = np.arange(len(vector), dtype=np.int64)
    lower = np.maximum(0, index - int(width) + 1)
    return (cumulative[index + 1] - cumulative[lower]) / (index - lower + 1)


def causal_episode_feature_matrix(
    *, online_logit: np.ndarray, static_logit: np.ndarray,
    shared_logit: np.ndarray, endpoint_memory_residual: np.ndarray,
    physics_support: np.ndarray, alarm_trace: dict[str, np.ndarray],
    threshold_logit: float, alarm_parameters: dict[str, float],
    raw: np.ndarray, raw_available: np.ndarray,
    model_branch_mask: np.ndarray, alarm_parameter_branch_mask: np.ndarray,
) -> np.ndarray:
    """Build the fixed causal base14 + robust raw6 episode representation."""
    online = np.asarray(online_logit, dtype=np.float64).reshape(-1)
    static = np.asarray(static_logit, dtype=np.float64).reshape(-1)
    shared = np.asarray(shared_logit, dtype=np.float64).reshape(-1)
    endpoint_residual = np.asarray(
        endpoint_memory_residual, dtype=np.float64,
    ).reshape(-1)
    support = np.asarray(physics_support, dtype=np.float64).reshape(-1)
    model_mask = np.asarray(model_branch_mask, dtype=bool).reshape(-1)
    parameter_mask = np.asarray(
        alarm_parameter_branch_mask, dtype=bool,
    ).reshape(-1)
    length = len(online)
    if (
        not length or any(len(item) != length for item in (
            static, shared, endpoint_residual, support, model_mask, parameter_mask,
        ))
    ):
        raise ValueError("episode logit and support traces must align")
    support = np.where(np.isfinite(support), np.clip(support, 0.0, 1.0), 0.0)
    arrays = (online, static, shared, endpoint_residual)
    if any(not np.isfinite(item).all() for item in arrays):
        raise ValueError("episode logit traces must be finite")
    for name in ("trend", "early_score", "high_mask"):
        if name not in alarm_trace or len(np.asarray(alarm_trace[name])) != length:
            raise ValueError(f"alarm trace missing aligned {name}")

    raw_values = np.asarray(raw, dtype=np.float64)
    raw_mask = np.asarray(raw_available, dtype=bool)
    if (
        raw_values.ndim != 2 or raw_values.shape != raw_mask.shape
        or raw_values.shape[1] != len(private23.DROPPED_RAW8)
        or len(raw_values) < length + alarm.W - 1
    ):
        raise ValueError("raw8 trace does not cover every causal risk window")

    probability = torch.sigmoid(torch.from_numpy(online)).numpy()
    margin = online - float(threshold_logit)
    lag20 = probability[np.maximum(0, np.arange(length) - 20)]
    risk_mean20 = _causal_rolling_mean(probability, 20)
    risk_square20 = _causal_rolling_mean(probability**2, 20)
    risk_std20 = np.sqrt(np.maximum(risk_square20 - risk_mean20**2, 0.0))
    trend = np.asarray(alarm_trace["trend"], dtype=np.float64)
    early_excess = (
        np.asarray(alarm_trace["early_score"], dtype=np.float64)
        - float(alarm_parameters["early_threshold"])
    )
    high_excess = margin - float(alarm_parameters["high_threshold"])
    high_route = np.asarray(alarm_trace["high_mask"], dtype=bool)

    starts = np.arange(length, dtype=np.int64)
    ends = starts + alarm.W
    middle = starts + alarm.W // 2

    def raw_stats(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        channel = private23.DROPPED_RAW8.index(name)
        values = raw_values[:, channel]
        available = raw_mask[:, channel] & np.isfinite(values)
        safe = np.where(available, values, 0.0)
        cumulative = np.cumsum(np.r_[0.0, safe])
        cumulative_square = np.cumsum(np.r_[0.0, safe**2])
        cumulative_count = np.cumsum(np.r_[0, available.astype(np.int64)])
        count = np.maximum(cumulative_count[ends] - cumulative_count[starts], 1)
        mean = (cumulative[ends] - cumulative[starts]) / count
        square = (cumulative_square[ends] - cumulative_square[starts]) / count
        std = np.sqrt(np.maximum(square - mean**2, 0.0))
        last_index = ends - 1
        last = np.where(available[last_index], safe[last_index], 0.0)
        first = np.where(available[starts], safe[starts], 0.0)
        middle_value = np.where(available[middle], safe[middle], 0.0)
        full_delta = np.where(
            available[last_index] & available[starts], last - first, 0.0,
        )
        recent_delta = np.where(
            available[last_index] & available[middle],
            last - middle_value, 0.0,
        )
        return last, std, recent_delta, full_delta

    ciii_last, _ciii_std, _ciii_recent, _ciii_full = raw_stats("CIII")
    ne_last, ne_std, ne_recent, ne_full = raw_stats("ne_nG")
    sxr_last, sxr_std, sxr_recent, sxr_full = raw_stats("sxr_mean")
    xuv_last, _xuv_std, _xuv_recent, _xuv_full = raw_stats("xuv_ratio")
    del ne_last, sxr_last, sxr_recent, sxr_full

    matrix = np.column_stack([
        probability,
        support,
        _causal_rolling_mean(support, 20),
        _causal_rolling_mean(support, 50),
        np.tanh(trend),
        np.tanh(early_excess / 2.0),
        np.tanh(high_excess / 2.0),
        np.tanh((online - shared - endpoint_residual) / 2.0),
        np.tanh(endpoint_residual / 2.0),
        probability - lag20,
        risk_std20,
        high_route.astype(np.float64),
        model_mask.astype(np.float64),
        parameter_mask.astype(np.float64),
        np.tanh(ne_std / 3.0),
        np.tanh(ne_recent / 3.0),
        np.tanh(ne_full / 3.0),
        np.tanh(sxr_std / 3.0),
        np.tanh(ciii_last / 3.0),
        np.tanh(xuv_last / 3.0),
    ]).astype(np.float64)
    if matrix.shape != (length, len(EPISODE_FEATURE_NAMES)):
        raise RuntimeError("episode feature schema changed")
    if not np.isfinite(matrix).all():
        raise RuntimeError("episode features became non-finite")
    return matrix


def endpoint_memory_key(
    *, frozen_state: np.ndarray, risk_logit: np.ndarray, risk_index: int,
    physics_support: float, physics_known: bool,
    raw: np.ndarray, raw_available: np.ndarray, raw_endpoint: int,
) -> np.ndarray:
    """Build a bounded endpoint key using only the current causal prefix."""
    hidden = np.asarray(frozen_state, dtype=np.float64).reshape(-1)
    if len(hidden) != dha.FEATURE_DIM or not np.isfinite(hidden).all():
        raise ValueError("invalid frozen endpoint state")
    hidden_scale = max(float(np.sqrt(np.mean(hidden**2))), 1e-6)
    hidden = np.tanh(hidden / hidden_scale)
    chunks = np.array_split(hidden, 8)
    hidden_summary = np.asarray(
        [item for chunk in chunks for item in (chunk.mean(), chunk.std())],
        dtype=np.float64,
    )

    logits = np.asarray(risk_logit, dtype=np.float64).reshape(-1)
    risk_index = int(risk_index)
    if risk_index < 0 or risk_index >= len(logits):
        raise ValueError("invalid risk endpoint index")
    current = float(logits[risk_index])

    def lagged(lag: int) -> float:
        return float(logits[max(0, risk_index - int(lag))])

    prefix = logits[:risk_index + 1]
    recent5 = prefix[-5:]
    recent50 = prefix[-50:]
    risk_summary = np.asarray([
        math.tanh(current / 3.0),
        math.tanh((current - lagged(5)) / 2.0),
        math.tanh((current - lagged(20)) / 2.0),
        math.tanh((current - lagged(100)) / 2.0),
        math.tanh((float(recent5.mean()) - float(recent50.mean())) / 2.0),
        2.0 * float(np.mean(recent5 >= 0.0)) - 1.0,
        2.0 * float(np.mean(recent50 >= 0.0)) - 1.0,
    ], dtype=np.float64)

    raw_values = np.asarray(raw, dtype=np.float64)
    raw_mask = np.asarray(raw_available, dtype=bool)
    raw_endpoint = int(raw_endpoint)
    if (
        raw_values.ndim != 2 or raw_mask.shape != raw_values.shape
        or raw_endpoint < 0 or raw_endpoint >= len(raw_values)
    ):
        raise ValueError("invalid causal raw8 endpoint")
    start = max(0, raw_endpoint - 127)
    raw_prefix = raw_values[start:raw_endpoint + 1]
    mask_prefix = raw_mask[start:raw_endpoint + 1] & np.isfinite(raw_prefix)
    raw_summary: list[float] = []
    for channel in range(raw_values.shape[1]):
        valid = mask_prefix[:, channel]
        if not bool(valid.any()):
            raw_summary.extend((0.0, 0.0, 0.0))
            continue
        observed = raw_prefix[valid, channel]
        raw_summary.extend((
            math.tanh(float(observed[-1]) / 3.0),
            math.tanh(float(observed.mean()) / 3.0),
            math.tanh(float(observed[-1] - observed[0]) / 3.0),
        ))
    raw_summary.append(2.0 * float(mask_prefix.mean()) - 1.0)
    physics = np.asarray([
        2.0 * float(np.clip(physics_support, 0.0, 1.0)) - 1.0,
        1.0 if bool(physics_known) else -1.0,
    ], dtype=np.float64)
    key = np.concatenate([
        hidden_summary, risk_summary,
        np.asarray(raw_summary, dtype=np.float64), physics,
    ])
    if not np.isfinite(key).all():
        raise RuntimeError("endpoint-memory key became non-finite")
    return np.clip(key, -1.0, 1.0).astype(np.float32)


def query_features(
    *, risk: np.ndarray, physics_support: np.ndarray,
    physics_known: np.ndarray, threshold: float,
    update_values: list[float], reconstruction: list[float],
    teacher: list[float], private_innovation: list[float],
) -> tuple[dict[str, float], dict[str, float]]:
    components = audit.query_components(
        risk=risk, physics_support=physics_support,
        physics_known=physics_known, threshold=threshold,
    )
    peak = float(np.max(risk))
    features = {
        "uncertainty": float(np.clip(components["uncertainty"], 0, 1)),
        "physics_contradiction": float(np.clip(max(
            components["fp_contradiction"], components["fn_contradiction"],
        ), 0, 1)),
        "self_supervised_surprise": float(np.clip(
            np.quantile(reconstruction, 0.90) if reconstruction else 0.0, 0, 1,
        )),
        "teacher_disagreement": float(np.clip(
            np.quantile(teacher, 0.90) if teacher else 0.0, 0, 1,
        )),
        "private_innovation": float(np.clip(
            np.quantile(private_innovation, 0.90) if private_innovation else 0.0, 0, 1,
        )),
        "expected_gradient": float(np.clip(4.0 * peak * (1.0 - peak), 0, 1)),
    }
    components["mean_update_value"] = float(np.mean(update_values)) if update_values else 0.0
    return features, components


def error_memory_phenotype(
    *, risk: np.ndarray, threshold: float,
    query_features: dict[str, float],
) -> dict[str, float]:
    """Summarize a sealed, label-free shot for causal error retrieval."""
    probability = np.clip(np.asarray(risk, np.float64).reshape(-1), 0.0, 1.0)
    if not len(probability) or not np.all(np.isfinite(probability)):
        raise ValueError("error-memory risk trace must be finite and non-empty")
    split = max(1, len(probability) // 3)
    early = float(np.mean(probability[:split]))
    late = float(np.mean(probability[-split:]))
    variation = (
        float(np.mean(np.abs(np.diff(probability)))) if len(probability) > 1 else 0.0
    )
    phenotype = {
        name: float(np.clip(value, 0.0, 1.0))
        for name, value in query_features.items()
    }
    phenotype.update({
        "risk_peak": float(np.max(probability)),
        "risk_mean": float(np.mean(probability)),
        "risk_late_minus_early": float(np.clip(0.5 + 0.5 * (late - early), 0, 1)),
        "risk_total_variation": float(np.clip(variation, 0, 1)),
        "risk_above_threshold_fraction": float(np.mean(probability >= threshold)),
    })
    return phenotype


def supervision_value(
    risk: np.ndarray, temporal_target: np.ndarray, *, group: str,
    lead_ms: float | None,
) -> float:
    probability = np.clip(np.asarray(risk, np.float64), 1e-6, 1 - 1e-6)
    target = np.asarray(temporal_target, np.float64)
    bce = -np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))
    risk_term = float(np.clip(bce / math.log(2.0), 0.0, 1.0))
    operation = {"FN": 1.0, "TARDY": 3.0 / 5.0, "FP": 1.0 / 5.0}.get(group, 0.0)
    lead_gap = 0.0
    if group == "FN":
        lead_gap = 1.0
    elif group == "TARDY" and lead_ms is not None:
        lead_gap = float(np.clip((30.0 - lead_ms) / 30.0, 0.0, 1.0))
    return float(np.clip(0.50 * risk_term + 0.30 * operation + 0.20 * lead_gap, 0, 1))


def episode_supervision_scale(
    base_value: float, *, discrepancy_group: str,
    episode_confidence: float,
) -> float:
    """Give a revealed FP enough optimization mass to change later shots."""
    value = float(np.clip(base_value, 0.0, 1.0))
    if discrepancy_group != "FP":
        return value
    confidence = float(np.clip(episode_confidence, 0.0, 1.0))
    return max(value, 0.30 + 0.20 * confidence)


def decision_query_realized_utility(
    *, discrepancy_group: str, supervision_value: float,
    episode_updated: bool, shared_accepted: bool,
    private_accepted: bool, novelty: float,
) -> float:
    """Value the revealed label by downstream causal learning it enabled.

    A queried FP remains valuable even when one gradient proposal is rolled
    back: it teaches both the episode classifier and future query targeting.
    Conversely, a redundant correct shot receives only a small protection and
    coverage value.  This target is computed only after reveal and can affect
    later shots, never the just-sealed prediction.
    """
    priorities = {"FP": 1.0, "FN": 1.0, "TARDY": 0.85, "TP": 0.25, "TN": 0.10}
    if discrepancy_group not in priorities:
        raise ValueError("invalid query-utility discrepancy group")
    accepted = 0.5 * (float(bool(shared_accepted)) + float(bool(private_accepted)))
    utility = (
        0.55 * priorities[discrepancy_group]
        + 0.20 * float(np.clip(supervision_value, 0.0, 1.0))
        + 0.15 * float(bool(episode_updated))
        + 0.07 * accepted
        + 0.03 * float(np.clip(novelty, 0.0, 1.0))
    )
    return float(np.clip(utility, 0.0, 1.0))


def operational_temporal_targets(
    shot: dict[str, Any], risk_length: int, *,
    minimum_lead_ms: float = 30.0, horizon_ms: float = 125.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build post-reveal targets without weakening the timely alarm interval.

    The historical soft target rises towards the disruption and is useful as a
    mechanism-state target.  It is not an operational alarm target: near the
    start of the legal 125 ms interval it is close to zero, so correcting a
    missed or tardy alarm there can accidentally train the private EAST branch
    downwards.  Keep that target for audit, but use an HDF5-derived binary
    target for risk correction and explicitly identify the 30--125 ms interval
    from which FN/TARDY windows must be selected.
    """
    if int(risk_length) <= 0:
        raise ValueError("risk_length must be positive")
    if not 0.0 < float(minimum_lead_ms) < float(horizon_ms):
        raise ValueError("minimum lead must lie inside the alarm horizon")
    binary, mechanism = old_private.temporal_targets(shot, int(risk_length))
    binary = np.asarray(binary, dtype=bool)
    mechanism = np.asarray(mechanism, dtype=np.float32)
    operational = binary.astype(np.float32)
    timely = np.zeros(int(risk_length), dtype=bool)
    if bool(shot["is_disrupt"]) and shot.get("down") is not None:
        times = (
            float(shot["start"])
            + np.arange(
                alarm.W - 1, alarm.W - 1 + int(risk_length), dtype=np.float64,
            ) / float(shot["rate"])
        )
        ttd_ms = (float(shot["down"]) - times) * 1000.0
        timely = (
            (ttd_ms >= float(minimum_lead_ms) - 1e-9)
            & (ttd_ms <= float(horizon_ms) + 1e-9)
        )
        timely &= binary
    return binary, mechanism, operational, timely


def sealed_alarm_boundary_logit(
    combined_logit: np.ndarray, *, physics_support: np.ndarray,
    threshold_logit: float, alarm_parameters: dict[str, float],
) -> np.ndarray:
    """Freeze the easiest pointwise dual-alarm boundary before reveal."""
    values = np.asarray(combined_logit, dtype=np.float64)
    support = np.asarray(physics_support, dtype=np.float64)
    if values.ndim != 1 or support.shape != values.shape or not len(values):
        raise ValueError("combined logits and physics support must align")
    margin = values - float(threshold_logit)
    trace = dual_alarm.dual_threshold_trace(
        margin, support,
        high_threshold=alarm_parameters["high_threshold"],
        physics_weight=alarm_parameters["physics_weight"],
        trend_weight=alarm_parameters["trend_weight"],
        early_threshold=alarm_parameters["early_threshold"],
        fast_width=5, slow_width=100,
    )
    early_required = (
        float(alarm_parameters["early_threshold"])
        - float(alarm_parameters["physics_weight"]) * trace["physics_support"]
        - float(alarm_parameters["trend_weight"]) * trace["trend"]
    )
    # The early route is unavailable when its physics input was unavailable;
    # mirror the registered alarm rather than treating missing support as zero.
    early_required = np.where(np.isfinite(support), early_required, np.inf)
    boundary_margin = np.minimum(
        float(alarm_parameters["high_threshold"]), early_required,
    )
    return (float(threshold_logit) + boundary_margin).astype(np.float32)


def supported_fusion_trace(
    *, online_logit: np.ndarray, static_logit: np.ndarray,
    physics_support: np.ndarray, threshold_logit: float,
    alarm_parameters: dict[str, float],
    novel_early_min_physics_support: float = 0.0,
    novel_early_min_trend: float = 0.0,
) -> dict[str, Any]:
    """Return every causally supported endpoint before selecting the first."""
    online = np.asarray(online_logit, dtype=np.float64)
    static = np.asarray(static_logit, dtype=np.float64)
    support = np.asarray(physics_support, dtype=np.float64)
    if (
        online.ndim != 1 or not len(online)
        or static.shape != online.shape or support.shape != online.shape
    ):
        raise ValueError("fusion alarm traces must align")
    online_margin = online - float(threshold_logit)
    static_margin = static - float(threshold_logit)
    actual = dual_alarm.dual_threshold_trace(
        online_margin, support, **alarm_parameters,
    )
    model_branch = dual_alarm.dual_threshold_trace(
        online_margin, support, **INITIAL_ALARM_PARAMETERS,
    )
    alarm_branch = dual_alarm.dual_threshold_trace(
        static_margin, support, **alarm_parameters,
    )
    actual_positions = np.flatnonzero(actual["alarm_mask"])
    supported = actual["alarm_mask"] & (
        model_branch["alarm_mask"] | alarm_branch["alarm_mask"]
    )
    minimum_support = float(novel_early_min_physics_support)
    if not 0.0 <= minimum_support <= 1.0:
        raise ValueError("novel early physics support must lie in [0, 1]")
    minimum_trend = float(novel_early_min_trend)
    if not np.isfinite(minimum_trend) or minimum_trend < 0.0:
        raise ValueError("novel early trend support must be finite and non-negative")
    novel_model_early = (
        actual["early_mask"] & model_branch["alarm_mask"]
        & ~alarm_branch["alarm_mask"]
    )
    below_physics = support < minimum_support
    if minimum_trend > 0.0:
        rejected_by_guard = (
            novel_model_early & below_physics
            & (actual["trend"] < minimum_trend)
        )
        guard_reason = "novel_early_support_or_trend_guard"
    else:
        # Preserve the sealed v1 physics-only behavior when no trend escape
        # threshold is registered.
        rejected_by_guard = novel_model_early & below_physics
        guard_reason = "novel_early_physics_guard"
    supported &= ~rejected_by_guard
    return {
        "actual": actual, "model_branch": model_branch,
        "alarm_parameter_branch": alarm_branch,
        "supported_mask": supported,
        "rejected_by_guard": rejected_by_guard,
        "guard_reason": guard_reason,
        "raw_alarm_index": (
            int(actual_positions[0]) if len(actual_positions) else None
        ),
    }


def supported_fusion_alarm(
    *, online_logit: np.ndarray, static_logit: np.ndarray,
    physics_support: np.ndarray, threshold_logit: float,
    alarm_parameters: dict[str, float],
    novel_early_min_physics_support: float = 0.0,
    novel_early_min_trend: float = 0.0,
) -> dict[str, Any]:
    """Reject alarms caused only by unvalidated model/parameter interaction.

    The adapted combined route may alarm only at a causal endpoint where either
    the adapted model under the immutable alarm parameters, or the immutable
    model under the adapted alarm parameters, independently alarms.  This gate
    is label-free and can only delay or veto the raw combined alarm.
    """
    trace = supported_fusion_trace(
        online_logit=online_logit, static_logit=static_logit,
        physics_support=physics_support, threshold_logit=threshold_logit,
        alarm_parameters=alarm_parameters,
        novel_early_min_physics_support=novel_early_min_physics_support,
        novel_early_min_trend=novel_early_min_trend,
    )
    actual = trace["actual"]
    model_branch = trace["model_branch"]
    alarm_branch = trace["alarm_parameter_branch"]
    supported = trace["supported_mask"]
    rejected_by_guard = trace["rejected_by_guard"]
    guard_reason = str(trace["guard_reason"])
    raw_index = trace["raw_alarm_index"]
    positions = np.flatnonzero(supported)
    if not len(positions):
        return {
            "alarmed": False, "alarm_index": None, "route": None,
            "early_score": None,
            "filter_reason": (
                guard_reason
                if raw_index is not None and np.any(rejected_by_guard)
                else "no_independent_branch_support" if raw_index is not None
                else "no_raw_combined_alarm"
            ),
            "unfiltered_alarmed": raw_index is not None,
            "unfiltered_alarm_index": raw_index,
            "model_branch_support": False,
            "alarm_parameter_branch_support": False,
        }
    index = int(positions[0])
    model_support = bool(model_branch["alarm_mask"][index])
    parameter_support = bool(alarm_branch["alarm_mask"][index])
    return {
        "alarmed": True, "alarm_index": index,
        "route": "high" if bool(actual["high_mask"][index]) else "early",
        "early_score": float(actual["early_score"][index]),
        "filter_reason": "independent_branch_supported",
        "unfiltered_alarmed": raw_index is not None,
        "unfiltered_alarm_index": raw_index,
        "model_branch_support": model_support,
        "alarm_parameter_branch_support": parameter_support,
    }


def deploy_episode_decision(
    learning_alarm: Mapping[str, Any],
    episode_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay a label-free veto while preserving the sealed learning alarm.

    The deployed decision may delay or veto the legacy supported-fusion alarm,
    but it cannot advance or create one.  Keeping the counterfactual learning
    alarm explicit lets a one-factor episode screen avoid changing the v2.1
    model/query/alarm-parameter update trajectory merely because a current FP
    was already corrected by the new deployment gate.
    """
    required = {"alarmed", "alarm_index", "route", "early_score"}
    missing = required.difference(episode_decision)
    if missing:
        raise KeyError(f"episode deployment decision missing {sorted(missing)}")
    deployed_alarm = bool(episode_decision["alarmed"])
    deployed_index = episode_decision["alarm_index"]
    learning_active = bool(learning_alarm["alarmed"])
    learning_index = learning_alarm["alarm_index"]
    if deployed_alarm:
        if not learning_active or deployed_index is None or learning_index is None:
            raise ValueError("episode decision cannot create an alarm")
        if int(deployed_index) < int(learning_index):
            raise ValueError("episode decision cannot advance an alarm")
    elif deployed_index is not None:
        raise ValueError("episode veto cannot retain an alarm index")
    receipt = copy.deepcopy(dict(learning_alarm))
    receipt.update({
        "alarmed": deployed_alarm,
        "alarm_index": None if deployed_index is None else int(deployed_index),
        "route": episode_decision["route"],
        "early_score": episode_decision["early_score"],
    })
    for name, value in episode_decision.items():
        if name not in required:
            receipt[name] = copy.deepcopy(value)
    receipt["learning_alarm_reference"] = copy.deepcopy(dict(learning_alarm))
    return receipt


def causal_discrepancy_groups(
    *, deployed_alarm: Mapping[str, Any], learning_alarm: Mapping[str, Any],
    times_s: np.ndarray, is_disrupt: bool, down_time_s: float | None,
) -> dict[str, str]:
    """Return deployed performance and counterfactual learning groups."""
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    return {
        "deployed": value_module.ProjectedAlarmController._group(
            deployed_alarm, times, is_disrupt=bool(is_disrupt),
            down_time_s=down_time_s,
        ),
        "learning": value_module.ProjectedAlarmController._group(
            learning_alarm, times, is_disrupt=bool(is_disrupt),
            down_time_s=down_time_s,
        ),
    }


def select_supervised_indices(
    *, risk: np.ndarray, labels: np.ndarray, margin: np.ndarray,
    physics_support: np.ndarray, alarm_parameters: dict[str, float],
    group: str, count: int = 16,
    timely_labels: np.ndarray | None = None,
    alarm_boundary_margin: np.ndarray | None = None,
) -> np.ndarray:
    risk = np.asarray(risk, float)
    labels = np.asarray(labels, bool)
    margin = np.asarray(margin, float)
    if risk.shape != labels.shape or margin.shape != risk.shape:
        raise ValueError("risk, margin, and labels must align")
    timely = None
    if timely_labels is not None:
        timely = np.asarray(timely_labels, bool)
        if timely.shape != labels.shape:
            raise ValueError("timely labels must align with risk")
    if alarm_boundary_margin is not None:
        boundary = np.asarray(alarm_boundary_margin, float)
        if boundary.shape != risk.shape:
            raise ValueError("alarm boundary margin must align with risk")
        evidence = margin - boundary
    else:
        trace = dual_alarm.dual_threshold_trace(
            margin, physics_support,
            high_threshold=alarm_parameters["high_threshold"],
            physics_weight=alarm_parameters["physics_weight"],
            trend_weight=alarm_parameters["trend_weight"],
            early_threshold=alarm_parameters["early_threshold"],
            fast_width=5, slow_width=100,
        )
        evidence = np.maximum(
            margin - alarm_parameters["high_threshold"],
            trace["early_score"] - alarm_parameters["early_threshold"],
        )
    if group in ("FN", "TARDY") and labels.any():
        priority = timely if timely is not None and timely.any() else labels
        pool = np.flatnonzero(priority)
        # Any timely crossing is sufficient.  Spend scarce queried-label
        # gradient on the closest legal evidence, not the lowest-risk point.
        order = np.argsort(evidence[pool])[::-1]
        return np.sort(pool[order[:min(int(count), len(pool))]])
    order = np.argsort(evidence)[::-1]
    return np.sort(order[:min(int(count), len(order))].astype(np.int64))


def select_fp_episode_indices(
    *, evidence: np.ndarray, alarm_index: int, maximum_points: int = 64,
    shoulder_margin: float = 0.50,
) -> np.ndarray:
    """Select one complete sealed false-alarm episode around its first hit.

    The outcome is already revealed when this helper is called, but the input
    evidence trace was sealed beforehand.  Selection therefore cannot rewrite
    the current alarm; it only localizes supervision for later shots.
    """
    values = np.asarray(evidence, dtype=np.float64).reshape(-1)
    index = int(alarm_index)
    if (
        not len(values) or not np.isfinite(values).all()
        or index < 0 or index >= len(values)
    ):
        raise ValueError("invalid sealed FP episode evidence")
    if maximum_points <= 0 or shoulder_margin < 0:
        raise ValueError("invalid FP episode selection limits")
    active = values >= -float(shoulder_margin)
    active[index] = True
    left = index
    while left > 0 and bool(active[left - 1]):
        left -= 1
    right = index
    while right + 1 < len(values) and bool(active[right + 1]):
        right += 1
    if right - left + 1 > int(maximum_points):
        half = int(maximum_points) // 2
        left = max(left, index - half)
        right = min(right, left + int(maximum_points) - 1)
        left = max(left, right - int(maximum_points) + 1)
    return np.arange(left, right + 1, dtype=np.int64)


def alarm_metrics(rows: list[dict], outcomes: dict[int, dict]) -> dict[str, Any]:
    disruptive = 0
    safe = 0
    tp = 0
    fp = 0
    timely = 0
    leads = []
    for row in rows:
        outcome = outcomes[int(row["shot"])]
        alarmed = bool(row["alarmed"])
        if outcome["is_disrupt"]:
            disruptive += 1
            if alarmed:
                tp += 1
                lead = (float(outcome["down_time_s"]) - float(row["alarm_time_s"])) * 1000.0
                leads.append(lead)
                timely += int(lead >= 30.0 - 1e-9)
        else:
            safe += 1
            fp += int(alarmed)
    return {
        "TP": tp, "FP": fp, "FN": disruptive - tp, "TN": safe - fp,
        "tpr": tp / max(disruptive, 1), "fpr": fp / max(safe, 1),
        "tpr_at_30ms": timely / max(disruptive, 1),
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "over_3000ms_count": int(sum(lead > 3000 for lead in leads)),
    }


def fit_support_query_cutoff(
    enc: torch.nn.Module, head: torch.nn.Module, transfer: list[dict],
    scales: dict[str, float], *, threshold: float, quantile: float = 0.80,
) -> float:
    """Fit the online value cutoff without reading support outcomes as features."""
    model = value_module.OnlineQueryValueModel()
    scores = []
    for shot in sorted(transfer, key=lambda row: int(row["shot"])):
        risk = alarm.full_risk(enc, head, shot)
        physics, known, _count = a1.physics_trace(
            int(shot["shot"]), scales, risk_length=len(risk),
        )
        components = audit.query_components(
            risk=risk, physics_support=physics,
            physics_known=known, threshold=threshold,
        )
        peak = float(np.max(risk))
        features = {
            "uncertainty": float(components["uncertainty"]),
            "physics_contradiction": float(max(
                components["fp_contradiction"], components["fn_contradiction"],
            )),
            "self_supervised_surprise": 0.50,
            "teacher_disagreement": 0.0,
            "private_innovation": 0.50,
            "expected_gradient": float(np.clip(4 * peak * (1 - peak), 0, 1)),
        }
        shot_id = int(shot["shot"])
        scores.append(model.score_and_seal(shot_id, features))
        model.discard_without_reveal(shot_id)
    return float(np.quantile(scores, float(quantile)))


def score_online_shot(
    shared_controller: two_scale.TwoTimescaleAdapterController,
    private_controller: private_continual.PrivateTwoTimescaleController,
    alarm_controller: value_module.ProjectedAlarmController,
    update_value: value_module.CausalUpdateValue,
    *, shot_id: int, base_risk: np.ndarray, endpoints: np.ndarray,
    frozen: np.ndarray, base_logit_dynamic: np.ndarray,
    physics_at_endpoint: np.ndarray, available_at_endpoint: np.ndarray,
    full_physics: np.ndarray, full_available: np.ndarray,
    physics_support: np.ndarray, physics_known: np.ndarray,
    raw: np.ndarray, raw_available: np.ndarray,
    static_logit_reference: np.ndarray,
    threshold_logit: float, shared_error_scale: float,
    private_error_scale: float, adaptation_events: int,
    intra_steps: int, start_time_s: float, sample_rate_hz: float,
    novel_early_min_physics_support: float = 0.0,
    novel_early_min_trend: float = 0.0,
    strict_window_active: bool = False,
    window_scheduler_options: dict[str, Any] | None = None,
    endpoint_memory: value_module.TypedEndpointResidualMemory | None = None,
    endpoint_memory_stride: int = 16,
    episode_head: (
        episode_value.CausalEpisodeFPHead
        | episode_value.CausalProtectedSequenceFPHead
        | None
    ) = None,
    episode_high_escape_margin: float = 0.75,
    episode_physics_escape: float = 0.85,
    episode_isolate_learning_feedback: bool = False,
    enable_intra: bool = True,
    state_shot_id: int | None = None,
) -> dict[str, Any]:
    state_id = int(shot_id if state_shot_id is None else state_shot_id)
    shared_controller.begin_shot(state_id)
    private_controller.begin_shot(state_id)
    alarm_controller.begin_shot(state_id)
    risk_length = len(base_risk)
    dynamic_offset = dha.LAG
    if len(endpoints) != risk_length - dynamic_offset:
        raise ValueError("dynamic endpoints do not match risk trace")
    combined_logit = np.log(
        np.clip(base_risk, 1e-6, 1 - 1e-6)
        / np.clip(1 - base_risk, 1e-6, 1 - 1e-6)
    ).astype(np.float32)
    shared_logit = combined_logit.copy()
    times = np.empty(risk_length, dtype=np.float64)
    # The caller supplies endpoints at the same 1 kHz timebase used by HDF5.
    times[:] = np.nan
    receipts: list[dict[str, Any]] = []
    values: list[float] = []
    reconstructions: list[float] = []
    teacher_disagreements: list[float] = []
    private_innovations: list[float] = []
    recent_states: list[np.ndarray] = []
    endpoint_memory_receipts: list[dict[str, Any]] = []
    endpoint_memory_residual = np.zeros(risk_length, dtype=np.float32)
    memory_running_peak = [-float("inf")]
    if int(endpoint_memory_stride) <= 0:
        raise ValueError("endpoint memory stride must be positive")
    window_clock = (
        value_module.PrequentialWindowClock()
        if enable_intra and strict_window_active else None
    )
    window_scheduler = (
        value_module.CausalWindowUpdateScheduler(
            max_updates=int(adaptation_events),
            **dict(window_scheduler_options or {}),
        )
        if enable_intra and strict_window_active else None
    )
    if endpoint_memory is not None:
        endpoint_memory.begin_shot(state_id)

    def score_risk_positions(risk_positions: np.ndarray) -> None:
        positions = np.asarray(risk_positions, dtype=np.int64)
        if not len(positions):
            return
        dynamic = positions - dynamic_offset
        if np.any(dynamic < 0):
            # The first lag has no legal previous Transformer window.  It still
            # receives the causal raw8 branch, but never a fabricated hidden delta.
            private_batch = make_private_batch(
                raw, raw_available, positions, shared_logit,
            )
            private_probability = torch.sigmoid(
                private_controller.predict(private_batch)
            ).numpy()
            combined_logit[positions] = np.log(
                np.clip(private_probability, 1e-6, 1 - 1e-6)
                / np.clip(1 - private_probability, 1e-6, 1 - 1e-6)
            )
        else:
            shared_batch = two_runner.make_batch(
                frozen, base_logit_dynamic, physics_at_endpoint,
                available_at_endpoint, dynamic,
            )
            predicted_shared = shared_controller.predict(shared_batch).numpy()
            shared_logit[positions] = predicted_shared
            private_batch = make_private_batch(
                raw, raw_available, positions, shared_logit,
            )
            combined_logit[positions] = private_controller.predict(private_batch).numpy()
        for position in positions:
            endpoint = alarm.W - 1 + int(position)
            time_s = float(start_time_s) + float(endpoint) / float(sample_rate_hz)
            times[position] = time_s
            current_base_logit = float(combined_logit[position])
            is_running_peak = current_base_logit > memory_running_peak[0]
            memory_running_peak[0] = max(memory_running_peak[0], current_base_logit)
            support_for_memory = (
                float(physics_support[endpoint])
                if bool(physics_known[endpoint]) else 0.0
            )
            margin_for_memory = current_base_logit - threshold_logit
            memory_eligible = (
                int(position) % int(endpoint_memory_stride) == 0
                or is_running_peak
                or margin_for_memory >= -(
                    endpoint_memory.max_positive_residual + 0.10
                ) if endpoint_memory is not None else False
            )
            if (
                endpoint_memory is not None
                and int(position) >= dynamic_offset
                and memory_eligible
            ):
                dynamic_index = int(position) - dynamic_offset
                memory_result = endpoint_memory.correct_and_seal(
                    state_id, endpoint_index=int(position),
                    key=endpoint_memory_key(
                        frozen_state=frozen[dynamic_index],
                        risk_logit=combined_logit, risk_index=int(position),
                        physics_support=support_for_memory,
                        physics_known=bool(physics_known[endpoint]),
                        raw=raw, raw_available=raw_available,
                        raw_endpoint=endpoint,
                    ),
                    base_logit=float(combined_logit[position]),
                    physics_support=support_for_memory,
                )
                combined_logit[position] = float(memory_result["corrected_logit"])
                endpoint_memory_residual[position] = float(memory_result["residual"])
                if float(memory_result["residual"]) != 0.0:
                    endpoint_memory_receipts.append({
                        "event": "typed_endpoint_residual", "shot": int(shot_id),
                        "risk_index": int(position),
                        "residual": float(memory_result["residual"]),
                        "positive_error_shots": int(
                            memory_result["positive_error_shots"]
                        ),
                        "negative_error_shots": int(
                            memory_result["negative_error_shots"]
                        ),
                        "tp_protection": float(memory_result["tp_protection"]),
                        "tn_protection": float(memory_result["tn_protection"]),
                    })
            if window_clock is not None:
                window_clock.seal_prediction(int(position))
            alarm_controller.observe_endpoint(
                state_id, float(combined_logit[position] - threshold_logit),
                float(physics_support[endpoint]), time_s,
            )

    score_risk_positions(np.arange(dynamic_offset, dtype=np.int64))
    legal_current = np.arange(dha.LAG, len(endpoints), dtype=np.int64)

    def adapt_at_current(
        current: int, *, scheduler_decision: dict[str, Any] | None = None,
    ) -> None:
        source = int(current - dha.LAG)
        source_risk = int(dynamic_offset + source)
        current_risk = int(dynamic_offset + current)
        if window_clock is not None:
            window_clock.permit_update(current_risk)
        safe_next = np.minimum(endpoints + dha.LAG, len(full_physics) - 1)
        shared_source = two_runner.make_batch(
            frozen, base_logit_dynamic, physics_at_endpoint,
            available_at_endpoint, np.asarray([source]),
            next_physics=full_physics[safe_next],
            next_available=full_available[safe_next],
        )
        private_source = make_private_batch(
            raw, raw_available, np.asarray([source_risk]), shared_logit,
            next_indices=np.asarray([current_risk]),
        )
        if shared_controller.intra_parameter_scope == "representation_only":
            shared_source["sealed_pre_adapt_logit"] = torch.tensor(
                [float(shared_logit[source_risk])], dtype=torch.float32,
            )
        if private_controller.intra_parameter_scope == "representation_only":
            private_source["sealed_pre_adapt_logit"] = torch.tensor(
                [float(combined_logit[source_risk])], dtype=torch.float32,
            )
        shared_observed_active = (
            shared_controller.temporal_risk_weight > 0.0
            or shared_controller.observed_physics_weight > 0.0
        )
        private_observed_active = (
            private_controller.temporal_risk_weight > 0.0
        )
        observed_active = shared_observed_active or private_observed_active
        shared_observed = None
        private_observed = None
        if observed_active:
            if not np.isfinite(times[current_risk]):
                raise RuntimeError(
                    "current endpoint prediction must be sealed before temporal update"
                )
            if shared_observed_active:
                shared_observed = two_runner.make_batch(
                    frozen, base_logit_dynamic, physics_at_endpoint,
                    available_at_endpoint, np.asarray([current]),
                )
            if private_observed_active:
                if shared_observed is None:
                    shared_for_private = two_runner.make_batch(
                        frozen, base_logit_dynamic, physics_at_endpoint,
                        available_at_endpoint, np.asarray([current]),
                    )
                else:
                    shared_for_private = shared_observed
                persistent_shared_current = float(
                    shared_controller.predict_persistent(shared_for_private)[0]
                )
                persistent_shared_trace = shared_logit.copy()
                persistent_shared_trace[current_risk] = persistent_shared_current
                private_observed = make_private_batch(
                    raw, raw_available, np.asarray([current_risk]),
                    persistent_shared_trace,
                )
        shared_error, shared_disagreement = _masked_shared_error(
            shared_controller, shared_source,
        )
        private_error, private_disagreement = _private_error(
            private_controller, private_source,
        )
        reconstruction = float(np.clip(0.5 * (
            shared_error / shared_error_scale + private_error / private_error_scale
        ), 0, 1))
        teacher = float(np.clip(
            2.0 * max(shared_disagreement, private_disagreement), 0, 1,
        ))
        endpoint = int(endpoints[current])
        support = float(physics_support[endpoint]) if physics_known[endpoint] else 0.0
        probability = float(torch.sigmoid(torch.tensor(combined_logit[source_risk])))
        physics_mismatch = max(probability * (1.0 - support), (1.0 - probability) * support)
        private_innovation = float(np.clip(private_error / private_error_scale, 0, 1))
        raw_fraction = float(raw_available[endpoint].mean())
        physics_fraction = float(available_at_endpoint[current].mean())
        availability_fraction = min(raw_fraction, max(physics_fraction, 0.25))
        state_vector = np.concatenate([
            physics_at_endpoint[current].astype(np.float64),
            private_source["next_features"][0].numpy().astype(np.float64),
        ])
        redundancy = redundancy_score(state_vector, recent_states)
        recent_states.append(state_vector)
        base_value = update_value.score(
            reconstruction_surprise=reconstruction,
            teacher_disagreement=teacher,
            physics_mismatch=physics_mismatch,
            private_innovation=private_innovation,
            availability_fraction=availability_fraction,
            redundancy=redundancy,
        )
        margin = float(combined_logit[source_risk] - threshold_logit)
        alarm_proximity = 1.0 / (1.0 + math.exp(-float(np.clip(
            (margin + 8.0 * support - 7.9) / 1.5, -30, 30,
        ))))
        mechanism_gate = max(
            alarm_proximity,
            1.0 if support >= 0.65 and reconstruction >= 0.75 else 0.0,
        )
        causal_value = float(np.clip(base_value * (0.25 + 0.75 * mechanism_gate), 0, 1))
        scale = update_value.learning_rate_scale(causal_value)
        shared_receipt = shared_controller.intra_update(
            source_endpoint=int(endpoints[source]),
            observed_endpoint=endpoint,
            batch=shared_source, observed_batch=shared_observed,
            steps=intra_steps, update_scale=scale,
        )
        private_receipt = private_controller.intra_update(
            source_endpoint=int(alarm.W - 1 + source_risk),
            observed_endpoint=int(alarm.W - 1 + current_risk),
            batch=private_source, observed_batch=private_observed,
            steps=intra_steps, update_scale=scale,
        )
        receipts.append({
            "event": "unified_intra_update", "shot": int(shot_id),
            "state_shot": state_id,
            "source_risk_index": source_risk, "current_risk_index": current_risk,
            "causal_value": causal_value, "learning_rate_scale": scale,
            "reconstruction": reconstruction, "teacher_disagreement": teacher,
            "physics_mismatch": physics_mismatch,
            "private_innovation": private_innovation,
            "availability_fraction": availability_fraction,
            "redundancy": redundancy,
            "shared_updated": bool(shared_receipt["updated"]),
            "shared_pseudo_loss": float(shared_receipt["pseudo_loss"]),
            "shared_temporal_risk_loss": float(
                shared_receipt["temporal_risk_loss"]
            ),
            "shared_temporal_target_delta": float(
                shared_receipt["temporal_target_delta"]
            ),
            "shared_observed_physics_loss": float(
                shared_receipt["observed_physics_loss"]
            ),
            "shared_risk_anchor_source": shared_receipt.get(
                "risk_anchor_source"
            ),
            "temporal_target_gain": float(
                shared_controller.temporal_target_gain
            ),
            "private_updated": bool(private_receipt["updated"]),
            "private_temporal_risk_loss": float(
                private_receipt["temporal_risk_loss"]
            ),
            "private_temporal_target_delta": float(
                private_receipt["temporal_target_delta"]
            ),
            "private_risk_anchor_source": private_receipt.get(
                "risk_anchor_source"
            ),
            "strict_prediction_sealed_before_update": bool(
                window_clock is not None
                and window_clock.last_sealed >= current_risk
            ),
            "last_sealed_risk_index_before_update": (
                int(window_clock.last_sealed)
                if window_clock is not None else None
            ),
            "observed_risk_index": current_risk,
            "window_scheduler": scheduler_decision,
        })
        values.append(causal_value)
        reconstructions.append(reconstruction)
        teacher_disagreements.append(teacher)
        private_innovations.append(private_innovation)

    if not enable_intra:
        score_risk_positions(np.arange(
            dynamic_offset, risk_length, dtype=np.int64,
        ))
    elif strict_window_active:
        assert window_scheduler is not None
        # Advance the immutable-parent/physics scheduler one endpoint at a
        # time.  A selected endpoint is scored and sealed immediately before
        # its gradient is applied, so the runtime path—not only the scheduling
        # formula—enforces prefix-causal predict -> seal -> update ordering.
        next_unscored_risk = dynamic_offset
        for causal_step, current_value in enumerate(legal_current):
            current = int(current_value)
            current_risk = int(dynamic_offset + current)
            endpoint = int(endpoints[current])
            support = (
                float(physics_support[endpoint])
                if bool(physics_known[endpoint]) else 0.0
            )
            probability = float(base_risk[current_risk])
            previous_probability = float(
                base_risk[max(0, current_risk - dha.LAG)]
            )
            immutable_logit = math.log(
                np.clip(probability, 1e-6, 1 - 1e-6)
                / np.clip(1.0 - probability, 1e-6, 1 - 1e-6)
            )
            margin = float(immutable_logit - threshold_logit)
            boundary_proximity = math.exp(-abs(margin) / 1.5)
            early_proximity = 1.0 / (1.0 + math.exp(-float(np.clip(
                (margin + 8.0 * support - 7.9) / 1.5, -30, 30,
            ))))
            cheap_value = float(np.clip(max(
                boundary_proximity, early_proximity, support,
                8.0 * abs(probability - previous_probability),
            ), 0.0, 1.0))
            scheduler_decision = window_scheduler.consider(
                causal_step=causal_step, value=cheap_value,
            )
            if bool(scheduler_decision["selected"]):
                score_risk_positions(np.arange(
                    next_unscored_risk, current_risk + 1, dtype=np.int64,
                ))
                next_unscored_risk = current_risk + 1
                adapt_at_current(
                    current, scheduler_decision=scheduler_decision,
                )
        if next_unscored_risk < risk_length:
            score_risk_positions(np.arange(
                next_unscored_risk, risk_length, dtype=np.int64,
            ))
    else:
        current_events = dha.evenly_limit(legal_current, int(adaptation_events))
        next_unscored = 0
        for current_value in current_events:
            current = int(current_value)
            if current > next_unscored:
                score_risk_positions(
                    np.arange(next_unscored, current, dtype=np.int64)
                    + dynamic_offset
                )
            adapt_at_current(current)
            score_risk_positions(np.asarray([dynamic_offset + current]))
            next_unscored = int(current + 1)
        if next_unscored < len(endpoints):
            score_risk_positions(
                np.arange(next_unscored, len(endpoints), dtype=np.int64)
                + dynamic_offset
            )
    shared_seal = shared_controller.seal_shot(state_id)
    private_seal = private_controller.seal_shot(state_id)
    alarm_parameters = alarm_controller.parameters_dict()
    fusion_trace = supported_fusion_trace(
        online_logit=combined_logit,
        static_logit=static_logit_reference,
        physics_support=physics_support[alarm.W - 1:],
        threshold_logit=threshold_logit,
        alarm_parameters=alarm_parameters,
        novel_early_min_physics_support=novel_early_min_physics_support,
        novel_early_min_trend=novel_early_min_trend,
    )
    fusion_decision = supported_fusion_alarm(
        online_logit=combined_logit,
        static_logit=static_logit_reference,
        physics_support=physics_support[alarm.W - 1:],
        threshold_logit=threshold_logit,
        alarm_parameters=alarm_parameters,
        novel_early_min_physics_support=novel_early_min_physics_support,
        novel_early_min_trend=novel_early_min_trend,
    )
    episode_receipt: dict[str, Any] | None = None
    episode_context: dict[str, Any] | None = None
    decision_filter = fusion_decision
    if episode_head is not None:
        support_trace = np.asarray(
            physics_support[alarm.W - 1:], dtype=np.float64,
        )
        feature_matrix = causal_episode_feature_matrix(
            online_logit=combined_logit,
            static_logit=static_logit_reference,
            shared_logit=shared_logit,
            endpoint_memory_residual=endpoint_memory_residual,
            physics_support=support_trace,
            alarm_trace=fusion_trace["actual"],
            threshold_logit=threshold_logit,
            alarm_parameters=alarm_parameters,
            raw=raw, raw_available=raw_available,
            model_branch_mask=fusion_trace["model_branch"]["alarm_mask"],
            alarm_parameter_branch_mask=(
                fusion_trace["alarm_parameter_branch"]["alarm_mask"]
            ),
        )
        supported_mask = np.asarray(
            fusion_trace["supported_mask"], dtype=bool,
        )
        actual_trace = fusion_trace["actual"]
        routes = np.full(risk_length, None, dtype=object)
        routes[supported_mask & np.asarray(actual_trace["early_mask"], bool)] = "early"
        routes[supported_mask & np.asarray(actual_trace["high_mask"], bool)] = "high"
        online_margin = combined_logit - float(threshold_logit)
        high_excess = online_margin - float(alarm_parameters["high_threshold"])
        escape_mask = supported_mask & (
            (
                np.asarray(actual_trace["high_mask"], dtype=bool)
                & (high_excess >= float(episode_high_escape_margin))
            )
            | (support_trace >= float(episode_physics_escape))
        )
        episode_receipt = episode_head.predict_and_seal(
            state_id,
            candidate_mask=supported_mask,
            feature_matrix=feature_matrix,
            route=routes,
            escape_mask=escape_mask,
        )
        decision_filter = dict(fusion_decision)
        selected_index = episode_receipt["alarm_index"]
        decision_filter.update({
            "alarmed": bool(episode_receipt["alarmed"]),
            "alarm_index": selected_index,
            "route": episode_receipt["route"],
            "early_score": (
                float(actual_trace["early_score"][int(selected_index)])
                if selected_index is not None else None
            ),
            "filter_reason": episode_receipt["filter_reason"],
            "episode_confirmation": copy.deepcopy(episode_receipt),
            "fusion_base_alarm_index": fusion_decision["alarm_index"],
            "fusion_base_filter_reason": fusion_decision["filter_reason"],
        })
        evidence = np.maximum(
            high_excess,
            np.asarray(actual_trace["early_score"], dtype=np.float64)
            - float(alarm_parameters["early_threshold"]),
        )
        reference_index = (
            int(episode_receipt["base_alarm_index"])
            if episode_receipt["base_alarm_index"] is not None
            else int(np.argmax(evidence))
        )
        model_support = bool(
            fusion_trace["model_branch"]["alarm_mask"][reference_index]
        )
        parameter_support = bool(
            fusion_trace["alarm_parameter_branch"]["alarm_mask"][reference_index]
        )
        changeability = math.exp(-abs(float(evidence[reference_index])) / 0.75)
        stability = float(np.clip(
            0.40 * support_trace[reference_index]
            + 0.30 * float(model_support and parameter_support)
            + 0.30 * (1.0 / (1.0 + math.exp(-float(np.clip(
                evidence[reference_index], -30.0, 30.0,
            ))))),
            0.0, 1.0,
        ))
        episode_context = {
            "reference_index": reference_index,
            "phenotype": feature_matrix[reference_index].copy(),
            "decision_changeability": float(changeability),
            "fp_probability": float(episode_receipt["fp_probability"]),
            "fp_confidence": float(episode_receipt["fp_confidence"]),
            "physics_reliability": float(np.mean(
                np.asarray(physics_known[alarm.W - 1:], dtype=bool)
            )),
            "stability_risk": stability,
            "evidence": float(evidence[reference_index]),
        }
    if episode_head is not None and episode_isolate_learning_feedback:
        learning_alarm_seal = alarm_controller.seal_observed_shot(
            state_id, decision_filter=fusion_decision,
        )
        alarm_seal = deploy_episode_decision(
            learning_alarm_seal, decision_filter,
        )
    else:
        alarm_seal = alarm_controller.seal_observed_shot(
            state_id, decision_filter=decision_filter,
        )
        learning_alarm_seal = alarm_seal
    if endpoint_memory is not None:
        endpoint_memory.seal_shot(state_id)
    return {
        "risk": torch.sigmoid(torch.from_numpy(combined_logit)).numpy(),
        "combined_logit": combined_logit, "shared_logit": shared_logit,
        "alarm": alarm_seal, "learning_alarm": learning_alarm_seal,
        "times_index": times,
        "intra_receipts": receipts, "update_values": values,
        "reconstruction": reconstructions,
        "teacher_disagreement": teacher_disagreements,
        "private_innovation": private_innovations,
        "shared_seal": shared_seal, "private_seal": private_seal,
        "endpoint_memory_receipts": endpoint_memory_receipts,
        "endpoint_memory_residual": endpoint_memory_residual,
        "episode_confirmation": episode_receipt,
        "episode_query_context": episode_context,
        "window_active_selected": (
            window_scheduler.total_selected if window_scheduler is not None else None
        ),
        "state_shot_id": state_id,
        "intra_enabled": bool(enable_intra),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-shots", type=int, default=360)
    parser.add_argument("--private-support-epochs", type=int, default=8)
    parser.add_argument("--adaptation-events", type=int, default=64)
    parser.add_argument("--intra-steps", type=int, default=1)
    parser.add_argument("--inter-steps", type=int, default=2)
    parser.add_argument("--intra-lr", type=float, default=1e-4)
    parser.add_argument("--private-intra-lr", type=float, default=1e-4)
    parser.add_argument("--intra-physics-weight", type=float, default=0.10)
    parser.add_argument(
        "--intra-observed-physics-weight", type=float, default=0.0,
    )
    parser.add_argument("--intra-temporal-risk-weight", type=float, default=0.0)
    parser.add_argument(
        "--intra-temporal-target-max-delta", type=float, default=0.5,
    )
    parser.add_argument("--intra-temporal-target-gain", type=float, default=1.0)
    parser.add_argument(
        "--intra-parameter-scope",
        choices=("legacy", "representation_only"), default="legacy",
    )
    parser.add_argument("--inter-lr", type=float, default=5e-4)
    parser.add_argument("--private-inter-lr", type=float, default=2e-4)
    parser.add_argument("--alarm-lr", type=float, default=0.02)
    parser.add_argument("--query-block-size", type=int, default=20)
    parser.add_argument("--query-quota", type=int, default=4)
    parser.add_argument("--query-history-size", type=int, default=80)
    parser.add_argument("--query-quantile", type=float, default=0.80)
    parser.add_argument("--query-uncertainty-weight", type=float, default=0.65)
    parser.add_argument("--query-value-lr", type=float, default=0.05)
    parser.add_argument(
        "--timescale-mode", choices=("intra_only", "inter_only", "intra_inter"),
        default="intra_inter",
    )
    parser.add_argument(
        "--query-cost", type=float, default=0.0,
        help="minimum normalized value required before paying for a label query",
    )
    parser.add_argument(
        "--query-score-mode", choices=("current", "gradient_blend"),
        default="current",
    )
    parser.add_argument("--fixed-query-manifest", type=Path, default=None)
    parser.add_argument(
        "--selector-compute-control", action="store_true",
        help=(
            "development-only selector isolation: fixed query manifest, no "
            "intra/alarm/memory learner, and one fixed-shape shared/private "
            "supervised update for every queried shot"
        ),
    )
    parser.add_argument("--fixed-compute-current-windows", type=int, default=16)
    parser.add_argument("--fixed-compute-replay-windows", type=int, default=16)
    parser.add_argument("--order-manifest", type=Path, default=None)
    parser.add_argument(
        "--diagnostic-reorder", action="store_true",
        help="permit a non-chronological 360-shot order as an upper-bound diagnostic",
    )
    parser.add_argument(
        "--causal-error-memory", action="store_true",
        help="target queries using only similar past queried/revealed mistakes",
    )
    parser.add_argument("--error-memory-radius", type=float, default=0.35)
    parser.add_argument("--error-memory-min-support", type=int, default=2)
    parser.add_argument("--error-memory-weight", type=float, default=0.60)
    parser.add_argument("--query-initial-cutoff", type=float, default=0.50)
    parser.add_argument(
        "--query-token-capacity", type=float, default=0.0,
        help="positive value enables carry-forward token budgeting",
    )
    parser.add_argument(
        "--novel-early-min-physics-support", type=float, default=0.0,
    )
    parser.add_argument("--novel-early-min-trend", type=float, default=0.0)
    parser.add_argument(
        "--strict-window-active", action="store_true",
        help="select causal intra-shot updates without using final shot length",
    )
    parser.add_argument("--window-candidate-stride", type=int, default=8)
    parser.add_argument("--window-update-rate", type=float, default=0.25)
    parser.add_argument("--window-token-capacity", type=float, default=4.0)
    parser.add_argument("--window-history-size", type=int, default=64)
    parser.add_argument("--window-value-quantile", type=float, default=0.80)
    parser.add_argument("--window-initial-cutoff", type=float, default=0.40)
    parser.add_argument("--window-minimum-history", type=int, default=8)
    parser.add_argument("--window-minimum-gap", type=int, default=8)
    parser.add_argument(
        "--typed-endpoint-memory", action="store_true",
        help="apply prior queried FP/FN/TARDY endpoint residuals with TP/TN guards",
    )
    parser.add_argument("--endpoint-memory-radius", type=float, default=0.35)
    parser.add_argument("--endpoint-memory-min-shots", type=int, default=2)
    parser.add_argument("--endpoint-memory-positive", type=float, default=0.35)
    parser.add_argument("--endpoint-memory-negative", type=float, default=0.25)
    parser.add_argument("--endpoint-memory-physics-gate", type=float, default=0.50)
    parser.add_argument("--endpoint-memory-protection-ratio", type=float, default=1.0)
    parser.add_argument("--endpoint-memory-stride", type=int, default=16)
    parser.add_argument(
        "--episode-value-v3", action="store_true",
        help="enable learned causal FP episodes and decision-aware label value",
    )
    parser.add_argument("--episode-fp-threshold", type=float, default=0.85)
    parser.add_argument("--episode-minimum-confidence", type=float, default=0.75)
    parser.add_argument("--episode-max-extra-points", type=int, default=8)
    parser.add_argument("--episode-min-history", type=int, default=12)
    parser.add_argument("--episode-min-per-class", type=int, default=3)
    parser.add_argument("--episode-high-escape-margin", type=float, default=0.75)
    parser.add_argument("--episode-physics-escape", type=float, default=0.85)
    parser.add_argument(
        "--episode-protected-sequence", action="store_true",
        help=(
            "replace the fixed endpoint FP gate with causal delta60 episode "
            "features and a past-query class-protection boundary"
        ),
    )
    parser.add_argument("--episode-sequence-l2", type=float, default=0.30)
    parser.add_argument("--episode-protection-margin", type=float, default=0.06)
    parser.add_argument("--episode-negative-quantile", type=float, default=0.90)
    parser.add_argument("--episode-safe-quantile", type=float, default=0.25)
    parser.add_argument(
        "--episode-isolate-learning-feedback", action="store_true",
        help=(
            "score deployment through the episode gate while preserving the "
            "counterfactual pre-gate v2.1 alarm for legacy learning updates"
        ),
    )
    parser.add_argument("--replay-max-distance", type=float, default=0.35)
    parser.add_argument("--episode-maximum-points", type=int, default=64)
    parser.add_argument("--episode-shoulder-margin", type=float, default=0.50)
    parser.add_argument("--query-exploration-per-block", type=int, default=1)
    parser.add_argument(
        "--episode-query-mode", choices=("shadow", "active"), default="active",
    )
    parser.add_argument(
        "--episode-gradient-mode", choices=("legacy", "episode"),
        default="episode",
    )
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--sealed-parent-state", type=Path, default=None,
        help=(
            "load a hash-verified support-only EAST parent instead of retraining "
            "the fixed110 adapter"
        ),
    )
    parser.add_argument("--a1-dir", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument(
        "--roster-manifest", type=Path, default=None,
        help=(
            "ordered JSONL rows containing shot IDs; required to replay an "
            "archived sealed parent without regenerating A1 artifacts"
        ),
    )
    parser.add_argument("--locked-best-v8", action="store_true")
    parser.add_argument(
        "--development-search", action="store_true",
        help="mark this 360-shot replay as architecture/parameter development",
    )
    parser.add_argument("--frozen-online-config", type=Path, default=None)
    args = parser.parse_args()
    run_started_at = time.perf_counter()
    cuda_available = bool(
        hasattr(torch, "cuda") and torch.cuda.is_available()
    )
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    if sum((bool(args.locked_best_v8), bool(args.development_search), args.frozen_online_config is not None)) > 1:
        raise ValueError("locked, frozen and development modes are mutually exclusive")
    frozen_contract = None
    if args.frozen_online_config is not None:
        frozen_contract = json.loads(args.frozen_online_config.read_text())
        validate_frozen_online_arguments(args, frozen_contract)
    validate_locked_best_v8_arguments(args)
    validate_episode_v3_arguments(args)
    validate_timescale_ablation_arguments(args)
    if args.locked_best_v8 and (
        args.strict_window_active or args.typed_endpoint_memory
    ):
        raise ValueError("locked best-v8 cannot enable experimental v2 controls")
    args.out = args.out or HERE / f"unified_online_active_seed{args.seed}_v1"
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "risk_traces").mkdir()
    (args.out / "static_risk_traces").mkdir()
    if args.max_shots <= 0 or args.max_shots > 360:
        raise ValueError("max-shots must lie in [1, 360]")

    sealed_parent_requested = args.sealed_parent_state is not None
    a1_dir, adapter_dir = resolve_parent_directories(
        seed=args.seed, a1_dir=args.a1_dir, adapter_dir=args.adapter_dir,
    )
    if sealed_parent_requested and args.roster_manifest is not None:
        a1_config = None
        threshold = float("nan")
        scales = None
        roster_rows = load_jsonl(args.roster_manifest.expanduser().resolve())
        base_roster = [int(row["shot"]) for row in roster_rows]
    else:
        a1_config = json.loads((a1_dir / "config.json").read_text())
        threshold = float(a1_config["threshold"])
        scales = a1_config["innovation_scales"]
        base_roster = [
            int(row["shot"]) for row in load_jsonl(a1_dir / "seals.jsonl")
        ]
    base_roster = base_roster[:int(args.max_shots)]
    if base_roster != sorted(base_roster) or len(base_roster) != len(set(base_roster)):
        raise RuntimeError("online roster must be unique and chronological")
    roster_resolution = timescale_ablation.resolve_online_roster(
        base_roster, order_manifest=args.order_manifest,
        allow_diagnostic_reorder=args.diagnostic_reorder,
    )
    roster = roster_resolution.roster
    mode_contract = timescale_ablation.timescale_contract(args.timescale_mode)

    print(f"[UNIFIED] support seed={args.seed} device={alarm.DEV}", flush=True)
    if alarm.DEV != "cuda":
        raise RuntimeError("unified registered replay requires rl_env CUDA")
    sealed_parent_path = (
        args.sealed_parent_state.expanduser().resolve()
        if args.sealed_parent_state is not None else None
    )
    sealed_parent_payload: dict[str, Any] | None = None
    sealed_parent_lineage: dict[str, Any] | None = None
    if sealed_parent_path is not None:
        torch.set_float32_matmul_precision("highest")
        if not sealed_parent_path.is_file():
            raise FileNotFoundError(sealed_parent_path)
        fidelity_path = sealed_parent_path.parent / "parent_fidelity.json"
        if not fidelity_path.is_file():
            raise RuntimeError("sealed parent is missing parent_fidelity.json")
        source_path = Path(
            args.source_checkpoint or alarm.source_checkpoint_path(args.seed)
        ).expanduser().resolve()
        source_sha256 = a1.sha256_file(source_path)
        sealed_parent_payload = torch.load(
            sealed_parent_path, map_location=alarm.DEV, weights_only=True,
        )
        sealed_parent_lineage = validate_sealed_parent_contract(
            sealed_parent_payload, json.loads(fidelity_path.read_text()),
            seed=args.seed, source_checkpoint_sha256=source_sha256,
        )
        enc, head, source_contract = alarm.GS.load_source_checkpoint(
            source_path, channels=len(alarm.GS.RH.REBUILD),
            expected_seed=args.seed, device=alarm.DEV,
        )
        enc.load_state_dict(sealed_parent_payload["encoder"], strict=True)
        head.load_state_dict(sealed_parent_payload["head"], strict=True)
        transfer, unexpected_test = alarm.GS.load_east(include_test=False)
        if unexpected_test:
            raise RuntimeError("sealed parent loader unexpectedly opened EAST test")
        transfer = alarm.GS.transform(transfer, "base")
        med17 = source_contract["scaler_center"].cpu().numpy()
        scale17 = source_contract["scaler_scale"].cpu().numpy()
        alarm.GS.apply_scale(transfer, med17, scale17)
        threshold = float(sealed_parent_lineage["threshold"])
        scales = a1.fit_support_physics_scales(transfer)
        print(
            f"[UNIFIED] loaded sealed support-only parent {sealed_parent_path}",
            flush=True,
        )
    else:
        enc, head, transfer, med17, scale17 = alarm.train_support_only(
            args.seed, args.source_checkpoint
        )
    threshold_logit = math.log(threshold / (1.0 - threshold))
    enc.eval()
    head.eval()
    base_hash = a1.model_sha256(enc, head)
    if sealed_parent_lineage is not None:
        if base_hash != sealed_parent_lineage["expected_base_sha256"]:
            raise RuntimeError("loaded sealed parent base hash mismatch")
    elif a1_config is None or base_hash != a1_config["model_sha256"]:
        raise RuntimeError("unified base differs from sealed A1 base")
    for parameter in enc.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    support_cache = dha.build_support_cache(enc, head, transfer, scales)
    guard_rng = np.random.default_rng(args.seed + 41000)
    shared_guard = dha.to_device(dha.sample_examples(
        support_cache, guard_rng, batch_size=512, balanced=True,
    ))
    shared_parent = dynamic.DynamicHazardAdapter(
        feature_dim=dha.FEATURE_DIM, physics_dim=dha.PHYSICS_DIM, hidden=16,
        max_logit_residual=1.0, min_physics_families=2,
    ).to(alarm.DEV)
    if sealed_parent_payload is not None:
        shared_parent.load_state_dict(sealed_parent_payload["shared"], strict=True)
    else:
        checkpoint = torch.load(
            adapter_dir / "adapter.pt", map_location=alarm.DEV, weights_only=True,
        )
        if checkpoint["base_sha256"] != base_hash:
            raise RuntimeError("shared adapter parent hash mismatch")
        shared_parent.load_state_dict(checkpoint["adapter"])
    shared_parent.eval()
    minimum_positive, positive_count = positive_gate.support_positive_threshold(
        shared_parent, support_cache, 0.10,
    )
    support_ids = [int(shot["shot"]) for shot in transfer]
    median8, scale8 = fit_causal_raw8_scaler(support_ids)
    if sealed_parent_payload is not None:
        private_support = build_private_support_batches(
            enc, head, shared_parent, transfer, scales, median8, scale8,
            minimum_positive=minimum_positive,
        )
        private_parent = private_continual.EastPrivateDynamicsResidual(
            channel_count=8, hidden=16, maximum_logit_residual=0.50,
        ).to(alarm.DEV)
        private_parent.load_state_dict(sealed_parent_payload["private"], strict=True)
        private_parent.eval()
        private_contract = {
            **private_reconstruction_contract(private_parent, private_support),
            "history_final_loss": None,
            "parent_initialization": "hash_verified_sealed_support_only_state",
        }
        if auxiliary.module_sha256(shared_parent) != sealed_parent_lineage[
            "expected_shared_sha256"
        ]:
            raise RuntimeError("loaded sealed shared adapter hash mismatch")
        if auxiliary.module_sha256(private_parent) != sealed_parent_lineage[
            "expected_private_sha256"
        ]:
            raise RuntimeError("loaded sealed private adapter hash mismatch")
    else:
        print("[UNIFIED] train raw8 causal dynamics on fixed 110", flush=True)
        private_parent, private_support, private_contract = fit_private_model(
            enc, head, shared_parent, transfer, scales, median8, scale8,
            threshold=threshold, minimum_positive=minimum_positive,
            seed=args.seed, epochs=args.private_support_epochs,
        )
    private_support_all = concatenate(*private_support)
    private_guard = sample_batch(
        private_support_all, count=min(512, len(private_support_all["label"])),
        seed=args.seed + 41001,
    )
    shared_error_q80 = shared_reconstruction_scale(shared_parent, support_cache)
    shared_controller = two_scale.TwoTimescaleAdapterController(
        copy.deepcopy(shared_parent), guard_batch=shared_guard,
        intra_lr=args.intra_lr, inter_lr=args.inter_lr,
        risk_anchor_weight=0.25, guard_anchor_weight=0.25,
        max_guard_harms=0, physics_constraint_weight=0.10,
        intra_physics_weight=args.intra_physics_weight,
        observed_physics_weight=args.intra_observed_physics_weight,
        physics_increment_gate=True,
        negative_max_physics_support=0.10,
        positive_min_physics_support=0.65,
        minimum_positive_residual=minimum_positive,
        threshold_logit=threshold_logit,
        temporal_risk_weight=args.intra_temporal_risk_weight,
        temporal_target_max_delta=args.intra_temporal_target_max_delta,
        temporal_target_gain=args.intra_temporal_target_gain,
        intra_parameter_scope=args.intra_parameter_scope,
    )
    private_controller = private_continual.PrivateTwoTimescaleController(
        copy.deepcopy(private_parent), guard_batch=private_guard,
        intra_lr=args.private_intra_lr, inter_lr=args.private_inter_lr,
        threshold_logit=threshold_logit, max_guard_harms=0,
        temporal_risk_weight=args.intra_temporal_risk_weight,
        temporal_target_max_delta=args.intra_temporal_target_max_delta,
        temporal_target_gain=args.intra_temporal_target_gain,
        intra_parameter_scope=args.intra_parameter_scope,
    )
    alarm_controller = value_module.ProjectedAlarmController(
        learning_rate=args.alarm_lr, max_guard_harms=0,
    )
    # Label-free smoke calibration: values below .03 are indistinguishable
    # from routine support noise; .15 already represents the extreme tail.
    # Scaling to this interval avoids the Adam-invariant "tiny loss weight"
    # mistake while keeping a hard no-update region.
    causal_value = value_module.CausalUpdateValue(
        minimum_value=0.03, saturation_value=0.15,
    )
    episode_head = None
    if args.episode_value_v3:
        if args.episode_protected_sequence:
            episode_head = episode_value.CausalProtectedSequenceFPHead(
                base_feature_dim=len(EPISODE_FEATURE_NAMES),
                minimum_history=args.episode_min_history,
                minimum_per_class=args.episode_min_per_class,
                l2=args.episode_sequence_l2,
                safety_margin=args.episode_protection_margin,
                negative_quantile=args.episode_negative_quantile,
                safe_quantile=args.episode_safe_quantile,
                minimum_confidence=args.episode_minimum_confidence,
                max_extra_points=args.episode_max_extra_points,
            )
        else:
            episode_head = episode_value.CausalEpisodeFPHead(
                feature_dim=len(EPISODE_FEATURE_NAMES),
                minimum_history=args.episode_min_history,
                minimum_per_class=args.episode_min_per_class,
                fp_probability_threshold=args.episode_fp_threshold,
                minimum_confidence=args.episode_minimum_confidence,
                max_extra_points=args.episode_max_extra_points,
            )
    decision_query_value_model = (
        episode_value.CausalDecisionQueryValueModel(
            feature_dim=len(EPISODE_FEATURE_NAMES),
            learning_rate=args.query_value_lr,
        )
        if args.episode_value_v3 else None
    )
    query_value_model = (
        decision_query_value_model
        if args.episode_value_v3 and args.episode_query_mode == "active" else
        value_module.OperationalQueryValueModel(
            uncertainty_weight=args.query_uncertainty_weight,
            learning_rate=args.query_value_lr,
        )
    )
    error_memory = (
        value_module.CausalErrorMemory(
            radius=args.error_memory_radius,
            minimum_support=args.error_memory_min_support,
            memory_weight=args.error_memory_weight,
        )
        if args.causal_error_memory else None
    )
    endpoint_memory = (
        value_module.TypedEndpointResidualMemory(
            radius=args.endpoint_memory_radius,
            minimum_error_shots=args.endpoint_memory_min_shots,
            max_positive_residual=args.endpoint_memory_positive,
            max_negative_residual=args.endpoint_memory_negative,
            positive_physics_gate=args.endpoint_memory_physics_gate,
            protection_ratio=args.endpoint_memory_protection_ratio,
        )
        if args.typed_endpoint_memory else None
    )
    fixed_query_shots: list[int] | None = None
    fixed_query_label_oracle_used = False
    if not mode_contract.enable_query:
        query_selector = None
    elif args.fixed_query_manifest is not None:
        if args.selector_compute_control:
            timescale_ablation.require_explicit_non_oracle_query_manifest(
                args.fixed_query_manifest
            )
        fixed_query_shots = timescale_ablation.read_query_manifest(
            args.fixed_query_manifest, roster,
        )
        fixed_query_label_oracle_used = (
            timescale_ablation.query_manifest_uses_label_oracle(
                args.fixed_query_manifest
            )
        )
        query_selector = timescale_ablation.CausalFixedQuerySelector(
            roster=roster, query_shots=fixed_query_shots,
            query_rate=args.query_quota / args.query_block_size,
            capacity=args.query_token_capacity,
            block_size=args.query_block_size, quota=args.query_quota,
            query_cost=args.query_cost,
            enforce_block_quota=bool(
                args.episode_value_v3 and args.episode_query_mode == "active"
            ),
        )
    elif args.episode_value_v3 and args.episode_query_mode == "active":
        query_selector = episode_value.StratifiedRollingTokenBudgetSelector(
            query_rate=args.query_quota / args.query_block_size,
            capacity=args.query_token_capacity,
            block_size=args.query_block_size, quota=args.query_quota,
            history_size=args.query_history_size,
            quantile=args.query_quantile,
            initial_cutoff=args.query_initial_cutoff,
            exploration_per_block=args.query_exploration_per_block,
            exploration_offset=3, exploration_stride=7,
            query_cost=args.query_cost,
        )
    elif args.query_token_capacity > 0:
        query_selector = value_module.RollingTokenBudgetSelector(
            query_rate=args.query_quota / args.query_block_size,
            capacity=args.query_token_capacity,
            history_size=args.query_history_size,
            quantile=args.query_quantile,
            initial_cutoff=args.query_initial_cutoff,
            exploration_interval=2 * args.query_block_size,
            query_cost=args.query_cost,
        )
    else:
        selector_type = (
            value_module.CappedAdaptiveBudgetSelector
            if error_memory is not None else value_module.AdaptiveBlockBudgetSelector
        )
        selector_options = dict(
            block_size=args.query_block_size, quota=args.query_quota,
            history_size=args.query_history_size, quantile=args.query_quantile,
            exploration_offset=3, exploration_stride=7,
        )
        if error_memory is not None:
            selector_options.update(
                initial_cutoff=args.query_initial_cutoff,
                exploration_per_block=1,
            )
        query_selector = selector_type(**selector_options)
    static_shared = copy.deepcopy(shared_parent).eval()
    static_private = copy.deepcopy(private_parent).eval()
    static_shared_hash = auxiliary.module_sha256(static_shared)
    static_private_hash = auxiliary.module_sha256(static_private)
    phase2_initial_shared_hash = auxiliary.module_sha256(shared_controller.persistent)
    phase2_initial_private_hash = auxiliary.module_sha256(private_controller.persistent)
    if phase2_initial_shared_hash != static_shared_hash:
        raise RuntimeError("phase2 shared model did not reset to static migrated parent")
    if phase2_initial_private_hash != static_private_hash:
        raise RuntimeError("phase2 private model did not reset to static migrated parent")

    # Phase 1 is completed and sealed for the entire roster before any online
    # query controller, outcome oracle, or update path is used.
    static_rows: list[dict[str, Any]] = []
    static_writer = a1.JsonlSealWriter(args.out / "static_predictions.jsonl")
    try:
        for stream_index, shot_id in enumerate(roster):
            shot = two_runner.load_east_signal_only(shot_id, med17, scale17)
            base_risk = base_risk_for_parent(
                enc, head, shot,
                cached_path=(
                    None if sealed_parent_path is not None else
                    a1_dir / "risk_traces" / f"{shot_id}.npz"
                ),
            )
            endpoints = np.arange(
                alarm.W - 1 + dha.LAG, len(shot["v"]), dtype=np.int64
            )
            frozen, direct_logit = dha.extract_frozen_batch(enc, head, shot["v"], endpoints)
            dynamic_risk_indices = endpoints - (alarm.W - 1)
            direct_probability = torch.sigmoid(torch.from_numpy(direct_logit)).numpy()
            if np.max(np.abs(direct_probability - base_risk[dynamic_risk_indices])) > 1e-5:
                raise RuntimeError(f"shot {shot_id}: base trace mismatch")
            base_logit_dynamic = np.log(
                np.clip(base_risk[dynamic_risk_indices], 1e-6, 1 - 1e-6)
                / np.clip(1 - base_risk[dynamic_risk_indices], 1e-6, 1 - 1e-6)
            ).astype(np.float32)
            physics, physics_available = a3.physics_family_trace(shot_id, scales)
            family_scores = {
                family: physics[:, index]
                for index, family in enumerate(a3.FAMILIES)
            }
            family_available = {
                family: physics_available[:, index]
                for index, family in enumerate(a3.FAMILIES)
            }
            physics_support, _physics_known, _count = audit.combine_family_scores(
                family_scores, family_available,
            )
            raw, raw_available = causal_raw8_signal_only(shot_id)
            raw, raw_available = scale_raw8(raw, raw_available, median8, scale8)
            if len(raw) != len(shot["v"]):
                raise RuntimeError(f"shot {shot_id}: shared/raw8 timebase mismatch")

            static_logit = np.log(
                np.clip(base_risk, 1e-6, 1 - 1e-6)
                / np.clip(1 - base_risk, 1e-6, 1 - 1e-6)
            ).astype(np.float32)
            initial_index = np.arange(dha.LAG, dtype=np.int64)
            initial_private = make_private_batch(
                raw, raw_available, initial_index, static_logit,
            )
            with torch.no_grad():
                static_logit[initial_index] = static_private(
                    initial_private["raw"].to(alarm.DEV),
                    initial_private["available"].to(alarm.DEV),
                    initial_private["base_logit"].to(alarm.DEV),
                )[0].cpu().numpy()
            static_logit[dynamic_risk_indices] = static_combined_logits(
                static_shared, static_private, frozen=frozen,
                base_logit=base_logit_dynamic,
                physics=physics[endpoints],
                physics_available=physics_available[endpoints],
                raw=raw, raw_available=raw_available,
                minimum_positive=minimum_positive, risk_offset=dha.LAG,
            )
            static_alarm = dual_alarm.dual_threshold_alarm(
                static_logit - threshold_logit,
                physics_support[alarm.W - 1:],
            )
            static_time = (
                float(shot["start"])
                + (alarm.W - 1 + int(static_alarm["alarm_index"])) / float(shot["rate"])
                if static_alarm["alarmed"] else None
            )
            static_row = {
                "stream_index": stream_index,
                "shot": shot_id,
                **static_alarm,
                "alarm_time_s": static_time,
                "shared_parent_sha256": static_shared_hash,
                "private_parent_sha256": static_private_hash,
            }
            static_rows.append(static_row)
            static_writer.append(static_row)
            seal_static_phase_trace(
                args.out / "static_risk_traces" / f"{shot_id}.npz",
                shot_id=shot_id, static_logit=static_logit,
            )
            if (stream_index + 1) % 20 == 0:
                print(
                    f"[STATIC PHASE] sealed {stream_index + 1}/{len(roster)}",
                    flush=True,
                )
    finally:
        static_writer.close()
    if len(static_rows) != len(roster):
        raise RuntimeError("static phase did not seal the complete roster")
    static_predictions_hash = a1.sha256_file(args.out / "static_predictions.jsonl")
    print(
        f"[STATIC PHASE] complete sha256={static_predictions_hash}; "
        "starting reset online phase",
        flush=True,
    )

    shared_replay = two_scale.CausalReplayBuffer(capacity_shots=64)
    replay_rng = np.random.default_rng(args.seed + 42000)
    vault = audit.LazyShotOutcomeOracle(
        roster, two_runner.read_hdf5_outcome,
        allow_nonmonotonic=roster_resolution.is_reordered,
    )

    prediction_writer = a1.JsonlSealWriter(args.out / "predictions.jsonl")
    query_writer = a1.JsonlSealWriter(args.out / "query_decisions.jsonl")
    reveal_writer = a1.JsonlSealWriter(args.out / "reveals.jsonl")
    update_writer = a1.JsonlSealWriter(args.out / "updates.jsonl")
    rows: list[dict[str, Any]] = []
    counts = {
        "intra_attempts": 0, "intra_nonzero": 0,
        "shared_inter_accepted": 0, "shared_inter_rollback": 0,
        "private_inter_accepted": 0, "private_inter_rollback": 0,
        "alarm_updates": 0, "query_value_updates": 0,
        "decision_value_updates": 0,
        "interaction_vetoes": 0, "interaction_delays": 0,
        "window_active_selected": 0, "endpoint_memory_corrections": 0,
        "episode_blocks": 0, "episode_delays": 0,
        "episode_escapes": 0, "episode_updates": 0,
        "fixed_compute_queries": 0,
        "fixed_compute_shared_steps": 0,
        "fixed_compute_private_steps": 0,
        "fixed_compute_current_windows": 0,
        "fixed_compute_replay_windows": 0,
    }
    try:
        for stream_index, shot_id in enumerate(roster):
            # Stateful online components use a monotone protocol identifier.
            # The actual EAST shot remains the data/outcome key.  Separating
            # them permits an explicitly marked order diagnostic without
            # weakening any controller's monotonicity checks.
            state_shot_id = int(stream_index + 1)
            shot = two_runner.load_east_signal_only(shot_id, med17, scale17)
            base_risk = base_risk_for_parent(
                enc, head, shot,
                cached_path=(
                    None if sealed_parent_path is not None else
                    a1_dir / "risk_traces" / f"{shot_id}.npz"
                ),
            )
            endpoints = np.arange(alarm.W - 1 + dha.LAG, len(shot["v"]), dtype=np.int64)
            frozen, direct_logit = dha.extract_frozen_batch(enc, head, shot["v"], endpoints)
            dynamic_risk_indices = endpoints - (alarm.W - 1)
            direct_probability = torch.sigmoid(torch.from_numpy(direct_logit)).numpy()
            if np.max(np.abs(direct_probability - base_risk[dynamic_risk_indices])) > 1e-5:
                raise RuntimeError(f"shot {shot_id}: base trace mismatch")
            base_logit_dynamic = np.log(
                np.clip(base_risk[dynamic_risk_indices], 1e-6, 1 - 1e-6)
                / np.clip(1 - base_risk[dynamic_risk_indices], 1e-6, 1 - 1e-6)
            ).astype(np.float32)
            physics, physics_available = a3.physics_family_trace(shot_id, scales)
            family_scores = {
                family: physics[:, index]
                for index, family in enumerate(a3.FAMILIES)
            }
            family_available = {
                family: physics_available[:, index]
                for index, family in enumerate(a3.FAMILIES)
            }
            physics_support, physics_known, _count = audit.combine_family_scores(
                family_scores, family_available,
            )
            raw, raw_available = causal_raw8_signal_only(shot_id)
            raw, raw_available = scale_raw8(raw, raw_available, median8, scale8)
            if len(raw) != len(shot["v"]):
                raise RuntimeError(f"shot {shot_id}: shared/raw8 timebase mismatch")

            static_logit = load_static_phase_trace(
                args.out / "static_risk_traces" / f"{shot_id}.npz",
                expected_shot_id=shot_id,
            )
            if len(static_logit) != len(base_risk):
                raise RuntimeError(f"shot {shot_id}: sealed static trace length mismatch")

            online = score_online_shot(
                shared_controller, private_controller, alarm_controller,
                causal_value, shot_id=shot_id, base_risk=base_risk,
                endpoints=endpoints, frozen=frozen,
                base_logit_dynamic=base_logit_dynamic,
                physics_at_endpoint=physics[endpoints],
                available_at_endpoint=physics_available[endpoints],
                full_physics=physics, full_available=physics_available,
                physics_support=physics_support, physics_known=physics_known,
                raw=raw, raw_available=raw_available,
                static_logit_reference=static_logit,
                threshold_logit=threshold_logit,
                shared_error_scale=shared_error_q80,
                private_error_scale=private_contract["private_reconstruction_q80"],
                adaptation_events=args.adaptation_events,
                intra_steps=args.intra_steps,
                start_time_s=float(shot["start"]),
                sample_rate_hz=float(shot["rate"]),
                novel_early_min_physics_support=(
                    args.novel_early_min_physics_support
                ),
                novel_early_min_trend=args.novel_early_min_trend,
                strict_window_active=args.strict_window_active,
                window_scheduler_options={
                    "candidate_stride": args.window_candidate_stride,
                    "update_rate": args.window_update_rate,
                    "token_capacity": args.window_token_capacity,
                    "history_size": args.window_history_size,
                    "quantile": args.window_value_quantile,
                    "initial_cutoff": args.window_initial_cutoff,
                    "minimum_history": args.window_minimum_history,
                    "minimum_gap": args.window_minimum_gap,
                },
                endpoint_memory=endpoint_memory,
                endpoint_memory_stride=args.endpoint_memory_stride,
                episode_head=episode_head,
                episode_high_escape_margin=args.episode_high_escape_margin,
                episode_physics_escape=args.episode_physics_escape,
                episode_isolate_learning_feedback=(
                    args.episode_isolate_learning_feedback
                ),
                enable_intra=mode_contract.enable_intra,
                state_shot_id=state_shot_id,
            )
            online_alarm = online["alarm"]
            learning_alarm = online["learning_alarm"]
            raw_alarm = online_alarm.get("unfiltered_alarm", online_alarm)
            counts["interaction_vetoes"] += int(
                bool(raw_alarm["alarmed"]) and not bool(online_alarm["alarmed"])
            )
            counts["interaction_delays"] += int(
                bool(online_alarm["alarmed"])
                and raw_alarm.get("alarm_index") != online_alarm.get("alarm_index")
            )
            episode_confirmation = online["episode_confirmation"]
            if episode_confirmation is not None:
                base_episode_index = episode_confirmation["base_alarm_index"]
                selected_episode_index = episode_confirmation["alarm_index"]
                counts["episode_blocks"] += int(
                    base_episode_index is not None and selected_episode_index is None
                )
                counts["episode_delays"] += int(
                    base_episode_index is not None
                    and selected_episode_index is not None
                    and int(selected_episode_index) != int(base_episode_index)
                )
                counts["episode_escapes"] += int(
                    episode_confirmation["filter_reason"]
                    == "independent_evidence_escape"
                )
            online_time = (
                float(shot["start"]) + (alarm.W - 1 + int(online_alarm["alarm_index"])) / float(shot["rate"])
                if online_alarm["alarmed"] else None
            )
            features, query_components = query_features(
                risk=online["risk"],
                physics_support=physics_support[alarm.W - 1:],
                physics_known=physics_known[alarm.W - 1:],
                threshold=threshold, update_values=online["update_values"],
                reconstruction=online["reconstruction"],
                teacher=online["teacher_disagreement"],
                private_innovation=online["private_innovation"],
            )
            legacy_query_features = dict(features)
            decision_value_receipt: dict[str, Any] | None = None
            decision_value_score: float | None = None
            if args.episode_value_v3:
                episode_context = online["episode_query_context"]
                if episode_context is None:
                    raise RuntimeError("episode v3 did not seal a query context")
                features.update({
                    name: float(episode_context[name])
                    for name in (
                        "decision_changeability", "fp_probability",
                        "fp_confidence", "physics_reliability",
                        "stability_risk",
                    )
                })
                decision_value_score, decision_value_receipt = (
                    decision_query_value_model.score_and_seal(
                        state_shot_id, features,
                        phenotype=np.asarray(
                            episode_context["phenotype"], dtype=np.float64,
                        ),
                    )
                )
                base_query_score = (
                    float(decision_value_score)
                    if args.episode_query_mode == "active" else
                    query_value_model.score_and_seal(
                        state_shot_id, legacy_query_features,
                    )
                )
            else:
                base_query_score = query_value_model.score_and_seal(
                    state_shot_id, features,
                )
            selection_base_score = timescale_ablation.sealed_query_score(
                args.query_score_mode,
                features=features,
                base_score=base_query_score,
            )
            memory_phenotype = error_memory_phenotype(
                risk=online["risk"], threshold=threshold,
                query_features=(
                    features
                    if args.episode_query_mode == "active"
                    else legacy_query_features
                ),
            )
            if error_memory is None:
                query_score = selection_base_score
                memory_evidence: dict[str, float] = {}
            else:
                query_score, memory_evidence = error_memory.score_and_seal(
                    state_shot_id, memory_phenotype,
                    base_score=selection_base_score,
                )
            vault.seal_current(shot_id)
            if query_selector is None:
                query_decision = {
                    "selected": False, "route": None,
                    "score": float(query_score), "cutoff": None,
                    "query_cost": 0.0,
                    "net_value": float(query_score),
                    "block": stream_index // args.query_block_size,
                    "position": stream_index % args.query_block_size,
                    "history_count": 0, "exploration_position": None,
                    "tokens_after": None,
                }
                selector_input_score = float(query_score)
            elif isinstance(
                query_selector, timescale_ablation.CausalFixedQuerySelector,
            ):
                selector_input_score = float(query_score)
                query_decision = query_selector.decide(
                    score=selector_input_score, stream_index=stream_index,
                    shot_id=shot_id,
                )
            elif (
                args.episode_value_v3 and args.episode_query_mode == "active"
            ):
                selector_input_score = float(query_score)
                query_decision = query_selector.decide(
                    score=selector_input_score, stream_index=stream_index,
                )
            else:
                selector_input_score = float(query_score)
                query_decision = query_selector.decide(
                    score=selector_input_score, stream_index=stream_index,
                )
                query_decision.setdefault("query_cost", float(args.query_cost))
                query_decision.setdefault(
                    "net_value", float(query_score) - float(args.query_cost),
                )
            selected = bool(query_decision["selected"])
            row = {
                "stream_index": stream_index, "shot": shot_id,
                "state_shot": state_shot_id,
                "alarmed": bool(online_alarm["alarmed"]),
                "alarm_index": online_alarm["alarm_index"],
                "alarm_time_s": online_time, "route": online_alarm["route"],
                "parameter_version": online_alarm["parameter_version"],
                "parameters": online_alarm["parameters"],
                "interaction_filter": online_alarm.get("filter_reason"),
                "learning_alarmed": bool(learning_alarm["alarmed"]),
                "learning_alarm_index": learning_alarm["alarm_index"],
                "learning_feedback_isolated": bool(
                    args.episode_isolate_learning_feedback
                ),
                "unfiltered_alarm_index": raw_alarm.get("alarm_index"),
                "model_branch_support": online_alarm.get("model_branch_support"),
                "alarm_parameter_branch_support": online_alarm.get(
                    "alarm_parameter_branch_support"
                ),
                "score": float(np.max(online["risk"])),
                "queried": bool(selected),
                "shared_persistent_sha256_before_prediction": online["shared_seal"]["persistent_sha256"],
                "private_persistent_sha256_before_prediction": online["private_seal"]["persistent_sha256"],
                "episode_confirmation": online["episode_confirmation"],
            }
            rows.append(row)
            prediction_writer.append(row)
            query_writer.append({
                "stream_index": stream_index, "shot": shot_id,
                "query_score": query_score, "selected": bool(selected),
                "selector_input_score": selector_input_score,
                "query_cost": query_decision.get("query_cost", args.query_cost),
                "net_query_value": query_decision.get(
                    "net_value", float(query_score) - float(args.query_cost),
                ),
                "base_query_score": base_query_score,
                "selection_base_score": selection_base_score,
                "query_score_mode": args.query_score_mode,
                "decision_value_score": decision_value_score,
                "query_control_mode": (
                    args.episode_query_mode if args.episode_value_v3 else "legacy"
                ),
                "features": features, "components": query_components,
                "decision_value": decision_value_receipt,
                "episode_confirmation": online["episode_confirmation"],
                "error_memory_phenotype": memory_phenotype,
                "error_memory_evidence": memory_evidence,
                "query_value_version": query_value_model.version,
                "decision_value_version": (
                    decision_query_value_model.version
                    if decision_query_value_model is not None else None
                ),
                "cutoff": query_decision["cutoff"],
                "selection_route": query_decision["route"],
                "selection_block": query_decision["block"],
                "selection_position": query_decision["position"],
                "selection_history_count": query_decision["history_count"],
                "exploration_position": query_decision["exploration_position"],
            })
            counts["intra_attempts"] += len(online["intra_receipts"])
            counts["intra_nonzero"] += sum(
                receipt["learning_rate_scale"] > 0
                for receipt in online["intra_receipts"]
            )
            for receipt in online["intra_receipts"]:
                update_writer.append(receipt)
            counts["window_active_selected"] += int(
                online["window_active_selected"] or 0
            )
            counts["endpoint_memory_corrections"] += len(
                online["endpoint_memory_receipts"]
            )
            for receipt in online["endpoint_memory_receipts"]:
                update_writer.append(receipt)
            np.savez_compressed(
                args.out / "risk_traces" / f"{shot_id}.npz",
                online_risk=online["risk"], static_risk=torch.sigmoid(
                    torch.from_numpy(static_logit)
                ).numpy(),
                online_shared_risk=torch.sigmoid(
                    torch.from_numpy(online["shared_logit"])
                ).numpy(),
                online_private_logit_residual=(
                    online["combined_logit"] - online["shared_logit"]
                    - online["endpoint_memory_residual"]
                ),
                endpoint_memory_logit_residual=online["endpoint_memory_residual"],
                physics_support=physics_support[alarm.W - 1:],
            )

            if not selected:
                shared_receipt = shared_controller.discard_without_reveal(state_shot_id)
                private_receipt = private_controller.discard_without_reveal(state_shot_id)
                alarm_receipt = alarm_controller.discard_without_reveal(state_shot_id)
                query_value_model.discard_without_reveal(state_shot_id)
                if (
                    decision_query_value_model is not None
                    and decision_query_value_model is not query_value_model
                ):
                    decision_query_value_model.discard_without_reveal(state_shot_id)
                if episode_head is not None:
                    episode_head.discard_without_reveal(state_shot_id)
                if error_memory is not None:
                    error_memory.discard_without_reveal(state_shot_id)
                if endpoint_memory is not None:
                    endpoint_memory.discard_without_reveal(state_shot_id)
                vault.discard_unqueried(shot_id)
                update_writer.append({
                    "event": "unified_unqueried_finalize", "shot": shot_id,
                    "state_shot": state_shot_id,
                    "shared": shared_receipt, "private": private_receipt,
                    "alarm": alarm_receipt,
                    "episode_confirmation": episode_confirmation,
                })
            else:
                outcome = vault.reveal_for_query(shot_id, selected=True)
                reveal_writer.append({
                    "stream_index": stream_index, "shot": shot_id,
                    "revealed_is_disrupt": bool(outcome["is_disrupt"]),
                    "revealed_down_time_s": outcome["down_time_s"],
                })
                revealed_shot = {
                    **shot, "is_disrupt": bool(outcome["is_disrupt"]),
                    "down": outcome["down_time_s"],
                }
                labels, mechanism_target, operational_target, timely_labels = (
                    operational_temporal_targets(revealed_shot, len(base_risk))
                )
                alarm_boundary_logit = sealed_alarm_boundary_logit(
                    online["combined_logit"],
                    physics_support=physics_support[alarm.W - 1:],
                    threshold_logit=threshold_logit,
                    alarm_parameters=online_alarm["parameters"],
                )
                supervised_positive = (
                    timely_labels if timely_labels.any() else labels
                )
                discrepancy_groups = causal_discrepancy_groups(
                    deployed_alarm=online_alarm,
                    learning_alarm=learning_alarm,
                    times_s=online["times_index"],
                    is_disrupt=bool(outcome["is_disrupt"]),
                    down_time_s=outcome["down_time_s"],
                )
                deployed_group = discrepancy_groups["deployed"]
                group = discrepancy_groups["learning"]
                learning_time = (
                    float(shot["start"])
                    + (
                        alarm.W - 1 + int(learning_alarm["alarm_index"])
                    ) / float(shot["rate"])
                    if learning_alarm["alarmed"] else None
                )
                learning_lead_ms = (
                    (
                        float(outcome["down_time_s"])
                        - float(learning_time)
                    ) * 1000.0
                    if outcome["is_disrupt"] and learning_time is not None
                    else None
                )
                deployed_lead_ms = (
                    (
                        float(outcome["down_time_s"])
                        - float(online_time)
                    ) * 1000.0
                    if outcome["is_disrupt"] and online_time is not None
                    else None
                )
                hdf5_value = supervision_value(
                    online["risk"], operational_target,
                    group=group, lead_ms=learning_lead_ms,
                )
                deployed_hdf5_value = supervision_value(
                    online["risk"], operational_target,
                    group=deployed_group, lead_ms=deployed_lead_ms,
                )
                if (
                    args.episode_value_v3
                    and args.episode_gradient_mode == "episode"
                ):
                    hdf5_value = episode_supervision_scale(
                        hdf5_value, discrepancy_group=group,
                        episode_confidence=float(
                            episode_confirmation["fp_confidence"]
                            if episode_confirmation is not None else 0.0
                        ),
                    )
                alarm_receipt = (
                    alarm_controller.finalize_revealed_no_update(
                        state_shot_id, discrepancy_group=group,
                    )
                    if args.selector_compute_control else
                    alarm_controller.update_after_reveal(
                        state_shot_id, selected=True,
                        is_disrupt=bool(outcome["is_disrupt"]),
                        down_time_s=outcome["down_time_s"],
                        update_value=hdf5_value,
                    )
                )
                if str(alarm_receipt["discrepancy_group"]) != group:
                    raise RuntimeError("sealed alarm group changed during reveal update")
                episode_update_receipt = (
                    episode_head.update_after_reveal(
                        state_shot_id, selected=True,
                        is_disrupt=bool(outcome["is_disrupt"]),
                        discrepancy_group=deployed_group,
                    )
                    if episode_head is not None else None
                )
                counts["episode_updates"] += int(
                    bool(episode_update_receipt)
                    and bool(episode_update_receipt["updated"])
                )
                query_value_receipt = (
                    query_value_model.finalize_revealed_no_update(state_shot_id)
                    if args.selector_compute_control else
                    (
                        None
                        if args.episode_value_v3
                        and args.episode_query_mode == "active" else
                        query_value_model.update_after_reveal(
                            state_shot_id, selected=True, discrepancy_group=group,
                        )
                    )
                )
                error_memory_receipt = (
                    error_memory.update_after_reveal(
                        state_shot_id, selected=True, discrepancy_group=group,
                    )
                    if error_memory is not None else None
                )
                if query_value_receipt is not None:
                    counts["query_value_updates"] += int(
                        query_value_receipt["updated"]
                    )
                counts["alarm_updates"] += int(alarm_receipt["updated"])
                inter_scale = float(np.clip(hdf5_value, 0, 1))
                controller_inter_steps, controller_inter_scale = inter_update_control(
                    args.inter_steps, inter_scale,
                )
                if args.selector_compute_control:
                    controller_inter_steps = int(args.inter_steps)
                    controller_inter_scale = 1.0
                similar_error = None
                endpoint_memory_indices: list[int] | None = None
                if args.selector_compute_control or group in ("FP", "FN", "TARDY"):
                    if args.selector_compute_control:
                        legal = fixed_compute_supervised_indices(
                            risk=online["risk"], positive=supervised_positive,
                            legal_start=dha.LAG,
                            legal_stop=len(base_risk) - dha.LAG,
                            count=args.fixed_compute_current_windows,
                        )
                    elif (
                        args.episode_value_v3
                        and args.episode_gradient_mode == "episode"
                        and group == "FP"
                        and online_alarm.get("alarm_index") is not None
                    ):
                        chosen_risk = select_fp_episode_indices(
                            evidence=(
                                online["combined_logit"] - alarm_boundary_logit
                            ),
                            alarm_index=int(online_alarm["alarm_index"]),
                            maximum_points=args.episode_maximum_points,
                            shoulder_margin=args.episode_shoulder_margin,
                        )
                    else:
                        chosen_risk = select_supervised_indices(
                            risk=online["risk"], labels=labels,
                            timely_labels=timely_labels,
                            margin=online["combined_logit"] - threshold_logit,
                            alarm_boundary_margin=(
                                alarm_boundary_logit - threshold_logit
                            ),
                            physics_support=physics_support[alarm.W - 1:],
                            alarm_parameters=online_alarm["parameters"], group=group,
                        )
                    if not args.selector_compute_control:
                        legal = chosen_risk[
                            (chosen_risk >= dha.LAG)
                            & (chosen_risk < len(base_risk) - dha.LAG)
                        ]
                    if not len(legal):
                        legal_pool = np.arange(dha.LAG, len(base_risk) - dha.LAG)
                        error = np.abs(
                            operational_target[legal_pool]
                            - online["risk"][legal_pool]
                        )
                        legal = np.sort(legal_pool[np.argsort(error)[-16:]])
                    chosen_dynamic = legal - dha.LAG
                    endpoint_memory_indices = [int(index) for index in legal]
                    safe_next = np.minimum(endpoints + dha.LAG, len(physics) - 1)
                    shared_current = two_runner.make_batch(
                        frozen, base_logit_dynamic, physics[endpoints],
                        physics_available[endpoints], chosen_dynamic,
                        next_physics=physics[safe_next],
                        next_available=physics_available[safe_next],
                        labels=supervised_positive[dha.LAG:],
                    )
                    shared_current["alarm_boundary"] = torch.from_numpy(
                        alarm_boundary_logit[legal].astype(np.float32)
                    )
                    phenotype_vector = [
                        memory_phenotype[name] for name in sorted(memory_phenotype)
                    ]
                    if args.selector_compute_control:
                        similar_error = None
                        opposite = None
                        replay_count = int(args.fixed_compute_replay_windows)
                    else:
                        similar_error = shared_replay.sample_similar_error_bag(
                            discrepancy_group=group, phenotype=phenotype_vector,
                            count=4, seed=args.seed * 100000 + stream_index + 17,
                            max_distance=(
                                args.replay_max_distance
                                if args.episode_value_v3
                                and args.episode_gradient_mode == "episode" else None
                            ),
                        )
                        opposite = shared_replay.sample_opposite_bag(
                            current_is_disruptive=bool(outcome["is_disrupt"]),
                            count=4 if similar_error is not None else 8,
                            seed=args.seed * 100000 + stream_index,
                        )
                        if similar_error is not None:
                            opposite = (
                                similar_error["batch"] if opposite is None
                                else concatenate(opposite, similar_error["batch"])
                            )
                        replay_count = 16 if opposite is None else 8
                    replay_np = dha.sample_examples(
                        support_cache, replay_rng,
                        batch_size=replay_count, balanced=True,
                    )
                    shared_support = {
                        name: torch.from_numpy(value) for name, value in replay_np.items()
                    }
                    shared_receipt = shared_controller.reveal_and_inter_update(
                        state_shot_id, selected=True, current_batch=shared_current,
                        replay_batch=shared_support, opposite_batch=opposite,
                        steps=controller_inter_steps,
                        update_scale=controller_inter_scale,
                        fixed_compute_objective=bool(
                            args.selector_compute_control
                        ),
                    )
                    counts[
                        "shared_inter_accepted" if shared_receipt["accepted"]
                        else "shared_inter_rollback"
                    ] += 1
                    if not args.selector_compute_control:
                        shared_replay.add(
                            state_shot_id, shared_current, discrepancy_group=group,
                            phenotype=phenotype_vector,
                        )
                    private_current = make_private_batch(
                        raw, raw_available, legal, online["shared_logit"],
                        target=operational_target,
                        sealed_logit=online["combined_logit"],
                        labels=supervised_positive,
                        alarm_boundary=alarm_boundary_logit,
                        next_indices=np.minimum(legal + dha.LAG, len(base_risk) - 1),
                    )
                    private_replay = sample_batch(
                        private_support_all,
                        count=(
                            args.fixed_compute_replay_windows
                            if args.selector_compute_control else 16
                        ),
                        seed=(
                            args.seed * 100000 + counts["fixed_compute_queries"]
                            if args.selector_compute_control else
                            args.seed * 100000 + stream_index
                        ),
                    )
                    private_receipt = private_controller.reveal_and_inter_update(
                        state_shot_id, selected=True, current_batch=private_current,
                        replay_batch=private_replay, steps=controller_inter_steps,
                        update_scale=controller_inter_scale,
                        fixed_compute_objective=bool(
                            args.selector_compute_control
                        ),
                    )
                    counts[
                        "private_inter_accepted" if private_receipt["accepted"]
                        else "private_inter_rollback"
                    ] += 1
                    if args.selector_compute_control:
                        counts["fixed_compute_queries"] += 1
                        counts["fixed_compute_shared_steps"] += int(
                            shared_receipt["steps"]
                        )
                        counts["fixed_compute_private_steps"] += int(
                            private_receipt["steps"]
                        )
                        counts["fixed_compute_current_windows"] += len(legal)
                        counts["fixed_compute_replay_windows"] += replay_count
                else:
                    shared_receipt = shared_controller.finalize_revealed_protection(
                        state_shot_id, group=group,
                    )
                    private_receipt = private_controller.finalize_revealed_protection(
                        state_shot_id, group=group,
                    )
                    protected_index = (
                        int(learning_alarm["alarm_index"])
                        if learning_alarm.get("alarm_index") is not None
                        else int(np.argmax(online["risk"]))
                    )
                    protected_index = int(np.clip(
                        protected_index, dha.LAG, len(base_risk) - 1,
                    ))
                    endpoint_memory_indices = [protected_index]
                endpoint_memory_receipt = (
                    endpoint_memory.update_after_reveal(
                        state_shot_id, selected=True, discrepancy_group=group,
                        endpoint_indices=endpoint_memory_indices or [dha.LAG],
                    )
                    if endpoint_memory is not None else None
                )
                realized_query_utility = None
                decision_query_value_update_receipt = None
                if args.episode_value_v3:
                    realized_query_utility = decision_query_realized_utility(
                        discrepancy_group=deployed_group,
                        supervision_value=deployed_hdf5_value,
                        episode_updated=bool(
                            episode_update_receipt
                            and episode_update_receipt["updated"]
                        ),
                        shared_accepted=bool(shared_receipt.get("accepted", False)),
                        private_accepted=bool(private_receipt.get("accepted", False)),
                        novelty=float(
                            decision_value_receipt["novelty"]
                            if decision_value_receipt is not None else 0.0
                        ),
                    )
                    decision_query_value_update_receipt = (
                    decision_query_value_model.update_after_reveal(
                        state_shot_id, selected=True,
                        discrepancy_group=deployed_group,
                        realized_utility=realized_query_utility,
                    )
                    )
                    counts["decision_value_updates"] += int(
                        decision_query_value_update_receipt["updated"]
                    )
                    if args.episode_query_mode == "active":
                        query_value_receipt = decision_query_value_update_receipt
                        counts["query_value_updates"] += int(
                            query_value_receipt["updated"]
                        )
                update_writer.append({
                    "event": "unified_revealed_finalize", "shot": shot_id,
                    "state_shot": state_shot_id,
                    "group": group,
                    "learning_group": group,
                    "deployed_group": deployed_group,
                    "learning_feedback_isolated": bool(
                        args.episode_isolate_learning_feedback
                    ),
                    "hdf5_supervision_value": hdf5_value,
                    "deployed_hdf5_supervision_value": deployed_hdf5_value,
                    "inter_update_scale": inter_scale,
                    "temporal_supervision": {
                        "mechanism_target": "linear 0-125 ms progression (audit only)",
                        "operational_target": "binary 0-125 ms HDF5 horizon",
                        "fn_tardy_selection": (
                            "closest sealed dual-alarm evidence in 30-125 ms"
                        ),
                        "operational_gradient": (
                            "closest 30-125 ms sealed dual-alarm evidence; "
                            "direct logit-to-boundary MIL"
                        ),
                        "timely_positive_windows": int(timely_labels.sum()),
                        "mechanism_target_mean": float(mechanism_target.mean()),
                    },
                    "fixed_compute": (
                        {
                            "query_ordinal_one_based": int(
                                counts["fixed_compute_queries"]
                            ),
                            "current_endpoint_indices": [
                                int(index) for index in endpoint_memory_indices
                            ],
                            "current_window_count": len(
                                endpoint_memory_indices
                            ),
                            "support_replay_window_count": int(replay_count),
                            "first_effect_stream_index_zero_based": (
                                int(stream_index + 1)
                                if stream_index + 1 < len(roster) else None
                            ),
                            "first_effect_shot": (
                                int(roster[stream_index + 1])
                                if stream_index + 1 < len(roster) else None
                            ),
                            "current_shot_prediction_precedes_update": True,
                        }
                        if args.selector_compute_control else None
                    ),
                    "shared": shared_receipt, "private": private_receipt,
                    "alarm": alarm_receipt, "query_value": query_value_receipt,
                    "decision_query_value_update": (
                        decision_query_value_update_receipt
                    ),
                    "decision_query_realized_utility": realized_query_utility,
                    "episode_confirmation": episode_confirmation,
                    "episode_update": episode_update_receipt,
                    "error_memory": error_memory_receipt,
                    "typed_endpoint_memory": endpoint_memory_receipt,
                    "same_error_replay": (
                        {
                            "source_shot": similar_error["source_shot"],
                            "distance": similar_error["distance"],
                            "group": group,
                        }
                        if group in ("FP", "FN", "TARDY")
                        and similar_error is not None else None
                    ),
                })
            if (stream_index + 1) % 20 == 0:
                selected_count = (
                    query_selector.total_selected
                    if query_selector is not None else 0
                )
                print(
                    f"[UNIFIED] sealed {stream_index + 1}/{len(roster)} "
                    f"queries={selected_count}", flush=True,
                )
    finally:
        prediction_writer.close()
        query_writer.close()
        reveal_writer.close()
        update_writer.close()

    query_count = query_selector.total_selected if query_selector is not None else 0
    expected_queries = (
        expected_query_count(
            len(roster), block_size=args.query_block_size, quota=args.query_quota,
        )
        if mode_contract.enable_query else 0
    )
    if not mode_contract.enable_query and query_count != 0:
        raise RuntimeError("intra-only mode accessed the label-query budget")
    if (
        mode_contract.enable_query and args.episode_value_v3
        and query_count > expected_queries
    ):
        raise RuntimeError("episode v3 causal query cap exceeded")
    if (
        mode_contract.enable_query and not args.episode_value_v3
        and error_memory is None and args.fixed_query_manifest is None
        and query_count != expected_queries
    ):
        raise RuntimeError("unified exact query budget failed")
    if (
        mode_contract.enable_query and not args.episode_value_v3
        and error_memory is not None and query_count > expected_queries
    ):
        raise RuntimeError("causal error-memory query cap exceeded")
    if fixed_query_shots is not None and query_count != len(fixed_query_shots):
        raise RuntimeError("fixed query manifest was not consumed exactly")
    if args.selector_compute_control:
        expected_current = query_count * int(args.fixed_compute_current_windows)
        expected_replay = query_count * int(args.fixed_compute_replay_windows)
        expected_steps = query_count * int(args.inter_steps)
        fixed_contract = {
            "queries": counts["fixed_compute_queries"] == query_count,
            "shared_steps": counts["fixed_compute_shared_steps"] == expected_steps,
            "private_steps": counts["fixed_compute_private_steps"] == expected_steps,
            "current_windows": counts["fixed_compute_current_windows"] == expected_current,
            "replay_windows": counts["fixed_compute_replay_windows"] == expected_replay,
            "alarm_frozen": counts["alarm_updates"] == 0,
            "intra_frozen": counts["intra_attempts"] == 0,
        }
        failed = [name for name, passed in fixed_contract.items() if not passed]
        if failed:
            raise RuntimeError(
                "selector compute control violated realized contract: "
                + ", ".join(failed)
            )
    if a1.model_sha256(enc, head) != base_hash:
        raise RuntimeError("unified online replay changed frozen J-TEXT/EAST base")
    if auxiliary.module_sha256(static_shared) != static_shared_hash:
        raise RuntimeError("immutable shared reference changed")
    if auxiliary.module_sha256(static_private) != static_private_hash:
        raise RuntimeError("immutable private reference changed")
    phase2_final_shared_hash = auxiliary.module_sha256(
        shared_controller.persistent
    )
    phase2_final_private_hash = auxiliary.module_sha256(
        private_controller.persistent
    )
    if not mode_contract.enable_intra and counts["intra_attempts"] != 0:
        raise RuntimeError("inter-only mode executed an intra-shot gradient")
    if not mode_contract.enable_inter:
        if query_count != 0 or counts["alarm_updates"] != 0:
            raise RuntimeError("intra-only mode executed a post-reveal update")
        if phase2_final_shared_hash != phase2_initial_shared_hash:
            raise RuntimeError("intra-only mode changed persistent shared weights")
        if phase2_final_private_hash != phase2_initial_private_hash:
            raise RuntimeError("intra-only mode changed persistent private weights")
    vault.open_final_evaluation()
    outcomes = vault.all_outcomes()
    results = {
        "static_unified": alarm_metrics(static_rows, outcomes),
        "online_unified": alarm_metrics(rows, outcomes),
    }
    static_groups = {
        int(row["shot"]): timescale_ablation.operational_group(
            row, outcomes[int(row["shot"])],
        )
        for row in static_rows
    }
    online_groups = {
        int(row["shot"]): timescale_ablation.operational_group(
            row, outcomes[int(row["shot"])],
        )
        for row in rows
    }
    transition_report = timescale_ablation.rescue_transition_report(
        static_groups, online_groups,
        queried_shots=[int(row["shot"]) for row in rows if row["queried"]],
        stream_order=roster,
    )
    abnormal = (
        results["online_unified"]["tpr"] < 0.60
        or results["online_unified"]["fpr"] > 0.40
    )
    torch.save({
        "shared_persistent": shared_controller.persistent.state_dict(),
        "private_persistent": private_controller.persistent.state_dict(),
        "alarm_parameters": alarm_controller.parameters_dict(),
        "query_value_weights": query_value_model.weights_dict(),
        "query_value_state": (
            query_value_model.state_dict()
            if args.episode_value_v3
            and args.episode_query_mode == "active" else None
        ),
        "decision_query_value_weights": (
            decision_query_value_model.weights_dict()
            if decision_query_value_model is not None else None
        ),
        "decision_query_value_state": (
            decision_query_value_model.state_dict()
            if decision_query_value_model is not None else None
        ),
        "query_selector_state": (
            query_selector.state_dict()
            if query_selector is not None
            and hasattr(query_selector, "state_dict") else None
        ),
        "episode_fp_head": (
            episode_head.state_dict() if episode_head is not None else None
        ),
        "episode_fp_head_type": (
            type(episode_head).__name__ if episode_head is not None else None
        ),
        "episode_isolate_learning_feedback": bool(
            args.episode_isolate_learning_feedback
        ),
        "causal_error_memory": (
            error_memory.state_dict() if error_memory is not None else None
        ),
        "typed_endpoint_memory": (
            endpoint_memory.state_dict() if endpoint_memory is not None else None
        ),
        "source_base_sha256": base_hash,
    }, args.out / "final_online_state.pt")
    summary = {
        "status": (
            "ABNORMAL_AUDIT_REQUIRED" if abnormal
            else roster_resolution.claim_status
            if roster_resolution.is_reordered else
            "QUERY_ORACLE_DIAGNOSTIC" if fixed_query_label_oracle_used else
            "ONLINE_DIAGNOSTIC"
        ),
        "stage": (
            "STATIC360_THEN_RESET_ONLINE360_DEVELOPMENT_SEARCH"
            if args.development_search else
            "STATIC360_THEN_RESET_ONLINE360_FROZEN_ARCHITECTURE"
            if frozen_contract is not None else
            "STATIC360_THEN_RESET_ONLINE360_LOCKED_BEST_V8"
        ),
        "seed": args.seed, "stream_shots": len(roster),
        "timescale_mode": args.timescale_mode,
        "timescale_contract": {
            "enable_intra": mode_contract.enable_intra,
            "enable_query": mode_contract.enable_query,
            "enable_reveal": mode_contract.enable_reveal,
            "enable_inter": mode_contract.enable_inter,
        },
        "selector_compute_control": (
            {
                "enabled": True,
                "claim_status": "BALANCED360_SELECTOR_ROSTER_DEVELOPMENT_CONTROL",
                "estimand": (
                    "archived active query roster versus label-free random "
                    "rosters under a new fixed-compute assimilation operator"
                ),
                "current_windows_per_query": args.fixed_compute_current_windows,
                "replay_windows_per_query": args.fixed_compute_replay_windows,
                "shared_steps_per_query": args.inter_steps,
                "private_steps_per_query": args.inter_steps,
                "fixed_learning_rate_scale": 1.0,
                "intra_frozen": True,
                "alarm_and_side_learners_frozen": True,
                "realized_contract": fixed_contract,
                "not_the_primary_full_mechanism": True,
                "not_independent_confirmation": True,
            }
            if args.selector_compute_control else {"enabled": False}
        ),
        "order_contract": {
            "is_reordered": roster_resolution.is_reordered,
            "claim_status": roster_resolution.claim_status,
            "source": roster_resolution.order_source,
            "query_selection_label_oracle_used": fixed_query_label_oracle_used,
            "chronological_main_result": (
                not roster_resolution.is_reordered
                and not fixed_query_label_oracle_used
            ),
            "reordered_result_is_upper_bound_only": roster_resolution.is_reordered,
            "query_oracle_result_is_upper_bound_only": (
                fixed_query_label_oracle_used
            ),
        },
        "execution_protocol": {
            "phase1": "complete 360-shot static inference and immutable seal",
            "phase2": "reset to identical migrated parents, then chronological online active replay",
            "phase1_static_complete_before_phase2": True,
            "hyperparameters_locked_before_phase1": bool(
                args.locked_best_v8 or frozen_contract is not None
            ),
            "selection_on_360_in_this_run": bool(args.development_search),
            "phase2_reset_to_same_migrated_initial_sha": True,
            "locked_best_v8": dict(LOCKED_BEST_V8),
            "frozen_online_config": frozen_contract,
            "parent_initialization": (
                "hash_verified_sealed_support_only_state"
                if sealed_parent_path is not None else
                "retrained_fixed110_with_current_sampler"
            ),
            "sealed_parent_state": (
                str(sealed_parent_path) if sealed_parent_path is not None else None
            ),
        },
        "query_count": query_count,
        "query_cost": args.query_cost,
        "query_score_mode": args.query_score_mode,
        "fixed_query_manifest": (
            str(args.fixed_query_manifest.resolve())
            if args.fixed_query_manifest is not None else None
        ),
        "query_cutoff": (
            f"past-only rolling q{args.query_quantile:.2f}, frozen per "
            f"{args.query_block_size}-shot block"
        ),
        "query_cutoff_source": (
            f"preceding {args.query_history_size} unlabeled sealed query scores"
        ),
        "query_selector": {
            "block_size": args.query_block_size, "quota": args.query_quota,
            "query_rate_per_shot": args.query_quota / args.query_block_size,
            "budget_semantics": (
                "hard_block_cap_plus_rolling_tokens"
                if args.episode_value_v3 and args.episode_query_mode == "active"
                else "rolling_tokens_with_carry"
            ),
            "hard_block_cap_enforced": bool(
                args.episode_value_v3 and args.episode_query_mode == "active"
            ),
            "history_size": args.query_history_size,
            "quantile": args.query_quantile, "exploration_offset": 3,
            "exploration_stride": 7,
            "mode": (
                "disabled-intra-only"
                if query_selector is None else
                "causal-fixed-manifest"
                if fixed_query_shots is not None else
                "stratified-decision-value-token-budget"
                if args.episode_value_v3
                and args.episode_query_mode == "active" else
                "legacy-v2.1-token-budget-with-decision-value-shadow"
                if args.episode_value_v3 else
                "rolling-token-budget-no-forced-fill"
                if args.query_token_capacity > 0 else
                "past-error-likelihood-cap-no-forced-fill"
                if error_memory is not None else "exact-budget-baseline"
            ),
            "token_capacity": args.query_token_capacity,
            "exploration_per_block": (
                args.query_exploration_per_block
                if args.episode_value_v3
                and args.episode_query_mode == "active" else None
            ),
        },
        "threshold": threshold, "threshold_source": "fixed110",
        "results": results, "counts": counts,
        "runtime": {
            "elapsed_wall_seconds": float(
                time.perf_counter() - run_started_at
            ),
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if cuda_available else None
            ),
            "cuda_device": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if cuda_available else None
            ),
        },
        "transition_report": transition_report,
        "value_contract": {
            "intra_components": [
                "next-state reconstruction", "persistent-teacher disagreement",
                "registered physics contradiction", "raw8 innovation",
                "availability", "non-redundancy", "alarm proximity",
            ],
            "optimizer_control": "value scales actual AdamW learning rate; fixed optimizer steps",
            "intra_value_dead_zone": causal_value.minimum_value,
            "intra_value_saturation": causal_value.saturation_value,
            "intra_physics_loss_weight": args.intra_physics_weight,
            "intra_temporal_risk_weight": args.intra_temporal_risk_weight,
            "intra_observed_physics_weight": (
                args.intra_observed_physics_weight
            ),
            "intra_temporal_target_max_delta": (
                args.intra_temporal_target_max_delta
            ),
            "intra_temporal_target_gain": args.intra_temporal_target_gain,
            "intra_parameter_scope": args.intra_parameter_scope,
            "intra_temporal_target_semantics": (
                "persistent-teacher trajectory extrapolation; not an outcome "
                "label or an HDF5-derived hazard target"
            ),
            "intra_observed_physics_semantics": (
                "after sealing the current endpoint, correct only strong "
                "causal model/physics contradictions in resettable fast weights"
            ),
            "query_components": list(query_value_model.weights_dict()),
            "query_weights_final": query_value_model.weights_dict(),
            "query_weight_bounds": (
                None
                if args.episode_value_v3
                and args.episode_query_mode == "active" else [0.55, 0.75]
            ),
            "query_initial_uncertainty_weight": args.query_uncertainty_weight,
            "query_value_learning_rate": args.query_value_lr,
            "episode_value_v3": (
                {
                    "head_type": (
                        "causal_delta60_protected_calibration"
                        if args.episode_protected_sequence
                        else "fixed_threshold_endpoint20"
                    ),
                    "fp_probability_threshold": args.episode_fp_threshold,
                    "minimum_confidence": args.episode_minimum_confidence,
                    "minimum_history": args.episode_min_history,
                    "minimum_per_class": args.episode_min_per_class,
                    "sequence_l2": (
                        args.episode_sequence_l2
                        if args.episode_protected_sequence else None
                    ),
                    "protection_margin": (
                        args.episode_protection_margin
                        if args.episode_protected_sequence else None
                    ),
                    "negative_quantile": (
                        args.episode_negative_quantile
                        if args.episode_protected_sequence else None
                    ),
                    "safe_quantile": (
                        args.episode_safe_quantile
                        if args.episode_protected_sequence else None
                    ),
                    "maximum_blocked_points_audit": args.episode_max_extra_points,
                    "high_escape_margin": args.episode_high_escape_margin,
                    "physics_escape": args.episode_physics_escape,
                    "trend_alone_can_escape": False,
                    "fp_episode_supervision_points": args.episode_maximum_points,
                    "fp_episode_shoulder_margin": args.episode_shoulder_margin,
                    "same_error_replay_max_distance": args.replay_max_distance,
                    "query_mode": args.episode_query_mode,
                    "gradient_mode": args.episode_gradient_mode,
                    "learning_feedback_isolated": bool(
                        args.episode_isolate_learning_feedback
                    ),
                    "decision_value_is_shadow_only": (
                        args.episode_query_mode == "shadow"
                    ),
                    "decision_query_weights_final": (
                        decision_query_value_model.weights_dict()
                    ),
                    "query_experts": [
                        "error_correction", "information_gain",
                        "physics_conflict", "gradient_actionability",
                        "coverage_exploration",
                    ],
                    "label_update_order": (
                        "predict-seal-query-reveal-episode/model-update-"
                        "realized-query-value-update"
                    ),
                }
                if args.episode_value_v3 else None
            ),
            "causal_error_memory": (
                {
                    "radius": args.error_memory_radius,
                    "minimum_support": args.error_memory_min_support,
                    "memory_weight": args.error_memory_weight,
                    "entries": len(error_memory.entries()),
                    "stores_only_selected_reveals": True,
                }
                if error_memory is not None else None
            ),
            "strict_window_active": (
                {
                    "candidate_stride": args.window_candidate_stride,
                    "update_rate": args.window_update_rate,
                    "token_capacity": args.window_token_capacity,
                    "history_size": args.window_history_size,
                    "quantile": args.window_value_quantile,
                    "initial_cutoff": args.window_initial_cutoff,
                    "minimum_history": args.window_minimum_history,
                    "minimum_gap": args.window_minimum_gap,
                    "max_updates_per_shot": args.adaptation_events,
                    "schedule_uses_final_shot_length": False,
                    "prediction_sealed_before_update": True,
                }
                if args.strict_window_active else None
            ),
            "typed_endpoint_memory": (
                {
                    "radius": args.endpoint_memory_radius,
                    "minimum_error_shots": args.endpoint_memory_min_shots,
                    "max_positive_residual": args.endpoint_memory_positive,
                    "max_negative_residual": args.endpoint_memory_negative,
                    "positive_physics_gate": args.endpoint_memory_physics_gate,
                    "protection_ratio": args.endpoint_memory_protection_ratio,
                    "stride": args.endpoint_memory_stride,
                    "entry_count": len(endpoint_memory.entries()),
                    "stores_only_selected_revealed_endpoints": True,
                }
                if endpoint_memory is not None else None
            ),
            "hdf5_error_cost": (
                "legacy 0.50 temporal BCE + 0.30 operation 5:3:1 + 0.20 lead gap"
                if args.episode_gradient_mode == "legacy" else
                "episode-aware legacy cost with confidence-scaled FP floor"
            ),
            "post_reveal_temporal_target": (
                "binary risk target in 0-125 ms; FN/TARDY gradients are selected "
                "from the closest sealed dual-alarm evidence in the 30-125 ms "
                "timely interval"
            ),
            "optimizer_base_learning_rates": {
                "shared_intra": args.intra_lr,
                "private_intra": args.private_intra_lr,
                "shared_inter": args.inter_lr,
                "private_inter": args.private_inter_lr,
            },
            "optimizer_steps": {
                "intra": args.intra_steps,
                "inter": args.inter_steps,
            },
        },
        "alarm_contract": {
            "formula": "margin + physics_weight*support + trend_weight*max(MA5-MA100,0)",
            "parameters_final": alarm_controller.parameters_dict(),
            "parameter_version": alarm_controller.version,
            "projection_bounds": alarm_controller.PARAMETER_BOUNDS,
            "minimum_lead_ms": 30.0,
            "over_3000ms_counts_as_tpr": True,
            "novel_early_min_physics_support": (
                args.novel_early_min_physics_support
            ),
            "novel_early_min_trend": args.novel_early_min_trend,
            "interaction_gate": (
                "combined adapted alarm requires same-endpoint support from "
                "model-only or alarm-parameter-only branch; gate can only delay/veto"
            ),
            "learned_episode_gate": (
                (
                    "queried-history delta60 FP classifier may only delay/veto "
                    "behind a separated past-query disruptive/safe boundary; "
                    "registered independent physics evidence can escape"
                    if args.episode_protected_sequence else
                    "queried-history FP classifier may only delay/veto supported "
                    "alarms; strong high-route or physics evidence can escape; "
                    "trend alone cannot"
                )
                if args.episode_value_v3 else None
            ),
        },
        "support_contract": {
            "hdf5_outcomes": {
                "disruptive": int(sum(bool(shot["is_disrupt"]) for shot in transfer)),
                "safe": int(sum(not bool(shot["is_disrupt"]) for shot in transfer)),
            },
            "raw8_scaler": "fixed110 per-shot-capped median/IQR with causal zero-sentinel availability",
            "shared_reconstruction_q80": shared_error_q80,
            **private_contract,
            "positive_residual_q10": minimum_positive,
            "positive_window_count": positive_count,
        },
        "parameter_scope": {
            "frozen": "J-TEXT/EAST rebuilt17 encoder, source/target head, shared parent anchor",
            "intra": "resettable shared adapter fast copy plus resettable raw8 dynamics fast copy",
            "inter": "shared residual head plus EAST-private dynamics/residual",
            "alarm": "projected high/physics/trend/early continuous parameters; MA widths fixed 5/100",
            "endpoint_memory": (
                "non-parametric prior queried endpoints; no backbone gradient"
                if endpoint_memory is not None else "disabled"
            ),
            "episode_fp_head": (
                (
                    "causal delta60 classifier with selected-query class-protection "
                    "calibration trained only after HDF5 reveal"
                    if args.episode_protected_sequence else
                    "small regularized causal classifier trained only after "
                    "selected HDF5 reveal"
                )
                if args.episode_value_v3 else "disabled"
            ),
            "episode_learning_feedback": (
                "deployment gate is evaluated for performance while the "
                "sealed pre-gate v2.1 alarm drives model, replay, query-value, "
                "and alarm-parameter learning"
                if args.episode_isolate_learning_feedback else
                "deployed alarm also drives subsequent learning"
                if args.episode_value_v3 else "disabled"
            ),
        },
        "hashes": {
            "source_checkpoint": a1.sha256_file(
                Path(args.source_checkpoint or alarm.source_checkpoint_path(args.seed))
                .expanduser().resolve()
            ),
            "base_before_after": base_hash,
            "static_shared": static_shared_hash,
            "static_private": static_private_hash,
            "phase2_initial_shared": phase2_initial_shared_hash,
            "phase2_initial_private": phase2_initial_private_hash,
            "phase2_final_shared": phase2_final_shared_hash,
            "phase2_final_private": phase2_final_private_hash,
            "static_predictions": static_predictions_hash,
            "predictions": a1.sha256_file(args.out / "predictions.jsonl"),
            "queries": a1.sha256_file(args.out / "query_decisions.jsonl"),
            "reveals": a1.sha256_file(args.out / "reveals.jsonl"),
            "updates": a1.sha256_file(args.out / "updates.jsonl"),
            "frozen_online_config": (
                a1.sha256_file(args.frozen_online_config)
                if args.frozen_online_config is not None else None
            ),
            "sealed_parent_state": (
                a1.sha256_file(sealed_parent_path)
                if sealed_parent_path is not None else None
            ),
        },
        "leakage_contract": (
            "phase1 static outputs for all 360 shots are sealed before phase2 starts; phase2 "
            "starts from the same migrated parent weights; endpoint outputs are emitted in "
            "order; shot prediction and dual alarm are sealed before query; HDF5 outcome is "
            "read only for selected shots; all supervised and alarm/query-parameter updates "
            "affect later shots only"
        ),
        "claim_boundary": (
            "this run is explicitly part of balanced360 architecture and parameter development; "
            "its result may select a later frozen replay but is not independent test evidence"
            if args.development_search else
            "this run uses the frozen architecture selected on balanced360; it is a deterministic "
            "development-stream replay, not an untouched future-stream confirmation"
            if frozen_contract is not None else
            "this run performs no parameter selection on balanced360 and uses a pre-locked "
            "configuration; balanced360 informed earlier method development, so this remains "
            "TEST_DRIVEN_EXPLORATION rather than untouched future-stream confirmation"
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
