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
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import dual_threshold_alarm as dual_alarm  # noqa: E402
import dynamic_hazard_adapter as dynamic  # noqa: E402
import east_private23_sidecar as private23  # noqa: E402
import east_private_continual as private_continual  # noqa: E402
import evaluate_dynamic_positive_gate as positive_gate  # noqa: E402
import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
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
    if mismatches:
        raise ValueError("locked best-v8 configuration mismatch: " + ", ".join(mismatches))


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


def causal_raw8_signal_only(shot_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Read raw8 without allowing future non-zero samples to mark the past valid."""
    data = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).resolve()
    path = data / "EAST/unified_hdf5" / str(shot_id // 100 * 100) / f"{shot_id}.hdf5"
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


def fit_private_model(
    enc: torch.nn.Module, head: torch.nn.Module,
    shared_adapter: dynamic.DynamicHazardAdapter, transfer: list[dict],
    scales: dict[str, float], median8: np.ndarray, scale8: np.ndarray,
    *, threshold: float, minimum_positive: float, seed: int, epochs: int,
) -> tuple[
    private_continual.EastPrivateDynamicsResidual,
    list[dict[str, torch.Tensor]], dict[str, float],
]:
    support_batches: list[dict[str, torch.Tensor]] = []
    for shot in transfer:
        shot_id = int(shot["shot"])
        base_risk = alarm.full_risk(enc, head, shot)
        shared_risk, _delta = dha.score_shot(
            shared_adapter, enc, head, shot, scales, base_risk,
            minimum_positive_residual=minimum_positive,
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
    return model, support_batches, {
        "private_reconstruction_q80": max(float(np.quantile(errors, 0.80)), 1e-6),
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


def supported_fusion_alarm(
    *, online_logit: np.ndarray, static_logit: np.ndarray,
    physics_support: np.ndarray, threshold_logit: float,
    alarm_parameters: dict[str, float],
) -> dict[str, Any]:
    """Reject alarms caused only by unvalidated model/parameter interaction.

    The adapted combined route may alarm only at a causal endpoint where either
    the adapted model under the immutable alarm parameters, or the immutable
    model under the adapted alarm parameters, independently alarms.  This gate
    is label-free and can only delay or veto the raw combined alarm.
    """
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
    positions = np.flatnonzero(supported)
    raw_index = int(actual_positions[0]) if len(actual_positions) else None
    if not len(positions):
        return {
            "alarmed": False, "alarm_index": None, "route": None,
            "early_score": None,
            "filter_reason": (
                "no_independent_branch_support" if raw_index is not None
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
) -> dict[str, Any]:
    shared_controller.begin_shot(shot_id)
    private_controller.begin_shot(shot_id)
    alarm_controller.begin_shot(shot_id)
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
            alarm_controller.observe_endpoint(
                shot_id, float(combined_logit[position] - threshold_logit),
                float(physics_support[endpoint]), time_s,
            )

    score_risk_positions(np.arange(dynamic_offset, dtype=np.int64))
    legal_current = np.arange(dha.LAG, len(endpoints), dtype=np.int64)
    current_events = dha.evenly_limit(legal_current, int(adaptation_events))
    next_unscored = 0
    for current in current_events:
        if current > next_unscored:
            score_risk_positions(
                np.arange(next_unscored, current, dtype=np.int64) + dynamic_offset
            )
        source = int(current - dha.LAG)
        source_risk = int(dynamic_offset + source)
        current_risk = int(dynamic_offset + current)
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
            batch=shared_source, steps=intra_steps, update_scale=scale,
        )
        private_receipt = private_controller.intra_update(
            source_endpoint=int(alarm.W - 1 + source_risk),
            observed_endpoint=int(alarm.W - 1 + current_risk),
            batch=private_source, steps=intra_steps, update_scale=scale,
        )
        receipts.append({
            "event": "unified_intra_update", "shot": int(shot_id),
            "source_risk_index": source_risk, "current_risk_index": current_risk,
            "causal_value": causal_value, "learning_rate_scale": scale,
            "reconstruction": reconstruction, "teacher_disagreement": teacher,
            "physics_mismatch": physics_mismatch,
            "private_innovation": private_innovation,
            "availability_fraction": availability_fraction,
            "redundancy": redundancy,
            "shared_updated": bool(shared_receipt["updated"]),
            "shared_pseudo_loss": float(shared_receipt["pseudo_loss"]),
            "private_updated": bool(private_receipt["updated"]),
        })
        values.append(causal_value)
        reconstructions.append(reconstruction)
        teacher_disagreements.append(teacher)
        private_innovations.append(private_innovation)
        score_risk_positions(np.asarray([current_risk]))
        next_unscored = int(current + 1)
    if next_unscored < len(endpoints):
        score_risk_positions(
            np.arange(next_unscored, len(endpoints), dtype=np.int64) + dynamic_offset
        )
    shared_seal = shared_controller.seal_shot(shot_id)
    private_seal = private_controller.seal_shot(shot_id)
    fusion_decision = supported_fusion_alarm(
        online_logit=combined_logit,
        static_logit=static_logit_reference,
        physics_support=physics_support[alarm.W - 1:],
        threshold_logit=threshold_logit,
        alarm_parameters=alarm_controller.parameters_dict(),
    )
    alarm_seal = alarm_controller.seal_observed_shot(
        shot_id, decision_filter=fusion_decision,
    )
    return {
        "risk": torch.sigmoid(torch.from_numpy(combined_logit)).numpy(),
        "combined_logit": combined_logit, "shared_logit": shared_logit,
        "alarm": alarm_seal, "times_index": times,
        "intra_receipts": receipts, "update_values": values,
        "reconstruction": reconstructions,
        "teacher_disagreement": teacher_disagreements,
        "private_innovation": private_innovations,
        "shared_seal": shared_seal, "private_seal": private_seal,
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
    parser.add_argument("--inter-lr", type=float, default=5e-4)
    parser.add_argument("--private-inter-lr", type=float, default=2e-4)
    parser.add_argument("--alarm-lr", type=float, default=0.02)
    parser.add_argument("--locked-best-v8", action="store_true")
    parser.add_argument("--a1-dir", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    args = parser.parse_args()
    validate_locked_best_v8_arguments(args)
    args.out = args.out or HERE / f"unified_online_active_seed{args.seed}_v1"
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "risk_traces").mkdir()
    (args.out / "static_risk_traces").mkdir()
    if args.max_shots <= 0 or args.max_shots > 360:
        raise ValueError("max-shots must lie in [1, 360]")

    a1_dir = args.a1_dir or HERE / f"online_active_a1_seed{args.seed}"
    adapter_dir = args.adapter_dir or HERE / f"dynamic_hazard_adapter_seed{args.seed}"
    a1_config = json.loads((a1_dir / "config.json").read_text())
    threshold = float(a1_config["threshold"])
    threshold_logit = math.log(threshold / (1.0 - threshold))
    scales = a1_config["innovation_scales"]
    roster = [int(row["shot"]) for row in load_jsonl(a1_dir / "seals.jsonl")]
    roster = roster[:int(args.max_shots)]
    if roster != sorted(roster) or len(roster) != len(set(roster)):
        raise RuntimeError("online roster must be unique and chronological")

    print(f"[UNIFIED] support seed={args.seed} device={alarm.DEV}", flush=True)
    enc, head, transfer, med17, scale17 = alarm.train_support_only(args.seed)
    if alarm.DEV != "cuda":
        raise RuntimeError("unified registered replay requires rl_env CUDA")
    base_hash = a1.model_sha256(enc, head)
    if base_hash != a1_config["model_sha256"]:
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
    checkpoint = torch.load(
        adapter_dir / "adapter.pt", map_location=alarm.DEV, weights_only=True,
    )
    if checkpoint["base_sha256"] != base_hash:
        raise RuntimeError("shared adapter parent hash mismatch")
    shared_parent = dynamic.DynamicHazardAdapter(
        feature_dim=dha.FEATURE_DIM, physics_dim=dha.PHYSICS_DIM, hidden=16,
        max_logit_residual=1.0, min_physics_families=2,
    ).to(alarm.DEV)
    shared_parent.load_state_dict(checkpoint["adapter"])
    minimum_positive, positive_count = positive_gate.support_positive_threshold(
        shared_parent, support_cache, 0.10,
    )
    support_ids = [int(shot["shot"]) for shot in transfer]
    median8, scale8 = fit_causal_raw8_scaler(support_ids)
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
        physics_increment_gate=True,
        negative_max_physics_support=0.10,
        positive_min_physics_support=0.65,
        minimum_positive_residual=minimum_positive,
        threshold_logit=threshold_logit,
    )
    private_controller = private_continual.PrivateTwoTimescaleController(
        copy.deepcopy(private_parent), guard_batch=private_guard,
        intra_lr=args.private_intra_lr, inter_lr=args.private_inter_lr,
        threshold_logit=threshold_logit, max_guard_harms=0,
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
    query_value_model = value_module.OperationalQueryValueModel(
        uncertainty_weight=0.65, learning_rate=0.05,
    )
    query_selector = value_module.AdaptiveBlockBudgetSelector(
        block_size=20, quota=4, history_size=80, quantile=0.80,
        exploration_offset=3, exploration_stride=7,
    )
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
            base_risk = np.load(
                a1_dir / "risk_traces" / f"{shot_id}.npz"
            )["risk"].astype(np.float32)
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
    vault = audit.LazyShotOutcomeOracle(roster, two_runner.read_hdf5_outcome)

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
        "interaction_vetoes": 0, "interaction_delays": 0,
    }
    try:
        for stream_index, shot_id in enumerate(roster):
            shot = two_runner.load_east_signal_only(shot_id, med17, scale17)
            base_risk = np.load(a1_dir / "risk_traces" / f"{shot_id}.npz")["risk"].astype(np.float32)
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
            )
            online_alarm = online["alarm"]
            raw_alarm = online_alarm.get("unfiltered_alarm", online_alarm)
            counts["interaction_vetoes"] += int(
                bool(raw_alarm["alarmed"]) and not bool(online_alarm["alarmed"])
            )
            counts["interaction_delays"] += int(
                bool(online_alarm["alarmed"])
                and raw_alarm.get("alarm_index") != online_alarm.get("alarm_index")
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
            query_score = query_value_model.score_and_seal(shot_id, features)
            vault.seal_current(shot_id)
            query_decision = query_selector.decide(
                score=query_score, stream_index=stream_index,
            )
            selected = bool(query_decision["selected"])
            row = {
                "stream_index": stream_index, "shot": shot_id,
                "alarmed": bool(online_alarm["alarmed"]),
                "alarm_index": online_alarm["alarm_index"],
                "alarm_time_s": online_time, "route": online_alarm["route"],
                "parameter_version": online_alarm["parameter_version"],
                "parameters": online_alarm["parameters"],
                "interaction_filter": online_alarm.get("filter_reason"),
                "unfiltered_alarm_index": raw_alarm.get("alarm_index"),
                "model_branch_support": online_alarm.get("model_branch_support"),
                "alarm_parameter_branch_support": online_alarm.get(
                    "alarm_parameter_branch_support"
                ),
                "score": float(np.max(online["risk"])),
                "queried": bool(selected),
                "shared_persistent_sha256_before_prediction": online["shared_seal"]["persistent_sha256"],
                "private_persistent_sha256_before_prediction": online["private_seal"]["persistent_sha256"],
            }
            rows.append(row)
            prediction_writer.append(row)
            query_writer.append({
                "stream_index": stream_index, "shot": shot_id,
                "query_score": query_score, "selected": bool(selected),
                "features": features, "components": query_components,
                "query_value_version": query_value_model.version,
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
                ),
                physics_support=physics_support[alarm.W - 1:],
            )

            if not selected:
                shared_receipt = shared_controller.discard_without_reveal(shot_id)
                private_receipt = private_controller.discard_without_reveal(shot_id)
                alarm_receipt = alarm_controller.discard_without_reveal(shot_id)
                query_value_model.discard_without_reveal(shot_id)
                vault.discard_unqueried(shot_id)
                update_writer.append({
                    "event": "unified_unqueried_finalize", "shot": shot_id,
                    "shared": shared_receipt, "private": private_receipt,
                    "alarm": alarm_receipt,
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
                group = value_module.ProjectedAlarmController._group(
                    online_alarm, online["times_index"],
                    is_disrupt=bool(outcome["is_disrupt"]),
                    down_time_s=outcome["down_time_s"],
                )
                lead_ms = (
                    (float(outcome["down_time_s"]) - float(online_time)) * 1000.0
                    if outcome["is_disrupt"] and online_time is not None else None
                )
                hdf5_value = supervision_value(
                    online["risk"], operational_target,
                    group=group, lead_ms=lead_ms,
                )
                alarm_receipt = alarm_controller.update_after_reveal(
                    shot_id, selected=True,
                    is_disrupt=bool(outcome["is_disrupt"]),
                    down_time_s=outcome["down_time_s"],
                    update_value=hdf5_value,
                )
                if str(alarm_receipt["discrepancy_group"]) != group:
                    raise RuntimeError("sealed alarm group changed during reveal update")
                query_value_receipt = query_value_model.update_after_reveal(
                    shot_id, selected=True, discrepancy_group=group,
                )
                counts["query_value_updates"] += int(query_value_receipt["updated"])
                counts["alarm_updates"] += int(alarm_receipt["updated"])
                inter_scale = float(np.clip(hdf5_value, 0, 1))
                if group in ("FP", "FN", "TARDY"):
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
                    opposite = shared_replay.sample_opposite_bag(
                        current_is_disruptive=bool(outcome["is_disrupt"]),
                        count=8, seed=args.seed * 100000 + stream_index,
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
                        shot_id, selected=True, current_batch=shared_current,
                        replay_batch=shared_support, opposite_batch=opposite,
                        steps=args.inter_steps, update_scale=inter_scale,
                    )
                    counts[
                        "shared_inter_accepted" if shared_receipt["accepted"]
                        else "shared_inter_rollback"
                    ] += 1
                    shared_replay.add(shot_id, shared_current)
                    private_current = make_private_batch(
                        raw, raw_available, legal, online["shared_logit"],
                        target=operational_target,
                        sealed_logit=online["combined_logit"],
                        labels=supervised_positive,
                        alarm_boundary=alarm_boundary_logit,
                        next_indices=np.minimum(legal + dha.LAG, len(base_risk) - 1),
                    )
                    private_replay = sample_batch(
                        private_support_all, count=16,
                        seed=args.seed * 100000 + stream_index,
                    )
                    private_receipt = private_controller.reveal_and_inter_update(
                        shot_id, selected=True, current_batch=private_current,
                        replay_batch=private_replay, steps=args.inter_steps,
                        update_scale=inter_scale,
                    )
                    counts[
                        "private_inter_accepted" if private_receipt["accepted"]
                        else "private_inter_rollback"
                    ] += 1
                else:
                    shared_receipt = shared_controller.finalize_revealed_protection(
                        shot_id, group=group,
                    )
                    private_receipt = private_controller.finalize_revealed_protection(
                        shot_id, group=group,
                    )
                update_writer.append({
                    "event": "unified_revealed_finalize", "shot": shot_id,
                    "group": group, "hdf5_supervision_value": hdf5_value,
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
                    "shared": shared_receipt, "private": private_receipt,
                    "alarm": alarm_receipt, "query_value": query_value_receipt,
                })
            if (stream_index + 1) % 20 == 0:
                print(
                    f"[UNIFIED] sealed {stream_index + 1}/{len(roster)} "
                    f"queries={query_selector.total_selected}", flush=True,
                )
    finally:
        prediction_writer.close()
        query_writer.close()
        reveal_writer.close()
        update_writer.close()

    expected_queries = 4 * (len(roster) // 20)
    if len(roster) % 20 == 0 and query_selector.total_selected != expected_queries:
        raise RuntimeError("unified exact query budget failed")
    if a1.model_sha256(enc, head) != base_hash:
        raise RuntimeError("unified online replay changed frozen J-TEXT/EAST base")
    if auxiliary.module_sha256(static_shared) != static_shared_hash:
        raise RuntimeError("immutable shared reference changed")
    if auxiliary.module_sha256(static_private) != static_private_hash:
        raise RuntimeError("immutable private reference changed")
    vault.open_final_evaluation()
    outcomes = vault.all_outcomes()
    results = {
        "static_unified": alarm_metrics(static_rows, outcomes),
        "online_unified": alarm_metrics(rows, outcomes),
    }
    abnormal = (
        results["online_unified"]["tpr"] < 0.60
        or results["online_unified"]["fpr"] > 0.40
    )
    torch.save({
        "shared_persistent": shared_controller.persistent.state_dict(),
        "private_persistent": private_controller.persistent.state_dict(),
        "alarm_parameters": alarm_controller.parameters_dict(),
        "query_value_weights": query_value_model.weights_dict(),
        "source_base_sha256": base_hash,
    }, args.out / "final_online_state.pt")
    summary = {
        "status": "ABNORMAL_AUDIT_REQUIRED" if abnormal else "ONLINE_DIAGNOSTIC",
        "stage": "STATIC360_THEN_RESET_ONLINE360_LOCKED_BEST_V8",
        "seed": args.seed, "stream_shots": len(roster),
        "execution_protocol": {
            "phase1": "complete 360-shot static inference and immutable seal",
            "phase2": "reset to identical migrated parents, then chronological online active replay",
            "phase1_static_complete_before_phase2": True,
            "hyperparameters_locked_before_phase1": bool(args.locked_best_v8),
            "selection_on_360_in_this_run": False,
            "phase2_reset_to_same_migrated_initial_sha": True,
            "locked_best_v8": dict(LOCKED_BEST_V8),
        },
        "query_count": query_selector.total_selected,
        "query_cutoff": "past-only rolling q80, frozen per 20-shot block",
        "query_cutoff_source": "preceding 80 unlabeled sealed query scores",
        "query_selector": {
            "block_size": 20, "quota": 4, "history_size": 80,
            "quantile": 0.80, "exploration_offset": 3,
            "exploration_stride": 7,
        },
        "threshold": threshold, "threshold_source": "fixed110",
        "results": results, "counts": counts,
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
            "query_components": list(query_value_model.weights_dict()),
            "query_weights_final": query_value_model.weights_dict(),
            "query_weight_bounds": [0.55, 0.75],
            "hdf5_error_cost": "0.50 temporal BCE + 0.30 operation 5:3:1 + 0.20 lead gap",
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
            "interaction_gate": (
                "combined adapted alarm requires same-endpoint support from "
                "model-only or alarm-parameter-only branch; gate can only delay/veto"
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
        },
        "hashes": {
            "base_before_after": base_hash,
            "static_shared": static_shared_hash,
            "static_private": static_private_hash,
            "phase2_initial_shared": phase2_initial_shared_hash,
            "phase2_initial_private": phase2_initial_private_hash,
            "static_predictions": static_predictions_hash,
            "predictions": a1.sha256_file(args.out / "predictions.jsonl"),
            "queries": a1.sha256_file(args.out / "query_decisions.jsonl"),
            "reveals": a1.sha256_file(args.out / "reveals.jsonl"),
            "updates": a1.sha256_file(args.out / "updates.jsonl"),
        },
        "leakage_contract": (
            "phase1 static outputs for all 360 shots are sealed before phase2 starts; phase2 "
            "starts from the same migrated parent weights; endpoint outputs are emitted in "
            "order; shot prediction and dual alarm are sealed before query; HDF5 outcome is "
            "read only for selected shots; all supervised and alarm/query-parameter updates "
            "affect later shots only"
        ),
        "claim_boundary": (
            "this run performs no parameter selection on balanced360 and uses the pre-locked v8 "
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
