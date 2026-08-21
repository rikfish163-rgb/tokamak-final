#!/usr/bin/env python3
"""EAST-private raw8 dynamics with strict fast/slow causal adaptation."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from east_private23_sidecar import causal_private_features, sealed_temporal_residual_loss
from two_timescale_online_adapter import operational_alarm_boundary_loss


_INTRA_OUTCOME_FIELDS = frozenset({
    "label", "target", "sealed_logit", "alarm_boundary", "alarm_target",
    "is_disrupt", "down", "down_time_s", "DownTime", "outcome",
    "revealed_is_disrupt", "revealed_down_time_s",
})


def _reject_intra_outcome_fields(values: Mapping[str, torch.Tensor]) -> None:
    forbidden = sorted(_INTRA_OUTCOME_FIELDS.intersection(values))
    if forbidden:
        raise ValueError(f"intra-shot batch contains outcome field: {forbidden[0]}")


def module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _batch(values: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value).to(device)
        for name, value in values.items()
    }


def _scale(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("update_scale must lie in [0, 1]")
    return result


@contextmanager
def _scaled_optimizer(optimizer: torch.optim.Optimizer, scale: float):
    rates = [float(group["lr"]) for group in optimizer.param_groups]
    try:
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate * scale
        yield
    finally:
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate


class EastPrivateDynamicsResidual(nn.Module):
    """Bounded raw8 risk residual plus next-causal-state reconstruction."""

    def __init__(
        self, *, channel_count: int = 8, hidden: int = 16,
        maximum_logit_residual: float = 0.50,
        minimum_available_channels: int = 2,
    ) -> None:
        super().__init__()
        if (
            channel_count <= 0 or hidden <= 0 or maximum_logit_residual <= 0
            or minimum_available_channels <= 0
        ):
            raise ValueError("invalid EAST-private dynamics dimensions")
        self.channel_count = int(channel_count)
        self.hidden = int(hidden)
        self.maximum_logit_residual = float(maximum_logit_residual)
        self.minimum_available_channels = int(minimum_available_channels)
        feature_dim = self.channel_count * 6
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.next_head = nn.Linear(hidden, feature_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self, raw: torch.Tensor, available: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = causal_private_features(raw, available)
        base = torch.as_tensor(
            base_logit, dtype=features.dtype, device=features.device,
        )
        if base.ndim != 1 or len(base) != len(features):
            raise ValueError("base logits must align with private windows")
        latent = self.trunk(torch.cat([features, base.clamp(-8, 8)[:, None]], dim=-1))
        raw_residual = self.residual_head(
            torch.cat([latent, base.clamp(-8, 8)[:, None]], dim=-1)
        ).squeeze(-1)
        channel_known = torch.as_tensor(available, device=features.device).any(dim=1)
        gate = (
            channel_known.to(torch.int64).sum(dim=-1) >= self.minimum_available_channels
        ).to(features.dtype)
        residual = self.maximum_logit_residual * torch.tanh(raw_residual) * gate
        return base + residual, residual, latent, features

    def predict_next(self, latent: torch.Tensor) -> torch.Tensor:
        return self.next_head(latent)


class PrivateTwoTimescaleController:
    """Resettable raw8 fast state and queried-label persistent state."""

    def __init__(
        self, model: EastPrivateDynamicsResidual, *,
        guard_batch: Mapping[str, torch.Tensor], intra_lr: float = 1e-4,
        inter_lr: float = 2e-4, teacher_weight: float = 0.50,
        next_weight: float = 1.0, guard_anchor_weight: float = 0.25,
        threshold_logit: float = 0.0, max_guard_harms: int = 0,
        temporal_risk_weight: float = 0.0,
        temporal_target_max_delta: float = 0.5,
        temporal_target_gain: float = 1.0,
        intra_parameter_scope: str = "legacy",
    ) -> None:
        if (
            intra_lr <= 0 or inter_lr <= 0 or teacher_weight < 0
            or next_weight < 0 or guard_anchor_weight < 0 or max_guard_harms < 0
            or not math.isfinite(float(temporal_risk_weight))
            or temporal_risk_weight < 0
            or not math.isfinite(float(temporal_target_max_delta))
            or temporal_target_max_delta <= 0
            or not math.isfinite(float(temporal_target_gain))
            or temporal_target_gain <= 0
            or intra_parameter_scope not in ("legacy", "representation_only")
        ):
            raise ValueError("invalid private two-timescale contract")
        self.persistent = model
        self.device = next(model.parameters()).device
        self.guard = _batch(guard_batch, self.device)
        if "label" not in self.guard:
            raise KeyError("private guard batch requires label")
        self.intra_lr = float(intra_lr)
        self.teacher_weight = float(teacher_weight)
        self.next_weight = float(next_weight)
        self.guard_anchor_weight = float(guard_anchor_weight)
        self.threshold_logit = float(threshold_logit)
        self.max_guard_harms = int(max_guard_harms)
        self.temporal_risk_weight = float(temporal_risk_weight)
        self.temporal_target_max_delta = float(temporal_target_max_delta)
        self.temporal_target_gain = float(temporal_target_gain)
        self.intra_parameter_scope = str(intra_parameter_scope)
        self.fast: EastPrivateDynamicsResidual | None = None
        self.fast_optimizer: torch.optim.Optimizer | None = None
        self.active_shot: int | None = None
        self.pending_shot: int | None = None
        self.last_finalized_shot: int | None = None
        for parameter in self.persistent.parameters():
            parameter.requires_grad_(True)
        self.inter_optimizer = torch.optim.AdamW(
            self.persistent.parameters(), lr=float(inter_lr), weight_decay=1e-4,
        )

    def persistent_hash(self) -> str:
        return module_hash(self.persistent)

    def fast_hash(self) -> str:
        if self.fast is None:
            raise RuntimeError("no active private fast model")
        return module_hash(self.fast)

    def begin_shot(self, shot_id: int) -> None:
        shot_id = int(shot_id)
        if self.active_shot is not None or self.pending_shot is not None:
            raise RuntimeError("previous private shot must be finalized")
        if self.last_finalized_shot is not None and shot_id <= self.last_finalized_shot:
            raise RuntimeError("private shot ids must be strictly increasing")
        self.fast = copy.deepcopy(self.persistent).to(self.device)
        for parameter in self.fast.parameters():
            parameter.requires_grad_(False)
        if self.intra_parameter_scope == "representation_only":
            prefixes = ("trunk.2", "trunk.3")
        else:
            prefixes = ("trunk", "next_head", "residual_head")
        fast_parameters = [
            parameter for name, parameter in self.fast.named_parameters()
            if name.startswith(prefixes)
        ]
        if not fast_parameters:
            raise RuntimeError("private intra-shot trainable parameter set is empty")
        for parameter in fast_parameters:
            parameter.requires_grad_(True)
        self.fast_optimizer = torch.optim.AdamW(
            fast_parameters, lr=self.intra_lr, weight_decay=1e-4,
        )
        self.active_shot = shot_id

    def _forward(
        self, model: EastPrivateDynamicsResidual,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return model(batch["raw"], batch["available"], batch["base_logit"])

    def predict(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.fast is None or self.active_shot is None:
            raise RuntimeError("begin_shot is required before private prediction")
        batch = _batch(values, self.device)
        self.fast.eval()
        with torch.no_grad():
            adjusted, _residual, _latent, _features = self._forward(self.fast, batch)
        return adjusted.detach().cpu()

    def predict_persistent(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.active_shot is None:
            raise RuntimeError("begin_shot is required before private prediction")
        batch = _batch(values, self.device)
        self.persistent.eval()
        with torch.no_grad():
            adjusted, _residual, _latent, _features = self._forward(self.persistent, batch)
        return adjusted.detach().cpu()

    @staticmethod
    def _next_loss(
        model: EastPrivateDynamicsResidual, latent: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if "next_features" not in batch:
            return latent.sum() * 0.0
        target = batch["next_features"].to(latent.dtype)
        prediction = model.predict_next(latent)
        valid = torch.isfinite(target)
        if not bool(valid.any()):
            return prediction.sum() * 0.0
        safe_target = torch.where(valid, target, prediction.detach())
        return F.smooth_l1_loss(prediction[valid], safe_target[valid])

    def intra_update(
        self, *, source_endpoint: int, observed_endpoint: int,
        batch: Mapping[str, torch.Tensor],
        observed_batch: Mapping[str, torch.Tensor] | None = None,
        steps: int = 1,
        update_scale: float = 1.0,
    ) -> dict[str, Any]:
        if self.fast is None or self.fast_optimizer is None or self.active_shot is None:
            raise RuntimeError("begin_shot is required before private intra update")
        if self.intra_parameter_scope == "representation_only":
            _reject_intra_outcome_fields(batch)
            if observed_batch is not None:
                _reject_intra_outcome_fields(observed_batch)
            if "sealed_pre_adapt_logit" not in batch:
                raise KeyError(
                    "representation-only TTT requires sealed_pre_adapt_logit"
                )
        if int(observed_endpoint) <= int(source_endpoint):
            raise ValueError("observed endpoint must be strictly later than source")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if self.temporal_risk_weight > 0.0 and observed_batch is None:
            raise ValueError(
                "observed_batch is required when temporal risk forecasting is enabled"
            )
        update_scale = _scale(update_scale)
        if update_scale == 0:
            return {
                "event": "private_intra_update", "shot": self.active_shot,
                "updated": False, "steps": 0, "loss": 0.0,
                "temporal_risk_weight": self.temporal_risk_weight,
                "temporal_target_gain": self.temporal_target_gain,
                "temporal_risk_loss": 0.0,
                "temporal_target_delta": 0.0,
                "update_scale": 0.0, "persistent_sha256": self.persistent_hash(),
                "fast_sha256": self.fast_hash(),
            }
        prepared = _batch(batch, self.device)
        observed = (
            _batch(observed_batch, self.device)
            if observed_batch is not None else None
        )
        self.persistent.eval()
        with torch.no_grad():
            teacher, _residual, _latent, _features = self._forward(
                self.persistent, prepared,
            )
            if self.intra_parameter_scope == "representation_only":
                risk_anchor_target = prepared["sealed_pre_adapt_logit"]
                if risk_anchor_target.shape != teacher.shape:
                    raise ValueError(
                        "sealed_pre_adapt_logit must align with private logits"
                    )
            else:
                risk_anchor_target = teacher
            if observed is not None and self.temporal_risk_weight > 0.0:
                observed_teacher, _residual, _latent, _features = self._forward(
                    self.persistent, observed,
                )
                if observed_teacher.shape != teacher.shape:
                    raise ValueError(
                        "observed_batch must align with the source batch"
                    )
                temporal_delta = torch.clamp(
                    self.temporal_target_gain * (observed_teacher - teacher),
                    min=-self.temporal_target_max_delta,
                    max=self.temporal_target_max_delta,
                )
                temporal_target = teacher + temporal_delta
            else:
                temporal_delta = torch.zeros_like(teacher)
                temporal_target = teacher
        losses: list[float] = []
        reconstructions: list[float] = []
        temporal_losses: list[float] = []
        self.fast.train()
        for _ in range(int(steps)):
            adjusted, _residual, latent, _features = self._forward(self.fast, prepared)
            reconstruction = self._next_loss(self.fast, latent, prepared)
            consistency = F.smooth_l1_loss(adjusted, risk_anchor_target)
            temporal = F.smooth_l1_loss(adjusted, temporal_target)
            loss = (
                self.next_weight * reconstruction
                + self.teacher_weight * consistency
                + self.temporal_risk_weight * temporal
            )
            self.fast_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fast.parameters(), 0.5)
            with _scaled_optimizer(self.fast_optimizer, update_scale):
                self.fast_optimizer.step()
            losses.append(float(loss.detach().cpu()))
            reconstructions.append(float(reconstruction.detach().cpu()))
            temporal_losses.append(float(temporal.detach().cpu()))
        return {
            "event": "private_intra_update", "shot": self.active_shot,
            "source_endpoint": int(source_endpoint),
            "observed_endpoint": int(observed_endpoint), "updated": True,
            "steps": int(steps), "loss": sum(losses) / len(losses),
            "reconstruction_loss": sum(reconstructions) / len(reconstructions),
            "temporal_risk_weight": self.temporal_risk_weight,
            "temporal_target_gain": self.temporal_target_gain,
            "temporal_risk_loss": sum(temporal_losses) / len(temporal_losses),
            "temporal_target_delta": float(
                temporal_delta.abs().mean().detach().cpu()
            ),
            "risk_anchor_source": (
                "sealed_pre_adapt_logit"
                if self.intra_parameter_scope == "representation_only"
                else "persistent_parent"
            ),
            "update_scale": update_scale, "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
        }

    def seal_shot(self, shot_id: int) -> dict[str, Any]:
        if self.active_shot != int(shot_id) or self.fast is None:
            raise RuntimeError("cannot seal inactive private shot")
        receipt = {
            "event": "private_shot_seal", "shot": int(shot_id),
            "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
        }
        self.active_shot = None
        self.fast = None
        self.fast_optimizer = None
        self.pending_shot = int(shot_id)
        return receipt

    def _require_sealed(self, shot_id: int) -> None:
        if self.pending_shot != int(shot_id):
            raise RuntimeError("a matching sealed private shot is required before reveal")

    def discard_without_reveal(self, shot_id: int) -> dict[str, Any]:
        self._require_sealed(shot_id)
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        return {"event": "private_discard_unqueried", "shot": int(shot_id), "updated": False}

    def finalize_revealed_protection(self, shot_id: int, *, group: str) -> dict[str, Any]:
        self._require_sealed(shot_id)
        if group not in ("TP", "TN"):
            raise ValueError("private protection-only group must be TP or TN")
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        return {
            "event": "private_revealed_protection", "shot": int(shot_id),
            "group": group, "updated": False,
            "persistent_sha256": self.persistent_hash(),
        }

    def reveal_and_inter_update(
        self, shot_id: int, *, selected: bool,
        current_batch: Mapping[str, torch.Tensor],
        replay_batch: Mapping[str, torch.Tensor], steps: int = 2,
        update_scale: float = 1.0,
    ) -> dict[str, Any]:
        self._require_sealed(shot_id)
        if not selected:
            raise PermissionError("private supervised update requires a selected query")
        if steps <= 0:
            raise ValueError("steps must be positive")
        update_scale = _scale(update_scale)
        current = _batch(current_batch, self.device)
        replay = _batch(replay_batch, self.device)
        for name in ("target", "sealed_logit"):
            if name not in current or name not in replay:
                raise KeyError(f"private supervised batch requires {name}")
        if update_scale == 0:
            self.pending_shot = None
            self.last_finalized_shot = int(shot_id)
            return {
                "event": "private_inter_update", "shot": int(shot_id),
                "accepted": True, "updated": False, "steps": 0,
                "loss": 0.0, "update_scale": 0.0,
                "persistent_sha256": self.persistent_hash(),
            }
        state_before = copy.deepcopy(self.persistent.state_dict())
        optimizer_before = copy.deepcopy(self.inter_optimizer.state_dict())
        self.persistent.eval()
        with torch.no_grad():
            guard_before, _r, _l, _f = self._forward(self.persistent, self.guard)
            guard_label = self.guard["label"].to(torch.bool)
            protected = (guard_before >= self.threshold_logit) == guard_label
            replay_before, _r, _l, _f = self._forward(self.persistent, replay)
        losses: list[float] = []
        self.persistent.train()
        for _ in range(int(steps)):
            current_logit, current_residual, current_latent, _f = self._forward(
                self.persistent, current,
            )
            replay_logit, replay_residual, replay_latent, _f = self._forward(
                self.persistent, replay,
            )
            guard_logit, _r, _l, _f = self._forward(self.persistent, self.guard)
            supervised = sealed_temporal_residual_loss(
                current_logit, current["sealed_logit"], current["target"],
            )
            if "alarm_boundary" in current:
                supervised = 0.50 * supervised + operational_alarm_boundary_loss(
                    current_logit, current["alarm_boundary"],
                    current["label"] > 0.5,
                )
            replay_anchor = F.smooth_l1_loss(replay_logit, replay_before)
            guard_anchor = F.smooth_l1_loss(guard_logit, guard_before)
            auxiliary = self._next_loss(self.persistent, current_latent, current)
            auxiliary = auxiliary + self._next_loss(self.persistent, replay_latent, replay)
            residual = 0.5 * (
                current_residual.square().mean() + replay_residual.square().mean()
            )
            loss = (
                supervised + 0.25 * replay_anchor
                + self.guard_anchor_weight * guard_anchor
                + 0.10 * auxiliary + 0.02 * residual
            )
            self.inter_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.persistent.parameters(), 0.5)
            with _scaled_optimizer(self.inter_optimizer, update_scale):
                self.inter_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.persistent.eval()
        with torch.no_grad():
            guard_after, _r, _l, _f = self._forward(self.persistent, self.guard)
            harms = int((
                protected & ((guard_after >= self.threshold_logit) != guard_label)
            ).sum().item())
            finite = bool(torch.isfinite(guard_after).all())
        accepted = finite and harms <= self.max_guard_harms
        if not accepted:
            self.persistent.load_state_dict(state_before)
            self.inter_optimizer.load_state_dict(optimizer_before)
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        return {
            "event": "private_inter_update", "shot": int(shot_id),
            "accepted": bool(accepted), "updated": bool(accepted),
            "objective": (
                "operational_alarm_boundary_residual"
                if "alarm_boundary" in current else "sealed_temporal_residual"
            ),
            "guard_harms": harms, "steps": int(steps),
            "loss": sum(losses) / len(losses), "update_scale": update_scale,
            "persistent_sha256": self.persistent_hash(),
        }


__all__ = [
    "EastPrivateDynamicsResidual", "PrivateTwoTimescaleController", "module_hash",
]
