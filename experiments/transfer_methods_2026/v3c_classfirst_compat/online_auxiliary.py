#!/usr/bin/env python3
"""Frozen-backbone A3 auxiliary online learning primitives.

The sidecar never feeds the base risk head.  Its API enforces the causal order
``predict_and_seal(t) -> observe_and_update(t+1)`` and all backbone features
are extracted under ``torch.no_grad``.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import sinusoidal_position_encoding


def support_novelty_contract(
    errors,
    *,
    quiet_quantile: float = 0.20,
    reference_quantile: float = 0.80,
) -> dict[str, float]:
    """Fit a label-free quiet/reference contract on support-shot errors."""
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or not 0 <= quiet_quantile < reference_quantile <= 1:
        raise ValueError("invalid support novelty reference")
    quiet = float(np.quantile(values, quiet_quantile))
    reference = max(float(np.quantile(values, reference_quantile)), 1e-8)
    return {
        "quiet_quantile": float(quiet_quantile),
        "reference_quantile": float(reference_quantile),
        "quiet_raw": quiet,
        "reference_raw": reference,
        "quiet_normalized": quiet / reference,
    }


def causal_candidate_confirmation(
    risk,
    *,
    threshold: float,
    novelty_events,
    quiet_raw: float,
    minimum_events: int = 4,
) -> dict[str, Any]:
    """Latch a risk candidate and confirm it using only novelty observed so far."""
    risk_values = np.asarray(risk, dtype=np.float64).reshape(-1)
    candidates = np.flatnonzero(risk_values >= float(threshold))
    if not len(candidates):
        return {
            "candidate_index": None, "confirmed": False,
            "confirmation_index": None, "event_count": 0,
        }
    candidate = int(candidates[0])
    events = sorted(
        (int(index), float(value))
        for index, value in novelty_events
        if 0 <= int(index) < len(risk_values) and np.isfinite(value)
    )
    observed: list[float] = []
    confirmation = None
    for index, value in events:
        observed.append(value)
        decision_index = max(candidate, index)
        if index <= decision_index and decision_index >= candidate:
            if len(observed) >= int(minimum_events) and float(np.mean(observed)) >= float(quiet_raw):
                confirmation = decision_index
                break
    return {
        "candidate_index": candidate,
        "confirmed": confirmation is not None,
        "confirmation_index": confirmation,
        "event_count": len(observed),
    }


def module_sha256(*modules: nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{module_index}:{name}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def frozen_token_features(enc: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact base CLS plus final patch token without enabling grads."""
    enc.eval()
    with torch.no_grad():
        h = enc.embed(x.transpose(1, 2)).transpose(1, 2)
        h = h + sinusoidal_position_encoding(h)
        cls = enc.cls.expand(x.shape[0], -1, -1)
        h = enc.encoder(torch.cat([cls, h], dim=1))
        cls_feature = enc.norm(h[:, 0])
        last_feature = enc.norm(h[:, -1])
    return cls_feature.detach(), last_feature.detach()


class AuxiliarySidecar(nn.Module):
    def __init__(self, feature_dim: int, physics_dim: int, hidden: int = 32):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.physics_dim = int(physics_dim)
        input_dim = self.feature_dim + 2 * self.physics_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.next_head = nn.Linear(hidden, self.physics_dim)
        self.direction_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )

    def encode(
        self, frozen: torch.Tensor, physics: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        values = torch.cat([frozen, physics, available.to(physics.dtype)], dim=-1)
        return self.projector(values)


class OnlineAuxiliaryLearner:
    """One-transition-at-a-time learner with a strict causal pending state."""

    def __init__(
        self,
        *,
        feature_dim: int,
        physics_dim: int,
        hidden: int = 32,
        lr: float = 1e-3,
        direction_weight: float = 0.1,
        consistency_weight: float = 0.05,
        ema_decay: float = 0.99,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.sidecar = AuxiliarySidecar(feature_dim, physics_dim, hidden).to(self.device)
        self.teacher = copy.deepcopy(self.sidecar.projector).to(self.device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.sidecar.parameters(), lr=lr, weight_decay=1e-4)
        self.direction_weight = float(direction_weight)
        self.consistency_weight = float(consistency_weight)
        self.ema_decay = float(ema_decay)
        self.pending: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def _tensor(self, value: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device).reshape(-1)

    def predict_and_seal(
        self,
        step_id: int,
        frozen: torch.Tensor,
        physics: torch.Tensor,
        available: torch.Tensor,
    ) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("previous prediction must be observed before another prediction")
        frozen_t = self._tensor(frozen).detach()
        physics_t = self._tensor(physics).detach()
        available_t = self._tensor(available, dtype=torch.bool).detach()
        latent = self.sidecar.encode(frozen_t, physics_t, available_t)
        prediction = self.sidecar.next_head(latent)
        self.pending = {
            "step_id": int(step_id),
            "frozen": frozen_t,
            "physics": physics_t,
            "available": available_t,
            "latent": latent,
            "prediction": prediction,
        }
        record = {
            "event": "predict_seal",
            "step_id": int(step_id),
            "predicted_next": prediction.detach().cpu().numpy().tolist(),
        }
        self.events.append(record)
        return record

    def observe_and_update(
        self,
        step_id: int,
        frozen: torch.Tensor,
        physics: torch.Tensor,
        available: torch.Tensor,
        *,
        update: bool = True,
    ) -> dict[str, Any]:
        if self.pending is None:
            raise RuntimeError("cannot observe before a prediction has been sealed")
        if int(step_id) != int(self.pending["step_id"]) + 1:
            raise RuntimeError("observation must be the strict next step")
        frozen_t = self._tensor(frozen).detach()
        physics_t = self._tensor(physics).detach()
        available_t = self._tensor(available, dtype=torch.bool).detach()
        current_latent = self.sidecar.encode(frozen_t, physics_t, available_t)
        prediction = self.pending["prediction"]
        valid = available_t & torch.isfinite(physics_t)
        if bool(valid.any()):
            next_loss = F.smooth_l1_loss(prediction[valid], physics_t[valid])
        else:
            next_loss = prediction.sum() * 0.0

        previous_latent = self.pending["latent"]
        true_logit = self.sidecar.direction_head(torch.cat([previous_latent, current_latent]))
        reverse_logit = self.sidecar.direction_head(torch.cat([current_latent, previous_latent]))
        direction_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(true_logit, torch.ones_like(true_logit))
            + F.binary_cross_entropy_with_logits(reverse_logit, torch.zeros_like(reverse_logit))
        )
        teacher_input = torch.cat([frozen_t, physics_t, available_t.to(physics_t.dtype)])
        with torch.no_grad():
            teacher_latent = self.teacher(teacher_input)
        consistency_loss = F.smooth_l1_loss(current_latent, teacher_latent)
        total = next_loss + self.direction_weight * direction_loss + self.consistency_weight * consistency_loss
        if update:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.sidecar.parameters(), 1.0)
            self.optimizer.step()
            with torch.no_grad():
                for teacher_parameter, student_parameter in zip(
                    self.teacher.parameters(), self.sidecar.projector.parameters()
                ):
                    teacher_parameter.mul_(self.ema_decay).add_(
                        student_parameter, alpha=1.0 - self.ema_decay
                    )

        error = torch.zeros_like(physics_t)
        error[valid] = torch.abs(prediction.detach()[valid] - physics_t[valid])
        receipt = {
            "event": "observe_update",
            "step_id": int(step_id),
            "updated": bool(update),
            "next_valid_count": int(valid.sum().item()),
            "next_loss": float(next_loss.detach().cpu()),
            "direction_loss": float(direction_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            "prediction_error_mean": float(error[valid].mean().cpu()) if bool(valid.any()) else 0.0,
        }
        self.events.append(receipt)
        self.pending = None
        return receipt

    def reset_shot(self) -> None:
        self.pending = None


__all__ = [
    "module_sha256", "frozen_token_features", "AuxiliarySidecar",
    "OnlineAuxiliaryLearner", "support_novelty_contract",
    "causal_candidate_confirmation",
]
