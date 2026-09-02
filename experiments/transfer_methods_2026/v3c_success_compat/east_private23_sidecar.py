#!/usr/bin/env python3
"""EAST-private residual branch for raw channels excluded from cross-device alignment."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


DROPPED_RAW8 = (
    "CIII", "MNM", "Mir_VV", "Mir_avg_amp",
    "ne_nG", "qa_proxy", "sxr_mean", "xuv_ratio",
)


def causal_private_features(
    values: torch.Tensor, available: torch.Tensor
) -> torch.Tensor:
    """Six bounded causal summaries per channel from an already scaled window."""
    raw = torch.as_tensor(values, dtype=torch.float32)
    valid = torch.as_tensor(available, dtype=torch.bool, device=raw.device)
    if raw.ndim != 3 or valid.shape != raw.shape:
        raise ValueError("private windows must have aligned [batch,time,channel] tensors")
    finite = torch.isfinite(raw)
    valid = valid & finite
    safe = torch.where(valid, raw, torch.zeros_like(raw))
    count = valid.sum(dim=1).clamp_min(1).to(raw.dtype)
    mean = safe.sum(dim=1) / count
    variance = torch.where(
        valid, (safe - mean[:, None, :]).square(), torch.zeros_like(safe)
    ).sum(dim=1) / count
    std = variance.sqrt()
    last = torch.where(valid[:, -1], safe[:, -1], torch.zeros_like(safe[:, -1]))
    first = torch.where(valid[:, 0], safe[:, 0], torch.zeros_like(safe[:, 0]))
    middle = safe.shape[1] // 2
    mid = torch.where(
        valid[:, middle], safe[:, middle], torch.zeros_like(safe[:, middle])
    )
    endpoint_valid = valid[:, -1] & valid[:, 0]
    recent_valid = valid[:, -1] & valid[:, middle]
    full_delta = torch.where(endpoint_valid, last - first, torch.zeros_like(last))
    recent_delta = torch.where(recent_valid, last - mid, torch.zeros_like(last))
    fraction = valid.to(raw.dtype).mean(dim=1)
    observed = valid.any(dim=1)
    summaries = torch.stack(
        [last, mean, std, full_delta, recent_delta, fraction], dim=-1
    )
    summaries = torch.where(observed[..., None], summaries, torch.zeros_like(summaries))
    return summaries.flatten(start_dim=1)


class EastPrivateResidual(nn.Module):
    """Small bounded residual; zero initialization preserves the shared model."""

    def __init__(
        self, *, channel_count: int = 8, hidden: int = 16,
        maximum_logit_residual: float = 0.50,
    ):
        super().__init__()
        if channel_count <= 0 or hidden <= 0 or maximum_logit_residual <= 0:
            raise ValueError("invalid EAST-private sidecar dimensions")
        self.channel_count = int(channel_count)
        self.maximum_logit_residual = float(maximum_logit_residual)
        input_dim = self.channel_count * 6 + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, values: torch.Tensor, available: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = causal_private_features(values, available)
        base = torch.as_tensor(
            base_logit, dtype=features.dtype, device=features.device
        )
        if base.ndim != 1 or base.shape[0] != features.shape[0]:
            raise ValueError("base logits must align with private windows")
        network_input = torch.cat([features, base.clamp(-8, 8)[:, None]], dim=-1)
        residual = self.maximum_logit_residual * torch.tanh(
            self.network(network_input).squeeze(-1)
        )
        return base + residual, residual, features


def sealed_temporal_residual_loss(
    current_logit: torch.Tensor,
    sealed_logit: torch.Tensor,
    temporal_target: torch.Tensor,
    *,
    correction_gain: float = 1.0,
    maximum_target_shift: float = 0.50,
    state_weight: float = 0.25,
) -> torch.Tensor:
    """Bind post-reveal learning to the actual prediction sealed before reveal."""
    current = torch.as_tensor(current_logit)
    sealed = torch.as_tensor(
        sealed_logit, dtype=current.dtype, device=current.device
    ).detach()
    target = torch.as_tensor(
        temporal_target, dtype=current.dtype, device=current.device
    ).detach()
    if current.shape != sealed.shape or current.shape != target.shape:
        raise ValueError("current, sealed, and temporal target must align")
    error = target - torch.sigmoid(sealed)
    shift = (float(correction_gain) * error).clamp(
        -float(maximum_target_shift), float(maximum_target_shift)
    )
    target_logit = (sealed + shift).detach()
    return (
        F.smooth_l1_loss(current, target_logit)
        + float(state_weight) * F.binary_cross_entropy_with_logits(current, target)
    )


class SameErrorReplayBuffer:
    """Replay earlier FP with FP and earlier FN with FN only."""

    def __init__(self, *, capacity_shots: int = 64):
        if capacity_shots <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity_shots = int(capacity_shots)
        self.entries: list[tuple[int, str, dict[str, torch.Tensor]]] = []

    def add(
        self, shot_id: int, *, error_group: str,
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        if error_group not in ("FP", "FN"):
            raise ValueError("same-error replay accepts only FP or FN")
        shot_id = int(shot_id)
        if self.entries and shot_id <= self.entries[-1][0]:
            raise RuntimeError("replay shots must be strictly increasing")
        stored = {
            name: torch.as_tensor(value).detach().cpu().clone()
            for name, value in batch.items()
        }
        self.entries.append((shot_id, error_group, stored))
        if len(self.entries) > self.capacity_shots:
            self.entries = self.entries[-self.capacity_shots:]

    def sample(
        self, *, error_group: str, count: int, seed: int
    ) -> dict[str, torch.Tensor] | None:
        if error_group not in ("FP", "FN") or count <= 0:
            raise ValueError("invalid same-error replay request")
        eligible = [batch for _shot, group, batch in self.entries if group == error_group]
        if not eligible:
            return None
        keys = tuple(eligible[0])
        joined = {
            name: torch.cat([batch[name] for batch in eligible], dim=0)
            for name in keys
        }
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randint(
            len(next(iter(joined.values()))), (int(count),), generator=generator
        )
        return {name: value.index_select(0, indices) for name, value in joined.items()}


__all__ = [
    "DROPPED_RAW8", "causal_private_features", "EastPrivateResidual",
    "sealed_temporal_residual_loss", "SameErrorReplayBuffer",
]
