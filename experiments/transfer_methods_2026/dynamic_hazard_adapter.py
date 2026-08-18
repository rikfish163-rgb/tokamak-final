#!/usr/bin/env python3
"""EAST-private auxiliary dynamics and bounded hazard residual."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def apply_positive_residual_gate(
    delta: torch.Tensor, *, minimum_positive: float
) -> torch.Tensor:
    """Keep negative corrections and only sufficiently supported positive ones."""
    threshold = torch.as_tensor(minimum_positive, dtype=delta.dtype, device=delta.device)
    return torch.where((delta > 0) & (delta < threshold), torch.zeros_like(delta), delta)


def median_margin_alarm(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Ensemble models after normalizing each probability by its own threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != len(thresholds):
        raise ValueError("scores must be [models, shots] with one threshold per model")
    clipped_scores = np.clip(scores, 1e-6, 1 - 1e-6)
    clipped_thresholds = np.clip(thresholds, 1e-6, 1 - 1e-6)
    score_logits = np.log(clipped_scores / (1 - clipped_scores))
    threshold_logits = np.log(clipped_thresholds / (1 - clipped_thresholds))
    margins = score_logits - threshold_logits[:, None]
    return np.median(margins, axis=0) >= 0.0


def support_weak_margin_contract(
    disruptive_margins, *, preserve_fraction: float = 0.90
) -> dict[str, float]:
    """Define the weak-alarm region using disruptive support shots only."""
    margins = np.asarray(disruptive_margins, dtype=np.float64)
    margins = margins[np.isfinite(margins)]
    if not len(margins) or not 0 < preserve_fraction <= 1:
        raise ValueError("invalid disruptive support margins")
    total_count = int(len(margins))
    protected = margins[margins >= 0.0]
    if not len(protected):
        raise ValueError("support has no correctly alarmed disruption")
    lower_quantile = 1.0 - float(preserve_fraction)
    cutoff = float(np.quantile(protected, lower_quantile, method="higher"))
    return {
        "preserve_fraction": float(preserve_fraction),
        "lower_quantile": lower_quantile,
        "weak_margin_max": cutoff,
        "support_disruption_count": total_count,
        "protected_support_disruption_count": int(len(protected)),
        "excluded_missed_disruption_count": total_count - int(len(protected)),
    }


def alarm_window_labels(
    times_s: np.ndarray,
    *,
    is_disrupt: bool,
    down_time_s: float | None,
    horizon_ms: float = 125.0,
) -> np.ndarray:
    """HDF5-authoritative EAST window labels for support-only training."""
    times = np.asarray(times_s, dtype=np.float64)
    if not is_disrupt or down_time_s is None:
        return np.zeros(len(times), dtype=bool)
    ttd_ms = (float(down_time_s) - times) * 1000.0
    return (ttd_ms >= -1e-9) & (ttd_ms <= float(horizon_ms) + 1e-9)


class DynamicHazardAdapter(nn.Module):
    """Target-private latent dynamics plus a missing-safe bounded residual."""

    def __init__(
        self,
        *,
        feature_dim: int,
        physics_dim: int,
        hidden: int = 32,
        max_logit_residual: float = 1.0,
        min_physics_families: int = 2,
    ):
        super().__init__()
        if max_logit_residual <= 0 or min_physics_families <= 0:
            raise ValueError("invalid adapter bounds")
        self.physics_dim = int(physics_dim)
        self.max_logit_residual = float(max_logit_residual)
        self.min_physics_families = int(min_physics_families)
        self.projector = nn.Sequential(
            nn.Linear(int(feature_dim) + 2 * self.physics_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.next_head = nn.Linear(hidden, self.physics_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def encode(
        self,
        frozen_features: torch.Tensor,
        physics: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat(
            [frozen_features, physics, available.to(physics.dtype)], dim=-1
        )
        return self.projector(values)

    def forward(
        self,
        frozen_features: torch.Tensor,
        physics: torch.Tensor,
        available: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(frozen_features, physics, available)
        raw = self.residual_head(
            torch.cat([latent, base_logit.reshape(-1, 1)], dim=-1)
        ).squeeze(-1)
        bounded = self.max_logit_residual * torch.tanh(raw)
        gate = (
            available.to(torch.int64).sum(dim=-1) >= self.min_physics_families
        ).to(base_logit.dtype)
        delta = bounded * gate
        return base_logit + delta, delta, gate, latent

    def predict_next(self, latent: torch.Tensor) -> torch.Tensor:
        return self.next_head(latent)


__all__ = [
    "alarm_window_labels", "apply_positive_residual_gate", "median_margin_alarm",
    "support_weak_margin_contract",
    "DynamicHazardAdapter",
]
