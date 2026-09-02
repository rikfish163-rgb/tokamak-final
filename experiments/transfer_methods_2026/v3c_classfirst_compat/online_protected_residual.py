#!/usr/bin/env python3
"""Protected inter-shot updates for an EAST-private hazard residual.

The disruption prediction is sealed before outcome access.  Intra-shot
auxiliary learning lives in a separate sidecar and is deliberately absent
from this module.  Once a selected outcome is revealed, only the residual
head may change; a fixed support guard can reject and roll back an update.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_subset_sha256(
    module: nn.Module,
    *,
    include_prefix: str | None = None,
    exclude_prefix: str | None = None,
) -> str:
    """Hash a deterministic state-dict subset for freeze audits."""
    digest = hashlib.sha256()
    matched = 0
    for name, tensor in sorted(module.state_dict().items()):
        if include_prefix is not None and not name.startswith(include_prefix):
            continue
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
        matched += 1
    if matched == 0:
        raise ValueError("state-dict subset is empty")
    return digest.hexdigest()


def bounded_update_weight(query_score: float) -> float:
    """Map a pre-reveal query score to a deliberately small bounded weight."""
    return float(min(1.5, max(0.5, 0.5 + float(query_score))))


def _clone_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    required = ("frozen", "physics", "available", "base_logit")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"batch missing {missing}")
    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        dtype = torch.bool if key == "available" else torch.float32
        result[key] = torch.as_tensor(value, dtype=dtype, device=device).detach().clone()
    return result


class ProtectedResidualUpdater:
    """Strict prequential residual updater with support-guard rollback."""

    def __init__(
        self,
        adapter: nn.Module,
        *,
        threshold_logit: float,
        guard_batch: Mapping[str, torch.Tensor],
        lr: float = 1e-4,
        max_guard_harms: int = 0,
        guard_anchor_weight: float = 0.25,
        minimum_positive_residual: float = 0.0,
    ):
        if lr <= 0 or max_guard_harms < 0:
            raise ValueError("invalid online update contract")
        self.adapter = adapter
        self.device = next(adapter.parameters()).device
        self.threshold_logit = float(threshold_logit)
        self.guard_batch = _clone_batch(guard_batch, self.device)
        if "label" not in self.guard_batch:
            raise KeyError("guard batch requires label")
        self.max_guard_harms = int(max_guard_harms)
        self.guard_anchor_weight = float(guard_anchor_weight)
        self.minimum_positive_residual = float(minimum_positive_residual)
        for name, parameter in self.adapter.named_parameters():
            parameter.requires_grad_(name.startswith("residual_head"))
        self.optimizer = torch.optim.AdamW(
            self.adapter.residual_head.parameters(), lr=float(lr), weight_decay=1e-4
        )
        self.pending: dict[str, Any] | None = None
        self.last_finalized_shot: int | None = None
        self.events: list[dict[str, Any]] = []

    def _forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        _adjusted, delta, _, _ = self.adapter(
            batch["frozen"], batch["physics"], batch["available"], batch["base_logit"]
        )
        effective_delta = torch.where(
            (delta > 0) & (delta < self.minimum_positive_residual),
            torch.zeros_like(delta),
            delta,
        )
        return batch["base_logit"] + effective_delta, effective_delta

    def predict_and_seal(
        self, shot_id: int, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("a prediction is already pending")
        shot_id = int(shot_id)
        if self.last_finalized_shot is not None and shot_id <= self.last_finalized_shot:
            raise RuntimeError("shot ids must be strictly increasing")
        prepared = _clone_batch(batch, self.device)
        self.adapter.eval()
        with torch.no_grad():
            adjusted, delta = self._forward(prepared)
        record = {
            "event": "predict_seal",
            "shot": shot_id,
            "adjusted_logit": adjusted.detach().cpu().clone(),
            "delta": delta.detach().cpu().clone(),
            "predictor_sha256": module_subset_sha256(self.adapter),
        }
        self.pending = {"shot": shot_id, "record": record}
        self.events.append({key: value for key, value in record.items() if key not in ("adjusted_logit", "delta")})
        return record

    def discard_without_reveal(self, shot_id: int) -> dict[str, Any]:
        self._require_pending(shot_id)
        receipt = {"event": "discard_unqueried", "shot": int(shot_id), "updated": False}
        self.pending = None
        self.last_finalized_shot = int(shot_id)
        self.events.append(receipt)
        return receipt

    def _require_pending(self, shot_id: int) -> None:
        if self.pending is None:
            raise RuntimeError("no sealed prediction is pending")
        if int(shot_id) != int(self.pending["shot"]):
            raise RuntimeError("outcome does not match the pending shot")

    def reveal_and_update(
        self,
        shot_id: int,
        *,
        selected: bool,
        current_batch: Mapping[str, torch.Tensor],
        replay_batch: Mapping[str, torch.Tensor],
        query_score: float,
        steps: int = 2,
    ) -> dict[str, Any]:
        self._require_pending(shot_id)
        if not selected:
            raise PermissionError("outcome cannot be revealed unless the shot was selected")
        if steps <= 0:
            raise ValueError("steps must be positive")
        current = _clone_batch(current_batch, self.device)
        replay = _clone_batch(replay_batch, self.device)
        if "label" not in current or "label" not in replay:
            raise KeyError("revealed and replay batches require labels")

        residual_before = copy.deepcopy(self.adapter.residual_head.state_dict())
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        immutable_before = module_subset_sha256(self.adapter, exclude_prefix="residual_head")
        with torch.no_grad():
            guard_before, _ = self._forward(self.guard_batch)
            guard_prediction_before = guard_before >= self.threshold_logit
            guard_label = self.guard_batch["label"].to(torch.bool)
            protected = guard_prediction_before == guard_label

        update_weight = bounded_update_weight(query_score)
        self.adapter.train()
        losses: list[float] = []
        for _ in range(int(steps)):
            current_logit, current_delta = self._forward(current)
            replay_logit, replay_delta = self._forward(replay)
            guard_logit, guard_delta = self._forward(self.guard_batch)
            current_loss = F.binary_cross_entropy_with_logits(current_logit, current["label"])
            replay_loss = F.binary_cross_entropy_with_logits(replay_logit, replay["label"])
            guard_anchor = F.smooth_l1_loss(guard_logit, guard_before.detach())
            residual_penalty = (
                current_delta.square().mean() + replay_delta.square().mean()
                + guard_delta.square().mean()
            ) / 3.0
            loss = (
                update_weight * current_loss + replay_loss
                + self.guard_anchor_weight * guard_anchor + 0.02 * residual_penalty
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.residual_head.parameters(), 0.5)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))

        self.adapter.eval()
        with torch.no_grad():
            guard_after, _ = self._forward(self.guard_batch)
            guard_prediction_after = guard_after >= self.threshold_logit
            guard_harms = int((protected & (guard_prediction_after != guard_label)).sum().item())
            finite = bool(torch.isfinite(guard_after).all())
        immutable_after = module_subset_sha256(self.adapter, exclude_prefix="residual_head")
        if immutable_after != immutable_before:
            raise RuntimeError("online update changed immutable adapter parameters")

        accepted = finite and guard_harms <= self.max_guard_harms
        rollback_reason: str | None = None
        if not accepted:
            rollback_reason = "non_finite" if not finite else "guard_harm"
            self.adapter.residual_head.load_state_dict(residual_before)
            self.optimizer.load_state_dict(optimizer_before)

        receipt = {
            "event": "reveal_update",
            "shot": int(shot_id),
            "selected": True,
            "accepted": bool(accepted),
            "updated": bool(accepted),
            "rollback_reason": rollback_reason,
            "guard_harms": guard_harms,
            "update_weight": update_weight,
            "steps": int(steps),
            "loss": float(sum(losses) / len(losses)),
            "predictor_sha256": module_subset_sha256(self.adapter),
        }
        self.pending = None
        self.last_finalized_shot = int(shot_id)
        self.events.append(receipt)
        return receipt


__all__ = [
    "ProtectedResidualUpdater", "bounded_update_weight", "module_subset_sha256",
]
