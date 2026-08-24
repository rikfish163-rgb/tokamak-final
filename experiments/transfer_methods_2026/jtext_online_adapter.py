#!/usr/bin/env python3
"""J-TEXT-only two-timescale residual and transactional controller.

The source encoder and source head are immutable parents.  Online learning is
confined to a small bounded residual: a per-shot fast copy can consume only
label-free causal prefixes, while the persistent copy can change only after a
sealed shot has been selected and revealed.
"""

import copy
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


CONTROLLER_RESUME_SCHEMA = "jtext-online-controller-resume-v1"
_INTRA_FIELDS = frozenset({
    "source_rebuilt",
    "source_logit",
    "source_available",
    "observed_rebuilt",
    "observed_source_logit",
    "observed_available",
})
_OUTCOME_FIELD_TOKENS = (
    "label", "target", "outcome", "downtime", "advancedtime", "ttd",
    "horizon", "isdisrupt", "reveal",
)
_LOSS_NAMES = frozenset({
    "next", "teacher", "risk_anchor", "physics", "supervised", "replay",
})


def module_sha256(module: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _optimizer_state_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_optimizer_state_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_optimizer_state_is_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _parameters_are_finite(module: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())


def _gradients_are_finite(module: nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


def _normalized_field(name: object) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def _reject_outcome_bearing_intra_fields(values: Mapping[str, Any]) -> None:
    for name in values:
        normalized = _normalized_field(name)
        if any(token in normalized for token in _OUTCOME_FIELD_TOKENS):
            raise ValueError(f"outcome-bearing intra field is forbidden: {name}")


def _as_bool_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.bool, device=device).detach()


def _as_float_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device).detach()


@dataclass(frozen=True)
class JTextIntraBatch:
    """Whitelist-only label-free pair of ordered causal observations."""

    source_rebuilt: torch.Tensor
    source_logit: torch.Tensor
    source_available: torch.Tensor
    observed_rebuilt: torch.Tensor
    observed_source_logit: torch.Tensor
    observed_available: torch.Tensor


@dataclass(frozen=True)
class JTextInterBatch:
    """Post-reveal temporal supervision for fixed sampled windows."""

    rebuilt: torch.Tensor
    source_logit: torch.Tensor
    available: torch.Tensor
    label: torch.Tensor


class JTextOnlineResidual(nn.Module):
    """Zero-initialized bounded residual over rebuilt17 summaries and source risk."""

    def __init__(
        self, *, feature_count: int = 17, hidden_size: int = 32,
        max_logit_shift: float = 0.5,
    ) -> None:
        super().__init__()
        if int(feature_count) <= 0 or int(hidden_size) <= 0:
            raise ValueError("J-TEXT residual dimensions must be positive")
        if not math.isfinite(float(max_logit_shift)) or float(max_logit_shift) <= 0:
            raise ValueError("J-TEXT residual max_logit_shift must be finite and positive")
        self.feature_count = int(feature_count)
        self.hidden_size = int(hidden_size)
        self.max_logit_shift = float(max_logit_shift)
        self.summary_encoder = nn.Sequential(
            nn.Linear(self.feature_count * 2 + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(self.hidden_size, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def online_parameter_count(self) -> int:
        return int(sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        ))

    def logit_shift(
        self, rebuilt: torch.Tensor, source_logit: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(rebuilt, dtype=torch.float32)
        source = torch.as_tensor(
            source_logit, dtype=values.dtype, device=values.device,
        )
        if values.ndim < 2 or values.shape[-1] != self.feature_count:
            raise ValueError("J-TEXT residual rebuilt summary shape changed")
        if source.shape == values.shape[:-1] + (1,):
            source = source.squeeze(-1)
        if source.shape != values.shape[:-1]:
            raise ValueError("J-TEXT source logits do not align with rebuilt summaries")
        known = (
            torch.ones_like(values, dtype=torch.bool)
            if available is None else
            torch.as_tensor(available, dtype=torch.bool, device=values.device)
        )
        if known.shape != values.shape:
            raise ValueError("J-TEXT availability does not align with rebuilt summaries")
        finite = torch.isfinite(values)
        known = known & finite
        masked = torch.where(known, values, torch.zeros_like(values))
        inputs = torch.cat(
            (masked, known.to(values.dtype), torch.sigmoid(source).unsqueeze(-1)),
            dim=-1,
        )
        raw = self.residual_head(self.summary_encoder(inputs)).squeeze(-1)
        return self.max_logit_shift * torch.tanh(raw)

    def forward(
        self, rebuilt: torch.Tensor, source_logit: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = torch.as_tensor(
            source_logit, dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        values = torch.as_tensor(rebuilt, dtype=torch.float32, device=source.device)
        known = None if available is None else torch.as_tensor(
            available, dtype=torch.bool, device=source.device,
        )
        shift = self.logit_shift(values, source, known)
        # IEEE addition by an exact +0 preserves finite source logits bitwise,
        # while retaining the residual gradient at the zero initialization.
        return source + shift


class JTextOnlineController:
    """Transaction boundary for J-TEXT fast and persistent online state."""

    def __init__(
        self,
        residual: JTextOnlineResidual,
        *,
        source_encoder: nn.Module | None = None,
        source_head: nn.Module | None = None,
        intra_lr: float = 1e-4,
        inter_lr: float = 5e-4,
        loss_weights: Mapping[str, float] | None = None,
        gradient_clip: float = 1.0,
    ) -> None:
        if not isinstance(residual, JTextOnlineResidual):
            raise TypeError("J-TEXT controller requires JTextOnlineResidual")
        if min(float(intra_lr), float(inter_lr), float(gradient_clip)) <= 0:
            raise ValueError("J-TEXT online learning rates and gradient clip must be positive")
        defaults = {
            "next": 1.0,
            "teacher": 0.5,
            "risk_anchor": 0.25,
            "physics": 0.0,
            "supervised": 1.0,
            "replay": 0.5,
        }
        if loss_weights is not None:
            unknown = set(loss_weights) - _LOSS_NAMES
            if unknown:
                raise ValueError(f"unknown J-TEXT online loss weights: {sorted(unknown)}")
            defaults.update({name: float(value) for name, value in loss_weights.items()})
        if any(not math.isfinite(value) or value < 0 for value in defaults.values()):
            raise ValueError("J-TEXT online loss weights must be finite and non-negative")

        self.persistent = residual
        self.device = next(residual.parameters()).device
        self.intra_lr = float(intra_lr)
        self.inter_lr = float(inter_lr)
        self.gradient_clip = float(gradient_clip)
        self.loss_weights = defaults
        self.inter_optimizer = torch.optim.Adam(
            self.persistent.parameters(), lr=self.inter_lr,
        )
        self.fast: JTextOnlineResidual | None = None
        self.fast_optimizer: torch.optim.Optimizer | None = None
        self.active_shot: int | None = None
        self.pending_shot: int | None = None
        self.last_finalized_shot: int | None = None
        self._source_encoder = source_encoder
        self._source_head = source_head
        for source_module in (self._source_encoder, self._source_head):
            if source_module is not None:
                source_module.eval()
                for parameter in source_module.parameters():
                    parameter.requires_grad_(False)
        self._initial_source_hashes = self.source_hashes

    @property
    def source_hashes(self) -> dict[str, str | None]:
        return {
            "encoder": (
                module_sha256(self._source_encoder)
                if self._source_encoder is not None else None
            ),
            "head": (
                module_sha256(self._source_head)
                if self._source_head is not None else None
            ),
        }

    @property
    def online_parameter_count(self) -> int:
        return self.persistent.online_parameter_count

    def assert_source_frozen(self) -> None:
        if self.source_hashes != self._initial_source_hashes:
            raise RuntimeError("frozen J-TEXT source encoder/head changed")

    def persistent_hash(self) -> str:
        return module_sha256(self.persistent)

    def fast_hash(self) -> str | None:
        return module_sha256(self.fast) if self.fast is not None else None

    def begin_shot(self, shot_id: int) -> dict[str, Any]:
        if self.active_shot is not None or self.pending_shot is not None:
            raise RuntimeError("previous J-TEXT shot is not finalized")
        self.assert_source_frozen()
        self.fast = copy.deepcopy(self.persistent).to(self.device)
        self.fast_optimizer = torch.optim.Adam(self.fast.parameters(), lr=self.intra_lr)
        self.active_shot = int(shot_id)
        return {
            "event": "begin_shot",
            "shot": int(shot_id),
            "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
            "online_parameter_count": self.online_parameter_count,
        }

    def predict(
        self, rebuilt: torch.Tensor, source_logit: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fast is None or self.active_shot is None:
            raise RuntimeError("begin_shot is required before J-TEXT prediction")
        self.assert_source_frozen()
        self.fast.eval()
        with torch.no_grad():
            return self.fast(rebuilt, source_logit, available).detach()

    def _prepare_intra(
        self, batch: JTextIntraBatch | Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        values = vars(batch) if isinstance(batch, JTextIntraBatch) else dict(batch)
        _reject_outcome_bearing_intra_fields(values)
        unknown = set(values) - _INTRA_FIELDS
        missing = _INTRA_FIELDS - set(values)
        if unknown or missing:
            raise ValueError(
                f"J-TEXT intra whitelist mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        prepared = {
            "source_rebuilt": _as_float_tensor(values["source_rebuilt"], device=self.device),
            "source_logit": _as_float_tensor(values["source_logit"], device=self.device),
            "source_available": _as_bool_tensor(values["source_available"], device=self.device),
            "observed_rebuilt": _as_float_tensor(values["observed_rebuilt"], device=self.device),
            "observed_source_logit": _as_float_tensor(
                values["observed_source_logit"], device=self.device,
            ),
            "observed_available": _as_bool_tensor(
                values["observed_available"], device=self.device,
            ),
        }
        feature_count = self.persistent.feature_count
        for prefix in ("source", "observed"):
            rebuilt = prepared[f"{prefix}_rebuilt"]
            logit_name = "source_logit" if prefix == "source" else "observed_source_logit"
            available = prepared[f"{prefix}_available"]
            if rebuilt.ndim != 2 or rebuilt.shape[1] != feature_count:
                raise ValueError("J-TEXT intra rebuilt batch shape changed")
            if available.shape != rebuilt.shape or prepared[logit_name].shape != rebuilt.shape[:-1]:
                raise ValueError("J-TEXT intra batch fields are not aligned")
        if prepared["source_rebuilt"].shape != prepared["observed_rebuilt"].shape:
            raise ValueError("J-TEXT intra source and observed batches must align")
        return prepared

    def _prepare_inter(
        self, batch: JTextInterBatch | Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        values = vars(batch) if isinstance(batch, JTextInterBatch) else dict(batch)
        required = {"rebuilt", "source_logit", "available", "label"}
        if set(values) != required:
            raise ValueError("J-TEXT inter batch contract changed")
        prepared = {
            "rebuilt": _as_float_tensor(values["rebuilt"], device=self.device),
            "source_logit": _as_float_tensor(values["source_logit"], device=self.device),
            "available": _as_bool_tensor(values["available"], device=self.device),
            "label": _as_float_tensor(values["label"], device=self.device),
        }
        rebuilt = prepared["rebuilt"]
        if rebuilt.ndim != 2 or rebuilt.shape[1] != self.persistent.feature_count:
            raise ValueError("J-TEXT inter rebuilt batch shape changed")
        if (
            prepared["available"].shape != rebuilt.shape
            or prepared["source_logit"].shape != rebuilt.shape[:-1]
            or prepared["label"].shape != rebuilt.shape[:-1]
        ):
            raise ValueError("J-TEXT inter batch fields are not aligned")
        return prepared

    def _loss_term(
        self, name: str, raw: torch.Tensor | None, *, available: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        weight = float(self.loss_weights[name])
        enabled = bool(available and weight > 0.0)
        if raw is None:
            zero = next(self.persistent.parameters()).sum() * 0.0
            raw_value = None
            weighted = zero
        else:
            raw_value = float(raw.detach().cpu())
            weighted = raw * weight if enabled else raw * 0.0
        receipt = {
            "enabled": enabled,
            "weight": weight,
            "raw": raw_value,
            "weighted": (
                float(weighted.detach().cpu()) if raw is not None else 0.0
            ),
        }
        return weighted, receipt

    def intra_update(
        self,
        batch: JTextIntraBatch | Mapping[str, Any],
        *,
        source_endpoint: int,
        observed_endpoint: int,
        steps: int = 1,
    ) -> dict[str, Any]:
        if self.fast is None or self.fast_optimizer is None or self.active_shot is None:
            raise RuntimeError("begin_shot is required before J-TEXT intra update")
        if int(source_endpoint) >= int(observed_endpoint):
            raise ValueError("J-TEXT intra update requires source_endpoint < observed_endpoint")
        if int(steps) <= 0:
            raise ValueError("J-TEXT intra update steps must be positive")
        prepared = self._prepare_intra(batch)
        persistent_before = self.persistent_hash()
        receipts: dict[str, Any] = {}
        self.fast.train()
        for _ in range(int(steps)):
            source_adjusted = self.fast(
                prepared["source_rebuilt"], prepared["source_logit"],
                prepared["source_available"],
            )
            observed_adjusted = self.fast(
                prepared["observed_rebuilt"], prepared["observed_source_logit"],
                prepared["observed_available"],
            )
            raw_losses = {
                "next": F.smooth_l1_loss(
                    source_adjusted, prepared["observed_source_logit"],
                ),
                "teacher": F.smooth_l1_loss(
                    observed_adjusted, prepared["observed_source_logit"],
                ),
                "risk_anchor": F.smooth_l1_loss(
                    source_adjusted, prepared["source_logit"],
                ),
                "physics": F.smooth_l1_loss(
                    observed_adjusted - source_adjusted,
                    prepared["observed_source_logit"] - prepared["source_logit"],
                ),
            }
            weighted = []
            receipts = {}
            for name in ("next", "teacher", "risk_anchor", "physics"):
                term, term_receipt = self._loss_term(name, raw_losses[name])
                weighted.append(term)
                receipts[name] = term_receipt
            loss = torch.stack(weighted).sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite J-TEXT intra loss")
            self.fast_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not _gradients_are_finite(self.fast):
                raise FloatingPointError("non-finite J-TEXT intra gradient")
            torch.nn.utils.clip_grad_norm_(self.fast.parameters(), self.gradient_clip)
            self.fast_optimizer.step()
            if not _parameters_are_finite(self.fast):
                raise FloatingPointError("non-finite J-TEXT fast parameter")
        self.assert_source_frozen()
        if self.persistent_hash() != persistent_before:
            raise RuntimeError("J-TEXT intra update changed persistent state")
        return {
            "event": "intra_update",
            "shot": self.active_shot,
            "source_endpoint": int(source_endpoint),
            "observed_endpoint": int(observed_endpoint),
            "steps": int(steps),
            "updated": True,
            "losses": receipts,
            "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
        }

    def seal_shot(self, shot_id: int) -> dict[str, Any]:
        if self.active_shot != int(shot_id) or self.fast is None:
            raise RuntimeError("cannot seal a J-TEXT shot that is not active")
        receipt = {
            "event": "seal",
            "shot": int(shot_id),
            "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
        }
        self.active_shot = None
        self.fast = None
        self.fast_optimizer = None
        self.pending_shot = int(shot_id)
        return receipt

    def _require_pending(self, shot_id: int) -> None:
        if self.pending_shot != int(shot_id):
            raise RuntimeError("matching sealed J-TEXT shot is required")

    def discard_unqueried(self, shot_id: int) -> dict[str, Any]:
        self._require_pending(shot_id)
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        return {
            "event": "discard_unqueried", "shot": int(shot_id),
            "updated": False, "persistent_sha256": self.persistent_hash(),
        }

    def finalize_revealed_without_update(self, shot_id: int) -> dict[str, Any]:
        self._require_pending(shot_id)
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        return {
            "event": "finalize_revealed", "shot": int(shot_id),
            "updated": False, "persistent_sha256": self.persistent_hash(),
        }

    def _restore_transaction(
        self, model_state: Mapping[str, torch.Tensor], optimizer_state: Mapping[str, Any],
    ) -> None:
        self.persistent.load_state_dict(model_state, strict=True)
        self.inter_optimizer.load_state_dict(optimizer_state)

    def inter_update(
        self,
        shot_id: int,
        *,
        current: JTextInterBatch | Mapping[str, Any],
        replay: JTextInterBatch | Mapping[str, Any] | None,
        guard_batch: JTextInterBatch | Mapping[str, Any] | None = None,
        steps: int = 1,
        selected: bool = True,
        guard: Callable[[torch.Tensor, torch.Tensor], bool] | None = None,
    ) -> dict[str, Any]:
        self._require_pending(shot_id)
        if not selected:
            raise PermissionError("J-TEXT persistent update requires selected reveal")
        if int(steps) <= 0:
            raise ValueError("J-TEXT inter update steps must be positive")
        current_batch = self._prepare_inter(current)
        replay_batch = self._prepare_inter(replay) if replay is not None else None
        protected_batch = (
            self._prepare_inter(guard_batch) if guard_batch is not None else current_batch
        )
        model_before = copy.deepcopy(self.persistent.state_dict())
        optimizer_before = copy.deepcopy(self.inter_optimizer.state_dict())
        self.persistent.eval()
        with torch.no_grad():
            protected_before = self.persistent(
                protected_batch["rebuilt"], protected_batch["source_logit"],
                protected_batch["available"],
            ).detach().clone()

        receipts: dict[str, Any] = {}
        rollback_reason: str | None = None
        self.persistent.train()
        for _ in range(int(steps)):
            adjusted = self.persistent(
                current_batch["rebuilt"], current_batch["source_logit"],
                current_batch["available"],
            )
            supervised = F.binary_cross_entropy_with_logits(
                adjusted, current_batch["label"],
            )
            replay_loss: torch.Tensor | None = None
            if replay_batch is not None:
                replay_adjusted = self.persistent(
                    replay_batch["rebuilt"], replay_batch["source_logit"],
                    replay_batch["available"],
                )
                replay_loss = F.binary_cross_entropy_with_logits(
                    replay_adjusted, replay_batch["label"],
                )
            anchor = F.smooth_l1_loss(adjusted, current_batch["source_logit"])
            physics = self.persistent.logit_shift(
                current_batch["rebuilt"], current_batch["source_logit"],
                current_batch["available"],
            ).square().mean()
            raw_losses = {
                "supervised": supervised,
                "replay": replay_loss,
                "risk_anchor": anchor,
                "physics": physics,
            }
            weighted = []
            receipts = {}
            for name in ("supervised", "replay", "risk_anchor", "physics"):
                term, term_receipt = self._loss_term(
                    name, raw_losses[name], available=(raw_losses[name] is not None),
                )
                weighted.append(term)
                receipts[name] = term_receipt
            loss = torch.stack(weighted).sum()
            if not bool(torch.isfinite(loss)):
                rollback_reason = "nonfinite_loss"
                break
            self.inter_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not _gradients_are_finite(self.persistent):
                rollback_reason = "nonfinite_gradient"
                break
            torch.nn.utils.clip_grad_norm_(
                self.persistent.parameters(), self.gradient_clip,
            )
            self.inter_optimizer.step()
            if not _parameters_are_finite(self.persistent):
                rollback_reason = "nonfinite_parameter"
                break
            if not _optimizer_state_is_finite(self.inter_optimizer.state_dict()):
                rollback_reason = "nonfinite_optimizer_state"
                break

        protected_after = protected_before
        if rollback_reason is None:
            self.persistent.eval()
            with torch.no_grad():
                protected_after = self.persistent(
                    protected_batch["rebuilt"], protected_batch["source_logit"],
                    protected_batch["available"],
                ).detach().clone()
            guard_accepted = (
                bool(guard(protected_before, protected_after))
                if guard is not None else
                bool(torch.isfinite(protected_after).all())
            )
            if not guard_accepted:
                rollback_reason = "guard_rejected"

        accepted = rollback_reason is None
        if not accepted:
            self._restore_transaction(model_before, optimizer_before)
        self.persistent.eval()
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        self.assert_source_frozen()
        return {
            "event": "update",
            "shot": int(shot_id),
            "selected": True,
            "accepted": bool(accepted),
            "updated": bool(accepted),
            "rollback_reason": rollback_reason,
            "steps_attempted": int(steps),
            "steps_committed": int(steps) if accepted else 0,
            "losses": receipts,
            "persistent_sha256": self.persistent_hash(),
            "source_hashes": self.source_hashes,
        }

    def reveal_and_inter_update(self, shot_id: int, **kwargs: Any) -> dict[str, Any]:
        """Compatibility spelling for callers that make reveal explicit."""

        return self.inter_update(shot_id, **kwargs)

    def resume_payload(self, *, extra_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.active_shot is not None or self.pending_shot is not None:
            raise RuntimeError("resume checkpoint requires a finalized shot boundary")
        if self.last_finalized_shot is None:
            raise RuntimeError("resume checkpoint requires a finalized shot boundary")
        self.assert_source_frozen()
        return {
            "schema_version": CONTROLLER_RESUME_SCHEMA,
            "persistent": copy.deepcopy(self.persistent.state_dict()),
            "inter_optimizer": copy.deepcopy(self.inter_optimizer.state_dict()),
            "last_finalized_shot": int(self.last_finalized_shot),
            "source_hashes": dict(self.source_hashes),
            "online_parameter_count": self.online_parameter_count,
            "extra_state": dict(extra_state or {}),
        }

    def load_resume_payload(self, payload: Mapping[str, Any]) -> None:
        if self.active_shot is not None or self.pending_shot is not None:
            raise RuntimeError("resume load requires a finalized controller boundary")
        if payload.get("schema_version") != CONTROLLER_RESUME_SCHEMA:
            raise ValueError("J-TEXT controller resume schema mismatch")
        if dict(payload.get("source_hashes", {})) != self.source_hashes:
            raise ValueError("J-TEXT controller resume source hashes mismatch")
        self.persistent.load_state_dict(payload["persistent"], strict=True)
        self.inter_optimizer.load_state_dict(payload["inter_optimizer"])
        self.last_finalized_shot = int(payload["last_finalized_shot"])

    def save_resume_checkpoint(
        self, path: Path, *, extra_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.resume_payload(extra_state=extra_state)
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        return {
            "event": "resume_checkpoint",
            "path": str(target),
            "last_finalized_shot": payload["last_finalized_shot"],
            "persistent_sha256": self.persistent_hash(),
        }


__all__ = [
    "CONTROLLER_RESUME_SCHEMA",
    "JTextIntraBatch",
    "JTextInterBatch",
    "JTextOnlineResidual",
    "JTextOnlineController",
    "module_sha256",
]
