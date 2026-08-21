#!/usr/bin/env python3
"""Two-timescale causal adaptation for an EAST-private hazard adapter."""

from __future__ import annotations

import copy
import hashlib
import math
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_INTRA_OUTCOME_FIELDS = frozenset({
    "label", "target", "sealed_logit", "alarm_boundary", "alarm_target",
    "is_disrupt", "down", "down_time_s", "DownTime", "outcome",
    "revealed_is_disrupt", "revealed_down_time_s",
})


def _reject_intra_outcome_fields(values: Mapping[str, torch.Tensor]) -> None:
    forbidden = sorted(_INTRA_OUTCOME_FIELDS.intersection(values))
    if forbidden:
        raise ValueError(f"intra-shot batch contains outcome field: {forbidden[0]}")


def _validate_update_scale(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("update_scale must lie in [0, 1]")
    return scale


@contextmanager
def _scaled_optimizer(optimizer: torch.optim.Optimizer, scale: float):
    original = [float(group["lr"]) for group in optimizer.param_groups]
    try:
        for group, learning_rate in zip(optimizer.param_groups, original):
            group["lr"] = learning_rate * scale
        yield
    finally:
        for group, learning_rate in zip(optimizer.param_groups, original):
            group["lr"] = learning_rate


def module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _batch(values: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    required = ("frozen", "physics", "available", "base_logit")
    missing = [name for name in required if name not in values]
    if missing:
        raise KeyError(f"batch missing {missing}")
    output = {}
    for name, value in values.items():
        dtype = torch.bool if name in ("available", "next_available") else torch.float32
        output[name] = torch.as_tensor(value, dtype=dtype, device=device).detach()
    return output


def _next_loss(adapter: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "next_physics" not in batch or "next_available" not in batch:
        raise KeyError("self-supervised update requires next_physics and next_available")
    latent = adapter.encode(batch["frozen"], batch["physics"], batch["available"])
    prediction = adapter.predict_next(latent)
    valid = batch["next_available"] & torch.isfinite(batch["next_physics"])
    if bool(valid.any()):
        return F.smooth_l1_loss(prediction[valid], batch["next_physics"][valid])
    return prediction.sum() * 0.0


def _adjusted(adapter: nn.Module, batch: Mapping[str, torch.Tensor]):
    return adapter(
        batch["frozen"], batch["physics"], batch["available"], batch["base_logit"]
    )


def protect_positive_fast_increment(
    fast_probability: torch.Tensor,
    persistent_probability: torch.Tensor,
    *,
    physics_support: torch.Tensor,
    physics_known: torch.Tensor,
    minimum_support: float,
) -> torch.Tensor:
    """Require causal physics support only when fast adaptation raises risk."""
    fast = torch.as_tensor(fast_probability)
    persistent = torch.as_tensor(
        persistent_probability, dtype=fast.dtype, device=fast.device
    )
    support = torch.as_tensor(physics_support, dtype=fast.dtype, device=fast.device)
    known = torch.as_tensor(physics_known, dtype=torch.bool, device=fast.device)
    if fast.shape != persistent.shape or fast.shape != support.shape or fast.shape != known.shape:
        raise ValueError("fast, persistent, support, and known must align")
    supported_raise = known & (support >= float(minimum_support))
    keep_fast = (fast <= persistent) | supported_raise
    return torch.where(keep_fast, fast, persistent)


def _smooth_topk(values: torch.Tensor, *, top_k: int, temperature: float) -> torch.Tensor:
    if values.ndim != 1 or not values.numel():
        raise ValueError("smooth top-k requires a non-empty one-dimensional tensor")
    if top_k <= 0 or temperature <= 0:
        raise ValueError("top_k and temperature must be positive")
    selected = torch.topk(values, k=min(int(top_k), values.numel())).values
    scale = float(temperature)
    return scale * (torch.logsumexp(selected / scale, dim=0) - torch.log(
        selected.new_tensor(float(selected.numel()))
    ))


def shot_level_mil_hazard_loss(
    logits: torch.Tensor,
    legal_positive: torch.Tensor,
    *,
    threshold_logit: float,
    top_k: int = 4,
    temperature: float = 0.25,
    hard_negative_weight: float = 0.25,
) -> torch.Tensor:
    """Deployment-aligned bag loss for one revealed shot.

    A safe shot is a negative bag whose highest-risk windows must stay below
    the frozen alarm boundary.  For a disruptive shot, only windows in the
    HDF5-authoritative alarm horizon can satisfy the positive bag; high-risk
    windows outside that horizon remain hard negatives.
    """
    values = torch.as_tensor(logits)
    positive = torch.as_tensor(
        legal_positive, dtype=torch.bool, device=values.device
    )
    if values.ndim != 1 or positive.shape != values.shape or not values.numel():
        raise ValueError("logits and legal_positive must be aligned non-empty vectors")
    if hard_negative_weight < 0:
        raise ValueError("hard_negative_weight must be non-negative")
    margin = values - float(threshold_logit)
    if not bool(positive.any()):
        return F.softplus(_smooth_topk(
            margin, top_k=top_k, temperature=temperature
        ))

    positive_margin = _smooth_topk(
        margin[positive], top_k=top_k, temperature=temperature
    )
    loss = F.softplus(-positive_margin)
    negative = margin[~positive]
    if negative.numel() and hard_negative_weight:
        hard_negative = _smooth_topk(
            negative, top_k=top_k, temperature=temperature
        )
        loss = loss + float(hard_negative_weight) * F.softplus(hard_negative)
    return loss


def operational_alarm_boundary_loss(
    logits: torch.Tensor,
    alarm_boundary_logit: torch.Tensor,
    positive: torch.Tensor,
    *,
    top_k: int = 4,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Optimize the same pointwise any-hit boundary used at deployment.

    ``alarm_boundary_logit`` is computed and sealed before outcome reveal from
    the dual high/physics/trend alarm.  A disruptive bag needs any timely point
    above that boundary; a safe error bag must move its strongest evidence
    below it.  The smooth top-k keeps the surrogate differentiable without
    pretending that every window in a disruptive shot must alarm.
    """
    values = torch.as_tensor(logits)
    boundary = torch.as_tensor(
        alarm_boundary_logit, dtype=values.dtype, device=values.device,
    ).detach()
    labels = torch.as_tensor(positive, dtype=torch.bool, device=values.device)
    if (
        values.ndim != 1 or not values.numel()
        or boundary.shape != values.shape or labels.shape != values.shape
    ):
        raise ValueError("alarm logits, boundary, and labels must align")
    evidence = values - boundary
    if bool(labels.any()):
        strongest = _smooth_topk(
            evidence[labels], top_k=top_k, temperature=temperature,
        )
        return F.softplus(-strongest)
    strongest = _smooth_topk(
        evidence, top_k=top_k, temperature=temperature,
    )
    return F.softplus(strongest)


def _deployment_aligned_hazard(
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    threshold_logit: float,
) -> torch.Tensor:
    positive = batch["label"] > 0.5
    if "alarm_boundary" in batch:
        return operational_alarm_boundary_loss(
            logits, batch["alarm_boundary"], positive,
        )
    return shot_level_mil_hazard_loss(
        logits, positive, threshold_logit=threshold_logit,
    )


def replay_distillation_loss(
    current_logits: torch.Tensor, sealed_logits: torch.Tensor
) -> torch.Tensor:
    """Anchor replay predictions without imposing a synthetic class prior."""
    current = torch.as_tensor(current_logits)
    sealed = torch.as_tensor(
        sealed_logits, dtype=current.dtype, device=current.device
    ).detach()
    if current.shape != sealed.shape:
        raise ValueError("current and sealed replay logits must align")
    return F.smooth_l1_loss(current, sealed)


def gate_signed_online_increment(
    increment: torch.Tensor,
    *,
    base_logit: torch.Tensor,
    threshold_logit: float,
    physics_support: torch.Tensor,
    physics_known: torch.Tensor,
    negative_max_support: float,
    positive_min_support: float,
) -> torch.Tensor:
    """Localize negative FP and positive FN corrections using causal physics."""
    values = torch.as_tensor(increment)
    base = torch.as_tensor(base_logit, dtype=values.dtype, device=values.device)
    support = torch.as_tensor(physics_support, dtype=values.dtype, device=values.device)
    known = torch.as_tensor(physics_known, dtype=torch.bool, device=values.device)
    if not (values.shape == base.shape == support.shape == known.shape):
        raise ValueError("increment gate inputs must align")
    negative_allowed = (
        known & (base >= float(threshold_logit))
        & (support <= float(negative_max_support))
    )
    positive_allowed = (
        known & (base < float(threshold_logit))
        & (support >= float(positive_min_support))
    )
    keep = ((values < 0) & negative_allowed) | ((values > 0) & positive_allowed)
    return torch.where(keep, values, torch.zeros_like(values))


def _physics_support(
    batch: Mapping[str, torch.Tensor], *, dtype: torch.dtype,
    activity_threshold: float, minimum_families: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(batch["physics"], dtype=dtype)
    available = torch.as_tensor(batch["available"], dtype=torch.bool, device=values.device)
    finite = torch.isfinite(values)
    valid = available & finite
    count = valid.sum(dim=-1)
    known = count >= int(minimum_families)
    safe_values = torch.where(valid, values.clamp(0.0, 1.0), torch.zeros_like(values))
    denominator = count.clamp_min(1).to(dtype)
    energy = safe_values.sum(dim=-1) / denominator
    concurrence = ((safe_values >= float(activity_threshold)) & valid).sum(dim=-1).to(
        dtype
    ) / denominator
    return (energy * concurrence).clamp(0.0, 1.0), known


def physics_consistency_loss(
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    threshold_logit: float,
    activity_threshold: float = 0.5,
    minimum_families: int = 2,
) -> torch.Tensor:
    """Availability-aware external physics constraint for revealed windows."""
    labels = torch.as_tensor(batch["label"], dtype=logits.dtype, device=logits.device)
    values = torch.as_tensor(batch["physics"], dtype=logits.dtype, device=logits.device)
    if values.ndim != 2 or labels.shape != logits.shape:
        raise ValueError("physics constraint inputs must align")
    support, known = _physics_support(
        batch, dtype=logits.dtype, activity_threshold=activity_threshold,
        minimum_families=minimum_families,
    )
    if not bool(known.any()):
        return logits.sum() * 0.0
    margin = logits - float(threshold_logit)
    positive = labels > 0.5
    per_window = torch.where(
        positive,
        support * F.softplus(-margin),
        (1.0 - support) * F.softplus(margin),
    )
    return per_window[known].mean()


def unlabeled_physics_pseudo_loss(
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    threshold_logit: float,
    activity_threshold: float = 0.5,
    minimum_families: int = 2,
    negative_max_support: float = 0.10,
    positive_min_support: float = 0.65,
) -> torch.Tensor:
    """Correct only strong causal risk/physics contradictions without labels."""
    values = torch.as_tensor(logits)
    base = torch.as_tensor(
        batch["base_logit"], dtype=values.dtype, device=values.device
    )
    if values.ndim != 1 or base.shape != values.shape:
        raise ValueError("pseudo-loss logits and base_logit must align")
    support, known = _physics_support(
        batch, dtype=values.dtype, activity_threshold=activity_threshold,
        minimum_families=minimum_families,
    )
    fp_like = (
        known & (base >= float(threshold_logit))
        & (support <= float(negative_max_support))
    )
    fn_like = (
        known & (base < float(threshold_logit))
        & (support >= float(positive_min_support))
    )
    active = fp_like | fn_like
    if not bool(active.any()):
        return values.sum() * 0.0
    margin = values - float(threshold_logit)
    per_window = torch.where(fp_like, F.softplus(margin), F.softplus(-margin))
    return per_window[active].mean()


class CausalReplayBuffer:
    """Fixed-shot replay containing only outcomes revealed in earlier shots."""

    def __init__(self, *, capacity_shots: int = 64):
        if capacity_shots <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity_shots = int(capacity_shots)
        self._entries: list[tuple[int, dict[str, torch.Tensor]]] = []
        self._metadata: dict[int, dict[str, Any]] = {}

    @property
    def shots(self) -> list[int]:
        return [shot for shot, _batch in self._entries]

    def add(
        self, shot: int, batch: Mapping[str, torch.Tensor], *,
        discrepancy_group: str | None = None,
        phenotype: Sequence[float] | None = None,
    ) -> None:
        shot = int(shot)
        if self._entries and shot <= self._entries[-1][0]:
            raise RuntimeError("replay shots must be strictly increasing")
        prepared = {
            name: torch.as_tensor(value).detach().cpu().clone()
            for name, value in batch.items()
        }
        if "label" not in prepared:
            raise KeyError("revealed replay batch requires label")
        if (discrepancy_group is None) != (phenotype is None):
            raise ValueError("replay discrepancy group and phenotype must be provided together")
        if discrepancy_group is not None:
            if discrepancy_group not in ("FP", "FN", "TARDY"):
                raise ValueError("replay metadata requires an operational error group")
            vector = np.asarray(phenotype, dtype=np.float64).reshape(-1)
            if not len(vector) or not np.all(np.isfinite(vector)):
                raise ValueError("replay phenotype must be finite and non-empty")
            self._metadata[shot] = {
                "group": discrepancy_group, "phenotype": vector.copy(),
            }
        self._entries.append((shot, prepared))
        if len(self._entries) > self.capacity_shots:
            self._entries = self._entries[-self.capacity_shots:]
            retained = set(self.shots)
            self._metadata = {
                key: value for key, value in self._metadata.items() if key in retained
            }

    def sample_similar_error_bag(
        self, *, discrepancy_group: str, phenotype: Sequence[float],
        count: int, seed: int, max_distance: float | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a labelled bag from the nearest earlier same-type mistake."""
        if count <= 0:
            raise ValueError("replay sample count must be positive")
        if max_distance is not None and (
            not math.isfinite(float(max_distance)) or float(max_distance) <= 0
        ):
            raise ValueError("replay maximum distance must be finite and positive")
        vector = np.asarray(phenotype, dtype=np.float64).reshape(-1)
        candidates = []
        for shot, batch in self._entries:
            metadata = self._metadata.get(shot)
            if metadata is None or metadata["group"] != discrepancy_group:
                continue
            stored = np.asarray(metadata["phenotype"], dtype=np.float64)
            if stored.shape != vector.shape:
                raise ValueError("replay phenotype shape changed")
            distance = float(
                np.linalg.norm(vector - stored) / math.sqrt(max(len(vector), 1))
            )
            candidates.append((distance, shot, batch))
        if not candidates:
            return None
        distance, source_shot, bag = min(candidates, key=lambda item: (item[0], item[1]))
        if max_distance is not None and distance > float(max_distance):
            return None
        generator = torch.Generator().manual_seed(int(seed))
        row_count = len(next(iter(bag.values())))
        indices = torch.randint(row_count, (int(count),), generator=generator)
        sampled = {
            name: value.index_select(0, indices) for name, value in bag.items()
        }
        return {
            "source_shot": int(source_shot), "distance": float(distance),
            "batch": sampled,
        }

    def sample(self, *, count: int, seed: int) -> dict[str, torch.Tensor] | None:
        if count <= 0:
            raise ValueError("replay sample count must be positive")
        if not self._entries:
            return None
        keys = tuple(self._entries[0][1])
        joined = {
            name: torch.cat([batch[name] for _shot, batch in self._entries], dim=0)
            for name in keys
        }
        labels = joined["label"] > 0.5
        positive = torch.nonzero(labels, as_tuple=False).flatten()
        negative = torch.nonzero(~labels, as_tuple=False).flatten()
        generator = torch.Generator().manual_seed(int(seed))

        def choose(pool: torch.Tensor, n: int) -> torch.Tensor:
            if n <= 0:
                return torch.zeros(0, dtype=torch.long)
            if not len(pool):
                pool = torch.arange(len(labels))
            return pool[torch.randint(len(pool), (n,), generator=generator)]

        positive_count = count // 2 if len(positive) and len(negative) else count if len(positive) else 0
        indices = torch.cat([
            choose(positive, positive_count),
            choose(negative, count - positive_count),
        ])
        return {name: value.index_select(0, indices) for name, value in joined.items()}

    def sample_opposite_bag(
        self, *, current_is_disruptive: bool, count: int, seed: int
    ) -> dict[str, torch.Tensor] | None:
        """Sample one earlier shot with the opposite revealed outcome."""
        if count <= 0:
            raise ValueError("replay sample count must be positive")
        eligible = [
            batch for _shot, batch in self._entries
            if bool((batch["label"] > 0.5).any()) != bool(current_is_disruptive)
        ]
        if not eligible:
            return None
        generator = torch.Generator().manual_seed(int(seed))
        bag = eligible[int(torch.randint(len(eligible), (1,), generator=generator))]
        labels = bag["label"] > 0.5
        positive = torch.nonzero(labels, as_tuple=False).flatten()
        negative = torch.nonzero(~labels, as_tuple=False).flatten()

        def choose(pool: torch.Tensor, n: int) -> torch.Tensor:
            if n <= 0:
                return torch.zeros(0, dtype=torch.long)
            if not len(pool):
                pool = torch.arange(len(labels))
            return pool[torch.randint(len(pool), (n,), generator=generator)]

        positive_count = count // 2 if len(positive) else 0
        indices = torch.cat([
            choose(positive, positive_count),
            choose(negative, count - positive_count),
        ])
        return {name: value.index_select(0, indices) for name, value in bag.items()}


class TwoTimescaleAdapterController:
    """Per-shot SSL fast weights plus persistent post-reveal supervision.

    The fast copy is reset from persistent weights at every shot.  Its
    self-supervised updates can affect only predictions made after the update.
    Persistent weights change only after a selected shot has been sealed and
    its outcome is explicitly revealed.
    """

    def __init__(
        self,
        adapter: nn.Module,
        *,
        guard_batch: Mapping[str, torch.Tensor],
        intra_lr: float = 1e-4,
        inter_lr: float = 5e-4,
        risk_anchor_weight: float = 0.25,
        guard_anchor_weight: float = 0.25,
        max_guard_harms: int = 0,
        physics_constraint_weight: float = 0.10,
        intra_physics_weight: float = 0.10,
        observed_physics_weight: float = 0.0,
        physics_activity_threshold: float = 0.5,
        physics_minimum_families: int = 2,
        physics_increment_gate: bool = False,
        negative_max_physics_support: float = 0.10,
        positive_min_physics_support: float = 0.65,
        minimum_positive_residual: float = 0.0,
        threshold_logit: float = 0.0,
        temporal_risk_weight: float = 0.0,
        temporal_target_max_delta: float = 0.5,
        temporal_target_gain: float = 1.0,
        intra_parameter_scope: str = "legacy",
    ):
        if (
            intra_lr <= 0 or inter_lr <= 0 or max_guard_harms < 0
            or physics_constraint_weight < 0 or intra_physics_weight < 0
            or not math.isfinite(float(temporal_risk_weight))
            or not math.isfinite(float(observed_physics_weight))
            or observed_physics_weight < 0
            or physics_minimum_families <= 0 or temporal_risk_weight < 0
            or not math.isfinite(float(temporal_target_max_delta))
            or temporal_target_max_delta <= 0
            or not math.isfinite(float(temporal_target_gain))
            or temporal_target_gain <= 0
            or intra_parameter_scope not in ("legacy", "representation_only")
        ):
            raise ValueError("invalid two-timescale update contract")
        self.persistent = adapter
        self.device = next(adapter.parameters()).device
        self.parent_anchor = copy.deepcopy(adapter).to(self.device).eval()
        for parameter in self.parent_anchor.parameters():
            parameter.requires_grad_(False)
        self.guard = _batch(guard_batch, self.device)
        if "label" not in self.guard:
            raise KeyError("guard batch requires labels")
        self.intra_lr = float(intra_lr)
        self.risk_anchor_weight = float(risk_anchor_weight)
        self.guard_anchor_weight = float(guard_anchor_weight)
        self.max_guard_harms = int(max_guard_harms)
        self.physics_constraint_weight = float(physics_constraint_weight)
        self.intra_physics_weight = float(intra_physics_weight)
        self.observed_physics_weight = float(observed_physics_weight)
        self.physics_activity_threshold = float(physics_activity_threshold)
        self.physics_minimum_families = int(physics_minimum_families)
        self.physics_increment_gate = bool(physics_increment_gate)
        self.negative_max_physics_support = float(negative_max_physics_support)
        self.positive_min_physics_support = float(positive_min_physics_support)
        self.minimum_positive_residual = float(minimum_positive_residual)
        self.threshold_logit = float(threshold_logit)
        self.temporal_risk_weight = float(temporal_risk_weight)
        self.temporal_target_max_delta = float(temporal_target_max_delta)
        self.temporal_target_gain = float(temporal_target_gain)
        self.intra_parameter_scope = str(intra_parameter_scope)
        self.fast: nn.Module | None = None
        self.fast_optimizer: torch.optim.Optimizer | None = None
        self.active_shot: int | None = None
        self.pending_shot: int | None = None
        self.last_finalized_shot: int | None = None

        inter_parameters = [
            parameter for name, parameter in self.persistent.named_parameters()
            if name.startswith("residual_head")
        ]
        if not inter_parameters:
            raise ValueError("persistent trainable parameter set is empty")
        for parameter in self.persistent.parameters():
            parameter.requires_grad_(False)
        for parameter in inter_parameters:
            parameter.requires_grad_(True)
        self.inter_optimizer = torch.optim.AdamW(
            inter_parameters, lr=float(inter_lr), weight_decay=1e-4
        )

    def persistent_hash(self) -> str:
        return module_hash(self.persistent)

    def fast_hash(self) -> str:
        if self.fast is None:
            raise RuntimeError("no active fast adapter")
        return module_hash(self.fast)

    def _forward(self, adapter: nn.Module, batch: Mapping[str, torch.Tensor]):
        _raw_adjusted, online_delta, gate, latent = _adjusted(adapter, batch)
        with torch.no_grad():
            _parent_adjusted, parent_delta, _parent_gate, _parent_latent = _adjusted(
                self.parent_anchor, batch
            )
        gated_parent = torch.where(
            (parent_delta > 0) & (parent_delta < self.minimum_positive_residual),
            torch.zeros_like(parent_delta), parent_delta,
        )
        # Preserve the registered static parent exactly, while allowing small
        # signed online changes around it.  Applying the old gate to the total
        # online residual would erase precisely the corrections adaptation is
        # meant to learn.
        increment = online_delta - parent_delta
        if self.physics_increment_gate:
            support, known = _physics_support(
                batch, dtype=increment.dtype,
                activity_threshold=self.physics_activity_threshold,
                minimum_families=self.physics_minimum_families,
            )
            increment = gate_signed_online_increment(
                increment, base_logit=batch["base_logit"],
                threshold_logit=self.threshold_logit,
                physics_support=support, physics_known=known,
                negative_max_support=self.negative_max_physics_support,
                positive_min_support=self.positive_min_physics_support,
            )
        effective = gated_parent + increment
        return batch["base_logit"] + effective, effective, gate, latent

    def _training_forward(
        self, adapter: nn.Module, batch: Mapping[str, torch.Tensor]
    ):
        """Expose a differentiable signed increment for supervised updates.

        The deployment path deliberately uses a hard sign/physics gate.  At
        the source-anchored initialization its increment is exactly zero, so
        routing the training objective through that hard gate would select the
        zero branch and kill every supervised gradient.  Training therefore
        optimizes the raw increment; prediction and guard acceptance still use
        :meth:`_forward`, where the registered physics gate is enforced.
        """
        _raw_adjusted, online_delta, gate, latent = _adjusted(adapter, batch)
        with torch.no_grad():
            _parent_adjusted, parent_delta, _parent_gate, _parent_latent = _adjusted(
                self.parent_anchor, batch
            )
        gated_parent = torch.where(
            (parent_delta > 0) & (parent_delta < self.minimum_positive_residual),
            torch.zeros_like(parent_delta), parent_delta,
        )
        increment = online_delta - parent_delta
        return batch["base_logit"] + gated_parent + increment, increment, gate, latent

    def begin_shot(self, shot_id: int) -> None:
        shot_id = int(shot_id)
        if self.active_shot is not None or self.pending_shot is not None:
            raise RuntimeError("previous shot must be finalized before begin_shot")
        if self.last_finalized_shot is not None and shot_id <= self.last_finalized_shot:
            raise RuntimeError("shot ids must be strictly increasing")
        self.fast = copy.deepcopy(self.persistent).to(self.device)
        for parameter in self.fast.parameters():
            parameter.requires_grad_(False)
        if self.intra_parameter_scope == "representation_only":
            prefixes = ("projector.2", "projector.3")
        else:
            prefixes = ("projector.3", "next_head", "residual_head")
        fast_parameters = [
            parameter for name, parameter in self.fast.named_parameters()
            if name.startswith(prefixes)
        ]
        if not fast_parameters:
            raise RuntimeError("intra-shot trainable parameter set is empty")
        for parameter in fast_parameters:
            parameter.requires_grad_(True)
        self.fast_optimizer = torch.optim.AdamW(
            fast_parameters, lr=self.intra_lr, weight_decay=1e-4
        )
        self.active_shot = shot_id

    def predict(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.fast is None or self.active_shot is None:
            raise RuntimeError("begin_shot is required before prediction")
        batch = _batch(values, self.device)
        self.fast.eval()
        with torch.no_grad():
            adjusted, _delta, _gate, _latent = self._forward(self.fast, batch)
        return adjusted.detach().cpu()

    def predict_persistent(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the current shot's slow-path prediction without fast updates."""
        if self.active_shot is None:
            raise RuntimeError("begin_shot is required before prediction")
        batch = _batch(values, self.device)
        self.persistent.eval()
        with torch.no_grad():
            adjusted, _delta, _gate, _latent = self._forward(self.persistent, batch)
        return adjusted.detach().cpu()

    def intra_update(
        self,
        *,
        source_endpoint: int,
        observed_endpoint: int,
        batch: Mapping[str, torch.Tensor],
        observed_batch: Mapping[str, torch.Tensor] | None = None,
        steps: int = 1,
        update_scale: float = 1.0,
    ) -> dict[str, Any]:
        if self.fast is None or self.fast_optimizer is None or self.active_shot is None:
            raise RuntimeError("begin_shot is required before intra_update")
        if self.intra_parameter_scope == "representation_only":
            _reject_intra_outcome_fields(batch)
            if observed_batch is not None:
                _reject_intra_outcome_fields(observed_batch)
            if "sealed_pre_adapt_logit" not in batch:
                raise KeyError(
                    "representation-only TTT requires sealed_pre_adapt_logit"
                )
        if int(observed_endpoint) <= int(source_endpoint):
            raise ValueError("the observed target endpoint must be strictly later than the source")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if (
            self.temporal_risk_weight > 0.0
            or self.observed_physics_weight > 0.0
        ) and observed_batch is None:
            raise ValueError(
                "observed_batch is required when temporal risk forecasting is enabled"
            )
        update_scale = _validate_update_scale(update_scale)
        if update_scale == 0.0:
            return {
                "event": "intra_observe_update", "shot": self.active_shot,
                "source_endpoint": int(source_endpoint),
                "observed_endpoint": int(observed_endpoint),
                "steps": 0, "loss": 0.0, "pseudo_loss": 0.0,
                "updated": False,
                "update_scale": 0.0,
                "temporal_risk_weight": self.temporal_risk_weight,
                "temporal_target_gain": self.temporal_target_gain,
                "temporal_risk_loss": 0.0,
                "temporal_target_delta": 0.0,
                "observed_physics_weight": self.observed_physics_weight,
                "observed_physics_loss": 0.0,
                "persistent_sha256": self.persistent_hash(),
                "fast_sha256": self.fast_hash(),
            }
        prepared = _batch(batch, self.device)
        observed = (
            _batch(observed_batch, self.device)
            if observed_batch is not None else None
        )
        self.persistent.eval()
        with torch.no_grad():
            persistent_adjusted, _delta, _gate, _latent = self._forward(self.persistent, prepared)
            if self.intra_parameter_scope == "representation_only":
                risk_anchor_target = prepared["sealed_pre_adapt_logit"]
                if risk_anchor_target.shape != persistent_adjusted.shape:
                    raise ValueError(
                        "sealed_pre_adapt_logit must align with source logits"
                    )
            else:
                risk_anchor_target = persistent_adjusted
            if observed is not None and self.temporal_risk_weight > 0.0:
                observed_adjusted, _delta, _gate, _latent = self._forward(
                    self.persistent, observed,
                )
                if observed_adjusted.shape != persistent_adjusted.shape:
                    raise ValueError(
                        "observed_batch must align with the source batch"
                    )
                temporal_delta = torch.clamp(
                    self.temporal_target_gain * (
                        observed_adjusted - persistent_adjusted
                    ),
                    min=-self.temporal_target_max_delta,
                    max=self.temporal_target_max_delta,
                )
                temporal_target = persistent_adjusted + temporal_delta
            else:
                temporal_delta = torch.zeros_like(persistent_adjusted)
                temporal_target = persistent_adjusted
        losses = []
        pseudo_losses = []
        temporal_losses = []
        observed_physics_losses = []
        self.fast.train()
        for _ in range(int(steps)):
            fast_adjusted, _increment, _gate, _latent = self._training_forward(
                self.fast, prepared
            )
            prediction_loss = _next_loss(self.fast, prepared)
            risk_anchor = F.smooth_l1_loss(fast_adjusted, risk_anchor_target)
            pseudo = unlabeled_physics_pseudo_loss(
                fast_adjusted, prepared, threshold_logit=self.threshold_logit,
                activity_threshold=self.physics_activity_threshold,
                minimum_families=self.physics_minimum_families,
                negative_max_support=self.negative_max_physics_support,
                positive_min_support=self.positive_min_physics_support,
            )
            temporal = F.smooth_l1_loss(fast_adjusted, temporal_target)
            if observed is not None and self.observed_physics_weight > 0.0:
                observed_adjusted, _increment, _gate, _latent = (
                    self._training_forward(self.fast, observed)
                )
                observed_physics = unlabeled_physics_pseudo_loss(
                    observed_adjusted, observed,
                    threshold_logit=self.threshold_logit,
                    activity_threshold=self.physics_activity_threshold,
                    minimum_families=self.physics_minimum_families,
                    negative_max_support=self.negative_max_physics_support,
                    positive_min_support=self.positive_min_physics_support,
                )
            else:
                observed_physics = fast_adjusted.sum() * 0.0
            loss = (
                prediction_loss + self.risk_anchor_weight * risk_anchor
                + self.intra_physics_weight * pseudo
                + self.temporal_risk_weight * temporal
                + self.observed_physics_weight * observed_physics
            )
            self.fast_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.fast.parameters() if p.requires_grad], 0.5
            )
            with _scaled_optimizer(self.fast_optimizer, update_scale):
                self.fast_optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pseudo_losses.append(float(pseudo.detach().cpu()))
            temporal_losses.append(float(temporal.detach().cpu()))
            observed_physics_losses.append(
                float(observed_physics.detach().cpu())
            )
        return {
            "event": "intra_observe_update",
            "shot": self.active_shot,
            "source_endpoint": int(source_endpoint),
            "observed_endpoint": int(observed_endpoint),
            "steps": int(steps),
            "loss": sum(losses) / len(losses),
            "pseudo_loss": sum(pseudo_losses) / len(pseudo_losses),
            "physics_weight": self.intra_physics_weight,
            "temporal_risk_weight": self.temporal_risk_weight,
            "temporal_target_gain": self.temporal_target_gain,
            "temporal_risk_loss": sum(temporal_losses) / len(temporal_losses),
            "temporal_target_delta": float(
                temporal_delta.abs().mean().detach().cpu()
            ),
            "observed_physics_weight": self.observed_physics_weight,
            "risk_anchor_source": (
                "sealed_pre_adapt_logit"
                if self.intra_parameter_scope == "representation_only"
                else "persistent_parent"
            ),
            "observed_physics_loss": (
                sum(observed_physics_losses) / len(observed_physics_losses)
            ),
            "updated": True,
            "update_scale": update_scale,
            "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
        }

    def intra_physics_update(
        self, *, batch: Mapping[str, torch.Tensor], steps: int = 1,
        update_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Use current causal observations to update only the per-shot fast copy."""
        if self.fast is None or self.fast_optimizer is None or self.active_shot is None:
            raise RuntimeError("begin_shot is required before intra_physics_update")
        if steps <= 0:
            raise ValueError("steps must be positive")
        update_scale = _validate_update_scale(update_scale)
        if update_scale == 0.0:
            return {
                "event": "intra_physics_pseudo_update", "shot": self.active_shot,
                "steps": 0, "loss": 0.0, "pseudo_loss": 0.0,
                "physics_weight": self.intra_physics_weight,
                "updated": False, "update_scale": 0.0,
                "persistent_sha256": self.persistent_hash(),
                "fast_sha256": self.fast_hash(),
            }
        prepared = _batch(batch, self.device)
        self.persistent.eval()
        with torch.no_grad():
            persistent_adjusted, _delta, _gate, _latent = self._forward(
                self.persistent, prepared
            )
            _parent_adjusted, parent_delta, _parent_gate, _parent_latent = _adjusted(
                self.parent_anchor, prepared
            )
            gated_parent = torch.where(
                (parent_delta > 0) & (parent_delta < self.minimum_positive_residual),
                torch.zeros_like(parent_delta), parent_delta,
            )

        losses = []
        pseudo_losses = []
        self.fast.train()
        for _ in range(int(steps)):
            _adjusted_fast, online_delta, _gate, _latent = _adjusted(
                self.fast, prepared
            )
            # Use the raw signed increment during optimization.  The hard
            # deployment gate is non-differentiable at the zero initialization.
            raw_logit = prepared["base_logit"] + gated_parent + (
                online_delta - parent_delta
            )
            pseudo = unlabeled_physics_pseudo_loss(
                raw_logit, prepared, threshold_logit=self.threshold_logit,
                activity_threshold=self.physics_activity_threshold,
                minimum_families=self.physics_minimum_families,
                negative_max_support=self.negative_max_physics_support,
                positive_min_support=self.positive_min_physics_support,
            )
            anchor = F.smooth_l1_loss(raw_logit, persistent_adjusted)
            loss = self.intra_physics_weight * pseudo + self.risk_anchor_weight * anchor
            self.fast_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.fast.parameters() if p.requires_grad], 0.5
            )
            with _scaled_optimizer(self.fast_optimizer, update_scale):
                self.fast_optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pseudo_losses.append(float(pseudo.detach().cpu()))
        return {
            "event": "intra_physics_pseudo_update",
            "shot": self.active_shot,
            "steps": int(steps),
            "loss": sum(losses) / len(losses),
            "pseudo_loss": sum(pseudo_losses) / len(pseudo_losses),
            "physics_weight": self.intra_physics_weight,
            "updated": True,
            "update_scale": update_scale,
            "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
        }

    def seal_shot(self, shot_id: int) -> dict[str, Any]:
        if self.active_shot != int(shot_id) or self.fast is None:
            raise RuntimeError("cannot seal a shot that is not active")
        receipt = {
            "event": "shot_prediction_seal",
            "shot": int(shot_id),
            "persistent_sha256": self.persistent_hash(),
            "fast_sha256": self.fast_hash(),
        }
        self.active_shot = None
        self.fast = None
        self.fast_optimizer = None
        self.pending_shot = int(shot_id)
        return receipt

    def _require_sealed(self, shot_id: int) -> None:
        if self.pending_shot is None or self.pending_shot != int(shot_id):
            raise RuntimeError("a matching sealed shot is required before reveal")

    def discard_without_reveal(self, shot_id: int) -> dict[str, Any]:
        self._require_sealed(shot_id)
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        return {"event": "discard_unqueried", "shot": int(shot_id), "updated": False}

    def finalize_revealed_protection(self, shot_id: int, *, group: str) -> dict[str, Any]:
        """Finalize a queried correct shot without using it as a gradient direction."""
        self._require_sealed(shot_id)
        if group not in ("TP", "TN"):
            raise ValueError("protection-only revealed shots must be TP or TN")
        self.pending_shot = None
        self.last_finalized_shot = int(shot_id)
        return {
            "event": "revealed_protection_only", "shot": int(shot_id),
            "group": group, "updated": False,
            "persistent_sha256": self.persistent_hash(),
        }

    def reveal_and_inter_update(
        self,
        shot_id: int,
        *,
        selected: bool,
        current_batch: Mapping[str, torch.Tensor],
        replay_batch: Mapping[str, torch.Tensor],
        opposite_batch: Mapping[str, torch.Tensor] | None = None,
        steps: int = 2,
        update_scale: float = 1.0,
    ) -> dict[str, Any]:
        self._require_sealed(shot_id)
        if not selected:
            raise PermissionError("supervised inter-shot update requires a selected query")
        if steps <= 0:
            raise ValueError("steps must be positive")
        update_scale = _validate_update_scale(update_scale)
        current = _batch(current_batch, self.device)
        replay = _batch(replay_batch, self.device)
        opposite = (
            _batch(opposite_batch, self.device)
            if opposite_batch is not None else None
        )
        if "label" not in current or "label" not in replay:
            raise KeyError("supervised batches require labels")
        if opposite is not None and "label" not in opposite:
            raise KeyError("opposite-outcome replay requires labels")

        if update_scale == 0.0:
            self.pending_shot = None
            self.last_finalized_shot = int(shot_id)
            return {
                "event": "inter_reveal_update", "shot": int(shot_id),
                "selected": True, "accepted": True, "updated": False,
                "objective": "shot_level_mil_hazard",
                "replay_objective": "sealed_logit_distillation",
                "opposite_outcome_replay": opposite is not None,
                "physics_constraint_weight": self.physics_constraint_weight,
                "physics_increment_gate": self.physics_increment_gate,
                "current_positive_windows": int((current["label"] > 0.5).sum().item()),
                "guard_harms": 0, "steps": 0, "loss": 0.0,
                "update_scale": 0.0,
                "persistent_sha256": self.persistent_hash(),
            }

        state_before = copy.deepcopy(self.persistent.state_dict())
        optimizer_before = copy.deepcopy(self.inter_optimizer.state_dict())
        self.persistent.eval()
        with torch.no_grad():
            guard_before, _delta, _gate, _latent = self._forward(self.persistent, self.guard)
            replay_before, _delta, _gate, _latent = self._forward(
                self.persistent, replay
            )
            guard_label = self.guard["label"].to(torch.bool)
            protected = (guard_before >= self.threshold_logit) == guard_label

        losses = []
        current_positive_windows = int((current["label"] > 0.5).sum().item())
        self.persistent.train()
        for _ in range(int(steps)):
            current_logit, current_delta, _gate, _latent = self._training_forward(
                self.persistent, current
            )
            replay_logit, replay_delta, _gate, _latent = self._training_forward(
                self.persistent, replay
            )
            guard_logit, _guard_delta, _gate, _latent = self._training_forward(
                self.persistent, self.guard
            )
            current_hazard = _deployment_aligned_hazard(
                current_logit, current,
                threshold_logit=self.threshold_logit,
            )
            replay_hazard = replay_distillation_loss(replay_logit, replay_before)
            if opposite is not None:
                opposite_logit, opposite_delta, _gate, _latent = self._training_forward(
                    self.persistent, opposite
                )
                opposite_hazard = _deployment_aligned_hazard(
                    opposite_logit, opposite,
                    threshold_logit=self.threshold_logit,
                )
                supervised_hazard = 0.5 * (current_hazard + opposite_hazard)
                physics = 0.5 * (
                    physics_consistency_loss(
                        current_logit, current, threshold_logit=self.threshold_logit,
                        activity_threshold=self.physics_activity_threshold,
                        minimum_families=self.physics_minimum_families,
                    )
                    + physics_consistency_loss(
                        opposite_logit, opposite, threshold_logit=self.threshold_logit,
                        activity_threshold=self.physics_activity_threshold,
                        minimum_families=self.physics_minimum_families,
                    )
                )
            else:
                opposite_delta = None
                supervised_hazard = current_hazard
                physics = physics_consistency_loss(
                    current_logit, current, threshold_logit=self.threshold_logit,
                    activity_threshold=self.physics_activity_threshold,
                    minimum_families=self.physics_minimum_families,
                )
            hazard = supervised_hazard + replay_hazard
            auxiliary = _next_loss(self.persistent, current) + _next_loss(self.persistent, replay)
            anchor = F.smooth_l1_loss(guard_logit, guard_before)
            residual = 0.5 * (current_delta.square().mean() + replay_delta.square().mean())
            if opposite_delta is not None:
                residual = (2.0 * residual + opposite_delta.square().mean()) / 3.0
            loss = (
                hazard + self.physics_constraint_weight * physics
                + 0.10 * auxiliary + self.guard_anchor_weight * anchor + 0.02 * residual
            )
            self.inter_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.persistent.parameters() if p.requires_grad], 0.5
            )
            with _scaled_optimizer(self.inter_optimizer, update_scale):
                self.inter_optimizer.step()
            losses.append(float(loss.detach().cpu()))

        self.persistent.eval()
        with torch.no_grad():
            guard_after, _delta, _gate, _latent = self._forward(self.persistent, self.guard)
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
            "event": "inter_reveal_update",
            "shot": int(shot_id),
            "selected": True,
            "accepted": bool(accepted),
            "updated": bool(accepted),
            "update_scale": update_scale,
            "objective": (
                "operational_alarm_boundary_mil"
                if "alarm_boundary" in current else "shot_level_mil_hazard"
            ),
            "replay_objective": "sealed_logit_distillation",
            "opposite_outcome_replay": opposite is not None,
            "physics_constraint_weight": self.physics_constraint_weight,
            "physics_increment_gate": self.physics_increment_gate,
            "current_positive_windows": current_positive_windows,
            "guard_harms": harms,
            "steps": int(steps),
            "loss": sum(losses) / len(losses),
            "persistent_sha256": self.persistent_hash(),
        }


__all__ = [
    "TwoTimescaleAdapterController", "module_hash",
    "protect_positive_fast_increment", "CausalReplayBuffer",
    "shot_level_mil_hazard_loss", "replay_distillation_loss",
    "operational_alarm_boundary_loss",
    "physics_consistency_loss",
    "unlabeled_physics_pseudo_loss",
    "gate_signed_online_increment",
]
