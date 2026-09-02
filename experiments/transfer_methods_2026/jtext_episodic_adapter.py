"""Causal, local neural adapters for the independent J-TEXT stream.

This module is deliberately narrower than the historical decision-level
active-Hedge controller and than the global ``LowRankRiskAdapter``.  A
revealed error creates a *local* episode.  The episode can be routed to later
shots only when its causal endpoint representation is closer than the
revealed correct-shot blockers.  The source encoder and source risk head are
never modified.

The module has no HDF5 access and accepts no outcome fields in its pre-reveal
API.  The runner is responsible for calling ``add_error_episode`` only after
``predict -> seal -> query -> reveal``.  This separation makes it possible to
test the causal boundary independently from the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCHEMA_VERSION = "jtext-ordered-episodic-adapter-v2"
ERROR_GROUPS = frozenset({"FP", "FN", "TARDY"})


def _finite_matrix(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    if width is not None and array.shape[1] != int(width):
        raise ValueError(f"{name} has an unexpected width")
    return np.ascontiguousarray(array)


def endpoint_context(
    source_logit: float,
    trend: float,
    elapsed_s: float,
) -> np.ndarray:
    """Return bounded, outcome-blind context available at one endpoint."""

    values = np.asarray(
        [
            math.tanh(float(source_logit) / 4.0),
            math.tanh(float(trend) / 2.0),
            math.tanh(float(elapsed_s) / 2.0),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("endpoint context is not finite")
    return values


def endpoint_descriptor(
    embedding: Sequence[float] | np.ndarray,
    *,
    source_logit: float,
    trend: float,
    elapsed_s: float,
) -> np.ndarray:
    """Build a normalized descriptor using only the current causal endpoint."""

    latent = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if not len(latent) or not np.isfinite(latent).all():
        raise ValueError("embedding must be a finite non-empty vector")
    norm = float(np.linalg.norm(latent))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("embedding norm is too small")
    vector = np.concatenate((latent / norm, endpoint_context(
        source_logit, trend, elapsed_s,
    )))
    vector_norm = float(np.linalg.norm(vector))
    if not math.isfinite(vector_norm) or vector_norm <= 1e-8:
        raise ValueError("endpoint descriptor norm is invalid")
    return np.ascontiguousarray(vector / vector_norm, dtype=np.float32)


def descriptor_matrix(
    embeddings: np.ndarray,
    source_logits: np.ndarray,
    endpoint_indices: Sequence[int] | np.ndarray,
    rate_hz: float,
) -> np.ndarray:
    """Vectorize descriptor construction while retaining endpoint causality."""

    z = _finite_matrix(embeddings, name="embeddings")
    logits = np.asarray(source_logits, dtype=np.float32).reshape(-1)
    indices = np.asarray(endpoint_indices, dtype=np.int64).reshape(-1)
    if len(logits) != len(z) or len(indices) != len(z):
        raise ValueError("endpoint descriptor arrays are misaligned")
    if not math.isfinite(float(rate_hz)) or float(rate_hz) <= 0.0:
        raise ValueError("rate_hz must be positive and finite")
    rows = []
    for position, index in enumerate(indices.tolist()):
        if int(index) != position:
            raise ValueError("endpoint indices must be contiguous and ordered")
        trend = 0.0 if position == 0 else float(logits[position] - logits[position - 1])
        rows.append(endpoint_descriptor(
            z[position], source_logit=float(logits[position]), trend=trend,
            elapsed_s=float(index) / float(rate_hz),
        ))
    return np.stack(rows).astype(np.float32, copy=False)


def hidden_prototype(embeddings: np.ndarray) -> np.ndarray:
    """Return a unit hidden-state prototype for one revealed error episode.

    The prototype is computed from the selected causal endpoints only.  A
    degenerate episode is rejected rather than silently replaced by a default
    vector, because a fabricated prototype would make the routing receipt
    impossible to interpret.
    """

    matrix = _finite_matrix(embeddings, name="prototype embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("prototype embeddings contain a zero-norm row")
    normalized = matrix / norms
    value = np.mean(normalized, axis=0)
    value_norm = float(np.linalg.norm(value))
    if not math.isfinite(value_norm) or value_norm <= 1e-8:
        raise ValueError("prototype has zero or non-finite norm")
    return np.ascontiguousarray(value / value_norm, dtype=np.float32)


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


class EpisodicRiskAdapter(nn.Module):
    """Small nonlinear residual that can alter a local hidden representation.

    ``latent_up`` and ``risk_up`` are zero initialized, so a new episode is an
    exact identity before fitting.  The risk path consumes the transformed
    latent state; consequently a successful fit can be inspected for both a
    risk residual and a representation delta rather than only a post-hoc
    threshold shift.
    """

    def __init__(
        self,
        dimension: int = 64,
        rank: int = 4,
        max_logit_delta: float = 2.0,
        latent_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if int(dimension) <= 0 or int(rank) <= 0:
            raise ValueError("adapter dimension and rank must be positive")
        for name, value in (
            ("max_logit_delta", max_logit_delta),
            ("latent_scale", latent_scale),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.max_logit_delta = float(max_logit_delta)
        self.latent_scale = float(latent_scale)
        self.latent_down = nn.Linear(self.dimension, self.rank, bias=False)
        self.latent_up = nn.Linear(self.rank, self.dimension, bias=False)
        self.risk_down = nn.Linear(self.dimension + 3, self.rank, bias=False)
        self.risk_up = nn.Linear(self.rank, 1, bias=True)
        nn.init.normal_(self.latent_down.weight, std=0.02)
        nn.init.normal_(self.risk_down.weight, std=0.02)
        nn.init.zeros_(self.latent_up.weight)
        nn.init.zeros_(self.risk_up.weight)
        nn.init.zeros_(self.risk_up.bias)

    def transform(self, frozen: Tensor) -> Tensor:
        if frozen.ndim != 2 or frozen.shape[1] != self.dimension:
            raise ValueError("adapter hidden input has an unexpected shape")
        return frozen + self.latent_scale * self.latent_up(
            torch.tanh(self.latent_down(frozen))
        )

    def forward(
        self,
        frozen: Tensor,
        base_logit: Tensor,
        context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if base_logit.ndim != 1 or context.ndim != 2:
            raise ValueError("adapter inputs have an unexpected rank")
        if frozen.shape[0] != base_logit.shape[0] or context.shape != (frozen.shape[0], 3):
            raise ValueError("adapter inputs are misaligned")
        transformed = self.transform(frozen)
        features = torch.cat((transformed, context), dim=1)
        residual = self.max_logit_delta * torch.tanh(
            self.risk_up(torch.tanh(self.risk_down(features))).squeeze(-1)
        )
        return base_logit + residual, transformed, residual


class PrototypeConditionedRiskAdapter(nn.Module):
    """Local residual adapter whose risk path is conditioned on an episode.

    ``prototype`` is a frozen, unit-norm summary of the revealed error
    trajectory.  The risk path receives the transformed displacement from
    that prototype and the causal endpoint context.  This makes the learned
    correction specific to a trajectory family, while the low-rank latent
    path still gives the update an inspectable representation effect.
    """

    def __init__(
        self,
        prototype: np.ndarray | Tensor,
        *,
        descriptor_prototype: np.ndarray | Tensor | None = None,
        direction: float = 1.0,
        rank: int = 4,
        max_logit_delta: float = 2.0,
        latent_scale: float = 0.10,
        locality_center: float = 0.80,
        locality_temperature: float = 12.0,
    ) -> None:
        super().__init__()
        if isinstance(prototype, Tensor):
            value = prototype.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        else:
            value = np.asarray(prototype, dtype=np.float32).reshape(-1)
        if not len(value) or not np.isfinite(value).all():
            raise ValueError("prototype must be a finite non-empty vector")
        norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError("prototype norm must be positive and finite")
        if descriptor_prototype is None:
            descriptor_value = np.concatenate((value / norm, np.zeros(3, dtype=np.float32)))
        elif isinstance(descriptor_prototype, Tensor):
            descriptor_value = descriptor_prototype.detach().cpu().numpy().astype(
                np.float32, copy=False,
            )
        else:
            descriptor_value = np.asarray(descriptor_prototype, dtype=np.float32)
        if descriptor_value.ndim == 1:
            descriptor_value = descriptor_value[None, :]
        if (
            descriptor_value.ndim != 2
            or descriptor_value.shape[1] != len(value) + 3
            or not np.isfinite(descriptor_value).all()
        ):
            raise ValueError("descriptor prototype has an unexpected shape")
        descriptor_norm = np.linalg.norm(descriptor_value, axis=1, keepdims=True)
        if np.any(descriptor_norm <= 1e-8) or not np.isfinite(descriptor_norm).all():
            raise ValueError("descriptor prototype norm must be positive and finite")
        if int(rank) <= 0:
            raise ValueError("adapter rank must be positive")
        if not math.isfinite(float(direction)) or abs(float(direction)) != 1.0:
            raise ValueError("direction must be exactly -1 or +1")
        for name, setting in (
            ("max_logit_delta", max_logit_delta),
            ("latent_scale", latent_scale),
            ("locality_temperature", locality_temperature),
        ):
            if not math.isfinite(float(setting)) or float(setting) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < float(locality_center) < 1.0:
            raise ValueError("locality_center must lie in (0, 1)")
        self.dimension = int(len(value))
        self.rank = int(rank)
        self.max_logit_delta = float(max_logit_delta)
        self.latent_scale = float(latent_scale)
        self.locality_center = float(locality_center)
        self.locality_temperature = float(locality_temperature)
        self.register_buffer(
            "prototype", torch.from_numpy(np.ascontiguousarray(value / norm)),
        )
        self.register_buffer(
            "descriptor_prototype",
            torch.from_numpy(np.ascontiguousarray(descriptor_value / descriptor_norm)),
        )
        self.register_buffer("direction", torch.tensor(float(direction), dtype=torch.float32))
        self.latent_down = nn.Linear(self.dimension, self.rank, bias=False)
        self.latent_up = nn.Linear(self.rank, self.dimension, bias=False)
        # transformed-prototype displacement, three context values, hidden
        # cosine and descriptor cosine.  The explicit locality feature makes
        # the learned correction decay outside the revealed trajectory family.
        self.risk_down = nn.Linear(self.dimension + 5, self.rank, bias=False)
        self.risk_up = nn.Linear(self.rank, 1, bias=True)
        # A signed, bounded local correction is learned from the revealed error
        # group.  It is zero before fitting and cannot act outside the locality
        # gate; this is not a global threshold update.
        self.correction_scale = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.latent_down.weight, std=0.02)
        nn.init.normal_(self.risk_down.weight, std=0.02)
        nn.init.zeros_(self.latent_up.weight)
        nn.init.zeros_(self.risk_up.weight)
        nn.init.zeros_(self.risk_up.bias)

    def transform(self, frozen: Tensor) -> Tensor:
        if frozen.ndim != 2 or frozen.shape[1] != self.dimension:
            raise ValueError("adapter hidden input has an unexpected shape")
        return frozen + self.latent_scale * self.latent_up(
            torch.tanh(self.latent_down(frozen))
        )

    def forward(
        self,
        frozen: Tensor,
        base_logit: Tensor,
        context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if base_logit.ndim != 1 or context.ndim != 2:
            raise ValueError("adapter inputs have an unexpected rank")
        if frozen.shape[0] != base_logit.shape[0] or context.shape != (frozen.shape[0], 3):
            raise ValueError("adapter inputs are misaligned")
        raw_transformed = self.transform(frozen)
        prototype = self.prototype.to(
            device=raw_transformed.device, dtype=raw_transformed.dtype,
        )
        descriptor_prototype = self.descriptor_prototype.to(
            device=raw_transformed.device, dtype=raw_transformed.dtype,
        )
        hidden_cosine = F.cosine_similarity(
            raw_transformed, prototype.unsqueeze(0).expand_as(raw_transformed), dim=1,
        )
        raw_descriptor = torch.cat((
            F.normalize(raw_transformed, dim=1, eps=1e-8), context,
        ), dim=1)
        raw_descriptor = F.normalize(raw_descriptor, dim=1, eps=1e-8)
        descriptor_cosine = torch.max(
            raw_descriptor @ descriptor_prototype.transpose(0, 1), dim=1,
        ).values
        locality = torch.sigmoid(
            float(self.locality_temperature) *
            (descriptor_cosine - float(self.locality_center))
        )
        # Only the local part of the latent update is exposed to later shots.
        transformed = frozen + locality.unsqueeze(1) * (raw_transformed - frozen)
        displacement = transformed - prototype.unsqueeze(0)
        cosine = F.cosine_similarity(
            transformed, prototype.unsqueeze(0).expand_as(transformed), dim=1,
        ).unsqueeze(1)
        features = torch.cat((displacement, context, cosine, descriptor_cosine.unsqueeze(1)), dim=1)
        nonlinear = self.max_logit_delta * torch.tanh(
            self.risk_up(torch.tanh(self.risk_down(features))).squeeze(-1)
        )
        signed = self.direction * self.max_logit_delta * torch.tanh(self.correction_scale)
        residual = locality * (signed + 0.5 * nonlinear)
        return base_logit + residual, transformed, residual


def adapter_parameter_count(adapter: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in adapter.parameters()))


def _relative_change(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name in before:
        left = before[name].detach().float()
        right = after[name].detach().float()
        numerator += float((right - left).square().sum())
        denominator += float(left.square().sum())
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1.0)


def _project_state(
    before: dict[str, Tensor], after: dict[str, Tensor], maximum: float,
) -> tuple[float, float, bool]:
    if not math.isfinite(float(maximum)) or float(maximum) <= 0.0:
        raise ValueError("adapter relative radius must be positive and finite")
    raw = _relative_change(before, after)
    if not math.isfinite(raw):
        return raw, raw, False
    if raw <= float(maximum):
        return raw, raw, False
    factor = float(maximum) / raw
    with torch.no_grad():
        for name, value in after.items():
            value.copy_(before[name] + factor * (value - before[name]))
    return raw, _relative_change(before, after), True


@dataclass(frozen=True)
class EpisodeRoute:
    """Outcome-free route decision returned before a current reveal."""

    episode_id: str | None
    group: str | None
    similarity: float
    blocker_similarity: float | None
    support_count: int
    accepted: bool
    reason: str
    outcome_free: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "group": self.group,
            "similarity": float(self.similarity),
            "blocker_similarity": (
                None if self.blocker_similarity is None
                else float(self.blocker_similarity)
            ),
            "support_count": int(self.support_count),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "outcome_free": True,
        }


@dataclass
class _Episode:
    episode_id: str
    source_shot: int
    source_index: int
    group: str
    prototypes: np.ndarray
    adapter: nn.Module
    source_risk_min: float
    source_risk_max: float


class OrderedEpisodicBank:
    """Chronological local-adapter registry with blocker-aware routing."""

    def __init__(
        self,
        *,
        threshold: float,
        minimum_similarity: float = 0.90,
        minimum_margin: float = 0.05,
        risk_margin: float = 0.18,
        minimum_episode_support: int = 1,
        maximum_episodes: int = 32,
    ) -> None:
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if not 0.0 < float(minimum_similarity) <= 1.0:
            raise ValueError("minimum similarity must lie in (0, 1]")
        if not math.isfinite(float(minimum_margin)) or float(minimum_margin) < 0.0:
            raise ValueError("minimum margin must be finite and non-negative")
        if not math.isfinite(float(risk_margin)) or float(risk_margin) < 0.0:
            raise ValueError("risk margin must be finite and non-negative")
        if int(minimum_episode_support) <= 0:
            raise ValueError("minimum_episode_support must be positive")
        if int(maximum_episodes) <= 0:
            raise ValueError("maximum_episodes must be positive")
        self.threshold = float(threshold)
        self.minimum_similarity = float(minimum_similarity)
        self.minimum_margin = float(minimum_margin)
        self.risk_margin = float(risk_margin)
        self.minimum_episode_support = int(minimum_episode_support)
        self.maximum_episodes = int(maximum_episodes)
        self._episodes: list[_Episode] = []
        self._blockers: list[tuple[int, int, np.ndarray]] = []
        self._committed_shots: set[int] = set()

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def blocker_count(self) -> int:
        return len(self._blockers)

    def add_blocker(
        self,
        *,
        shot_id: int,
        stream_index: int,
        descriptors: np.ndarray,
    ) -> dict[str, Any]:
        """Register a revealed correct shot as a negative routing example."""

        shot = int(shot_id)
        index = int(stream_index)
        if shot in self._committed_shots:
            raise RuntimeError("a shot cannot be committed to memory twice")
        matrix = _finite_matrix(descriptors, name="blocker descriptors")
        self._blockers.append((shot, index, matrix.copy()))
        self._committed_shots.add(shot)
        return {
            "schema_version": SCHEMA_VERSION,
            "shot": shot,
            "stream_index": index,
            "kind": "correct_blocker",
            "points": int(len(matrix)),
            "blocker_count": self.blocker_count,
        }

    def add_error_episode(
        self,
        *,
        shot_id: int,
        stream_index: int,
        group: str,
        descriptors: np.ndarray,
        adapter: nn.Module,
        source_risk_min: float,
        source_risk_max: float,
    ) -> dict[str, Any]:
        """Commit a fitted adapter only after a selected reveal."""

        shot = int(shot_id)
        index = int(stream_index)
        if group not in ERROR_GROUPS:
            raise ValueError("only revealed error groups can create episodes")
        if shot in self._committed_shots:
            raise RuntimeError("a shot cannot be committed to memory twice")
        if int(index) < 0:
            raise ValueError("stream_index must be non-negative")
        prototypes = _finite_matrix(descriptors, name="episode descriptors")
        if prototypes.shape[1] < 2:
            raise ValueError("episode descriptors are too narrow")
        norms = np.linalg.norm(prototypes, axis=1)
        if np.any(norms <= 1e-8):
            raise ValueError("episode descriptors must have non-zero rows")
        if not math.isfinite(float(source_risk_min)) or not math.isfinite(float(source_risk_max)):
            raise ValueError("source risk range must be finite")
        if float(source_risk_min) > float(source_risk_max):
            raise ValueError("source risk range is reversed")
        digest = hashlib.sha256()
        digest.update(f"{shot}:{index}:{group}".encode("utf-8"))
        digest.update(np.ascontiguousarray(prototypes).tobytes())
        episode_id = digest.hexdigest()[:24]
        if any(item.episode_id == episode_id for item in self._episodes):
            raise RuntimeError("duplicate episode identity")
        self._episodes.append(_Episode(
            episode_id=episode_id, source_shot=shot, source_index=index,
            group=str(group), prototypes=prototypes.copy(),
            adapter=copy.deepcopy(adapter).eval(),
            source_risk_min=float(source_risk_min),
            source_risk_max=float(source_risk_max),
        ))
        self._committed_shots.add(shot)
        if len(self._episodes) > self.maximum_episodes:
            self._episodes = self._episodes[-self.maximum_episodes:]
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "shot": shot,
            "stream_index": index,
            "group": str(group),
            "prototype_points": int(len(prototypes)),
            "adapter_sha256": module_sha256(adapter),
            "episode_count": self.episode_count,
        }

    @staticmethod
    def _max_similarity(query: np.ndarray, prototypes: np.ndarray) -> float:
        value = np.asarray(query, dtype=np.float32).reshape(-1)
        matrix = _finite_matrix(prototypes, name="stored prototypes")
        if matrix.shape[1] != len(value):
            raise ValueError("descriptor dimensions do not match")
        value = value / max(float(np.linalg.norm(value)), 1e-8)
        rows = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        return float(np.max(rows @ value))

    def route(
        self,
        *,
        stream_index: int,
        descriptor: np.ndarray,
        source_probability: float,
    ) -> tuple[_Episode | None, EpisodeRoute]:
        """Route a current endpoint using only strict-past memory."""

        index = int(stream_index)
        probability = float(source_probability)
        if index < 0 or not math.isfinite(probability):
            raise ValueError("route inputs are invalid")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("source_probability must lie in [0, 1]")
        query = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        if not len(query) or not np.isfinite(query).all():
            raise ValueError("route descriptor must be finite and non-empty")
        candidates: list[tuple[float, float | None, _Episode]] = []
        for episode in self._episodes:
            if episode.source_index >= index:
                continue
            if episode.group == "FP":
                in_risk_regime = probability >= self.threshold - self.risk_margin
            else:
                in_risk_regime = probability <= self.threshold + self.risk_margin
            if not in_risk_regime:
                continue
            similarity = self._max_similarity(query, episode.prototypes)
            if similarity < self.minimum_similarity:
                continue
            blocker_similarity: float | None = None
            if self._blockers:
                past_blockers = [
                    matrix for _shot, blocker_index, matrix in self._blockers
                    if blocker_index < index
                ]
                if past_blockers:
                    blocker_similarity = max(
                        self._max_similarity(query, matrix) for matrix in past_blockers
                    )
                    if similarity - blocker_similarity < self.minimum_margin:
                        continue
            candidates.append((similarity, blocker_similarity, episode))
        if not candidates:
            return None, EpisodeRoute(
                episode_id=None, group=None, similarity=0.0,
                blocker_similarity=None, support_count=0, accepted=False,
                reason="no_past_compatible_episode",
            )
        supported_candidates: list[tuple[float, float | None, int, _Episode]] = []
        for similarity, blocker_similarity, episode in candidates:
            support_count = sum(
                1 for other in self._episodes
                if other.group == episode.group
                and other.source_index < index
                and self._max_similarity(query, other.prototypes) >= self.minimum_similarity
            )
            if support_count >= self.minimum_episode_support:
                supported_candidates.append(
                    (similarity, blocker_similarity, support_count, episode)
                )
        if not supported_candidates:
            return None, EpisodeRoute(
                episode_id=None, group=None, similarity=0.0,
                blocker_similarity=None, support_count=0, accepted=False,
                reason="insufficient_past_episode_support",
            )
        similarity, blocker_similarity, support_count, selected = max(
            supported_candidates, key=lambda item: (item[0], item[2], -item[3].source_index)
        )
        return selected, EpisodeRoute(
            episode_id=selected.episode_id, group=selected.group,
            similarity=similarity, blocker_similarity=blocker_similarity,
            support_count=support_count, accepted=True,
            reason="past_episode_support_and_blocker_gate",
        )

    def state_dict(self) -> dict[str, Any]:
        """Return an auditable registry without exposing outcome fields pre-reveal."""

        return {
            "schema_version": SCHEMA_VERSION,
            "threshold": self.threshold,
            "minimum_similarity": self.minimum_similarity,
            "minimum_margin": self.minimum_margin,
            "risk_margin": self.risk_margin,
            "minimum_episode_support": self.minimum_episode_support,
            "maximum_episodes": self.maximum_episodes,
            "episodes": [
                {
                    "episode_id": item.episode_id,
                    "source_shot": item.source_shot,
                    "source_index": item.source_index,
                    "group": item.group,
                    "prototype_points": int(len(item.prototypes)),
                    "adapter_sha256": module_sha256(item.adapter),
                    "source_risk_min": item.source_risk_min,
                    "source_risk_max": item.source_risk_max,
                }
                for item in self._episodes
            ],
            "blockers": [
                {"shot": shot, "stream_index": index, "points": int(len(matrix))}
                for shot, index, matrix in self._blockers
            ],
        }


def fit_episode_adapter(
    adapter: nn.Module,
    *,
    embeddings: np.ndarray,
    base_logits: np.ndarray,
    contexts: np.ndarray,
    labels: np.ndarray,
    device: str,
    learning_rate: float = 1e-2,
    steps: int = 8,
    max_relative_change: float = 0.20,
    hard_example_weight: float = 4.0,
) -> dict[str, Any]:
    """Fit one local adapter and return a transactional receipt.

    The caller supplies only post-reveal samples.  This function never reads
    shot IDs, outcomes, or terminal times.  It returns an adapter that is
    still owned by the caller; committing it to a bank is a separate action.
    """

    z_np = _finite_matrix(embeddings, name="episode embeddings")
    base_np = np.asarray(base_logits, dtype=np.float32).reshape(-1)
    context_np = _finite_matrix(contexts, name="episode contexts", width=3)
    labels_np = np.asarray(labels, dtype=np.float32).reshape(-1)
    if len(z_np) != len(base_np) or len(z_np) != len(context_np) or len(z_np) != len(labels_np):
        raise ValueError("episode fit arrays are misaligned")
    if not np.isin(labels_np, [0.0, 1.0]).all():
        raise ValueError("episode labels must be binary")
    for name, value in (
        ("learning_rate", learning_rate),
        ("hard_example_weight", hard_example_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if int(steps) <= 0:
        raise ValueError("episode fit steps must be positive")
    if not 0.0 < float(max_relative_change):
        raise ValueError("episode fit radius must be positive")
    if z_np.shape[1] != adapter.dimension:
        raise ValueError("episode embedding dimension does not match adapter")
    before = copy.deepcopy(adapter.state_dict())
    model = adapter.to(device)
    model.train()
    z = torch.from_numpy(z_np).to(device)
    base = torch.from_numpy(base_np).to(device)
    context = torch.from_numpy(context_np).to(device)
    target = torch.from_numpy(labels_np).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    losses: list[float] = []
    for _ in range(int(steps)):
        adjusted, transformed, residual = model(z, base, context)
        hardness = torch.where(
            target > 0.5,
            1.0 - torch.sigmoid(base.detach()),
            torch.sigmoid(base.detach()),
        )
        weights = 1.0 + float(hard_example_weight) * hardness
        bce = F.binary_cross_entropy_with_logits(adjusted, target, reduction="none")
        representation_penalty = (transformed - z).square().mean()
        residual_penalty = residual.square().mean()
        loss = (weights * bce).sum() / weights.sum().clamp_min(1e-6)
        loss = loss + 0.01 * residual_penalty + 0.001 * representation_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    after = model.state_dict()
    finite = all(bool(torch.isfinite(value).all()) for value in after.values())
    raw_change = float("nan")
    realized_change = float("nan")
    projected = False
    if finite:
        raw_change, realized_change, projected = _project_state(
            before, after, float(max_relative_change),
        )
        finite = math.isfinite(realized_change)
    accepted = bool(finite and realized_change > 0.0)
    if not accepted:
        model.load_state_dict(before, strict=True)
    model.eval()
    with torch.no_grad():
        adjusted, transformed, residual = model(z, base, context)
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted": accepted,
        "reason": "finite_local_fit" if accepted else "nonfinite_or_noop",
        "loss": float(np.mean(losses)) if losses else 0.0,
        "raw_relative_parameter_change": (
            float(raw_change) if math.isfinite(raw_change) else None
        ),
        "relative_parameter_change": float(realized_change if accepted else 0.0),
        "projected_to_radius": bool(projected and accepted),
        "mean_abs_logit_residual": float(residual.abs().mean().cpu()),
        "mean_representation_delta": float((transformed - z).norm(dim=1).mean().cpu()),
        "adapter_sha256": module_sha256(model),
        "points": int(len(labels_np)),
    }


__all__ = [
    "ERROR_GROUPS",
    "EpisodicRiskAdapter",
    "PrototypeConditionedRiskAdapter",
    "EpisodeRoute",
    "OrderedEpisodicBank",
    "SCHEMA_VERSION",
    "adapter_parameter_count",
    "descriptor_matrix",
    "endpoint_context",
    "endpoint_descriptor",
    "fit_episode_adapter",
    "hidden_prototype",
    "module_sha256",
]
