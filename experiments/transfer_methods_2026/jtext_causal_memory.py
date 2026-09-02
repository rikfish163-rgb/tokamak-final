"""Causal, typed trajectory memory for the independent J-TEXT online branch.

The memory in this module is deliberately narrower than a generic nearest
neighbour classifier.  A shot can enter the memory only after its prediction
has been sealed and its query has been revealed.  At inference time a match is
accepted only when it has support from distinct earlier shots and when the
positive/negative evidence is not contradictory.  This makes the component
useful for testing whether a queried error trajectory can be reused, while
keeping it bounded, auditable, and reversible.

No outcome or terminal-time field is accepted by :func:`endpoint_feature` or
:func:`causal_query_components`; those values belong to the post-reveal path
in the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "jtext-causal-typed-memory-v1"


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def endpoint_feature(
    embedding: Sequence[float] | np.ndarray,
    *,
    source_logit: float,
    trend: float,
    elapsed_s: float,
) -> np.ndarray:
    """Encode one causal endpoint without using a terminal label.

    The latent direction carries the source model's local representation.  The
    three bounded scalars retain the risk regime, local temporal direction and
    elapsed time.  Elapsed time is measured from the shot start and is
    available online; total shot duration is intentionally not used.
    """

    latent = _finite_array(embedding, name="embedding")
    scalars = np.asarray(
        [
            math.tanh(float(source_logit) / 4.0),
            math.tanh(float(trend) / 2.0),
            math.tanh(float(elapsed_s) / 2.0),
        ],
        dtype=np.float64,
    )
    vector = np.concatenate((latent / max(float(np.linalg.norm(latent)), 1e-12), scalars))
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("endpoint feature has an invalid norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float64)


@dataclass(frozen=True)
class MemoryMatch:
    """A typed nearest-support result returned in every correction receipt."""

    label: int
    support_shots: int
    support_points: int
    mean_similarity: float
    maximum_similarity: float
    local_probability: float


class CausalTypedTrajectoryMemory:
    """Bounded post-reveal endpoint memory with positive/negative typing."""

    def __init__(
        self,
        *,
        capacity: int = 512,
        neighbors: int = 3,
        minimum_similarity: float = 0.90,
        minimum_support_shots: int = 2,
        agreement_margin: float = 0.10,
        maximum_logit_shift: float = 4.5,
        candidate_margin: float = 0.18,
        positive_floor: float = 0.01,
    ) -> None:
        if int(capacity) <= 0 or int(neighbors) <= 0:
            raise ValueError("trajectory memory capacity and neighbors must be positive")
        if not 0.0 < float(minimum_similarity) < 1.0:
            raise ValueError("minimum similarity must lie in (0, 1)")
        if int(minimum_support_shots) <= 0:
            raise ValueError("minimum support shots must be positive")
        for name, value in (
            ("agreement_margin", agreement_margin),
            ("maximum_logit_shift", maximum_logit_shift),
            ("candidate_margin", candidate_margin),
            ("positive_floor", positive_floor),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"memory {name} must be finite and non-negative")
        if float(positive_floor) >= 1.0:
            raise ValueError("memory positive floor must be below one")
        self.capacity = int(capacity)
        self.neighbors = int(neighbors)
        self.minimum_similarity = float(minimum_similarity)
        self.minimum_support_shots = int(minimum_support_shots)
        self.agreement_margin = float(agreement_margin)
        self.maximum_logit_shift = float(maximum_logit_shift)
        self.candidate_margin = float(candidate_margin)
        self.positive_floor = float(positive_floor)
        self._features: list[np.ndarray] = []
        self._labels: list[int] = []
        self._shots: list[int] = []
        self._roles: list[str] = []
        self._last_shot: int | None = None

    @property
    def size(self) -> int:
        return len(self._labels)

    @property
    def shot_count(self) -> int:
        return len(set(self._shots))

    @property
    def feature_dimension(self) -> int | None:
        return None if not self._features else int(self._features[0].shape[0])

    def add_after_reveal(
        self,
        shot_id: int,
        features: Sequence[Sequence[float] | np.ndarray],
        labels: Sequence[int | float],
        *,
        roles: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Commit endpoint supports after the selected shot is revealed.

        A shot may be committed once and commits must follow the stream order.
        The method has no way to read a label itself; the caller must provide
        the post-reveal endpoint targets explicitly.
        """

        shot = int(shot_id)
        if self._last_shot is not None and shot <= self._last_shot:
            raise RuntimeError("trajectory memory commits must be strictly chronological")
        if len(features) != len(labels) or not len(features):
            raise ValueError("memory features and labels must have equal non-zero length")
        if roles is not None and len(roles) != len(features):
            raise ValueError("memory roles must align with features")
        converted: list[np.ndarray] = []
        converted_labels: list[int] = []
        for feature, label in zip(features, labels):
            value = _finite_array(feature, name="memory feature")
            if self._features and value.shape != self._features[0].shape:
                raise ValueError("trajectory memory feature dimension changed")
            numeric_label = float(label)
            if numeric_label not in (0.0, 1.0):
                raise ValueError("trajectory memory labels must be binary")
            converted.append(np.ascontiguousarray(value, dtype=np.float64))
            converted_labels.append(int(numeric_label))
        role_values = ["endpoint" for _ in converted] if roles is None else [str(v) for v in roles]
        self._features.extend(converted)
        self._labels.extend(converted_labels)
        self._shots.extend([shot] * len(converted))
        self._roles.extend(role_values)
        if len(self._features) > self.capacity:
            start = len(self._features) - self.capacity
            self._features = self._features[start:]
            self._labels = self._labels[start:]
            self._shots = self._shots[start:]
            self._roles = self._roles[start:]
        self._last_shot = shot
        return {
            "schema_version": SCHEMA_VERSION,
            "shot": shot,
            "added_points": len(converted),
            "memory_size": self.size,
            "memory_shots": self.shot_count,
        }

    def _match(
        self,
        feature: np.ndarray,
        label: int,
        *,
        error_only: bool = False,
    ) -> MemoryMatch | None:
        if not self._features:
            return None
        query = _finite_array(feature, name="query feature")
        matrix = np.stack(self._features, axis=0)
        norms = np.linalg.norm(matrix, axis=1)
        query_norm = max(float(np.linalg.norm(query)), 1e-12)
        similarities = (matrix @ query) / np.maximum(norms * query_norm, 1e-12)
        candidates = np.flatnonzero(np.asarray(self._labels, dtype=np.int64) == int(label))
        if error_only:
            candidates = np.asarray([
                index for index in candidates
                if str(self._roles[int(index)]).startswith("error_")
            ], dtype=np.int64)
        if not len(candidates):
            return None
        # Keep at most one point from each earlier shot.  Multiple endpoints
        # from one queried shot must not masquerade as independent evidence.
        best_by_shot: dict[int, int] = {}
        for index in candidates:
            shot = self._shots[int(index)]
            if shot not in best_by_shot or similarities[int(index)] > similarities[best_by_shot[shot]]:
                best_by_shot[shot] = int(index)
        eligible = [
            index for index in best_by_shot.values()
            if float(similarities[index]) >= self.minimum_similarity
        ]
        eligible.sort(key=lambda index: float(similarities[index]), reverse=True)
        eligible = eligible[: self.neighbors]
        if not eligible:
            return None
        weights = np.maximum(
            np.asarray([similarities[index] for index in eligible], dtype=np.float64)
            - self.minimum_similarity,
            1e-8,
        )
        probability = float(np.average(
            np.asarray([self._labels[index] for index in eligible], dtype=np.float64),
            weights=weights,
        ))
        return MemoryMatch(
            label=int(label),
            support_shots=len({self._shots[index] for index in eligible}),
            support_points=len(eligible),
            mean_similarity=float(np.mean([similarities[index] for index in eligible])),
            maximum_similarity=float(np.max([similarities[index] for index in eligible])),
            local_probability=probability,
        )

    def correct(
        self,
        feature: Sequence[float] | np.ndarray,
        source_logit: float,
        *,
        threshold: float,
        error_only: bool = False,
        allow_negative: bool = True,
        allow_positive: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Return a bounded correction based only on earlier revealed shots."""

        base = float(source_logit)
        if not math.isfinite(base):
            raise ValueError("source logit must be finite")
        threshold_value = float(threshold)
        if not 0.0 < threshold_value < 1.0:
            raise ValueError("memory threshold must lie in (0, 1)")
        if not isinstance(allow_negative, (bool, np.bool_)):
            raise ValueError("allow_negative must be boolean")
        if not isinstance(allow_positive, (bool, np.bool_)):
            raise ValueError("allow_positive must be boolean")
        positive = (
            self._match(np.asarray(feature), 1, error_only=error_only)
            if bool(allow_positive) else None
        )
        negative = (
            self._match(np.asarray(feature), 0, error_only=error_only)
            if bool(allow_negative) else None
        )
        receipt: dict[str, Any] = {
            "applied": False,
            "reason": "no_local_support",
            "memory_size": self.size,
            "positive_support_shots": 0 if positive is None else positive.support_shots,
            "negative_support_shots": 0 if negative is None else negative.support_shots,
            "positive_similarity": None if positive is None else positive.mean_similarity,
            "negative_similarity": None if negative is None else negative.mean_similarity,
            "logit_shift": 0.0,
            "error_only": bool(error_only),
            "allow_negative": bool(allow_negative),
            "allow_positive": bool(allow_positive),
        }
        if positive is None and negative is None:
            return base, receipt
        probability = 1.0 / (1.0 + math.exp(-max(min(base, 40.0), -40.0)))
        chosen: MemoryMatch | None = None
        route: str | None = None
        if (
            negative is not None
            and negative.support_shots >= self.minimum_support_shots
            and probability >= threshold_value - self.candidate_margin
            and probability <= threshold_value + self.candidate_margin
            and (
                positive is None
                or positive.support_shots < self.minimum_support_shots
                or negative.local_probability + self.agreement_margin
                < positive.local_probability
            )
        ):
            # Negative support is represented by label zero, so its local
            # probability should be low.  The explicit comparison above is
            # written in terms of confidence to make contradictory evidence
            # fail closed.
            chosen = negative
            route = "negative_error_memory"
        elif (
            positive is not None
            and positive.support_shots >= self.minimum_support_shots
            and probability < threshold_value
            and probability >= max(0.0, threshold_value - self.candidate_margin)
            and (
                negative is None
                or negative.support_shots < self.minimum_support_shots
                or positive.local_probability
                > negative.local_probability + self.agreement_margin
            )
        ):
            chosen = positive
            route = "positive_error_memory"
        if chosen is None:
            receipt["reason"] = "typed_support_conflict_or_gate"
            return base, receipt
        confidence = float(np.clip(
            (chosen.mean_similarity - self.minimum_similarity)
            / max(1.0 - self.minimum_similarity, 1e-8),
            0.0,
            1.0,
        ))
        confidence *= float(np.clip(
            chosen.support_shots / max(self.minimum_support_shots, 1), 0.0, 1.0,
        ))
        target_probability = 0.05 if chosen.label == 0 else 0.95
        target_logit = math.log(target_probability / (1.0 - target_probability))
        shift = float(np.clip(
            confidence * (target_logit - base),
            -self.maximum_logit_shift,
            self.maximum_logit_shift,
        ))
        corrected = base + shift
        receipt.update({
            "applied": bool(abs(shift) > 0.0),
            "reason": route,
            "support_shots": chosen.support_shots,
            "support_points": chosen.support_points,
            "mean_similarity": chosen.mean_similarity,
            "maximum_similarity": chosen.maximum_similarity,
            "local_probability": chosen.local_probability,
            "confidence": confidence,
            "logit_shift": shift,
        })
        return corrected, receipt

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "capacity": self.capacity,
            "neighbors": self.neighbors,
            "minimum_similarity": self.minimum_similarity,
            "minimum_support_shots": self.minimum_support_shots,
            "agreement_margin": self.agreement_margin,
            "maximum_logit_shift": self.maximum_logit_shift,
            "candidate_margin": self.candidate_margin,
            "positive_floor": self.positive_floor,
            "features": [value.tolist() for value in self._features],
            "labels": list(self._labels),
            "shots": list(self._shots),
            "roles": list(self._roles),
            "last_shot": self._last_shot,
        }


def causal_query_components(
    risk: Sequence[float] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    novelty: float,
) -> dict[str, float]:
    """Compute a label-free query value with an explicit low-risk reserve.

    The low-risk novelty term is intentional: a selector based only on
    uncertainty cannot discover a false-negative trajectory whose source risk
    is uniformly small.  It uses no outcome, terminal time, or future shot.
    """

    probabilities = _finite_array(risk, name="risk")
    source_logits = _finite_array(logits, name="logits")
    if probabilities.shape != source_logits.shape:
        raise ValueError("risk and logits must have equal shape")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("query threshold must lie in (0, 1)")
    novelty_value = float(novelty)
    if not math.isfinite(novelty_value):
        raise ValueError("query novelty must be finite")
    novelty_value = float(np.clip(novelty_value, 0.0, 1.0))
    if not len(probabilities):
        return {
            "uncertainty": 0.0,
            "trajectory_conflict": 0.0,
            "novelty": novelty_value,
            "low_risk_novelty": novelty_value,
            "query_score": 0.25 * novelty_value,
            "max_risk": 0.0,
        }
    threshold_logit = math.log(float(threshold) / (1.0 - float(threshold)))
    peak = float(np.max(probabilities))
    peak_logit = float(source_logits[int(np.argmax(probabilities))])
    uncertainty = math.exp(-abs(peak_logit - threshold_logit))
    volatility = float(np.mean(np.abs(np.diff(probabilities)))) if len(probabilities) > 1 else 0.0
    recovery = max(0.0, peak - float(np.median(probabilities[-min(20, len(probabilities)):])))
    trajectory_conflict = float(np.clip(
        0.55 * min(1.0, 12.0 * volatility) + 0.45 * min(1.0, 4.0 * recovery),
        0.0,
        1.0,
    ))
    low_risk_novelty = novelty_value * (1.0 - peak)
    score = (
        0.35 * float(np.clip(uncertainty, 0.0, 1.0))
        + 0.20 * trajectory_conflict
        + 0.20 * novelty_value
        + 0.25 * float(np.clip(low_risk_novelty, 0.0, 1.0))
    )
    return {
        "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
        "trajectory_conflict": trajectory_conflict,
        "novelty": novelty_value,
        "low_risk_novelty": float(np.clip(low_risk_novelty, 0.0, 1.0)),
        "query_score": float(np.clip(score, 0.0, 1.0)),
        "max_risk": peak,
    }


__all__ = [
    "CausalTypedTrajectoryMemory",
    "MemoryMatch",
    "SCHEMA_VERSION",
    "causal_query_components",
    "endpoint_feature",
]
