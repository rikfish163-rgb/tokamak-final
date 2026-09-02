"""Validation-only error-prior query policy for the J-TEXT online branch.

The policy is deliberately small and deterministic.  It is fitted only on
VAL160 shot-level outcomes, then used as a label-free prior on TEST220.  At
deployment time the feature vector contains the current shot's observed
signal summary and source risk trace; it never contains ``IsDisrupt``,
``DownTime`` or ``AdvancedTime``.  The policy is a query-value model, not a
replacement risk model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "jtext-validation-error-prior-v1"


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def _quantile(value: np.ndarray, fraction: float) -> float:
    return float(np.quantile(value, fraction)) if len(value) else 0.0


def trace_features(
    risk: Sequence[float] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    values: np.ndarray,
    available: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Extract a bounded, outcome-blind shot vector for query scoring."""

    probabilities = _finite_vector(risk, name="query risk")
    source_logits = _finite_vector(logits, name="query logits")
    signal = np.asarray(values, dtype=np.float64)
    known = np.asarray(available, dtype=bool)
    if signal.ndim != 2 or known.shape != signal.shape:
        raise ValueError("query signal and availability must be aligned matrices")
    if len(probabilities) != len(source_logits) or len(probabilities) != len(signal):
        raise ValueError("query trace fields must have equal endpoint length")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("query threshold must lie in (0, 1)")
    tail = probabilities[-min(20, len(probabilities)):]
    head = probabilities[:min(20, len(probabilities))]
    delta = np.diff(probabilities) if len(probabilities) > 1 else np.zeros(1)
    first_alarm = np.flatnonzero(probabilities >= float(threshold))
    peak = float(np.max(probabilities))
    features: list[float] = [
        peak,
        _quantile(probabilities, 0.50),
        _quantile(probabilities, 0.90),
        _quantile(probabilities, 0.95),
        float(probabilities[-1]),
        float(np.mean(tail)),
        float(np.mean(head)),
        float(np.mean(np.abs(delta))),
        float(np.max(np.abs(delta))),
        max(0.0, peak - float(np.median(tail))),
        float(first_alarm[0] / len(probabilities)) if len(first_alarm) else 1.0,
        float(np.mean(known)),
    ]
    # Recent robust signal statistics preserve cross-channel operating-state
    # information while avoiding any terminal-time or outcome field.
    recent = signal[-min(128, len(signal)):]
    recent_known = known[-len(recent):]
    for column in range(signal.shape[1]):
        observed = recent[recent_known[:, column], column]
        if len(observed):
            features.extend([
                float(np.median(observed)),
                float(np.quantile(observed, 0.75) - np.quantile(observed, 0.25)),
                float(np.mean(np.abs(np.diff(observed)))) if len(observed) > 1 else 0.0,
                float(np.mean(recent_known[:, column])),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    result = np.asarray(features, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("query feature extraction produced a non-finite vector")
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class ValidationErrorPrior:
    """A class-balanced ridge logistic surrogate fitted on VAL160 only."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    l2: float
    positive_count: int
    negative_count: int

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        error_labels: Sequence[int | bool] | np.ndarray,
        *,
        l2: float = 1.0,
    ) -> "ValidationErrorPrior":
        matrix = np.asarray(features, dtype=np.float64)
        labels = np.asarray(error_labels, dtype=np.float64).reshape(-1)
        if matrix.ndim != 2 or not len(matrix) or matrix.shape[0] != len(labels):
            raise ValueError("validation query features and labels are misaligned")
        if not np.isfinite(matrix).all() or not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError("validation query training data must be finite and binary")
        if not math.isfinite(float(l2)) or float(l2) <= 0.0:
            raise ValueError("validation error-prior l2 must be finite and positive")
        positives = int(np.sum(labels > 0.5))
        negatives = int(len(labels) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("validation error-prior needs both outcome classes")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        standardized = (matrix - mean) / scale
        design = np.column_stack((standardized, np.ones(len(standardized))))
        weights = np.where(
            labels > 0.5,
            len(labels) / (2.0 * positives),
            len(labels) / (2.0 * negatives),
        )
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_target = (2.0 * labels - 1.0) * np.sqrt(weights)
        regularizer = np.eye(design.shape[1], dtype=np.float64) * float(l2)
        regularizer[-1, -1] = 0.0
        try:
            solution = np.linalg.solve(
                weighted_design.T @ weighted_design + regularizer,
                weighted_design.T @ weighted_target,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("validation error-prior fit is singular") from exc
        if not np.isfinite(solution).all():
            raise ValueError("validation error-prior fit produced non-finite coefficients")
        return cls(
            mean=np.ascontiguousarray(mean),
            scale=np.ascontiguousarray(scale),
            coefficients=np.ascontiguousarray(solution[:-1]),
            intercept=float(solution[-1]),
            l2=float(l2),
            positive_count=positives,
            negative_count=negatives,
        )

    @property
    def feature_dimension(self) -> int:
        return int(len(self.coefficients))

    def score(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_dimension:
            raise ValueError("query feature dimension changed")
        if not np.isfinite(matrix).all():
            raise ValueError("query features must be finite")
        return _sigmoid(
            ((matrix - self.mean) / self.scale) @ self.coefficients + self.intercept
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "l2": self.l2,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.state_dict(), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["SCHEMA_VERSION", "ValidationErrorPrior", "trace_features"]
