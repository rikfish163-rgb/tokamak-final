#!/usr/bin/env python3
"""Causal dual-threshold alarm for weak early risk plus physics concurrence."""

from __future__ import annotations

import copy

import numpy as np


def causal_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Trailing mean including the current sample, with no future access."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(width) <= 0:
        raise ValueError("moving-average width must be positive")
    safe = np.where(np.isfinite(array), array, 0.0)
    valid = np.isfinite(array).astype(np.int64)
    total = np.r_[0.0, np.cumsum(safe)]
    count = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(array) + 1)
    start = np.maximum(0, end - int(width))
    numerator = total[end] - total[start]
    denominator = count[end] - count[start]
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(array), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def dual_threshold_trace(
    margin_trace,
    physics_support,
    *,
    high_threshold: float = 0.0,
    physics_weight: float = 8.0,
    trend_weight: float = 12.0,
    early_threshold: float = 7.9,
    fast_width: int = 5,
    slow_width: int = 100,
) -> dict[str, np.ndarray]:
    """Return pointwise causal alarm evidence for both registered routes.

    The immutable migrated-model route is preserved exactly.  The early route
    can only add an alarm when weak model risk coincides with observable EAST
    physics innovation and a positive trailing risk trend.
    """
    margin = np.asarray(margin_trace, dtype=np.float64).reshape(-1)
    support = np.asarray(physics_support, dtype=np.float64).reshape(-1)
    if margin.shape != support.shape or not len(margin):
        raise ValueError("margin and physics support must be aligned non-empty vectors")
    if fast_width <= 0 or slow_width <= fast_width:
        raise ValueError("require 0 < fast_width < slow_width")
    finite_margin = np.isfinite(margin)
    finite_support = np.isfinite(support)
    bounded_support = np.where(finite_support, np.clip(support, 0.0, 1.0), 0.0)
    fast = causal_moving_average(margin, fast_width)
    slow = causal_moving_average(margin, slow_width)
    trend = np.maximum(fast - slow, 0.0)
    trend = np.where(np.isfinite(trend), trend, 0.0)
    early_score = (
        np.where(finite_margin, margin, -np.inf)
        + float(physics_weight) * bounded_support
        + float(trend_weight) * trend
    )
    high_mask = finite_margin & (margin >= float(high_threshold))
    early_mask = finite_margin & finite_support & (early_score >= float(early_threshold))
    return {
        "high_mask": high_mask,
        "early_mask": early_mask,
        "alarm_mask": high_mask | early_mask,
        "early_score": early_score,
        "trend": trend,
        "physics_support": bounded_support,
    }


def dual_threshold_alarm(
    margin_trace,
    physics_support,
    **kwargs,
) -> dict[str, object]:
    trace = dual_threshold_trace(margin_trace, physics_support, **kwargs)
    positions = np.flatnonzero(trace["alarm_mask"])
    if not len(positions):
        return {
            "alarmed": False,
            "alarm_index": None,
            "route": None,
            "early_score": None,
        }
    index = int(positions[0])
    route = "high" if bool(trace["high_mask"][index]) else "early"
    return {
        "alarmed": True,
        "alarm_index": index,
        "route": route,
        "early_score": float(trace["early_score"][index]),
    }


def persistent_precursor_trace(
    risk_probability,
    physics_support,
    *,
    decision_threshold: float,
    physics_weight: float = 8.0,
    trend_weight: float = 12.0,
    score_threshold: float = 4.5,
    minimum_physics_support: float = 0.20,
    minimum_risk_probability: float = 0.40,
    required_points: int = 3,
    window: int = 10,
    fast_width: int = 5,
    slow_width: int = 100,
) -> dict[str, object]:
    """Return a prefix-causal low-threshold, persistent precursor route.

    A point contributes evidence only when model risk, registered physics
    support, and the existing early-score geometry all agree.  The route fires
    after ``required_points`` qualifying endpoints occur in the trailing
    ``window``.  No shot outcome or future endpoint is used.
    """
    risk = np.asarray(risk_probability, dtype=np.float64).reshape(-1)
    support = np.asarray(physics_support, dtype=np.float64).reshape(-1)
    if risk.shape != support.shape or not len(risk):
        raise ValueError("risk and physics support must be aligned non-empty vectors")
    if not 0.0 < float(decision_threshold) < 1.0:
        raise ValueError("decision threshold must lie in (0, 1)")
    if not 0.0 < float(minimum_risk_probability) < 1.0:
        raise ValueError("minimum precursor risk must lie in (0, 1)")
    if not 0.0 <= float(minimum_physics_support) <= 1.0:
        raise ValueError("minimum precursor physics support must lie in [0, 1]")
    if not np.isfinite(float(score_threshold)):
        raise ValueError("precursor score threshold must be finite")
    if int(required_points) <= 0 or int(window) < int(required_points):
        raise ValueError("require 0 < required_points <= window")
    if int(fast_width) <= 0 or int(slow_width) <= int(fast_width):
        raise ValueError("require 0 < fast_width < slow_width")

    finite_risk = np.isfinite(risk) & (risk >= 0.0) & (risk <= 1.0)
    safe_risk = np.clip(risk, 1e-6, 1.0 - 1e-6)
    decision_logit = np.log(
        float(decision_threshold) / (1.0 - float(decision_threshold))
    )
    margin = np.full(len(risk), np.nan, dtype=np.float64)
    margin[finite_risk] = (
        np.log(safe_risk[finite_risk] / (1.0 - safe_risk[finite_risk]))
        - decision_logit
    )
    bounded_support = np.where(
        np.isfinite(support), np.clip(support, 0.0, 1.0), 0.0,
    )
    fast = causal_moving_average(margin, int(fast_width))
    slow = causal_moving_average(margin, int(slow_width))
    trend = np.maximum(fast - slow, 0.0)
    trend = np.where(np.isfinite(trend), trend, 0.0)
    early_score = (
        np.where(np.isfinite(margin), margin, -np.inf)
        + float(physics_weight) * bounded_support
        + float(trend_weight) * trend
    )
    qualifying = (
        finite_risk
        & np.isfinite(support)
        & (risk >= float(minimum_risk_probability))
        & (bounded_support >= float(minimum_physics_support))
        & (early_score >= float(score_threshold))
    )
    cumulative = np.r_[0, np.cumsum(qualifying.astype(np.int64))]
    end = np.arange(1, len(qualifying) + 1)
    start = np.maximum(0, end - int(window))
    rolling_count = cumulative[end] - cumulative[start]
    alarm_mask = rolling_count >= int(required_points)
    positions = np.flatnonzero(alarm_mask)
    index = int(positions[0]) if len(positions) else None
    return {
        "alarmed": index is not None,
        "alarm_index": index,
        "route": "persistent_precursor" if index is not None else None,
        "early_score": float(early_score[index]) if index is not None else None,
        "qualifying_mask": qualifying,
        "rolling_count": rolling_count,
        "alarm_mask": alarm_mask,
        "margin": margin,
        "trend": trend,
        "physics_support": bounded_support,
    }


def apply_precursor_overlay(
    base_alarm: dict[str, object], precursor_alarm: dict[str, object],
) -> dict[str, object]:
    """Merge an independent precursor route without delaying a base alarm."""
    base = copy.deepcopy(dict(base_alarm))
    precursor = copy.deepcopy(dict(precursor_alarm))
    for name, row in (("base", base), ("precursor", precursor)):
        if "alarmed" not in row or "alarm_index" not in row:
            raise KeyError(f"{name} alarm is missing its decision fields")
        if bool(row["alarmed"]) != (row["alarm_index"] is not None):
            raise ValueError(f"{name} alarm decision is inconsistent")
    base_index = int(base["alarm_index"]) if bool(base["alarmed"]) else None
    precursor_index = (
        int(precursor["alarm_index"]) if bool(precursor["alarmed"]) else None
    )
    applied = (
        precursor_index is not None
        and (base_index is None or precursor_index < base_index)
    )
    result = copy.deepcopy(base)
    if applied:
        result.update({
            "alarmed": True,
            "alarm_index": precursor_index,
            "route": precursor.get("route", "persistent_precursor"),
            "early_score": precursor.get("early_score"),
            "filter_reason": "persistent_precursor_overlay",
        })
    result["precursor_overlay_applied"] = bool(applied)
    result["base_alarm_reference"] = base
    result["precursor_alarm"] = precursor
    return result


__all__ = [
    "apply_precursor_overlay", "causal_moving_average", "dual_threshold_alarm",
    "dual_threshold_trace", "persistent_precursor_trace",
]
