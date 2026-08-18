#!/usr/bin/env python3
"""Causal dual-threshold alarm for weak early risk plus physics concurrence."""

from __future__ import annotations

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


__all__ = ["causal_moving_average", "dual_threshold_trace", "dual_threshold_alarm"]
