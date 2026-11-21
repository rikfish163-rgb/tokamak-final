import numpy as np
import pytest

from experiments.risk_aligned_online.cold20_compare_20261104.calibration import (
    choose_thresholds,
    threshold_at,
)


def test_prefix_predictions_use_original_threshold_even_after_calibration():
    assert threshold_at(1, 0.7, 0.3) == 0.7
    assert threshold_at(20, 0.7, 0.3) == 0.7
    assert threshold_at(21, 0.7, 0.3) == 0.3


def test_threshold_selection_uses_only_explicit_ended_safe_shots():
    safe = list(range(12))
    maxima = {"native": dict(zip(safe, np.linspace(0.1, 0.9, 12, dtype=np.float32)))}
    result = choose_thresholds(maxima, safe)
    assert result["0.1"] == result["0.15"]
    for cap, row in result.items():
        assert sum(v >= row["native"] for v in maxima["native"].values()) <= int(
            float(cap) * 12
        )
    maxima["native"][13] = 0.99
    with pytest.raises(ValueError, match="safe"):
        choose_thresholds(maxima, safe)


def test_tied_maxima_are_not_arbitrarily_split():
    maxima = {"native": {i: np.float32(0.8) for i in range(12)}}
    result = choose_thresholds(maxima, list(range(12)))
    assert all(r["native"] > np.float32(0.8) for r in result.values())
