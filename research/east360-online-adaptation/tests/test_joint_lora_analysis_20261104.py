import numpy as np
import pytest

from experiments.risk_aligned_online.joint_lora_analysis_20261104 import (
    forecast_error,
    capped_metric,
)


def test_forecast_error_excludes_pending_nans_and_unobserved_coordinates():
    target = np.array([[2.0, 0.0], [np.nan, np.nan]], np.float32)
    prediction = np.array([[1.0, 99.0], [8.0, 9.0]], np.float32)
    mask = np.array([[True, False], [False, False]])
    result = forecast_error(prediction, target, mask, np.array([135, -1]))
    assert result["mse"] == 1.0 and result["hold_mse"] == 4.0
    assert result["observed_elements"] == 1 and result["pending"] == 1
    assert result["unsupported_matured"] == 0


def test_no_supported_targets_are_not_fabricated_zero_mse():
    result = forecast_error(
        np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2), bool), np.array([135])
    )
    assert result["mse"] is None and result["hold_mse"] is None
    assert result["unsupported_matured"] == 1


def test_pending_mask_cannot_claim_real_observations():
    with pytest.raises(ValueError):
        forecast_error(
            np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2), bool), np.array([-1])
        )


def test_capped_sensitivity_does_not_relabel_main_metric():
    rows = [
        dict(disruptive=True, timely=True, lead_ms=3001, alarm=True),
        dict(disruptive=True, timely=True, lead_ms=31, alarm=True),
        dict(disruptive=False, timely=False, lead_ms=None, alarm=True),
    ]
    original = [dict(r) for r in rows]
    result = capped_metric(rows)
    assert result["tp"] == 1 and result["fp"] == 1
    assert result["J"] == -0.5
    assert rows == original
