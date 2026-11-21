import numpy as np
import pytest

from experiments.risk_aligned_online.predictive_risk_pilot_20261102.core import (
    future_windows,
    decode_forecast,
)


def test_direct_future_windows_do_not_recur_on_earlier_model_outputs():
    past = np.repeat(np.arange(128)[:, None], 23, axis=1).astype(np.float32)
    forecast = np.full((2, 23), 500.0, np.float32)
    forecast[1] = 600.0
    x, m = future_windows(
        past, np.ones_like(past, bool), forecast, np.ones_like(forecast, bool)
    )
    assert x.shape == m.shape == (2, 128, 23)
    assert np.array_equal(x[0, -2:, 0], [127.0, 500.0])
    assert np.array_equal(x[1, -2:, 0], [500.0, 600.0])
    assert x[1, 0, 0] == 2.0


def test_decode_carries_discrete_mode_and_preserves_missing_without_future_truth():
    pred = np.zeros((25, 23), np.float32)
    pred[:, 10] = 1e8
    latest = np.ones(23, np.float32) * 3
    mask = np.ones(23, bool)
    mask[1] = False
    mask[2] = False
    values, available = decode_forecast(pred, np.zeros(23), np.ones(23), latest, mask)
    assert np.all(values[:, 10] == 3) and np.all(available[:, 10])
    assert not available[:, 1:3].any() and not values[:, 1:3].any()
    pred[:, 0] = float("nan")
    with pytest.raises(ValueError):
        decode_forecast(pred, np.zeros(23), np.ones(23), latest, mask)
