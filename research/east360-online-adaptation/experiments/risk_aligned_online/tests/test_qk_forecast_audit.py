import numpy as np
import pytest

from experiments.risk_aligned_online.qk_forecast_stream_20261029.audit import decision


def test_recount_strict_lead_and_first_alarm_not_best_alarm():
    times = np.array([0.5, 0.96, 0.97, 1.01])
    row = decision(1, times, np.array([0.1, 0.1, 0.8, 0.9]), True, 1.0, 0.8)
    assert row["alarm"] and row["lead_ms"] == 30 and not row["timely"]
    row = decision(1, times, np.array([0.8, 0.1, 0.8, 0.9]), True, 1.0, 0.8)
    assert row["lead_ms"] == 500 and row["timely"]
    row = decision(1, times, np.array([0.1, 0.1, 0.1, 0.9]), True, 1.0, 0.8)
    assert not row["alarm"] and row["lead_ms"] is None


def test_safe_shot_uses_entire_trace_and_invalid_data_raise():
    times = np.array([0.1, 0.2, 0.3])
    row = decision(2, times, np.array([0.1, 0.1, 0.8]), False, 0.0, 0.8)
    assert row["alarm"] and not row["timely"] and row["lead_ms"] is None
    for p in [np.array([0.1, np.nan, 0.8]), np.array([0.1, 1.1, 0.8])]:
        with pytest.raises(ValueError):
            decision(2, times, p, False, 0.0, 0.8)
    with pytest.raises(ValueError):
        decision(2, times[::-1], np.array([0.1, 0.1, 0.8]), False, 0.0, 0.8)
