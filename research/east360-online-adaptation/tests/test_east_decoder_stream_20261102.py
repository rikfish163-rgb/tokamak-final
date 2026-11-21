import pytest

from experiments.risk_aligned_online.east_decoder_calibration_20261102.stream360 import (
    specs,
    predict,
)


def test_feedback_controls_share_exact_same_calibrated_head():
    prior, forecast, worst = object(), object(), object()
    s = specs(dict(prior=prior, forecast_only=forecast, worst_meta=worst))
    assert s["worst_real"] == (worst, "real")
    assert s["worst_reverse"] == (worst, "reverse")
    assert s["worst_hold"] == (worst, "hold")
    assert s["forecast_real"] == (forecast, "real")


def test_stream_refuses_label_fields():
    with pytest.raises(ValueError):
        predict(None, dict(values=None, available=None, label=1), None, {})
