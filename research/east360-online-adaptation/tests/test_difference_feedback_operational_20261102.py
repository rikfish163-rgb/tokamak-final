import pytest

from experiments.risk_aligned_online.difference_feedback_probe_20261102.operational_sensitivity import (
    operational_delta,
)


def test_raising_risk_helps_a_timely_disruptive_point_but_hurts_a_safe_point():
    assert operational_delta(0.0, 1.0, True) < 0
    assert operational_delta(0.0, 1.0, False) > 0


def test_extreme_logits_are_finite():
    assert operational_delta(-1000.0, -999.0, True) == pytest.approx(-1)
    assert operational_delta(1000.0, 1001.0, False) == pytest.approx(1)
