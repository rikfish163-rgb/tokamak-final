import pytest

from experiments.risk_aligned_online.east_decoder_calibration_20261102.parameter_controls import (
    evidence_gate,
    predict,
)


def test_generic_gain_does_not_prove_a_benefit_from_real_feedback():
    scores = dict(
        static=dict(tp=125, fp=35),
        real=dict(tp=126, fp=34),
        reverse=dict(tp=126, fp=34),
        hold=dict(tp=126, fp=34),
    )
    gate = evidence_gate(scores)
    assert gate["operational"] and not gate["passed"]
    scores["reverse"] = scores["hold"] = dict(tp=125, fp=35)
    assert evidence_gate(scores)["passed"]


def test_parameter_control_predictor_refuses_current_outcome():
    with pytest.raises(ValueError):
        predict(None, dict(values=None, available=None, label=1), None, None)
