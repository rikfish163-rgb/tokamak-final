import copy

import pytest

from experiments.risk_aligned_online.direct_decoder_meta_20261102.audit import (
    check_numeric,
    gate,
)


def example():
    return dict(
        before_logit=0.0,
        after_logit=0.0,
        label=1.0,
        before_bce=0.69314718056,
        after_bce=0.69314718056,
        delta_bce=0.0,
        query_rms=0.0,
        accepted=False,
        step_norm=0.0,
        gradient_norm=0.0,
        loss_before=0.2,
        loss_proposal=0.2,
        absolute_diagnostic_before=0.3,
    )


def test_no_artificial_event_probabilities_needed_for_numeric_audit():
    assert check_numeric(example()) < 1e-10
    bad = example()
    bad["after_bce"] += 0.001
    with pytest.raises(ValueError):
        check_numeric(bad)


def test_decoder_gate_requires_forecast_and_ordered_feedback_controls():
    v = dict(
        safe=dict(mean_delta_bce=-0.1),
        danger=dict(mean_delta_bce=-0.1),
        compatible_danger=dict(mean_delta_bce=-0.1),
        balanced=-0.1,
    )
    s = {
        t: {
            a: copy.deepcopy(v)
            for a in (
                "worst_meta_real",
                "mean_meta_real",
                "forecast_only_real",
                "worst_meta_reverse",
                "worst_meta_hold",
            )
        }
        for t in ("native", "no_MNM_CIII")
    }
    assert not gate(s)["passed"]
    s["no_MNM_CIII"]["worst_meta_real"]["balanced"] = -0.2
    assert gate(s)["passed"]
