import pytest

from experiments.risk_aligned_online.event_conditional_feedback_20261102.pilot import (
    summarize,
    gates,
)


def rows(delta=-0.01):
    return [
        dict(shot=1, disruptive=True, actual_lead_ms=60.0, delta_bce=delta),
        dict(shot=1, disruptive=True, actual_lead_ms=250.0, delta_bce=delta * 2),
        dict(shot=2, disruptive=False, actual_lead_ms=None, delta_bce=delta),
    ]


def test_danger_safe_and_timely_window_are_not_point_label_mixtures():
    value = summarize(rows())
    assert value["disruptive"]["n_shots"] == 1
    assert value["disruptive"]["mean_delta_bce"] == pytest.approx(-0.015)
    assert value["danger_window"]["mean_delta_bce"] == -0.01
    assert value["safe"]["mean_delta_bce"] == -0.01
    assert value["balanced"] == pytest.approx(-0.0125)


def test_no_event_context_advantage_cannot_pass_gate():
    summary = {
        a: summarize(rows())
        for a in (
            "shared_real",
            "true_real",
            "shuffled_real",
            "true_reverse",
            "true_hold",
        )
    }
    assert not gates(summary)["passed"]
    summary["true_real"] = summarize(rows(-0.02))
    assert gates(summary)["passed"]
    summary["true_real"]["safe"]["mean_delta_bce"] = 0.001
    assert not gates(summary)["passed"]
