import numpy as np
import pytest

from experiments.risk_aligned_online.source_anticipation_20261104 import (
    history_eligibility as h,
)


def test_full_history_policy_does_not_count_early_padded_scores():
    times = np.arange(200) / 1000
    scores = np.zeros(200)
    scores[0] = 0.9
    assert h.score_history(times, scores, False, None, 0.7, deadline=30, history=1)[
        "alarm"
    ]
    assert not h.score_history(
        times, scores, False, None, 0.7, deadline=30, history=128
    )["alarm"]
    scores[130] = 0.9
    assert h.score_history(times, scores, False, None, 0.7, deadline=30, history=128)[
        "alarm"
    ]


def test_short_records_stay_in_the_denominator_without_fabricated_predictions():
    row = h.score_history(
        np.arange(90) / 1000, np.ones(90), True, 0.091, 0.7, deadline=30, history=128
    )
    assert row == dict(
        alarm=False,
        lead_ms=None,
        timely=False,
        history_ready=False,
        timely_history_feasible=False,
    )


def test_history_feasibility_uses_the_device_deadline_and_strict_boundary():
    times = np.arange(200) / 1000
    scores = np.ones(200)
    east = h.score_history(times, scores, True, 0.157, 0.7, deadline=30, history=128)
    source = h.score_history(times, scores, True, 0.157, 0.7, deadline=10, history=128)
    assert (
        east["lead_ms"] == 30
        and not east["timely_history_feasible"]
        and not east["timely"]
    )
    assert source["timely_history_feasible"] and source["timely"]


@pytest.mark.parametrize("history", [0, True, 2.5])
def test_invalid_history_policy_is_rejected(history):
    with pytest.raises(ValueError):
        h.score_history(
            np.arange(5) / 1000,
            np.ones(5),
            False,
            None,
            0.7,
            deadline=30,
            history=history,
        )
