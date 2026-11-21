import numpy as np

from audit_layer0_action_utility_20261106 import best_first, coverage, miss_reason


def test_oracle_can_choose_silence_even_if_each_full_arm_alarms():
    clock = np.arange(5) * 0.1
    p = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    truth = dict(shot=1, disruptive=False, down=None)
    assert best_first(clock, p, truth, capped=True, first=1, stride=2) == len(clock)


def test_capped_capacity_separates_unavoidable_early_from_no_candidate():
    clock = np.array([0.0, 1.0, 2.0, 3.5, 3.99])
    p = np.array([[1, 0, 0, 1, 1], [1, 0, 0, 0, 0]])
    truth = dict(shot=1, disruptive=True, down=4.0)
    assert best_first(clock, p, truth, capped=True, first=1, stride=2) == 0
    assert (
        miss_reason(clock, p, truth, 0, capped=True)
        == "eligible_crossing_cannot_be_first"
    )
    p[:, 3] = 0
    assert (
        miss_reason(clock, p, truth, 0, capped=True) == "no_eligible_candidate_crossing"
    )


def test_credit_coverage_counts_unique_shots_not_correlated_anchors():
    rows = [
        dict(shot=1, disruptive=True, capped_isolated=np.ones(50, dtype=int)),
        dict(shot=2, disruptive=False, capped_isolated=-np.ones(12, dtype=int)),
        dict(shot=3, disruptive=True, capped_isolated=np.zeros(20, dtype=int)),
    ]
    info = coverage(rows, "capped_isolated")
    assert info["benefit"]["shots"] == [1] and info["benefit"]["anchors"] == 50
    assert info["harm"]["shots"] == [2] and info["harm"]["anchors"] == 12
