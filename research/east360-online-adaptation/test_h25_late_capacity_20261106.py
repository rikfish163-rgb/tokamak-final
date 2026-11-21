import numpy as np
import pytest

from run_h25_late_capacity_20261106 import gate, match_late, pick_trial


def test_match_uses_complete_raw_label_window_and_requires_same_rows():
    maturity = np.array([63, 95])
    positive = np.zeros((2, 32), dtype=bool)
    positive[0, [2, 4]] = True
    positive[1, 0] = True
    rows = [(11, dict(maturity=63, eligible=[66, 68]))]
    assert match_late(rows, maturity, positive, np.array([True, False])) == {63: 11}
    rows[0][1]["eligible"] = [68]
    with pytest.raises(ValueError, match="canonical"):
        match_late(rows, maturity, positive, np.array([True, False]))


def test_oracle_keeps_no_write_and_all_verified_candidates():
    trials = [
        dict(target=1, projected=None, optimized={"peak": 0.3, "state": [0]}),
        dict(target=2, projected={"peak": 0.2, "state": [1]}, optimized=None),
    ]
    assert pick_trial(trials, 0.5) == (0.5, None)
    peak, best = pick_trial(trials, 0.0)
    assert peak == 0.3 and best["target"] == 1 and best["source"] == "optimized"


def test_promotion_requires_new_verified_shot_or_real_capture_and_safe_gain():
    reach = dict(
        old=[[1], [1], [1]], old_true=[[1, 2], [1, 2], [1]], old_reverse=[[], [2], []]
    )
    safe = dict(raw_harm_by_seed=[[0, 0, 0, 0]])
    summary = dict(
        positive=dict(
            n_windows=10,
            n_supported=7,
            paired_capture_gain=dict(real_minus_reverse=0.03),
            raw_peak_delta=[[0, 0, 0.04, 0.05]],
        ),
        safe=safe,
        protect=safe,
    )
    assert not gate(summary, reach)["promote"]
    reach["old_reverse"][1] = []
    assert gate(summary, reach)["new_shot_two_seeds"] == [2]
    summary["positive"]["n_supported"] = 6
    assert not gate(summary, reach)["promote"]
    summary["positive"]["n_supported"] = 7
    reach["old_reverse"][1] = [2]
    summary["positive"]["raw_peak_delta"] = [[0, 0, 0.06, 0.05]]
    assert gate(summary, reach)["promote"]
    summary["safe"] = dict(raw_harm_by_seed=[[0, 0, 1, 0]])
    assert not gate(summary, reach)["promote"]
