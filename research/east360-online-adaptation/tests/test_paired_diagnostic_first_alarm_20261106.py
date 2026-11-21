import numpy as np

from paired_diagnostic_first_alarm_20261106 import attainable, shifted_peak


def test_cannot_pick_different_arms_inside_same_causal_block():
    scores = np.zeros((160, 2))
    scores[65, 0] = 1
    scores[71, 1] = 1
    scores[110, :] = 1
    hits, silent = attainable(scores, 0.5)
    assert hits == [65, 71]
    assert not silent


def test_allow_skip_a_block_only_if_one_arm_is_silent():
    scores = np.zeros((130, 2))
    scores[68, 1] = 1
    scores[100, 0] = 1
    hits, silent = attainable(scores, 0.5)
    assert hits == [68, 100] and silent
    scores[30] = 1
    assert attainable(scores, 0.5) == ([30], False)


def test_shifted_peak_distinguishes_untrained_new_false_alarm_location():
    a = np.zeros(110)
    b = a.copy()
    a[70], b[70], b[82] = 0.48, 0.49, 0.6
    row = shifted_peak(a, b, 82, 0.5)
    assert row["static_peak_index"] == 70
    assert row["real_peak_index"] == 82
    assert row["original_peak_stays_below_threshold"]
