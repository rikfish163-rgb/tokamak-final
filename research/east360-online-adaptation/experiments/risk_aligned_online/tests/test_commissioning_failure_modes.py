import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_failure_modes_20261111 import (
    consensus, opportunity,
)


def test_update_can_be_large_enough_at_wrong_point_without_rescuing_the_alarm():
    t = np.arange(7) / 1000
    fixed = np.array([.10, .49, .20, .12, .80, .80, .80])
    updated = np.array([.10, .49, .29, .12, .90, .90, .90])
    r = opportunity(t, fixed, updated, True, .034, .5)
    assert r['eligible_points'] == 4  # Exactly 30 ms does not qualify.
    assert r['largest_lift'] > r['gap'] > 0
    assert r['lift_at_best_static_point'] == 0
    assert r['updated_gap'] == pytest.approx(r['gap'])
    assert r['first_change_index'] == 2


def test_false_alarm_before_first_score_change_cannot_be_retracted():
    t = np.arange(4) / 1000
    r = opportunity(t, np.array([.1, .6, .6, .2]), np.array([.1, .6, .2, .1]),
                    False, None, .5)
    assert r['first_fixed_alarm_index'] == 1
    assert r['first_change_index'] == 2
    assert r['alarm_before_any_change']


def test_consensus_uses_same_shot_and_keeps_tardy_distinct_from_no_alarm():
    def row(shot, d, alarm, timely=False):
        return dict(shot=shot, disruptive=d, alarm=alarm, timely=timely)

    a = [row(1, True, True), row(2, True, False), row(3, False, True)]
    b = [row(1, True, True, True), row(2, True, True), row(3, False, True)]
    r = consensus(dict(a=a, b=b))
    assert r['always_untimely'] == [2] and r['always_false_alarm'] == [3]
    assert r['always_no_alarm'] == [] and r['seed_or_condition_sensitive_disruptions'] == [1]
    with pytest.raises(ValueError, match='same ordered'):
        consensus(dict(a=a, b=b[::-1]))
