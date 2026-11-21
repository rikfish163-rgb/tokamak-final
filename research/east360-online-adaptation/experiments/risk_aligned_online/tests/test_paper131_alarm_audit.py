"""Retrospective alarm diagnostics must not backdate a confirmed warning."""
import numpy as np
import pytest

from experiments.risk_aligned_online.paper131_alarm_audit import alarm_endpoints


def test_isolated_peak_and_continuous_confirmation():
    times = np.arange(15) * .001
    scores = np.array([.8, .1, .1, *([.8] * 12)])
    assert alarm_endpoints(times, scores, .8) == (0, 0)
    assert alarm_endpoints(times, scores, .8, points=10) == (3, 12)
    assert alarm_endpoints(times, scores, .8, duration_ms=10.) == (3, 13)
    assert alarm_endpoints(times, scores, .8, duration_ms=1.) == (3, 4)


def test_first_run_too_short_later_run_and_no_alarm():
    t = np.arange(8) * .001
    s = np.array([.8, .8, .2, .8, .8, .8, .8, .2])
    assert alarm_endpoints(t, s, .7, points=3) == (3, 5)
    assert alarm_endpoints(t, s, .7, points=5) is None
    assert alarm_endpoints(t, np.zeros(8), .7) is None


def test_time_not_point_count_and_never_use_future():
    t = 3. + np.arange(20) * .0005
    s = np.ones(20)
    assert alarm_endpoints(t, s, .7, duration_ms=1.) == (0, 2)
    assert alarm_endpoints(t, s, .7, points=10) == (0, 9)
    assert alarm_endpoints(t[:9], s[:9], .7, points=10) is None


def test_confirmation_can_turn_early_crossing_into_late_alarm():
    t = np.arange(20) * .001
    start, confirm = alarm_endpoints(t, np.ones(20), .7, points=10)
    assert (.035 - t[start]) * 1000 > 30.
    assert (.035 - t[confirm]) * 1000 < 30.


@pytest.mark.parametrize('kwargs', [{'points': 0}, {'points': True}, {'duration_ms': -1}, {'points': 2, 'duration_ms': 1.}])
def test_bad_rule_rejected(kwargs):
    with pytest.raises(ValueError):
        alarm_endpoints(np.arange(3), np.ones(3), .7, **kwargs)


@pytest.mark.parametrize('t,s', [([0, 1, 1], [0, 0, 0]), ([0, 1], [0, np.nan]), ([0, 1], [0, 2])])
def test_bad_trace_rejected(t, s):
    with pytest.raises(ValueError):
        alarm_endpoints(t, s, .7)
