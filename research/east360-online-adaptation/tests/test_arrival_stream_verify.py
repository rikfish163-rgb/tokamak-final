import numpy as np
import pytest

from experiments.risk_aligned_online.arrival_stream_20261030.verify import decide, check_timing


def test_strict_30ms_first_crossing_and_safe_alarm():
    assert not decide(1, np.array([.970]), np.array([.8]), True, 1., .8)['timely']
    assert decide(1, np.array([.969]), np.array([.8]), True, 1., .8)['timely']
    r = decide(2, np.array([.9, .99]), np.array([.8, .9]), True, 1., .8)
    assert r['lead_ms'] == 100. and r['timely']
    r = decide(3, np.array([0.]), np.array([.8]), False, 1., .8)
    assert r['alarm'] and not r['timely'] and r['lead_ms'] is None


def test_timing_covers_tail_and_rejects_same_point_update():
    rows = [dict(origin=127, maturity=152, first_affected=153, apply_count=25),
            dict(origin=152, maturity=177, first_affected=178, apply_count=2)]
    check_timing(rows, 180)
    with pytest.raises(AssertionError):
        check_timing([dict(rows[0], first_affected=152), rows[1]], 180)
    with pytest.raises(AssertionError):
        check_timing(rows[:1], 180)
