import numpy as np
import pytest

from FOUNDATION_EXPOSURE_AUDIT_20261031 import first_crossing, prefix_summary


def test_first_crossing_counts_equality_and_never_retracts_an_alarm():
    assert first_crossing([.1, .5, .2], .5) == 1
    assert first_crossing([.1, .2], .5) is None


@pytest.mark.parametrize('scores', [[], [[.1]], [.1, float('nan')]])
def test_invalid_curve_is_rejected(scores):
    with pytest.raises(ValueError):
        first_crossing(scores, .5)


def test_prefix_boundary_and_short_records_have_explicit_denominators():
    rows = [dict(points=8, rate=1000., first=7),
            dict(points=20, rate=1000., first=8),
            dict(points=4, rate=1000., first=3),
            dict(points=20, rate=1000., first=None)]
    result = prefix_summary(rows, 8.)
    assert result['false_alarms_before_ms'] == 2  # indices 7 and 3, not 8
    assert result['all_safe_shots'] == 4
    assert result['fraction_of_all_safe'] == .5
    assert result['observed_through_ms'] == 2  # indices through >=8 exist
    assert result['false_alarms_among_observed_through_ms'] == 0


def test_summary_uses_time_not_assumed_point_count():
    rows = [dict(points=20, rate=2000., first=12)]
    assert prefix_summary(rows, 8.)['false_alarms_before_ms'] == 1
    assert prefix_summary(rows, 6.)['false_alarms_before_ms'] == 0


def test_prefix_counts_monotonically_include_all_alarm_times():
    rows = [dict(points=20, rate=1000., first=x) for x in (0, 3, 10, 19, None)]
    got = [prefix_summary(rows, ms)['false_alarms_before_ms'] for ms in (1, 4, 11, 20)]
    assert np.array_equal(got, [1, 2, 3, 4])
