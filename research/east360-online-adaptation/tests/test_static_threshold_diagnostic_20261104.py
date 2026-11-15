import numpy as np
import pytest

from experiments.risk_aligned_online.static_threshold_diagnostic_20261104 import (
    cutoff_from_safe,
    operating_point,
    shot_maximum,
)


def test_disruption_score_uses_only_strictly_timely_points():
    times = np.array([0.969, 0.970, 0.980, 1.001])
    scores = np.array([0.2, 0.8, 0.9, 1.0])
    assert shot_maximum(times, scores, True, 1.0) == 0.2
    assert shot_maximum(np.array([0.970]), np.array([0.8]), True, 1.0) is None


def test_safe_score_uses_full_record_not_disruption_deadline():
    assert shot_maximum(np.array([0.0, 2.0]), np.array([0.2, 0.9]), False, None) == 0.9


@pytest.mark.parametrize('scores', [[np.nan], [-0.1], [1.1]])
def test_invalid_probability_is_rejected(scores):
    with pytest.raises(ValueError):
        shot_maximum(np.array([0.0]), np.array(scores), False, None)


def test_preparation_cutoff_obeys_cap_including_ties():
    safe = np.array([0.8, 0.8, 0.8, 0.3, 0.2], dtype=np.float32)
    cutoff = cutoff_from_safe(safe, 0.2)
    assert np.sum(safe.astype(np.float64) >= cutoff) == 0
    assert cutoff > float(safe.max())


def test_preparation_cutoff_allows_exactly_budgeted_safe_alarms():
    safe = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    cutoff = cutoff_from_safe(safe, 0.2)
    assert np.sum(safe.astype(np.float64) >= cutoff) == 1
    assert float(safe[1]) < cutoff <= float(safe[0])


@pytest.mark.parametrize('safe,cap', [([], 0.2), ([np.nan], 0.2), ([0.2], 1.1)])
def test_invalid_preparation_data_rejected(safe, cap):
    with pytest.raises(ValueError):
        cutoff_from_safe(safe, cap)


def test_operating_point_retains_unobservable_disruptions_in_denominator():
    rows = [dict(disruptive=True, maximum=0.8), dict(disruptive=True, maximum=None),
            dict(disruptive=False, maximum=0.7), dict(disruptive=False, maximum=0.2)]
    assert operating_point(rows, 0.7) == dict(tp=1, fp=1, n_disruptive=2, n_safe=2,
                                            tpr=0.5, fpr=0.5, J=0.0)
