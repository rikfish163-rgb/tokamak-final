import numpy as np
import pytest

from experiments.risk_aligned_online.heldout_threshold_20261023.calibration import safe_threshold


def test_whole_shot_order_statistic_with_strict_tie_handling():
    scores = np.arange(10, dtype=float) / 10
    r = safe_threshold(scores, .2)
    assert r['rank'] == 9 and r['safe_count'] == 10
    assert r['order_statistic'] == .8 and r['threshold'] > .8
    assert sum(scores >= r['threshold']) == 1
    assert safe_threshold(scores[::-1], .2) == r


def test_exact_integer_rank_and_duplicate_peaks_are_not_false_alarms():
    r = safe_threshold([.5] * 9, .1)
    assert r['rank'] == 9 and r['threshold'] > .5
    assert not (np.full(9, .5) >= r['threshold']).any()


def test_small_sample_is_explicitly_infeasible_not_a_fake_safe_model():
    r = safe_threshold([.3] * 8, .1)
    assert r['status'] == 'INSUFFICIENT_FOR_NONTRIVIAL_THRESHOLD'
    assert r['threshold'] is None and r['rank'] == 9


@pytest.mark.parametrize('scores,alpha', [([], .2), ([np.nan], .2), ([1.1], .2),
    ([.3], 0), ([.3], 1), ([[.3]], .2), ([.3], np.nan)])
def test_invalid_calibration_rejected(scores, alpha):
    with pytest.raises(ValueError): safe_threshold(scores, alpha)
