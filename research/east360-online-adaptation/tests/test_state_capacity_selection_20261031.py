import numpy as np
import pytest

from experiments.risk_aligned_online.state_capacity_20261031.prepare import choose_endpoint


def test_fn_selects_only_strictly_timely_and_actionable_maximum():
    times = np.arange(80)/1000
    scores = np.zeros(80)
    scores[0], scores[49], scores[50], scores[51] = .99, .7, .8, .9
    assert choose_endpoint(times, scores, True, .08) == 49  # index50 lead=30, excluded


def test_fp_keeps_early_maximum_instead_of_hiding_it():
    assert choose_endpoint(np.arange(80)/1000, np.r_[.9, np.zeros(79)], False, None) == 0


def test_ties_choose_first_and_no_legal_fn_is_recorded():
    assert choose_endpoint(np.arange(80)/1000, np.ones(80), True, .08) == 16
    assert choose_endpoint(np.arange(20)/1000, np.ones(20), True, .03) is None


@pytest.mark.parametrize('scores', [[], [np.nan]*80, np.ones((80, 1))])
def test_invalid_curve_is_rejected(scores):
    with pytest.raises(ValueError):
        choose_endpoint(np.arange(80)/1000, scores, False, None)
