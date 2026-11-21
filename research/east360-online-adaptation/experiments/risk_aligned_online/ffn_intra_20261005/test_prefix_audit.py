"""Irrevocable-prefix bounds are diagnostics, not achievable online scores."""
import numpy as np
import pytest

from .prefix_audit import prefix_bound, summarize_bounds


def case(scores, *, disruptive=False, down=None, times=None, valid=None):
    n = len(scores)
    return dict(shot=1, scores=np.asarray(scores),
                times=np.arange(n) * .001 if times is None else np.asarray(times),
                valid=np.ones(n, bool) if valid is None else np.asarray(valid),
                is_disrupt=disruptive, down=down)


def test_alarm_at_effect_index_is_not_in_fixed_prefix():
    row = case([.1, .8, .2])
    assert not prefix_bound(row, .8, 1)['unavoidable_fp']
    assert prefix_bound(row, .8, 2)['unavoidable_fp']


def test_safe_prefix_floor_does_not_claim_later_alarm_is_rescuable():
    result = prefix_bound(case([.1, .8]), .5, 1)
    assert result['static_class'] == 'FP'
    assert not result['unavoidable_fp']  # Upper freedom, not evidence of rescue.


def test_strict_30ms_boundary_is_not_a_timely_opportunity():
    row = case([.1, .1], disruptive=True, down=.040, times=[.009, .010])
    assert not prefix_bound(row, .5, 0)['unavoidable_miss']
    assert prefix_bound(row, .5, 1)['unavoidable_miss']


def test_existing_timely_alarm_remains_locked_even_without_future_opportunity():
    row = case([.8, .1], disruptive=True, down=.040, times=[.009, .010])
    result = prefix_bound(row, .5, 1)
    assert result['locked_timely'] and not result['unavoidable_miss']
    assert result['static_class'] == 'TP'


def test_invalid_prefix_does_not_compress_time_index():
    row = case([.8, .8, .1], valid=[False, True, True])
    assert not prefix_bound(row, .5, 1)['unavoidable_fp']
    assert prefix_bound(row, .5, 2)['unavoidable_fp']


def test_no_effect_means_every_emitted_point_is_fixed():
    row = case([.1, .8])
    assert prefix_bound(row, .5, None)['unavoidable_fp']


def test_elapsed_time_uses_actual_grid_not_nominal_ms():
    row = case([.1, .8], times=[2., 2.0021])
    assert prefix_bound(row, .5, 1)['first_alarm_elapsed_ms'] == pytest.approx(2.1)


def test_denominators_keep_unrescuable_and_short_shots():
    rows = [prefix_bound(case([.8]), .5, None),
            prefix_bound(case([.1]), .5, None),
            prefix_bound(case([.1], disruptive=True, down=.030), .5, 0),
            prefix_bound(case([.8], disruptive=True, down=.031), .5, None)]
    result = summarize_bounds(rows)
    assert result['n_safe'] == result['n_disrupt'] == 2
    assert result['fpr_lower_bound'] == .5
    assert result['tpr_upper_bound'] == .5
    assert result['static_tp'] == 1 and result['static_fp'] == 1


@pytest.mark.parametrize('effect', [-1, 1.5, True])
def test_invalid_effect_rejected(effect):
    with pytest.raises(ValueError):
        prefix_bound(case([.1, .2]), .5, effect)


def test_bad_valid_mask_rejected():
    with pytest.raises(ValueError):
        prefix_bound(case([.1, .2], valid=[1, 1]), .5, 1)


def test_post_terminal_score_cannot_lock_a_false_timely_alarm():
    row = case([.1, .9], disruptive=True, down=.001, times=[0., .002])
    result = prefix_bound(row, .5, 2)
    assert result['static_class'] == 'FN'
    assert result['unavoidable_miss']
