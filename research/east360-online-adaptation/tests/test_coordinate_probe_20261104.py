import numpy as np
import pytest

from experiments.risk_aligned_online.coordinate_probe_20261104 import (
    compare_raw, prepare_views, select_cohorts, summarize, uniform_indices,
)


def _shot(device, n=3):
    values = np.zeros((n, 23), dtype=np.float32)
    available = np.ones_like(values, dtype=bool)
    if device == 'EAST':
        available[:, 2] = False  # simple_data's actual EAST model contract
    return dict(device=device, shot=1, values=values, available=available)


def _map():
    gain, offset = np.ones(23), np.zeros(23)
    gain[12], offset[12] = 2., 3.
    return dict(gain=gain.tolist(), offset=offset.tolist())


def test_uniform_sampling_is_bounded_repeatable_and_covers_endpoints():
    assert np.array_equal(uniform_indices(3), [0, 1, 2])
    indices = uniform_indices(500)
    assert len(indices) == 128 and indices[0] == 0 and indices[-1] == 499
    assert np.all(np.diff(indices) > 0)
    assert np.array_equal(indices, uniform_indices(500))
    with pytest.raises(ValueError):
        uniform_indices(0)


def test_east_sentinel_and_effective_mnm_mask_but_zeros_remain_observed():
    shot = _shot('EAST')
    shot['values'][:, 1] = [-10., 0., -9.99]
    shot['values'][:, 12] = [0., 1., 2.]
    original_values, original_mask = shot['values'].copy(), shot['available'].copy()
    views = prepare_views(shot, _map())
    before, after = views['unmapped'], views['mapped']
    assert not before['available'][0, 1]
    assert before['available'][1, 1] and before['values'][1, 1] == 0
    assert not before['available'][:, 2].any()
    assert before['available'][0, 12] and before['values'][0, 12] == 0
    assert np.array_equal(before['available'], after['available'])
    assert np.array_equal(after['values'][:, 12], [3., 5., 7.])
    assert np.array_equal(shot['values'], original_values)
    assert np.array_equal(shot['available'], original_mask)


def test_source_ciii_does_not_use_the_east_sentinel():
    shot = _shot('J-TEXT')
    shot['values'][0, 1] = -10.
    views = prepare_views(shot, _map())
    assert views['unmapped']['available'][0, 1]
    assert 'mapped' not in views


def test_prefix_and_full_have_separate_sampling_and_true_mask_counts():
    shot = _shot('EAST', 256)
    shot['values'][128:, 0] = 100.
    shot['available'][:64, 0] = False
    physical_finite = np.ones((256, 23), bool)
    record = {
        'values': shot['values'], 'available': shot['available'],
        'raw_finite': physical_finite,
    }
    center, scale = np.zeros(23, np.float32), np.ones(23, np.float32)
    prefix = summarize([record], center, scale, prefix=True)
    full = summarize([record], center, scale, prefix=False)
    assert prefix[0]['total_points'] == 128
    assert prefix[0]['available_points'] == 64
    assert prefix[0]['asinh_median'] == 0
    assert full[0]['total_points'] == 128
    assert full[0]['available_points'] == 96
    assert full[0]['sample_points'] == 128
    assert full[0]['asinh_median'] > 5
    assert full[2]['available_points'] == 0
    assert full[2]['finite_hdf_points'] == 128
    assert full[2]['asinh_median'] is None


def test_thresholds_strict_and_ratios_refuse_undefined_denominators():
    shot = _shot('J-TEXT', 3)
    shot['values'][:, 0] = np.sinh(np.array([0., 6., 11.], np.float32))
    rec = dict(values=shot['values'], available=shot['available'],
               raw_finite=np.ones((3, 23), bool))
    stats = summarize([rec], np.zeros(23, np.float32), np.ones(23, np.float32), prefix=False)
    assert stats[0]['frac_abs_gt5'] == pytest.approx(2/3)
    assert stats[0]['frac_abs_gt10'] == pytest.approx(1/3)
    assert compare_raw({'raw_median': 0., 'raw_iqr': 0.}, stats[0]) == {
        'median_east_over_source': None, 'iqr_east_over_source': None,
    }
    assert compare_raw({'raw_median': 2., 'raw_iqr': 4.},
                       {'raw_median': 6., 'raw_iqr': 12.}) == {
        'median_east_over_source': 3., 'iqr_east_over_source': 3.,
    }


def test_cohorts_are_exact_sorted_train_and_locked_first20():
    roles = {'train': list(range(70, 0, -1)), 'val': [71], 'test': [72]}
    order = list(range(100, 460))
    source, east = select_cohorts(roles, order, order, order[:20])
    assert source == list(range(1, 65)) and east == order[:20]
    wrong = order.copy()
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(ValueError):
        select_cohorts(roles, order, wrong, order[:20])
    with pytest.raises(ValueError):
        select_cohorts({**roles, 'val': [1]}, order, order, order[:20])
    with pytest.raises(ValueError):
        select_cohorts(roles, order, order, wrong[:20])
