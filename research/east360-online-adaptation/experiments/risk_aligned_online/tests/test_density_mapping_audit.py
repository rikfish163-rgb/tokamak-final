import numpy as np
import pytest

from experiments.risk_aligned_online.density_mapping_audit import (
    density_coordinates, eligible_peak, change_type, source_coordinate_path,
    initial_density,
)


def test_density_coordinates_are_original_source_and_ended_target_normalization():
    v = np.array([0., 2., 7.], np.float32)
    old = v.copy()
    result = density_coordinates(v, source_center=2., source_scale=3.,
                                 target_center=5., target_scale=2.)
    np.testing.assert_allclose(result['raw_source'], np.arcsinh((v-2.)/3.))
    np.testing.assert_allclose(result['mapped_source'], np.arcsinh((v-5.)/2.))
    np.testing.assert_array_equal(v, old)


def test_density_summary_uses_same_scale_floor_as_existing_provider():
    result = density_coordinates(np.array([2.]), source_center=0., source_scale=1.,
                                 target_center=2., target_scale=.001)
    assert result['effective_target_scale'] == .02
    np.testing.assert_array_equal(result['mapped_source'], [0.])


@pytest.mark.parametrize('field', ['source_scale', 'target_scale'])
def test_nonpositive_scale_rejected(field):
    kwargs = dict(source_center=0., source_scale=1., target_center=0., target_scale=1.)
    kwargs[field] = 0.
    with pytest.raises(ValueError, match='scale'):
        density_coordinates(np.ones(2), **kwargs)


def test_peak_rejects_exact_thirty_ms_and_after_disruption():
    times = np.array([0., .069, .070, .099, .101])
    scores = np.array([.4, .5, .99, 1., 1.])
    assert eligible_peak(times, scores, disruptive=True, down=.100) == 1
    assert eligible_peak(times, scores, disruptive=False, down=None) == 3


def test_no_eligible_peak_is_explicit_not_an_invalid_argmax():
    assert eligible_peak(np.array([.09]), np.array([.99]),
                         disruptive=True, down=.1) is None


@pytest.mark.parametrize('d,a,b,expected', [
    (True, True, False, 'lost'), (True, False, True, 'rescued'),
    (False, True, False, 'removed_fp'), (False, False, True, 'new_fp'),
    (True, True, True, 'unchanged'), (False, False, False, 'unchanged'),
])
def test_paired_changes_separate_disruptive_and_safe(d, a, b, expected):
    assert change_type(d, a, b) == expected


def test_path_interpolates_the_coordinate_actually_supplied_to_embedding():
    x = np.full((128, 23), 3., np.float32)
    mapped = x.copy(); mapped[:, 12] = .001
    center, scale = .002, .001
    path = source_coordinate_path(x, mapped, (0., .5, 1.),
                                  source_center=center, source_scale=scale)
    z = np.arcsinh((path[:, :, 12].astype(float)-center)/scale)
    np.testing.assert_allclose(z[1], (z[0]+z[2])/2, atol=2e-6)
    np.testing.assert_array_equal(path[0], x)
    np.testing.assert_array_equal(path[-1], mapped)
    other = np.arange(23) != 12
    for row in path:
        np.testing.assert_array_equal(row[:, other], x[:, other])


def test_initial_density_uses_elapsed_timestamp_not_a_claimed_physical_start():
    values = np.zeros((4, 23), np.float32); values[:, 12] = [1., 2., 99., 100.]
    s = dict(values=values, available=np.ones_like(values, bool),
             times=np.array([3., 3.099, 3.100, 3.2]))
    np.testing.assert_array_equal(initial_density(s), [1., 2.])
    s['available'][0, 12] = False
    np.testing.assert_array_equal(initial_density(s), [2.])
