import numpy as np
import pytest

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.scale_policy import scale_variant


def fixture():
    source_center = np.arange(23, dtype=np.float32)
    source_scale = np.full(23, .3, np.float32)
    statistics = dict(center=np.arange(23, dtype=np.float32) + 100,
                      scale=np.full(23, .002, np.float32),
                      usable=np.ones(23, bool))
    return source_center, source_scale, statistics


def test_floor_matches_source_scaler_arithmetic_not_an_extra_tuned_parameter():
    center, scale, stats = fixture()
    stats['center'][0] = 0
    stats['scale'][0] = 1e-9
    variant = scale_variant(center, scale, stats, 'source_floor')
    np.testing.assert_array_equal(variant['center'], stats['center'])
    np.testing.assert_allclose(variant['scale'],
                              np.maximum.reduce((stats['scale'], abs(stats['center'])*.01,
                                                 np.full(23, 1e-6))))


def test_preserve_controls_really_keeps_bt_and_ip_coordinates():
    center, scale, stats = fixture()
    result = scale_variant(center, scale, stats, 'source_floor_preserve_controls')
    for name in ('Bt', 'ip'):
        c = io23.RAW23.index(name)
        assert result['center'][c] == center[c]
        assert result['scale'][c] == scale[c]
    assert result['center'][io23.RAW23.index('CIII')] == stats['center'][1]


def test_density_only_changes_exactly_one_channel_not_the_risk_output():
    center, scale, stats = fixture()
    result = scale_variant(center, scale, stats, 'density_only')
    changed = np.flatnonzero((result['center'] != center) | (result['scale'] != scale))
    assert changed.tolist() == [io23.RAW23.index('ne_nG')]


@pytest.mark.parametrize('policy', ['target_robust_scale', 'source_floor',
                                  'source_floor_preserve_controls', 'density_only'])
def test_unusable_channel_mask_is_identical_and_no_inputs_are_modified(policy):
    center, scale, stats = fixture()
    stats['usable'][3] = False
    original = {k: v.copy() for k, v in stats.items()}
    source_center, source_scale = center.copy(), scale.copy()
    result = scale_variant(center, scale, stats, policy)
    np.testing.assert_array_equal(result['usable'], stats['usable'])
    np.testing.assert_array_equal(center, source_center)
    np.testing.assert_array_equal(scale, source_scale)
    for key, value in original.items():
        np.testing.assert_array_equal(stats[key], value)
    result['center'][0] += 1
    result['usable'][0] = False
    np.testing.assert_array_equal(center, source_center)
    np.testing.assert_array_equal(stats['center'], original['center'])
    np.testing.assert_array_equal(stats['usable'], original['usable'])


def test_unfloored_variant_reproduces_original_calibration():
    center, scale, stats = fixture()
    result = scale_variant(center, scale, stats, 'target_robust_scale')
    np.testing.assert_array_equal(result['center'], stats['center'])
    np.testing.assert_array_equal(result['scale'], stats['scale'])


@pytest.mark.parametrize('fault', ['unknown', 'shape', 'nonfinite', 'zero_scale', 'mask_type'])
def test_invalid_numeric_contract_rejected(fault):
    center, scale, stats = fixture()
    policy = 'source_floor'
    if fault == 'unknown':
        policy = 'new_untested_fallback'
    elif fault == 'shape':
        center = center[:22]
    elif fault == 'nonfinite':
        stats['center'][2] = np.nan
    elif fault == 'zero_scale':
        stats['scale'][2] = 0
    else:
        stats['usable'] = stats['usable'].astype(float)
    with pytest.raises(ValueError):
        scale_variant(center, scale, stats, policy)
