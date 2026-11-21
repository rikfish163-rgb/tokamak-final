import numpy as np
import pytest

from experiments.risk_aligned_online.feedback_curve_audit_20261028 import contrast


def test_constant_shift_is_not_temporal_variation():
    r = contrast([.2, .3], [.3, .4], [.3, .4], .5)
    assert r['fast_rms'] == pytest.approx(.1)
    assert r['fast_mean'] == pytest.approx(.1)
    assert r['centered_energy_fraction'] == pytest.approx(0.)
    assert r['reverse_over_fast_rms'] == pytest.approx(0.)
    assert r['up_fraction'] == 1.


def test_opposite_changes_have_zero_mean_but_nonzero_energy():
    r = contrast([.5, .5], [.6, .4], [.4, .6], .5)
    assert r['fast_rms'] == pytest.approx(.1)
    assert r['centered_energy_fraction'] == pytest.approx(1.)
    assert r['reverse_over_fast_rms'] == pytest.approx(2.)
    assert r['new_above_threshold'] == 0
    assert r['removed_above_threshold'] == 1
    assert r['reverse_threshold_disagreements'] == 2


def test_zero_effect_is_explicit_not_divide_by_epsilon():
    r = contrast([.2, .2], [.2, .2], [.2, .2], .5)
    assert r['centered_energy_fraction'] is None
    assert r['reverse_over_fast_rms'] is None
    assert r['fast_rms'] == 0.


def test_threshold_equality_is_alarm_and_does_not_imply_shot_flip():
    r = contrast([.4, .7], [.5, .7], [.49, .7], .5)
    assert r['new_above_threshold'] == 1
    assert r['reverse_threshold_disagreements'] == 1


@pytest.mark.parametrize('a,b,c,t', [
    ([], [], [], .5), ([.2], [.2, .3], [.2], .5),
    ([np.nan], [.2], [.2], .5), ([1.1], [.2], [.2], .5),
    ([[.2]], [[.2]], [[.2]], .5), ([.2], [.2], [.2], 1.),
])
def test_invalid_curves_fail(a, b, c, t):
    with pytest.raises(ValueError):
        contrast(a, b, c, t)
