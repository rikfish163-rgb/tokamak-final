import numpy as np
import pytest

from experiments.risk_aligned_online.alignment import fit_ended_shot_statistics


def shot(sid, values):
    return dict(shot=sid, device='EAST', values=values,
                available=np.ones_like(values, dtype=bool))


def test_fit_is_affine_equivariant_without_labels():
    x = np.tile(np.arange(256, dtype=np.float32)[:, None], (1, 23))
    first = fit_ended_shot_statistics([shot(1, x)], ended_ids=[1])
    second = fit_ended_shot_statistics([shot(1, 3*x+17)], ended_ids=[1])
    np.testing.assert_allclose(second['center'], first['center']*3+17)
    np.testing.assert_allclose(second['scale'], first['scale']*3)
    assert first['usable'].all()


def test_unended_or_duplicate_shot_is_rejected():
    x = np.ones((256, 23), np.float32)
    with pytest.raises(ValueError, match='ended'):
        fit_ended_shot_statistics([shot(2, x)], ended_ids=[1])
    with pytest.raises(ValueError, match='duplicate'):
        fit_ended_shot_statistics([shot(1, x), shot(1, x)], ended_ids=[1])


def test_constant_and_missing_channels_are_explicitly_unusable():
    x = np.tile(np.arange(256, dtype=np.float32)[:, None], (1, 23))
    x[:, 0] = -9.999695
    s = shot(1, x)
    s['available'][:, 1] = False
    stats = fit_ended_shot_statistics([s], ended_ids=[1])
    assert not stats['usable'][0] and not stats['usable'][1]
    assert stats['usable'][2:].all()
    assert np.isfinite(stats['center']).all()
    assert (stats['scale'] > 0).all()


def test_unavailable_extremes_do_not_enter_statistics():
    x = np.tile(np.arange(256, dtype=np.float32)[:, None], (1, 23))
    a, b = shot(1, x.copy()), shot(1, x.copy())
    a['available'][:20] = b['available'][:20] = False
    b['values'][:20] = 1e30
    one = fit_ended_shot_statistics([a], ended_ids=[1])
    two = fit_ended_shot_statistics([b], ended_ids=[1])
    np.testing.assert_array_equal(one['center'], two['center'])
    np.testing.assert_array_equal(one['scale'], two['scale'])


def test_small_physical_units_are_not_confused_with_constant_channels():
    x = np.tile(np.arange(256, dtype=np.float32)[:, None], (1, 23))*1e-12
    stats = fit_ended_shot_statistics([shot(1, x)], ended_ids=[1])
    assert stats['usable'].all()
    assert (stats['scale'] < 1e-8).all()
