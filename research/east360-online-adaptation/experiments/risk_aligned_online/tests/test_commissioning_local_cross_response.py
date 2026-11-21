import numpy as np

from experiments.risk_aligned_online import commissioning_local_cross_response_20261111 as local


def test_the_observed_block_changes_only_subsequent_forecasts():
    own = np.zeros((7, 1))
    real = np.ones((7, 1))
    lagged = real.copy()
    y = np.tile([.2, .3], (7, 1))
    valid = np.ones(len(y), dtype=bool)
    base, _ = local.replay(own, real, lagged, y, valid)
    changed = y.copy()
    changed[2] *= -3
    other, _ = local.replay(own, real, lagged, changed, valid)
    np.testing.assert_array_equal(base[:3], other[:3])
    j = local.ARMS.index('real')
    k = local.ARMS.index('late_feedback')
    assert np.any(base[3, j] != other[3, j])
    np.testing.assert_array_equal(base[:4, k], other[:4, k])
    assert np.any(base[4, k] != other[4, k])


def test_a_missing_target_cannot_write_parameters_and_shot_reset_is_exact():
    own, real = np.ones((6, 2)), np.ones((6, 1))
    y = np.tile([.4, .7], (6, 1))
    valid = np.ones(len(y), dtype=bool)
    valid[2] = False
    a, _ = local.replay(own, real, real, y, valid)
    y[2] = 10000
    b, _ = local.replay(own, real, real, y, valid)
    np.testing.assert_array_equal(a, b)
    assert not a[0].any()
    assert np.any(a[1:])


def test_online_cross_history_can_learn_a_non_autoregressive_response():
    rng = np.random.default_rng(131)
    real, lagged = rng.normal(size=(50, 2)), rng.normal(size=(50, 2))
    own = rng.normal(size=(50, 2))
    y = real @ np.array([[.3, .1, .4], [.1, .4, -.3]])
    p, _ = local.replay(own, real, lagged, y, np.ones(len(y), dtype=bool))
    err = np.square(p[15:]-y[15:, None]).mean(axis=(0, 2))
    assert err[local.ARMS.index('real')] < .3 * err[local.ARMS.index('own')]
    assert err[local.ARMS.index('real')] < .3 * err[local.ARMS.index('lagged_source')]
