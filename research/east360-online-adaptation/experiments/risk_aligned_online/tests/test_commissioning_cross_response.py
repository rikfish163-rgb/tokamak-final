import numpy as np

from experiments.risk_aligned_online import commissioning_cross_response_20261111 as response


def test_prefix_and_first_release_are_independent_of_unseen_signal():
    rng = np.random.default_rng(31)
    x = rng.normal(scale=.08, size=(250, 6))
    m = np.ones_like(x, dtype=bool)
    q = np.array([127, 159])
    full = response.predictors(x, m, q, response.MOTION)
    partial = response.predictors(x[:160], m[:160], q, response.MOTION)
    for k in ('own', 'real', 'lagged', 'support', 'lag'):
        np.testing.assert_array_equal(full[k], partial[k])
    assert full['support'].tolist() == [True, True]
    assert np.all(full['lag'] >= 32)
    incomplete = response.observations(x[:153], m[:153], full, response.MOTION)
    assert incomplete['use'].size == 0
    first = response.observations(x[:154], m[:154], full, response.MOTION)
    np.testing.assert_array_equal(first['use'], [0])
    assert first['target'].shape == (1, 50)


def test_missing_at_release_does_not_change_issued_features():
    x = np.ones((154, 6))
    m = np.ones_like(x, dtype=bool)
    q = np.array([127])
    issued = response.predictors(x, m, q, response.MOTION)
    m[144, response.MOTION[0]] = False
    checked = response.predictors(x, m, q, response.MOTION)
    np.testing.assert_array_equal(issued['real'], checked['real'])
    assert checked['support'].tolist() == [True]
    assert response.observations(x, m, checked, response.MOTION)['use'].size == 0


def test_fit_detects_new_information_not_present_in_wrong_time_input():
    rng = np.random.default_rng(32)
    own, real, lagged = (rng.normal(size=(600, k)) for k in (5, 2, 2))
    real += .9 * own[:, :2]
    y = own[:, :2] + real * 2.
    fit = response.fit(own[:300], real[:300], lagged[:300], y[:300], np.repeat(np.arange(6), 50))
    p = response.forecasts(fit, own[300:], real[300:], lagged[300:])
    mae = abs(p-y[300:, None]).mean(axis=(0, 2))
    assert mae[2] < .1 * mae[1]
    assert mae[2] < .1 * mae[3]
    zero = response.forecasts(fit, own[300:], real[300:], lagged[300:])[:, 0]
    assert not zero.any()
