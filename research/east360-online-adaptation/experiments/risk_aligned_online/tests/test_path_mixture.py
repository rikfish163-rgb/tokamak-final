import numpy as np
import pytest

from experiments.risk_aligned_online.path_mixture_20261023.learning import (
    signal_context, fit_context, whiten, neighbors, mixture_energy, score_paths,
)


def test_context_is_past_only_and_marks_missing():
    x = np.arange(80*23).reshape(80, 23).astype(float)
    signal = dict(values=x, available=np.ones_like(x, bool))
    active = np.ones(23, bool)
    h, valid = signal_context(signal, [60], np.zeros(23), np.ones(23), active)
    q = np.arcsinh(x)
    np.testing.assert_allclose(h[0], np.r_[q[60], q[36:61].mean(0)-q[11:36].mean(0), q[60]-q[52]])
    x[61:] = -1e15
    np.testing.assert_array_equal(signal_context(signal, [60], np.zeros(23), np.ones(23), active)[0], h)
    signal['available'][15, 2] = False
    assert not signal_context(signal, [60], np.zeros(23), np.ones(23), active)[1][0]
    with pytest.raises(ValueError):
        signal_context(dict(signal, y=1), [60], np.zeros(23), np.ones(23), active)
    assert valid[0]


def test_whitening_and_one_neighbor_per_shot_class():
    train = np.array([[0., 1.], [1., 0.], [3., 1.], [4., 0.], [10., 2.], [11., 2.]])
    ids = np.array([1, 1, 2, 2, 3, 3]); y = np.array([0, 1, 0, 1, 0, 1])
    transform = fit_context(train, np.ones(6)/6)
    xx = whiten(train, transform); query = whiten(np.array([[.2, 1.1]]), transform)
    result = neighbors(xx, query, ids, y, device='cpu')
    for c in (0, 1):
        indexes, distances, shotids = result[c]
        assert shotids.tolist() == [1, 2, 3]
        assert indexes.shape == (1, 3)
        assert (y[indexes] == c).all()
        np.testing.assert_allclose(distances[0], ((xx[indexes[0]]-query)**2).mean(1), atol=1e-12)
    with pytest.raises(ValueError):
        fit_context(np.ones((5, 2)), np.ones(5)/5)


def test_mixture_retains_opposite_paths_and_is_numerically_stable():
    paths = np.array([[[[-2.]], [[2.]]]])  # one query, two paths, one time/channel
    target = np.array([[[2.]]]); valid = np.ones_like(target, bool)
    mean, mix, individual = score_paths(paths, target, valid, np.eye(1), np.ones(1, bool))
    assert mean[0] == pytest.approx(4.)
    assert mix[0] == pytest.approx(-np.log((np.exp(-16)+1)/2))
    np.testing.assert_allclose(individual, [[16., 0.]])
    np.testing.assert_allclose(mixture_energy(np.array([[10000., 10000.]])), [10000.])
    assert np.isnan(mixture_energy(np.array([[np.nan, np.nan]])))[0]


def test_forecast_tail_and_unobserved_targets_do_not_change_partial_score():
    rng = np.random.default_rng(3)
    paths = rng.normal(size=(2, 3, 25, 2)); target = rng.normal(size=(2, 5, 2))
    valid = np.ones_like(target, bool); cov = np.array([[1., .4], [.4, 2.]])
    active = np.ones(2, bool)
    first = score_paths(paths, target, valid, cov, active)
    paths[:, :, 5:] = 1e15
    for a, b in zip(first, score_paths(paths, target, valid, cov, active)):
        np.testing.assert_array_equal(a, b)
    assert np.isnan(score_paths(paths, target, np.zeros_like(valid), cov, active)[0]).all()
