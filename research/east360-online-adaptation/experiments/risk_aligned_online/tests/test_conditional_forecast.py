import numpy as np
import pytest

from experiments.risk_aligned_online.conditional_forecast_20261023.learning import (
    shot_weights, ridge_fit, ridge_predict, misplaced_labels, maturity_error,
    fit_metric, direct_targets,
)


def test_shot_weights_and_wrong_time_control():
    ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    w = shot_weights(ids, y)
    assert w[y == 0].sum() == pytest.approx(.5)
    assert w[y == 1].sum() == pytest.approx(.5)
    for c in (0, 1):
        assert w[(ids == 1) & (y == c)].sum() == pytest.approx(.25)
    wrong = misplaced_labels(ids, y)
    for sid in (1, 2):
        assert wrong[ids == sid].sum() == y[ids == sid].sum()
    assert not np.array_equal(wrong, y)


def test_ridge_matches_independent_augmented_lstsq():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(21, 4)); y = rng.normal(size=(21, 7))
    w = rng.uniform(.1, 1, 21); w /= w.sum()
    b = ridge_fit(x, y, w, .1, device='cpu')
    design = np.column_stack((x, np.ones(len(x))))
    penalty = np.diag([.1**.5]*4 + [0.])
    want = np.linalg.lstsq(np.vstack((design*w[:, None]**.5, penalty)),
                          np.vstack((y*w[:, None]**.5, np.zeros((5, 7)))), rcond=None)[0]
    np.testing.assert_allclose(b, want, atol=1e-12)
    np.testing.assert_allclose(ridge_predict(x, b), design @ want, atol=1e-12)


def test_partial_feedback_reads_only_matured_values_and_keeps_missing_missing():
    pred = np.zeros((2, 25, 3)); truth = np.ones((2, 5, 3)); mask = np.ones_like(truth, bool)
    cov = np.eye(3); active = np.ones(3, bool)
    e = maturity_error(pred, truth, mask, cov, active)
    np.testing.assert_allclose(e, 1.)
    pred[:, 5:] = 1e9
    np.testing.assert_allclose(maturity_error(pred, truth, mask, cov, active), e)
    truth[:, :, 1] = np.nan; mask[:, :, 1] = False
    np.testing.assert_allclose(maturity_error(pred, truth, mask, cov, active), e)
    assert np.isnan(maturity_error(pred, truth, np.zeros_like(mask), cov, active)).all()
    with pytest.raises(ValueError):
        maturity_error(pred, np.ones((2, 26, 3)), np.ones((2, 26, 3), bool), cov, active)


def test_covariance_uses_complete_channels_and_is_positive_definite():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(30, 25, 4)); z[:, :, 1] = z[:, :, 0]; z[:, :, 3] = 0
    mask = np.ones_like(z, bool); mask[0, :, 2] = False
    cov, active = fit_metric(z, mask, np.ones(30)/30)
    assert active.tolist() == [True, True, False, False]
    assert np.linalg.eigvalsh(cov[np.ix_(active, active)]).min() > 0


def test_direct_targets_are_future_changes_not_recursive_and_do_not_accept_labels():
    values = np.arange(40*23).reshape(40, 23).astype(float)
    sig = dict(values=values, available=np.ones_like(values, bool))
    target, valid = direct_targets(sig, [10], np.zeros(23), np.ones(23), 5)
    expected = np.arcsinh(values[11:16])-np.arcsinh(values[10])
    np.testing.assert_allclose(target[0], expected)
    before = target.copy(); sig['values'][16:] = -1e9
    np.testing.assert_array_equal(direct_targets(sig, [10], np.zeros(23), np.ones(23), 5)[0], before)
    with pytest.raises(ValueError):
        direct_targets(dict(sig, y=1), [10], np.zeros(23), np.ones(23), 5)
    assert valid.all()


def test_partial_covariance_marginal_not_precision_slice():
    from experiments.risk_aligned_online.conditional_forecast_20261023.audit import cholesky_error
    pred = np.array([[[1., 999.], [2., 3.]]])
    truth = np.zeros_like(pred); valid = np.array([[[True, False], [True, True]]])
    cov = np.array([[1., .8], [.8, 1.]])
    active = np.ones(2, bool)
    result = maturity_error(pred, truth, valid, cov, active)
    expected = (1. + np.array([2., 3.]) @ np.linalg.inv(cov) @ np.array([2., 3.])/2)/2
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result, cholesky_error(pred, truth, valid, cov, active))
