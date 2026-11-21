import numpy as np
import pytest

from experiments.risk_aligned_online.arrival_increment_20261030.diagnose import contributions


def test_exact_additive_contributions_and_missing_channel():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(3, 25, 23))
    mask = np.ones((3, 23), bool)
    mask[:, 2] = False
    coef = rng.normal(size=180)
    fit = dict(q1_coef=coef.tolist(), arrival_scale=np.full(92, 2.).tolist())
    actual = contributions(fit, values, mask)
    assert actual.shape == (3, 23, 4)
    np.testing.assert_array_equal(actual[:, 2], 0.)
    mean = values.mean(1)
    summary = np.stack((mean, values[:, -1]-values[:, 0], values.std(1), values[:, -1]), axis=2)
    expected = (summary.reshape(3, 92)/2 * coef[87:-1]).reshape(3, 23, 4)
    expected[:, 2] = 0
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_zero_arrival_gives_no_contribution():
    fit = dict(q1_coef=np.ones(180), arrival_scale=np.ones(92))
    actual = contributions(fit, np.zeros((1, 25, 23)), np.ones((1, 23), bool))
    assert not actual.any()


def test_reject_invalid_scale():
    fit = dict(q1_coef=np.ones(180), arrival_scale=np.zeros(92))
    with pytest.raises(ValueError):
        contributions(fit, np.zeros((1, 25, 23)), np.ones((1, 23), bool))
