"""Tests for the diagnostic decomposition, not a new predictive model."""
import numpy as np
import pytest
from scipy.stats import multivariate_t

from experiments.risk_aligned_online.conditional_likelihood_20261029.diagnose import terms


def test_terms_match_normalized_scipy_density():
    rng = np.random.default_rng(130)
    target, mean = rng.normal(size=(2, 7, 25))
    a = rng.normal(size=(25, 25))
    scale = a @ a.T + np.eye(25)
    normalizer, tail, distance = terms(target, mean, np.linalg.cholesky(scale))
    expected = [multivariate_t.logpdf(t, loc=m, shape=scale, df=4)
                for t, m in zip(target, mean)]
    np.testing.assert_allclose(normalizer + tail, expected, rtol=1e-12)
    np.testing.assert_allclose(distance, np.einsum('ni,ij,nj->n',
        target-mean, np.linalg.inv(scale), target-mean), rtol=1e-12)


def test_exact_mean_has_zero_distance_and_tail():
    x = np.ones((3, 25))
    _, tail, distance = terms(x, x, np.eye(25))
    np.testing.assert_array_equal(tail, 0)
    np.testing.assert_array_equal(distance, 0)


def test_nonfinite_input_is_rejected():
    with pytest.raises(ValueError):
        terms(np.full((1, 25), np.nan), np.zeros((1, 25)), np.eye(25))
