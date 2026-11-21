import numpy as np

from experiments.risk_aligned_online.path_mixture_20261023.audit import reference_context, reference_neighbors, reference_scores
from experiments.risk_aligned_online.path_mixture_20261023.learning import signal_context, neighbors, score_paths


def test_independent_context_and_retrieval():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(80, 23)); a = np.ones_like(x, bool)
    a[12, 2] = False
    args = (dict(values=x, available=a), np.array([60, 70]), np.zeros(23), np.ones(23), np.ones(23, bool))
    for actual, expected in zip(reference_context(*args), signal_context(*args)):
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    bank = rng.normal(size=(30, 4)); q = rng.normal(size=(3, 4))
    ids = np.repeat([1, 2, 3], 10); labels = np.tile([0, 1], 15)
    actual = reference_neighbors(bank, q, ids, labels)
    expected = neighbors(bank, q, ids, labels, device='cpu')
    for c in (0, 1):
        for left, right in zip(actual[c], expected[c]):
            np.testing.assert_allclose(left, right, atol=1e-12)


def test_independent_marginal_errors_and_mixture():
    rng = np.random.default_rng(38)
    paths = rng.normal(size=(4, 3, 25, 3)); truth = rng.normal(size=(4, 5, 3))
    valid = np.ones_like(truth, bool); valid[0] = False; valid[1, :, 1] = False
    cov = np.eye(3)+.2*np.ones((3, 3)); active = np.ones(3, bool)
    for actual, expected in zip(reference_scores(paths, truth, valid, cov, active), score_paths(paths, truth, valid, cov, active)):
        np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-12)
