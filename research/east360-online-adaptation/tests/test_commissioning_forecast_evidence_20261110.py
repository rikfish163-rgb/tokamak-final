import numpy as np

from experiments.risk_aligned_online.commissioning_forecast_evidence_20261110 import joined


def test_forecast_evidence_rescales_only_comparison_and_persistence():
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    evidence = np.array([[.5, -.1, .2, .4, 1., .5], [0., .2, 0., .1, 0., 1.]], np.float32)
    scale = np.array([.1, .2, .1, .4], np.float32)
    x = joined(raw, evidence, scale)
    np.testing.assert_array_equal(x[:, :3], raw)
    np.testing.assert_allclose(x[:, 3:], [[4., -.5, 2., 1., 1., .5], [0., 1., 0., .25, 0., 1.]])
    np.testing.assert_array_equal(evidence[:, :2], np.array([[.5, -.1], [0., .2]], np.float32))
