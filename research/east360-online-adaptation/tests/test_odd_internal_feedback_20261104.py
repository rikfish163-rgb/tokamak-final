import numpy as np

from odd_internal_feedback_20261104 import odd_direction


def test_odd_part_cancels_intercept_state_and_center():
    rng = np.random.default_rng(8)
    m = rng.normal(size=(5, 21))
    state = rng.normal(size=(5, 64))
    scale = rng.uniform(0.5, 2, 85)
    coef = rng.normal(size=(86, 4))
    fit = dict(mean=rng.normal(size=85), scale=scale, coef=coef)
    got = odd_direction(fit, m)
    plus = np.c_[np.ones(5), (np.c_[state, m] - fit["mean"]) / scale] @ coef
    minus = np.c_[np.ones(5), (np.c_[state, -m] - fit["mean"]) / scale] @ coef
    np.testing.assert_allclose(got, (plus - minus) / 2, atol=1e-14)
    np.testing.assert_allclose(odd_direction(fit, -m), -got)
    np.testing.assert_array_equal(odd_direction(fit, np.zeros_like(m)), 0)
