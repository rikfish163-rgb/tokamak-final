import numpy as np
import pytest

from experiments.risk_aligned_online.readout_identifiability_20261029.audit import (
    recount,
    numpy_fit,
)


def test_independent_recount_equal_shots_and_rejects_supported_nan():
    a = dict(
        ids=np.array([1, 1, 1, 2]),
        current=np.zeros((4, 23)),
        future=np.zeros((4, 25, 23)),
        mask=np.ones((4, 23), bool),
    )
    p = np.zeros((4, 25, 23))
    p[-1] = 2
    a.update(prediction_hold=np.zeros_like(p), prediction_model=p)
    r = recount(a)
    assert r["model"]["mse"] == 2 and r["model"]["smooth_l1"] == 0.75
    assert r["model"]["beats_hold_shots"] == 0
    a["prediction_model"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        recount(a)


def test_independent_numpy_solver_has_unpenalized_intercept_and_constant_zero():
    x = np.array([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0]])
    y = np.ones((3, 2)) * [3, -4]
    f = numpy_fit(x, y, np.array([1, 1, 2]))
    np.testing.assert_allclose(f["coef"], 0, atol=1e-12)
    np.testing.assert_array_equal(f["target_mean"], [3, -4])
    assert f["scale"][1] == 1
