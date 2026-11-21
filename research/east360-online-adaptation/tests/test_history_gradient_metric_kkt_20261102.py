"""Independent CPU checks for the bounded diagonal quadratic solver."""

import numpy as np
import pytest

from experiments.risk_aligned_online.history_gradient_metric_20261102.core import solve


def _objective(a, b, p):
    return float(np.sum(a * p**2 - 2.0 * b * p))


def _assert_kkt(a, b, p, mu):
    """Check the primal and dual conditions without reimplementing the solver."""
    assert isinstance(p, np.ndarray)
    assert p.dtype == np.float64
    assert p.shape == a.shape
    assert np.all(np.isfinite(p))
    assert np.isscalar(mu) and np.isfinite(mu)
    assert np.all(p >= 0.0)
    assert np.all(p <= 4.0)
    np.testing.assert_allclose(p.sum(), float(a.size), rtol=0.0, atol=1e-10)

    gradient = a * p - b + mu
    lower = p == 0.0
    upper = p == 4.0
    interior = (p > 0.0) & (p < 4.0)
    assert np.all(gradient[lower] >= -1e-10)
    assert np.all(gradient[upper] <= 1e-10)
    np.testing.assert_allclose(gradient[interior], 0.0, rtol=0.0, atol=1e-10)

    reference = np.ones_like(a)
    assert _objective(a, b, p) <= _objective(a, b, reference) + 1e-10


@pytest.mark.parametrize(
    ("a", "b", "expected_p", "expected_mu"),
    [
        ([1.0, 2.0], [2.0, 2.0], [4.0 / 3.0, 2.0 / 3.0], 2.0 / 3.0),
        ([1.0, 1.0], [-2.0, 3.0], [0.0, 2.0], 1.0),
        (
            [1.0, 2.0, 4.0],
            [2.0, 2.0, 2.0],
            [12.0 / 7.0, 6.0 / 7.0, 3.0 / 7.0],
            2.0 / 7.0,
        ),
        ([1.0, 1.0, 1.0], [-2.0, 2.0, 3.0], [0.0, 1.0, 2.0], 1.0),
        (
            [1.0] * 5,
            [10.0, 1.0, 1.0, 1.0, 1.0],
            [4.0, 0.25, 0.25, 0.25, 0.25],
            0.75,
        ),
    ],
)
def test_hand_calculated_solutions(a, b, expected_p, expected_mu):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    p, mu = solve(a, b)

    np.testing.assert_allclose(p, expected_p, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(mu, expected_mu, rtol=0.0, atol=1e-10)
    _assert_kkt(a, b, p, mu)


@pytest.mark.parametrize(
    ("a_value", "b_value"), [(1.0 / 256.0, -1.0), (0.25, 0.0), (1.0, 2.0)]
)
def test_single_coordinate_is_forced_to_one(a_value, b_value):
    a = np.array([a_value], dtype=np.float64)
    b = np.array([b_value], dtype=np.float64)
    p, mu = solve(a, b)

    np.testing.assert_allclose(p, [1.0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(mu, b_value - a_value, rtol=0.0, atol=1e-10)
    _assert_kkt(a, b, p, mu)


def test_only_bound_coordinates_allow_nonunique_multiplier():
    a = np.ones(8, dtype=np.float64)
    b = np.array([10.0, 9.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0])
    p, mu = solve(a, b)

    np.testing.assert_array_equal(p, [4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _assert_kkt(a, b, p, mu)


def test_seeded_random_kkt_feasibility_and_objective():
    rng = np.random.default_rng(20261102)
    observed = set()
    for dimension in (2, 3, 5, 8, 32, 256):
        for _ in range(8):
            a = np.exp(rng.uniform(np.log(1.0 / 256.0), 0.0, size=dimension))
            b = rng.normal(0.0, 1.0, size=dimension)
            p, mu = solve(a, b)
            _assert_kkt(a, b, p, mu)
            if np.any(p == 0.0):
                observed.add("lower")
            if np.any((p > 0.0) & (p < 4.0)):
                observed.add("interior")
            if np.any(p == 4.0):
                observed.add("upper")
    assert observed == {"lower", "interior", "upper"}


@pytest.mark.parametrize("scale", [1e-4, 0.25, 17.0, 1e4])
def test_common_positive_scaling_preserves_solution(scale):
    a = np.array([0.25, 0.5, 0.75, 1.0, 0.125, 0.2, 0.4, 0.6])
    expected = np.array([4.0, 0.0, 0.5, 1.0, 1.5, 0.3, 0.7, 0.0])
    b = a * expected + 0.3
    b[0] += 1.0
    b[[1, 7]] -= [0.5, 2.0]
    p, mu = solve(a, b)
    scaled_p, scaled_mu = solve(scale * a, scale * b)

    np.testing.assert_allclose(p, expected, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(scaled_p, p, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        scaled_mu, scale * mu, rtol=0.0, atol=1e-10 * max(1.0, scale)
    )
    _assert_kkt(a, b, p, mu)


@pytest.mark.parametrize("dimension", [2, 3, 5, 8])
def test_matches_independent_slsqp(dimension):
    # Import locally so a missing optional SciPy fails this check visibly while
    # leaving the independent KKT checks runnable and their evidence explicit.
    from scipy.optimize import minimize

    rng = np.random.default_rng(29000 + dimension)
    for _ in range(4):
        a = rng.uniform(0.25, 2.0, size=dimension)
        b = rng.normal(0.0, 3.0, size=dimension)
        p, mu = solve(a, b)
        result = minimize(
            lambda x: _objective(a, b, x),
            np.ones(dimension),
            jac=lambda x: 2.0 * (a * x - b),
            bounds=[(0.0, 4.0)] * dimension,
            constraints={
                "type": "eq",
                "fun": lambda x: np.sum(x) - dimension,
                "jac": lambda x: np.ones_like(x),
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        assert result.success, result.message
        _assert_kkt(a, b, p, mu)
        np.testing.assert_allclose(p, result.x, rtol=0.0, atol=2e-5)
        np.testing.assert_allclose(_objective(a, b, p), result.fun, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([np.nan, 1.0], [0.0, 0.0]),
        ([np.inf, 1.0], [0.0, 0.0]),
        ([-np.inf, 1.0], [0.0, 0.0]),
        ([1.0, 1.0], [np.nan, 0.0]),
        ([1.0, 1.0], [np.inf, 0.0]),
        ([1.0, 1.0], [-np.inf, 0.0]),
        ([0.0, 1.0], [0.0, 0.0]),
        ([-1.0, 1.0], [0.0, 0.0]),
        ([1.0, 1.0], [0.0]),
        ([[1.0, 1.0]], [[0.0, 0.0]]),
        ([[1.0, 1.0]], [0.0, 0.0]),
    ],
)
def test_invalid_inputs_raise_value_error(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with pytest.raises(ValueError):
        solve(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([np.finfo(np.float64).max, 1.0], [0.0, 0.0]),
        (
            [1.0 / 256.0, 1.0],
            [-np.finfo(np.float64).max, np.finfo(np.float64).max],
        ),
    ],
)
def test_extreme_finite_inputs_raise_instead_of_returning_identity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with pytest.raises(FloatingPointError):
        solve(a, b)
