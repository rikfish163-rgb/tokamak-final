"""Contract tests for the isolated, CPU-only linear direction solver."""

from types import SimpleNamespace

import numpy as np
import pytest

import joint_paired_ball_20261106 as solver


def assert_solution(g, rows, caps, radius, u, upper, optimum=None):
    g = np.asarray(g, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, g.size)
    caps = np.asarray(caps, dtype=np.float64)
    assert isinstance(u, np.ndarray)
    assert u.dtype == np.float64
    assert u.shape == g.shape
    assert np.all(np.isfinite(u))
    assert type(upper) is float and np.isfinite(upper)
    assert np.linalg.norm(u) <= radius + 1e-8
    assert np.all(rows @ u <= caps + 1e-8)
    assert g @ u <= upper + 1e-10
    if optimum is not None:
        assert g @ u == pytest.approx(optimum, abs=1e-7, rel=1e-7)
        assert upper >= optimum - 1e-10
        assert upper == pytest.approx(optimum, abs=1e-7, rel=1e-7)


@pytest.mark.parametrize("rows,caps", [([], []), ([[1.0, 0.0]], [1.3]), ([[1.0, 0.0]], [np.inf])])
def test_unconstrained_ball_fast_path(monkeypatch, rows, caps):
    def unexpected(*args, **kwargs):
        raise AssertionError("an optimizer must not run on a feasible ball optimum")

    monkeypatch.setattr(solver, "minimize", unexpected)
    g = np.array([3.0, 4.0])
    u, upper = solver.solve(g, rows, caps, 2.0)
    np.testing.assert_allclose(u, [1.2, 1.6], atol=1e-14)
    assert_solution(g, rows, caps, 2.0, u, upper, 10.0)


def test_single_active_cap_has_analytic_optimum_and_dual_bound():
    g = np.array([1.0, 1.0])
    rows = np.array([[1.0, 0.0]])
    caps = np.array([0.25])
    optimum = 0.25 + np.sqrt(1.0 - 0.25**2)
    u, upper = solver.solve(g, rows, caps, 1.0)
    np.testing.assert_allclose(u, [0.25, np.sqrt(1.0 - 0.25**2)], atol=1e-7)
    assert_solution(g, rows, caps, 1.0, u, upper, optimum)


def test_simultaneous_guard_and_safe_constraints():
    g = np.array([1.0, -1.0, 1.0])
    rows = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    caps = np.array([0.2, 0.3])
    optimum = 0.2 + 0.3 + np.sqrt(1.0 - 0.2**2 - 0.3**2)
    u, upper = solver.solve(g, rows, caps, 1.0)
    np.testing.assert_allclose(u, [0.2, -0.3, np.sqrt(0.87)], atol=1e-7)
    assert_solution(g, rows, caps, 1.0, u, upper, optimum)


def test_ball_can_be_inactive_at_the_optimum():
    g = np.array([1.0, 1.0])
    rows = np.eye(2)
    caps = np.array([0.2, 0.3])
    u, upper = solver.solve(g, rows, caps, 10.0)
    assert_solution(g, rows, caps, 10.0, u, upper, 0.5)


@pytest.mark.parametrize("rows", [[[1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]])
def test_zero_cap_does_not_erase_a_useful_tangent_solution(rows):
    g = np.array([1.0, 1.0])
    caps = np.zeros(len(rows))
    u, upper = solver.solve(g, rows, caps, 1.0)
    np.testing.assert_allclose(u, [0.0, 1.0], atol=1e-7)
    assert_solution(g, rows, caps, 1.0, u, upper, 1.0)


def test_scaled_opposing_zero_caps_keep_the_tangent_direction():
    g = np.array([1.0, 1.0])
    rows = np.array([[1e12, 0.0], [-1e12, 0.0]])
    caps = np.zeros(2)
    u, upper = solver.solve(g, rows, caps, 1.0)
    np.testing.assert_allclose(u, [0.0, 1.0], atol=1e-7)
    assert_solution(g, rows, caps, 1.0, u, upper, 1.0)


@pytest.mark.parametrize("g,radius", [([0.0, 0.0], 1.0), ([1.0, -1.0], 0.0)])
def test_zero_objective_or_radius(g, radius):
    rows = [[1.0, 0.0]]
    caps = [0.0]
    u, upper = solver.solve(g, rows, caps, radius)
    np.testing.assert_array_equal(u, [0.0, 0.0])
    assert_solution(g, rows, caps, radius, u, upper, 0.0)


def test_known_kkt_solution_at_representative_dimensions():
    rng = np.random.default_rng(20261106)
    n, m, active = 64, 75, 5
    expected = rng.normal(size=n)
    expected /= np.linalg.norm(expected)
    rows = rng.normal(size=(m, n))
    rows /= np.linalg.norm(rows, axis=1)[:, None]
    rows *= np.where(rows @ expected < 0.0, -1.0, 1.0)[:, None]
    caps = rows @ expected + 0.5
    caps[:active] = rows[:active] @ expected
    multipliers = rng.uniform(0.3, 1.0, size=active)
    g = 0.8 * expected + rows[:active].T @ multipliers
    optimum = float(g @ expected)
    u, upper = solver.solve(g, rows, caps, 1.0)
    assert_solution(g, rows, caps, 1.0, u, upper, optimum)


def test_failed_optimizer_projects_candidate_and_warns(monkeypatch):
    def failed(*args, **kwargs):
        return SimpleNamespace(success=False, message="iteration limit", x=np.array([2.0, 1.0]))

    monkeypatch.setattr(solver, "minimize", failed)
    g = np.array([1.0, 1.0])
    rows = np.array([[1.0, 0.0]])
    caps = np.array([0.25])
    with pytest.warns(RuntimeWarning, match="iteration limit"):
        u, upper = solver.solve(g, rows, caps, 1.0)
    assert_solution(g, rows, caps, 1.0, u, upper)
    assert g @ u > 0.0
    assert upper >= 0.25 + np.sqrt(1.0 - 0.25**2) - 1e-10


def test_optimizer_success_with_nonfinite_result_is_not_silent(monkeypatch):
    def invalid(*args, **kwargs):
        return SimpleNamespace(success=True, message="success", x=np.array([np.nan, np.nan]))

    monkeypatch.setattr(solver, "minimize", invalid)
    g = np.array([1.0, 1.0])
    rows = np.array([[1.0, 0.0]])
    caps = np.array([0.25])
    with pytest.warns(RuntimeWarning, match="candidate"):
        u, upper = solver.solve(g, rows, caps, 1.0)
    assert_solution(g, rows, caps, 1.0, u, upper)
    assert g @ u > 0.0


def test_optimizer_failure_without_a_sensible_direction_is_visible(monkeypatch):
    def failed(*args, **kwargs):
        return SimpleNamespace(success=False, message="nonfinite", x=np.array([np.nan, np.nan]))

    monkeypatch.setattr(solver, "minimize", failed)
    with pytest.raises(RuntimeError, match="nonfinite|candidate|direction"):
        solver.solve([1.0, 1.0], [[1.0, 0.0]], [0.0], 1.0)


@pytest.mark.parametrize(
    "g,rows,caps,radius",
    [
        ([[1.0, 2.0]], [], [], 1.0),
        ([1.0, np.nan], [], [], 1.0),
        ([1.0, 2.0], [[np.inf, 0.0]], [0.0], 1.0),
        ([1.0, 2.0], [[1.0]], [0.0], 1.0),
        ([1.0, 2.0], [[1.0, 0.0]], [], 1.0),
        ([1.0, 2.0], [[1.0, 0.0]], [-1.0], 1.0),
        ([1.0, 2.0], [[1.0, 0.0]], [np.nan], 1.0),
        ([1.0, 2.0], [], [], -1.0),
        ([1.0, 2.0], [], [], np.inf),
    ],
)
def test_invalid_inputs_fail_visibly(g, rows, caps, radius):
    with pytest.raises(ValueError):
        solver.solve(g, rows, caps, radius)
