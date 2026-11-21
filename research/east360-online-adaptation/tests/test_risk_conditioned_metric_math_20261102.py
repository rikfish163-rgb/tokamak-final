"""Synthetic CPU-only checks for the two-state channel quadratic program."""

import numpy as np
import pytest

from experiments.risk_aligned_online.channel_feedback_20261102 import metric
from experiments.risk_aligned_online.risk_conditioned_feedback_20261102 import core


def test_coupled_qp_respects_both_sums_and_an_active_bound():
    # The unique optimum is [0, 2, 1/2, 3/2].  Its two block-sum
    # constraints and the first coordinate's lower bound are all active.
    # Off-diagonal entries couple channels both within and across states.
    a = np.array(
        [
            [3.0, 0.35, 0.5, -0.2],
            [0.35, 2.5, -0.15, 0.4],
            [0.5, -0.15, 2.8, 0.25],
            [-0.2, 0.4, 0.25, 2.2],
        ]
    )
    b = np.array([0.15, 5.775, 0.725, 3.475])
    actual, receipt = core.solve(a, b)

    np.testing.assert_allclose(actual, [0.0, 2.0, 0.5, 1.5], atol=2e-6, rtol=0)
    np.testing.assert_allclose(np.reshape(actual, (2, 2)).sum(axis=1), 2.0, atol=1e-8, rtol=0)
    assert np.all(np.asarray(actual) >= 0.0)
    assert np.all(np.asarray(actual) <= 4.0)
    assert receipt["success"] is True
    assert receipt["projected_gradient_residual"] <= 1e-5
    assert receipt["iterations"] > 0


def test_coupled_qp_can_reach_both_box_boundaries():
    expected = np.array([4.0, 0.0, 0.25, 0.5, 0.25, 0.5, 1.0, 1.5, 1.0, 1.0])
    v = np.array([1.0, -2.0, 0.4, 0.6, -0.7, 1.5, -1.0, 0.3, -0.5, 0.8])
    a = 2 * np.eye(10) + 0.05 * np.outer(v, v)
    # These equality multipliers and bound multipliers make expected the
    # unique KKT solution of this strictly convex problem.
    b = a @ expected + np.array([0.8, -0.2, 0.3, 0.3, 0.3] + [-0.4] * 5)

    actual, receipt = core.solve(a, b)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.reshape(actual, (2, 5)).sum(axis=1), 5, atol=1e-8, rtol=0)
    assert np.all((np.asarray(actual) >= 0) & (np.asarray(actual) <= 4))
    assert receipt["success"] is True
    assert 0 <= receipt["projected_gradient_residual"] <= 1e-5


@pytest.fixture
def training():
    # Uneven points per shot and shots per label. Channel 2 is never observed;
    # channel 4 is observed only on an undefined (zero-signal) training row.
    g = np.array(
        [
            [[2, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 2], [1, -0.5], [0, 0], [1, 0], [0, 0]],
            [[1, 1], [0.5, 1], [0, 0], [2, -1], [0, 0]],
            [[-1, 2], [1, 2], [0, 0], [0.25, 0.5], [0, 0]],
            [[1, 2], [2, -0.5], [0, 0], [1, 1], [0, 0]],
            [[0.5, -1], [1, 1], [0, 0], [1, 0.5], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        ],
        dtype=float,
    )
    r = np.array([[0, 2], [1, 0], [1, 2], [-1, 1], [0.3, -2], [2, 1], [1, 0]])
    rows = [
        {"shot": shot, "query": query, "label": label, "disruptive": disruptive}
        for shot, query, label, disruptive in [
            (10, 0, 0, False),
            (10, 1, 0, False),
            (20, 0, 0, True),
            (30, 0, 1, True),
            (30, 1, 1, True),
            (30, 2, 1, True),
            (40, 0, 0, False),
        ]
    ]
    obs = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    u = np.array([0.05, 0.2, 0.35, 0.6, 0.85, 1.0, 0.4])
    return g, r, rows, obs, u


def _project_block(values):
    """Independently project onto sum=C and 0<=w<=4 by bisection."""
    lo, hi = float(np.min(values - 4)), float(np.max(values))
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.clip(values - mid, 0, 4).sum() > len(values):
            lo = mid
        else:
            hi = mid
    return np.clip(values - (lo + hi) / 2, 0, 4)


def test_fit_matches_independent_balanced_two_state_objective(training):
    g, r, rows, obs, u = training
    fitted = core.fit(g, r, rows, obs, u)
    active = [0, 1, 3, 4]
    assert fitted["active_indices"] == active
    assert fitted["training_ids"] == [10, 20, 30, 40]
    assert fitted["n_total"] == 7
    assert fitted["n_defined"] == 6
    assert fitted["undefined_pairs"] == [{"shot": 40, "query": 0}]
    weights = np.asarray(fitted["weights"])
    assert weights.shape == (2, 5)
    np.testing.assert_array_equal(weights[:, 2], [0, 0])
    active_weights = weights[:, active].reshape(-1)
    np.testing.assert_allclose(weights[:, active].sum(axis=1), 4, atol=1e-8, rtol=0)
    assert np.all((weights >= 0) & (weights <= 4))

    # Independently expand sum_i p_i ||[(1-u_i)g_i,u_i g_i]w-r_i||²
    # + mean((w-1)²); the discarded undefined row receives no sample mass.
    sample_weights = [1 / 8, 1 / 8, 1 / 4, 1 / 6, 1 / 6, 1 / 6]
    c = len(active)
    a = np.eye(2 * c) / (2 * c)
    b = np.ones(2 * c) / (2 * c)
    objective = float(np.mean((active_weights - 1) ** 2))
    identity = 0.0
    for i, p in enumerate(sample_weights):
        signal = g[i, active] / np.linalg.norm(g[i].sum(axis=0))
        target = r[i] / np.linalg.norm(r[i])
        design = np.concatenate(((1 - u[i]) * signal, u[i] * signal), axis=0)
        a += p * (design @ design.T)
        b += p * (design @ target)
        objective += p * np.sum((active_weights @ design - target) ** 2)
        identity += p * np.sum((signal.sum(axis=0) - target) ** 2)
    assert abs(a[0, c + 1]) > 0.01  # across both state and channel
    np.testing.assert_allclose(fitted["quadratic"], a, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fitted["linear"], b, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fitted["objective_fit"], objective, atol=1e-10, rtol=0)
    np.testing.assert_allclose(fitted["objective_identity"], identity, atol=1e-10, rtol=0)
    assert objective <= identity + 1e-9

    z = active_weights - 2 * (a @ active_weights - b)
    projected = np.concatenate((_project_block(z[:c]), _project_block(z[c:])))
    residual = np.linalg.norm(active_weights - projected)
    assert residual <= 1e-5
    np.testing.assert_allclose(
        fitted["projected_gradient_residual"], residual, atol=1e-10, rtol=0
    )


def test_constant_half_condition_recovers_the_channel_metric(training):
    g, r, rows, obs, _ = training
    constant = metric.fit(g, r, rows, obs)
    fitted = core.fit(g, r, rows, obs, np.full(len(rows), 0.5))
    weights = np.asarray(fitted["weights"])
    assert fitted["active_indices"] == constant["active_indices"]
    np.testing.assert_allclose(weights[0], weights[1], atol=2e-6, rtol=0)
    np.testing.assert_allclose(weights[0], constant["weights"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(fitted["objective_fit"], constant["objective_fit"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        fitted["objective_identity"], constant["objective_identity"], atol=1e-10, rtol=0
    )


def test_effective_weights_are_convex_at_endpoints_and_interior(training):
    g, r, rows, obs, u = training
    fitted = core.fit(g, r, rows, obs, u)
    low, high = np.asarray(fitted["weights"])
    for value in (0.0, 0.5, 1.0, 0.37):
        actual = core.effective_weights(fitted, value)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (g.shape[1],)
        np.testing.assert_allclose(actual, (1 - value) * low + value * high, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual.sum(), len(fitted["active_indices"]), atol=1e-8, rtol=0)
        assert np.all((actual >= 0) & (actual <= 4))


@pytest.mark.parametrize("bad_u", [-1e-12, 1 + 1e-12, np.nan, np.inf, -np.inf])
def test_invalid_condition_is_rejected_even_on_an_undefined_row(training, bad_u):
    g, r, rows, obs, u = training
    fitted = core.fit(g, r, rows, obs, u)
    u[-1] = bad_u
    with pytest.raises(ValueError):
        core.fit(g, r, rows, obs, u)
    with pytest.raises(ValueError):
        core.effective_weights(fitted, bad_u)


def test_observability_is_training_only_and_nonzero_unobserved_gradient_fails(training):
    g, r, rows, obs, u = training
    fitted = core.fit(g, r, rows, obs, u)
    # A hypothetical held-out shot alone observes channel 2. It must not
    # change the fit unless explicitly included in the training arguments.
    held_g = np.array([[[0, 0], [0, 0], [3, 2], [0, 0], [0, 0]]])
    held_r = np.array([[1, 2]])
    held_row = {"shot": 99, "query": 0, "label": 1, "disruptive": True}
    held_obs = np.array([[0, 0, 1, 0, 0]])
    assert 99 not in fitted["training_ids"]
    np.testing.assert_array_equal(np.asarray(fitted["weights"])[:, 2], [0, 0])
    expanded = core.fit(
        np.concatenate((g, held_g)),
        np.concatenate((r, held_r)),
        rows + [held_row],
        np.concatenate((obs, held_obs)),
        np.append(u, 0.7),
    )
    assert expanded["active_indices"] == [0, 1, 2, 3, 4]
    assert 99 in expanded["training_ids"]

    invalid_obs = obs.copy()
    invalid_obs[0, 0] = 0
    with pytest.raises(ValueError):
        core.fit(g, r, rows, invalid_obs, u)


def test_fit_is_invariant_to_independent_positive_per_sample_scales(training):
    g, r, rows, obs, u = training
    original = core.fit(g, r, rows, obs, u)
    g_scale = np.array([0.1, 10, 2, 0.5, 3, 20, 5])
    r_scale = np.array([3, 0.5, 10, 2, 0.1, 4, 7])
    scaled = core.fit(g * g_scale[:, None, None], r * r_scale[:, None], rows, obs, u)
    assert scaled["active_indices"] == original["active_indices"]
    np.testing.assert_allclose(scaled["quadratic"], original["quadratic"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(scaled["linear"], original["linear"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(scaled["weights"], original["weights"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(scaled["objective_fit"], original["objective_fit"], atol=1e-10, rtol=0)
