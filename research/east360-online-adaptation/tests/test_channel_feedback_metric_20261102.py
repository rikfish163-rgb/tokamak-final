"""Synthetic CPU checks of the channel-weight quadratic program."""

import numpy as np
import pytest

from experiments.risk_aligned_online.channel_feedback_20261102.metric import fit, solve
from experiments.risk_aligned_online.history_gradient_metric_20261102.core import (
    solve as diagonal_solve,
)


def _check_qp(a, b, weights, receipt):
    w = np.asarray(weights, dtype=np.float64)
    assert w.shape == b.shape
    assert np.isfinite(w).all()
    assert np.all((w >= -1e-10) & (w <= 4.0 + 1e-10))
    np.testing.assert_allclose(w.sum(), len(w), rtol=0, atol=1e-8)
    projection, _ = diagonal_solve(np.ones_like(w), w - 2 * (a @ w - b))
    assert np.linalg.norm(w - projection) <= 1e-5
    assert receipt["success"]
    assert 0 <= receipt["projected_gradient_residual"] <= 1e-5
    assert receipt["iterations"] >= 0

    def objective(x):
        return x @ a @ x - 2 * b @ x

    assert objective(w) <= objective(np.ones_like(w)) + 1e-9

    interior = (w > 1e-7) & (w < 4.0 - 1e-7)
    if interior.any():
        mu = np.mean((b - a @ w)[interior])
        gradient = a @ w - b + mu
        np.testing.assert_allclose(gradient[interior], 0, rtol=0, atol=1e-5)
        assert np.all(gradient[w <= 1e-7] >= -1e-5)
        assert np.all(gradient[w >= 4.0 - 1e-7] <= 1e-5)


@pytest.mark.parametrize(
    ("dtype", "scale"), [(np.float64, 1.0), (np.float32, 0.1), (np.float32, 10.0)]
)
def test_cross_terms_change_the_hand_calculated_solution(dtype, scale):
    a = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=dtype) * scale
    b = np.array([0.0, 3.0], dtype=dtype) * scale
    w, receipt = solve(a, b)
    np.testing.assert_allclose(w, [1.0 / 3, 5.0 / 3], rtol=0, atol=2e-6)
    diagonal, _ = diagonal_solve(np.diag(a), b)
    assert np.linalg.norm(np.asarray(w) - diagonal) > 0.3
    _check_qp(a, b, w, receipt)


def test_qp_with_active_upper_and_lower_bounds():
    a = np.diag([1.0, 2.0, 1.0, 2.0, 1.0]) + 0.1 * np.ones((5, 5))
    expected = np.array([4.0, 0.0, 0.25, 0.5, 0.25])
    b = a @ expected + 0.3
    b[0] += 0.5
    b[1] -= 0.5
    w, receipt = solve(a, b)
    np.testing.assert_allclose(w, expected, rtol=0, atol=2e-6)
    _check_qp(a, b, w, receipt)


def test_single_channel_qp_is_forced_to_one():
    a = np.array([[7.0]])
    b = np.array([-3.0])
    w, receipt = solve(a, b)
    np.testing.assert_allclose(w, [1.0], rtol=0, atol=1e-10)
    _check_qp(a, b, w, receipt)


@pytest.fixture
def synthetic_training():
    # Six points, deliberately uneven by label and shot. Channels 2 and 4
    # have zero gradients, but only channel 4 has training observations.
    g = np.array(
        [
            [[2, 0], [1, 1], [0, 0], [-0.5, 1], [0, 0]],
            [[0, 2], [1, -0.5], [0, 0], [1, 0], [0, 0]],
            [[1, 1], [0.5, 1], [0, 0], [2, -1], [0, 0]],
            [[-1, 2], [1, 2], [0, 0], [0.25, 0.5], [0, 0]],
            [[1, 2], [2, -0.5], [0, 0], [1, 1], [0, 0]],
            [[0.5, -1], [1, 1], [0, 0], [1, 0.5], [0, 0]],
        ],
        dtype=np.float64,
    )
    r = np.array([[0, 2], [1, 0], [1, 2], [-1, 1], [0.3, -2], [2, 1]])
    rows = [
        {"shot": shot, "query": query, "label": label, "disruptive": disruptive}
        for shot, query, label, disruptive in [
            (10, 0, 0, False),
            (10, 1, 0, False),
            (20, 0, 0, True),
            (30, 0, 1, True),
            (30, 1, 1, True),
            (30, 2, 1, True),
        ]
    ]
    observations = np.tile([1, 1, 0, 1, 1], (len(rows), 1))
    return g, r, rows, observations


def test_fit_normalization_cross_terms_balancing_and_observability(synthetic_training):
    g, r, rows, observations = synthetic_training
    result = fit(g, r, rows, observations)
    active = np.array([0, 1, 3, 4])
    weights = np.asarray(result["weights"])
    np.testing.assert_array_equal(result["active_indices"], active)
    assert weights.shape == (5,)
    assert weights.dtype == np.float64
    assert weights[2] == 0
    assert result["training_ids"] == [10, 20, 30]

    # Independent expansion of the stated normalized, label/shot-balanced loss.
    g_normalized = g[:, active] / np.linalg.norm(g.sum(axis=1), axis=1)[:, None, None]
    r_normalized = r / np.linalg.norm(r, axis=1)[:, None]
    sample_weights = np.array([1 / 8, 1 / 8, 1 / 4, 1 / 6, 1 / 6, 1 / 6])
    a = np.einsum("n,ncd,ned->ce", sample_weights, g_normalized, g_normalized)
    a += np.eye(len(active)) / len(active)
    b = np.einsum("n,ncd,nd->c", sample_weights, g_normalized, r_normalized)
    b += 1 / len(active)
    assert abs(a[0, 1]) > 0.01
    np.testing.assert_allclose(result["quadratic"], a, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["linear"], b, rtol=0, atol=1e-12)
    active_weights = weights[active]
    projected, _ = diagonal_solve(
        np.ones_like(active_weights), active_weights - 2 * (a @ active_weights - b)
    )
    assert np.linalg.norm(active_weights - projected) <= 1e-5
    assert result["projected_gradient_residual"] <= 1e-5
    np.testing.assert_allclose(active_weights.sum(), len(active), rtol=0, atol=1e-8)
    assert np.all((active_weights >= 0) & (active_weights <= 4))

    def objective(w):
        difference = np.einsum("c,ncd->nd", w, g_normalized) - r_normalized
        return sample_weights @ np.sum(difference**2, axis=1) + np.mean((w - 1) ** 2)

    actual = objective(active_weights)
    identity = objective(np.ones(len(active)))
    np.testing.assert_allclose(result["objective_fit"], actual, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        result["objective_identity"], identity, rtol=0, atol=1e-10
    )
    assert actual <= identity + 1e-10


def test_fit_one_observed_channel_has_identity_objective(synthetic_training):
    g, r, rows, observations = synthetic_training
    result = fit(g[:, :1], r, rows, observations[:, :1])
    np.testing.assert_allclose(result["weights"], [1.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        result["objective_fit"], result["objective_identity"], rtol=0, atol=1e-10
    )


def test_fit_positive_sample_scales_and_float32_inputs(synthetic_training):
    g, r, rows, observations = synthetic_training
    reference = fit(g, r, rows, observations)
    g_scale = np.array([0.1, 10.0, 2.0, 0.5, 3.0, 20.0])
    r_scale = np.array([3.0, 0.5, 10.0, 2.0, 0.1, 4.0])
    scaled = fit(
        (g * g_scale[:, None, None]).astype(np.float32),
        (r * r_scale[:, None]).astype(np.float32),
        rows,
        observations.astype(np.float32),
    )
    np.testing.assert_array_equal(scaled["active_indices"], reference["active_indices"])
    np.testing.assert_allclose(
        scaled["weights"], reference["weights"], rtol=0, atol=2e-6
    )
    np.testing.assert_allclose(
        scaled["objective_fit"], reference["objective_fit"], rtol=0, atol=2e-6
    )
