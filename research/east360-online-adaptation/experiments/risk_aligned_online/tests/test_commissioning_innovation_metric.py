"""Synthetic contracts for delayed, masked innovation updates."""

import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_innovation_metric_20261111 as metric
from experiments.risk_aligned_online import commissioning_weak_shape_data_20261111 as weak
from experiments.risk_aligned_online.commissioning_transition_net_20261110 import FIRST, STRIDE


def test_constants_match_the_requested_protocol():
    assert metric.HORIZON == 25
    assert (metric.RETAIN, metric.STEP, metric.STABILIZER) == (0.85, 0.225, 0.02)


def test_observed_mask_requires_three_finite_samples_in_every_future_bin():
    q = 12
    z = np.ones((80, 10), dtype=np.float64)
    mask = np.ones_like(z, dtype=bool)
    for b in range(5):
        mask[q + 1 + 5*b:q + 3 + 5*b, 0] = False  # Exactly three in each bin.
    mask[q + 6:q + 9, 1] = False  # Two in one bin invalidates the channel.
    z[q + 11:q + 13, 2] = [np.nan, np.inf]
    mask[q + 13, 2] = False
    mask[q, 3] = mask[q + 26, 3] = False  # Outside the horizon.
    z[q, 4] = z[q + 26, 4] = np.nan

    actual = metric.observed_mask(z, mask, [q])
    expected = np.ones((5, 10), dtype=bool)
    expected[:, 1:3] = False
    assert actual.shape == (1, 50)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual[0].reshape(5, 10), expected)

    # Both endpoints belong to the horizon, not the adjacent samples.
    for boundary, missing in ((q + 1, [q + 2, q + 3]),
                              (q + 25, [q + 22, q + 23])):
        edge_mask = np.ones_like(mask)
        edge_mask[missing, 5] = False
        assert metric.observed_mask(z, edge_mask, [q])[0].reshape(5, 10)[:, 5].all()
        edge_mask[boundary, 5] = False
        assert not metric.observed_mask(z, edge_mask, [q])[0].reshape(5, 10)[:, 5].any()


def test_observed_mask_empty_range_and_input_validation():
    z = np.ones((40, 10))
    mask = np.ones_like(z, dtype=bool)
    empty = metric.observed_mask(z, mask, [])
    assert empty.shape == (0, 50) and empty.dtype == np.bool_
    assert metric.observed_mask(z, mask, [0, len(z) - metric.HORIZON - 1]).all()
    for origin in (-1, len(z) - metric.HORIZON):
        with pytest.raises(ValueError):
            metric.observed_mask(z, mask, [origin])
    with pytest.raises(TypeError):
        metric.observed_mask(z, mask, [1.5])
    with pytest.raises(ValueError):
        metric.observed_mask(z[:, :9], mask[:, :9], [0])
    with pytest.raises(ValueError):
        metric.observed_mask(z, mask.astype(int), [0])


def test_anchor_steps_match_finite_difference_of_masked_half_squared_loss():
    rng = np.random.default_rng(543)
    m, h, rank, edges = 3, 4, 2, 7
    pre = rng.normal(size=(m, h))
    key = rng.normal(size=(m, rank))
    decoder = rng.normal(size=(edges, h))
    bias = rng.normal(size=edges)
    target = rng.normal(size=(m, edges))
    observed = rng.random((m, edges)) > 0.3
    actual = metric.anchor_steps(pre, key, target, observed, decoder, bias)
    assert actual.shape == (m, h, rank) and actual.dtype == np.float32

    eps = 1e-6
    expected = np.empty_like(actual, dtype=np.float64)
    for i in range(m):
        def loss(state):
            prediction = np.tanh(pre[i] + state @ key[i]) @ decoder.T + bias
            residual = np.where(observed[i], prediction - target[i], 0.0)
            return 0.5 * np.sum(residual**2)

        gradient = np.empty((h, rank))
        for a in range(h):
            for b in range(rank):
                perturbation = np.zeros((h, rank))
                perturbation[a, b] = eps
                gradient[a, b] = (loss(perturbation) - loss(-perturbation)) / (2*eps)
        expected[i] = -metric.STEP * gradient / (np.linalg.norm(gradient) + metric.STABILIZER)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)


def test_anchor_steps_ignore_unobserved_targets_and_allow_exact_zero():
    pre = np.array([[0.2, -0.4], [0.1, 0.3], [-0.2, 0.5]])
    key = np.array([[0.7, -0.2], [0.0, 0.0], [0.8, 0.4]])
    decoder = np.array([[1.0, 2.0], [-0.5, 0.4], [0.2, -1.0]])
    bias = np.array([0.3, -0.7, 1.2])
    target = np.array([[0.5, 0.1, -0.2], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]])
    observed = np.array([[True, False, True], [True, True, True], [False, False, False]])
    baseline = metric.anchor_steps(pre, key, target, observed, decoder, bias)
    altered = target.copy()
    altered[~observed] = [np.nan, np.inf, -np.inf, 1e20]
    np.testing.assert_array_equal(
        metric.anchor_steps(pre, key, altered, observed, decoder, bias), baseline
    )
    assert np.isfinite(baseline).all()
    assert np.any(baseline[0] != 0)
    np.testing.assert_array_equal(baseline[1:], 0)  # Zero key and no observed target.

    exact_target = np.tanh(pre) @ decoder.T + bias
    zero = metric.anchor_steps(pre, key, exact_target, np.ones_like(observed), decoder, bias)
    np.testing.assert_array_equal(zero, 0)
    # A bias residual alone cannot update a decoder-independent fast state.
    zero = metric.anchor_steps(pre, key, target, observed, np.zeros_like(decoder), bias)
    np.testing.assert_array_equal(zero, 0)


def test_release_starts_only_at_the_first_index_after_the_horizon():
    q = FIRST
    first = q + metric.HORIZON + 1
    key = np.tile(np.array([[0.5, -1.0]], dtype=np.float32), (first + 1, 1))
    steps = np.array([[[1.0, 2.0], [-1.0, 0.5]]], dtype=np.float32)
    before, count = metric.released_displacement(key[:first], [q], steps, [True])
    assert before.shape == (first, 2) and before.dtype == np.float32
    assert count == 0 and not before.any()
    delta, count = metric.released_displacement(key, [q], steps, [True])
    assert count == 1 and isinstance(count, int)
    np.testing.assert_array_equal(delta[:first], 0)
    np.testing.assert_allclose(delta[first], steps[0] @ key[first])


def test_release_retains_state_through_invalid_blocks_and_resets_per_shot():
    origins = FIRST + STRIDE * np.arange(5)
    releases = origins + metric.HORIZON + 1
    length = int(releases[3] + 4)
    t = np.arange(length, dtype=np.float32)
    key = np.column_stack((1 + t / 100, -0.5 + t / 200)).astype(np.float32)
    steps = np.array([
        [[1, 2], [3, 4]],
        [[999, 999], [999, 999]],  # Ineligible: neither add nor decay.
        [[-1, 0.5], [0.2, -0.3]],
        [[0, 0], [0, 0]],          # Eligible zero step still retains once.
        [[999, 999], [999, 999]],  # Release would be beyond the record.
    ], dtype=np.float32)
    eligible = np.array([True, False, True, True, True])
    delta, count = metric.released_displacement(key, origins, steps, eligible)
    assert count == 3
    np.testing.assert_array_equal(delta[:releases[0]], 0)
    first_state = steps[0]
    second_state = metric.RETAIN * first_state + steps[2]
    third_state = metric.RETAIN * second_state
    for start, stop, state in ((releases[0], releases[2], first_state),
                               (releases[2], releases[3], second_state),
                               (releases[3], length, third_state)):
        np.testing.assert_allclose(delta[start:stop], key[start:stop] @ state.T, rtol=1e-6)

    # A separate shot never inherits state from the preceding call.
    empty, n_empty = metric.released_displacement(key, [], np.empty((0, 2, 2)), [])
    invalid, n_invalid = metric.released_displacement(key, origins, steps, np.zeros(5, dtype=bool))
    assert n_empty == n_invalid == 0
    np.testing.assert_array_equal(empty, 0)
    np.testing.assert_array_equal(invalid, 0)
    repeat, n_repeat = metric.released_displacement(key, origins, steps, eligible)
    assert n_repeat == count
    np.testing.assert_array_equal(repeat, delta)


def test_releases_follow_time_order_even_when_origins_are_not_sorted():
    q = FIRST
    origins = np.array([q + STRIDE, q])
    steps = np.array([[[3.0]], [[2.0]]], dtype=np.float32)
    last = int(origins.max() + metric.HORIZON + 1)
    delta, count = metric.released_displacement(np.ones((last + 2, 1)), origins, steps, [True, True])
    assert count == 2
    np.testing.assert_allclose(delta[q + metric.HORIZON + 1:last], 2.0)
    np.testing.assert_allclose(delta[last:], metric.RETAIN * 2.0 + 3.0)


def test_truncated_prefix_matches_full_run_with_actual_future_shape_feedback():
    rng = np.random.default_rng(61)
    length = FIRST + 3*STRIDE + metric.HORIZON + 20
    z = rng.normal(size=(length, 10))
    mask = np.ones_like(z, dtype=bool)
    pre = rng.normal(size=(length, 3))
    key = rng.normal(size=(length, 2))
    decoder = rng.normal(size=(50, 3))
    bias = rng.normal(size=50)
    origins = np.arange(FIRST, length - metric.HORIZON, STRIDE)

    def evaluate(stop):
        available = origins[origins + metric.HORIZON < stop]
        target, _, eligible = weak.feedback(z[:stop], mask[:stop], available)
        observed = metric.observed_mask(z[:stop], mask[:stop], available)
        steps = metric.anchor_steps(pre[available], key[available], target, observed, decoder, bias)
        return metric.released_displacement(key[:stop], available, steps, eligible)

    complete, count = evaluate(length)
    assert count > 1 and np.any(complete)
    first = FIRST + metric.HORIZON + 1
    for stop in (first - 1, first, first + 1,
                 first + STRIDE, first + STRIDE + 1,
                 first + 2*STRIDE + 1, length):
        prefix, _ = evaluate(stop)
        np.testing.assert_allclose(prefix, complete[:stop], rtol=1e-6, atol=1e-7)
