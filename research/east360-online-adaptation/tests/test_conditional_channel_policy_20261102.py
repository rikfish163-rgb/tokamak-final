"""CPU-only synthetic checks; no claim about neural or alarm performance."""

from copy import deepcopy
import inspect
import json
import math

import numpy as np
import pytest

from experiments.risk_aligned_online.conditional_channel_policy_20261102.core import (
    fit,
    group_changes,
    predict,
)


@pytest.fixture(scope="module")
def separable():
    x = np.eye(6, dtype=np.float64)[[0, 1, 0, 1]]
    contributions = np.zeros((4, 23), dtype=np.float64)
    contributions[[0, 2], :2] = [0.02, -0.02]
    contributions[[1, 3], :2] = [-0.02, 0.02]
    ids = np.array([10, 11, 20, 21])
    groups = np.array(["safe", "safe", "danger30_125", "danger30_125"])
    return x, contributions, ids, groups


@pytest.fixture(scope="module")
def fitted(separable):
    return fit(*separable)


def test_zero_initialization_and_zero_signal_yield_uniform_weights(separable):
    x, contributions, ids, groups = separable
    result = fit(x, np.zeros_like(contributions), ids, groups)
    np.testing.assert_array_equal(predict(x, result), np.ones((len(x), 23)))
    np.testing.assert_array_equal(result["A"], np.zeros((6, 23)))
    np.testing.assert_array_equal(result["b"], np.zeros(23))
    assert len(result["history"]) == result["steps"] == 256
    assert result["history"][0]["group_deltas"] == {
        "safe": 0.0,
        "danger30_125": 0.0,
    }
    assert result["history"][0]["obj"] == 0.0
    assert result["history"][-1]["obj"] == 0.0


def test_bounds_mean_and_explicit_tanh_formula(separable, fitted):
    x, _, _, _ = separable
    extreme = deepcopy(fitted)
    extreme["A"] = np.zeros((6, 23)).tolist()
    extreme["A"][0] = np.linspace(-1000, 1000, 23).tolist()
    extreme["b"] = np.linspace(200, -200, 23).tolist()
    weights = predict(x, extreme)
    u = np.tanh(x @ np.asarray(extreme["A"]) + np.asarray(extreme["b"]))
    np.testing.assert_allclose(weights, 1 + 0.25 * (u - u.mean(1, keepdims=True)))
    assert weights.dtype == np.float64
    assert ((0.5 <= weights) & (weights <= 1.5)).all()
    np.testing.assert_allclose(weights.mean(1), 1, rtol=0, atol=2e-16)


def test_conditional_toy_direction_is_learnable_but_global_baseline_cannot(separable):
    x, contributions, ids, groups = separable
    conditional = fit(x, contributions, ids, groups)
    global_fit = fit(np.empty((len(x), 0)), contributions, ids, groups)
    weights = predict(x, conditional)
    changes = group_changes(contributions, weights, ids, groups)
    assert weights[0, 0] < weights[0, 1]
    assert weights[1, 0] > weights[1, 1]
    assert changes["safe"] < -0.001
    assert changes["danger30_125"] < -0.001
    global_weights = predict(np.empty((len(x), 0)), global_fit)
    np.testing.assert_allclose(global_weights, 1, rtol=0, atol=1e-14)
    assert global_fit["feature_dim"] == 0
    assert global_fit["A"] == []
    assert len(global_fit["b"]) == 23
    assert len(conditional["A"]) * 23 + len(conditional["b"]) == 161
    assert len(global_fit["b"]) == 23  # Capacity differs; not a conditional control.


def test_global_baseline_also_learns_when_one_direction_helps_both_groups(separable):
    x, contributions, ids, groups = separable
    contributions = np.zeros_like(contributions)
    contributions[:, :2] = [0.02, -0.02]
    global_x = np.empty((len(x), 0))
    result = fit(global_x, contributions, ids, groups)
    weights = predict(global_x, result)
    assert result["A"] == []
    assert weights[0, 0] < 1 < weights[0, 1]
    np.testing.assert_array_equal(weights, np.broadcast_to(weights[0], weights.shape))
    assert all(
        change < 0
        for change in group_changes(contributions, weights, ids, groups).values()
    )


def test_json_roundtrip_feature_dim_and_true_shuffled_capacity(separable, fitted):
    x, contributions, ids, groups = separable
    shuffled = fit(x[[1, 0, 3, 2]], contributions, ids, groups)
    for result in (fitted, shuffled):
        encoded = json.dumps(result, allow_nan=False)
        restored = json.loads(encoded)
        assert restored["feature_dim"] == 6
        assert restored["fit_ids"] == [10, 11, 20, 21]
        assert len(restored["A"]) == 6
        assert all(len(row) == 23 for row in restored["A"])
        assert len(restored["b"]) == 23
        assert len(restored["history"]) == 256
        assert isinstance(restored["scope"], str)
        np.testing.assert_array_equal(predict(x, restored), predict(x, result))
    global_fit = fit(np.empty((len(x), 0)), contributions, ids, groups)
    restored_global = json.loads(json.dumps(global_fit, allow_nan=False))
    np.testing.assert_array_equal(
        predict(np.empty((len(x), 0)), global_fit),
        predict(np.empty((len(x), 0)), restored_global),
    )


def test_group_changes_are_shot_macro_averages_and_early_may_share_danger_id():
    contributions = np.zeros((7, 23))
    contributions[:, 0] = [1, 1, 1, 9, -3, -5, 4]
    weights = np.ones((7, 23))
    weights[:, 0] = 1.5
    ids = [1, 1, 1, 2, 3, 3, 3]
    groups = ["safe"] * 4 + ["danger30_125"] * 2 + ["disruptive_early"]
    result = group_changes(contributions, weights, ids, groups)
    assert result == {"safe": 2.5, "danger30_125": -2.0, "disruptive_early": 2.0}
    assert result["safe"] != np.mean([0.5, 0.5, 0.5, 4.5])


def test_fit_uses_shot_macro_not_query_average(separable):
    x, contributions, ids, groups = separable
    # Asymmetric coefficients avoid a persistent exact tie between groups.
    contributions = contributions.copy()
    contributions[1] *= 0.7
    contributions[3] *= 0.6
    result = fit(x, contributions, ids, groups)
    # Replicating all queries of just one shot leaves that shot's mean unchanged.
    ix = [0, 0, 0, 1, 2, 3]
    duplicated = fit(x[ix], contributions[ix], ids[ix], groups[ix])
    np.testing.assert_allclose(duplicated["A"], result["A"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(duplicated["b"], result["b"], rtol=0, atol=1e-12)
    for actual, expected in zip(duplicated["history"], result["history"], strict=True):
        np.testing.assert_allclose(actual["obj"], expected["obj"], rtol=0, atol=1e-12)


def test_fit_receipt_and_forward_do_not_use_heldout_rows_or_ids(separable):
    x, contributions, ids, groups = separable
    heldout_x = np.vstack([x, np.eye(6)[2]])
    heldout_a = np.vstack([contributions, np.full(23, 1e6)])
    heldout_ids = np.append(ids, 9999)
    heldout_groups = np.append(groups, "disruptive_early")
    training_indices = np.arange(4)

    def fit_training_subset():
        return fit(
            heldout_x[training_indices],
            heldout_a[training_indices],
            heldout_ids[training_indices],
            heldout_groups[training_indices],
        )

    original = fit_training_subset()
    heldout_x[-1] = np.eye(6)[5]
    heldout_a[-1] *= -100
    assert fit_training_subset() == original
    assert original["fit_ids"] == [10, 11, 20, 21]
    renamed = fit(x, contributions, np.array([101, 102, 103, 104]), groups)
    np.testing.assert_allclose(renamed["A"], original["A"], atol=1e-14, rtol=0)
    np.testing.assert_allclose(renamed["b"], original["b"], atol=1e-14, rtol=0)
    assert list(inspect.signature(predict).parameters) == ["x", "fitted"]
    np.testing.assert_allclose(predict(x, renamed), predict(x, original), atol=1e-14)


def test_fit_and_predict_do_not_mutate_inputs_or_numpy_rng_state(separable):
    before = tuple(value.copy() for value in separable)
    state = np.random.get_state()
    result = fit(*separable)
    saved = deepcopy(result)
    predict(separable[0], result)
    current = np.random.get_state()
    for actual, expected in zip(separable, before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert result == saved
    assert state[0] == current[0] and state[2:] == current[2:]
    np.testing.assert_array_equal(state[1], current[1])


def test_adam_and_all_history_match_independent_scalar_solution(fitted):
    # For the symmetric toy there are only four nonzero, equal-magnitude
    # parameters. Its exact group deltas and gradient reduce to one scalar.
    magnitude = first = second = 0.0
    for step in range(1, 257):
        tanh = math.tanh(magnitude)
        delta = -0.01 * tanh
        objective = -tanh + 0.04 * magnitude**2 / 161
        record = fitted["history"][step - 1]
        assert record["step"] == step
        np.testing.assert_allclose(record["group_deltas"]["safe"], delta, atol=1e-14)
        np.testing.assert_allclose(
            record["group_deltas"]["danger30_125"], delta, atol=1e-14
        )
        np.testing.assert_allclose(record["obj"], objective, atol=1e-13)
        gradient = 0.25 * (1 - tanh**2) - 0.02 * magnitude / 161
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        magnitude += (
            0.01
            * (first / (1 - 0.9**step))
            / (math.sqrt(second / (1 - 0.999**step)) + 1e-8)
        )
    expected = np.zeros((6, 23))
    expected[0, :2] = [-magnitude, magnitude]
    expected[1, :2] = [magnitude, -magnitude]
    np.testing.assert_allclose(fitted["A"], expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(fitted["b"], 0, rtol=0, atol=1e-13)


def test_first_step_matches_finite_difference_with_unequal_shot_sizes():
    x = np.eye(6)[[0, 1, 0, 1, 0, 1, 0]]
    ids = np.array([1, 1, 1, 2, 3, 3, 4])
    groups = np.array(["safe"] * 4 + ["danger30_125"] * 3)
    contributions = np.sin(np.arange(7 * 23).reshape(7, 23)) * 0.02
    result = fit(x, contributions, ids, groups)

    def evaluate(parameters):
        u = np.tanh(x @ parameters[:6] + parameters[6])
        weights = 1 + 0.25 * (u - u.mean(axis=1, keepdims=True))
        rows = ((weights - 1) * contributions).sum(axis=1)
        deltas = {
            group: np.mean(
                [
                    rows[(groups == group) & (ids == shot)].mean()
                    for shot in set(ids[groups == group])
                ]
            )
            for group in ("safe", "danger30_125")
        }
        return deltas, max(deltas.values()) / 0.01 + 0.01 * np.mean(parameters**2)

    parameters = np.zeros((7, 23))
    gradient = np.zeros_like(parameters)
    epsilon = 1e-6
    for position in np.ndindex(parameters.shape):
        parameters[position] = epsilon
        plus = evaluate(parameters)[1]
        parameters[position] = -epsilon
        minus = evaluate(parameters)[1]
        parameters[position] = 0
        gradient[position] = (plus - minus) / (2 * epsilon)
    parameters -= 0.01 * gradient / (np.abs(gradient) + 1e-8)
    deltas, objective = evaluate(parameters)
    for group, value in deltas.items():
        np.testing.assert_allclose(
            result["history"][1]["group_deltas"][group], value, atol=1e-9
        )
    np.testing.assert_allclose(result["history"][1]["obj"], objective, atol=1e-7)


@pytest.mark.parametrize("steps", [0, 1, 255, 257, -1, True, 256.0, "256"])
def test_only_fixed_256_training_steps_are_accepted(separable, steps):
    with pytest.raises(ValueError):
        fit(*separable, steps=steps)


@pytest.mark.parametrize(
    "bad_x",
    [
        np.ones((4, 5)),
        np.ones((4, 7)),
        np.ones(6),
        np.full((4, 6), np.nan),
        np.full((4, 6), np.inf),
        np.tile([-0.1, 1.1, 0, 0, 0, 0], (4, 1)),
        np.tile([0.5, 0.4, 0, 0, 0, 0], (4, 1)),
    ],
)
def test_bad_probability_shape_or_values_rejected(separable, fitted, bad_x):
    _, contributions, ids, groups = separable
    with pytest.raises(ValueError):
        fit(bad_x, contributions, ids, groups)
    with pytest.raises(ValueError):
        predict(bad_x, fitted)


@pytest.mark.parametrize(
    "bad_contributions",
    [np.zeros((4, 22)), np.zeros((3, 23)), np.full((4, 23), np.nan)],
)
def test_bad_contribution_shape_or_values_rejected(separable, bad_contributions):
    x, _, ids, groups = separable
    with pytest.raises(ValueError):
        fit(x, bad_contributions, ids, groups)
    with pytest.raises(ValueError):
        group_changes(bad_contributions, np.ones((4, 23)), ids, groups)


@pytest.mark.parametrize(
    ("ids", "groups"),
    [
        ([1, 2, 3], ["safe", "safe", "danger30_125", "danger30_125"]),
        ([1.0, 2.0, 3.0, 4.0], ["safe", "safe", "danger30_125", "danger30_125"]),
        ([True, False, True, False], ["safe", "safe", "danger30_125", "danger30_125"]),
        ([1, 2, 3, 4], ["safe", "safe", "danger30_125"]),
        ([1, 2, 3, 4], ["safe", "safe", "danger30_125", "other"]),
        ([1, 2, 3, 4], ["safe", "safe", "safe", "safe"]),
        ([1, 2, 3, 4], ["danger30_125"] * 4),
        ([1, 2, 3, 4], ["safe", "safe", "danger30_125", "disruptive_early"]),
        ([1, 2, 1, 4], ["safe", "safe", "danger30_125", "danger30_125"]),
    ],
)
def test_fit_rejects_bad_ids_groups_missing_groups_early_or_mixed_shot(
    separable, ids, groups
):
    x, contributions, _, _ = separable
    with pytest.raises(ValueError):
        fit(x, contributions, ids, groups)


@pytest.mark.parametrize("other_group", ["danger30_125", "disruptive_early"])
def test_group_changes_rejects_safe_and_disruptive_same_shot(other_group):
    with pytest.raises(ValueError):
        group_changes(
            np.zeros((2, 23)), np.ones((2, 23)), [7, 7], ["safe", other_group]
        )


@pytest.mark.parametrize("weights", [np.ones((4, 22)), np.full((4, 23), np.nan)])
def test_group_changes_rejects_bad_weight_shape_or_values(separable, weights):
    _, contributions, ids, groups = separable
    with pytest.raises(ValueError):
        group_changes(contributions, weights, ids, groups)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "unknown"),
        ("feature_dim", 5),
        ("feature_dim", True),
        ("A", [[0.0] * 22] * 6),
        ("A", [[float("nan")] * 23] * 6),
        ("b", [0.0] * 22),
        ("b", [float("inf")] * 23),
    ],
)
def test_predict_rejects_invalid_fitted_model_schema(separable, fitted, key, value):
    malformed = deepcopy(fitted)
    malformed[key] = value
    with pytest.raises(ValueError):
        predict(separable[0], malformed)


def test_predict_rejects_missing_schema_and_feature_dimension_mismatch(
    separable, fitted
):
    malformed = deepcopy(fitted)
    del malformed["schema"]
    with pytest.raises(ValueError):
        predict(separable[0], malformed)
    with pytest.raises(ValueError):
        predict(np.empty((4, 0)), fitted)
