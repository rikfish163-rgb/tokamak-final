"""CPU-only regression tests for the offline output-trust policy."""

import io

import numpy as np
import pytest

from output_trust_policy_20261106 import choose, fit_heads, predictions


def sample_data(actions=1):
    """Six distinct shots in each utility and guard role."""
    n = 12
    roles = np.repeat([0, 1], 6)
    return {
        "x": np.zeros((n, 1)),
        "shot": np.arange(n),
        "utility_role": roles.copy(),
        "guard_role": roles.copy(),
        "raw": np.full((n, actions), 0.2),
        "joint": np.full((n, actions), 0.3),
        "harm": np.full((n, actions), 0.05),
        "available": np.ones((n, actions), dtype=bool),
    }


def append_rows(data, indices):
    return {
        key: np.concatenate((value, value[indices]), axis=0)
        for key, value in data.items()
    }


def constant_heads(raw, joint, safe, first):
    """Create eligible heads with exactly controlled constant predictions."""
    heads = fit_heads(**sample_data(len(raw)))
    heads["coef"][:] = 0.0
    heads["intercept"][:] = np.asarray([raw, joint, safe, first])
    return heads


def test_predictions_are_deterministic_serializable_and_outcome_free():
    data = sample_data(2)
    data["raw"][:, 0] = np.linspace(-2, 2, 12)
    data["joint"][:, 1] = np.linspace(0, 3, 12)
    data["x"][:, 0] = np.arange(12) / 10
    first = fit_heads(**data)
    second = fit_heads(**data)
    assert set(first) == {
        "head_names",
        "feature_mean",
        "feature_scale",
        "coef",
        "intercept",
        "shot_counts",
        "eligible",
        "ridge",
    }
    assert all(isinstance(value, np.ndarray) for value in first.values())
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    buffer = io.BytesIO()
    np.savez(buffer, **first)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        restored = {key: archive[key] for key in archive.files}
    for actual, expected in zip(
        predictions(restored, data["x"]), predictions(first, data["x"]), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        choose(restored, data["x"], data["available"]),
        choose(first, data["x"], data["available"]),
    )


def test_unknown_roles_missing_targets_and_unavailable_targets_are_ignored():
    base = sample_data(2)
    expected = fit_heads(**base)
    augmented = append_rows(base, [0, 0, 0])
    augmented["shot"][-3:] = [100, 101, 102]
    augmented["x"][-3:, 0] = [100, 200, 300]
    augmented["utility_role"][-3] = -1
    augmented["guard_role"][-3] = -1
    augmented["available"][-2] = [False, True]
    for name in ("raw", "joint", "harm"):
        augmented[name][-3] = [np.inf, -np.inf]  # unknown role
        augmented[name][-2] = [np.inf, np.nan]  # unavailable / unknown target
        augmented[name][-1] = np.nan  # known roles, unknown targets
    actual = fit_heads(**augmented)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_roles_shots_and_episodes_receive_equal_mass():
    data = sample_data()
    data["raw"][:6, 0] = [0, 1, 1, 1, 1, 1]
    data["raw"][6:, 0] = 0
    data["harm"][:6, 0] = [0, 0.6, 0.6, 0.6, 0.6, 0.6]
    data["harm"][6:, 0] = 0.08
    once = fit_heads(**data)
    duplicated = fit_heads(**append_rows(data, [0] * 25))
    for actual, expected in zip(
        predictions(duplicated, [[0.0]]), predictions(once, [[0.0]]), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    raw, _, safe, first = predictions(duplicated, [[0.0]])
    np.testing.assert_allclose(raw, [[5 / 12]])
    np.testing.assert_allclose(safe, [[0.5]])
    np.testing.assert_allclose(first, [[0.08]])


def test_episodes_within_a_shot_are_averaged_before_shots():
    data = sample_data()
    data["raw"][:6, 0] = [0, 1, 1, 1, 1, 1]
    data["raw"][6:, 0] = 0
    data = append_rows(data, [0])
    data["raw"][-1, 0] = 1
    heads = fit_heads(**data)
    np.testing.assert_allclose(predictions(heads, [[0]])[0], [[11 / 24]])


def test_ridge_uses_distinct_shot_scale_and_equal_utility_roles():
    data = append_rows(sample_data(), list(range(6, 12)))
    data["shot"][12:] = np.arange(12, 18)
    data["x"][:, 0] = data["utility_role"]
    data["raw"][:, 0] = 2 * data["utility_role"]
    data["joint"][:, 0] = data["raw"][:, 0]
    heads = fit_heads(**data, ridge=4)
    raw, joint, _, _ = predictions(heads, [[0], [1]])
    slope = 18 / 22
    np.testing.assert_allclose(raw[:, 0], [1 - slope, 1 + slope])
    np.testing.assert_allclose(joint, raw)
    np.testing.assert_array_equal(heads["shot_counts"][0, 0], [6, 12])
    np.testing.assert_allclose(heads["feature_mean"][0, 0], [0.5])
    np.testing.assert_allclose(heads["feature_scale"][0, 0], [0.5])


def test_ridge_counts_a_shot_only_once_if_it_occurs_in_both_roles():
    data = sample_data()
    data["shot"][6:] = data["shot"][:6]
    data["x"][:, 0] = data["utility_role"]
    data["raw"][:, 0] = 2 * data["utility_role"]
    heads = fit_heads(**data, ridge=4)
    raw, _, _, _ = predictions(heads, [[0], [1]])
    np.testing.assert_allclose(raw[:, 0], [0.4, 1.6])
    assert heads["eligible"][0]


def test_missing_targets_are_masked_independently_by_head():
    data = append_rows(sample_data(), [0, 6])
    data["shot"][-2:] = [12, 13]
    data["raw"][0, 0] = np.nan
    data["joint"][12, 0] = np.nan
    data["harm"][6, 0] = np.nan
    heads = fit_heads(**data)
    np.testing.assert_array_equal(
        heads["shot_counts"][:, 0], [[6, 7], [6, 7], [7, 0], [0, 6]]
    )
    assert heads["eligible"][0]
    assert all(np.isfinite(value[0, 0]) for value in predictions(heads, [[0]]))


def test_guard_heads_use_independent_shot_scaled_ridge_fits():
    data = sample_data()
    data["x"][:, 0] = [0, 0, 0, 1, 1, 1] * 2
    data["harm"][:6, 0] = [0, 0, 0, 1, 1, 1]
    data["harm"][6:, 0] = [1, 1, 1, 0, 0, 0]
    heads = fit_heads(**data, ridge=4)
    _, _, safe, first = predictions(heads, [[0], [1]])
    np.testing.assert_allclose(safe[:, 0], [0.2, 0.8])
    np.testing.assert_allclose(first[:, 0], [0.8, 0.2])


@pytest.mark.parametrize(
    ("target", "row", "head_index"),
    [("raw", 0, 0), ("joint", 0, 1), ("harm", 0, 2), ("harm", 6, 3)],
)
def test_each_head_needs_six_distinct_shots_even_with_guards_off(
    target, row, head_index
):
    data = append_rows(sample_data(), [1] * 20)
    data[target][row, 0] = np.nan
    heads = fit_heads(**data)
    assert not heads["eligible"][0]
    assert np.isnan(predictions(heads, [[0]])[head_index][0, 0])
    np.testing.assert_array_equal(choose(heads, [[0]], [[True]]), [-1])
    np.testing.assert_array_equal(choose(heads, [[0]], [[True]], guard=False), [-1])


def test_availability_and_training_counts_are_per_action():
    data = sample_data(2)
    data["available"][0, 1] = False
    data["raw"][0, 1] = np.inf
    data["joint"][0, 1] = np.inf
    data["harm"][0, 1] = np.inf
    heads = fit_heads(**data)
    np.testing.assert_array_equal(heads["eligible"], [True, False])
    np.testing.assert_array_equal(heads["shot_counts"][0, :, 0], [6, 5])
    np.testing.assert_array_equal(choose(heads, [[0]], [[False, True]]), [-1])
    np.testing.assert_array_equal(choose(heads, [[0]], [[True, True]]), [0])


def test_objectives_guard_veto_thresholds_and_unavailability():
    heads = constant_heads(
        raw=[0.9, 0.8, 0.02, 0.01],
        joint=[0.3, 0.5, 0.4, 0.01],
        safe=[0.11, 0.0, 0.1, 0.0],
        first=[0.0, 0.11, 0.1, 0.0],
    )
    x = np.zeros((4, 1))
    available = np.array(
        [
            [True, True, True, True],
            [False, False, False, True],
            [False] * 4,
            [False, False, True, False],
        ]
    )
    np.testing.assert_array_equal(
        choose(heads, x, available, objective="raw"), [-1, -1, -1, 2]
    )
    np.testing.assert_array_equal(choose(heads, x, available), [-1, -1, -1, 2])
    np.testing.assert_array_equal(
        choose(heads, x, available, objective="raw", guard=False), [0, -1, -1, 2]
    )
    np.testing.assert_array_equal(
        choose(heads, x, available, guard=False), [1, -1, -1, 2]
    )


def test_fitted_constant_targets_respect_exact_selection_boundaries():
    data = sample_data()
    data["raw"][:] = 0.01
    data["joint"][:] = 0.01
    data["harm"][:] = 0.1
    heads = fit_heads(**data)
    assert predictions(heads, [[0]])[0][0, 0] == 0.01
    np.testing.assert_array_equal(choose(heads, [[0]], [[True]], guard=False), [-1])
    data["raw"][:] = 0.02
    data["joint"][:] = 0.02
    heads = fit_heads(**data)
    assert predictions(heads, [[0]])[2][0, 0] == 0.1
    assert predictions(heads, [[0]])[3][0, 0] == 0.1
    np.testing.assert_array_equal(choose(heads, [[0]], [[True]]), [0])


def test_end_to_end_fitted_objectives_and_independent_guards():
    data = sample_data(3)
    data["raw"][:] = [0.9, 0.8, 0.2]
    data["joint"][:] = [0.2, 0.9, 0.3]
    data["harm"][:6] = [0.11, 0.02, 0.05]
    data["harm"][6:] = [0.02, 0.11, 0.05]
    heads = fit_heads(**data)
    available = [[True, True, True]]
    np.testing.assert_array_equal(
        choose(heads, [[0]], available, objective="raw"), [-1]
    )
    np.testing.assert_array_equal(
        choose(heads, [[0]], available, objective="joint"), [-1]
    )
    np.testing.assert_array_equal(
        choose(heads, [[0]], available, objective="raw", guard=False), [0]
    )
    np.testing.assert_array_equal(
        choose(heads, [[0]], available, objective="joint", guard=False), [1]
    )


def test_ties_choose_first_action_and_empty_batches_are_supported():
    heads = constant_heads([0.2, 0.2], [0.4, 0.4], [0.0] * 2, [0.0] * 2)
    np.testing.assert_array_equal(choose(heads, [[0]], [[True, True]]), [0])
    assert choose(heads, np.empty((0, 1)), np.empty((0, 2), dtype=bool)).shape == (0,)
    assert all(
        values.shape == (0, 2) for values in predictions(heads, np.empty((0, 1)))
    )


def test_veto_of_first_tied_action_does_not_try_another_action():
    heads = constant_heads([0.2, 0.2], [0.4, 0.4], [0.11, 0], [0, 0])
    np.testing.assert_array_equal(choose(heads, [[0]], [[True, True]]), [-1])
    np.testing.assert_array_equal(
        choose(heads, [[0]], [[True, True]], guard=False), [0]
    )


def test_std_floor_clipping_and_only_harm_predictions_are_clamped():
    heads = fit_heads(**sample_data())
    np.testing.assert_array_equal(heads["feature_scale"], np.full((4, 1, 1), 0.05))
    heads["coef"][:, 0, 0] = 1
    heads["intercept"][:, 0] = 0
    raw, joint, safe, first = predictions(heads, [[-1], [-0.2], [0], [0.2], [1]])
    expected = [-4, -4, 0, 4, 4]
    np.testing.assert_allclose(raw[:, 0], expected)
    np.testing.assert_allclose(joint[:, 0], expected)
    np.testing.assert_allclose(safe[:, 0], [0, 0, 0, 1, 1])
    np.testing.assert_allclose(first[:, 0], [0, 0, 0, 1, 1])


def test_training_clips_standardized_features_before_ridge_fit():
    data = append_rows(sample_data(), list(range(12)) * 8)
    n = len(data["shot"])
    data["shot"] = np.arange(n)
    data["x"][-1, 0] = 10
    data["raw"][-1, 0] = 2
    heads = fit_heads(**data, ridge=4)
    # Here both roles have the same number of distinct shots and equal weights.
    x = data["x"][:, 0]
    y = data["raw"][:, 0]
    z = np.clip((x - x.mean()) / x.std(), -4, 4)
    slope = np.sum((z - z.mean()) * (y - y.mean())) / (np.sum((z - z.mean()) ** 2) + 4)
    intercept = y.mean() - slope * z.mean()
    np.testing.assert_allclose(heads["coef"][0, 0, 0], slope)
    np.testing.assert_allclose(heads["intercept"][0, 0], intercept)


def test_zero_ridge_with_rank_deficiency_and_no_action_case():
    data = sample_data()
    data["x"] = np.zeros((12, 2))
    heads = fit_heads(**data, ridge=0)
    np.testing.assert_allclose(predictions(heads, [[0, 0]])[0], [[0.2]])
    no_actions = sample_data(0)
    heads = fit_heads(**no_actions)
    np.testing.assert_array_equal(
        choose(heads, [[0], [0]], np.empty((2, 0), bool)), [-1, -1]
    )


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("x", np.zeros(12)),
        ("x", np.full((12, 1), np.nan)),
        ("shot", np.arange(12, dtype=float)),
        ("shot", np.arange(11)),
        ("utility_role", np.full(12, 2)),
        ("utility_role", np.full(12, 0.5)),
        ("guard_role", np.full(12, -2)),
        ("raw", np.zeros((12, 2))),
        ("joint", np.full((12, 1), np.inf)),
        ("harm", np.full((12, 1), -0.01)),
        ("harm", np.full((12, 1), 1.01)),
        ("available", np.ones((12, 1), dtype=int)),
        ("available", np.ones((12, 2), dtype=bool)),
    ],
)
def test_invalid_fit_inputs_raise_value_error(key, bad_value):
    data = sample_data()
    data[key] = bad_value
    with pytest.raises(ValueError):
        fit_heads(**data)


@pytest.mark.parametrize("ridge", [-1, np.nan, np.inf, [1, 2]])
def test_invalid_ridge_raises_value_error(ridge):
    with pytest.raises(ValueError):
        fit_heads(**sample_data(), ridge=ridge)


def test_invalid_inference_inputs_raise_value_error():
    heads = fit_heads(**sample_data())
    with pytest.raises(ValueError):
        predictions({}, [[0]])
    with pytest.raises(ValueError):
        predictions(heads, [[0, 1]])
    with pytest.raises(ValueError):
        predictions(heads, [[np.nan]])
    with pytest.raises(ValueError):
        choose(heads, [[0]], [[True]], objective="wrong")
    with pytest.raises(ValueError):
        choose(heads, [[0]], [[True]], guard=1)
    with pytest.raises(ValueError):
        choose(heads, [[0]], [[1]])
    with pytest.raises(ValueError):
        choose(heads, [[0]], [[True, False]])
