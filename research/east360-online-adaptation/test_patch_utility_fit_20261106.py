"""CPU-only synthetic checks for the fixed action-utility estimator."""

from dataclasses import FrozenInstanceError
import builtins
import hashlib
import os

import numpy as np
import pytest

from patch_utility_fit_20261106 import choices, fit_model, shot_folds, shot_weights


@pytest.fixture
def sample():
    rng = np.random.default_rng(20261106)
    allowed = np.array(
        [
            [1, 1, 1, 0],  # class 0, shot 0, first block
            [1, 0, 0, 1],  # class 0, shot 0, second block
            [1, 1, 1, 1],  # class 0, shot 1
            [1, 0, 1, 1],  # class 1, shot 2, first block
            [1, 1, 0, 0],  # class 1, shot 2, second block
            [1, 1, 1, 1],  # held-out shot 3
            [1, 0, 0, 0],  # selected shot 4, but no eligible blocks
        ],
        dtype=bool,
    )
    x = rng.normal(size=(7, 4, 3))
    x[..., 0] = 5.0  # A constant must be represented by scale 1 and zero effect.
    y = rng.normal(size=(7, 4))
    y[:, 0] = 0.0
    return {
        "x": x,
        "y": y,
        "allowed": allowed,
        "block_shot": np.array([0, 0, 1, 2, 2, 3, 4]),
        "disruptive": np.array([False, False, True, True, False]),
        "train_shots": np.array([True, True, True, False, True]),
    }


def test_shot_weights_balance_classes_shots_blocks_and_actions(sample):
    w = shot_weights(
        sample["block_shot"], sample["disruptive"], sample["train_shots"], sample["allowed"]
    )
    expected = np.array(
        [
            [0, 1 / 16, 1 / 16, 0],
            [0, 0, 0, 1 / 8],
            [0, 1 / 12, 1 / 12, 1 / 12],
            [0, 0, 1 / 8, 1 / 8],
            [0, 1 / 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-16)
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w[:3].sum(), 0.5)
    np.testing.assert_allclose(w[3:5].sum(), 0.5)


def test_single_supported_class_gets_all_weight(sample):
    train = sample["train_shots"].copy()
    train[2] = False
    w = shot_weights(sample["block_shot"], sample["disruptive"], train, sample["allowed"])
    np.testing.assert_allclose([w[:2].sum(), w[2].sum(), w.sum()], [0.5, 0.5, 1.0])
    assert not w[3:].any()


def test_fit_matches_independent_weighted_ridge_with_unpenalized_intercept(sample):
    alpha = 0.37
    w = shot_weights(
        sample["block_shot"], sample["disruptive"], sample["train_shots"], sample["allowed"]
    )
    selected = w > 0
    features = sample["x"][selected]
    target = sample["y"][selected]
    weight = w[selected]
    mean = np.average(features, axis=0, weights=weight)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weight))
    mean[0] = 5.0
    scale[0] = 1.0
    z = (features - mean) / scale
    target_mean = np.average(target, weights=weight)
    coefficient = np.linalg.solve(
        z.T @ (weight[:, None] * z) + alpha * np.eye(z.shape[1]),
        z.T @ (weight * (target - target_mean)),
    )
    model = fit_model(**sample, alpha=alpha)
    assert model.mean[0] == 5.0
    assert model.scale[0] == 1.0
    assert abs(model.coefficient[0]) < 1e-14
    prediction = model.predict(sample["x"], sample["allowed"])
    expected = target_mean + ((sample["x"] - mean) / scale) @ coefficient
    positive = sample["allowed"].copy()
    positive[:, 0] = False
    np.testing.assert_allclose(prediction[positive], expected[positive], rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(prediction[:, 0], 0.0)
    assert np.all(np.isneginf(prediction[~sample["allowed"]]))


@pytest.mark.parametrize("alpha", [0.0, 1000.0])
def test_constant_features_and_unpenalized_intercept(sample, alpha):
    x = np.full_like(sample["x"], 7.0)
    y = np.full_like(sample["y"], 3.25)
    y[:, 0] = 0.0
    model = fit_model(**{**sample, "x": x, "y": y}, alpha=alpha)
    pred = model.predict(x, sample["allowed"])
    positive = sample["allowed"].copy()
    positive[:, 0] = False
    np.testing.assert_allclose(pred[positive], 3.25, rtol=0, atol=1e-13)


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_finite_feature_magnitudes_do_not_underflow_or_overflow_std(magnitude):
    x = np.array([[[0.0], [-magnitude]], [[0.0], [magnitude]]])
    y = np.array([[0.0, -1.0], [0.0, 1.0]])
    allowed = np.ones((2, 2), dtype=bool)
    model = fit_model(x, y, allowed, np.array([0, 1]), np.array([False, True]),
                      np.ones(2, dtype=bool))
    np.testing.assert_allclose(model.scale, [magnitude], rtol=2e-15)
    np.testing.assert_allclose(model.predict(x, allowed)[:, 1], [-1 / 1.03, 1 / 1.03],
                               rtol=2e-14, atol=0)


def test_fit_has_no_dependency_on_heldout_or_unavailable_values(sample):
    model = fit_model(**sample)
    w = shot_weights(
        sample["block_shot"], sample["disruptive"], sample["train_shots"], sample["allowed"]
    )
    x2, y2 = sample["x"].copy(), sample["y"].copy()
    x2[w == 0] = 10_000.0
    excluded_positive = (w == 0).copy()
    excluded_positive[:, 0] = False
    y2[excluded_positive] = -30_000.0
    disruptive2 = sample["disruptive"].copy()
    disruptive2[3:] = ~disruptive2[3:]  # held out or no eligible rows
    allowed2 = sample["allowed"].copy()
    allowed2[5, 1:] = False  # changing held-out availability also cannot affect fit
    changed = fit_model(**{**sample, "x": x2, "y": y2, "disruptive": disruptive2,
                           "allowed": allowed2})
    np.testing.assert_allclose(
        model.predict(sample["x"], sample["allowed"]),
        changed.predict(sample["x"], sample["allowed"]),
        rtol=0,
        atol=0,
    )


def test_repeatability_and_block_permutation(sample):
    order = np.array([6, 4, 1, 5, 0, 3, 2])
    shuffled = {**sample}
    for key in ("x", "y", "allowed", "block_shot"):
        shuffled[key] = sample[key][order]
    model = fit_model(**sample)
    repeated = fit_model(**sample)
    permuted = fit_model(**shuffled)
    original = model.predict(sample["x"], sample["allowed"])
    np.testing.assert_array_equal(original, repeated.predict(sample["x"], sample["allowed"]))
    np.testing.assert_allclose(
        original, permuted.predict(sample["x"], sample["allowed"]), rtol=2e-13, atol=2e-13
    )
    original_weights = shot_weights(
        sample["block_shot"], sample["disruptive"], sample["train_shots"], sample["allowed"]
    )
    shuffled_weights = shot_weights(
        shuffled["block_shot"], shuffled["disruptive"], shuffled["train_shots"], shuffled["allowed"]
    )
    np.testing.assert_array_equal(shuffled_weights, original_weights[order])


def test_relabeling_shot_positions_does_not_change_the_fit(sample):
    shot_order = np.array([3, 1, 4, 0, 2])
    inverse = np.argsort(shot_order)
    relabeled = {
        **sample,
        "block_shot": inverse[sample["block_shot"]],
        "disruptive": sample["disruptive"][shot_order],
        "train_shots": sample["train_shots"][shot_order],
    }
    original = fit_model(**sample).predict(sample["x"], sample["allowed"])
    reindexed = fit_model(**relabeled).predict(sample["x"], sample["allowed"])
    np.testing.assert_allclose(original, reindexed, rtol=2e-13, atol=2e-13)


def test_fixed_alpha_is_invariant_to_duplicating_every_block(sample):
    doubled = {**sample}
    for key in ("x", "y", "allowed", "block_shot"):
        doubled[key] = np.concatenate([sample[key], sample[key]], axis=0)
    original = fit_model(**sample, alpha=0.8).predict(sample["x"], sample["allowed"])
    duplicate = fit_model(**doubled, alpha=0.8).predict(sample["x"], sample["allowed"])
    np.testing.assert_allclose(original, duplicate, rtol=2e-13, atol=2e-13)


def test_permuting_positive_action_columns_preserves_scores(sample):
    action_order = np.array([0, 3, 1, 2])
    shuffled = {**sample}
    for key in ("x", "y", "allowed"):
        shuffled[key] = sample[key][:, action_order]
    original = fit_model(**sample).predict(sample["x"], sample["allowed"])
    permuted = fit_model(**shuffled).predict(shuffled["x"], shuffled["allowed"])
    np.testing.assert_allclose(permuted, original[:, action_order], rtol=2e-13, atol=2e-13)


def test_offset_of_positive_targets_shifts_only_the_intercept(sample):
    shifted_y = sample["y"].copy()
    shifted_y[:, 1:] += 100.0
    original = fit_model(**sample)
    shifted = fit_model(**{**sample, "y": shifted_y})
    np.testing.assert_allclose(original.coefficient, shifted.coefficient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shifted.intercept, original.intercept + 100.0, rtol=1e-14)


def test_model_is_frozen_and_operations_do_not_mutate_inputs(sample):
    before = {key: value.copy() for key, value in sample.items()}
    model = fit_model(**sample)
    prediction = model.predict(sample["x"], sample["allowed"])
    prediction_before = prediction.copy()
    choices(prediction, sample["allowed"], 0.0)
    for key in sample:
        np.testing.assert_array_equal(sample[key], before[key])
    np.testing.assert_array_equal(prediction, prediction_before)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        model.intercept = 999.0
    with pytest.raises(TypeError):
        model.mean[0] = 999.0


def test_fit_and_prediction_do_not_access_files(sample, monkeypatch):
    def reject_file_access(*args, **kwargs):
        raise AssertionError("unexpected file access")

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "open", reject_file_access)
        guarded.setattr(os, "open", reject_file_access)
        shot_folds([42, 100, -5])
        model = fit_model(**sample)
        prediction = model.predict(sample["x"], sample["allowed"])
        choices(prediction, sample["allowed"], 0.0)


def test_folds_are_sha256_round_robin_and_stable_under_reordering():
    ids = np.array([13, -8, 10000, 21, 0, 4, 7, 19, 51, 99, -1, 112, 23])
    folds = shot_folds(ids, n=5)
    expected_order = sorted(ids.tolist(), key=lambda s: (hashlib.sha256(str(s).encode("ascii")).digest(), s))
    expected = {shot: rank % 5 for rank, shot in enumerate(expected_order)}
    np.testing.assert_array_equal(folds, [expected[int(s)] for s in ids])
    assert folds.shape == ids.shape and np.issubdtype(folds.dtype, np.integer)
    assert np.ptp(np.bincount(folds, minlength=5)) <= 1
    order = np.array([7, 0, 12, 4, 2, 10, 3, 11, 9, 8, 1, 6, 5])
    np.testing.assert_array_equal(shot_folds(ids[order]), folds[order])
    np.testing.assert_array_equal(shot_folds(ids, n=1), 0)
    np.testing.assert_array_equal(shot_folds(np.array([], dtype=int)), np.array([], dtype=int))
    assert np.unique(shot_folds(ids[:3], n=7)).size == 3


@pytest.mark.parametrize("ids,n", [([1, 1], 2), ([1.0, 2.0], 2), ([True, False], 2), ([[1, 2]], 2), ([1], 0), ([1], -2), ([1], 1.5), ([1], True)])
def test_folds_reject_invalid_arguments(ids, n):
    with pytest.raises((ValueError, TypeError)):
        shot_folds(ids, n)


def test_choice_strict_margin_stable_real_action_ties_and_nonfinite_scores():
    prediction = np.array(
        [
            [0, 1.0, 1.0, -5.0],
            [0, 99.0, 1.0, 1.0],
            [0, 0.5, 0.2, -np.inf],
            [0, np.nan, np.inf, -np.inf],
            [0, np.nan, np.inf, 0.8],
            [0, -1.0, -2.0, -3.0],
        ]
    )
    allowed = np.ones_like(prediction, dtype=bool)
    allowed[1, 1] = False
    np.testing.assert_array_equal(choices(prediction, allowed, 0.5), [1, 2, 0, 0, 3, 0])
    np.testing.assert_array_equal(choices(prediction, allowed, np.inf), np.zeros(6, dtype=int))
    np.testing.assert_array_equal(choices(np.zeros((2, 1)), np.ones((2, 1), dtype=bool), 0), [0, 0])


@pytest.mark.parametrize("margin", [-1.0, -np.inf, np.nan, True, [0.0], "0.1"])
def test_choice_rejects_invalid_margin(margin):
    with pytest.raises((ValueError, TypeError)):
        choices(np.zeros((1, 2)), np.ones((1, 2), dtype=bool), margin)


@pytest.mark.parametrize(
    "key,value",
    [
        ("x", np.zeros((7, 4))),
        ("x", np.full((7, 4, 3), np.nan)),
        ("x", np.full((7, 4, 3), np.inf)),
        ("x", np.full((7, 4, 3), "1")),
        ("y", np.zeros((7, 3))),
        ("y", np.full((7, 4), np.nan)),
        ("y", np.ones((7, 4))),
        ("allowed", np.ones((7, 4), dtype=int)),
        ("allowed", np.ones((7, 3), dtype=bool)),
        ("block_shot", np.array([0, 0, 1, 2, 2, 3, 5])),
        ("block_shot", np.array([0, 0, 1, 2, 2, 3, -1])),
        ("block_shot", np.zeros(7, dtype=float)),
        ("disruptive", np.zeros(5, dtype=int)),
        ("train_shots", np.ones(4, dtype=bool)),
    ],
)
def test_fit_rejects_invalid_shapes_types_values(sample, key, value):
    with pytest.raises((ValueError, TypeError)):
        fit_model(**{**sample, key: value})


def test_fit_rejects_nonfinite_values_even_outside_training_support(sample):
    x = sample["x"].copy()
    x[0, 3, 1] = np.nan  # unavailable action
    y = sample["y"].copy()
    y[5, 1] = np.inf  # held-out shot
    for key, value in (("x", x), ("y", y)):
        with pytest.raises(ValueError):
            fit_model(**{**sample, key: value})


@pytest.mark.parametrize("alpha", [-1, np.nan, np.inf, True, "0.03"])
def test_fit_rejects_invalid_alpha(sample, alpha):
    with pytest.raises((ValueError, TypeError)):
        fit_model(**sample, alpha=alpha)


def test_fit_and_weights_reject_no_support_and_abstain_not_allowed(sample):
    no_train = np.zeros_like(sample["train_shots"])
    no_positive = sample["allowed"].copy()
    no_positive[:, 1:] = False
    no_abstain = sample["allowed"].copy()
    no_abstain[5, 0] = False  # all rows, even held out, must allow abstention
    for train, allowed in [(no_train, sample["allowed"]), (sample["train_shots"], no_positive),
                           (sample["train_shots"], no_abstain)]:
        with pytest.raises((ValueError, TypeError)):
            fit_model(**{**sample, "train_shots": train, "allowed": allowed})
        with pytest.raises((ValueError, TypeError)):
            shot_weights(sample["block_shot"], sample["disruptive"], train, allowed)


def test_predict_and_choices_reject_shape_and_boolean_contract_violations(sample):
    model = fit_model(**sample)
    bad_allowed = sample["allowed"].copy()
    bad_allowed[0, 0] = False
    for x, allowed in [(sample["x"][:, :, :2], sample["allowed"]),
                       (sample["x"], bad_allowed),
                       (sample["x"], sample["allowed"].astype(int)),
                       (np.full_like(sample["x"], np.nan), sample["allowed"])]:
        with pytest.raises((ValueError, TypeError)):
            model.predict(x, allowed)
    for pred, allowed in [(np.zeros((7, 3)), sample["allowed"]),
                          (np.zeros((7, 4)), bad_allowed),
                          (np.zeros((7, 4)), sample["allowed"].astype(int))]:
        with pytest.raises((ValueError, TypeError)):
            choices(pred, allowed, 0.0)
