import importlib
import inspect
import json

import numpy as np
import pytest

from experiments.risk_aligned_online.conditional_forecast_20261023.learning import (
    shot_weights,
)
from experiments.risk_aligned_online.feedback_capacity_20261024.learning import (
    logistic_fit,
)


@pytest.fixture(scope="module")
def learning():
    return importlib.import_module(
        "experiments.risk_aligned_online.arrival_increment_20261030.learning"
    )


@pytest.fixture(scope="module")
def training():
    rng = np.random.default_rng(481)
    ids = np.array([10, 10, 10, 11, 12, 12, 12, 13, 13])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    prefix = rng.normal(size=(len(ids), 64))
    prefix[:, 0] = np.arange(len(ids))
    prefix[:, -1] = 7.0
    target = rng.normal(size=(len(ids), 25, 23))
    target += labels[:, None, None] * np.linspace(0.0, 2.0, 25)[None, :, None]
    valid = np.ones((len(ids), 23), dtype=bool)
    valid[2] = False
    valid[[0, 4, 7], 1] = False
    valid[:, 22] = False
    target[~np.broadcast_to(valid[:, None, :], target.shape)] = np.nan
    return prefix, target, valid, ids, labels


@pytest.fixture(scope="module")
def fitted(learning, training):
    return learning.fit_pair(*training, device="cpu")


def _independent_summary(target, valid):
    safe = np.where(valid[:, None, :], target, 0.0)
    mean = safe.mean(axis=1)
    return np.stack(
        (
            mean,
            safe[:, -1] - safe[:, 0],
            np.sqrt(np.mean((safe - mean[:, None, :]) ** 2, axis=1)),
            safe[:, -1],
        ),
        axis=2,
    ).reshape(len(target), 92)


def test_summary_has_exact_four_channel_major_statistics(learning):
    t = np.arange(25, dtype=float)
    target = np.empty((1, 25, 23))
    target[0] = np.arange(23)[None, :]
    target[0, :, 0] = t
    target[0, :, 1] = 3.0
    target[0, :, 2] = t**2
    actual = learning.summarize_arrival(target, np.ones((1, 23), bool))
    assert actual.shape == (1, 92)
    np.testing.assert_allclose(actual[0, :4], [12.0, 24.0, np.sqrt(52.0), 24.0])
    np.testing.assert_allclose(actual[0, 4:8], [3.0, 0.0, 0.0, 3.0])
    np.testing.assert_allclose(
        actual[0, 8:12], [196.0, 576.0, np.std(t**2), 576.0]
    )
    np.testing.assert_allclose(actual, _independent_summary(target, np.ones((1, 23), bool)))


def test_reversal_preserves_mean_and_rms_but_reverses_drift_and_endpoint(learning):
    rng = np.random.default_rng(44)
    target = rng.normal(size=(2, 25, 23))
    valid = rng.random((2, 23)) > 0.25
    forward = learning.summarize_arrival(target, valid).reshape(2, 23, 4)
    reversed_ = learning.summarize_arrival(target[:, ::-1], valid).reshape(2, 23, 4)
    np.testing.assert_allclose(reversed_[:, :, 0], forward[:, :, 0], atol=1e-15)
    np.testing.assert_allclose(reversed_[:, :, 1], -forward[:, :, 1], atol=1e-15)
    np.testing.assert_allclose(reversed_[:, :, 2], forward[:, :, 2], atol=1e-15)
    np.testing.assert_allclose(reversed_[:, :, 3], np.where(valid, target[:, 0], 0))


def test_missing_channels_are_zero_even_when_nonfinite(learning):
    target = np.ones((2, 25, 23))
    valid = np.ones((2, 23), dtype=bool)
    valid[0, 2] = False
    valid[1] = False
    target[0, :, 2] = np.nan
    target[1] = np.inf
    result = learning.summarize_arrival(target, valid).reshape(2, 23, 4)
    np.testing.assert_array_equal(result[0, 2], np.zeros(4))
    np.testing.assert_array_equal(result[1], np.zeros((23, 4)))
    np.testing.assert_array_equal(result[0, 1], [1.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_value_in_any_valid_channel_is_rejected(learning, value):
    target = np.zeros((1, 25, 23))
    target[0, 8, 9] = value
    with pytest.raises(ValueError, match="finite"):
        learning.summarize_arrival(target, np.ones((1, 23), bool))


@pytest.mark.parametrize(
    "target,valid",
    [
        (np.zeros((2, 24, 23)), np.ones((2, 23), bool)),
        (np.zeros((2, 25, 22)), np.ones((2, 22), bool)),
        (np.zeros((2, 25, 23)), np.ones((2, 25, 23), bool)),
        (np.zeros((2, 25, 23)), np.ones((2, 23), int)),
        (np.zeros((2, 25, 23)), np.ones((1, 23), bool)),
    ],
)
def test_summary_requires_complete_channel_boolean_mask_and_h25(learning, target, valid):
    with pytest.raises(ValueError):
        learning.summarize_arrival(target, valid)


def test_fit_records_only_training_ids_counts_and_weighted_normalization(learning, training, fitted):
    prefix, target, valid, ids, labels = training
    weights = shot_weights(ids, labels)
    center = np.sum(weights[:, None] * prefix, axis=0)
    std = np.maximum(np.sqrt(np.sum(weights[:, None] * (prefix - center) ** 2, axis=0)), 1e-3)
    summary = _independent_summary(target, valid)
    available = np.repeat(valid, 4, axis=1)
    denominator = np.sum(weights[:, None] * available, axis=0)
    numerator = np.sum(weights[:, None] * summary**2, axis=0)
    rms = np.sqrt(np.divide(numerator, denominator, out=np.zeros(92), where=denominator > 0))
    rms = np.maximum(rms, 1e-3)
    np.testing.assert_allclose(fitted["prefix_center"], center)
    np.testing.assert_allclose(fitted["prefix_scale"], std)
    np.testing.assert_allclose(fitted["arrival_scale"], rms)
    assert fitted["prefix_scale"][-1] == 1e-3
    np.testing.assert_array_equal(fitted["arrival_scale"][-4:], np.full(4, 1e-3))
    assert fitted["fit_ids"] == [10, 11, 12, 13]
    assert fitted["label_shot_counts"] == {"0": 2, "1": 2}
    assert fitted["label_window_counts"] == {"0": 4, "1": 5}
    assert fitted["l2"] == 0.1
    assert fitted["q0_trace"][-1]["gradient_max"] <= 1e-8
    assert fitted["q1_trace"][-1]["gradient_max"] <= 1e-8


def test_pair_uses_same_weights_masks_and_independent_old_solver(learning, training, monkeypatch):
    prefix, target, valid, ids, labels = training
    calls = []

    def record(x, y, weights, **kwargs):
        calls.append((x.copy(), y.copy(), weights.copy(), dict(kwargs)))
        return logistic_fit(x, y, weights, **kwargs)

    monkeypatch.setattr(learning, "logistic_fit", record)
    fit = learning.fit_pair(*training, device="cpu")
    assert len(calls) == 2
    weights = shot_weights(ids, labels)
    summary = _independent_summary(target, valid)
    h = (prefix - np.asarray(fit["prefix_center"])) / np.asarray(fit["prefix_scale"])
    q0 = np.column_stack((h, valid))
    q1 = np.column_stack((h, valid, summary / np.asarray(fit["arrival_scale"])))
    for (x, y, w, kwargs), expected in zip(calls, (q0, q1), strict=True):
        np.testing.assert_allclose(x, expected)
        np.testing.assert_array_equal(y, labels)
        np.testing.assert_allclose(w, weights)
        assert kwargs["device"] == "cpu"
        assert kwargs.get("l2", 0.1) == 0.1
        assert kwargs.get("offset") is None
        assert kwargs.get("intercept", True) is True
    expected0, trace0 = logistic_fit(q0, labels, weights, device="cpu", l2=0.1)
    expected1, trace1 = logistic_fit(q1, labels, weights, device="cpu", l2=0.1)
    np.testing.assert_allclose(fit["q0_coef"], expected0, atol=1e-12)
    np.testing.assert_allclose(fit["q1_coef"], expected1, atol=1e-12)
    assert fit["q0_trace"] == trace0
    assert fit["q1_trace"] == trace1
    assert not np.allclose(expected0[:-1], expected1[:87])
    scores = learning.score_pair(fit, prefix, target, valid)
    manual0 = q0 @ expected0[:-1] + expected0[-1]
    manual1 = q1 @ expected1[:-1] + expected1[-1]
    constant = q0 @ expected1[:87] + expected1[-1]
    np.testing.assert_allclose(scores["q0_logit"], manual0, atol=1e-12)
    np.testing.assert_allclose(scores["q1_logit"], manual1, atol=1e-12)
    np.testing.assert_allclose(scores["constant_logit"], constant, atol=1e-12)
    present = valid.any(axis=1)
    np.testing.assert_allclose(scores["paired_increment"][present], (manual1 - manual0)[present])
    np.testing.assert_allclose(scores["arrival_increment"][present], (manual1 - constant)[present])


def test_zero_arrival_cancels_q1_prefix_and_mask_exactly(learning, training, fitted):
    prefix, _, valid, _, _ = training
    result = learning.score_pair(fitted, prefix, np.zeros((len(prefix), 25, 23)), valid)
    np.testing.assert_array_equal(result["q1_logit"], result["constant_logit"])
    np.testing.assert_array_equal(result["arrival_increment"], np.zeros(len(prefix)))


def test_independent_models_can_disagree_on_mask_alone(learning, fitted):
    fit = json.loads(json.dumps(fitted, allow_nan=False))
    fit["q0_coef"] = [0.0] * len(fit["q0_coef"])
    fit["q1_coef"] = [0.0] * len(fit["q1_coef"])
    fit["q1_coef"][64] = 2.0
    prefix = np.asarray(fit["prefix_center"])[None, :]
    valid = np.zeros((1, 23), dtype=bool)
    valid[0, 0] = True
    result = learning.score_pair(fit, prefix, np.zeros((1, 25, 23)), valid)
    np.testing.assert_array_equal(result["paired_increment"], [2.0])
    np.testing.assert_array_equal(result["arrival_increment"], [0.0])


def test_all_missing_returns_zero_increments_but_keeps_diagnostic_logits(learning, fitted):
    fit = json.loads(json.dumps(fitted, allow_nan=False))
    fit["q0_coef"][-1] = 1.0
    fit["q1_coef"][-1] = 3.0
    prefix = np.tile(fit["prefix_center"], (2, 1))
    valid = np.zeros((2, 23), dtype=bool)
    target = np.full((2, 25, 23), np.nan)
    result = learning.score_pair(fit, prefix, target, valid)
    np.testing.assert_allclose(result["q0_logit"], 1.0)
    np.testing.assert_allclose(result["q1_logit"], 3.0)
    np.testing.assert_array_equal(result["paired_increment"], np.zeros(2))
    np.testing.assert_array_equal(result["arrival_increment"], np.zeros(2))


def test_held_extremes_do_not_refit_or_mutate_training_state(learning, training, fitted, monkeypatch):
    from experiments.risk_aligned_online import io23, simple_data

    before = json.dumps(fitted, sort_keys=True, allow_nan=False)
    prefix, target, valid, _, _ = training
    held_prefix = prefix[:2].copy() + 1e6
    held_target = np.where(valid[:2, None, :], 1e6, np.inf)
    held_target = np.broadcast_to(held_target, (2, 25, 23)).copy()
    prefix_before = held_prefix.copy()
    target_before = held_target.copy()
    valid_before = valid.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("scoring may not fit, compute weights, or read an outcome/shot")

    monkeypatch.setattr(learning, "logistic_fit", forbidden)
    monkeypatch.setattr(learning, "shot_weights", forbidden)
    monkeypatch.setattr(io23, "read_outcome", forbidden)
    monkeypatch.setattr(simple_data, "read_shot", forbidden)
    assert tuple(inspect.signature(learning.score_pair).parameters) == ("fit", "prefix", "target", "valid")
    score_only_fit = {
        key: fitted[key]
        for key in ("prefix_center", "prefix_scale", "arrival_scale", "q0_coef", "q1_coef")
    }
    result = learning.score_pair(score_only_fit, held_prefix, held_target, valid[:2])
    assert all(np.isfinite(v).all() for v in result.values())
    assert json.dumps(fitted, sort_keys=True, allow_nan=False) == before
    np.testing.assert_array_equal(held_prefix, prefix_before)
    np.testing.assert_array_equal(held_target, target_before)
    np.testing.assert_array_equal(valid, valid_before)
    with pytest.raises(TypeError):
        learning.score_pair(fitted, held_prefix, held_target, valid[:2], labels=[0, 1])


def test_fit_json_round_trip_reproduces_scoring(learning, training, fitted):
    restored = json.loads(json.dumps(fitted, allow_nan=False))
    prefix, target, valid, _, _ = training
    actual = learning.score_pair(restored, prefix, target, valid)
    expected = learning.score_pair(fitted, prefix, target, valid)
    assert set(actual) == {"q0_logit", "q1_logit", "constant_logit", "paired_increment", "arrival_increment"}
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_optimizer_errors_propagate_without_fallback(learning, training, monkeypatch, fail_on_call):
    calls = 0

    def failing_solver(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise RuntimeError("optimizer failed deliberately")
        return logistic_fit(*args, **kwargs)

    monkeypatch.setattr(learning, "logistic_fit", failing_solver)
    with pytest.raises(RuntimeError, match="optimizer failed deliberately"):
        learning.fit_pair(*training, device="cpu")
    assert calls == fail_on_call


def test_counts_track_each_class_even_when_one_shot_has_both(learning, training):
    prefix, target, valid, ids, labels = training
    mixed = labels.copy()
    mixed[1] = 1
    fit = learning.fit_pair(prefix, target, valid, ids, mixed)
    assert fit["fit_ids"] == [10, 11, 12, 13]
    assert fit["label_shot_counts"] == {"0": 2, "1": 3}
    assert fit["label_window_counts"] == {"0": 3, "1": 6}


def test_fit_rejects_bad_inputs_and_score_rejects_bad_parameters(learning, training, fitted):
    prefix, target, valid, ids, labels = training
    with pytest.raises(ValueError):
        learning.fit_pair(prefix[:, :-1], target, valid, ids, labels)
    bad_prefix = prefix.copy()
    bad_prefix[0, 0] = np.nan
    with pytest.raises(ValueError):
        learning.fit_pair(bad_prefix, target, valid, ids, labels)
    with pytest.raises(ValueError):
        learning.fit_pair(prefix, target, valid, ids[:-1], labels)
    with pytest.raises(ValueError):
        learning.fit_pair(prefix, target, valid, ids, np.zeros_like(labels))
    bad_fit = json.loads(json.dumps(fitted))
    bad_fit["arrival_scale"][0] = 0.0
    with pytest.raises(ValueError):
        learning.score_pair(bad_fit, prefix, target, valid)
