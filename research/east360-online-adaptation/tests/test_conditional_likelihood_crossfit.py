"""CPU-only contracts for nested, leave-one-shot-out density scales."""

from collections import Counter
from importlib import import_module
from io import BytesIO

import numpy as np
import pytest
from scipy.stats import multivariate_t
import torch

from experiments.risk_aligned_online.conditional_likelihood_20261029.density import (
    fit_density,
    log_densities,
)


@pytest.fixture(scope="module")
def crossfit():
    return import_module(
        "experiments.risk_aligned_online.conditional_likelihood_20261029.crossfit"
    )


@pytest.fixture
def sample():
    rng = np.random.default_rng(6173)
    # Shot 201 occurs in both classes: exclusion must be by shot, not row label.
    ids = np.repeat([101, 102, 201, 201, 202], [2, 4, 3, 2, 5])
    labels = np.repeat([0, 0, 0, 1, 1], [2, 4, 3, 2, 5])
    x = rng.normal(size=(len(ids), 64))
    x[:, 0] = np.arange(len(ids))  # A unique row marker for the leakage recorder.
    x[:, 1] = 1.0  # Exercise constant columns, including one-shot inner fits.
    horizon = np.linspace(0.4, 1.6, 25)
    offsets = np.select([ids == 101, ids == 102, ids == 201], [0.0, 2.0, -1.0], 4.0)
    target = np.full((len(ids), 25, 23), np.nan)
    target[:, :, 0] = (offsets + 0.8 * x[:, 2] + 0.4 * labels)[
        :, None
    ] * horizon + rng.normal(scale=0.1, size=(len(ids), 25))
    target[:, :, 1] = (0.5 * offsets - x[:, 3])[:, None] * horizon[::-1] + rng.normal(
        scale=0.15, size=(len(ids), 25)
    )
    target[:, :, 2] = target[:, :, 0] + 1
    target[:, :, 3] = target[:, :, 1] - 1
    valid = np.zeros((len(ids), 23), dtype=bool)
    valid[:, [0, 1, 3]] = True
    valid[:, 2] = (labels == 0) | (ids == 201)
    valid[[0, 5, 11], 1] = False
    target[8, 7, 1] = np.nan  # Incomplete even though the validity flag is true.
    target[(ids == 202) & (labels == 1), 7, 3] = np.nan
    return x, target, valid, ids, labels


def _weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    return weight / weight.sum()


def _numpy_ridge_predict(train_x, train_y, train_ids, query_x):
    """Independent NumPy implementation of the specified weighted ridge."""
    weight = _weights(train_ids)
    mean = (train_x * weight[:, None]).sum(axis=0)
    constant = np.max(train_x, axis=0) == np.min(train_x, axis=0)
    mean[constant] = train_x[0, constant]
    scale = np.sqrt(((train_x - mean) ** 2 * weight[:, None]).sum(axis=0))
    scale[constant] = 1.0
    target_mean = (train_y * weight[:, None]).sum(axis=0)
    design = (train_x - mean) / scale * np.sqrt(weight)[:, None]
    centered = (train_y - target_mean) * np.sqrt(weight)[:, None]
    # Always use the primal equations independently of the implementation's
    # primal/dual choice. The fixture is intentionally N < D.
    coef = np.linalg.solve(
        design.T @ design + 0.1 * np.eye(train_x.shape[1]),
        design.T @ centered,
    )
    return (query_x - mean) / scale @ coef + target_mean


def _reference(sample, channel, label):
    x, target, valid, ids, labels = sample
    eligible = valid[:, channel] & np.isfinite(target[:, :, channel]).all(axis=1)
    eligible &= labels == label
    selected = np.flatnonzero(eligible)
    residual = np.empty((len(selected), 25))
    selected_ids = ids[selected]
    for shot in np.unique(selected_ids):
        held = selected_ids == shot
        train = eligible & (ids != shot)
        residual[held] = target[selected[held], :, channel] - _numpy_ridge_predict(
            x[train], target[train, :, channel], ids[train], x[selected[held]]
        )
    weight = _weights(selected_ids)
    moment = residual.T @ (weight[:, None] * residual)
    scale = 0.5 * moment + 0.5 * np.diag(np.diag(moment)) + 1e-6 * np.eye(25)
    return scale, residual, weight, eligible


def test_crossfit_scale_matches_independent_numpy_and_scipy(sample, crossfit):
    x, target, valid, ids, labels = sample
    fitted = crossfit.fit_crossfit_density(*sample, device="cpu")
    query_valid = np.zeros_like(valid[:3])
    query_valid[:, 0] = True
    logs, counts = log_densities(fitted, x[:3], target[:3], query_valid)
    np.testing.assert_array_equal(counts, 1)
    for channel in (0, 1):
        for label in (0, 1):
            expected, _, _, eligible = _reference(sample, channel, label)
            component = fitted["channels"][channel]["classes"][label]
            np.testing.assert_allclose(
                component["t_scale"], expected, rtol=2e-10, atol=2e-11
            )
            chol = component["t_cholesky"].numpy()
            np.testing.assert_allclose(chol @ chol.T, expected, rtol=2e-10, atol=2e-11)
            if channel == 0:
                mean = _numpy_ridge_predict(
                    x[eligible], target[eligible, :, channel], ids[eligible], x[:3]
                )
                expected_logs = [
                    multivariate_t.logpdf(y, loc=mu, shape=expected, df=4)
                    for y, mu in zip(target[:3, :, channel], mean, strict=True)
                ]
                np.testing.assert_allclose(
                    logs[:, label], expected_logs, rtol=2e-10, atol=2e-9
                )


def test_residual_second_moment_is_uncentered_and_includes_bias(sample, crossfit):
    fitted = crossfit.fit_crossfit_density(*sample)
    scale, residual, weight, _ = _reference(sample, 0, 0)
    residual_mean = np.sum(weight[:, None] * residual, axis=0)
    centered = residual - residual_mean
    centered_moment = centered.T @ (weight[:, None] * centered)
    centered_scale = (
        0.5 * centered_moment
        + 0.5 * np.diag(np.diag(centered_moment))
        + 1e-6 * np.eye(25)
    )
    assert np.linalg.norm(residual_mean) > 0.1
    assert np.linalg.norm(scale - centered_scale) > 0.01
    actual = fitted["channels"][0]["classes"][0]["t_scale"]
    np.testing.assert_allclose(actual, scale, rtol=2e-10, atol=2e-11)
    assert not np.allclose(actual, centered_scale, rtol=1e-4, atol=1e-4)


def test_inner_fits_and_predictions_exclude_entire_held_shot(
    sample, crossfit, monkeypatch
):
    x, target, valid, ids, labels = sample
    records = []
    real_ridge, real_predict = crossfit.ridge, crossfit.predict

    def record_ridge(train_x, train_y, train_ids, penalty, *, device):
        records.append(
            {
                "train_rows": train_x[:, 0].astype(int),
                "train_ids": train_ids.copy(),
                "train_y": train_y.copy(),
            }
        )
        assert penalty == 0.1
        assert device == "cpu"
        return real_ridge(train_x, train_y, train_ids, penalty, device=device)

    def record_predict(model, held_x):
        assert "held_rows" not in records[-1]
        records[-1]["held_rows"] = held_x[:, 0].astype(int)
        return real_predict(model, held_x)

    monkeypatch.setattr(crossfit, "ridge", record_ridge)
    monkeypatch.setattr(crossfit, "predict", record_predict)
    fitted = crossfit.fit_crossfit_density(*sample, device="cpu")

    expected = Counter()
    for channel in (0, 1):
        for label in (0, 1):
            _, _, _, eligible = _reference(sample, channel, label)
            for shot in np.unique(ids[eligible]):
                train = np.flatnonzero(eligible & (ids != shot))
                held = np.flatnonzero(eligible & (ids == shot))
                expected[(tuple(train), tuple(held), channel)] += 1

    actual = Counter()
    for record in records:
        train, held = record["train_rows"], record["held_rows"]
        np.testing.assert_array_equal(record["train_ids"], ids[train])
        assert len(np.unique(ids[held])) == 1
        assert set(ids[train]).isdisjoint(ids[held])
        matches = [
            channel
            for channel in (0, 1)
            if np.array_equal(record["train_y"], target[train, :, channel])
        ]
        assert len(matches) == 1
        actual[(tuple(train), tuple(held), matches[0])] += 1
    assert actual == expected
    # Exactly two eligible class-1 shots imply each inner fit uses one shot.
    for channel in (0, 1):
        folds = fitted["channels"][channel]["classes"][1]["crossfit"]["folds"]
        assert len(folds) == 2
        assert all(fold["fit_shots"] == 1 for fold in folds)


def test_metadata_reports_complete_residual_coverage_and_smaller_inner_fits(
    sample, crossfit
):
    _, _, _, ids, _ = sample
    fitted = crossfit.fit_crossfit_density(*sample)
    assert fitted["crossfit"]["method"] == "leave_one_shot_out"
    assert "smaller" in fitted["crossfit"]["note"].lower()
    assert "calibration" in fitted["crossfit"]["note"].lower()
    for channel in (0, 1):
        for label in (0, 1):
            _, _, _, eligible = _reference(sample, channel, label)
            meta = fitted["channels"][channel]["classes"][label]["crossfit"]
            shots = set(ids[eligible].tolist())
            assert meta["eligible_rows"] == int(eligible.sum())
            assert meta["eligible_shots"] == len(shots)
            assert meta["residual_rows"] == int(eligible.sum())
            assert meta["residual_shots"] == len(shots)
            assert set(meta["residual_shot_ids"]) == shots
            assert {fold["held_shot_id"] for fold in meta["folds"]} == shots
            for fold in meta["folds"]:
                held = fold["held_shot_id"]
                assert set(fold["fit_shot_ids"]) == shots - {held}
                assert fold["fit_shots"] == len(shots) - 1
                assert fold["held_rows"] == int((eligible & (ids == held)).sum())
                assert fold["fit_rows"] == int((eligible & (ids != held)).sum())
    for channel in range(23):
        if not fitted["channels"][channel]["supported"]:
            for component in fitted["channels"][channel]["classes"]:
                meta = component["crossfit"]
                assert meta["eligible_rows"] == component["nrows"]
                assert meta["eligible_shots"] == component["nshots"]
                assert meta["residual_rows"] == meta["residual_shots"] == 0
                assert meta["residual_shot_ids"] == meta["folds"] == []


def test_mean_models_and_unsupported_channels_match_original(sample, crossfit):
    originals = tuple(value.copy() for value in sample)
    baseline = fit_density(*sample, device="cpu")
    fitted = crossfit.fit_crossfit_density(*sample, device="cpu")
    for key in ("version", "feature_dim", "horizon", "n_channels", "df"):
        assert fitted[key] == baseline[key]
    for original, result in zip(baseline["channels"], fitted["channels"], strict=True):
        assert original["supported"] == result["supported"]
        for base_class, new_class in zip(
            original["classes"], result["classes"], strict=True
        ):
            for key in ("nrows", "nshots", "shot_ids"):
                assert new_class[key] == base_class[key]
            if not original["supported"]:
                for key in ("ridge", "t_scale", "t_cholesky"):
                    assert base_class[key] is new_class[key] is None
                continue
            assert new_class["ridge"].keys() == base_class["ridge"].keys()
            for key, value in base_class["ridge"].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        new_class["ridge"][key], value, rtol=0, atol=0
                    )
                else:
                    assert new_class["ridge"][key] == value
    assert not fitted["channels"][2]["supported"]
    assert not fitted["channels"][3]["supported"]
    assert not fitted["channels"][22]["supported"]
    for actual, before in zip(sample, originals, strict=True):
        np.testing.assert_equal(actual, before)


def test_repeating_a_shot_keeps_equal_shot_weight(sample, crossfit):
    x, target, valid, ids, labels = sample
    repeat = np.flatnonzero(ids == 101)
    duplicated = tuple(np.concatenate((a, a[repeat])) for a in sample)
    original = crossfit.fit_crossfit_density(x, target, valid, ids, labels)
    repeated = crossfit.fit_crossfit_density(*duplicated)
    for channel in (0, 1):
        for label in (0, 1):
            original_scale = original["channels"][channel]["classes"][label]["t_scale"]
            repeated_scale = repeated["channels"][channel]["classes"][label]["t_scale"]
            np.testing.assert_allclose(
                original_scale, repeated_scale, rtol=2e-10, atol=2e-11
            )
    before, counts = log_densities(original, x, target, valid)
    after, repeated_counts = log_densities(repeated, x, target, valid)
    np.testing.assert_allclose(before, after, rtol=2e-9, atol=2e-9)
    np.testing.assert_array_equal(counts, repeated_counts)


@pytest.mark.parametrize("danger_rows", [52, 56])
def test_high_dimensional_danger_fit_and_single_shot_inner_fits(danger_rows, crossfit):
    rng = np.random.default_rng(danger_rows)
    ids = np.repeat([11, 12, 21, 22], [3, 4, danger_rows // 2, danger_rows // 2])
    labels = np.repeat([0, 0, 1, 1], [3, 4, danger_rows // 2, danger_rows // 2])
    x = rng.normal(size=(len(ids), 64))
    x[:, 1] = 1.0
    target = np.full((len(ids), 25, 23), np.nan)
    target[:, :, 0] = (x[:, 2] - 0.7 * x[:, 3] + 2.0 * (ids == 22))[
        :, None
    ] * np.linspace(0.5, 1.5, 25) + rng.normal(scale=0.1, size=(len(ids), 25))
    valid = np.zeros((len(ids), 23), dtype=bool)
    valid[:, 0] = True
    data = x, target, valid, ids, labels
    fitted = crossfit.fit_crossfit_density(*data, device="cpu")
    component = fitted["channels"][0]["classes"][1]
    expected, _, _, _ = _reference(data, 0, 1)
    assert component["ridge"]["rows"] == danger_rows
    assert component["ridge"]["solver"] == "dual"
    assert component["crossfit"]["residual_rows"] == danger_rows
    assert [fold["fit_rows"] for fold in component["crossfit"]["folds"]] == [
        danger_rows // 2,
        danger_rows // 2,
    ]
    assert all(fold["fit_shots"] == 1 for fold in component["crossfit"]["folds"])
    np.testing.assert_allclose(component["t_scale"], expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("as_strings", [False, True])
def test_weights_only_roundtrip_preserves_scores_and_metadata(
    sample, crossfit, as_strings
):
    x, target, valid, ids, labels = sample
    fitted = crossfit.fit_crossfit_density(
        x, target, valid, ids.astype(str) if as_strings else ids, labels
    )
    buffer = BytesIO()
    torch.save(fitted, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    assert loaded["crossfit"] == fitted["crossfit"]
    assert (
        loaded["channels"][0]["classes"][0]["crossfit"]
        == (fitted["channels"][0]["classes"][0]["crossfit"])
    )
    before, counts = log_densities(fitted, x, target, valid)
    after, loaded_counts = log_densities(loaded, x, target, valid)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(loaded_counts, counts)


def test_inner_fit_errors_propagate_without_fallback(sample, crossfit, monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError("sentinel inner fit failure")

    monkeypatch.setattr(crossfit, "ridge", fail)
    with pytest.raises(FloatingPointError, match="sentinel inner fit failure"):
        crossfit.fit_crossfit_density(*sample)


@pytest.mark.parametrize("bad", [2, np.nan])
def test_original_input_validation_is_preserved(sample, crossfit, bad):
    x, target, valid, ids, labels = sample
    invalid = labels.astype(float)
    invalid[-1] = bad
    with pytest.raises(ValueError, match="binary"):
        crossfit.fit_crossfit_density(x, target, valid, ids, invalid)
