import copy
import importlib
import inspect
import json

import numpy as np
import pytest


@pytest.fixture(scope="module")
def learning():
    return importlib.import_module(
        "experiments.risk_aligned_online.arrival_reference_20261030.learning"
    )


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(2140)
    ids = np.array([100, 100, 100, 101, 101, 102, 200, 201])
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    prefix = rng.normal(size=(len(ids), 64))
    target = rng.normal(size=(len(ids), 25, 23))
    target += np.arange(len(ids))[:, None, None] * 0.3
    valid = np.zeros((len(ids), 23), dtype=bool)
    valid[:, 0] = True
    valid[[0, 2, 3, 6, 7], 1] = True
    valid[[0, 1, 2, 6], 2] = True  # Only one negative-label shot.
    valid[[6, 7], 3] = True  # No negative-label windows.
    target[~np.broadcast_to(valid[:, None, :], target.shape)] = np.nan
    pair = {
        "prefix_center": np.linspace(-0.8, 0.7, 64).tolist(),
        "prefix_scale": np.linspace(0.2, 2.0, 64).tolist(),
        "arrival_scale": np.linspace(0.3, 2.1, 92).tolist(),
        "q1_coef": np.linspace(-1.1, 1.3, 180).tolist(),
    }
    return pair, prefix, target, valid, ids, labels


@pytest.fixture(scope="module")
def fitted(learning, sample):
    return learning.fit_reference(*sample, device="cpu")


def independent_summary(target, valid):
    safe = np.where(valid[:, None, :], target, 0.0)
    mean = safe.mean(axis=1)
    return np.stack(
        [
            mean,
            safe[:, -1, :] - safe[:, 0, :],
            np.sqrt(np.mean((safe - mean[:, None, :]) ** 2, axis=1)),
            safe[:, -1, :],
        ],
        axis=-1,
    )


def independent_ridge(pair, prefix, target, valid, ids, labels, channel):
    take = (labels == 0) & valid[:, channel]
    shots, inverse, counts = np.unique(
        ids[take], return_inverse=True, return_counts=True
    )
    weights = 1.0 / (len(shots) * counts[inverse])
    h = (prefix[take] - pair["prefix_center"]) / pair["prefix_scale"]
    design = np.column_stack((h, np.ones(len(h))))
    y = (
        independent_summary(target, valid)[take, channel]
        / np.asarray(pair["arrival_scale"]).reshape(23, 4)[channel]
    )
    penalty = np.eye(65) * 0.1
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty,
        design.T @ (weights[:, None] * y),
    )
    return coef, np.sum(weights[:, None] * y, axis=0)


def test_fit_matches_independent_weighted_ridge_and_provenance(
    learning, sample, fitted
):
    pair, prefix, target, valid, ids, labels = sample
    before = copy.deepcopy(pair)
    assert fitted["supported"] == [True, True] + [False] * 21
    assert fitted["fit_ids"] == [100, 101, 102, 200, 201]
    assert fitted["fit_row_indices"] == [0, 1, 2, 3, 4, 5]
    channels = fitted["channels"]
    assert [c["fit_ids"] for c in channels[:4]] == [
        [100, 101, 102],
        [100, 101],
        [100],
        [],
    ]
    assert [c["nrows"] for c in channels[:4]] == [6, 3, 3, 0]
    assert [c["nshots"] for c in channels[:4]] == [3, 2, 1, 0]
    assert [c["fit_row_indices"] for c in channels[:4]] == [
        [0, 1, 2, 3, 4, 5],
        [0, 2, 3],
        [0, 1, 2],
        [],
    ]
    assert [c["unsupported_reason"] for c in channels[:2]] == [None, None]
    assert channels[2]["unsupported_reason"] and channels[3]["unsupported_reason"]
    assert fitted["l2"] == 0.1
    for channel in (0, 1):
        expected_coef, expected_mean = independent_ridge(*sample, channel)
        assert np.shape(channels[channel]["coefficients"]) == (65, 4)
        np.testing.assert_allclose(
            channels[channel]["coefficients"], expected_coef, atol=2e-12
        )
        np.testing.assert_allclose(channels[channel]["mean"], expected_mean, atol=2e-12)
    for channel in range(2, 23):
        assert channels[channel]["coefficients"] is None
        assert len(channels[channel]["mean"]) == 4
    assert pair == before
    assert json.loads(json.dumps(fitted, allow_nan=False)) == fitted


def test_y1_adverse_values_do_not_change_fit(learning, sample, fitted):
    pair, prefix, target, valid, ids, labels = copy.deepcopy(sample)
    negative = labels == 1
    prefix[negative] = 1e12
    target[negative] = 1e12
    valid[negative] = True
    assert learning.fit_reference(pair, prefix, target, valid, ids, labels) == fitted


def test_y1_nonfinite_payload_does_not_enter_training(learning, sample, fitted):
    pair, prefix, target, valid, ids, labels = copy.deepcopy(sample)
    prefix[labels == 1] = np.inf
    target[labels == 1] = np.nan
    valid[labels == 1] = True
    assert learning.fit_reference(pair, prefix, target, valid, ids, labels) == fitted


def test_existing_ridge_receives_only_prefix_and_equal_shot_weights(
    learning, sample, monkeypatch
):
    pair, prefix, target, valid, ids, labels = sample
    original = learning.ridge_fit
    calls = []

    def record(x, y, weights, penalty, *, device):
        calls.append((x.copy(), y.copy(), weights.copy(), penalty, device))
        return original(x, y, weights, penalty, device=device)

    monkeypatch.setattr(learning, "ridge_fit", record)
    learning.fit_reference(*sample, device="cpu")
    assert len(calls) == 2
    for channel, (x, y, weights, penalty, device) in enumerate(calls):
        take = (labels == 0) & valid[:, channel]
        h = (prefix[take] - pair["prefix_center"]) / pair["prefix_scale"]
        expected_y = (
            independent_summary(target, valid)[take, channel]
            / np.asarray(pair["arrival_scale"]).reshape(23, 4)[channel]
        )
        np.testing.assert_allclose(x, h)
        np.testing.assert_allclose(y, expected_y)
        assert x.shape[1] == 64
        assert penalty == 0.1 and device == "cpu"
        totals = [weights[ids[take] == sid].sum() for sid in np.unique(ids[take])]
        np.testing.assert_allclose(totals, np.full(len(totals), 1.0 / len(totals)))


def test_duplicating_all_windows_of_one_shot_preserves_fit(learning, sample, fitted):
    pair, prefix, target, valid, ids, labels = sample
    duplicate = np.flatnonzero(ids == 100)
    repeated = [
        np.concatenate((a, a[duplicate])) for a in (prefix, target, valid, ids, labels)
    ]
    other = learning.fit_reference(pair, *repeated)
    for channel in (0, 1):
        np.testing.assert_allclose(
            other["channels"][channel]["coefficients"],
            fitted["channels"][channel]["coefficients"],
            atol=2e-12,
        )
        np.testing.assert_allclose(
            other["channels"][channel]["mean"],
            fitted["channels"][channel]["mean"],
            atol=2e-12,
        )
    assert other["channels"][0]["nrows"] == 9
    assert other["channels"][0]["nshots"] == 3


def test_prediction_uses_only_normalized_prefix_not_future_mask(
    learning, sample, fitted
):
    pair, prefix, _, valid, _, _ = sample
    prediction = learning.predict_reference(fitted, pair, prefix, valid)
    h = (prefix - pair["prefix_center"]) / pair["prefix_scale"]
    for channel in (0, 1):
        expected = (
            np.column_stack((h, np.ones(len(h))))
            @ fitted["channels"][channel]["coefficients"]
        )
        expected[:, 2] = np.maximum(0.0, expected[:, 2])
        np.testing.assert_allclose(
            prediction[valid[:, channel], channel], expected[valid[:, channel]]
        )
        np.testing.assert_array_equal(prediction[~valid[:, channel], channel], 0.0)
    changed_valid = valid.copy()
    changed_valid[:, 1:] = ~changed_valid[:, 1:]
    changed = learning.predict_reference(fitted, pair, prefix, changed_valid)
    np.testing.assert_array_equal(changed[:, 0], prediction[:, 0])
    np.testing.assert_array_equal(prediction[:, 2:], 0.0)


def test_mean_mode_is_prefix_invariant(learning, sample, fitted):
    pair, prefix, _, valid, _, _ = sample
    a = learning.predict_reference(fitted, pair, prefix, valid, mode="mean")
    b = learning.predict_reference(fitted, pair, prefix + 1e5, valid, mode="mean")
    np.testing.assert_array_equal(a, b)
    for channel in (0, 1):
        expected = np.asarray(fitted["channels"][channel]["mean"])
        np.testing.assert_allclose(
            a[valid[:, channel], channel],
            np.tile(expected, (valid[:, channel].sum(), 1)),
        )


def test_rms_projection_has_nonnegative_convex_range_and_keeps_signed_stats(
    learning, sample, fitted
):
    pair, prefix, _, valid, _, _ = sample
    model = copy.deepcopy(fitted)
    coef = np.zeros((65, 4))
    coef[-1] = [-3.0, -2.0, -5.0, -1.0]
    model["channels"][0]["coefficients"] = coef.tolist()
    model["channels"][0]["mean"] = [-3.0, -2.0, -5.0, -1.0]
    for mode in ("context", "mean"):
        result = learning.predict_reference(model, pair, prefix, valid, mode=mode)
        np.testing.assert_array_equal(
            result[:, 0], np.tile([-3.0, -2.0, 0.0, -1.0], (len(prefix), 1))
        )
        for true_rms in (0.0, 0.25, 2.0, 100.0):
            assert (0.0 - true_rms) ** 2 <= (-5.0 - true_rms) ** 2


def test_feedback_closed_form_zero_identity_and_mask_counts(learning, sample, fitted):
    pair, _, target, valid, _, _ = sample
    supported = np.asarray(fitted["supported"])
    observed = independent_summary(target, valid) / np.asarray(
        pair["arrival_scale"]
    ).reshape(23, 4)
    rng = np.random.default_rng(900)
    reference = rng.normal(size=observed.shape)
    increment, counts = learning.feedback(pair, target, valid, reference, supported)
    mask = valid & supported[None, :]
    coef = np.asarray(pair["q1_coef"])[87:-1].reshape(23, 4)
    expected = np.sum(
        np.where(mask[:, :, None], (observed - reference) * coef, 0.0), axis=(1, 2)
    )
    np.testing.assert_allclose(increment, expected, atol=1e-12)
    np.testing.assert_array_equal(counts, mask.sum(axis=1))
    zero, count = learning.feedback(pair, target, valid, observed, supported)
    np.testing.assert_array_equal(zero, np.zeros(len(target)))
    np.testing.assert_array_equal(count, mask.sum(axis=1))


def test_missing_and_unsupported_are_explicit_no_ops(learning, sample, fitted):
    pair, prefix, _, _, _, _ = sample
    valid = np.zeros((len(prefix), 23), dtype=bool)
    valid[:, 2:4] = True  # Observed but unsupported.
    target = np.full((len(prefix), 25, 23), np.nan)
    target[:, :, 2:4] = 10.0
    prediction = learning.predict_reference(fitted, pair, prefix, valid)
    np.testing.assert_array_equal(prediction, 0.0)
    increment, counts = learning.feedback(
        pair, target, valid, np.ones_like(prediction), fitted["supported"]
    )
    np.testing.assert_array_equal(increment, 0.0)
    np.testing.assert_array_equal(counts, 0)
    valid[:] = False
    target[:] = np.nan
    increment, counts = learning.feedback(
        pair, target, valid, np.ones_like(prediction), fitted["supported"]
    )
    np.testing.assert_array_equal(increment, 0.0)
    np.testing.assert_array_equal(counts, 0)


def test_scoring_has_no_label_fitting_or_outcome_reads(
    learning, sample, fitted, monkeypatch
):
    from experiments.risk_aligned_online import io23, simple_data

    def forbidden(*args, **kwargs):
        raise AssertionError("scoring must not fit or read outcomes/shots")

    monkeypatch.setattr(learning, "ridge_fit", forbidden)
    monkeypatch.setattr(io23, "read_outcome", forbidden)
    monkeypatch.setattr(simple_data, "read_shot", forbidden)
    assert tuple(inspect.signature(learning.predict_reference).parameters) == (
        "ref",
        "pair",
        "prefix",
        "valid",
        "mode",
    )
    assert tuple(inspect.signature(learning.feedback).parameters) == (
        "pair",
        "target",
        "valid",
        "reference",
        "supported",
    )
    pair, prefix, target, valid, _, _ = sample
    original = json.dumps(fitted, sort_keys=True, allow_nan=False)
    predicted = learning.predict_reference(fitted, pair, prefix, valid)
    learning.feedback(pair, target, valid, predicted, fitted["supported"])
    assert json.dumps(fitted, sort_keys=True, allow_nan=False) == original
    with pytest.raises(TypeError):
        learning.predict_reference(
            fitted, pair, prefix, valid, labels=np.zeros(len(prefix))
        )
    with pytest.raises(TypeError):
        learning.feedback(
            pair,
            target,
            valid,
            predicted,
            fitted["supported"],
            labels=np.zeros(len(prefix)),
        )


def test_held_out_rows_do_not_enter_fit_metadata_or_mutate_it(learning, sample):
    pair, prefix, target, valid, ids, labels = sample
    train = np.flatnonzero(ids < 200)
    held = np.flatnonzero(ids >= 200)
    ref = learning.fit_reference(
        pair, prefix[train], target[train], valid[train], ids[train], labels[train]
    )
    before = json.dumps(ref, sort_keys=True, allow_nan=False)
    prediction = learning.predict_reference(ref, pair, prefix[held], valid[held])
    score, counts = learning.feedback(
        pair, target[held], valid[held], prediction, ref["supported"]
    )
    assert np.isfinite(score).all() and np.isfinite(counts).all()
    assert set(ref["fit_ids"]).isdisjoint(ids[held])
    assert all(set(c["fit_ids"]).isdisjoint(ids[held]) for c in ref["channels"])
    assert json.dumps(ref, sort_keys=True, allow_nan=False) == before


def test_fit_solver_failure_propagates(learning, sample, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate ridge failure")

    monkeypatch.setattr(learning, "ridge_fit", fail)
    with pytest.raises(RuntimeError, match="deliberate ridge failure"):
        learning.fit_reference(*sample)


def test_empty_or_single_shot_cohorts_report_unsupported(learning, sample):
    pair, prefix, target, valid, ids, labels = sample
    for changed_labels in (np.ones_like(labels), np.where(ids == 100, 0, 1)):
        ref = learning.fit_reference(pair, prefix, target, valid, ids, changed_labels)
        assert not any(ref["supported"])
        assert all(c["unsupported_reason"] for c in ref["channels"])
        assert all(c["coefficients"] is None for c in ref["channels"])


@pytest.mark.parametrize(
    "which",
    [
        "prefix_shape",
        "prefix_nan",
        "target_shape",
        "target_nan",
        "valid_shape",
        "valid_type",
        "ids_shape",
        "ids_nan",
        "labels_shape",
        "labels_value",
        "labels_nan",
        "center_shape",
        "scale_zero",
        "scale_nan",
        "arrival_scale_zero",
        "arrival_scale_nan",
    ],
)
def test_fit_rejects_invalid_inputs(learning, sample, which):
    pair, prefix, target, valid, ids, labels = copy.deepcopy(sample)
    if which == "prefix_shape":
        prefix = prefix[:, :-1]
    if which == "prefix_nan":
        prefix[0, 0] = np.nan
    if which == "target_shape":
        target = target[:, :-1]
    if which == "target_nan":
        target[0, 1, 0] = np.nan
    if which == "valid_shape":
        valid = valid[:, :-1]
    if which == "valid_type":
        valid = valid.astype(int)
    if which == "ids_shape":
        ids = ids[:-1]
    if which == "ids_nan":
        ids = ids.astype(float)
        ids[0] = np.nan
    if which == "labels_shape":
        labels = labels[:-1]
    if which == "labels_value":
        labels[0] = 2
    if which == "labels_nan":
        labels = labels.astype(float)
        labels[0] = np.nan
    if which == "center_shape":
        pair["prefix_center"].pop()
    if which == "scale_zero":
        pair["prefix_scale"][0] = 0.0
    if which == "scale_nan":
        pair["prefix_scale"][0] = np.nan
    if which == "arrival_scale_zero":
        pair["arrival_scale"][0] = 0.0
    if which == "arrival_scale_nan":
        pair["arrival_scale"][0] = np.nan
    with pytest.raises(ValueError):
        learning.fit_reference(pair, prefix, target, valid, ids, labels)


def test_scoring_rejects_invalid_models_masks_and_scales(learning, sample, fitted):
    pair, prefix, target, valid, _, _ = sample
    reference = np.zeros((len(prefix), 23, 4))
    supported = np.asarray(fitted["supported"])
    with pytest.raises(ValueError):
        learning.predict_reference(fitted, pair, prefix, valid, mode="other")
    with pytest.raises(ValueError):
        learning.predict_reference(fitted, pair, prefix[:, :-1], valid)
    with pytest.raises(ValueError):
        learning.predict_reference(fitted, pair, prefix, valid.astype(int))
    bad = copy.deepcopy(fitted)
    bad["supported"][0] = False
    bad["channels"][0]["coefficients"] = [[np.nan] * 4] * 65
    # Inactive coefficients are not used; supported ones must be finite and correctly shaped.
    bad["supported"][0] = True
    with pytest.raises(ValueError):
        learning.predict_reference(bad, pair, prefix, valid)
    bad = copy.deepcopy(fitted)
    bad["channels"][0]["mean"] = [0.0] * 3
    with pytest.raises(ValueError):
        learning.predict_reference(bad, pair, prefix, valid, mode="mean")
    bad = copy.deepcopy(fitted)
    bad["supported"] = [1] * 23
    with pytest.raises(ValueError):
        learning.predict_reference(bad, pair, prefix, valid)
    with pytest.raises(ValueError):
        learning.feedback(pair, target, valid, reference[:, :, :3], supported)
    reference[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        learning.feedback(pair, target, valid, reference, supported)
    with pytest.raises(ValueError):
        learning.feedback(
            pair, target, valid, np.zeros_like(reference), supported.astype(int)
        )
    bad_pair = copy.deepcopy(pair)
    bad_pair["q1_coef"][87] = np.inf
    with pytest.raises(ValueError):
        learning.feedback(bad_pair, target, valid, np.zeros_like(reference), supported)
    bad_pair = copy.deepcopy(pair)
    bad_pair["arrival_scale"][0] = -1.0
    with pytest.raises(ValueError):
        learning.feedback(bad_pair, target, valid, np.zeros_like(reference), supported)
