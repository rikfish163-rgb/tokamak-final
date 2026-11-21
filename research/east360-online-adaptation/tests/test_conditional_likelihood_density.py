"""CPU-only contract tests for conditional predictive-density arithmetic."""

from io import BytesIO
import inspect

import numpy as np
import pytest
from scipy.stats import multivariate_t
import torch

from experiments.risk_aligned_online.conditional_likelihood_20261029.density import (
    channel_log_densities,
    fit_density,
    log_densities,
    mixture_feedback,
)
from experiments.risk_aligned_online.readout_identifiability_20261029.learning import (
    predict as reference_predict,
    ridge as reference_ridge,
)


@pytest.fixture
def sample():
    rng = np.random.default_rng(82031)
    ids = np.repeat([101, 102, 201, 202], 3)
    labels = np.repeat([0, 0, 1, 1], 3)
    x = rng.normal(size=(len(ids), 3))
    coefficients = rng.normal(scale=0.3, size=(3, 25, 23))
    target = np.einsum("nd,dhc->nhc", x, coefficients)
    target += labels[:, None, None] * 0.5
    target += rng.normal(scale=0.2, size=target.shape)
    valid = np.zeros((len(ids), 23), dtype=bool)
    valid[:, [0, 1, 10]] = True
    # Channel 2 has many class-1 rows, but they all come from a single shot.
    valid[:, 2] = (labels == 0) | (ids == 201)
    # A nominally valid incomplete target must not manufacture a second shot.
    valid[:, 3] = True
    target[ids == 201, 7, 3] = np.nan
    target[~np.broadcast_to(valid[:, None, :], target.shape)] = np.nan
    return x, target, valid, ids, labels


def _independent_t_score(x, target, ids, label, channel, query_x, query_y, labels):
    selected = (
        (labels == label)
        & np.isfinite(target[:, :, channel]).all(axis=1)
    )
    fit = reference_ridge(
        x[selected], target[selected, :, channel], ids[selected], 0.1, device="cpu"
    )
    residual = target[selected, :, channel] - reference_predict(fit, x[selected])
    _, inverse, counts = np.unique(ids[selected], return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    centered = residual - np.average(residual, axis=0, weights=weights)
    covariance = (centered * weights[:, None]).T @ centered
    shape = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
    shape += 1e-6 * np.eye(25)
    prediction = reference_predict(fit, query_x)
    result = np.array(
        [
            multivariate_t.logpdf(y, loc=mu, shape=shape, df=4)
            for y, mu in zip(query_y, prediction, strict=True)
        ]
    )
    return result, shape


def test_normalized_t_log_densities_match_scipy_per_channel(sample):
    x, target, valid, ids, labels = sample
    fit = fit_density(x, target, valid, ids, labels, device="cpu")
    # Only channel 0 is exposed to the scoring function in this comparison.
    query_valid = np.zeros_like(valid[:4])
    query_valid[:, 0] = True
    logs, counts = log_densities(fit, x[:4], target[:4], query_valid)
    assert np.array_equal(counts, np.ones(4, dtype=int))
    for label in (0, 1):
        expected, scale = _independent_t_score(
            x, target, ids, label, 0, x[:4], target[:4, :, 0], labels
        )
        np.testing.assert_allclose(logs[:, label], expected, rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(
            fit["channels"][0]["classes"][label]["t_scale"],
            scale,
            rtol=1e-12,
            atol=1e-12,
        )


def test_per_channel_export_sums_to_total_and_respects_shared_mask(sample):
    x, target, valid, ids, labels = sample
    fit = fit_density(x, target, valid, ids, labels)
    by_channel, used = channel_log_densities(fit, x, target, valid)
    total, counts = log_densities(fit, x, target, valid)
    assert by_channel.shape == (len(x), 23, 2)
    assert used.shape == (len(x), 23)
    assert used.dtype == np.bool_
    assert np.array_equal(used.sum(axis=1), counts)
    np.testing.assert_allclose(by_channel.sum(axis=1), total, rtol=0, atol=1e-12)
    assert np.all(by_channel[~used] == 0)
    assert np.all(used[:, 10])  # Exclusion is the caller's mask, not a hardcoded rule.
    assert not np.any(used[:, 2])
    assert not np.any(used[:, 3])


def test_each_class_requires_two_distinct_complete_shots_and_keeps_metadata(sample):
    x, target, valid, ids, labels = sample
    fit = fit_density(x, target, valid, ids, labels)
    channels = fit["channels"]
    assert channels[0]["supported"]
    assert channels[0]["classes"][0]["nrows"] == 6
    assert channels[0]["classes"][0]["nshots"] == 2
    assert set(channels[0]["classes"][0]["shot_ids"]) == {101, 102}
    assert not channels[2]["supported"]
    assert channels[2]["classes"][1]["nrows"] == 3
    assert channels[2]["classes"][1]["nshots"] == 1
    assert not channels[3]["supported"]
    assert channels[3]["classes"][1]["nrows"] == 3
    assert channels[3]["classes"][1]["nshots"] == 1
    assert not channels[22]["supported"]
    assert channels[22]["classes"][0]["nrows"] == 0


def test_repeating_every_row_of_one_shot_preserves_shot_weighting(sample):
    x, target, valid, ids, labels = sample
    fit = fit_density(x, target, valid, ids, labels)
    repeat = np.flatnonzero(ids == 101)
    duplicated = tuple(np.concatenate((a, a[repeat]), axis=0) for a in sample)
    fit_dup = fit_density(*duplicated)
    qvalid = np.zeros_like(valid)
    qvalid[:, 0] = True
    actual, _ = log_densities(fit, x, target, qvalid)
    repeated, _ = log_densities(fit_dup, x, target, qvalid)
    np.testing.assert_allclose(actual, repeated, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        fit["channels"][0]["classes"][0]["t_scale"],
        fit_dup["channels"][0]["classes"][0]["t_scale"],
        rtol=1e-11,
        atol=1e-12,
    )


def test_missing_and_incomplete_targets_are_not_evidence(sample):
    x, target, valid, ids, labels = sample
    fit = fit_density(x, target, valid, ids, labels)
    qtarget = target[:3].copy()
    qvalid = np.zeros_like(valid[:3])
    qvalid[:, 0] = True
    expected, counts = log_densities(fit, x[:3], qtarget, qvalid)
    qtarget[:, :, 1] = np.nan
    qtarget[:, :, 10] = 1e200
    actual, altered_counts = log_densities(fit, x[:3], qtarget, qvalid)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(altered_counts, counts)
    # A true validity flag is insufficient if even one target time is missing.
    qvalid[:, 1] = True
    qtarget[:, :, 1] = target[:3, :, 1]
    qtarget[:, 9, 1] = np.nan
    incomplete, incomplete_counts = log_densities(fit, x[:3], qtarget, qvalid)
    np.testing.assert_array_equal(incomplete, expected)
    np.testing.assert_array_equal(incomplete_counts, counts)
    qvalid[:] = False
    qtarget[:] = np.nan
    empty, empty_counts = log_densities(fit, x[:3], qtarget, qvalid)
    np.testing.assert_array_equal(empty, np.zeros((3, 2)))
    np.testing.assert_array_equal(empty_counts, np.zeros(3, dtype=int))


def test_finite_extreme_target_has_finite_student_t_log_density(sample):
    x, target, valid, ids, labels = sample
    fit = fit_density(x, target, valid, ids, labels)
    qtarget = np.full((1, 25, 23), np.nan)
    qtarget[0, :, 0] = 1e200
    qvalid = np.zeros((1, 23), dtype=bool)
    qvalid[:, 0] = True
    logs, counts = log_densities(fit, x[:1], qtarget, qvalid)
    assert np.isfinite(logs).all()
    assert (logs < 0).all()
    np.testing.assert_array_equal(counts, [1])


def test_no_labels_are_accepted_at_scoring_and_fit_does_not_modify_inputs(sample):
    x, target, valid, ids, labels = sample
    before = tuple(a.copy() for a in sample)
    fit = fit_density(x, target, valid, ids, labels)
    log_densities(fit, x, target, valid)
    for actual, original in zip(sample, before, strict=True):
        np.testing.assert_equal(actual, original)
    assert "labels" not in inspect.signature(log_densities).parameters
    assert "labels" not in inspect.signature(channel_log_densities).parameters


def test_fit_can_be_torch_weights_only_round_tripped_in_memory(sample):
    x, target, valid, ids, labels = sample
    fit = fit_density(x, target, valid, ids, labels)
    buffer = BytesIO()
    torch.save(fit, buffer)
    buffer.seek(0)
    reloaded = torch.load(buffer, map_location="cpu", weights_only=True)
    before, counts_before = log_densities(fit, x, target, valid)
    after, counts_after = log_densities(reloaded, x, target, valid)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(counts_before, counts_after)


def test_mixture_feedback_matches_autograd_and_detaches_density():
    z = torch.tensor([-7.0, -1.2, 0.0, 1.1, 8.0], dtype=torch.float64, requires_grad=True)
    logs = torch.tensor(
        [[-50.0, -49.0], [2.5, -0.5], [0.0, 2.0], [-10.0, -8.0], [0.5, 0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss, posterior, gradient = mixture_feedback(z, logs)
    automatic, density_gradient = torch.autograd.grad(
        loss.sum(), (z, logs), allow_unused=True
    )
    assert density_gradient is None
    torch.testing.assert_close(automatic, gradient, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(gradient, z.sigmoid() - posterior, rtol=0, atol=0)
    expected = -torch.logsumexp(
        torch.stack((torch.nn.functional.logsigmoid(-z), torch.nn.functional.logsigmoid(z)), 1)
        + logs.detach(),
        dim=1,
    )
    torch.testing.assert_close(loss, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equal_likelihoods_are_exactly_logit_independent(dtype):
    z = torch.tensor([-1000.0, -20.0, 0.0, 3.5, 1000.0], dtype=dtype, requires_grad=True)
    logs = torch.tensor(
        [[-1e6, -1e6], [0.0, 0.0], [1.0, 1.0], [-37.0, -37.0], [1e6, 1e6]],
        dtype=dtype,
    )
    loss, posterior, gradient = mixture_feedback(z, logs)
    automatic, = torch.autograd.grad(loss.sum(), z)
    torch.testing.assert_close(loss, -logs[:, 0], rtol=0, atol=1e-14)
    torch.testing.assert_close(posterior, z.sigmoid(), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(z), rtol=0, atol=0)
    torch.testing.assert_close(automatic, torch.zeros_like(z), rtol=0, atol=1e-14)


def test_class_exchange_preserves_loss_and_reverses_gradient():
    z = torch.tensor([-4.0, 0.0, 3.0], dtype=torch.float64)
    logs = torch.tensor([[1.0, -5.0], [-10.0, 4.0], [2.0, 2.0]], dtype=torch.float64)
    loss, posterior, gradient = mixture_feedback(z, logs)
    swapped_loss, swapped_posterior, swapped_gradient = mixture_feedback(-z, logs.flip(1))
    torch.testing.assert_close(swapped_loss, loss, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(swapped_posterior, 1 - posterior, rtol=0, atol=1e-15)
    torch.testing.assert_close(swapped_gradient, -gradient, rtol=0, atol=1e-15)


def test_extreme_finite_logits_and_log_densities_stay_finite():
    z = torch.tensor([-1000.0, 1000.0, -800.0, 800.0, 0.0], dtype=torch.float64, requires_grad=True)
    logs = torch.tensor(
        [[-1e6, 0.0], [0.0, -1e6], [-1e6, -999200.0], [-1e6, -1000800.0], [-1e6, -1e6]],
        dtype=torch.float64,
    )
    loss, posterior, gradient = mixture_feedback(z, logs)
    automatic, = torch.autograd.grad(loss.sum(), z)
    for value in (loss, posterior, gradient, automatic):
        assert torch.isfinite(value).all()
    torch.testing.assert_close(automatic, gradient, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("bad_labels", [[0] * 11 + [2], [0] * 11 + [np.nan]])
def test_fit_rejects_nonbinary_training_labels(sample, bad_labels):
    x, target, valid, ids, _ = sample
    with pytest.raises(ValueError):
        fit_density(x, target, valid, ids, np.asarray(bad_labels))
