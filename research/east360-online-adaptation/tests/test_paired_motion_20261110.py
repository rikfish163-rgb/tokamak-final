"""Synthetic, CPU-only contract tests for the paired motion views."""

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.commissioning_transfer_20261108 import contrast
from experiments.risk_aligned_online.paired_motion_20261110 import motion_views


def inputs(batch=2):
    rng = torch.Generator().manual_seed(931)
    center = torch.linspace(-4.0, 7.0, 23, dtype=torch.float64)
    scale = torch.linspace(0.4, 3.0, 23, dtype=torch.float64)
    past = center + scale * torch.randn(batch, 63, 23, generator=rng, dtype=torch.float64)
    truth = center + scale * torch.randn(batch, 25, 23, generator=rng, dtype=torch.float64)
    forecast = torch.randn(batch, 25, 23, generator=rng, dtype=torch.float64)
    past_available = torch.rand(batch, 63, 23, generator=rng) > 0.12
    observed = torch.rand(batch, 25, 23, generator=rng) > 0.12
    return [forecast, truth, observed, past, past_available, center, scale]


def numpy_reference(args):
    forecast, truth, observed, past, past_available, center, scale = args
    arrays = [x.detach().cpu().numpy() for x in args]
    _, y, observed_np, x, past_mask, c, s = arrays
    prediction = (center.double() + scale.double() * forecast.double().sinh()).detach().numpy()
    hold = np.zeros((x.shape[0], 25, 23), dtype=np.float64)
    for b in range(len(x)):
        for channel in range(23):
            indices = np.flatnonzero(past_mask[b, :, channel])
            if len(indices):
                hold[b, :, channel] = x[b, indices[-1], channel]

    outputs = [[], [], []]
    validity = []
    for b in range(len(x)):
        mask = np.concatenate((past_mask[b], observed_np[b]), axis=0)
        for bucket, future in zip(outputs, (prediction, y, hold)):
            values = np.concatenate((x[b], future[b]), axis=0)
            bucket.append(contrast(values, mask)[63:])
        validity.append(np.stack([
            (mask[t - 31:t + 1] & mask[t - 63:t - 31]).sum(axis=0) >= 16
            for t in range(63, 88)
        ]))
    return tuple(np.stack(x) for x in outputs), np.stack(validity)


def test_batched_views_match_existing_numpy_contrast_and_pair_validity():
    args = inputs()
    # Include entirely missing channels and channels that cross the 16-pair boundary.
    args[4][0, :, 3] = False
    args[2][1, ::2, 7] = False
    actual = motion_views(*args)
    expected, valid = numpy_reference(args)
    for result, reference in zip(actual[:3], expected):
        assert result.shape == (2, 25, 23)
        assert result.dtype == torch.float64
        np.testing.assert_allclose(result.detach().numpy(), reference, rtol=2e-7, atol=5e-8)
        assert torch.all(result.abs() <= 1)
    assert actual[3].shape == (2, 25, 23)
    assert actual[3].dtype == torch.bool
    np.testing.assert_array_equal(actual[3].numpy(), valid)


def test_exact_reference_values_for_linear_constant_and_persistent_paths():
    past = torch.arange(63, dtype=torch.float64)[None, :, None].expand(1, -1, 23).clone()
    past[:, :, 1] *= -1
    past[:, :, 2] = 5
    truth = torch.arange(63, 88, dtype=torch.float64)[None, :, None].expand(1, -1, 23).clone()
    truth[:, :, 1] *= -1
    truth[:, :, 2] = 5
    args = [truth.asinh(), truth, torch.ones_like(truth, dtype=torch.bool), past,
            torch.ones_like(past, dtype=torch.bool), torch.zeros(23), torch.ones(23)]
    actual = motion_views(*args)
    expected, valid = numpy_reference(args)
    for result, reference in zip(actual[:3], expected):
        np.testing.assert_array_equal(result.detach().numpy().astype(np.float32), reference)
    np.testing.assert_array_equal(actual[3].numpy(), valid)


def test_latest_observed_hold_and_masked_payloads_do_not_change_any_view():
    args = inputs()
    args[4][0, 52:, 0] = False
    args[4][0, :, 1] = False
    original = motion_views(*args)

    changed = [x.clone() for x in args]
    changed[0] = torch.where(changed[2], changed[0], torch.nan).requires_grad_()
    changed[1] = torch.where(changed[2], changed[1], torch.inf)
    changed[3] = torch.where(changed[4], changed[3], torch.nan)
    masked = motion_views(*changed)
    for before, after in zip(original, masked):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    masked[0].sum().backward()
    assert torch.isfinite(changed[0].grad).all()
    assert torch.count_nonzero(changed[0].grad[~changed[2]]) == 0
    assert not masked[3][0, :, 1].any()

    # A forecast holding the last available sample (not necessarily q) is the
    # persistence branch, under exactly the same future-observation mask.
    last = torch.zeros(2, 23, dtype=torch.float64)
    for b in range(2):
        for channel in range(23):
            indices = torch.nonzero(args[4][b, :, channel]).flatten()
            if len(indices):
                last[b, channel] = (args[3][b, indices[-1], channel] - args[5][channel]) / args[6][channel]
    hold_forecast = last.asinh()[:, None, :].expand(-1, 25, -1)
    predicted, _, held, _ = motion_views(hold_forecast, *args[1:])
    torch.testing.assert_close(predicted, held, rtol=1e-14, atol=1e-14)


def test_sixteen_observed_pairs_are_valid_but_fifteen_are_not_even_at_zero_energy():
    args = inputs(batch=1)
    args[0].zero_()
    args[1].zero_()
    args[3].zero_()
    args[5].zero_()
    args[6].fill_(1)
    args[2].fill_(True)
    args[4].fill_(False)
    args[4][0, :16, 0] = True
    args[4][0, 32:48, 0] = True
    args[4][0, :15, 1] = True
    args[4][0, 32:47, 1] = True
    pred, truth, hold, valid = motion_views(*args)
    assert valid[0, 0, 0]
    assert not valid[0, 0, 1]
    for view in (pred, truth, hold):
        assert view[0, 0, 0] == 0
        assert view[0, 0, 1] == 0


@pytest.mark.parametrize("missing", [False, True])
def test_constant_and_fully_missing_paths_have_finite_zero_gradients(missing):
    args = inputs(batch=1)
    args[0] = torch.zeros_like(args[0]).requires_grad_()
    args[3] = args[5].expand_as(args[3]).clone()
    args[1] = args[5].expand_as(args[1]).clone()
    args[2].fill_(not missing)
    args[4].fill_(not missing)
    if missing:
        args[0] = torch.full_like(args[0], float("nan"), requires_grad=True)
        args[1].fill_(float("nan"))
        args[3].fill_(float("nan"))
    pred, truth, hold, valid = motion_views(*args)
    for view in (pred, truth, hold):
        assert torch.isfinite(view).all()
        assert torch.count_nonzero(view) == 0
    assert bool(valid.all()) != missing
    pred.sum().backward()
    assert torch.isfinite(args[0].grad).all()
    assert torch.count_nonzero(args[0].grad) == 0


def test_non_degenerate_forecast_gradient_matches_finite_differences():
    args = inputs(batch=1)
    args[2].fill_(True)
    args[4].fill_(True)
    basis = torch.zeros(3, 25, 23, dtype=torch.float64)
    basis[0, 2, 0] = 1
    basis[1, 8, 0] = 1
    basis[2, 17, 0] = 1
    probes = torch.tensor([0.1, -0.15, 0.2], dtype=torch.float64, requires_grad=True)

    def selected(p):
        forecast = args[0] + (p[:, None, None] * basis).sum(dim=0)[None]
        return motion_views(forecast, *args[1:])[0][0, [5, 12, 20], 0]

    assert torch.autograd.gradcheck(selected, (probes,), eps=1e-6, atol=1e-6, rtol=1e-4)
    gradient, = torch.autograd.grad(selected(probes).sum(), probes)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 3


def test_earlier_endpoints_and_gradients_never_use_later_horizons():
    args = inputs(batch=1)
    args[0].requires_grad_()
    original = motion_views(*args)
    later = [x.detach().clone() for x in args]
    later[2][:, 7:] = False
    later[0][:, 7:] = float("nan")
    later[1][:, 7:] = float("nan")
    changed = motion_views(*later)
    for before, after in zip(original, changed):
        torch.testing.assert_close(before[:, :7], after[:, :7], rtol=0, atol=0)
    gradient, = torch.autograd.grad(original[0][:, :7].sum(), args[0])
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, 7:]) == 0
    assert torch.count_nonzero(gradient[~args[2]]) == 0


def test_sinh_is_computed_in_float64_before_differences():
    args = inputs(batch=1)
    args[0] = torch.zeros(1, 25, 23, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        args[0][0, 0, 0] = 90.0  # float32 sinh overflows, float64 sinh is finite
    args[2].fill_(True)
    pred, _, _, _ = motion_views(*args)
    assert torch.isfinite(pred).all()
    pred.sum().backward()
    assert torch.isfinite(args[0].grad).all()


@pytest.mark.parametrize("invalid", [
    "short_forecast", "short_truth", "short_past", "mask_shape", "center_shape",
    "scale_shape", "mask_dtype", "integer_values", "empty_batch", "non_tensor",
])
def test_invalid_shapes_and_types_are_rejected(invalid):
    args = inputs(batch=1)
    if invalid == "short_forecast":
        args[0] = args[0][:, :24]
    elif invalid == "short_truth":
        args[1] = args[1][:, :24]
    elif invalid == "short_past":
        args[3] = args[3][:, :62]
    elif invalid == "mask_shape":
        args[2] = args[2][:, :, :22]
    elif invalid == "center_shape":
        args[5] = args[5][:, None]
    elif invalid == "scale_shape":
        args[6] = args[6][:22]
    elif invalid == "mask_dtype":
        args[4] = args[4].int()
    elif invalid == "integer_values":
        args[3] = args[3].int()
    elif invalid == "empty_batch":
        args[:5] = [x[:0] for x in args[:5]]
    else:
        args[0] = args[0].numpy()
    with pytest.raises((TypeError, ValueError)):
        motion_views(*args)


@pytest.mark.parametrize("invalid", [
    "zero_scale", "negative_scale", "infinite_scale", "nan_center",
    "nan_observed_truth", "infinite_observed_past", "nan_observed_forecast",
    "sinh_overflow",
])
def test_invalid_observed_values_and_source_coordinates_are_rejected(invalid):
    args = inputs(batch=1)
    args[2].fill_(True)
    args[4].fill_(True)
    if invalid == "zero_scale":
        args[6][0] = 0
    elif invalid == "negative_scale":
        args[6][0] = -1
    elif invalid == "infinite_scale":
        args[6][0] = float("inf")
    elif invalid == "nan_center":
        args[5][0] = float("nan")
    elif invalid == "nan_observed_truth":
        args[1][0, 0, 0] = float("nan")
    elif invalid == "infinite_observed_past":
        args[3][0, 0, 0] = float("inf")
    elif invalid == "nan_observed_forecast":
        args[0][0, 0, 0] = float("nan")
    else:
        args[0][0, 0, 0] = 1000
    with pytest.raises(ValueError, match="finite|scale"):
        motion_views(*args)
