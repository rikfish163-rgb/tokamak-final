"""Synthetic CPU checks for the arrived-forecast gradient contract."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.simple23 import Config, Simple23
from paired_diagnostic_features_20261106 import GROUPS

import h25_arrived_gradient_20261106 as arrived


@pytest.fixture
def sample():
    torch.manual_seed(842)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.linspace(-0.7, 0.9, 23),
            torch.linspace(0.6, 1.7, 23),
        )
        .double()
        .eval()
        .requires_grad_(False)
    )
    x = torch.randn(2, 128, 23, dtype=torch.double)
    available = torch.rand(x.shape) > 0.2
    available[0, 16:24] = False
    available[1, 80:88] = False
    raw = torch.randn(2, 25, 23, dtype=torch.double)
    raw += torch.linspace(-1.2, 1.3, 25, dtype=torch.double)[None, :, None]
    valid = torch.rand(raw.shape) > 0.12
    return base, x, available, raw, valid


def _mask(raw, available):
    observed = available & torch.isfinite(raw)
    keep = observed.sum(dim=1) >= 13
    selected = torch.zeros(23, dtype=torch.bool, device=raw.device)
    for _, channels in GROUPS:
        selected[list(channels)] = True
    return observed & keep[:, None, :] & selected[None, None, :]


def _reverse(raw, mask):
    result = raw.clone()
    for b in range(raw.shape[0]):
        for channel in range(23):
            indices = torch.nonzero(mask[b, :, channel], as_tuple=True)[0]
            result[b, indices, channel] = raw[b, indices.flip(0), channel]
    return result


def _forecast(base, x, available, u):
    """Independent reconstruction of the specified first-norm write."""
    patches = available.reshape(len(x), 16, 8, 23).any(dim=(2, 3))
    route = torch.cat((patches, torch.ones_like(patches[:, :1])), dim=1)
    calls = []

    def write(norm, args, output):
        calls.append(1)
        gain, bias = u.chunk(2, dim=1)
        normalized = F.layer_norm(args[0], norm.normalized_shape, eps=norm.eps)
        return output + route[:, :, None] * (
            normalized * gain[:, None, :] + bias[:, None, :]
        )

    handle = base.encoder.layers[0].norm1.register_forward_hook(write)
    try:
        forecast = base(x, available)["forecast"]
    finally:
        handle.remove()
    assert calls == [1]
    return forecast


def _losses(base, forecast, raw, mask):
    safe = torch.where(mask, raw, base.center)
    target = torch.asinh((safe - base.center) / base.scale)
    prediction = torch.where(mask, forecast, torch.zeros_like(forecast))
    point_loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    point_loss = point_loss.masked_fill(~mask, 0.0)
    result = []
    for _, channels in GROUPS:
        count = mask[:, :, list(channels)].sum(dim=(1, 2))
        result.append(
            point_loss[:, :, list(channels)].sum(dim=(1, 2)) / count.clamp_min(1)
        )
    return torch.stack(result, dim=1)


def test_arrived_targets_exact_indices_and_prefix_poison():
    values = np.arange(90 * 23, dtype=np.float64).reshape(90, 23)
    available = np.ones_like(values, dtype=bool)
    available[12, 3] = False
    values[13, 4] = np.nan
    origins = np.array([0, 10, 20], dtype=np.int64)
    maturity = np.array([25, 37, 46], dtype=np.int64)
    raw, mask = arrived.arrived_targets(values, available, origins, maturity)
    assert raw.shape == mask.shape == (3, 25, 23)
    assert mask.dtype == np.bool_
    np.testing.assert_equal(raw[1], values[11:36])
    np.testing.assert_array_equal(
        mask[1], available[11:36] & np.isfinite(values[11:36])
    )
    assert not mask[1, 1, 3] and not mask[1, 2, 4]
    for b, (origin, mature) in enumerate(zip(origins, maturity, strict=True)):
        poisoned = values.copy()
        poisoned[mature + 1 :] = np.nan
        poisoned_available = available.copy()
        poisoned_available[mature + 1 :] = ~poisoned_available[mature + 1 :]
        full = arrived.arrived_targets(poisoned, poisoned_available, [origin], [mature])
        prefix = arrived.arrived_targets(
            values[: mature + 1], available[: mature + 1], [origin], [mature]
        )
        np.testing.assert_equal(full[0][0], raw[b])
        np.testing.assert_array_equal(full[1][0], mask[b])
        np.testing.assert_equal(full[0], prefix[0])
        np.testing.assert_array_equal(full[1], prefix[1])


@pytest.mark.parametrize(
    "origins,maturity",
    [([1], [25]), ([0], [40]), ([-1], [24]), ([0, 1], [25]), ([0.5], [25])],
)
def test_arrived_targets_rejects_unmatured_out_of_range_and_bad_indices(
    origins, maturity
):
    values = np.zeros((40, 23))
    with pytest.raises(ValueError):
        arrived.arrived_targets(
            values, np.ones_like(values, dtype=bool), origins, maturity
        )


def test_arrived_targets_rejects_bad_data_shapes_and_nonboolean_mask():
    values = np.zeros((26, 23))
    with pytest.raises(ValueError):
        arrived.arrived_targets(
            values[:, :22], np.ones((26, 22), dtype=bool), [0], [25]
        )
    with pytest.raises(ValueError):
        arrived.arrived_targets(values, np.ones_like(values), [0], [25])
    with pytest.raises(ValueError):
        arrived.arrived_targets(values, np.ones((25, 23), dtype=bool), [0], [25])


def test_actual_head_matches_family_losses_and_signed_vjps(sample, monkeypatch):
    base, x, available, raw, valid = sample
    raw[0, 3, 3] = float("nan")
    x[1, 8:16] = float("nan")
    available[1, 8:16] = True
    state = {name: value.clone() for name, value in base.state_dict().items()}
    head_calls = []
    original_routes = arrived.latent.routes_all
    route_masks = []

    def capture_route(mask):
        route_masks.append(mask.clone())
        return original_routes(mask)

    def capture_head(*_):
        head_calls.append(
            tuple(len(block.norm1._forward_hooks) for block in base.encoder.layers)
        )

    monkeypatch.setattr(arrived.latent, "routes_all", capture_route)
    handle = base.forecast_head.register_forward_hook(capture_head)
    try:
        result = arrived.gradients(base, x, available, raw.numpy(), valid.numpy())
    finally:
        handle.remove()
    assert head_calls == [(1, 0)]
    assert len(route_masks) == 1 and torch.equal(route_masks[0], available)
    assert set(result) == {"true", "reverse", "groups", "mask", "loss"}
    assert result["true"].shape == result["reverse"].shape == (2, 3, 128)
    assert result["loss"].shape == (2, 2, 3)
    assert result["groups"].shape == (2, 3) and result["groups"].dtype == torch.bool
    assert result["mask"].shape == (2, 25, 23) and result["mask"].dtype == torch.bool
    assert not any(value.requires_grad for value in result.values())
    expected_mask = _mask(raw, valid)
    assert torch.equal(result["mask"], expected_mask)
    u = torch.zeros(2, 128, dtype=x.dtype, requires_grad=True)
    forecast = _forecast(base, x, available, u)
    assert forecast.shape == (2, 25, 23)
    for arm, target in enumerate((raw, _reverse(raw, expected_mask))):
        losses = _losses(base, forecast, target, expected_mask)
        torch.testing.assert_close(result["loss"][:, arm], losses.detach())
        for group in range(3):
            (expected,) = torch.autograd.grad(
                losses[:, group].sum(), u, retain_graph=True
            )
            name = ("true", "reverse")[arm]
            torch.testing.assert_close(
                result[name][:, group], expected, atol=1e-11, rtol=1e-9
            )
    assert result["groups"].all()
    assert result["true"].norm() > 0
    assert not torch.allclose(result["true"], result["reverse"])
    assert all(
        torch.equal(value, state[name]) for name, value in base.state_dict().items()
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in base.parameters()
    )
    assert not base.encoder.layers[0].norm1._forward_hooks
    assert not base.encoder.layers[1].norm1._forward_hooks


def test_family_reduction_reconstructs_direct_global_forecast_loss_and_gradient(sample):
    base, x, available, raw, valid = sample
    result = arrived.gradients(base, x, available, raw, valid)
    mask = _mask(raw, valid)
    counts = torch.stack([mask[:, :, list(ch)].sum((1, 2)) for _, ch in GROUPS], dim=1)
    weights = counts / counts.sum()
    u = torch.zeros(2, 128, dtype=x.dtype, requires_grad=True)
    forecast = _forecast(base, x, available, u)
    direct = base.forecast_loss(forecast, raw, mask)
    (direct_gradient,) = torch.autograd.grad(direct, u)
    torch.testing.assert_close((result["loss"][:, 0] * weights).sum(), direct)
    torch.testing.assert_close(
        (result["true"] * weights[:, :, None]).sum(1), direct_gradient
    )


def test_float32_model_accepts_numpy_float64_targets(sample):
    base, x, available, raw, valid = sample
    base = base.float()
    x = x.float()
    result = arrived.gradients(base, x, available, raw.numpy(), valid.numpy())
    assert result["true"].dtype == result["reverse"].dtype == torch.float32
    assert result["loss"].dtype == torch.float32
    mask = _mask(raw, valid)
    counts = torch.stack([mask[:, :, list(ch)].sum((1, 2)) for _, ch in GROUPS], dim=1)
    weights = counts / counts.sum()
    u = torch.zeros(2, 128, dtype=x.dtype, requires_grad=True)
    direct = base.forecast_loss(_forecast(base, x, available, u), raw.float(), mask)
    (reference,) = torch.autograd.grad(direct, u)
    torch.testing.assert_close(
        (result["loss"][:, 0] * weights).sum(), direct, atol=2e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        (result["true"] * weights[:, :, None]).sum(1), reference, atol=2e-7, rtol=1e-5
    )


def test_reversal_preserves_irregular_mask_and_observed_channel_multisets(sample):
    base, x, available, raw, valid = sample
    valid[:] = False
    raw[:] = 0
    for b in range(2):
        for channel in (3, 14, 19):
            positions = torch.tensor([0, 1, 3, 5, 6, 8, 9, 12, 14, 15, 19, 21, 24])
            valid[b, positions, channel] = True
            raw[b, positions, channel] = (
                torch.arange(13, dtype=raw.dtype) + b + channel / 10
            )
    mask = _mask(raw, valid)
    reversed_raw = _reverse(raw, mask)
    for b in range(2):
        for channel in (3, 14, 19):
            values = raw[b, :, channel][mask[b, :, channel]]
            control = reversed_raw[b, :, channel][mask[b, :, channel]]
            assert torch.equal(control, values.flip(0))
            assert torch.equal(control.sort().values, values.sort().values)
    result = arrived.gradients(base, x, available, raw, valid)
    independent_control = arrived.gradients(base, x, available, reversed_raw, valid)
    assert torch.equal(result["mask"], mask)
    torch.testing.assert_close(result["reverse"], independent_control["true"])
    torch.testing.assert_close(result["loss"][:, 1], independent_control["loss"][:, 0])
    assert not torch.allclose(result["reverse"], -result["true"])


def test_threshold_missing_groups_and_nonfinite_unobserved_values(sample):
    base, x, available, raw, valid = sample
    raw[:] = float("nan")
    valid[:] = True
    raw[0, :12, 3] = 0.2
    raw[0, :13, 14] = 0.4
    raw[0, :, 0] = 0.9  # Not a member of any diagnostic family.
    raw[1, :14, 19] = 0.8
    raw[1, 4, 19] = float("inf")
    valid[0, 0, 14] = False  # Exactly twelve finite observed values.
    result = arrived.gradients(base, x, available, raw, valid)
    assert torch.equal(result["mask"], _mask(raw, valid))
    assert result["groups"].tolist() == [[False, False, False], [False, False, True]]
    assert result["mask"][1, :, 19].sum() == 13
    assert not result["mask"][:, :, 0].any()
    absent = ~result["groups"]
    assert (result["true"][absent] == 0).all()
    assert (result["reverse"][absent] == 0).all()
    assert (result["loss"].transpose(1, 2)[absent] == 0).all()
    assert all(torch.isfinite(result[k]).all() for k in ("true", "reverse", "loss"))


@pytest.mark.parametrize("arm", [0, 1])
def test_directional_derivative_by_finite_difference(sample, arm):
    base, x, available, raw, valid = (
        item[:1] if isinstance(item, torch.Tensor) else item for item in sample
    )
    result = arrived.gradients(base, x, available, raw, valid)
    mask = _mask(raw, valid)
    target = raw if arm == 0 else _reverse(raw, mask)
    direction = torch.linspace(-0.4, 0.7, 128, dtype=x.dtype)[None, :]
    eps = 1e-5
    with torch.no_grad():
        plus = _losses(
            base, _forecast(base, x, available, eps * direction), target, mask
        )
        minus = _losses(
            base, _forecast(base, x, available, -eps * direction), target, mask
        )
    numeric = (plus - minus) / (2 * eps)
    analytic = (result[("true", "reverse")[arm]] * direction[:, None, :]).sum(dim=2)
    torch.testing.assert_close(analytic, numeric, rtol=2e-6, atol=2e-8)


def test_invalid_inputs_do_not_leave_hooks_and_unfrozen_model_is_rejected(sample):
    base, x, available, raw, valid = sample
    with pytest.raises(ValueError):
        arrived.gradients(base, x, available, raw[:, :24], valid[:, :24])
    with pytest.raises(ValueError):
        arrived.gradients(base, x, available.float(), raw, valid)
    with pytest.raises(ValueError):
        arrived.gradients(base, x, available, raw, valid.float())
    base.forecast_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        arrived.gradients(base, x, available, raw, valid)
    assert base.forecast_head.weight.requires_grad
    assert not base.encoder.layers[0].norm1._forward_hooks


def test_gradients_work_inside_no_grad_without_writing_to_base(sample):
    base, x, available, raw, valid = sample
    before = {name: value.clone() for name, value in base.state_dict().items()}
    with torch.no_grad():
        native = base(x, available)["forecast"].clone()
        first = arrived.gradients(base, x, available, raw, valid)
        second = arrived.gradients(base, x, available, raw, valid)
        after = base(x, available)["forecast"]
    torch.testing.assert_close(native, after, rtol=0, atol=0)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert all(
        torch.equal(before[name], value) for name, value in base.state_dict().items()
    )


def test_full_and_poisoned_prefix_produce_identical_gradient(sample):
    base, _, _, _, _ = sample
    rng = np.random.default_rng(92)
    values = rng.normal(size=(220, 23))
    available = rng.random(values.shape) > 0.1
    origin, maturity = 150, 180
    x = torch.from_numpy(values[origin - 127 : origin + 1].copy()).unsqueeze(0)
    origin_mask = torch.from_numpy(
        available[origin - 127 : origin + 1].copy()
    ).unsqueeze(0)
    prefix = arrived.arrived_targets(
        values[: maturity + 1], available[: maturity + 1], [origin], [maturity]
    )
    poisoned = values.copy()
    poisoned[origin + 26 :] = np.nan
    poisoned_available = available.copy()
    poisoned_available[origin + 26 :] = ~poisoned_available[origin + 26 :]
    full = arrived.arrived_targets(poisoned, poisoned_available, [origin], [maturity])
    result_prefix = arrived.gradients(base, x, origin_mask, *prefix)
    result_poisoned = arrived.gradients(base, x, origin_mask, *full)
    for key in result_prefix:
        torch.testing.assert_close(
            result_prefix[key], result_poisoned[key], rtol=0, atol=0
        )


def test_failing_forward_removes_only_temporary_hook(sample):
    base, x, available, raw, valid = sample
    block = base.encoder.layers[0].norm1
    prior = block.register_forward_hook(lambda _norm, _args, output: output)

    def fail(*_):
        raise RuntimeError("synthetic head failure")

    failure = base.forecast_head.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="synthetic head failure"):
            arrived.gradients(base, x, available, raw, valid)
        assert list(block._forward_hooks) == [prior.id]
        assert not base.encoder.layers[1].norm1._forward_hooks
    finally:
        failure.remove()
        prior.remove()


def test_empty_batch_keeps_documented_shapes(sample):
    base, x, available, raw, valid = sample
    result = arrived.gradients(base, x[:0], available[:0], raw[:0], valid[:0])
    assert result["true"].shape == result["reverse"].shape == (0, 3, 128)
    assert result["groups"].shape == (0, 3)
    assert result["mask"].shape == (0, 25, 23)
    assert result["loss"].shape == (0, 2, 3)
    indices = np.empty(0, dtype=np.int64)
    targets, mask = arrived.arrived_targets(
        np.empty((0, 23)), np.empty((0, 23), dtype=bool), indices, indices
    )
    assert targets.shape == mask.shape == (0, 25, 23)
