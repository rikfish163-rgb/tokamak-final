"""CPU-only synthetic tests of the frozen H25 lag-8 increment helper."""

import hashlib

import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.simple23 import Config, Simple23
from paired_diagnostic_features_20261106 import GROUPS

import h25_arrived_gradient_20261106 as arrived
import h25_increment_gradient_20261106 as increment


@pytest.fixture
def sample():
    torch.manual_seed(2286)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.linspace(-0.8, 0.7, 23),
            torch.linspace(0.6, 1.8, 23),
        )
        .double()
        .eval()
        .requires_grad_(False)
    )
    x = torch.randn(2, 128, 23, dtype=torch.double)
    available = torch.rand(x.shape) > 0.18
    available[0, 16:24] = False
    x[1, 8:16] = float("nan")
    available[1, 8:16] = True
    time = torch.linspace(-1.3, 1.5, 25, dtype=torch.double)
    raw = torch.randn(2, 25, 23, dtype=torch.double) * 0.2 + time[None, :, None]
    valid = torch.rand(raw.shape) > 0.12
    return base, x, available, raw, valid


def _level_mask(raw, available):
    observed = available & torch.isfinite(raw)
    selected = torch.zeros(23, dtype=torch.bool, device=raw.device)
    for _, channels in GROUPS:
        selected[list(channels)] = True
    return observed & (observed.sum(1) >= 13)[:, None, :] & selected[None, None, :]


def _reverse_observed(raw, mask):
    reversed_raw = raw.clone()
    for batch in range(len(raw)):
        for channel in range(raw.shape[-1]):
            positions = torch.nonzero(mask[batch, :, channel], as_tuple=True)[0]
            reversed_raw[batch, positions, channel] = raw[
                batch, positions.flip(0), channel
            ]
    return reversed_raw


def _forecast(base, x, available, u):
    """Reconstruct the existing effective-state write independently."""
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
        output = base(x, available)["forecast"]
    finally:
        handle.remove()
    assert calls == [1]
    return output


def _reference_losses(base, forecast, raw, level_mask):
    """Explicitly enumerate selected endpoint pairs, independently of the helper."""
    rows = []
    for batch in range(len(raw)):
        families = []
        for _, channels in GROUPS:
            predicted, target = [], []
            for channel in channels:
                for left in range(17):
                    right = left + 8
                    if level_mask[batch, left, channel] and level_mask[batch, right, channel]:
                        predicted.append(
                            forecast[batch, right, channel] - forecast[batch, left, channel]
                        )
                        hi = torch.asinh(
                            (raw[batch, right, channel] - base.center[channel])
                            / base.scale[channel]
                        )
                        lo = torch.asinh(
                            (raw[batch, left, channel] - base.center[channel])
                            / base.scale[channel]
                        )
                        target.append(hi.to(forecast.dtype) - lo.to(forecast.dtype))
            if predicted:
                families.append(
                    F.smooth_l1_loss(torch.stack(predicted), torch.stack(target), beta=1.0)
                )
            else:
                families.append(forecast[batch].sum() * 0.0)
        rows.append(torch.stack(families))
    return torch.stack(rows)


def _checksum(base):
    digest = hashlib.sha256()
    for name, tensor in base.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_original_head_hook_loss_mask_and_vjps_match_independent_reference(sample):
    base, x, available, raw, valid = sample
    raw[0, 3, 3] = float("nan")
    expected_level = _level_mask(raw, valid)
    expected_pairs = expected_level[:, 8:] & expected_level[:, :-8]
    base.forecast_head.weight.grad = torch.full_like(base.forecast_head.weight, 0.125)
    before_grad = base.forecast_head.weight.grad.clone()
    before_checksum = _checksum(base)
    before_flags = [module.training for module in base.modules()]
    before_requires = [p.requires_grad for p in base.parameters()]
    with torch.no_grad():
        original = base(x, available)["forecast"].clone()
    calls = []
    captured = []
    head_hook = base.forecast_head.register_forward_hook(lambda *_: calls.append(1))
    output_hook = base.register_forward_hook(
        lambda _module, _args, output: captured.append(output["forecast"].detach().clone())
    )
    try:
        with torch.no_grad():
            result = increment.gradients(base, x, available, raw.numpy(), valid.numpy())
    finally:
        output_hook.remove()
        head_hook.remove()

    assert calls == [1] and len(captured) == 1
    # Registering a hook can change PyTorch's encoder kernel dispatch.
    torch.testing.assert_close(captured[0], original, rtol=1e-13, atol=1e-14)
    with torch.no_grad():
        after = base(x, available)["forecast"]
    torch.testing.assert_close(after, original, rtol=0, atol=0)
    assert set(result) == {"true", "reverse", "groups", "mask", "loss"}
    assert result["true"].shape == result["reverse"].shape == (2, 3, 128)
    assert result["groups"].shape == (2, 3) and result["groups"].dtype == torch.bool
    assert result["mask"].shape == (2, 17, 23) and result["mask"].dtype == torch.bool
    assert result["loss"].shape == (2, 2, 3)
    assert all(not value.requires_grad for value in result.values())
    assert torch.equal(result["mask"], expected_pairs)
    old_mask = arrived.gradients(base, x, available, raw, valid)["mask"]
    assert torch.equal(old_mask, expected_level)
    assert torch.equal(result["mask"], old_mask[:, 8:] & old_mask[:, :-8])
    expected_groups = torch.stack(
        [expected_pairs[:, :, list(channels)].any((1, 2)) for _, channels in GROUPS], 1
    )
    assert torch.equal(result["groups"], expected_groups)

    u = torch.zeros(2, 128, dtype=x.dtype, requires_grad=True)
    forecast = _forecast(base, x, available, u)
    for arm, target in enumerate((raw, _reverse_observed(raw, expected_level))):
        losses = _reference_losses(base, forecast, target, expected_level)
        torch.testing.assert_close(
            result["loss"][:, arm], losses.detach(), rtol=1e-11, atol=1e-12
        )
        for group in range(3):
            (expected,) = torch.autograd.grad(losses[:, group].sum(), u, retain_graph=True)
            torch.testing.assert_close(
                result[("true", "reverse")[arm]][:, group], expected, rtol=1e-9, atol=1e-11
            )

    assert result["groups"].all()
    assert result["true"].norm() > 0
    assert not torch.allclose(result["true"], result["reverse"])
    assert _checksum(base) == before_checksum
    assert [module.training for module in base.modules()] == before_flags
    assert [p.requires_grad for p in base.parameters()] == before_requires
    torch.testing.assert_close(
        base.forecast_head.weight.grad, before_grad, rtol=0, atol=0
    )
    assert all(
        p.grad is None for p in base.parameters() if p is not base.forecast_head.weight
    )
    assert not base.encoder.layers[0].norm1._forward_hooks
    assert not base.encoder.layers[1].norm1._forward_hooks


@pytest.mark.parametrize("arm", [0, 1])
def test_gradient_matches_central_finite_difference(sample, arm):
    base, x, available, raw, valid = sample
    x, available, raw, valid = x[:1], available[:1], raw[:1], valid[:1]
    result = increment.gradients(base, x, available, raw, valid)
    level_mask = _level_mask(raw, valid)
    target = raw if arm == 0 else _reverse_observed(raw, level_mask)
    direction = torch.linspace(-0.4, 0.7, 128, dtype=x.dtype)[None, :]
    epsilon = 1e-5
    with torch.no_grad():
        plus = _reference_losses(
            base, _forecast(base, x, available, epsilon * direction), target, level_mask
        )
        minus = _reference_losses(
            base, _forecast(base, x, available, -epsilon * direction), target, level_mask
        )
    numeric = (plus - minus) / (2 * epsilon)
    analytic = (result[("true", "reverse")[arm]] * direction[:, None, :]).sum(2)
    torch.testing.assert_close(analytic, numeric, rtol=3e-6, atol=3e-8)


def test_loss_stage_is_invariant_to_constant_shifts_in_normalized_levels(sample):
    base, _, _, raw, valid = sample
    mask = _level_mask(raw, valid)
    forecast = torch.randn_like(raw, requires_grad=True)
    shifts = torch.linspace(-0.7, 0.9, 23, dtype=raw.dtype)[None, None, :]
    # Shift each already-normalized target level, then map it back to raw units.
    normalized = torch.asinh((raw - base.center) / base.scale)
    shifted_raw = base.center + base.scale * torch.sinh(normalized + shifts)
    expected = increment._family_losses(base, forecast, raw, mask)
    shifted_prediction = increment._family_losses(base, forecast + 2 * shifts, raw, mask)
    shifted_target = increment._family_losses(base, forecast, shifted_raw, mask)
    torch.testing.assert_close(shifted_prediction, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(shifted_target, expected, rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(expected.sum(), forecast, retain_graph=True)[0]
    for shifted in (shifted_prediction, shifted_target):
        actual = torch.autograd.grad(shifted.sum(), forecast, retain_graph=True)[0]
        torch.testing.assert_close(actual, gradient, rtol=1e-12, atol=1e-12)


def test_threshold_nonfinite_mask_pairs_and_zero_pair_family(sample):
    base, x, available, raw, valid = sample
    raw[:] = float("nan")
    valid[:] = True
    raw[0, :12, 3] = 0.2  # Too few observed levels.
    raw[0, :13, 14] = torch.arange(13, dtype=raw.dtype) / 10  # Five valid pairs.
    raw[0, [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20], 19] = 0.3
    raw[1, :14, 19] = 0.4
    raw[1, 4, 19] = float("inf")  # Thirteen finite levels, five valid pairs.
    raw[:, :, 0] = 9.0  # Not in a diagnostic family.
    result = increment.gradients(base, x, available, raw, valid)
    level = _level_mask(raw, valid)
    assert level[0, :, 19].sum() == 13
    expected_pairs = level[:, 8:] & level[:, :-8]
    assert torch.equal(result["mask"], expected_pairs)
    assert result["mask"][0, :, 14].sum() == 5
    assert result["mask"][1, :, 19].sum() == 5
    assert not result["mask"][0, :, 19].any()
    assert not result["mask"][:, :, 0].any()
    assert result["groups"].tolist() == [[False, True, False], [False, False, True]]
    absent = ~result["groups"]
    assert (result["true"][absent] == 0).all()
    assert (result["reverse"][absent] == 0).all()
    assert (result["loss"].transpose(1, 2)[absent] == 0).all()
    assert all(torch.isfinite(result[key]).all() for key in ("true", "reverse", "loss"))


def test_irregular_reversal_preserves_observed_order_and_pair_mask(sample):
    base, x, available, raw, valid = sample
    raw.zero_()
    valid.zero_()
    positions = torch.tensor([0, 1, 3, 5, 6, 8, 9, 12, 14, 15, 19, 21, 24])
    for batch in range(2):
        for channel in (3, 14, 19):
            valid[batch, positions, channel] = True
            raw[batch, positions, channel] = (
                torch.arange(13, dtype=raw.dtype) + batch + channel / 10
            )
    mask = _level_mask(raw, valid)
    reversed_raw = _reverse_observed(raw, mask)
    for batch in range(2):
        for channel in (3, 14, 19):
            assert torch.equal(
                reversed_raw[batch, positions, channel],
                raw[batch, positions.flip(0), channel],
            )
    result = increment.gradients(base, x, available, raw, valid)
    control = increment.gradients(base, x, available, reversed_raw, valid)
    assert torch.equal(result["mask"], control["mask"])
    torch.testing.assert_close(result["reverse"], control["true"], rtol=0, atol=0)
    torch.testing.assert_close(
        result["loss"][:, 1], control["loss"][:, 0], rtol=0, atol=0
    )
    assert not torch.allclose(result["true"], result["reverse"])


def test_unobserved_targets_and_origins_do_not_contaminate_results(sample):
    base, x, available, raw, valid = sample
    valid[0, :, 4] = False
    valid[0, :12, 4] = True  # Observed but ineligible channel.
    level = _level_mask(raw, valid)
    poisoned_raw = raw.clone()
    poisoned_raw[~level] = float("nan")
    poisoned_x = x.clone()
    poisoned_x[~available] = float("inf")
    result = increment.gradients(base, x, available, raw, valid)
    poisoned = increment.gradients(base, poisoned_x, available, poisoned_raw, valid)
    for key in result:
        torch.testing.assert_close(poisoned[key], result[key], rtol=0, atol=0)


def test_all_groups_absent_with_no_available_origin(sample):
    base, x, available, raw, valid = sample
    available.zero_()
    valid.zero_()
    raw.fill_(float("nan"))
    result = increment.gradients(base, x, available, raw, valid)
    assert not result["groups"].any()
    assert not result["mask"].any()
    assert not result["true"].any()
    assert not result["reverse"].any()
    assert not result["loss"].any()


def test_beta_one_loss_averages_all_valid_pairs_in_family(sample):
    base, _, _, raw, _ = sample
    raw = raw[:1].clone()
    mask = torch.zeros_like(raw, dtype=torch.bool)
    mask[0, :13, 3] = True
    mask[0, :16, 4] = True
    normalized = torch.arange(25, dtype=raw.dtype).square() / 40
    for channel in (3, 4):
        raw[0, :, channel] = (
            base.center[channel] + base.scale[channel] * torch.sinh(normalized)
        )
    forecast = torch.zeros_like(raw)
    forecast[0, :, 3] = torch.arange(25, dtype=raw.dtype) * 0.1
    forecast[0, :, 4] = torch.arange(25, dtype=raw.dtype) * -0.2
    residuals = []
    for channel, stop in ((3, 13), (4, 16)):
        for left in range(stop - 8):
            error = (forecast[0, left + 8, channel] - forecast[0, left, channel]) - (
                normalized[left + 8] - normalized[left]
            )
            residuals.append(error.abs())
    residuals = torch.stack(residuals)
    assert (residuals < 1).any() and (residuals > 1).any()
    expected = torch.where(
        residuals < 1, residuals.square() / 2, residuals - 0.5
    ).mean()
    actual = increment._family_losses(base, forecast, raw, mask)
    torch.testing.assert_close(actual[0, 0], expected, rtol=1e-12, atol=1e-12)
    assert not actual[0, 1:].any()


def test_masked_endpoint_forecasts_do_not_contaminate_loss_or_gradients(sample):
    base, _, _, raw, valid = sample
    level = _level_mask(raw, valid)
    pairs = level[:, 8:] & level[:, :-8]
    used = torch.zeros_like(level)
    used[:, 8:] |= pairs
    used[:, :-8] |= pairs
    forecast = torch.randn_like(raw, requires_grad=True)
    clean = increment._family_losses(base, forecast, raw, level)
    expected_grad = torch.autograd.grad(clean.sum(), forecast)[0]
    poisoned = forecast.detach().clone()
    poisoned[~used] = float("nan")
    poisoned.requires_grad_()
    actual = increment._family_losses(base, poisoned, raw, level)
    actual_grad = torch.autograd.grad(actual.sum(), poisoned)[0]
    torch.testing.assert_close(actual, clean, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert torch.isfinite(actual_grad).all()


def test_float32_numpy_targets_and_empty_batch(sample):
    base, x, available, raw, valid = sample
    base = base.float()
    x = x.float()
    result = increment.gradients(base, x, available, raw.numpy(), valid.numpy())
    assert (
        result["true"].dtype
        == result["reverse"].dtype
        == result["loss"].dtype
        == torch.float32
    )
    level = _level_mask(raw, valid)
    u = torch.zeros(2, 128, dtype=x.dtype, requires_grad=True)
    reference = _reference_losses(base, _forecast(base, x, available, u), raw, level)
    torch.testing.assert_close(result["loss"][:, 0], reference, rtol=2e-6, atol=2e-7)
    for group in range(3):
        expected = torch.autograd.grad(reference[:, group].sum(), u, retain_graph=True)[0]
        torch.testing.assert_close(
            result["true"][:, group], expected, rtol=1e-5, atol=3e-7
        )
    empty = increment.gradients(base, x[:0], available[:0], raw[:0], valid[:0])
    assert empty["true"].shape == empty["reverse"].shape == (0, 3, 128)
    assert empty["groups"].shape == (0, 3)
    assert empty["mask"].shape == (0, 17, 23)
    assert empty["loss"].shape == (0, 2, 3)


def test_failure_removes_only_its_hook_and_does_not_mutate_base(sample):
    base, x, available, raw, valid = sample
    block = base.encoder.layers[0].norm1
    prior = block.register_forward_hook(lambda _module, _args, output: output)
    failure = base.forecast_head.register_forward_hook(
        lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic forward failure"))
    )
    checksum = _checksum(base)
    flags = [p.requires_grad for p in base.parameters()]
    try:
        with pytest.raises(RuntimeError, match="synthetic forward failure"):
            increment.gradients(base, x, available, raw, valid)
        assert list(block._forward_hooks) == [prior.id]
        assert not base.encoder.layers[1].norm1._forward_hooks
        assert _checksum(base) == checksum
        assert [p.requires_grad for p in base.parameters()] == flags
    finally:
        failure.remove()
        prior.remove()


def test_nonfinite_active_pair_is_rejected_after_removing_hook(sample):
    base, x, available, raw, valid = sample
    valid[:] = True

    def poison_head(_module, _args, output):
        result = output.clone()
        result[0, 3] = float("nan")  # First level of an active lag-8 pair.
        return result

    handle = base.forecast_head.register_forward_hook(poison_head)
    checksum = _checksum(base)
    try:
        with pytest.raises(FloatingPointError, match="observed target pair"):
            increment.gradients(base, x, available, raw, valid)
        assert not base.encoder.layers[0].norm1._forward_hooks
        assert _checksum(base) == checksum
    finally:
        handle.remove()


def test_bad_inputs_and_unfrozen_model_are_rejected_without_mutation(sample):
    base, x, available, raw, valid = sample
    with pytest.raises(ValueError):
        increment.gradients(base, x, available, raw[:, :24], valid[:, :24])
    with pytest.raises(ValueError):
        increment.gradients(base, x, available.float(), raw, valid)
    with pytest.raises(ValueError):
        increment.gradients(base, x, available, raw, valid.float())
    base.forecast_head.weight.requires_grad_(True)
    checksum = _checksum(base)
    with pytest.raises(ValueError, match="frozen"):
        increment.gradients(base, x, available, raw, valid)
    assert base.forecast_head.weight.requires_grad
    assert _checksum(base) == checksum
    assert not base.encoder.layers[0].norm1._forward_hooks
