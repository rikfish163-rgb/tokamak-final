import math

import pytest
import torch

from experiments.risk_aligned_online.adaptivity_capacity_20261102 import geometry
from experiments.risk_aligned_online.dual_history_ttt_20261101.model import project_gates


@pytest.fixture
def pack():
    generator = torch.Generator().manual_seed(20261102)
    valid = torch.rand((6, 17), generator=generator) > 0.3
    valid[0, :16] = False
    valid[0, 16] = True  # CLS is not a data token.
    return {
        "h": torch.randn(6, 17, 64, generator=generator),
        "valid": valid,
        "long": torch.randn(6, 16, 64, generator=generator),
        "snapshot": torch.randn(6, 16, 64, generator=generator),
        "phi": torch.randn(6, 16, 4, generator=generator),
    }


@pytest.fixture
def enabled():
    return torch.tensor([True, False, True, True, False, True])


def make_state(family, *, dtype=torch.float32):
    generator = torch.Generator().manual_seed(42)
    shape = (16, 2) if family == "gates" else (4, 64)
    return torch.randn(shape, generator=generator, dtype=dtype)


def direct_shift(pack, state, family):
    if family == "gates":
        raw = (
            state[None, :, :1] * pack["long"]
            + state[None, :, 1:] * pack["snapshot"]
        )
    else:
        raw = pack["phi"] @ state
    return raw.masked_fill(~pack["valid"][:, :16, None], 0)


def direct_max_rms(pack, state, family, enabled):
    delta = direct_shift(
        {key: value.double() if key != "valid" else value for key, value in pack.items()},
        state.double(),
        family,
    )[enabled]
    denom = (pack["valid"][enabled, :16].sum(dim=1) * 64).clamp_min(1)
    if not len(delta):
        return state.new_zeros((), dtype=torch.float64)
    return (delta.square().sum(dim=(1, 2)) / denom).sqrt().max()


@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_shift_and_metric_match_direct_masked_rms(pack, enabled, family):
    state = make_state(family)
    delta = geometry.shift(pack, state, family)
    assert delta.shape == (6, 16, 64)
    torch.testing.assert_close(delta, direct_shift(pack, state, family))
    assert torch.count_nonzero(delta[~pack["valid"][:, :16]]) == 0

    metric = geometry.build_metric(pack, enabled)
    assert set(metric) == {"gate_gram", "lora_gram", "denom"}
    assert metric["gate_gram"].shape == (4, 16, 2, 2)
    assert metric["lora_gram"].shape == (4, 4, 4)
    assert metric["denom"].shape == (4,)
    assert all(value.dtype == torch.float64 for value in metric.values())
    assert metric["denom"][0] == 1  # No valid data even though CLS is valid.
    rms = geometry.max_rms(metric, state, family)
    assert rms.shape == () and rms.dtype == torch.float64
    torch.testing.assert_close(rms, direct_max_rms(pack, state, family, enabled))


def test_metric_contains_enabled_masked_per_token_grams(pack, enabled):
    metric = geometry.build_metric(pack, enabled)
    for j, i in enumerate(enabled.nonzero(as_tuple=True)[0]):
        mask = pack["valid"][i, :16]
        for k in range(16):
            views = torch.stack((pack["long"][i, k], pack["snapshot"][i, k])).double()
            expected = views @ views.T if mask[k] else torch.zeros(2, 2, dtype=torch.float64)
            torch.testing.assert_close(metric["gate_gram"][j, k], expected)
        phi = pack["phi"][i, :16][mask].double()
        torch.testing.assert_close(metric["lora_gram"][j], phi.T @ phi)
        assert metric["denom"][j] == max(1, int(mask.sum()) * 64)


@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_invalid_padding_cannot_overflow_shift(pack, family):
    largest = torch.finfo(torch.float32).max
    for name in ("long", "snapshot", "phi"):
        pack[name][0] = largest
    state = torch.full_like(make_state(family), 2, requires_grad=True)
    delta = geometry.shift(pack, state, family)
    assert torch.count_nonzero(delta[0]) == 0
    delta.sum().backward()
    assert torch.isfinite(state.grad).all()


@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_projection_is_feasible_bounded_idempotent_and_does_not_mutate(pack, enabled, family):
    metric = geometry.build_metric(pack, enabled)
    original = make_state(family) * 8
    before = original.clone()
    projected = geometry.project_state(metric, original, family)
    assert torch.equal(original, before)
    assert geometry.max_rms(metric, projected, family) <= 0.25 + 1e-7
    assert direct_max_rms(pack, projected, family, enabled) <= 0.25 + 1e-7
    torch.testing.assert_close(geometry.project_state(metric, projected, family), projected)
    if family == "gates":
        assert (projected >= 0).all()
        assert (projected.sum(dim=1) <= 1 + 1e-7).all()
        before_scaling = project_gates(original)
    else:
        before_scaling = original
    rms = geometry.max_rms(metric, before_scaling, family)
    expected = before_scaling * min(1.0, 0.25 / float(rms)) if rms else before_scaling
    torch.testing.assert_close(projected, expected)


@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_zero_state_and_empty_or_invalid_endpoints(pack, family):
    zero = torch.zeros_like(make_state(family), requires_grad=True)
    none_enabled = torch.zeros(6, dtype=torch.bool)
    metric = geometry.build_metric(pack, none_enabled)
    assert metric["gate_gram"].shape == (0, 16, 2, 2)
    assert metric["lora_gram"].shape == (0, 4, 4)
    assert metric["denom"].shape == (0,)
    assert geometry.max_rms(metric, zero, family) == 0
    assert torch.equal(geometry.project_state(metric, zero, family), zero)

    invalid_only = torch.tensor([True, False, False, False, False, False])
    metric = geometry.build_metric(pack, invalid_only)
    assert geometry.max_rms(metric, make_state(family), family) == 0
    assert torch.equal(geometry.project_state(metric, zero, family), zero)

    empty_pack = {key: value[:0] for key, value in pack.items()}
    empty_metric = geometry.build_metric(empty_pack, torch.empty(0, dtype=torch.bool))
    assert geometry.shift(empty_pack, zero, family).shape == (0, 16, 64)
    assert geometry.max_rms(empty_metric, zero, family) == 0


@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_shift_metric_and_projection_preserve_finite_gradients(pack, enabled, family):
    state = (make_state(family) * 2).requires_grad_()
    delta = geometry.shift(pack, state, family)
    delta.square().sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0

    state = (make_state(family) * 2).requires_grad_()
    metric = geometry.build_metric(pack, enabled)
    geometry.max_rms(metric, state, family).backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0

    state = (make_state(family) * 2).requires_grad_()
    geometry.project_state(metric, state, family).square().sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()

    zero = torch.zeros_like(state, requires_grad=True)
    geometry.max_rms(metric, zero, family).backward()
    assert zero.grad is not None and torch.isfinite(zero.grad).all()
    assert torch.count_nonzero(zero.grad) == 0


@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_metric_gradient_matches_direct_masked_rms(pack, enabled, family):
    state = make_state(family, dtype=torch.float64).requires_grad_()
    metric_rms = geometry.max_rms(geometry.build_metric(pack, enabled), state, family)
    direct_rms = direct_max_rms(pack, state, family, enabled)
    metric_grad = torch.autograd.grad(metric_rms, state, retain_graph=True)[0]
    direct_grad = torch.autograd.grad(direct_rms, state)[0]
    torch.testing.assert_close(metric_rms, direct_rms, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(metric_grad, direct_grad, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_quadratic_roundoff_is_clamped_but_material_negative_is_rejected(family):
    metric = {
        "gate_gram": torch.zeros(1, 16, 2, 2, dtype=torch.float64),
        "lora_gram": torch.zeros(1, 4, 4, dtype=torch.float64),
        "denom": torch.ones(1, dtype=torch.float64),
    }
    matrix = torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0 - torch.finfo(torch.float64).eps]],
        dtype=torch.float64,
    )
    state = torch.zeros_like(make_state(family, dtype=torch.float64))
    if family == "gates":
        metric["gate_gram"][0, 0] = matrix
        state[0] = 1
        diagonal = metric["gate_gram"][0, 0, 1, 1]
    else:
        metric["lora_gram"][0, :2, :2] = matrix
        state[:2, 0] = 1
        diagonal = metric["lora_gram"][0, 1, 1]
    assert geometry.max_rms(metric, state, family) == 0
    diagonal.fill_(0.999)
    with pytest.raises((ValueError, FloatingPointError, AssertionError)):
        geometry.max_rms(metric, state, family)


@pytest.mark.parametrize("name", ["h", "valid", "long", "snapshot", "phi"])
def test_missing_pack_keys_are_rejected(pack, enabled, name):
    del pack[name]
    with pytest.raises((KeyError, ValueError)):
        geometry.build_metric(pack, enabled)
    with pytest.raises((KeyError, ValueError)):
        geometry.shift(pack, make_state("gates"), "gates")


@pytest.mark.parametrize("radius", [0, -1, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("family", ["gates", "ffn_lora"])
def test_nonpositive_or_nonfinite_radius_is_rejected(pack, enabled, family, radius):
    with pytest.raises((ValueError, FloatingPointError)):
        geometry.project_state(geometry.build_metric(pack, enabled), make_state(family), family, radius)


def test_bad_shapes_types_and_nonfinite_inputs_are_rejected(pack, enabled):
    for name in ("h", "long", "snapshot", "phi"):
        invalid = dict(pack)
        invalid[name] = invalid[name].clone()
        invalid[name].reshape(-1)[0] = math.nan
        with pytest.raises((ValueError, FloatingPointError)):
            geometry.build_metric(invalid, enabled)
    for bad_valid in (pack["valid"].float(), pack["valid"][:, :16]):
        with pytest.raises((ValueError, TypeError)):
            geometry.build_metric(dict(pack, valid=bad_valid), enabled)
    for bad_enabled in (enabled.float(), enabled[:, None], enabled[:-1]):
        with pytest.raises((ValueError, TypeError)):
            geometry.build_metric(pack, bad_enabled)
    for name in ("h", "long", "snapshot", "phi"):
        with pytest.raises((ValueError, TypeError)):
            geometry.build_metric(dict(pack, **{name: pack[name][..., :-1]}), enabled)
    for family in ("gates", "ffn_lora"):
        state = make_state(family)
        state[0, 0] = math.inf
        with pytest.raises((ValueError, FloatingPointError)):
            geometry.shift(pack, state, family)
        with pytest.raises((ValueError, FloatingPointError)):
            geometry.max_rms(geometry.build_metric(pack, enabled), state, family)
        with pytest.raises((ValueError, TypeError)):
            geometry.shift(pack, torch.zeros(1), family)
    with pytest.raises(ValueError):
        geometry.shift(pack, make_state("gates"), "unknown")


@pytest.mark.parametrize("name", ["gate_gram", "lora_gram", "denom"])
def test_missing_or_nonfinite_metric_values_are_rejected(pack, enabled, name):
    metric = geometry.build_metric(pack, enabled)
    del metric[name]
    with pytest.raises((KeyError, ValueError)):
        geometry.max_rms(metric, make_state("gates"), "gates")
    metric = geometry.build_metric(pack, enabled)
    metric[name] = metric[name].clone()
    metric[name].reshape(-1)[0] = math.nan
    with pytest.raises((ValueError, FloatingPointError)):
        geometry.max_rms(metric, make_state("gates"), "gates")


def test_gate_projection_errors_are_not_swallowed(pack, enabled, monkeypatch):
    def fail(_state):
        raise RuntimeError("projection dependency failed")

    monkeypatch.setattr(geometry, "project_gates", fail)
    with pytest.raises(RuntimeError, match="projection dependency failed"):
        geometry.project_state(geometry.build_metric(pack, enabled), make_state("gates"), "gates")
