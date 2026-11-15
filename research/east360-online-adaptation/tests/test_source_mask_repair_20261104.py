"""CPU-only synthetic coverage of the source embedding repair contract."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.mask_repair_20261104 import core
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
    sha256,
)


@pytest.fixture(scope="session")
def base():
    torch.set_num_threads(1)
    assert sha256(DEFAULT_PARENT) == PARENT_SHA256
    model, _ = load_parent(DEFAULT_PARENT)
    return model


@pytest.fixture(autouse=True)
def unchanged_source(base):
    state = {name: value.clone() for name, value in base.state_dict().items()}
    flags = [(module, module.training) for module in base.modules()]
    hooks = dict(base.embedding._forward_hooks)
    yield
    assert all(torch.equal(value, base.state_dict()[name]) for name, value in state.items())
    assert all(module.training == flag for module, flag in flags)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in base.parameters())
    assert dict(base.embedding._forward_hooks) == hooks


@pytest.fixture
def example(base):
    generator = torch.Generator().manual_seed(324)
    x = base.center + base.scale * (0.3 * torch.randn(2, 128, 23, generator=generator))
    available = torch.ones_like(x, dtype=torch.bool)
    # First patch: 5 padding frames, then 3 real frames. Missing CIII on one
    # real frame and MNM on two, i.e. availability 2/3 and 1/3.
    available[0, :5] = False
    available[0, 5, 1] = False
    available[0, 6:8, 2] = False
    available[0, 8:16] = False  # all padding: no repair
    available[0, 16:21] = False  # 3 valid and complete frames: no repair
    available[1, :, 2] = False
    return x, available


def test_architecture_and_initial_native_equality(base, example):
    repair = core.TokenRepair()
    assert [type(m) for m in repair.net] == [nn.LayerNorm, nn.Linear, nn.SiLU, nn.Linear]
    assert (repair.net[1].in_features, repair.net[1].out_features) == (66, 16)
    assert (repair.net[3].in_features, repair.net[3].out_features) == (16, 64)
    assert sum(p.numel() for p in repair.parameters()) == 2292
    assert torch.count_nonzero(repair.net[-1].weight) == 0
    assert torch.count_nonzero(repair.net[-1].bias) == 0
    assert all(module is not base for module in repair.modules())
    x, available = example
    x0, a0 = x.clone(), available.clone()
    for fastpath in (True, False):
        previous = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(fastpath)
            with torch.no_grad():
                native = base(x, available)
                result = core.forward(base, repair, x, available)
            assert torch.equal(result["logit"], native["logit"])
            assert torch.equal(result["forecast"], native["forecast"])
            assert torch.count_nonzero(result["embedding_delta"]) == 0
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)
    assert torch.equal(x, x0) and torch.equal(available, a0)


def test_valid_frame_fractions_and_mixed_batch(base, example):
    repair = core.TokenRepair()
    x, available = example
    seen = []
    hook = repair.net[0].register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach()))
    try:
        result = core.forward(base, repair, x, available)
    finally:
        hook.remove()
    missing = result["missing_fraction"]
    tokens = result["repair_tokens"]
    assert missing.shape == (2, 16, 2) and tokens.shape == (2, 16)
    torch.testing.assert_close(missing[0, 0], torch.tensor([1 / 3, 2 / 3]))
    assert torch.equal(missing[0, 1:3], torch.zeros(2, 2))
    assert torch.equal(tokens[0], torch.tensor([True] + [False] * 15))
    assert tokens[1].all()
    # The plan specifies availability, not missingness, as the two inputs.
    assert len(seen) == 1 and seen[0].shape == (17, 66)
    torch.testing.assert_close(seen[0][0, -2:], torch.tensor([2 / 3, 1 / 3]))


@pytest.mark.parametrize("fastpath", [True, False])
@pytest.mark.parametrize("disabled", [True, False])
def test_noop_keeps_exact_native_path(base, example, fastpath, disabled):
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].bias.fill_(2)
    x, masked = example
    available = masked if disabled else torch.ones_like(masked)
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        native = base(x, available)
        # A disabled or fully observed batch must not call the network, even
        # when it contains trainable parameters and grad mode is enabled.
        with patch.object(repair, "forward", side_effect=AssertionError("unexpected repair")):
            result = core.forward(base, repair, x, available, enabled=not disabled)
        assert result["embedding_before"] is result["embedding_after"]
        assert not result["embedding_delta"].requires_grad
        assert not result["repair_tokens"].any()
        assert not result["logit"].requires_grad
        assert torch.equal(result["logit"], native["logit"])
        assert torch.equal(result["forecast"], native["forecast"])
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_nonzero_repair_is_gated_bounded_and_does_not_mutate_inputs(base, example):
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].bias.fill_(3)
    x, available = example
    x0, a0 = x.clone(), available.clone()
    result = core.forward(base, repair, x, available)
    delta, tokens = result["embedding_delta"], result["repair_tokens"]
    assert delta.shape == (2, 16, 64)
    assert torch.count_nonzero(delta[~tokens]) == 0
    assert torch.count_nonzero(delta[tokens]) == 17 * 64
    assert delta.double().square().mean(-1).sqrt().max() <= 0.25000002
    assert torch.equal(result["embedding_before"][~tokens], result["embedding_after"][~tokens])
    torch.testing.assert_close(result["embedding_after"] - result["embedding_before"], delta)
    assert torch.equal(x, x0) and torch.equal(available, a0)


def test_mixed_batch_preserves_row_with_only_other_diagnostics_missing(base, example):
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].bias.fill_(1)
    x, available = example
    available = available.clone()
    available[1] = True
    available[1, :, 3] = False  # not CIII or MNM
    result = core.forward(base, repair, x, available)
    assert result["repair_tokens"][0].any()
    assert not result["repair_tokens"][1].any()
    assert not result["embedding_delta"][1].any()
    assert torch.equal(result["embedding_before"][1], result["embedding_after"][1])


def test_strict_rms_bound_with_varied_capped_tokens(base):
    generator = torch.Generator().manual_seed(712)
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].weight.normal_(generator=generator)
        repair.net[-1].bias.normal_(generator=generator)
    x = base.center + base.scale * torch.randn(48, 128, 23, generator=generator)
    available = torch.ones_like(x, dtype=torch.bool)
    available[:, :, 1] = False
    result = core.forward(base, repair, x, available)
    rms = result["embedding_delta"].double().square().mean(-1).sqrt()
    assert result["repair_tokens"].all()
    assert (rms <= 0.25).all()


@pytest.mark.parametrize("bias", [1e37, 1e38, 1.9189731294592223e38, 2e38, 3e38])
def test_rms_bound_for_extreme_finite_raw_values(base, example, bias):
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].bias.fill_(bias)
    x, available = example
    result = core.forward(base, repair, x, available)
    rms = result["embedding_delta"].double().square().mean(-1).sqrt()
    assert torch.isfinite(rms).all() and (rms <= 0.25).all()
    assert (rms[result["repair_tokens"]] > 0.249).all()


@pytest.mark.parametrize("fastpath", [True, False])
@pytest.mark.parametrize("input_grad", [True, False])
def test_zero_init_and_differentiable_source_paths(base, example, fastpath, input_grad):
    repair = core.TokenRepair()
    x, available = example
    x = x.clone().requires_grad_(input_grad)
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        native = base(x, available)
        result = core.forward(base, repair, x, available)
        if fastpath and not input_grad:
            # Frozen native MHA can use a different CPU kernel from the
            # differentiable repair path. Do not misreport that as bitwise.
            torch.testing.assert_close(result["logit"], native["logit"], rtol=0, atol=2e-6)
            torch.testing.assert_close(result["forecast"], native["forecast"], rtol=0, atol=2e-6)
        else:
            assert torch.equal(result["logit"], native["logit"])
            assert torch.equal(result["forecast"], native["forecast"])
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_all_padding_is_noop_without_network(base, example):
    repair = core.TokenRepair()
    x, _ = example
    available = torch.zeros_like(x, dtype=torch.bool)
    with patch.object(repair, "forward", side_effect=AssertionError("unexpected repair")):
        result = core.forward(base, repair, x, available)
    assert not result["repair_tokens"].any()
    assert not result["missing_fraction"].any()
    assert result["embedding_after"] is result["embedding_before"]
    assert not result["embedding_delta"].requires_grad


def test_gradients_reach_only_repair_and_are_finite_at_zero(base, example):
    repair = core.TokenRepair()
    x, available = example
    result = core.forward(base, repair, x, available)
    loss = result["logit"].square().mean() + result["forecast"].square().mean()
    loss.backward()
    assert repair.net[-1].weight.grad is not None
    assert repair.net[-1].bias.grad is not None
    assert repair.net[-1].weight.grad.abs().max() > 0
    assert repair.net[-1].bias.grad.abs().max() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in repair.parameters())


def test_clipped_branch_gradients_are_finite(base, example):
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].bias.fill_(1)
    x, available = example
    result = core.forward(base, repair, x, available)
    result["logit"].square().mean().backward()
    assert repair.net[-1].bias.grad is not None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in repair.parameters())


def test_hooks_cleaned_on_exception(base, example):
    repair = core.TokenRepair()
    x, available = example
    sentinel = base.embedding.register_forward_hook(lambda _module, _args, _out: None)
    expected = dict(base.embedding._forward_hooks)
    try:
        with patch.object(repair, "forward", side_effect=RuntimeError("repair failed")):
            with pytest.raises(RuntimeError, match="repair failed"):
                core.forward(base, repair, x, available)
        assert dict(base.embedding._forward_hooks) == expected
        with patch.object(base, "forward", side_effect=RuntimeError("source failed")):
            with pytest.raises(RuntimeError, match="source failed"):
                core.forward(base, repair, x, available)
        assert dict(base.embedding._forward_hooks) == expected
    finally:
        sentinel.remove()


@pytest.mark.parametrize("case", [
    "shape", "empty", "value_dtype", "mask_dtype", "mask_shape", "observed_nan",
    "value_device", "mask_device", "repair_dtype", "repair_device", "repair_nan", "enabled",
])
def test_invalid_inputs_rejected(base, example, case):
    repair = core.TokenRepair()
    x, available = example
    enabled = True
    if case == "shape":
        x = x[:, :-1]
    elif case == "empty":
        x, available = x[:0], available[:0]
    elif case == "value_dtype":
        x = x.double()
    elif case == "mask_dtype":
        available = available.float()
    elif case == "mask_shape":
        available = available[:, :-1]
    elif case == "observed_nan":
        x = x.clone()
        x[1, 0, 0] = float("nan")
    elif case == "value_device":
        x = x.to("meta")
    elif case == "mask_device":
        available = available.to("meta")
    elif case == "repair_dtype":
        repair.double()
    elif case == "repair_device":
        repair.to("meta")
    elif case == "repair_nan":
        with torch.no_grad():
            repair.net[-1].bias[0] = float("nan")
    elif case == "enabled":
        enabled = 1
    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        core.forward(base, repair, x, available, enabled=enabled)


def test_unobserved_nonfinite_is_not_promoted_to_observation(base, example):
    repair = core.TokenRepair()
    x, available = example
    x = x.clone()
    x[0, 0, 0] = float("nan")
    result = core.forward(base, repair, x, available)
    assert torch.isfinite(result["logit"]).all() and torch.isfinite(result["forecast"]).all()
    assert torch.isnan(x[0, 0, 0]) and not available[0, 0, 0]


def test_invalid_source_and_shared_storage_rejected(base, example):
    repair = core.TokenRepair()
    x, available = example
    with patch.object(base, "cfg", replace(base.cfg, horizon=24)):
        with pytest.raises(ValueError, match="source"):
            core.forward(base, repair, x, available)
    with patch.object(base, "training", True):
        with pytest.raises(ValueError, match="frozen|eval"):
            core.forward(base, repair, x, available)
    parameter = base.risk_head.weight
    try:
        parameter.requires_grad_(True)
        with pytest.raises(ValueError, match="frozen|eval"):
            core.forward(base, repair, x, available)
    finally:
        parameter.requires_grad_(False)
    repair.net[-1].weight = nn.Parameter(base.embedding.weight[:, :16].detach())
    with pytest.raises(ValueError, match="storage|alias|share"):
        core.forward(base, repair, x, available)
