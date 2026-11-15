"""CPU tests for the frozen source-native temporary token interface.

Only hash-locked model artifacts and synthetic tensors are used; no shot data.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.risk_aligned_online.centered_gate_stream_20261101.model import (
    loss as centered_loss,
)
from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.native_token_probe_20261104 import core
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from experiments.risk_aligned_online.simple_online import FAST_RADIUS, FAST_STEP


SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
DECODER_SHA = "cdde96086517cbc468a911bbb5926dde1c783b3711b0a8abc09caf0a6d2e6730"
PRIOR = Path(__file__).resolve().parents[1] / "probe_outputs/direct_decoder_source_v1/decoders.pt"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def source():
    assert sha(DEFAULT_PARENT) == SOURCE_SHA and sha(PRIOR) == DECODER_SHA
    torch.set_num_threads(1)
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    stored = torch.load(PRIOR, map_location="cpu", weights_only=True)
    assert stored["source_sha256"] == SOURCE_SHA
    decoder = deepcopy(base.forecast_head)
    decoder.load_state_dict(stored["heads"]["forecast_only"], strict=True)
    decoder.eval().requires_grad_(False)
    assert base.fast_origin.device.type == "cpu"
    yield base, decoder
    torch.backends.mha.set_fastpath_enabled(old_fastpath)


@pytest.fixture(autouse=True)
def frozen_models_unchanged(source):
    base, decoder = source
    states = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in source]
    hooks = [
        {n: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for n, m in obj.named_modules()}
        for obj in source
    ]
    yield
    for obj, before, before_hooks in zip(source, states, hooks):
        assert not obj.training
        assert all(not p.requires_grad and p.grad is None for p in obj.parameters())
        assert all(torch.equal(v, obj.state_dict()[k]) for k, v in before.items())
        assert {n: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for n, m in obj.named_modules()} == before_hooks


@pytest.fixture
def example(source):
    base, _ = source
    rng = torch.Generator(device="cpu").manual_seed(20261104)
    x = base.center + 0.3 * base.scale * torch.randn(2, 128, 23, generator=rng)
    a = torch.rand(2, 128, 23, generator=rng) > 0.1
    truth = base.center + 0.7 * base.scale * torch.randn(2, 25, 23, generator=rng)
    mask = torch.rand(2, 25, 23, generator=rng) > 0.2
    assert torch.isfinite(x).all() and torch.isfinite(truth).all()
    return x, a, truth, mask


def loss_for(base, kind, prediction, truth, mask):
    if kind == "absolute":
        return base.forecast_loss(prediction, truth, mask)
    return centered_loss(SimpleNamespace(base=base), prediction, truth, mask)


def test_zero_offset_is_bitwise_native_forward(source, example):
    base, _ = source
    x, a, _, _ = example
    with torch.no_grad():
        native = base(x, a)
        actual = core.forward(base, x, a, torch.zeros(2, 128))
    assert torch.equal(actual["logit"], native["logit"])
    assert torch.equal(actual["forecast"], native["forecast"])
    assert actual["logit"].shape == (2,) and actual["forecast"].shape == (2, 25, 23)


def test_external_head_exact_coordinates_and_delta_gradient(source, example):
    base, decoder = source
    x, a, _, _ = example
    delta = (torch.arange(128, dtype=x.dtype).repeat(2, 1) / 2048).requires_grad_()
    captured = []
    hook = base.forecast_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    try:
        native = base(x, a, fast=base.initial_fast(len(x)) + delta)
    finally:
        hook.remove()
    assert len(captured) == 1 and captured[0].shape == (2, 1088)
    actual = core.forward(base, x, a, delta, decoder=decoder)
    expected = decoder(captured[0]).reshape(2, 25, 23)
    assert decoder.weight.shape == (575, 1088) and decoder.bias.shape == (575,)
    assert torch.equal(actual["logit"], native["logit"])
    assert torch.equal(actual["forecast"], expected)
    flat = decoder(captured[0])
    for h in (0, 12, 24):
        for channel in (0, 1, 2, 12, 22):
            assert torch.equal(actual["forecast"][:, h, channel], flat[:, h * 23 + channel])
    risk_gradient = torch.autograd.grad(actual["logit"].sum(), delta, retain_graph=True)[0]
    forecast_gradient = torch.autograd.grad(actual["forecast"].square().mean(), delta)[0]
    assert torch.isfinite(risk_gradient).all() and risk_gradient.abs().sum() > 0
    assert torch.isfinite(forecast_gradient).all() and forecast_gradient.abs().sum() > 0


@pytest.mark.parametrize("kind", ["absolute", "centered"])
@pytest.mark.parametrize("external", [False, True], ids=["native", "external"])
def test_advance_is_original_normalized_step_with_honest_receipt(source, example, kind, external):
    base, decoder = source
    x, a, truth, mask = example
    head = decoder if external else None
    state = torch.zeros(2, 128, requires_grad=True)
    origin = base.initial_fast(2).detach()
    variable = state.detach().clone().requires_grad_(True)
    prediction = core.forward(base, x, a, variable, decoder=head)["forecast"]
    objective = loss_for(base, kind, prediction, truth, mask)
    gradient = torch.autograd.grad(objective, variable)[0]
    expected = normalized_step(origin + state.detach(), origin, gradient.detach(), step=FAST_STEP, radius=FAST_RADIUS) - origin
    with torch.no_grad():  # advance must locally enable gradients for its state.
        after, receipt = core.advance(base, x, a, state, truth, mask, decoder=head, loss_kind=kind)
    assert torch.equal(after, expected) and not after.requires_grad and state.grad is None
    assert torch.equal(state.detach(), torch.zeros_like(state))
    assert (after.norm(dim=1) <= FAST_RADIUS + 1e-6).all()
    actual_after = loss_for(base, kind, core.forward(base, x, a, after, decoder=head)["forecast"], truth, mask)
    assert receipt["reason"] == "normalized_step" and receipt["gradient_computed"] is True
    assert receipt["observed"] == int(mask.sum())
    assert receipt["loss_before"] == pytest.approx(float(objective.detach()), abs=1e-8)
    assert receipt["loss_after"] == pytest.approx(float(actual_after.detach()), abs=1e-8)
    assert torch.allclose(torch.tensor(receipt["gradient_norm"]), gradient.norm(dim=1))
    assert torch.allclose(torch.tensor(receipt["step_norm"]), after.norm(dim=1))
    assert torch.equal(torch.tensor(receipt["state_offset"]), after)


@pytest.mark.parametrize("kind", ["absolute", "centered"])
def test_missing_targets_and_mixed_batch_are_not_updated(source, example, kind):
    base, decoder = source
    x, a, truth, mask = example
    mask[1] = False
    state = torch.zeros(2, 128)
    state[1, 0] = 0.2
    changed_truth = truth.clone()
    changed_truth[~mask] = float("nan")
    x_with_missing = x.clone()
    x_with_missing[~a] = float("nan")
    after, receipt = core.advance(base, x_with_missing, a, state, changed_truth, mask, decoder=decoder, loss_kind=kind)
    reference, other = core.advance(base, x, a, state, truth, mask, decoder=decoder, loss_kind=kind)
    assert torch.equal(after, reference) and torch.equal(after[1], state[1])
    assert receipt["observed_per_sample"] == [int(mask[0].sum()), 0]
    assert receipt["gradient_norm"][1] == 0 and receipt["step_norm"][1] == 0
    assert receipt["loss_before"] == other["loss_before"] and receipt["loss_after"] == other["loss_after"]


@pytest.mark.parametrize("kind", ["absolute", "centered"])
def test_all_unobserved_returns_explicit_no_target_without_forward(source, example, monkeypatch, kind):
    base, decoder = source
    x, a, truth, mask = example
    mask[:] = False
    truth[:] = float("nan")
    state = torch.full((2, 128), 0.02, requires_grad=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no-target case must not execute the model")

    monkeypatch.setattr(base, "forward", forbidden)
    after, receipt = core.advance(base, x, a, state, truth, mask, decoder=decoder, loss_kind=kind)
    assert torch.equal(after, state.detach()) and not after.requires_grad and state.grad is None
    assert receipt["reason"] == "no_target" and receipt["observed"] == 0
    assert not receipt["gradient_computed"] and receipt["loss_before"] is None and receipt["loss_after"] is None
    assert receipt["gradient_norm"] == [0.0, 0.0] and receipt["step_norm"] == [0.0, 0.0]
    assert torch.equal(torch.tensor(receipt["state_offset"]), after)


def test_single_observed_centered_target_has_zero_gradient_but_is_finite(source, example):
    base, decoder = source
    x, a, truth, mask = example
    mask[:] = False
    mask[0, 0, 0] = True
    after, receipt = core.advance(base, x, a, torch.zeros(2, 128), truth, mask, decoder=decoder, loss_kind="centered")
    assert torch.isfinite(after).all() and not after.any()
    assert receipt["observed"] == 1 and receipt["loss_before"] == 0
    assert receipt["gradient_computed"] and receipt["gradient_norm"] == [0.0, 0.0]


def test_radius_is_projected_around_learned_origin(source, example):
    base, _ = source
    x, a, truth, mask = example
    state = torch.zeros(2, 128)
    state[:, 0] = 4.0
    after, receipt = core.advance(base, x, a, state, truth, mask)
    assert (after.norm(dim=1) <= FAST_RADIUS + 1e-6).all()
    assert all(n > 3.9 for n in receipt["offset_norm_before"])
    assert all(n <= FAST_RADIUS + 1e-6 for n in receipt["offset_norm_after"])


def test_external_hook_is_removed_on_error_without_removing_existing_hook(source, example, monkeypatch):
    base, decoder = source
    x, a, _, _ = example
    sentinel = base.forecast_head.register_forward_pre_hook(lambda *_args: None)
    before = tuple(base.forecast_head._forward_pre_hooks)

    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate source forward failure")

    try:
        monkeypatch.setattr(base, "forward", fail)
        with pytest.raises(RuntimeError, match="deliberate source forward failure"):
            core.forward(base, x, a, torch.zeros(2, 128), decoder=decoder)
        assert tuple(base.forecast_head._forward_pre_hooks) == before
    finally:
        sentinel.remove()


@pytest.mark.parametrize("failure", ["site", "rule", "view", "dim", "trainable_base", "training_base"])
def test_invalid_source_configuration_is_rejected(source, example, failure):
    original, _ = source
    x, a, _, _ = example
    bad = deepcopy(original)
    if failure == "site":
        bad.cfg = replace(bad.cfg, fast_site="output")
    elif failure == "rule":
        bad.cfg = replace(bad.cfg, fast_rule="raw")
    elif failure == "view":
        bad.cfg = replace(bad.cfg, input_view="window")
    elif failure == "dim":
        bad.cfg = replace(bad.cfg, dim=32)
    elif failure == "trainable_base":
        bad.risk_head.weight.requires_grad_(True)
    else:
        bad.train()
    with pytest.raises((ValueError, TypeError)):
        core.forward(bad, x, a, torch.zeros(2, 128))


@pytest.mark.parametrize("failure", ["shape", "training", "trainable", "dtype", "aliased"])
def test_invalid_decoder_is_rejected(source, example, failure):
    base, decoder = source
    x, a, _, _ = example
    if failure == "shape":
        bad = torch.nn.Linear(1088, 574).eval().requires_grad_(False)
    else:
        bad = deepcopy(decoder)
        if failure == "training":
            bad.train()
        elif failure == "trainable":
            bad.weight.requires_grad_(True)
        elif failure == "dtype":
            bad.double()
        else:
            bad.weight = base.forecast_head.weight
            bad.bias = base.forecast_head.bias
    with pytest.raises((ValueError, TypeError)):
        core.forward(base, x, a, torch.zeros(2, 128), decoder=bad)


@pytest.mark.parametrize("failure", ["offset_shape", "offset_nan", "x_shape", "availability_dtype", "observed_x_nan", "truth_shape", "mask_dtype", "observed_truth_nan", "loss_kind"])
def test_invalid_inputs_are_rejected(source, example, failure):
    base, decoder = source
    x, a, truth, mask = example
    state = torch.zeros(2, 128)
    kind = "absolute"
    if failure == "offset_shape":
        state = torch.zeros(2, 127)
    elif failure == "offset_nan":
        state[0, 0] = float("nan")
    elif failure == "x_shape":
        x, a = x[:, :-1], a[:, :-1]
    elif failure == "availability_dtype":
        a = a.float()
    elif failure == "observed_x_nan":
        x[0, 0, 0], a[0, 0, 0] = float("nan"), True
    elif failure == "truth_shape":
        truth, mask = truth[:, :-1], mask[:, :-1]
    elif failure == "mask_dtype":
        mask = mask.float()
    elif failure == "observed_truth_nan":
        truth[0, 0, 0], mask[0, 0, 0] = float("nan"), True
    else:
        kind = "unknown"
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        core.advance(base, x, a, state, truth, mask, decoder=decoder, loss_kind=kind)
