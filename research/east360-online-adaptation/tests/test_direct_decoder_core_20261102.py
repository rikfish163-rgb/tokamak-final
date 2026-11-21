"""Random CPU models only; no real checkpoint, HDF, or training run."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.adaptivity_capacity_20261102.model import (
    source_basis,
)
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.direct_decoder_meta_20261102 import core
from experiments.risk_aligned_online.full_path_meta_20261102 import core as old
from experiments.risk_aligned_online.lora_stream_20261102 import core as lora
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture
def case():
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(57)
        model = (
            ContextAdapter(
                Simple23(
                    Config(
                        fast_site="tokens", fast_rule="normalized", input_view="source"
                    ),
                    torch.zeros(23),
                    torch.ones(23),
                )
            )
            .eval()
            .requires_grad_(False)
        )
    rng = np.random.default_rng(58)
    signal = dict(
        values=rng.normal(size=(220, 23)).astype(np.float32),
        available=np.ones((220, 23), dtype=bool),
    )
    basis = source_basis(model)
    try:
        yield model, signal, basis
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def inputs(case):
    model, signal, basis = case
    origin = lora.encode(model, signal, [134], basis)
    later = lora.encode(model, signal, [180], basis)
    zero = model.base.center.new_zeros(4, 64)
    baseline = lora.forward(model, origin, zero, forecast=True)["forecast"].detach()
    wave = torch.sin(torch.arange(25, dtype=baseline.dtype) * 0.4)[None, :, None]
    truth = (baseline + 0.3 * wave).sinh()
    mask = torch.ones_like(truth, dtype=torch.bool)
    mask[:, :, 10] = False
    return origin, later, zero, truth, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clone_is_exact_independent_trainable_head_with_inherited_dtype(case, dtype):
    model, signal, basis = case
    pack = lora.encode(model, signal, [127, 159], basis)
    model.to(dtype=dtype)
    pack = {
        k: v.to(dtype=dtype) if v.is_floating_point() else v for k, v in pack.items()
    }
    decoder = core.clone_head(model)
    assert isinstance(decoder, torch.nn.Linear)
    assert (decoder.in_features, decoder.out_features) == (1088, 575)
    for external, source in zip(
        decoder.parameters(), model.base.forecast_head.parameters()
    ):
        assert (
            external.dtype == source.dtype == dtype and external.device == source.device
        )
        assert external.data_ptr() != source.data_ptr() and torch.equal(
            external, source
        )
        assert external.requires_grad and not source.requires_grad
    state = torch.full((4, 64), 0.03, dtype=dtype, requires_grad=True)
    expected = lora.forward(model, pack, state, forecast=True)["forecast"]
    actual = core.prediction(model, pack, state, decoder)
    assert torch.equal(actual, expected)
    assert torch.autograd.grad(actual.sum(), state)[0].norm() > 0
    assert not model.base.forecast_head._forward_pre_hooks


def test_prediction_has_no_risk_bridge_and_external_optimization_cannot_change_source(
    case,
):
    model, _, _ = case
    origin, _, zero, _, _ = inputs(case)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    decoder = core.clone_head(model)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)
    core.prediction(model, origin, zero, decoder).square().mean().backward()
    optimizer.step()
    assert not torch.equal(decoder.weight, model.base.forecast_head.weight)
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.zero_()
    assert lora.forward(model, origin, zero)["logit"].abs().sum() > 0
    assert core.prediction(model, origin, zero, decoder).count_nonzero() == 0


def test_prediction_uses_the_same_input_as_actual_direct_ffn_hook(case):
    model, signal, basis = case
    pack = lora.encode(model, signal, [127, 180], basis)
    decoder = core.clone_head(model)
    with torch.no_grad():
        decoder.weight.mul_(0.8)
    state = torch.full((4, 64), 0.04)
    captured = []
    handle = model.base.forecast_head.register_forward_pre_hook(
        lambda m, args: captured.append(args[0])
    )
    try:
        lora.direct(model, signal, [127, 180], basis, state, forecast=True)
    finally:
        handle.remove()
    actual = decoder(captured[0]).reshape(2, 25, 23)
    torch.testing.assert_close(
        core.prediction(model, pack, state, decoder), actual, rtol=0, atol=0
    )
    assert not model.base.forecast_head._forward_pre_hooks
    assert not model.base.encoder.layers[0].linear2._forward_hooks


def test_inner_numeric_algorithm_matches_old_zero_bridge_and_receipts(case):
    model, _, _ = case
    origin, _, zero, truth, mask = inputs(case)
    after, receipt = core.advance(
        model, origin, zero, truth, mask, core.clone_head(model)
    )
    reference, recorded = old.advance(
        model, origin, zero, truth, mask, zero.new_zeros(25, 23)
    )
    torch.testing.assert_close(after, reference, rtol=0, atol=0)
    assert receipt == recorded and receipt["accepted"]
    assert receipt["loss_proposal"] < receipt["loss_before"]
    assert len(receipt["gradient256"]) == len(receipt["state256_after"]) == 256
    assert not after.requires_grad


def test_later_risk_only_head_gradient_fd_and_direct_hook(case, monkeypatch):
    model, signal, _ = case
    model.double()
    original_batch = lora.batch

    def double_batch(*args):
        x, mask = original_batch(*args)
        return x.double(), mask

    monkeypatch.setattr(lora, "batch", double_batch)
    basis = source_basis(model).double()
    double_case = model, signal, basis
    origin, later, zero, truth, mask = inputs(double_case)
    decoder = core.clone_head(model)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    after, receipt = core.advance(
        model, origin, zero, truth, mask, decoder, create_graph=True
    )
    assert receipt["accepted"]
    logit = lora.forward(model, later, after)["logit"]
    risk = F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
    gradients = torch.autograd.grad(risk, tuple(decoder.parameters()))
    norm = torch.cat([v.flatten() for v in gradients]).norm()
    assert torch.isfinite(norm) and norm > 1e-10
    directions = [g / norm for g in gradients]
    initial = [p.detach().clone() for p in decoder.parameters()]
    for eps in (1e-4, 2e-4):
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                for p, value, direction in zip(
                    decoder.parameters(), initial, directions
                ):
                    p.copy_(value + sign * eps * direction)
            state, row = core.advance(model, origin, zero, truth, mask, decoder)
            assert row["accepted"] == receipt["accepted"]
            z = lora.forward(model, later, state)["logit"]
            values.append(
                float(F.binary_cross_entropy_with_logits(z, torch.ones_like(z)))
            )
        assert (values[1] - values[0]) / (2 * eps) == pytest.approx(
            float(norm), rel=2e-3, abs=1e-8
        )
    direct = lora.direct(model, signal, [180], basis, after.detach())["logit"]
    torch.testing.assert_close(direct, logit.detach(), rtol=1e-10, atol=1e-10)
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())
    assert not model.base.forecast_head._forward_pre_hooks
    assert not model.base.encoder.layers[0].linear2._forward_hooks


@pytest.mark.parametrize("graph", [False, True])
def test_no_target_keeps_state_and_only_graph_mode_keeps_incoming_dependence(
    case, graph
):
    model, _, _ = case
    decoder = core.clone_head(model)
    state = decoder.weight[0, 0] * torch.ones(4, 64)
    truth = torch.full((1, 25, 23), float("nan"))
    after, row = core.advance(
        model,
        {},
        state,
        truth,
        torch.zeros_like(truth, dtype=torch.bool),
        decoder,
        create_graph=graph,
    )
    assert (
        torch.equal(after, state)
        and not row["accepted"]
        and row["reason"] == "no_target"
    )
    assert row["step_norm"] == row["gradient_norm"] == 0
    assert after.requires_grad == graph
    if graph:
        assert torch.autograd.grad(after.sum(), decoder.weight)[0][0, 0] == 256


def test_equal_proposal_rejected_and_incoming_graph_is_not_detached(case, monkeypatch):
    model, _, _ = case
    origin, _, zero, truth, mask = inputs(case)
    decoder = core.clone_head(model)
    incoming = zero + decoder.weight[0, 0] * 0.01
    monkeypatch.setattr(core, "metric_direction", lambda pack, gradient: gradient * 0)
    after, row = core.advance(
        model, origin, incoming, truth, mask, decoder, create_graph=True
    )
    assert row["loss_before"] == row["loss_proposal"] and not row["accepted"]
    assert torch.equal(after, incoming) and row["step_norm"] == 0
    assert torch.autograd.grad(after.sum(), decoder.weight)[0][0, 0] == pytest.approx(
        2.56
    )


@pytest.mark.parametrize("failure", ["source", "decoder", "twice"])
def test_prediction_hooks_are_removed_on_every_failure(case, monkeypatch, failure):
    model, _, _ = case
    origin, _, zero, _, _ = inputs(case)
    decoder = core.clone_head(model)
    calls = []
    existing = model.base.forecast_head.register_forward_pre_hook(
        lambda m, a: calls.append(1)
    )
    ids = list(model.base.forecast_head._forward_pre_hooks)

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    if failure == "source":
        monkeypatch.setattr(core, "forward", fail)
    elif failure == "decoder":
        monkeypatch.setattr(decoder, "forward", fail)
    else:

        def twice(*args, **kwargs):
            lora.forward(*args, **kwargs)
            return lora.forward(*args, **kwargs)

        monkeypatch.setattr(core, "forward", twice)
    try:
        with pytest.raises(RuntimeError):
            core.prediction(model, origin, zero, decoder)
        assert list(model.base.forecast_head._forward_pre_hooks) == ids
    finally:
        existing.remove()
