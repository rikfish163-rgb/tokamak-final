"""CPU-first contracts for the isolated input-patch-embedding alternative."""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004 import inter
from experiments.risk_aligned_online.redesign_20261004.embedding_model import EmbeddingModel


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case():
    generator = torch.Generator().manual_seed(20261004)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                        torch.linspace(-1, 1, 23), torch.linspace(.5, 2, 23)).eval()
    with torch.no_grad():
        base.fast_origin.copy_(torch.linspace(-.3, .4, 128))
    values = torch.randn(2, 128, 23, generator=generator)
    available = torch.rand(values.shape, generator=generator) > .15
    return base, values, available


def snapshot(base):
    return {
        "state": {name: tensor.detach().clone() for name, tensor in base.state_dict().items()},
        "registered": {
            name: (tensor, tensor.data_ptr(), tensor._version)
            for name, tensor in [*base.named_parameters(), *base.named_buffers()]
        },
    }


def assert_base_unchanged(base, before):
    assert set(base.state_dict()) == set(before["state"])
    for name, tensor in base.state_dict().items():
        torch.testing.assert_close(tensor, before["state"][name], rtol=0, atol=0)
    registered = dict([*base.named_parameters(), *base.named_buffers()])
    assert set(registered) == set(before["registered"])
    for name, (original, pointer, version) in before["registered"].items():
        assert registered[name] is original, name
        assert registered[name].data_ptr() == pointer, name
        assert registered[name]._version == version, name
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in base.parameters())
    assert all(not module.training for module in base.modules())


def assert_exact(actual, expected):
    assert set(actual) == set(expected) == {"logit", "forecast"}
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["grad", "no_grad", "inference"])
@pytest.mark.parametrize("explicit", [False, True])
def test_zero_identity_uses_matched_source_dispatch(mode, explicit):
    base, values, available = case()
    model = EmbeddingModel(base).eval()
    before = snapshot(base)
    context = {"grad": torch.enable_grad, "no_grad": torch.no_grad,
               "inference": torch.inference_mode}[mode]
    with context():
        # A frozen source with trainable token state selects the same grad path
        # as the wrapper's differentiable embedding; don't compare train/eval kernels.
        source_fast = base.initial_fast(2)
        if mode == "grad":
            source_fast.requires_grad_()
        expected = base(values, available, source_fast)
        fast = model.initial_fast(2) if explicit else None
        assert_exact(model(values, available, fast), expected)
    assert_base_unchanged(base, before)


def test_fast_delta_matches_absolute_source_and_its_gradient():
    base, values, available = case()
    model = EmbeddingModel(base)
    delta = torch.linspace(-.2, .3, 128).repeat(2, 1).requires_grad_()
    source_delta = delta.detach().clone().requires_grad_()
    expected = base(values, available, base.initial_fast(2) + source_delta)
    actual = model(values, available, delta)
    assert_exact(actual, expected)
    for head in ("logit", "forecast"):
        reference, = torch.autograd.grad(expected[head].square().mean(), source_delta, retain_graph=True)
        derivative, = torch.autograd.grad(actual[head].square().mean(), delta, retain_graph=True)
        torch.testing.assert_close(derivative, reference, rtol=0, atol=0)
    with torch.no_grad():
        assert_exact(model(values, available, delta),
                     base(values, available, base.initial_fast(2) + delta))


def test_interface_parameter_count_and_rng_are_exact():
    base, _, _ = case()
    before = snapshot(base)
    rng = torch.random.get_rng_state().clone()
    model = EmbeddingModel(base)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert model.cfg is base.cfg and model.center is base.center and model.scale is base.scale
    parameters = tuple(model.persistent_parameters())
    assert len(parameters) == 2
    assert parameters[0] is model.delta_weight and parameters[1] is model.delta_bias
    assert model.delta_weight.shape == (64, 368) and model.delta_bias.shape == (64,)
    assert sum(p.numel() for p in parameters) == 23552 + 64 == 23616
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    assert not ({id(p) for p in parameters} & {id(p) for p in base.parameters()})
    assert all(torch.count_nonzero(p) == 0 for p in parameters)
    assert set(dict(model.named_children())) == {"base"}
    fast = model.initial_fast(2)
    assert fast.shape == (2, 128) and not fast.requires_grad and not fast.any()
    assert fast.dtype == base.embedding.weight.dtype and fast.device.type == "cpu"
    fast[0, 0] = 1
    assert fast[1, 0] == 0 and not model.initial_fast(2).any()
    assert_base_unchanged(base, before)


@pytest.mark.parametrize("parameter", ["delta_weight", "delta_bias"])
def test_each_embedding_delta_changes_tokens_and_both_original_heads(parameter):
    base, values, available = case()
    model = EmbeddingModel(base)
    frozen = snapshot(base)
    tokens, risks, forecasts = [], [], []
    hooks = [
        base.encoder.register_forward_pre_hook(lambda module, args: tokens.append(args[0].detach().clone())),
        base.risk_head.register_forward_pre_hook(lambda module, args: risks.append(args[0].detach().clone())),
        base.forecast_head.register_forward_pre_hook(lambda module, args: forecasts.append(args[0].detach().clone())),
    ]
    try:
        with torch.no_grad():
            before = model(values, available)
            p = getattr(model, parameter)
            p.copy_(torch.linspace(-.03, .04, p.numel()).reshape_as(p))
            after = model(values, available)
    finally:
        for hook in hooks:
            hook.remove()
    assert not torch.equal(tokens[0][:, :-1], tokens[1][:, :-1])
    torch.testing.assert_close(tokens[0][:, -1], tokens[1][:, -1], rtol=0, atol=0)
    for index in (0, 1):
        torch.testing.assert_close(risks[index], forecasts[index].reshape(2, 17, 64)[:, -1], rtol=0, atol=0)
    assert not torch.equal(risks[0], risks[1]) and not torch.equal(forecasts[0], forecasts[1])
    assert not torch.equal(before["logit"], after["logit"])
    assert not torch.equal(before["forecast"], after["forecast"])
    assert_base_unchanged(base, frozen)


@pytest.mark.parametrize("head", ["logit", "forecast"])
def test_gradients_and_sgd_touch_only_deltas_and_temporary_fast(head):
    base, values, available = case()
    model = EmbeddingModel(base).train()
    frozen = snapshot(base)
    params = tuple(model.persistent_parameters())
    optimizer = torch.optim.SGD(params, lr=.03)
    fast = model.initial_fast(2).requires_grad_()
    model(values, available, fast)[head].square().mean().backward()
    for gradient in (fast.grad, *(p.grad for p in params)):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
    assert_base_unchanged(base, frozen)
    optimizer.step()
    assert all(torch.count_nonzero(p) > 0 for p in params)
    assert_base_unchanged(base, frozen)


def test_double_backward_through_forecast_fast_step_and_multiple_live_graphs():
    base, values, available = case()
    model = EmbeddingModel(base).eval()
    frozen = snapshot(base)
    fast = model.initial_fast(2).requires_grad_()
    first = model(values, available, fast)
    assert_base_unchanged(base, frozen)
    truth = torch.zeros_like(first["forecast"])
    loss = model.forecast_loss(first["forecast"], truth, torch.ones_like(truth, dtype=torch.bool))
    gradient, = torch.autograd.grad(loss, fast, create_graph=True)
    assert_base_unchanged(base, frozen)
    updated = fast - .05 * gradient
    outer = model(values * .9, available, updated)["logit"].square().mean()
    derivatives = torch.autograd.grad(outer, (fast, *model.persistent_parameters()))
    for derivative in derivatives:
        assert torch.isfinite(derivative).all() and derivative.norm() > 0
    assert_base_unchanged(base, frozen)


def test_functional_call_restores_registered_tensors_on_exception():
    base, values, available = case()
    model = EmbeddingModel(base)
    frozen = snapshot(base)

    def fail(module, args):
        raise RuntimeError("intentional forward failure")

    hook = base.encoder.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="intentional forward failure"):
            model(values, available)
    finally:
        hook.remove()
    assert_base_unchanged(base, frozen)
    assert torch.isfinite(model(values, available)["logit"]).all()


@pytest.mark.parametrize("endpoint", [0, 63, 127])
def test_causal_window_does_not_read_changed_future_suffix(endpoint):
    base, values, available = case()
    model = EmbeddingModel(base)
    with torch.no_grad():
        model.delta_weight.fill_(.001)
        model.delta_bias.copy_(torch.linspace(-.01, .02, 64))
    shot = dict(values=values.reshape(-1, 23).numpy().copy(),
                available=available.reshape(-1, 23).numpy().copy())
    changed = {name: tensor.copy() for name, tensor in shot.items()}
    changed["values"][endpoint + 1:] = np.nan
    changed["available"][endpoint + 1:] = False
    x, mask = simple_data.window(shot, endpoint, model.cfg.window)
    y, other_mask = simple_data.window(changed, endpoint, model.cfg.window)
    with torch.no_grad():
        assert_exact(model(torch.from_numpy(x)[None], torch.from_numpy(mask)[None]),
                     model(torch.from_numpy(y)[None], torch.from_numpy(other_mask)[None]))


def test_missing_and_nonfinite_values_are_finite_and_ignore_unavailable_payloads():
    base, values, available = case()
    model = EmbeddingModel(base)
    with torch.no_grad():
        model.delta_weight.fill_(.002)
        model.delta_bias.copy_(torch.linspace(-.01, .02, 64))
    assert_exact(model(values, available), model(values.masked_fill(~available, float("nan")), available))
    missing = torch.zeros_like(available)
    output = model(torch.full_like(values, float("nan")), missing)
    assert all(torch.isfinite(tensor).all() for tensor in output.values())
    assert_exact(output, model(torch.full_like(values, float("inf")), available))
    assert_exact(output, model(values, missing))
    prediction = output["forecast"]
    loss = model.forecast_loss(prediction, torch.full_like(prediction, float("nan")),
                               torch.ones_like(prediction, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.persistent_parameters())


def test_forecast_loss_is_exact_delegate_and_truth_is_detached():
    base, values, available = case()
    model = EmbeddingModel(base)
    prediction = model(values, available)["forecast"]
    truth = torch.randn_like(prediction, requires_grad=True)
    available = torch.ones_like(truth, dtype=torch.bool)
    torch.testing.assert_close(model.forecast_loss(prediction, truth, available),
                               base.forecast_loss(prediction, truth, available), rtol=0, atol=0)
    model.forecast_loss(prediction, truth, available).backward()
    assert truth.grad is None


def test_mode_and_dtype_follow_base_without_freezing_delta_gradients():
    base, values, available = case()
    model = EmbeddingModel(base).double().train()
    assert model.training and not base.training
    assert model.center.dtype == model.scale.dtype == model.delta_weight.dtype == torch.float64
    first = model(values.double(), available)
    model.eval()
    assert_exact(model(values.double(), available), first)
    first["logit"].sum().backward()
    assert model.delta_weight.grad is not None and all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize("change", [
    {"fast_site": "output"}, {"input_view": "window"}, {"input_view": "hybrid"},
    {"input_view": "residual_window"}, {"dim": 32}, {"window": 64}, {"patch": 4},
])
def test_incompatible_source_is_rejected_before_freezing(change):
    base, _, _ = case()
    invalid = Simple23(replace(base.cfg, **change), base.center, base.scale)
    flags = [p.requires_grad for p in invalid.parameters()]
    with pytest.raises(ValueError):
        EmbeddingModel(invalid)
    assert flags == [p.requires_grad for p in invalid.parameters()]


def test_wrong_source_type_and_malformed_embedding_are_rejected():
    with pytest.raises(ValueError):
        EmbeddingModel(nn.Linear(23, 64))
    base, _, _ = case()
    base.embedding = nn.Linear(368, 64, bias=False)
    with pytest.raises(ValueError):
        EmbeddingModel(base)


def test_invalid_inputs_and_fast_are_rejected():
    base, values, available = case()
    model = EmbeddingModel(base)
    for invalid in (torch.zeros(128), torch.zeros(1, 128), torch.zeros(2, 64),
                    torch.zeros(2, 128, dtype=torch.int64), torch.zeros(2, 128, dtype=torch.float64),
                    torch.full((2, 128), float("nan")), "invalid"):
        with pytest.raises(ValueError):
            model(values, available, invalid)
    for invalid in (values[:, :-1], values[:, :, :-1], values.long(), values.double(), None):
        with pytest.raises(ValueError):
            model(invalid, available)
    for invalid in (available.float(), available[:, :-1], None):
        with pytest.raises(ValueError):
            model(values, invalid)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            model.initial_fast(invalid)


def test_existing_inter_sgd_updates_only_embedding_deltas_with_global_radius():
    base, values, _ = case()
    model = EmbeddingModel(base)
    frozen = snapshot(base)
    shot = dict(shot=1, device="EAST", values=values[0].numpy().copy(),
                available=np.ones((128, 23), bool), times=np.arange(128) / 1000.,
                y=np.zeros(128, np.float32), valid=np.ones(128, bool),
                is_disrupt=False, down=None, horizon=125., _sealed_scores=np.full(128, .5))
    result = inter.postshot_update(model, shot, [], "cpu", np.random.default_rng(13),
                                   .5, lr=.03, radius=1., steps=2)
    assert result["accepted"] and result["learned_neural_update"]
    assert result["scope"]["parameter_names"] == ["delta_weight", "delta_bias"]
    assert result["scope"]["parameter_count"] == 23616
    assert result["drift"]["anchor"] == "zero" and result["drift"]["radius"] == 1.
    assert 0 < result["drift"]["after"] <= 1.000001
    assert all(p.grad is None for p in model.persistent_parameters())
    assert_base_unchanged(base, frozen)


def test_hash_locked_qualified_source_is_bitwise_identical_on_cpu():
    path = Path(__file__).resolve().parents[1] / "runs/gradient_aligned512_20260929/source.pt"
    assert path.is_file(), "qualified local source artifact is required"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    base = Simple23(Config(**checkpoint["model_config"]),
                    checkpoint["source_center"], checkpoint["source_scale"]).eval()
    base.load_state_dict(checkpoint["model"])
    model = EmbeddingModel(base).eval()
    frozen = snapshot(base)
    _, values, available = case()
    assert_exact(model(values, available), base(values, available, base.initial_fast(2).requires_grad_()))
    with torch.no_grad():
        assert_exact(model(values, available), base(values, available))
    with torch.inference_mode():
        assert_exact(model(values, available), base(values, available))
    for name, tensor in base.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint["model"][name], rtol=0, atol=0)
    assert_base_unchanged(base, frozen)
