"""CPU acceptance tests for the independent, flat23 context redesign."""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.model import ContextModel


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


def assert_exact_outputs(actual, expected):
    assert set(actual) == {"logit", "forecast"}
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_zero_offsets_preserve_source_forward_and_nonzero_origin_exactly():
    base, values, available = case()
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    base.requires_grad_(False)
    # Compare the same frozen-source autograd dispatch, not the unfrozen
    # classifier-training path (which can round differently on CPU).
    source_fast = base.initial_fast(2).requires_grad_()
    expected_with_grad = base(values, available, source_fast)
    with torch.no_grad():
        expected = base(values, available)
    model = ContextModel(base)
    fast = model.initial_fast(2)
    assert fast.shape == (2, 128)
    assert not fast.requires_grad
    assert torch.count_nonzero(fast) == 0
    assert model.cfg is base.cfg
    assert model.center is base.center and model.scale is base.scale
    assert model(values, available)["forecast"].shape == (2, 25, 23)
    assert_exact_outputs(model(values, available), expected_with_grad)
    with torch.no_grad():
        assert_exact_outputs(model(values, available), expected)
        assert_exact_outputs(model(values, available, fast), expected)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    fast[0, 0] = 1
    assert fast[1, 0] == 0
    assert torch.count_nonzero(model.initial_fast(2)) == 0


@pytest.mark.parametrize("state", ["fast", "persistent"])
def test_internal_update_changes_shared_representation_and_both_original_heads(state):
    base, values, available = case()
    model = ContextModel(base)
    encoder_inputs, risk_inputs, forecast_inputs = [], [], []
    hooks = [
        base.encoder.register_forward_pre_hook(
            lambda module, args: encoder_inputs.append(args[0].detach().clone())),
        base.risk_head.register_forward_pre_hook(
            lambda module, args: risk_inputs.append(args[0].detach().clone())),
        base.forecast_head.register_forward_pre_hook(
            lambda module, args: forecast_inputs.append(args[0].detach().clone())),
    ]
    try:
        with torch.no_grad():
            before = model(values, available)
            if state == "persistent":
                model.context_map.weight.copy_(
                    torch.linspace(-.2, .2, 1024).reshape(128, 8))
                after = model(values, available)
            else:
                fast = torch.linspace(-.4, .6, 128).expand(2, -1)
                after = model(values, available, fast)
    finally:
        for hook in hooks:
            hook.remove()
    assert not torch.equal(encoder_inputs[0][:, :-1], encoder_inputs[1][:, :-1])
    torch.testing.assert_close(encoder_inputs[0][:, -1], encoder_inputs[1][:, -1], rtol=0, atol=0)
    for index, representation in enumerate(forecast_inputs):
        torch.testing.assert_close(risk_inputs[index], representation.reshape(2, 17, 64)[:, -1],
                                   rtol=0, atol=0)
    assert not torch.equal(risk_inputs[0], risk_inputs[1])
    assert not torch.equal(forecast_inputs[0], forecast_inputs[1])
    assert not torch.equal(before["logit"], after["logit"])
    assert not torch.equal(before["forecast"], after["forecast"])
    assert not any(parameter.requires_grad for parameter in base.parameters())


def test_context_is_fixed_projection_of_valid_source_token_mean_and_innovation():
    base, values, available = case()
    model = ContextModel(base)
    available[:, :16] = False
    available[:, -8:] = False
    with torch.no_grad():
        observed = available & torch.isfinite(values)
        clean = torch.where(observed, values, base.center)
        inputs = torch.cat((torch.asinh((clean - base.center) / base.scale),
                            observed.to(values.dtype)), -1)
        tokens = base.embedding(inputs.reshape(2, 16, -1))
        valid = observed.reshape(2, 16, -1).any(-1)
        summary = torch.cat((tokens.masked_fill(~valid[..., None], 0).sum(1)
                             / valid.sum(1).clamp_min(1)[:, None],
                             tokens[:, 14] - tokens[:, 2]), -1)
        expected = summary @ model.context_projection
    torch.testing.assert_close(model._context(values, available), expected, rtol=0, atol=0)
    assert model.context_projection.shape == (128, 8)
    assert not model.context_projection.requires_grad
    torch.testing.assert_close(model.context_projection.T @ model.context_projection,
                               torch.eye(8), rtol=1e-6, atol=1e-6)
    assert not model._context(values.requires_grad_(), available).requires_grad


@pytest.mark.parametrize("endpoint", [0, 63, 127])
def test_future_suffix_isolation_through_causal_window(endpoint):
    base, values, available = case()
    model = ContextModel(base)
    with torch.no_grad():
        model.context_map.weight.fill_(.07)
    shot = dict(values=values.reshape(-1, 23).numpy().copy(),
                available=available.reshape(-1, 23).numpy().copy())
    changed = {name: value.copy() for name, value in shot.items()}
    changed["values"][endpoint + 1:] = np.nan
    changed["available"][endpoint + 1:] = False
    x, mask = simple_data.window(shot, endpoint, model.cfg.window)
    y, other_mask = simple_data.window(changed, endpoint, model.cfg.window)
    fast = model.initial_fast(1) + .2
    with torch.no_grad():
        assert_exact_outputs(model(torch.from_numpy(x)[None], torch.from_numpy(mask)[None], fast),
                             model(torch.from_numpy(y)[None], torch.from_numpy(other_mask)[None], fast))


@pytest.mark.parametrize("head", ["logit", "forecast"])
def test_each_head_has_nonzero_persistent_and_fast_gradients_but_base_is_frozen(head):
    base, values, available = case()
    model = ContextModel(base)
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    parameters = tuple(model.persistent_parameters())
    optimizer = torch.optim.SGD(parameters, lr=.1)
    fast = model.initial_fast(2).requires_grad_()
    model(values, available, fast)[head].square().mean().backward()
    assert fast.grad is not None and torch.isfinite(fast.grad).all() and fast.grad.norm() > 0
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    assert all(parameter.grad is None for parameter in base.parameters())
    optimizer.step()
    assert torch.count_nonzero(model.context_map.weight) > 0
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_persistent_parameter_scope_and_projection_determinism():
    base, _, _ = case()
    model = ContextModel(base)
    parameters = tuple(model.persistent_parameters())
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert list(dict(model.named_parameters()).keys())[-1] == "context_map.weight"
    assert parameters == (model.context_map.weight,)
    assert sum(parameter.numel() for parameter in parameters) == 1024
    assert all(torch.count_nonzero(parameter) == 0 for parameter in parameters)
    assert not any(parameter.requires_grad for parameter in base.parameters())
    rng_before = torch.random.get_rng_state().clone()
    other = ContextModel(base)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    torch.testing.assert_close(model.context_projection, other.context_projection, rtol=0, atol=0)


def test_second_backward_through_fast_update_reaches_persistent_map():
    base, values, available = case()
    model = ContextModel(base)
    fast = model.initial_fast(2).requires_grad_()
    forecast = model(values, available, fast)["forecast"]
    truth = torch.zeros_like(forecast)
    loss = model.forecast_loss(forecast, truth, torch.ones_like(truth, dtype=torch.bool))
    gradient, = torch.autograd.grad(loss, fast, create_graph=True)
    updated = fast - .05 * gradient
    outer = model(values * .9, available, updated)["logit"].square().mean()
    fast_gradient, map_gradient = torch.autograd.grad(outer, (fast, model.context_map.weight))
    for derivative in (fast_gradient, map_gradient):
        assert torch.isfinite(derivative).all() and derivative.norm() > 0
    assert all(parameter.grad is None for parameter in base.parameters())


def test_all_missing_nonfinite_inputs_and_targets_are_finite():
    base, values, available = case()
    model = ContextModel(base)
    with torch.no_grad():
        model.context_map.weight.fill_(.1)
    missing = torch.zeros_like(available)
    output = model(torch.full_like(values, float("nan")), missing)
    for tensor in output.values():
        assert torch.isfinite(tensor).all()
    assert torch.count_nonzero(model._context(values, missing)) == 0
    assert_exact_outputs(output, model(torch.full_like(values, float("inf")), available))
    assert_exact_outputs(output, model(torch.randn_like(values), missing))
    loss = model.forecast_loss(output["forecast"], torch.full_like(output["forecast"], float("nan")),
                               torch.ones_like(output["forecast"], dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(model.context_map.weight.grad).all()


def test_forecast_loss_is_exact_delegate_and_detaches_truth():
    base, values, available = case()
    model = ContextModel(base)
    prediction = model(values, available)["forecast"]
    truth = torch.randn_like(prediction, requires_grad=True)
    mask = torch.ones_like(truth, dtype=torch.bool)
    torch.testing.assert_close(model.forecast_loss(prediction, truth, mask),
                               base.forecast_loss(prediction, truth, mask), rtol=0, atol=0)
    model.forecast_loss(prediction, truth, mask).backward()
    assert truth.grad is None


def test_modes_are_consistent_and_base_remains_in_inference_mode():
    base, values, available = case()
    model = ContextModel(base)
    fast = model.initial_fast(2) + .3
    with torch.no_grad():
        model.context_map.weight.fill_(.1)
    model.train()
    assert model.training and not base.training
    train_output = model(values, available, fast)
    model.eval()
    assert not model.training and not base.training
    assert_exact_outputs(model(values, available, fast), train_output)
    with torch.no_grad():
        eval_output = model(values, available, fast)
        model.train()
        assert_exact_outputs(model(values, available, fast), eval_output)
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_missing_values_and_other_batch_member_do_not_change_context_or_prediction():
    base, values, available = case()
    model = ContextModel(base)
    with torch.no_grad():
        model.context_map.weight.fill_(.1)
    changed = values.masked_fill(~available, float("nan"))
    assert_exact_outputs(model(values, available), model(changed, available))
    unrelated = changed.clone()
    unrelated[1] *= 1000
    with torch.no_grad():
        original = model(changed, available)
        altered = model(unrelated, available)
        for head in original:
            torch.testing.assert_close(original[head][0], altered[head][0], rtol=0, atol=0)


@pytest.mark.parametrize("change", [{"fast_site": "output"}, {"input_view": "window"}, {"dim": 32}])
def test_incompatible_source_contract_is_rejected(change):
    base, _, _ = case()
    incompatible = Simple23(replace(base.cfg, **change), base.center, base.scale)
    with pytest.raises(ValueError):
        ContextModel(incompatible)


def test_invalid_input_fast_shape_and_context_dimension_are_rejected():
    base, values, available = case()
    model = ContextModel(base)
    for dimension in (0, -1, 129, 1.5, True):
        with pytest.raises(ValueError):
            ContextModel(base, context_dim=dimension)
    for invalid in (torch.zeros(128), torch.zeros(1, 128), torch.zeros(2, 64)):
        with pytest.raises(ValueError):
            model(values, available, invalid)
    with pytest.raises(ValueError):
        model(values[:, :-1], available[:, :-1])
    with pytest.raises(ValueError):
        model(values, available.float())


def test_locked_qualified_source_checkpoint_identity_on_cpu():
    checkpoint_path = Path(__file__).resolve().parents[1] / "runs/gradient_aligned512_20260929/source.pt"
    assert checkpoint_path.is_file(), "locked source artifact is required for this acceptance test"
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == (
        "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02")
    # Only the hash-verified, trusted local artifact is deserialized.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = Simple23(Config(**checkpoint["model_config"]),
                    checkpoint["source_center"], checkpoint["source_scale"]).eval()
    base.load_state_dict(checkpoint["model"])
    _, values, available = case()
    base.requires_grad_(False)
    source_fast = base.initial_fast(2).requires_grad_()
    expected_with_grad = base(values, available, source_fast)
    with torch.no_grad():
        expected = base(values, available)
    model = ContextModel(base).eval()
    assert_exact_outputs(model(values, available), expected_with_grad)
    with torch.no_grad():
        assert_exact_outputs(model(values, available), expected)
        assert_exact_outputs(model(values, available, model.initial_fast(2)), expected)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)
