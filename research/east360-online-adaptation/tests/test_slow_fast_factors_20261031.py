"""CPU checks of a post-hoc intervention, not a deployment policy."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.slow_fast_attribution_20261031.factors import (
    logits_with_components,
)


@pytest.fixture(autouse=True)
def math_attention_path():
    # Match the replay configuration and restore process-wide state afterward.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture
def example():
    # A real two-block source model, but with a small embedding dimension.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31031)
        cfg = Config(
            dim=8,
            heads=2,
            layers=2,
            window=128,
            patch=8,
            horizon=25,
            fast_site="tokens",
            input_view="source",
        )
        model = (
            ContextAdapter(Simple23(cfg, torch.zeros(23), torch.ones(23)))
            .double()
            .eval()
        )
        with torch.no_grad():
            model.slow.weight.normal_(0, 0.2)
            model.memory.output.weight.normal_(0, 0.12)
            model.base.fast_origin.normal_(0, 0.6)
            values = torch.randn(3, 128, 23, dtype=torch.float64)
            available = torch.ones_like(values, dtype=torch.bool)
            available[1, -8:] = False  # last observation token is missing
            available[2, 8:16] = False  # an earlier token is missing
            h, valid = model.tokens(values, available)
            delta = torch.randn(3, 92, dtype=torch.float64)
        slow = model.slow.weight.detach().clone()
        output = model.memory.output.weight.detach().clone()
    return model, h, valid, delta, slow, output


def _snapshot(model, *tensors):
    return (
        {key: value.detach().clone() for key, value in model.state_dict().items()},
        [(module, module.training) for module in model.modules()],
        [
            (
                parameter,
                parameter.requires_grad,
                parameter._version,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in model.parameters()
        ],
        [(tensor, tensor.detach().clone(), tensor._version) for tensor in tensors],
    )


def _assert_unchanged(model, snapshot):
    state, modes, parameters, tensors = snapshot
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert all(module.training == training for module, training in modes)
    for parameter, requires_grad, version, grad in parameters:
        assert (
            parameter.requires_grad == requires_grad and parameter._version == version
        )
        if grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
    for tensor, value, version in tensors:
        torch.testing.assert_close(tensor, value, rtol=0, atol=0)
        assert tensor._version == version


@pytest.mark.parametrize("zero_delta", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("weights_require_grad", [False, True])
def test_identical_components_exactly_reproduce_real_risk(
    example, zero_delta, grad_enabled, weights_require_grad
):
    model, h, valid, delta, slow, output = example
    if zero_delta:
        delta = torch.zeros_like(delta)
    slow.requires_grad_(weights_require_grad)
    output.requires_grad_(weights_require_grad)
    before = _snapshot(model, h, valid, delta, slow, output)
    with torch.set_grad_enabled(grad_enabled):
        expected = model.risk(h, valid, delta)
        actual = logits_with_components(model, h, valid, delta, slow, output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_unchanged(model, before)


def test_default_64_dimension_exactly_matches_the_real_risk():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3164)
        cfg = Config(fast_site="tokens", input_view="source")
        model = ContextAdapter(Simple23(cfg, torch.zeros(23), torch.ones(23))).eval()
        with torch.no_grad():
            model.slow.weight.normal_(0, 0.1)
            model.memory.output.weight.normal_(0, 0.1)
            model.base.fast_origin.normal_(0, 0.3)
            values = torch.randn(2, 128, 23)
            h, valid = model.tokens(values, torch.ones_like(values, dtype=torch.bool))
            delta = torch.randn(2, 92)
            actual = logits_with_components(
                model,
                h,
                valid,
                delta,
                model.slow.weight.detach().clone(),
                model.memory.output.weight.detach().clone(),
            )
            expected = model.risk(h, valid, delta)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_fully_frozen_adapter_matches_and_preserves_single_item_batch(
    example, grad_enabled
):
    model, h, valid, delta, slow, output = example
    model.requires_grad_(False)
    h, valid, delta = h[:1], valid[:1], delta[:1]
    before = _snapshot(model, h, valid, delta, slow, output)
    with torch.set_grad_enabled(grad_enabled):
        actual = logits_with_components(model, h, valid, delta, slow, output)
        expected = model.risk(h, valid, delta)
    assert actual.shape == (1,)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_unchanged(model, before)


def _row_reference(model, h, valid, delta, slow, output):
    """Compute each intervention row separately without the helper or F.linear."""
    result = []
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for row in range(len(h)):
            sequence = h[row : row + 1].clone()
            observed = valid[row, -2]
            sequence[0, -2] = h[row, -2] + (slow @ h[row, -2]) * observed
            sequence[0, -2] = sequence[0, -2] + (output @ delta[row]) * observed
            encoded = model.base.encoder.layers[1](
                sequence, src_key_padding_mask=~valid[row : row + 1]
            )
            gain, bias = model.base.initial_fast(1).chunk(2, -1)
            normalized = model.base.norm(encoded) * (1 + 0.1 * gain[:, None].tanh())
            normalized = normalized + 0.1 * bias[:, None].tanh()
            result.append(model.base.risk_head(normalized[:, -1]).squeeze(-1))
    return torch.cat(result)


def test_rowwise_factorial_reference_includes_the_combined_interaction(example):
    model, h, valid, delta, slow, output = example
    # These are external matrices, deliberately different from the model's own.
    slow, output = -1.3 * slow, 1.7 * output
    zero_slow, zero_output = torch.zeros_like(slow), torch.zeros_like(output)
    before = _snapshot(model, h, valid, delta, slow, output, zero_slow, zero_output)
    results, reference = {}, {}
    with torch.no_grad():
        for name, s, o in [
            ("neither", zero_slow, zero_output),
            ("slow", slow, zero_output),
            ("output", zero_slow, output),
            ("both", slow, output),
        ]:
            results[name] = logits_with_components(model, h, valid, delta, s, o)
            reference[name] = _row_reference(model, h, valid, delta, s, o)
            torch.testing.assert_close(
                results[name], reference[name], rtol=1e-12, atol=1e-12
            )
    interaction = (
        results["both"] - results["slow"] - results["output"] + results["neither"]
    )
    reference_interaction = (
        reference["both"]
        - reference["slow"]
        - reference["output"]
        + reference["neither"]
    )
    torch.testing.assert_close(
        interaction, reference_interaction, rtol=1e-12, atol=1e-12
    )
    assert interaction.abs().max() > 1e-8
    for result in results.values():
        torch.testing.assert_close(result[1], results["neither"][1], rtol=0, atol=0)
    _assert_unchanged(model, before)


def test_missing_last_token_gates_both_corrections_before_the_encoder(example):
    model, h, valid, delta, slow, output = example
    captured = []

    def record_input(_module, args, kwargs):
        captured.append(
            (args[0].detach().clone(), kwargs["src_key_padding_mask"].clone())
        )

    handle = model.base.encoder.layers[1].register_forward_pre_hook(
        record_input, with_kwargs=True
    )
    try:
        with torch.no_grad():
            for s, o in [
                (slow, torch.zeros_like(output)),
                (torch.zeros_like(slow), output),
                (slow, output),
            ]:
                logits_with_components(model, h, valid, delta, s, o)
    finally:
        handle.remove()

    assert len(captured) == 3
    for (sequence, padding), (s, o) in zip(
        captured,
        [
            (slow, torch.zeros_like(output)),
            (torch.zeros_like(slow), output),
            (slow, output),
        ],
    ):
        last = h[:, -2] + F.linear(h[:, -2], s) * valid[:, -2, None]
        last = last + F.linear(delta, o) * valid[:, -2, None]
        torch.testing.assert_close(sequence[:, -2], last, rtol=0, atol=0)
        torch.testing.assert_close(sequence[:, :-2], h[:, :-2], rtol=0, atol=0)
        torch.testing.assert_close(sequence[:, -1], h[:, -1], rtol=0, atol=0)
        assert torch.equal(padding, ~valid)
        torch.testing.assert_close(sequence[1, -2], h[1, -2], rtol=0, atol=0)
        assert not torch.equal(sequence[0, -2], h[0, -2])
        assert not torch.equal(sequence[2, -2], h[2, -2])


@pytest.mark.parametrize("component", ["slow", "output"])
def test_each_nonzero_component_can_flip_a_logit(example, component):
    model, h, valid, delta, slow, output = example
    zero_slow, zero_output = torch.zeros_like(slow), torch.zeros_like(output)
    s = slow if component == "slow" else zero_slow
    o = output if component == "output" else zero_output
    with torch.no_grad():
        baseline = logits_with_components(
            model, h, valid, delta, zero_slow, zero_output
        )[0]
        intervened = logits_with_components(model, h, valid, delta, s, o)[0]
        assert (baseline - intervened).abs() > 1e-8
        # Move the fixed head's bias once during test setup so zero lies between
        # the two outputs; the helper must not change it during either call.
        model.base.risk_head.bias.sub_((baseline + intervened) / 2)
        before = _snapshot(model, h, valid, delta, s, o)
        baseline = logits_with_components(
            model, h, valid, delta, zero_slow, zero_output
        )[0]
        intervened = logits_with_components(model, h, valid, delta, s, o)[0]
    assert baseline * intervened < 0
    _assert_unchanged(model, before)


def test_autograd_reaches_both_supplied_matrices_and_inputs_without_mutation(example):
    model, h, valid, delta, slow, output = example
    h = h.detach().requires_grad_()
    delta = delta.detach().requires_grad_()
    slow.requires_grad_()
    output.requires_grad_()
    before = _snapshot(model, h, valid, delta, slow, output)
    logits = logits_with_components(model, h, valid, delta, slow, output)
    grads = torch.autograd.grad(logits.sum(), (h, delta, slow, output))
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)
    _assert_unchanged(model, before)


def test_fully_frozen_adapter_allows_gradients_to_external_matrices(example):
    model, h, valid, delta, slow, output = example
    model.requires_grad_(False)
    slow.requires_grad_()
    output.requires_grad_()
    before = _snapshot(model, h, valid, delta, slow, output)
    expected = model.risk(h, valid, delta)
    actual = logits_with_components(model, h, valid, delta, slow, output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grads = torch.autograd.grad(actual.sum(), (slow, output))
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)
    _assert_unchanged(model, before)


def test_detached_inputs_do_not_create_an_artificial_gradient_leaf(example):
    model, h, valid, delta, slow, output = example
    assert model.slow.weight.requires_grad and model.memory.output.weight.requires_grad
    before = _snapshot(model, h, valid, delta, slow, output)
    with torch.enable_grad():
        actual = logits_with_components(model, h, valid, delta, slow, output)
    assert not actual.requires_grad
    assert not torch.backends.mha.get_fastpath_enabled()
    _assert_unchanged(model, before)


def test_unused_components_do_not_have_to_share_the_readout_device(example):
    model, h, valid, delta, slow, output = example
    with torch.no_grad():
        expected = logits_with_components(model, h, valid, delta, slow, output)
        # These modules are not consulted when h and both weights are supplied.
        model.base.encoder.layers[0].to("meta")
        model.slow.to("meta")
        model.memory.output.to("meta")
        actual = logits_with_components(model, h, valid, delta, slow, output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "component", ["encoder", "norm", "head", "initial_fast", "buffer"]
)
def test_relevant_model_device_must_match_inputs(example, component):
    model, h, valid, delta, slow, output = example
    if component == "encoder":
        model.base.encoder.layers[1].to("meta")
    elif component == "norm":
        model.base.norm.to("meta")
    elif component == "head":
        model.base.risk_head.to("meta")
    elif component == "initial_fast":
        model.base.fast_origin = torch.nn.Parameter(model.base.fast_origin.to("meta"))
    elif component == "buffer":
        model.base.norm.register_buffer("device_probe", torch.empty(0, device="meta"))
    with pytest.raises(ValueError, match="device"):
        logits_with_components(model, h, valid, delta, slow, output)


@pytest.mark.parametrize(
    "invalid",
    [
        "h_rank",
        "h_tokens",
        "h_dim",
        "empty_batch",
        "valid_shape",
        "valid_dtype",
        "delta_batch",
        "delta_width",
        "delta_dtype",
        "slow_shape",
        "output_shape",
        "slow_nan",
        "output_inf",
        "matrix_dtype",
        "matrix_device",
        "slow_device",
        "valid_device",
        "delta_device",
        "not_tensor",
    ],
)
def test_invalid_inputs_are_rejected(example, invalid):
    model, h, valid, delta, slow, output = example
    if invalid == "h_rank":
        h = h[0]
    elif invalid == "h_tokens":
        h = h[:, :-1]
    elif invalid == "h_dim":
        h = h[:, :, :-1]
    elif invalid == "empty_batch":
        h, valid, delta = h[:0], valid[:0], delta[:0]
    elif invalid == "valid_shape":
        valid = valid[:, :-1]
    elif invalid == "valid_dtype":
        valid = valid.float()
    elif invalid == "delta_batch":
        delta = delta[:-1]
    elif invalid == "delta_width":
        delta = delta[:, :-1]
    elif invalid == "delta_dtype":
        delta = delta.float()
    elif invalid == "slow_shape":
        slow = slow[:-1]
    elif invalid == "output_shape":
        output = output[:, :-1]
    elif invalid == "slow_nan":
        slow[0, 0] = float("nan")
    elif invalid == "output_inf":
        output[0, 0] = float("inf")
    elif invalid == "matrix_dtype":
        slow = slow.float()
    elif invalid == "matrix_device":
        output = output.to("meta")
    elif invalid == "slow_device":
        slow = slow.to("meta")
    elif invalid == "valid_device":
        valid = valid.to("meta")
    elif invalid == "delta_device":
        delta = delta.to("meta")
    elif invalid == "not_tensor":
        slow = None
    modes = [(module, module.training) for module in model.modules()]
    with pytest.raises(ValueError):
        logits_with_components(model, h, valid, delta, slow, output)
    assert all(module.training == training for module, training in modes)


@pytest.mark.parametrize(
    "invalid",
    ["layers", "fast_site", "input_view", "readout", "root_train", "child_train"],
)
def test_invalid_configuration_or_training_mode_is_rejected_without_mode_changes(
    example, invalid
):
    model, h, valid, delta, slow, output = example
    if invalid in ("layers", "fast_site", "input_view"):
        value = {"layers": 1, "fast_site": "output", "input_view": "window"}[invalid]
        model.base.cfg = model.cfg = replace(model.cfg, **{invalid: value})
    elif invalid == "readout":
        model.readout = "shape"
    elif invalid == "root_train":
        model.train()
    elif invalid == "child_train":
        model.memory.output.train()
    before = _snapshot(model, h, valid, delta, slow, output)
    with pytest.raises(ValueError):
        logits_with_components(model, h, valid, delta, slow, output)
    _assert_unchanged(model, before)
