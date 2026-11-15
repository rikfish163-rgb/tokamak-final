from dataclasses import replace

import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.two_band_fast_20261110 import banded_forward


def make_case(dtype=torch.float32):
    torch.manual_seed(20261110)
    cfg = Config(window=16, patch=4, dim=8, heads=2, layers=1, horizon=3,
                 fast_site="tokens", input_view="source")
    base = Simple23(cfg, torch.zeros(23), torch.ones(23)).to(dtype=dtype)
    base.eval().requires_grad_(False)
    with torch.no_grad():
        base.fast_origin.copy_(torch.linspace(-0.7, 0.7, 2 * cfg.dim, dtype=dtype))
    values = torch.randn(2, cfg.window, 23, dtype=dtype)
    available = torch.ones_like(values, dtype=torch.bool)
    available[0, :cfg.patch] = False
    available[1, -cfg.patch:] = False
    available[:, 5:7, 3] = False
    values[~available] = float("nan")
    return base, values, available


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_delta_exactly_matches_default_and_preserves_state(dtype):
    base, values, available = make_case(dtype)
    delta = torch.zeros(len(values), 2, 2 * base.cfg.dim, dtype=dtype)
    original_values, original_available = values.clone(), available.clone()
    state = {name: value.clone() for name, value in base.state_dict().items()}
    existing = base.embedding.register_forward_hook(lambda *_: None)
    hooks = dict(base.embedding._forward_hooks)
    try:
        expected = base(values, available)
        actual = banded_forward(base, values, available, delta)
        after = base(values, available)
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(after[key], expected[key], rtol=0, atol=0)
        assert dict(base.embedding._forward_hooks) == hooks
        assert all(torch.equal(value, state[name]) for name, value in base.state_dict().items())
        torch.testing.assert_close(values, original_values, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(available, original_available)
    finally:
        existing.remove()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equal_bands_match_original_uniform_fast_state(dtype):
    base, values, available = make_case(dtype)
    batch, dim = len(values), base.cfg.dim
    shared = torch.linspace(-1.2, 1.3, batch * 2 * dim, dtype=dtype).reshape(batch, 2 * dim)
    delta = shared[:, None, :].expand(-1, 2, -1)
    original = base(values, available, base.initial_fast(batch) + shared)
    actual = banded_forward(base, values, available, delta)
    tolerance = 2e-6 if dtype == torch.float32 else 3e-15
    for key in original:
        torch.testing.assert_close(actual[key], original[key], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("band", [0, 1])
def test_direct_effect_is_confined_to_selected_time_band(band):
    base, values, available = make_case()
    batch, dim = len(values), base.cfg.dim
    patches = base.cfg.window // base.cfg.patch
    delta = torch.zeros(batch, 2, 2 * dim)
    delta[:, band] = torch.linspace(-1.1, 1.3, batch * 2 * dim).reshape(batch, 2 * dim)
    embeddings, encoder_inputs = [], []

    def capture_embedding(_module, _args, output):
        embeddings.append(output.detach().clone())

    def capture_encoder(_module, args, kwargs):
        encoder_inputs.append((args[0].detach().clone(), kwargs["src_key_padding_mask"].clone()))

    embedding_hook = base.embedding.register_forward_hook(capture_embedding)
    encoder_hook = base.encoder.register_forward_pre_hook(capture_encoder, with_kwargs=True)
    try:
        base(values, available)
        banded_forward(base, values, available, delta)
    finally:
        embedding_hook.remove()
        encoder_hook.remove()

    before, before_mask = encoder_inputs[0]
    after, after_mask = encoder_inputs[1]
    start, end = band * (patches // 2), (band + 1) * (patches // 2)
    other_start, other_end = (1 - band) * (patches // 2), (2 - band) * (patches // 2)
    gain, bias = delta[:, band].chunk(2, dim=-1)
    expected = embeddings[1][:, start:end] * (1 + 0.1 * gain[:, None].tanh())
    expected = expected + 0.1 * bias[:, None].tanh() + base.position[:, start:end]
    torch.testing.assert_close(after[:, start:end], expected, rtol=0, atol=0)
    assert not torch.equal(after[:, start:end], before[:, start:end])
    torch.testing.assert_close(after[:, other_start:other_end], before[:, other_start:other_end], rtol=0, atol=0)
    torch.testing.assert_close(after[:, -1], before[:, -1], rtol=0, atol=0)
    assert torch.equal(after_mask, before_mask)
    valid = available.reshape(batch, patches, -1).any(-1)
    valid = torch.cat((valid, torch.ones(batch, 1, dtype=torch.bool)), dim=1)
    assert torch.equal(after_mask, ~valid)


def test_both_band_gain_and_bias_receive_gradients():
    base, values, available = make_case()
    delta = torch.randn(len(values), 2, 2 * base.cfg.dim).requires_grad_()
    output = banded_forward(base, values, available, delta)
    weights = torch.linspace(-1.0, 1.0, output["forecast"].numel()).reshape_as(output["forecast"])
    loss = (output["forecast"] * weights).sum() + output["logit"].sum()
    (gradient,) = torch.autograd.grad(loss, delta)
    assert gradient.shape == delta.shape
    assert torch.isfinite(gradient).all()
    assert (gradient.reshape(len(values), 2, 2, base.cfg.dim).abs().sum(-1) > 0).all()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert not base.embedding._forward_hooks


def test_embedding_hook_is_removed_on_forward_exception():
    base, values, available = make_case()
    delta = torch.zeros(len(values), 2, 2 * base.cfg.dim)
    existing = base.embedding.register_forward_hook(lambda *_: None)
    hooks = dict(base.embedding._forward_hooks)

    def fail(_module, _args):
        raise RuntimeError("injected encoder failure")

    failing = base.encoder.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected encoder failure"):
            banded_forward(base, values, available, delta)
        assert dict(base.embedding._forward_hooks) == hooks
    finally:
        failing.remove()
        existing.remove()
    base(values, available)


@pytest.mark.parametrize("change", [
    {"fast_site": "output"},
    {"input_view": "window"},
    {"window": 12},
])
def test_rejects_unsupported_configuration(change):
    base, _, _ = make_case()
    cfg = replace(base.cfg, **change)
    base = Simple23(cfg, torch.zeros(23), torch.ones(23)).eval()
    values = torch.randn(2, cfg.window, 23)
    available = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(ValueError):
        banded_forward(base, values, available, torch.zeros(2, 2, 2 * cfg.dim))
    assert not base.embedding._forward_hooks


@pytest.mark.parametrize("shape", [(1, 2, 16), (2, 1, 16), (2, 2, 17)])
def test_rejects_mismatched_delta_shape(shape):
    base, values, available = make_case()
    with pytest.raises(ValueError):
        banded_forward(base, values, available, torch.zeros(shape))
    assert not base.embedding._forward_hooks
