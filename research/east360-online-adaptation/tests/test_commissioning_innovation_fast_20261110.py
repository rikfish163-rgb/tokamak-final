from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.commissioning_innovation_fast_20261110 import (
    arrival_inputs,
    lagged_forward,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture(autouse=True)
def _production_attention_path():
    original = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(original)


def _case(dtype=torch.float32, fast_site="tokens"):
    torch.manual_seed(20261110)
    cfg = Config(dim=8, heads=2, layers=2, patch=8, window=128,
                 fast_site=fast_site, input_view="source")
    base = Simple23(cfg, np.zeros(23), np.ones(23)).to(dtype=dtype)
    base.eval().requires_grad_(False)
    with torch.no_grad():
        base.fast_origin.copy_(torch.linspace(-0.7, 0.7, 2 * cfg.dim, dtype=dtype))
    values = torch.randn(2, 128, 23, dtype=dtype)
    available = torch.ones_like(values, dtype=torch.bool)
    available[0, 0:8] = False
    available[0, 48:56] = False
    values[1, 40:48] = float("nan")  # available but not a valid observation
    values[~available] = float("nan")
    return base, values, available


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fast_site", ["tokens", "output"])
@pytest.mark.parametrize("differentiable", [False, True])
@pytest.mark.parametrize("existing_hook", [False, True])
def test_zero_state_matches_each_original_output_exactly(dtype, fast_site, differentiable, existing_hook):
    base, values, available = _case(dtype, fast_site)
    state = torch.zeros(len(values), base.cfg.dim, dtype=dtype, requires_grad=differentiable)
    saved = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    existing = base.encoder.layers[0].register_forward_hook(lambda *_: None) if existing_hook else None
    hooks = dict(base.encoder.layers[0]._forward_hooks)
    try:
        with torch.set_grad_enabled(differentiable):
            original = base(values, available)
            actual = lagged_forward(base, values, available, state)
            after = base(values, available)
        assert actual.keys() == original.keys()
        for key in original:
            torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
            torch.testing.assert_close(after[key], original[key], rtol=0, atol=0)
        assert dict(base.encoder.layers[0]._forward_hooks) == hooks
        assert all(torch.equal(value, saved[name]) for name, value in base.state_dict().items())
    finally:
        if existing is not None:
            existing.remove()


def test_exact_layer_zero_write_and_original_padding_mask():
    base, values, available = _case(torch.float64)
    batch, dim = len(values), base.cfg.dim
    state = torch.linspace(-1.1, 1.2, batch * dim, dtype=values.dtype).reshape(batch, dim)
    raw_outputs, layer1_inputs = [], []

    def capture_output(_module, _args, output):
        raw_outputs.append(output.detach().clone())

    def capture_next(_module, args, kwargs):
        layer1_inputs.append((args[0].detach().clone(), kwargs["src_key_padding_mask"].clone()))

    first = base.encoder.layers[0].register_forward_hook(capture_output)
    second = base.encoder.layers[1].register_forward_pre_hook(capture_next, with_kwargs=True)
    try:
        base(values, available)
        lagged_forward(base, values, available, state)
    finally:
        first.remove()
        second.remove()
    before, before_mask = layer1_inputs[0]
    after, after_mask = layer1_inputs[1]
    hidden = raw_outputs[1]
    valid = (available & torch.isfinite(values)).reshape(batch, 16, 8 * 23).any(-1)
    eligible = valid[:, 4:] & valid[:, :-4]
    expected = hidden.clone()
    delta = 0.5 * state.tanh()[:, None, :] * (hidden[:, 4:16] - hidden[:, :12])
    expected[:, 4:16] += torch.where(eligible[:, :, None], delta, 0.0)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)
    torch.testing.assert_close(after[:, :4], before[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(after[:, -1], before[:, -1], rtol=0, atol=0)
    assert (after[:, 4:16] != before[:, 4:16]).any()
    assert torch.equal(after_mask, before_mask)
    assert torch.equal(torch.isneginf(after_mask[:, :-1]), ~valid)
    assert (after_mask[:, :-1][valid] == 0).all()
    assert not after_mask[:, -1].any()


def test_zero_state_has_real_gradients_through_both_model_outputs():
    base, values, available = _case(torch.float64)
    state = torch.zeros(len(values), base.cfg.dim, dtype=values.dtype, requires_grad=True)
    output = lagged_forward(base, values, available, state)
    for key in ("logit", "forecast"):
        weights = torch.linspace(-1, 1, output[key].numel(), dtype=values.dtype).reshape_as(output[key])
        gradient, = torch.autograd.grad((output[key] * weights).sum(), state, retain_graph=True)
        assert gradient.shape == state.shape
        assert torch.isfinite(gradient).all()
        assert (gradient.abs().sum(dim=1) > 0).all()
    assert all(p.grad is None for p in base.parameters())
    assert not base.encoder.layers[0]._forward_hooks


def test_layer_hook_is_removed_after_forward_failure():
    base, values, available = _case()
    existing = base.encoder.layers[0].register_forward_hook(lambda *_: None)
    hooks = dict(base.encoder.layers[0]._forward_hooks)

    def fail(_module, _args):
        raise RuntimeError("injected failure")

    failing = base.encoder.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            lagged_forward(base, values, available, torch.ones(2, base.cfg.dim))
        assert dict(base.encoder.layers[0]._forward_hooks) == hooks
    finally:
        failing.remove()
        existing.remove()
    base(values, available)


@pytest.mark.parametrize("changes", [
    {"window": 64}, {"patch": 4}, {"layers": 1}, {"input_view": "window"},
])
def test_model_rejects_unsupported_layout(changes):
    base, _, _ = _case()
    cfg = replace(base.cfg, **changes)
    base = Simple23(cfg, np.zeros(23), np.ones(23)).eval()
    values = torch.zeros(2, cfg.window, 23)
    with pytest.raises(ValueError):
        lagged_forward(base, values, torch.ones_like(values, dtype=torch.bool), torch.zeros(2, cfg.dim))
    assert not base.encoder.layers[0]._forward_hooks


def test_model_rejects_incorrect_state_shape():
    base, values, available = _case()
    with pytest.raises(ValueError):
        lagged_forward(base, values, available, torch.zeros(2, base.cfg.dim + 1))
    assert not base.encoder.layers[0]._forward_hooks


def _scalar_arrival(values, available, center, scale, origin):
    result = np.zeros(46, dtype=np.float64)
    for channel in range(23):
        past = [t for t in range(origin - 127, origin + 1)
                if available[t, channel] and np.isfinite(values[t, channel])]
        if not past:
            continue
        coordinate = lambda x: np.arcsinh((float(x) - center[channel]) / scale[channel])
        reference = coordinate(values[past[-1], channel])
        future = [(t, coordinate(values[t, channel]) - reference)
                  for t in range(origin + 1, origin + 26)
                  if available[t, channel] and np.isfinite(values[t, channel])]
        early = [v for t, v in future if t <= origin + 8]
        late = [v for t, v in future if t >= origin + 18]
        if future:
            result[channel] = np.mean([v for _, v in future])
        if early and late:
            result[23 + channel] = np.mean(late) - np.mean(early)
    return result.astype(np.float32)


def _arrival_case():
    rng = np.random.default_rng(1110)
    values = rng.normal(size=(260, 23)).cumsum(axis=0).astype(np.float32)
    available = rng.random(size=values.shape) > 0.3
    center = np.linspace(-1.2, 1.4, 23)
    scale = np.linspace(0.3, 2.7, 23)
    return values, available, center, scale


def test_arrival_features_match_scalar_math_and_missing_cases():
    values, available, center, scale = _arrival_case()
    q = 150
    # No in-window reference, including a tempting observation just before it.
    available[q - 127:q + 1, 0] = False
    available[q - 128, 0] = True
    # Latest available nonfinite sample is ignored in favor of an earlier valid reference.
    available[q, 1] = True
    values[q, 1] = np.nan
    available[q - 1, 1] = True
    # No future data; early-only and late-only data have no slope.
    available[q + 1:q + 26, 2] = False
    available[q + 1:q + 26, 3:5] = False
    available[q + 1:q + 9, 3] = True
    available[q + 18:q + 26, 4] = True
    available[q + 3, 5] = True
    values[q + 3, 5] = np.inf
    origins = np.array([127, q, 200, q], dtype=np.int64)
    actual = arrival_inputs(values, available, center, scale, origins)
    expected = np.stack([_scalar_arrival(values, available, center, scale, x) for x in origins])
    assert actual.shape == (4, 46) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(
        arrival_inputs(values, available, center, scale, origins.astype(np.uint64)), actual
    )
    np.testing.assert_array_equal(actual[1, [0, 23, 2, 25, 26, 27]], 0)
    assert np.isfinite(actual).all()


def test_arrival_reversal_and_strict_endpoint_causality():
    values, available, center, scale = _arrival_case()
    q = 150
    actual = arrival_inputs(values, available, center, scale, np.array([q]))
    reversed_values, reversed_available = values.copy(), available.copy()
    reversed_values[q + 1:q + 26] = values[q + 1:q + 26][::-1]
    reversed_available[q + 1:q + 26] = available[q + 1:q + 26][::-1]
    reversed_ = arrival_inputs(reversed_values, reversed_available, center, scale, np.array([q]))
    np.testing.assert_allclose(reversed_[:, :23], actual[:, :23], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(reversed_[:, 23:], -actual[:, 23:], rtol=2e-6, atol=2e-6)
    changed_values, changed_available = values.copy(), available.copy()
    changed_values[q + 26:] = np.nan
    changed_available[q + 26:] = ~available[q + 26:]
    changed_values[:q - 127] = np.inf
    changed_available[:q - 127] = ~available[:q - 127]
    changed_values[~available] = np.nan
    untouched = arrival_inputs(changed_values, changed_available, center, scale, np.array([q]))
    np.testing.assert_array_equal(untouched, actual)
    truncated = arrival_inputs(values[:q + 26], available[:q + 26], center, scale, np.array([q]))
    np.testing.assert_array_equal(truncated, actual)


def test_arrival_empty_origins_and_contract_validation():
    values, available, center, scale = _arrival_case()
    empty = arrival_inputs(values, available, center, scale, np.array([], dtype=np.int64))
    assert empty.shape == (0, 46) and empty.dtype == np.float32
    for origins in (np.array([126]), np.array([len(values) - 25]), np.array([127.0]),
                    np.array([[127]])):
        with pytest.raises((ValueError, TypeError)):
            arrival_inputs(values, available, center, scale, origins)
    with pytest.raises((ValueError, TypeError)):
        arrival_inputs(values, available.astype(float), center, scale, np.array([127]))
    bad_scale = scale.copy()
    bad_scale[0] = 0
    with pytest.raises(ValueError):
        arrival_inputs(values, available, center, bad_scale, np.array([127]))
