"""CPU-only, synthetic contracts for the isolated patch intervention."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.risk_aligned_online.simple23 import Config, Simple23
import adjacent_patch_operator_20261106 as op


@pytest.fixture
def base():
    torch.manual_seed(802)
    model = Simple23(
        Config(fast_site="tokens", fast_rule="normalized", input_view="source"),
        torch.linspace(-0.4, 0.3, 23),
        torch.linspace(0.7, 1.4, 23),
    ).double().eval().requires_grad_(False)
    with torch.no_grad():
        model.fast_origin.copy_(torch.linspace(-0.4, 0.3, 128, dtype=torch.float64))
        model.cls.copy_(torch.linspace(-0.2, 0.2, 64, dtype=torch.float64))
    return model


def synthetic(base, batch=4):
    generator = torch.Generator().manual_seed(719)
    normalized = torch.randn(batch, 128, 23, generator=generator, dtype=base.center.dtype)
    x = base.center + base.scale * normalized.sinh()
    available = torch.rand(batch, 128, 23, generator=generator) > 0.27
    return x, available


def reference(base, x, available, projection):
    """Literal per-pair computation, independent of the vectorized implementation."""
    batch = len(x)
    features = x.new_zeros(batch, 2, 16, 8)
    support = torch.zeros(batch, 2, dtype=torch.bool)
    observed = available & torch.isfinite(x)
    level = torch.asinh((torch.where(observed, x, base.center) - base.center) / base.scale)
    weights = base.embedding.weight.reshape(64, 8, 46)[:, :, :23].reshape(64, 184)
    for row in range(batch):
        eligible = []
        for patch in range(1, 16):
            prev = slice((patch - 1) * 8, patch * 8)
            curr = slice(patch * 8, (patch + 1) * 8)
            common = observed[row, prev] & observed[row, curr]
            if int(common.sum()) < 8:
                continue
            previous = level[row, prev].masked_fill(~common, 0)
            current = level[row, curr].masked_fill(~common, 0)
            eligible.append((patch, common, current - previous, current))
        for mode in range(2):
            if len(eligible) < 2:
                continue
            coord_sq = sum(item[2 + mode].square().sum() for item in eligible)
            coord_count = sum(item[1].sum() for item in eligible)
            if (coord_sq / coord_count).sqrt() < 1e-3:
                continue
            latents = torch.stack([
                F.linear(item[2 + mode].reshape(184), weights) @ projection
                for item in eligible
            ])
            latent_rms = (latents.square().sum() / len(eligible)).sqrt()
            if not torch.isfinite(latent_rms) or latent_rms <= 0:
                continue
            support[row, mode] = True
            for item, latent in zip(eligible, latents, strict=True):
                features[row, mode, item[0]] = latent / latent_rms
    return features, support


def test_basis_is_oriented_cpu_float64_value_only_svd(base):
    projection = op.basis(base)
    weights = base.embedding.weight.reshape(64, 8, 46)[:, :, :23].reshape(64, 184)
    u, _, _ = torch.linalg.svd(weights.cpu().double(), full_matrices=False)
    u = u[:, :8]
    pivots = u.abs().argmax(dim=0)
    u = u * torch.sign(u[pivots, torch.arange(8)])[None, :]
    torch.testing.assert_close(projection, u, rtol=0, atol=1e-12)
    torch.testing.assert_close(projection.T @ projection, torch.eye(8, dtype=u.dtype), atol=1e-12, rtol=0)
    assert not projection.requires_grad and projection.shape == (64, 8)
    with torch.no_grad():
        base.embedding.weight.reshape(64, 8, 46)[:, :, 23:].normal_()
        base.embedding.bias.normal_()
    torch.testing.assert_close(op.basis(base), projection, rtol=0, atol=0)


def test_reference_formula_and_native_tokens(base):
    x, available = synthetic(base)
    available[0, 72:80] = False
    x[1, 33, 9] = float("nan")
    x[2, 18, 3] = float("inf")
    projection = op.basis(base)
    packed = op.encode(base, x, available, projection)
    expected, support = reference(base, x, available, projection)
    assert isinstance(packed, op.Packed)
    assert packed.tokens.shape == (4, 17, 64)
    assert packed.valid.shape == (4, 17) and packed.valid.dtype == torch.bool
    assert packed.features.shape == (4, 2, 16, 8)
    assert packed.support.shape == (4, 2) and packed.support.dtype == torch.bool
    assert packed.projection is projection
    torch.testing.assert_close(packed.features, expected, atol=2e-12, rtol=2e-12)
    assert torch.equal(packed.support, support)
    assert not packed.features[:, :, 0].any()
    observed = available & torch.isfinite(x)
    expected_valid = torch.cat((observed.reshape(4, 16, 184).any(-1), torch.ones(4, 1, dtype=torch.bool)), 1)
    assert torch.equal(packed.valid, expected_valid)
    clean = torch.where(observed, x, base.center)
    normalized = torch.asinh((clean - base.center) / base.scale)
    native_input = torch.cat((normalized, observed.to(normalized)), -1).reshape(4, 16, 368)
    expected_tokens = torch.cat((base.embedding(native_input), base.cls.expand(4, -1, -1)), 1) + base.position
    torch.testing.assert_close(packed.tokens, expected_tokens, atol=0, rtol=0)
    torch.testing.assert_close(packed.tokens[:, -1], (base.cls + base.position[:, -1:])[:, 0].expand(4, -1), atol=0, rtol=0)


def test_constant_normalized_signal_and_changing_mask_separate_modes(base):
    level = torch.linspace(-0.7, 1.1, 23, dtype=base.center.dtype)
    x = (base.center + base.scale * level.sinh()).expand(2, 128, 23).clone()
    _, available = synthetic(base, 2)
    p = op.encode(base, x, available, op.basis(base))
    assert not p.support[:, 0].any()
    assert p.support[:, 1].all()
    assert not p.features[:, 0].any()
    eye = torch.eye(8, dtype=x.dtype).reshape(1, 64).expand(2, -1)
    assert not op.delta(p, eye, 0).any()
    torch.testing.assert_close(op.delta(p, eye, 1), p.features[:, 1] @ p.projection.T)
    assert op.delta(p, eye, 1).norm() > 0
    for mode in range(2):
        assert not op.delta(p, torch.zeros_like(eye), mode).any()


def test_noncommon_values_and_embedding_mask_columns_are_not_in_the_edit(base):
    x, available = synthetic(base, 1)
    available[:] = False
    available[:, :, :10] = True
    available[:, ::16, 10:15] = True
    p = op.basis(base)
    original = op.encode(base, x, available, p)
    changed_x = x.clone()
    changed_x[:, ::16, 10:15] += 1000
    changed_x[~available] = float("nan")
    changed = op.encode(base, changed_x, available, p)
    torch.testing.assert_close(original.features, changed.features, atol=0, rtol=0)
    assert not torch.equal(original.tokens, changed.tokens)
    with torch.no_grad():
        base.embedding.weight.reshape(64, 8, 46)[:, :, 23:].add_(2.0)
        base.embedding.bias.add_(4.0)
    changed_embedding = op.encode(base, x, available, p)
    torch.testing.assert_close(original.features, changed_embedding.features, atol=0, rtol=0)
    assert not torch.equal(original.tokens, changed_embedding.tokens)


def test_eligibility_floor_empty_and_zero_projection_response(base):
    projection = op.basis(base)
    level = torch.zeros(6, 128, 23, dtype=base.center.dtype)
    level[2] = 0.000999
    level[3] = 0.001001
    for patch in range(16):
        level[4, patch * 8:(patch + 1) * 8] = patch * 0.000999
        level[5, patch * 8:(patch + 1) * 8] = patch * 0.001001
    x = base.center + base.scale * level.sinh()
    available = torch.ones_like(x, dtype=torch.bool)
    available[0] = False
    available[1, 24:] = False
    available[1, :24] = False
    available[1, 0, :8] = True
    available[1, 8, :8] = True  # one eligible pair, not two
    p = op.encode(base, x, available, projection)
    assert not p.support[0].any() and not p.support[1].any()
    assert not p.support[2, 1] and p.support[3, 1]
    assert not p.support[4, 0] and p.support[5, 0]
    assert not p.features[0].any() and not p.features[1].any()
    assert p.valid[0, -1] and not p.valid[0, :-1].any()

    sparse = torch.zeros(1, 128, 23, dtype=torch.bool)
    sparse[0, [0, 8, 16], :7] = True
    constant = (base.center + base.scale).expand(1, 128, 23).clone()
    assert not op.encode(base, constant, sparse, projection).support.any()
    sparse[0, [0, 8, 16], 7] = True
    assert op.encode(base, constant, sparse, projection).support[0, 1]

    with torch.no_grad():
        base.embedding.weight.reshape(64, 8, 46)[:, :, :23].zero_()
    dead = op.encode(base, constant, sparse, projection)
    assert not dead.support.any() and not dead.features.any()


def test_eligible_zero_latent_counts_in_per_patch_rms(base):
    level = torch.zeros(1, 128, 23, dtype=base.center.dtype)
    level[:, 16:24, :8] = 1
    x = base.center + base.scale * level.sinh()
    available = torch.zeros_like(x, dtype=torch.bool)
    available[:, [0, 8, 16], :8] = True
    p = op.encode(base, x, available, op.basis(base))
    assert p.support.all()
    assert not p.features[:, :, 1].any()  # Eligible, but its latent is zero.
    for mode in range(2):
        torch.testing.assert_close(p.features[0, mode].square().sum().sqrt(),
                                   torch.tensor(2.0, dtype=x.dtype).sqrt(), atol=1e-12, rtol=0)
        state = torch.randn(1, 64, dtype=x.dtype)
        write_rms = (op.delta(p, state, mode).square().sum() / 2).sqrt()
        assert write_rms <= state.norm() + 1e-12


@pytest.mark.parametrize("index", [1, [2, 0], np.array([2, 0]), torch.tensor([2, 0]),
                                    np.array([True, False, True]), torch.tensor([True, False, True]), slice(1, 3), []])
def test_take_selects_endpoint_axis_and_shares_projection(base, index):
    x, available = synthetic(base, 3)
    p = op.encode(base, x, available, op.basis(base))
    selected = p.take(index)
    ix = [index] if isinstance(index, int) else index
    if isinstance(ix, np.ndarray):
        ix = torch.as_tensor(ix)
    if isinstance(ix, list) and not ix:
        ix = torch.empty(0, dtype=torch.long)
    for name in ("tokens", "valid", "features", "support"):
        torch.testing.assert_close(getattr(selected, name), getattr(p, name)[ix])
    assert selected.projection is p.projection


def test_updates_follow_low_rank_formula_and_induced_rms_bound(base):
    x, available = synthetic(base, 3)
    p = op.encode(base, x, available, op.basis(base))
    state = torch.randn(3, 64, dtype=x.dtype)
    for mode in range(2):
        actual = op.delta(p, state, mode)
        expected = (p.features[:, mode] @ state.reshape(3, 8, 8)) @ p.projection.T
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not actual[:, 0].any()
        eligible = p.features[:, mode].norm(dim=-1) > 0
        rms = (actual.square().sum((1, 2)) / eligible.sum(1)).sqrt()
        assert torch.all(rms <= state.norm(dim=1) + 1e-12)


@pytest.mark.parametrize("fastpath", [True, False])
@pytest.mark.parametrize("mode", [0, 1])
def test_zero_state_matches_original_frozen_model_and_optional_forecast(base, fastpath, mode):
    x, available = synthetic(base, 3)
    available[0, :104] = False
    available[1] = False
    x[2, 12, 1] = float("nan")
    p = op.encode(base, x, available, op.basis(base))
    before = {key: value.clone() for key, value in base.state_dict().items()}
    packed_tokens = p.tokens.clone()
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        with torch.no_grad():
            native = base(x, available)
            result = op.forward(base, p, torch.zeros(3, 64, dtype=x.dtype), mode, forecast=True)
            no_forecast = op.forward(base, p, torch.zeros(3, 64, dtype=x.dtype), mode)
        assert set(result) == {"logit", "forecast"} and set(no_forecast) == {"logit"}
        assert result["forecast"].shape == (3, 25, 23)
        torch.testing.assert_close(result["logit"], native["logit"], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(result["forecast"], native["forecast"], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(no_forecast["logit"], native["logit"], atol=2e-12, rtol=2e-12)
        if not fastpath:
            # Native and intervention both use math SDPA, with identical operations.
            assert torch.equal(result["logit"], native["logit"])
            assert torch.equal(result["forecast"], native["forecast"])
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    assert torch.equal(p.tokens, packed_tokens)
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in before.items())


def test_nonzero_forward_matches_direct_native_encoder_with_only_patch_write(base):
    x, available = synthetic(base, 2)
    p = op.encode(base, x, available, op.basis(base))
    state = torch.randn(2, 64, dtype=x.dtype) * 0.1
    for mode in range(2):
        write = op.delta(p, state, mode)
        tokens = torch.cat((p.tokens[:, :16] + write, p.tokens[:, -1:]), 1)
        with sdpa_kernel(SDPBackend.MATH):
            h = base.encoder(tokens, src_key_padding_mask=~p.valid)
        gain, bias = base.fast_origin.chunk(2)
        h = base.norm(h) * (1 + 0.1 * gain.tanh()) + 0.1 * bias.tanh()
        h = h.masked_fill(~p.valid[..., None], 0)
        expected = dict(logit=base.risk_head(h[:, -1]).squeeze(-1),
                        forecast=base.forecast_head(h.flatten(1)).reshape(2, 25, 23))
        actual = op.forward(base, p, state, mode, forecast=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=2e-12, rtol=2e-12)


def test_take_and_per_endpoint_states_are_batch_independent(base):
    x, available = synthetic(base, 3)
    p = op.encode(base, x, available, op.basis(base))
    state = torch.randn(3, 64, dtype=x.dtype) * 0.2
    for mode in range(2):
        whole = op.forward(base, p, state, mode, forecast=True)
        for endpoint in (2, 0, 1):
            single = op.forward(base, p.take(endpoint), state[endpoint:endpoint + 1], mode, forecast=True)
            for key in whole:
                torch.testing.assert_close(single[key][0], whole[key][endpoint], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("mode", [0, 1])
def test_float32_native_parity_and_nonzero_gradient_at_zero(base, mode):
    base.float()
    x, available = synthetic(base, 2)
    available[0, :80] = False
    p = op.encode(base, x, available, op.basis(base))
    state = torch.zeros(2, 64, dtype=x.dtype, requires_grad=True)
    with torch.no_grad():
        native = base(x, available)
    actual = op.forward(base, p, state, mode, forecast=True)
    for key in native:
        torch.testing.assert_close(actual[key], native[key], atol=2e-6, rtol=2e-6)
    gradient, = torch.autograd.grad(actual["logit"].sum() + actual["forecast"].square().mean(), state)
    assert torch.isfinite(gradient).all() and torch.all(gradient.norm(dim=1) > 0)


@pytest.mark.parametrize("mode", [0, 1])
def test_full_forward_gradcheck_and_forecast_gradient(base, mode):
    x, available = synthetic(base, 1)
    p = op.encode(base, x, available, op.basis(base))
    state = (torch.randn(1, 64, dtype=x.dtype) * 0.01).requires_grad_()
    assert torch.autograd.gradcheck(lambda s: op.forward(base, p, s, mode)["logit"],
                                    (state,), eps=1e-6, atol=1e-6, rtol=1e-4)
    output = op.forward(base, p, state, mode, forecast=True)
    grad, = torch.autograd.grad(output["forecast"].square().mean(), state, create_graph=True)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    grad2, = torch.autograd.grad(grad.square().sum(), state)
    assert torch.isfinite(grad2).all()


def test_invalid_contracts_fail_without_mutating_base(base):
    x, available = synthetic(base, 1)
    projection = op.basis(base)
    with pytest.raises((ValueError, TypeError)):
        op.encode(base, x[:, :-1], available[:, :-1], projection)
    with pytest.raises((ValueError, TypeError)):
        op.encode(base, x, available.float(), projection)
    with pytest.raises((ValueError, TypeError)):
        op.encode(base, x, available, projection[:, :-1])
    with pytest.raises((ValueError, TypeError)):
        op.encode(base, x, available, torch.full_like(projection, float("nan")))
    with pytest.raises((ValueError, TypeError)):
        op.encode(base, x, available, projection * 2)
    p = op.encode(base, x, available, projection)
    with pytest.raises((ValueError, TypeError)):
        op.delta(p, torch.zeros(1, 63, dtype=x.dtype), 0)
    with pytest.raises((ValueError, TypeError)):
        op.delta(p, torch.zeros(1, 64, dtype=x.dtype), 2)
    with pytest.raises((ValueError, RuntimeError)):
        op.delta(p, torch.full((1, 64), float("nan"), dtype=x.dtype), 0)
    base.train()
    with pytest.raises((ValueError, TypeError)):
        op.encode(base, x, available, projection)
    base.eval()
    base.embedding.weight.requires_grad_(True)
    with pytest.raises((ValueError, TypeError)):
        op.encode(base, x, available, projection)
