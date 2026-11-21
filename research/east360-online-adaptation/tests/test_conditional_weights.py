import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
from experiments.risk_aligned_online.conditional_weights_20261030.learning import (
    ConditionalWeights,
    controlled_contexts,
)
from experiments.risk_aligned_online.context_vector_20261030.teaching import context_controls
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture(scope='module', autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def deterministic_torch():
    with torch.random.fork_rng():
        torch.manual_seed(20261030)
        yield


def nonzero_generator(radius=0.1):
    generator = ConditionalWeights(radius=radius)
    with torch.no_grad():
        generator.b_head.weight.normal_(0, 0.1)
        generator.b_head.bias.normal_(0, 0.1)
    return generator


def test_initialization_and_matrix_attention_layout():
    generator = ConditionalWeights()
    assert isinstance(generator.encoder[0], nn.LayerNorm)
    assert generator.encoder[0].normalized_shape == (358,)
    assert isinstance(generator.encoder[1], nn.Linear)
    assert (generator.encoder[1].in_features, generator.encoder[1].out_features) == (358, 32)
    assert isinstance(generator.encoder[2], nn.SiLU)
    assert (generator.a_head.in_features, generator.a_head.out_features) == (32, 256)
    assert (generator.b_head.in_features, generator.b_head.out_features) == (32, 256)
    assert torch.count_nonzero(generator.b_head.weight) == 0
    assert torch.count_nonzero(generator.b_head.bias) == 0
    assert torch.count_nonzero(generator.a_head.weight) > 0

    context = torch.randn(3, 358)
    state = generator(context, torch.tensor([True, False, True]))
    assert state.shape == (3, 8192)
    assert torch.count_nonzero(state) == 0
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23)))
    parts = net.deltas(state, 'qk')
    assert torch.equal(parts['q'], state.reshape(3, 2, 64, 64)[:, 0])
    assert torch.equal(parts['k'], state.reshape(3, 2, 64, 64)[:, 1])


def test_generator_learns_through_real_frozen_attention_and_detaches_context():
    generator = ConditionalWeights()
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23))).eval()
    frozen_before = {key: value.clone() for key, value in net.state_dict().items()}
    initial_parameters = {key: value.detach().clone() for key, value in generator.named_parameters()}
    context = torch.randn(3, 358, requires_grad=True)
    support = torch.tensor([True, True, False])
    values = torch.randn(3, 128, 23)
    available = torch.ones_like(values, dtype=torch.bool)
    targets = torch.tensor([0., 1., 0.])
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        state = generator(context, support)
        logits = net(values, available, state, None, 'qk', forecast=False)['logit']
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        assert context.grad is None
        assert all(parameter.grad is None for parameter in net.parameters())
        assert generator.b_head.weight.grad.abs().sum() > 0
        if step > 0:
            assert generator.a_head.weight.grad.abs().sum() > 0
            assert generator.encoder[1].weight.grad.abs().sum() > 0
        optimizer.step()

    with torch.no_grad():
        state = generator(context, support)
        hidden = generator.encoder(context.detach())
        a = generator.a_head(hidden)
        b = generator.b_head(hidden)
    assert torch.count_nonzero(state[:2]) > 0
    assert not torch.allclose(state[0], state[1], rtol=1e-5, atol=1e-9)
    assert not torch.allclose(a[0], a[1])
    assert not torch.allclose(b[0], b[1])
    assert torch.count_nonzero(state[2]) == 0
    assert any(not torch.equal(value, initial_parameters[key])
               for key, value in generator.named_parameters())
    assert all(torch.equal(value, frozen_before[key]) for key, value in net.state_dict().items())


def test_nonzero_factors_generate_rank_two_projected_qk_matrices():
    generator = nonzero_generator()
    context = torch.randn(4, 358)
    state = generator(context, torch.ones(4, dtype=torch.bool))
    hidden = generator.encoder(context)
    a = generator.a_head(hidden).reshape(4, 2, 2, 64)
    b = generator.b_head(hidden).reshape(4, 2, 2, 64)
    raw = (b.transpose(-1, -2) @ a).flatten(1)
    norms = raw.norm(dim=1, keepdim=True)
    expected = raw * (0.1 / norms.clamp_min(1e-12)).clamp(max=1)
    torch.testing.assert_close(state, expected, atol=1e-7, rtol=1e-5)
    assert torch.isfinite(state).all()
    assert (state.double().norm(dim=1) <= 0.1000001).all()
    singular_values = torch.linalg.svdvals(state.reshape(4, 2, 64, 64))
    assert (singular_values[:, :, :2] > 1e-7).all()
    assert (singular_values[:, :, 2:] < 1e-6).all()


def test_ball_projection_keeps_small_updates_and_clips_large_ones():
    generator = ConditionalWeights()
    context = torch.randn(2, 358)
    with torch.no_grad():
        generator.a_head.weight.zero_()
        generator.a_head.bias.fill_(0.01)
        generator.b_head.weight.zero_()
        generator.b_head.bias.fill_(0.01)
    support = torch.ones(2, dtype=torch.bool)
    small = generator(context, support)
    expected_small = torch.full_like(small, 2 * 0.01 * 0.01)
    torch.testing.assert_close(small, expected_small)
    assert (small.double().norm(dim=1) < 0.1).all()
    with torch.no_grad():
        generator.b_head.bias.fill_(1.)
    large = generator(context, support)
    torch.testing.assert_close(large.double().norm(dim=1), torch.full((2,), 0.1, dtype=torch.float64),
                               atol=1e-8, rtol=0)
    assert torch.isfinite(large).all()
    with torch.no_grad():
        generator.b_head.bias.fill_(1e30)
    huge = generator(context, support)
    assert torch.isfinite(huge).all()
    torch.testing.assert_close(huge.double().norm(dim=1), torch.full((2,), 0.1, dtype=torch.float64),
                               atol=1e-8, rtol=0)


def test_projection_is_per_sample_and_honors_smaller_radius():
    generator = ConditionalWeights(radius=0.025)
    with torch.no_grad():
        generator.encoder[1].weight.zero_()
        generator.encoder[1].bias.zero_()
        generator.encoder[1].weight[0, 0] = 1.
        generator.a_head.weight.zero_()
        generator.a_head.bias.fill_(0.01)
        generator.b_head.weight.zero_()
        generator.b_head.bias.zero_()
        generator.b_head.weight[:, 0] = 1.
    context = torch.zeros(2, 358)
    context[:, 0] = torch.tensor([1e-5, 0.1])
    support = torch.ones(2, dtype=torch.bool)
    state = generator(context, support)
    raw = (generator.b_head(generator.encoder(context)).reshape(2, 2, 2, 64).transpose(-1, -2)
           @ generator.a_head(generator.encoder(context)).reshape(2, 2, 2, 64)).flatten(1)
    assert 0 < state[0].double().norm() < 0.025
    torch.testing.assert_close(state[0], raw[0])
    assert state[1].double().norm().item() == pytest.approx(0.025, abs=1e-8)
    torch.testing.assert_close(state, torch.cat([generator(context[i:i + 1], support[i:i + 1])
                                               for i in range(2)]), atol=1e-8, rtol=1e-5)


def test_batch_independence_no_mutation_and_unsupported_zero_gradients():
    generator = nonzero_generator()
    context = torch.randn(358, 4).T
    support = torch.tensor([True, False, True, False])
    context_before, support_before = context.clone(), support.clone()
    parameters_before = {key: value.clone() for key, value in generator.state_dict().items()}
    state = generator(context, support)
    individual = torch.cat([generator(context[i:i + 1], support[i:i + 1]) for i in range(4)])
    torch.testing.assert_close(state, individual, atol=1e-8, rtol=1e-5)
    assert torch.count_nonzero(state[~support]) == 0
    assert torch.equal(context, context_before)
    assert torch.equal(support, support_before)
    assert all(torch.equal(value, parameters_before[key])
               for key, value in generator.state_dict().items())
    generator.zero_grad(set_to_none=True)
    state[~support].sum().backward()
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
               for parameter in generator.parameters())


def test_zero_radius_and_empty_batch():
    generator = nonzero_generator(radius=0.)
    context = torch.randn(2, 358)
    state = generator(context, torch.ones(2, dtype=torch.bool))
    assert torch.count_nonzero(state) == 0
    state.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in generator.parameters())
    assert generator(torch.empty(0, 358), torch.empty(0, dtype=torch.bool)).shape == (0, 8192)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_floating_context_dtypes_are_supported(dtype):
    generator = nonzero_generator()
    state = generator(torch.randn(2, 358).to(dtype), torch.ones(2, dtype=torch.bool))
    assert state.dtype == next(generator.parameters()).dtype
    assert torch.isfinite(state).all()
    assert (state.double().norm(dim=1) <= 0.1000001).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_extreme_but_finite_context_succeeds_or_fails_explicitly(dtype):
    generator = nonzero_generator()
    context = torch.full((2, 358), torch.finfo(dtype).max, dtype=dtype)
    context[0, ::2] *= -1
    converted = context.to(generator.encoder[0].weight.dtype)
    if torch.isfinite(converted).all() and torch.isfinite(generator.encoder[0](converted)).all():
        state = generator(context, torch.ones(2, dtype=torch.bool))
        assert torch.isfinite(state).all()
        assert (state.double().norm(dim=1) <= 0.1000001).all()
    else:
        with pytest.raises((ValueError, FloatingPointError), match='finite'):
            generator(context, torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize('site', ['encoder', 'head'])
def test_nonfinite_internal_results_raise_explicit_error(site):
    generator = nonzero_generator()
    with torch.no_grad():
        if site == 'encoder':
            generator.encoder[1].weight[0, 0] = math.nan
        else:
            generator.b_head.bias[0] = math.inf
    with pytest.raises(FloatingPointError, match='nonfinite'):
        generator(torch.randn(2, 358), torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize('supported', [True, False])
def test_rejects_nonfinite_context_even_on_unsupported_rows(value, supported):
    context = torch.randn(2, 358)
    context[1, 10] = value
    with pytest.raises(ValueError):
        ConditionalWeights()(context, torch.tensor([True, supported]))


@pytest.mark.parametrize('context', [
    None,
    [[0.] * 358],
    torch.zeros(358),
    torch.zeros(1, 357),
    torch.zeros(1, 358, 1),
    torch.zeros(1, 358, dtype=torch.int64),
    torch.zeros(1, 358, dtype=torch.bool),
    torch.zeros(1, 358, dtype=torch.complex64),
])
def test_rejects_invalid_context(context):
    with pytest.raises((TypeError, ValueError)):
        ConditionalWeights()(context, torch.tensor([True]))


@pytest.mark.parametrize('support', [
    None,
    [True, False],
    torch.tensor([1, 0]),
    torch.tensor([1., 0.]),
    torch.tensor(True),
    torch.ones(2, 1, dtype=torch.bool),
    torch.ones(3, dtype=torch.bool),
    torch.empty(2, dtype=torch.bool, device='meta'),
])
def test_rejects_invalid_or_wrong_device_support(support):
    with pytest.raises((TypeError, ValueError)):
        ConditionalWeights()(torch.randn(2, 358), support)


@pytest.mark.parametrize('radius', [
    -0.001, 0.100001, math.nan, math.inf, -math.inf, True, None, '0.1', torch.tensor(0.1),
])
def test_rejects_invalid_radius(radius):
    with pytest.raises((TypeError, ValueError)):
        ConditionalWeights(radius=radius)


def test_controlled_contexts_reuses_existing_control_definitions():
    context = torch.arange(3 * 358, dtype=torch.float32).reshape(3, 358)
    train_mean = torch.arange(358, dtype=torch.float32) / 10
    context_before, mean_before = context.clone(), train_mean.clone()
    actual = controlled_contexts(context, train_mean=train_mean)
    expected = context_controls(context, train_mean)
    assert set(actual) == {'real', 'reverse', 'mean_feedback', 'constant'}
    for key in expected:
        assert torch.equal(actual[key], expected[key])
    assert torch.equal(context, context_before)
    assert torch.equal(train_mean, mean_before)
    with pytest.raises(ValueError):
        controlled_contexts(context[:, :-1], train_mean)
