import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
import layer0_local_memory_20261105 as memory
import trajectory_policy_20261105 as policy


def key_rows(*values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype)[:, None].expand(-1, 192).clone()


def vector(value, *, dtype=torch.float64):
    v = torch.zeros(128, dtype=dtype)
    v[0] = value
    return v


def test_keys_use_only_standardized_native_state_and_causal_history():
    mean = torch.linspace(-1, 1, 194)
    scale = torch.linspace(0.5, 2, 194)
    state = torch.arange(3 * 66, dtype=torch.float32).reshape(3, 66)
    history = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128) / 100
    ctx = (None, None, None, None, state, history, None)
    rule = SimpleNamespace(mean=mean, scale=scale)
    expected = torch.asinh((torch.cat((state[:, 2:], history), 1) - mean[2:]) / scale[2:])
    torch.testing.assert_close(memory.keys_for(rule, ctx), expected)
    state[:, :2] = float("nan")
    rule.mean[:2] = float("nan")
    rule.scale[:2] = 0
    torch.testing.assert_close(memory.keys_for(rule, ctx), expected)
    rule.scale[8] = 0
    with pytest.raises(ValueError):
        memory.keys_for(rule, ctx)


def test_bandwidth_is_pooled_75th_percentile_of_nearest_other_shot():
    groups = [key_rows(0, 1), key_rows(3), key_rows(10)]
    # Every row: nearest *other* shot distances are 3, 2, 2, 7.
    assert memory.fit_bandwidth(groups) == pytest.approx(4.0)
    assert memory.fit_bandwidth([key_rows(0, 0.01), key_rows(4, 4.01)]) == pytest.approx(4.0)
    assert memory.fit_bandwidth([key_rows(i) for i in range(20)]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "groups",
    [
        [key_rows(0)],
        [key_rows(0), key_rows(0)],
        [key_rows(0), torch.empty(0, 192, dtype=torch.float64)],
        [key_rows(0), key_rows(float("nan"))],
        [key_rows(0), torch.zeros(1, 191, dtype=torch.float64)],
    ],
)
def test_bandwidth_rejects_missing_degenerate_or_nonfinite_input(groups):
    with pytest.raises((ValueError, FloatingPointError)):
        memory.fit_bandwidth(groups)


def test_episode_kernel_max_and_shared_denominator_with_compact_support():
    first = memory.Episode(key_rows(0, 1, 1), vector(0.6))
    second = memory.Episode(key_rows(2), vector(-0.4))
    result = memory.memory_actions(key_rows(0, 1, 3, 4), [first, second], 1.0)
    expected = torch.stack((vector(0.6), vector(0.24), vector(-0.225), vector(0)))
    torch.testing.assert_close(result, expected)
    assert torch.equal(memory.memory_actions(key_rows(2), [], 1.0), torch.zeros(1, 128, dtype=torch.float64))
    assert memory.memory_actions(key_rows(), [first], 1.0).shape == (0, 128)


def test_episode_kernel_is_differentiable_in_episode_delta():
    delta = vector(0.4).requires_grad_()
    out = memory.memory_actions(key_rows(1), [memory.Episode(key_rows(0), delta)], 1.0)
    out.sum().backward()
    torch.testing.assert_close(delta.grad, torch.full_like(delta, 0.5625))


def test_memory_handles_positive_bandwidth_below_float32_squared_range():
    keys = key_rows(0, 1e-30, dtype=torch.float32)
    episode = memory.Episode(keys[:1], vector(0.4, dtype=torch.float32))
    actual = memory.memory_actions(keys, [episode], 1e-30)
    torch.testing.assert_close(actual[:, 0], torch.tensor([0.4, 0.225]))


@pytest.mark.parametrize("bandwidth", [0, -1, float("nan"), float("inf")])
def test_memory_rejects_invalid_bandwidth_even_when_empty(bandwidth):
    with pytest.raises(ValueError):
        memory.memory_actions(key_rows(0), [], bandwidth)


def test_memory_rejects_nonfinite_or_oversized_episodes():
    with pytest.raises((ValueError, FloatingPointError)):
        memory.memory_actions(key_rows(float("inf")), [], 1.0)
    for episode in (
        lambda: memory.Episode(key_rows(0), vector(0.61)),
        lambda: memory.Episode(key_rows(0), vector(float("nan"))),
        lambda: memory.Episode(key_rows(), vector(0.1)),
    ):
        with pytest.raises((ValueError, FloatingPointError)):
            memory.memory_actions(key_rows(0), [episode()], 1.0)
    mutated = memory.Episode(key_rows(0), vector(0.1))
    mutated.delta[0] = 0.7
    with pytest.raises(ValueError):
        memory.memory_actions(key_rows(0), [mutated], 1.0)


def test_episode_kernel_chunking_and_max_match_a_direct_reference():
    generator = torch.Generator().manual_seed(91)
    keys = torch.randn(65, 192, generator=generator, dtype=torch.float64)
    episodes = []
    for i in range(17):
        prototypes = torch.randn(10, 192, generator=generator, dtype=torch.float64)
        episodes.append(memory.Episode(prototypes, vector(0.01 * (i - 8))))
    actual = memory.memory_actions(keys, episodes, 1.0)
    weight = torch.stack([
        (1 - (keys[:, None, :] - episode.keys).square().mean(2) / 4).clamp_min(0).square().max(1).values
        for episode in episodes
    ], dim=1)
    expected = weight @ torch.stack([episode.delta for episode in episodes])
    expected /= weight.sum(1, keepdim=True).clamp_min(1)
    torch.testing.assert_close(actual, expected)


def test_total_action_uses_largest_feasible_lambda_without_rescaling_fast():
    fast = torch.stack((vector(1.5), vector(1.5), vector(2.0), vector(0.3)))
    slow = torch.stack((vector(1), vector(-4), vector(1), vector(0)))
    actual = memory.total_action(fast, slow)
    expected = torch.stack((vector(2), vector(-2), vector(2), vector(0.3)))
    torch.testing.assert_close(actual[:, :128], expected)
    assert torch.equal(actual[3, :128], fast[3])
    assert torch.count_nonzero(actual[:, 128:]) == 0
    assert (actual[:, :128].norm(dim=1) <= 2).all()
    with pytest.raises(ValueError):
        memory.total_action(torch.stack([vector(2.01)]), torch.stack([vector(0)]))


def test_total_action_has_finite_gradients_at_zero_and_when_clipped():
    fast = torch.stack((vector(0.5), vector(1.9), vector(2))).requires_grad_()
    slow = torch.stack((vector(0), vector(0.5), vector(0)))
    slow[2, 1] = 0.1  # tangent at the boundary: only lambda=0 is feasible
    slow.requires_grad_()
    result = memory.total_action(fast, slow)
    result.square().sum().backward()
    assert torch.isfinite(result).all()
    assert torch.isfinite(fast.grad).all()
    assert torch.isfinite(slow.grad).all()
    assert torch.equal(result[0, :128], fast.detach()[0])
    torch.testing.assert_close(result[2, :128], fast.detach()[2])


def test_total_action_random_clipped_rows_use_the_maximal_feasible_amount():
    generator = torch.Generator().manual_seed(912)
    fast = torch.randn(30, 128, generator=generator, dtype=torch.float64)
    fast = 1.99 * fast / fast.norm(dim=1, keepdim=True)
    slow = torch.randn(30, 128, generator=generator, dtype=torch.float64) * 0.1
    combined = memory.total_action(fast, slow)[:, :128]
    assert (combined.norm(dim=1) <= 2 + 1e-12).all()
    a = slow.square().sum(dim=1)
    b = (fast * slow).sum(dim=1)
    c = 4 - fast.square().sum(dim=1)
    expected = ((-b + torch.sqrt(b.square() + a * c)) / a).clamp(max=1)
    torch.testing.assert_close(combined, fast + expected[:, None] * slow, rtol=1e-12, atol=1e-12)


class TinyBase(nn.Module):
    """A small differentiable network exercising the actual norm1 hook path."""

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(dim=64, layers=2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for block in self.encoder.layers:
            block.norm1 = nn.LayerNorm(64)
        self.readout = nn.Linear(64, 1, bias=False)
        with torch.no_grad():
            self.readout.weight.zero_()
            self.readout.weight[0, 0] = 3.0
            self.readout.weight[0, 1] = -2.0
        self.eval().requires_grad_(False)

    def forward(self, x, a):
        tokens = torch.linspace(-1, 1, 64, device=x.device, dtype=x.dtype)
        tokens = tokens.view(1, 1, 64).expand(len(x), 17, 64)
        tokens = tokens + x[:, :1, :1] * torch.linspace(0, 0.1, 64, device=x.device).view(1, 1, 64)
        for block in self.encoder.layers:
            tokens = tokens + 0.1 * block.norm1(tokens)
        return {"logit": self.readout(tokens[:, -1]).squeeze(1)}


@pytest.fixture
def fit_case():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = TinyBase()
        x = torch.linspace(-1, 1, 201).reshape(201, 1, 1).expand(-1, 128, 23)
        a = torch.ones(1, 128, 23, dtype=torch.bool).expand(201, -1, -1)
        state, history = torch.zeros(3, 66), torch.zeros(3, 128)
        ctx = (x, a, None, None, state, history, history)
        keys = key_rows(0, 0.4, 0.8, dtype=torch.float32)
        fast = torch.stack([vector(v, dtype=torch.float32) for v in (0.1, -0.2, 0.3)])
        old = [memory.Episode(key_rows(0, dtype=torch.float32), vector(0.4, dtype=torch.float32))]
        indices = np.array([135, 136, 167, 168, 200], dtype=np.int64)
        yield base, ctx, fast, keys, old, indices
    finally:
        torch.set_num_threads(old_threads)


def exact_objective(base, ctx, fast, keys, episodes, indices, disruptive, delta):
    q = torch.from_numpy(indices)
    k = (q - 135) // 32
    selected = keys[k]
    slow = memory.memory_actions(selected, episodes, 1.0)
    xx, aa = ctx[0][q], ctx[1][q]
    route = latent.routes_all(aa)
    zero = torch.zeros_like(fast[k])
    logits = torch.stack([
        policy.routed_pair(base, xx, aa, route, memory.total_action(zero, slow)),
        policy.routed_pair(base, xx, aa, route, memory.total_action(fast[k], slow)),
    ])
    threshold_logit = math.log(0.7258946895599365 / (1 - 0.7258946895599365))
    sign = -1 if disruptive else 1
    return F.softplus(sign * (logits - threshold_logit)).mean() + 0.10 * delta.square().sum()


@pytest.mark.parametrize("disruptive", [False, True])
def test_fit_episode_exact_two_branch_hook_loss_and_frozen_source(fit_case, disruptive, monkeypatch):
    base, ctx, fast, keys, old, indices = fit_case
    state_before = {k: v.clone() for k, v in base.state_dict().items()}
    old_before = old[0].delta.clone()
    prototype = keys[torch.tensor([0, 1, 2])]
    initial = memory.Episode(prototype, torch.zeros(128))
    expected_initial = exact_objective(base, ctx, fast, keys, old + [initial], indices, disruptive, initial.delta)
    routed = policy.routed_pair
    writes = []

    def capture(base, x, a, routes, delta):
        writes.append(delta.detach().clone())
        return routed(base, x, a, routes, delta)

    monkeypatch.setattr(memory.policy, "routed_pair", capture)
    episode, receipt = memory.fit_episode(base, ctx, fast, keys, old, 1.0, indices, disruptive)
    assert len(writes) == 44  # initial + twenty steps + final, both branches each
    q = torch.from_numpy(indices)
    k = (q - 135) // 32
    slow = memory.memory_actions(keys[k], old + [initial], 1.0)
    torch.testing.assert_close(writes[0], memory.total_action(torch.zeros_like(fast[k]), slow))
    torch.testing.assert_close(writes[1], memory.total_action(fast[k], slow))
    assert all(torch.count_nonzero(w[:, 128:]) == 0 for w in writes)
    assert set(receipt) == {"steps", "initial_loss", "final_loss", "norm", "n_keys"}
    assert receipt["steps"] == 20 and receipt["n_keys"] == 3
    torch.testing.assert_close(episode.keys, prototype)
    assert 0 < episode.delta.norm() <= 0.600001
    assert receipt["norm"] == pytest.approx(float(episode.delta.norm()))
    assert receipt["initial_loss"] == pytest.approx(float(expected_initial))
    final = exact_objective(base, ctx, fast, keys, old + [episode], indices, disruptive, episode.delta)
    assert receipt["final_loss"] == pytest.approx(float(final))
    assert receipt["final_loss"] < receipt["initial_loss"]
    assert torch.equal(old[0].delta, old_before)
    assert not base.training and all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in state_before.items())


@pytest.mark.parametrize(
    "indices",
    [np.array([], dtype=np.int64), np.array([135, 135], dtype=np.int64),
     np.array([136, 135], dtype=np.int64), np.array([134], dtype=np.int64),
     np.array([201], dtype=np.int64), np.array([135], dtype=np.int32)],
)
def test_fit_episode_rejects_invalid_indices(fit_case, indices):
    base, ctx, fast, keys, old, _ = fit_case
    with pytest.raises(ValueError):
        memory.fit_episode(base, ctx, fast, keys, old, 1.0, indices, True)


def test_fit_episode_requires_an_untouched_frozen_eval_source(fit_case):
    base, ctx, fast, keys, old, indices = fit_case
    base.train()
    with pytest.raises(ValueError):
        memory.fit_episode(base, ctx, fast, keys, old, 1.0, indices, False)
    assert base.training
    base.eval()
    next(base.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):
        memory.fit_episode(base, ctx, fast, keys, old, 1.0, indices, False)
    assert next(base.parameters()).requires_grad


def test_fit_episode_rejects_a_training_child_without_changing_it(fit_case):
    base, ctx, fast, keys, old, indices = fit_case
    base.encoder.layers[0].train()
    with pytest.raises(ValueError):
        memory.fit_episode(base, ctx, fast, keys, old, 1.0, indices, False)
    assert base.encoder.layers[0].training


def test_fit_optimizes_only_new_delta_even_inside_no_grad(fit_case):
    base, ctx, fast, keys, old, indices = fit_case
    fast.requires_grad_()
    keys.requires_grad_()
    old[0].keys.requires_grad_()
    old[0].delta.requires_grad_()
    x = ctx[0].detach().requires_grad_()
    ctx = (x, *ctx[1:])
    with torch.no_grad():
        episode, receipt = memory.fit_episode(base, ctx, fast, keys, old, 1.0, indices, True)
    assert receipt["steps"] == 20 and not episode.delta.requires_grad
    assert all(v.grad is None for v in (fast, keys, old[0].keys, old[0].delta, x))


def test_synthetic_simple23_exact_hook_and_full_episode_fit(fit_case):
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        base = Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23), torch.ones(23),
        ).eval().requires_grad_(False)
        x = torch.randn(168, 128, 23)
        available = torch.ones_like(x, dtype=torch.bool)
        state, history = torch.zeros(2, 66), torch.zeros(2, 128)
        ctx = (x, available, None, None, state, history, history)
        keys = key_rows(0, 0.4, dtype=torch.float32)
        fast = torch.zeros(2, 128)
        q = np.array([135, 167], dtype=np.int64)
        before = {k: v.clone() for k, v in base.state_dict().items()}
        delta = torch.zeros(2, 128, requires_grad=True)
        xx, aa = x[q], available[q]
        logits = policy.routed_pair(base, xx, aa, latent.routes_all(aa), memory.total_action(fast, delta))
        (grad,) = torch.autograd.grad(logits.sum(), delta)
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
        episode, receipt = memory.fit_episode(base, ctx, fast, keys, [], 1.0, q, True)
        assert receipt["steps"] == 20 and receipt["n_keys"] == 2
        assert torch.isfinite(episode.delta).all() and 0 < episode.delta.norm() <= 0.6
        assert all(torch.equal(v, base.state_dict()[k]) for k, v in before.items())
        assert all(p.grad is None for p in base.parameters())
        assert all(not block.norm1._forward_hooks for block in base.encoder.layers)
