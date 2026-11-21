import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
import layer0_local_memory_20261105 as memory
import layer0_phase_kernel_20261105 as phase
import trajectory_policy_20261105 as policy


def key_rows(*values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype).reshape(-1, 1).expand(-1, 192).clone()


def vector(value, *, dtype=torch.float64):
    result = torch.zeros(128, dtype=dtype)
    result[0] = value
    return result


def episode(pos, early, delta, *, dtype=torch.float64):
    return phase.PhaseEpisode(key_rows(*pos, dtype=dtype), vector(delta, dtype=dtype),
                              key_rows(*early, dtype=dtype))


def reference_actions(keys, episodes, h):
    """Direct, independent expression of the phase rule for small fixtures."""
    if not episodes:
        return keys.new_zeros((len(keys), 128))
    weights = []
    for ep in episodes:
        positive = ((keys[:, None] - ep.keys[None]).square().mean(2)).min(1).values
        early = ((keys[:, None] - ep.early_keys[None]).square().mean(2)).min(1).values
        weights.append((1 - positive / (4 * h * h)).clamp_min(0).square()
                       * (positive < early))
    weights = torch.stack(weights, dim=1)
    return weights @ torch.stack([ep.delta for ep in episodes]) / weights.sum(
        1, keepdim=True
    ).clamp_min(1)


def test_early_anchors_are_actual_causal_samples_and_matched_are_uniform():
    times = np.arange(950, dtype=np.float64) / 1000
    all_indices = phase.early_anchor_indices(times, 0.8, 3, "all")
    expected = np.array([135, 167, 199, 231, 263, 295], dtype=np.int64)
    np.testing.assert_array_equal(all_indices, expected)
    np.testing.assert_array_equal(
        phase.early_anchor_indices(times, 0.8, 3, "matched"), expected[[0, 2, 5]]
    )
    np.testing.assert_array_equal(
        phase.early_anchor_indices(times, 0.8, 6, "matched"), expected
    )
    assert all_indices.dtype == np.int64
    assert np.all(np.round((0.8 - times[all_indices]) * 1000, 9) >= 500)
    assert phase.early_anchor_indices(times, 0.8, 7, "matched").size == 0
    assert phase.early_anchor_indices(times[:135], 0.8, 1, "all").size == 0
    np.testing.assert_array_equal(
        phase.early_anchor_indices(times[:136], 0.8, 1, "all"), expected[:1]
    )


def test_early_anchor_cutoff_uses_the_specified_nine_decimal_rounding():
    times = np.arange(300, dtype=np.float64) / 1000
    np.testing.assert_array_equal(
        phase.early_anchor_indices(times, 0.635 - 1e-13, 1, "matched"), [135]
    )
    assert phase.early_anchor_indices(times, 0.635 - 1e-10, 1, "all").size == 0


@pytest.mark.parametrize(
    "times,down,n_pos,mode",
    [([0.0], 1.0, -1, "all"), ([0.0], 1.0, 1.2, "matched"),
     ([0.0], 1.0, 1, "other"), ([[0.0]], 1.0, 1, "all"),
     ([float("nan")], 1.0, 1, "all"), ([0.0], float("inf"), 1, "all")],
)
def test_early_anchors_reject_invalid_inputs(times, down, n_pos, mode):
    with pytest.raises(ValueError):
        phase.early_anchor_indices(times, down, n_pos, mode)


def test_phase_episode_extends_memory_and_detaches_early_keys():
    early = key_rows(2).requires_grad_()
    ep = phase.PhaseEpisode(key_rows(0), vector(0.3), early)
    assert isinstance(ep, memory.Episode)
    assert ep.early_keys.shape == (1, 192)
    assert not ep.early_keys.requires_grad
    assert ep.early_keys.dtype == ep.keys.dtype and ep.early_keys.device == ep.keys.device
    torch.testing.assert_close(ep.early_keys, early)


@pytest.mark.parametrize("early", [key_rows(), torch.zeros(1, 191), key_rows(float("nan")),
                                   key_rows(1, dtype=torch.float32)])
def test_phase_episode_rejects_missing_invalid_or_mismatched_early_keys(early):
    with pytest.raises(ValueError):
        phase.PhaseEpisode(key_rows(0), vector(0.2), early)


def test_phase_gate_vetoes_ties_and_keeps_original_compact_kernel():
    eps = [episode([0], [2], 0.6), episode([2], [0], -0.4)]
    keys = key_rows(0, 0.5, 1, 1.5, 2, 4, -2)
    weight = (1 - 0.5**2 / 4) ** 2
    expected = torch.stack([vector(x) for x in
                            (0.6, 0.6 * weight, 0, -0.4 * weight, -0.4, 0, 0)])
    torch.testing.assert_close(phase.phase_actions(keys, eps, 1.0), expected)
    assert list(inspect.signature(phase.phase_actions).parameters) == [
        "keys", "episodes", "bandwidth", "fixed_prefix"
    ]


def test_denominator_counts_only_accepted_weights_and_never_amplifies():
    first = episode([0], [2], 0.6)
    second = episode([0], [3], -0.2)
    vetoed = episode([0], [0], 0.5)
    actual = phase.phase_actions(key_rows(0, 0.5, -1.5), [first, second, vetoed], 1.0)
    small_weight = (1 - 1.5**2 / 4) ** 2
    expected = torch.stack([vector(0.2), vector(0.2), vector(0.4 * small_weight)])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_phase_actions_matches_direct_nearest_prototype_reference(dtype):
    generator = torch.Generator().manual_seed(91)
    keys = torch.randn(65, 192, generator=generator, dtype=dtype)
    eps = []
    for i in range(17):
        pos = torch.randn(10, 192, generator=generator, dtype=dtype)
        early = torch.randn(10, 192, generator=generator, dtype=dtype)
        eps.append(phase.PhaseEpisode(pos, vector((i - 8) / 20, dtype=dtype), early))
    actual = phase.phase_actions(keys, eps, 1.0)
    torch.testing.assert_close(actual, reference_actions(keys, eps, 1.0), rtol=2e-6, atol=1e-7)
    assert actual.dtype == dtype and actual.device == keys.device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_open_gate_is_bit_exact_with_native_kernel_and_delta_gradient(dtype):
    generator = torch.Generator().manual_seed(12)
    keys = torch.randn(9, 192, generator=generator, dtype=dtype)
    eps = []
    for i in range(4):
        pos = torch.randn(3, 192, generator=generator, dtype=dtype)
        delta = vector((i + 1) / 20, dtype=dtype).requires_grad_()
        eps.append(phase.PhaseEpisode(pos, delta, pos + 100))
    actual = phase.phase_actions(keys, eps, 1.0, fixed_prefix=1)
    native = memory._memory_actions(keys, eps, 1.0, fixed_prefix=1)
    assert torch.equal(actual, native)
    actual_grad = torch.autograd.grad(actual.sum(), eps[-1].delta, retain_graph=True)[0]
    native_grad = torch.autograd.grad(native.sum(), eps[-1].delta)[0]
    assert torch.equal(actual_grad, native_grad)


def test_gate_distance_is_stable_at_extreme_finite_scales():
    tiny = episode([1e-300], [2e-300], 0.4)
    torch.testing.assert_close(phase.phase_actions(key_rows(0), [tiny], 1e-300),
                               vector(0.225)[None])
    # The early separation exceeds float64's range, but its half-RMS is finite.
    huge = episode([4e307], [9e307], 0.4)
    torch.testing.assert_close(phase.phase_actions(key_rows(-1e308), [huge], 1e308),
                               vector(0.4 * (1 - 1.4**2 / 4)**2)[None])


def test_empty_cases_and_tiny_bandwidth_are_supported():
    ep = episode([0], [2e-30], 0.4, dtype=torch.float32)
    torch.testing.assert_close(
        phase.phase_actions(key_rows(0, 1e-30, dtype=torch.float32), [ep], 1e-30)[:, 0],
        torch.tensor([0.4, 0.0]),
    )
    keys = key_rows(1)
    assert torch.equal(phase.phase_actions(keys, [], 1), torch.zeros(1, 128, dtype=keys.dtype))
    assert phase.phase_actions(key_rows(), [episode([0], [2], 0.1)], 1).shape == (0, 128)
    assert phase.phase_actions(key_rows(), [], 1).shape == (0, 128)


@pytest.mark.parametrize("h", [0, -1, float("nan"), float("inf")])
def test_invalid_bandwidth_rejected_even_with_empty_memory(h):
    with pytest.raises(ValueError):
        phase.phase_actions(key_rows(0), [], h)


def test_invalid_or_mutated_episode_and_prefix_are_rejected():
    ep = episode([0], [2], 0.2)
    for prefix in (-1, 2, 0.5):
        with pytest.raises(ValueError):
            phase.phase_actions(key_rows(0), [ep], 1.0, fixed_prefix=prefix)
    with pytest.raises(ValueError):
        phase.phase_actions(key_rows(0), [memory.Episode(key_rows(0), vector(0.2))], 1.0)
    with pytest.raises(ValueError):
        phase.phase_actions(key_rows(0, dtype=torch.float32), [ep], 1.0)
    ep.early_keys[0, 0] = float("nan")
    with pytest.raises(ValueError):
        phase.phase_actions(key_rows(0), [ep], 1.0)


def test_fixed_prefix_freezes_old_keys_and_deltas_but_new_delta_has_gradient():
    old_keys = key_rows(0).requires_grad_()
    old_delta = vector(0.2).requires_grad_()
    new_delta = vector(0.1).requires_grad_()
    early = key_rows(2).requires_grad_()
    old = phase.PhaseEpisode(old_keys, old_delta, early)
    new = phase.PhaseEpisode(key_rows(0.5), new_delta, key_rows(3))
    phase.phase_actions(key_rows(0), [old, new], 1.0, fixed_prefix=1).sum().backward()
    assert old_keys.grad is None and old_delta.grad is None and early.grad is None
    assert new_delta.grad is not None and torch.isfinite(new_delta.grad).all()
    assert torch.count_nonzero(new_delta.grad) == 128


class TinyBase(nn.Module):
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
        tokens = tokens + x[:, :1, :1] * torch.linspace(
            0, 0.1, 64, device=x.device, dtype=x.dtype
        ).view(1, 1, 64)
        for block in self.encoder.layers:
            tokens = tokens + 0.1 * block.norm1(tokens)
        return {"logit": self.readout(tokens[:, -1]).squeeze(1)}


@pytest.fixture
def fit_case():
    base = TinyBase()
    x = torch.linspace(-1, 1, 201).reshape(201, 1, 1).expand(-1, 128, 23)
    a = torch.ones(1, 128, 23, dtype=torch.bool).expand(201, -1, -1)
    state, history = torch.zeros(3, 66), torch.zeros(3, 128)
    ctx = (x, a, None, None, state, history, history)
    keys = key_rows(0, 0.4, 0.8, dtype=torch.float32).requires_grad_()
    fast = torch.stack([vector(v, dtype=torch.float32) for v in (0.1, -0.2, 0.3)]).requires_grad_()
    old = phase.PhaseEpisode(key_rows(0, dtype=torch.float32).requires_grad_(),
                             vector(0.4, dtype=torch.float32).requires_grad_(),
                             key_rows(0.8, dtype=torch.float32))
    early = key_rows(0.8, dtype=torch.float32).requires_grad_()
    indices = np.array([135, 136, 167, 168, 199], dtype=np.int64)
    return base, ctx, fast, keys, [old], indices, early


def exact_objective(base, ctx, fast, keys, episodes, indices, disruptive, delta):
    q = torch.from_numpy(indices)
    k = (q - 135) // 32
    slow = reference_actions(keys[k], episodes, 1.0)
    xx, aa = ctx[0][q], ctx[1][q]
    route = latent.routes_all(aa)
    zero = torch.zeros_like(fast[k])
    logits = torch.stack([
        policy.routed_pair(base, xx, aa, route, memory.total_action(zero, slow)),
        policy.routed_pair(base, xx, aa, route, memory.total_action(fast[k], slow)),
    ])
    threshold = memory.THRESHOLD
    threshold_logit = math.log(threshold / (1 - threshold))
    sign = -1 if disruptive else 1
    return F.softplus(sign * (logits - threshold_logit)).mean() + 0.1 * delta.square().sum()


@pytest.mark.parametrize("disruptive", [False, True])
def test_fit_reuses_original_optimizer_with_phase_gate_and_exact_hooks(
    fit_case, disruptive, monkeypatch
):
    base, ctx, fast, keys, old, indices, early = fit_case
    initial_state = {k: v.clone() for k, v in base.state_dict().items()}
    old_value = old[0].delta.detach().clone()
    before_private = memory._memory_actions
    prototype = keys.detach().clone()
    initial = phase.PhaseEpisode(prototype, torch.zeros(128), early)
    expected_initial = exact_objective(base, ctx, fast, keys, old + [initial], indices,
                                       disruptive, initial.delta).item()
    original_fit = phase._ORIGINAL_FIT_EPISODE
    real_actions, real_routed = phase.phase_actions, policy.routed_pair
    calls, writes, native_calls = [], [], []

    def fit_spy(*args, **kwargs):
        native_calls.append(1)
        return original_fit(*args, **kwargs)

    def action_spy(query, episodes, bandwidth, *, fixed_prefix=0):
        assert fixed_prefix == 1 and episodes[0] is old[0]
        assert isinstance(episodes[-1], phase.PhaseEpisode)
        assert episodes[-1].delta.requires_grad
        assert not episodes[-1].keys.requires_grad and not episodes[-1].early_keys.requires_grad
        out = real_actions(query, episodes, bandwidth, fixed_prefix=fixed_prefix)
        torch.testing.assert_close(out, reference_actions(query, episodes, bandwidth),
                                   rtol=2e-6, atol=1e-7)
        # Both the old and new phase banks veto the final query exactly.
        assert torch.count_nonzero(out[-1]) == 0
        old_grad, new_grad = torch.autograd.grad(
            out.sum(), (old[0].delta, episodes[-1].delta), allow_unused=True, retain_graph=True
        )
        assert old_grad is None and torch.count_nonzero(new_grad) > 0
        calls.append((out.detach().clone(), episodes[-1].delta.detach().clone()))
        return out

    def routed_spy(model, x, a, route, action):
        writes.append(action.detach().clone())
        return real_routed(model, x, a, route, action)

    # Simulate the caller's temporary core.learn injection. The fitter must use
    # the captured original rather than recursively calling memory.fit_episode.
    def injected_fit(*args):
        return phase.fit_phase(*args, early)

    monkeypatch.setattr(phase, "_ORIGINAL_FIT_EPISODE", fit_spy)
    monkeypatch.setattr(phase, "phase_actions", action_spy)
    monkeypatch.setattr(policy, "routed_pair", routed_spy)
    monkeypatch.setattr(memory, "fit_episode", injected_fit)
    with torch.no_grad():
        fitted, receipt = memory.fit_episode(base, ctx, fast, keys, old, 1.0,
                                            indices, disruptive)
    assert native_calls == [1] and len(calls) == 22 and len(writes) == 44
    assert any(torch.count_nonzero(candidate) for _, candidate in calls[1:])
    assert memory._memory_actions is before_private
    assert isinstance(fitted, phase.PhaseEpisode)
    assert receipt["steps"] == 20 and receipt["n_keys"] == 3
    assert receipt["initial_loss"] == pytest.approx(expected_initial)
    assert 0 < fitted.delta.norm() <= memory.EPISODE_RADIUS
    assert not fitted.delta.requires_grad and not fitted.early_keys.requires_grad
    torch.testing.assert_close(fitted.keys, prototype)
    torch.testing.assert_close(fitted.early_keys, early)
    q = torch.from_numpy(indices)
    k = (q - 135) // 32
    torch.testing.assert_close(writes[0], memory.total_action(torch.zeros_like(fast[k]), calls[0][0]))
    torch.testing.assert_close(writes[1], memory.total_action(fast[k], calls[0][0]))
    assert all(torch.count_nonzero(write[:, 128:]) == 0 for write in writes)
    monkeypatch.setattr(policy, "routed_pair", real_routed)
    final = exact_objective(base, ctx, fast, keys, old + [fitted], indices,
                            disruptive, fitted.delta).item()
    assert receipt["final_loss"] == pytest.approx(final)
    assert receipt["final_loss"] < receipt["initial_loss"]
    assert torch.equal(old[0].delta, old_value)
    assert all(t.grad is None for t in (keys, fast, old[0].keys, old[0].delta, early))
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in initial_state.items())
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    assert all(not block.norm1._forward_hooks for block in base.encoder.layers)


def test_fit_restores_private_hook_when_native_optimizer_raises(fit_case):
    base, ctx, fast, keys, old, _, early = fit_case
    before = memory._memory_actions
    with pytest.raises(ValueError):
        phase.fit_phase(base, ctx, fast, keys, old, 1.0, np.array([], dtype=np.int64), True, early)
    assert memory._memory_actions is before


def test_fit_through_real_simple23_forward_with_synthetic_data(monkeypatch):
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                        torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
        x = torch.randn(168, 128, 23)
        available = torch.ones_like(x, dtype=torch.bool)
        state, history = torch.zeros(2, 66), torch.zeros(2, 128)
        ctx = (x, available, None, None, state, history, history)
        keys = key_rows(0, 0.4, dtype=torch.float32)
        fast = torch.zeros(2, 128)
        old = phase.PhaseEpisode(key_rows(0, dtype=torch.float32),
                                 vector(0.2, dtype=torch.float32).requires_grad_(),
                                 key_rows(0.4, dtype=torch.float32))
        early = key_rows(0.4, dtype=torch.float32)
        indices = np.array([135, 167], dtype=np.int64)
        before = {k: v.clone() for k, v in base.state_dict().items()}
        real_routed = policy.routed_pair
        hook_calls = []

        def capture(model, xx, aa, route, action):
            hook_calls.append(action.detach().clone())
            return real_routed(model, xx, aa, route, action)

        monkeypatch.setattr(policy, "routed_pair", capture)
        fitted, receipt = phase.fit_phase(base, ctx, fast, keys, [old], 1.0,
                                          indices, True, early)
        assert receipt["steps"] == 20 and len(hook_calls) == 44
        assert 0 < fitted.delta.norm() <= memory.EPISODE_RADIUS
        assert all(torch.count_nonzero(action[1]) == 0 for action in hook_calls)
        assert old.delta.grad is None and all(p.grad is None for p in base.parameters())
        assert all(torch.equal(v, base.state_dict()[k]) for k, v in before.items())
        assert all(not block.norm1._forward_hooks for block in base.encoder.layers)
