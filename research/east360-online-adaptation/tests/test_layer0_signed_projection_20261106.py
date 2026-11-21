from dataclasses import is_dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import latent_trajectory_adapt_20261105 as latent
import trajectory_policy_20261105 as policy
from layer0_signed_projection_20261106 import (
    EPS,
    SignedProjection,
    project_layer0,
    project_unsupported,
)


def actions_with_first(values, *, dtype=torch.float64):
    actions = torch.zeros((len(values), 128), dtype=dtype)
    actions[:, 0] = torch.tensor(values, dtype=dtype)
    return actions


def test_signed_linear_projection_keeps_downward_and_tolerance_rows_exact():
    actions = actions_with_first([1, -2, -3, 2, 5e-6, 0])
    actions[:, 1] = torch.tensor([3, -4, 5, 6, 7, 8], dtype=actions.dtype)
    actions.requires_grad_()
    before = actions.detach().clone()
    slopes = torch.tensor([2, 3, -4, -5, 1, 2], dtype=actions.dtype)
    offsets = torch.tensor([10, -1, 2, -5, 0, 3], dtype=actions.dtype)

    def evaluate(indices, values):
        return offsets[indices] + slopes[indices] * values[:, 0]

    def gradient(indices, values):
        result = torch.zeros_like(values)
        result[:, 0] = slopes[indices]
        return result

    result = project_unsupported(actions, evaluate, gradient)
    assert is_dataclass(result) and isinstance(result, SignedProjection)
    assert EPS == 1e-5
    assert result.restricted.tolist() == [True, False, True, False, False, False]
    assert not result.at_zero.any()
    assert torch.equal(result.values[[1, 3, 4, 5]], before[[1, 3, 4, 5]])
    assert torch.equal(result.values[:, 1], before[:, 1])
    torch.testing.assert_close(result.values[[0, 2], 0], torch.zeros(2, dtype=actions.dtype))
    torch.testing.assert_close(result.native, offsets)
    torch.testing.assert_close(result.initial, evaluate(torch.arange(6), before))
    torch.testing.assert_close(result.final, evaluate(torch.arange(6), result.values))
    assert ((result.final - result.native) <= EPS).all()
    assert (result.values.norm(dim=1) <= before.norm(dim=1)).all()
    assert torch.equal(actions.detach(), before) and actions.grad is None
    assert all(not tensor.requires_grad for tensor in (
        result.values, result.native, result.initial, result.final,
    ))


def test_nonlinear_first_order_estimates_are_not_a_safety_certificate():
    actions = actions_with_first([1, -1])
    actions[:, 1] = torch.tensor([3.0, 4.0])
    evaluated, gradients = [], []

    def evaluate(indices, values):
        evaluated.append((indices.clone(), values.clone()))
        return values[:, 0].square()

    def gradient(indices, values):
        gradients.append((indices.clone(), values.clone()))
        result = torch.zeros_like(values)
        result[:, 0] = 2 * values[:, 0]
        return result

    result = project_unsupported(actions, evaluate, gradient)
    assert len(gradients) == 2
    torch.testing.assert_close(gradients[0][1][:, 0], torch.tensor([1.0, -1.0], dtype=actions.dtype))
    torch.testing.assert_close(gradients[1][1][:, 0], torch.tensor([0.5, -0.5], dtype=actions.dtype))
    assert result.restricted.tolist() == [True, True]
    assert result.at_zero.tolist() == [True, True]
    assert torch.equal(result.values, torch.zeros_like(actions))
    assert torch.equal(result.final, result.native)
    # Both nonlinear proposals were evaluated before falling back to zero.
    assert any(torch.allclose(v[:, 0].abs(), torch.tensor([0.25, 0.25], dtype=actions.dtype)) for _, v in evaluated)


def test_second_correction_is_evaluated_only_for_remaining_violators():
    actions = actions_with_first([1, 1])
    actions[:, 1] = 2
    gradients = []
    examined = []

    def evaluate(indices, values):
        examined.append((indices.clone(), values.clone()))
        x = values[:, 0]
        nonlinear = torch.where(x > 0.75, 2 * x - 1.15,
                                torch.where(x > 0.2, 0.2 + 0.2 * x, 1.2 * x))
        return torch.where(indices == 0, x, nonlinear)

    def gradient(indices, values):
        gradients.append(indices.clone())
        x = values[:, 0]
        nonlinear = torch.where(x > 0.75, 2.0, torch.where(x > 0.2, 0.2, 1.2))
        result = torch.zeros_like(values)
        result[:, 0] = torch.where(indices == 0, 1.0, nonlinear)
        return result

    result = project_unsupported(actions, evaluate, gradient)
    assert [ids.tolist() for ids in gradients] == [[0, 1], [1]]
    assert result.restricted.tolist() == [True, True]
    assert result.at_zero.tolist() == [False, False]
    assert result.values[0, 0] == 0 and result.values[0, 1] == 2
    assert result.values[1, 0] < 0 and result.values[1, 1] == 2
    assert result.final[1] < result.native[1]
    assert any(ids.tolist() == [1] and torch.equal(v, result.values[1:2]) for ids, v in examined)


def test_invalid_gradient_or_proposal_falls_back_to_exact_zero():
    actions = actions_with_first([1, 1, 1])
    actions[:, 1] = 1

    def evaluate(indices, values):
        result = values[:, 0].clone()
        candidate = (indices == 2) & (values[:, 0] != 0) & (values[:, 0] != 1)
        result[candidate] = float("nan")
        return result

    def gradient(indices, values):
        result = torch.zeros_like(values)
        result[:, 0] = torch.tensor([float("nan"), 0.0, 2.0], dtype=values.dtype)[indices]
        return result

    result = project_unsupported(actions, evaluate, gradient)
    assert result.restricted.tolist() == [True, True, True]
    assert result.at_zero.tolist() == [True, True, True]
    assert torch.equal(result.values, torch.zeros_like(actions))
    assert torch.equal(result.final, result.native)


def test_overshooting_proposal_is_capped_to_the_original_norm():
    actions = actions_with_first([1], dtype=torch.float32)
    actions[:, 1] = 0.1
    examined = []

    def evaluate(indices, values):
        examined.append(values.clone())
        return values[:, 0]

    def gradient(indices, values):
        result = torch.zeros_like(values)
        result[:, 0] = 0.1
        return result

    result = project_unsupported(actions, evaluate, gradient)
    assert not result.at_zero.item()
    assert result.final.item() < result.native.item()
    assert result.values.double().norm() <= actions.double().norm()
    assert all(v.double().norm() <= actions.double().norm() for v in examined[2:])


def test_empty_input_returns_empty_result_without_invoking_callbacks():
    actions = torch.empty((0, 128))

    def unexpected(*_args):
        pytest.fail("empty inputs must not call the evaluator or gradient")

    result = project_unsupported(actions, unexpected, unexpected)
    assert result.values.shape == (0, 128)
    for field in (result.native, result.initial, result.final, result.restricted, result.at_zero):
        assert field.shape == (0,)
    assert result.restricted.dtype == result.at_zero.dtype == torch.bool


def test_epsilon_boundary_and_a_validated_zero_candidate():
    actions = actions_with_first([EPS, EPS + 1e-10, 1.0, 0.0])

    def gradient(_indices, values):
        result = torch.zeros_like(values)
        result[:, 0] = 1
        return result

    result = project_unsupported(actions, lambda _i, v: v[:, 0], gradient)
    assert result.restricted.tolist() == [False, True, True, False]
    assert result.at_zero.tolist() == [False, True, True, False]
    assert torch.equal(result.values[0], actions[0])
    assert torch.equal(result.values[1:], torch.zeros_like(actions[1:]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_projection_preserves_dtype_and_never_expands_represented_norm(dtype):
    actions = actions_with_first([1, 2, -2], dtype=dtype)
    actions[:, 1] = 0.1

    def gradient(_indices, values):
        result = torch.zeros_like(values)
        result[:, 0] = 0.1
        return result

    result = project_unsupported(actions, lambda _i, v: v[:, 0], gradient)
    assert result.values.dtype == result.native.dtype == result.final.dtype == dtype
    assert torch.equal(result.values[2], actions[2])
    assert (result.values.double().norm(dim=1) <= actions.double().norm(dim=1)).all()
    assert ((result.final.double() - result.native.double()) <= EPS).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_random_linear_rows_preserve_safe_actions_and_validate_every_result(dtype):
    generator = torch.Generator().manual_seed(104)
    actions = torch.randn(97, 128, generator=generator, dtype=dtype)
    weights = torch.randn(97, 128, generator=generator, dtype=dtype)
    offsets = torch.randn(97, generator=generator, dtype=dtype)

    def evaluate(indices, values):
        return offsets[indices] + (weights[indices] * values).sum(dim=1)

    def gradient(indices, _values):
        return weights[indices]

    result = project_unsupported(actions, evaluate, gradient)
    assert result.restricted.any() and (~result.restricted).any()
    assert torch.equal(result.values[~result.restricted], actions[~result.restricted])
    torch.testing.assert_close(result.final, evaluate(torch.arange(len(actions)), result.values),
                               rtol=0, atol=2e-6 if dtype == torch.float32 else 1e-12)
    assert ((result.final.double() - result.native.double()) <= EPS).all()
    assert (result.values.double().norm(dim=1) <= actions.double().norm(dim=1)).all()


def test_extreme_finite_steps_fall_back_without_propagating_nonfinite_values():
    actions = actions_with_first([1, 1])

    def evaluate(indices, values):
        extreme = torch.where(values[:, 0] == 0, values.new_full((len(values),), -1e308),
                              values.new_full((len(values),), 1e308))
        return torch.where(indices == 0, values[:, 0], extreme)

    def gradient(_indices, values):
        result = torch.zeros_like(values)
        result[:, 0] = 1e308
        return result

    result = project_unsupported(actions, evaluate, gradient)
    assert result.at_zero.tolist() == [True, True]
    assert torch.equal(result.final, result.native)
    assert torch.isfinite(result.final).all() and torch.isfinite(result.values).all()


@pytest.mark.parametrize("actions", [
    torch.zeros(128), torch.zeros(2, 127), torch.zeros(1, 128, dtype=torch.int64),
    torch.full((1, 128), float("nan")), torch.full((1, 128), float("inf")),
    [[0] * 128],
])
def test_actions_must_be_finite_floating_128_vectors(actions):
    with pytest.raises(ValueError):
        project_unsupported(actions, lambda *_: None, lambda *_: None)


@pytest.mark.parametrize("bad", [
    lambda v: torch.zeros(len(v), 1, dtype=v.dtype),
    lambda v: torch.zeros(len(v), dtype=torch.float32),
    lambda v: torch.full((len(v),), float("nan"), dtype=v.dtype),
    lambda v: torch.zeros(len(v), dtype=v.dtype, device="meta"),
    lambda v: [0] * len(v),
])
def test_initial_evaluations_require_matching_finite_vector(bad):
    actions = actions_with_first([1])
    with pytest.raises(ValueError):
        project_unsupported(actions, lambda _i, v: bad(v), lambda *_: None)


@pytest.mark.parametrize("bad", [
    lambda v: torch.zeros(len(v), 127, dtype=v.dtype),
    lambda v: torch.zeros_like(v, dtype=torch.float32),
    lambda v: torch.zeros(v.shape, dtype=v.dtype, device="meta"),
    lambda v: [0] * len(v),
])
def test_gradients_require_matching_matrix_dtype_and_device(bad):
    actions = actions_with_first([1])
    with pytest.raises(ValueError):
        project_unsupported(actions, lambda _i, v: v[:, 0], lambda _i, v: bad(v))


class TinyBase(nn.Module):
    """Frozen CPU model using both real routed_pair norm1 hooks."""

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
            self.readout.weight[0, 0] = 3
            self.readout.weight[0, 1] = -2
        self.eval().requires_grad_(False)

    def forward(self, x, available):
        features = torch.linspace(-1, 1, 64, dtype=x.dtype, device=x.device)
        tokens = features.view(1, 1, 64).expand(len(x), 17, 64)
        tokens = tokens + x[:, :1, :1] * features.view(1, 1, 64) * 0.01
        for block in self.encoder.layers:
            tokens = tokens + 0.1 * block.norm1(tokens)
        return {"logit": self.readout(tokens[:, -1]).squeeze(1)}


def test_layer0_uses_actual_hooks_anchors_frozen_state_and_bounded_batches(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = TinyBase()
        count = 130
        length = policy.FIRST + policy.STRIDE * (count - 1) + 1
        time_values = torch.arange(length, dtype=torch.float32, requires_grad=True)
        x = time_values[:, None, None].expand(-1, 128, 23)
        available = torch.ones((length, 128, 23), dtype=torch.bool)
        available[:, :8] = False
        actions = torch.zeros(count, 128, requires_grad=True)
        with torch.no_grad():
            actions[:, 64] = 0.4
            actions[1::2, 64] = -0.4
            actions[:, 65] = 0.1
        saved = actions.detach().clone()
        state = {name: value.clone() for name, value in base.state_dict().items()}
        original = policy.routed_pair
        calls = []

        def capture(model, xx, aa, routes, delta):
            assert len(xx) <= 128
            assert torch.equal(routes, latent.routes_all(aa))
            assert not torch.count_nonzero(delta[:, 128:])
            calls.append((xx[:, 0, 0].detach().clone(), delta.detach().clone()))
            return original(model, xx, aa, routes, delta)

        monkeypatch.setattr(policy, "routed_pair", capture)
        with torch.no_grad():
            result = project_layer0(base, (x, available), actions)
        assert result.restricted.any() and (~result.restricted).any()
        assert torch.equal(actions.detach(), saved)
        assert not actions.grad and time_values.grad is None
        assert all(p.grad is None and not p.requires_grad for p in base.parameters())
        assert all(torch.equal(value, base.state_dict()[name]) for name, value in state.items())
        assert all(not block.norm1._forward_hooks for block in base.encoder.layers)
        anchors = torch.arange(count) * policy.STRIDE + policy.FIRST
        assert torch.equal(torch.cat([c[0] for c in calls[:2]]), anchors.float())
        assert all(not torch.count_nonzero(delta) for _, delta in calls[:2])
        assert ((result.final.double() - result.native.double()) <= EPS).all()
        assert (result.values.double().norm(dim=1) <= saved.double().norm(dim=1)).all()
        with torch.no_grad():
            exact = []
            for lo in range(0, count, 128):
                q = anchors[lo:lo + 128]
                delta = result.values[lo:lo + 128]
                exact.append(original(base, x[q], available[q], latent.routes_all(available[q]),
                                      torch.cat((delta, torch.zeros_like(delta)), dim=1)))
        torch.testing.assert_close(result.final, torch.cat(exact), rtol=0, atol=1e-6)

        changed_time = time_values.detach().clone()
        changed_time[policy.FIRST + policy.STRIDE * 5:] = float("nan")
        changed_x = changed_time[:, None, None].expand(-1, 128, 23)
        prefix = project_layer0(base, (changed_x, available), actions[:5])
        torch.testing.assert_close(prefix.values, result.values[:5], rtol=0, atol=2e-5)
        torch.testing.assert_close(prefix.final, result.final[:5], rtol=0, atol=2e-5)
        assert torch.equal(prefix.restricted, result.restricted[:5])
        assert torch.equal(prefix.at_zero, result.at_zero[:5])
    finally:
        torch.set_num_threads(old_threads)


def test_layer0_validates_source_context_and_available_anchor_count():
    base = TinyBase()
    x = torch.zeros(168, 128, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    action = torch.zeros(2, 128)
    assert project_layer0(base, (x, mask), torch.zeros(0, 128)).values.shape == (0, 128)
    with pytest.raises(ValueError):
        project_layer0(base, (x, mask), torch.zeros(3, 128))
    with pytest.raises(ValueError):
        project_layer0(base, (x, mask.float()), action)
    with pytest.raises(ValueError):
        project_layer0(base, (x.double(), mask), action)
    poisoned = x.clone()
    poisoned[policy.FIRST, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        project_layer0(base, (poisoned, mask), action)
    base.train()
    with pytest.raises(ValueError):
        project_layer0(base, (x, mask), action)
    base.eval()
    next(base.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):
        project_layer0(base, (x, mask), action)


def test_real_simple23_source_on_synthetic_anchors_with_inference_mode():
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(911)
            base = Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                torch.zeros(23), torch.ones(23),
            ).eval().requires_grad_(False)
            x = torch.randn(168, 128, 23, requires_grad=True)
            a = torch.ones_like(x, dtype=torch.bool)
            anchors = torch.tensor([policy.FIRST, policy.FIRST + policy.STRIDE])
            xx, aa = x.detach()[anchors], a[anchors]
            routes = latent.routes_all(aa)
            temporary = torch.zeros(2, 128, requires_grad=True)
            z = policy.routed_pair(base, xx, aa, routes,
                                   torch.cat((temporary, torch.zeros_like(temporary)), dim=1))
            (first_gradient,) = torch.autograd.grad(z.sum(), temporary)
            assert first_gradient.norm(dim=1).min() > 0
            actions = 0.5 * first_gradient / first_gradient.norm(dim=1, keepdim=True)
            actions[1] *= -1
            actions.requires_grad_()
            state = {name: value.clone() for name, value in base.state_dict().items()}
            with torch.inference_mode():
                result = project_layer0(base, (x, a), actions)
            assert result.restricted.tolist() == [True, False]
            assert torch.equal(result.values[1], actions.detach()[1])
            with torch.no_grad():
                actual = policy.routed_pair(
                    base, xx, aa, routes,
                    torch.cat((result.values, torch.zeros_like(result.values)), dim=1),
                )
            torch.testing.assert_close(result.final, actual, rtol=0, atol=1e-6)
            assert ((actual.double() - result.native.double()) <= EPS).all()
            assert (result.values.double().norm(dim=1) <= actions.detach().double().norm(dim=1)).all()
            assert x.grad is None and actions.grad is None
            assert all(p.grad is None and not p.requires_grad for p in base.parameters())
            assert all(torch.equal(value, base.state_dict()[name]) for name, value in state.items())
            assert all(not block.norm1._forward_hooks for block in base.encoder.layers)
    finally:
        torch.set_num_threads(old_threads)
