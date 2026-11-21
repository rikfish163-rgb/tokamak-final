"""Synthetic CPU checks for the differentiable joint-energy helper."""

import math

import pytest
import torch
import torch.nn.functional as F

from joint_trajectory_20261107 import joint_energy_score
from meta_energy_20261107 import joint_energy_no_self


def _sample_batch():
    generator = torch.Generator().manual_seed(20261107)
    encoded = torch.randn(2, 3, 2, 3, generator=generator, dtype=torch.float64)
    probabilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.45, 0.35, 0.2]], dtype=torch.float64
    )
    target = torch.randn(2, 2, 3, generator=generator, dtype=torch.float64)
    available = torch.tensor(
        [
            [[True, False, True], [True, True, False]],
            [[False, True, True], [True, False, True]],
        ]
    )
    return encoded, probabilities, target, available


def _leaves(encoded, probabilities, target):
    return tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (encoded, probabilities, target)
    )


def _score_and_first_second(encoded, probabilities, target, available):
    inputs = _leaves(encoded, probabilities, target)
    score = joint_energy_no_self(*inputs, available)
    first = torch.autograd.grad(score.sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(gradient.sum() for gradient in first), inputs)
    return score, first, second


def test_matches_original_value_and_all_first_derivatives():
    encoded, probabilities, target, available = _sample_batch()
    results = []
    for function in (joint_energy_score, joint_energy_no_self):
        inputs = _leaves(encoded, probabilities, target)
        score = function(*inputs, available)
        gradients = torch.autograd.grad(score.sum(), inputs)
        results.append((score, gradients))

    assert results[1][0].shape == (2,)
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-12, atol=1e-12)
    for actual, expected in zip(results[1][1], results[0][1]):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_single_scenario_is_masked_rms_with_no_spread():
    encoded = torch.tensor([[[[3.0, float("nan")], [4.0, 9.0]]]], dtype=torch.float64)
    target = torch.tensor([[[0.0, float("nan")], [0.0, -5.0]]], dtype=torch.float64)
    probabilities = torch.ones(1, 1, dtype=torch.float64)
    available = torch.tensor([[[True, False], [True, False]]])
    score, first, second = _score_and_first_second(encoded, probabilities, target, available)

    torch.testing.assert_close(score, torch.tensor([5.0 / math.sqrt(2.0)], dtype=torch.float64))
    assert all(torch.isfinite(gradient).all() for gradient in (*first, *second))


def test_scenario_permutation_preserves_value_and_gradient_equivariance():
    encoded, probabilities, target, available = _sample_batch()
    permutation = torch.tensor([2, 0, 1])
    original = _leaves(encoded, probabilities, target)
    reordered = _leaves(encoded[:, permutation], probabilities[:, permutation], target)
    original_score = joint_energy_no_self(*original, available)
    reordered_score = joint_energy_no_self(*reordered, available)
    original_grads = torch.autograd.grad(original_score.sum(), original)
    reordered_grads = torch.autograd.grad(reordered_score.sum(), reordered)

    torch.testing.assert_close(reordered_score, original_score, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(reordered_grads[0], original_grads[0][:, permutation])
    torch.testing.assert_close(reordered_grads[1], original_grads[1][:, permutation])
    torch.testing.assert_close(reordered_grads[2], original_grads[2])


def test_unobserved_nan_has_same_value_and_derivatives_as_finite_values():
    encoded, probabilities, target, available = _sample_batch()
    encoded_mask = available[:, None].expand_as(encoded)
    nan_encoded = torch.where(encoded_mask, encoded, float("nan"))
    nan_target = torch.where(available, target, float("nan"))
    finite_encoded = torch.where(encoded_mask, encoded, 1234.0)
    finite_target = torch.where(available, target, -1234.0)

    actual = _score_and_first_second(nan_encoded, probabilities, nan_target, available)
    expected = _score_and_first_second(finite_encoded, probabilities, finite_target, available)
    torch.testing.assert_close(actual[0], expected[0])
    for actual_group, expected_group in zip(actual[1:], expected[1:]):
        for actual_gradient, expected_gradient in zip(actual_group, expected_group):
            assert torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.count_nonzero(actual_group[0][~encoded_mask]) == 0
        assert torch.count_nonzero(actual_group[2][~available]) == 0


def test_entirely_missing_batch_has_connected_zero_first_and_second_derivatives():
    encoded = torch.full((2, 3, 2, 3), float("nan"), dtype=torch.float64)
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], dtype=torch.float64)
    target = torch.full((2, 2, 3), float("nan"), dtype=torch.float64)
    available = torch.zeros(2, 2, 3, dtype=torch.bool)
    score, first, second = _score_and_first_second(encoded, probabilities, target, available)

    torch.testing.assert_close(score, torch.zeros_like(score), rtol=0, atol=0)
    for gradient in (*first, *second):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


def test_missing_row_in_mixed_batch_has_connected_zero_first_and_second_derivatives():
    encoded, probabilities, target, available = _sample_batch()
    available[0] = False
    encoded[0] = float("nan")
    target[0] = float("nan")
    inputs = _leaves(encoded, probabilities, target)
    score = joint_energy_no_self(*inputs, available)
    assert score[0].item() == 0.0
    assert torch.isfinite(score[1])

    first = torch.autograd.grad(score[0], inputs, create_graph=True)
    second = torch.autograd.grad(sum(gradient.sum() for gradient in first), inputs)
    for gradient in (*first, *second):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


def test_hessian_vector_product_matches_float64_central_difference():
    encoded, probabilities, target, available = _sample_batch()
    inputs = _leaves(encoded, probabilities, target)
    generator = torch.Generator().manual_seed(504)
    direction = [
        torch.randn(value.shape, generator=generator, dtype=value.dtype) for value in inputs
    ]
    direction[1] = direction[1] - direction[1].mean(dim=1, keepdim=True)
    score = joint_energy_no_self(*inputs, available).sum()
    first = torch.autograd.grad(score, inputs, create_graph=True)
    hvp = torch.autograd.grad(
        sum((gradient * vector).sum() for gradient, vector in zip(first, direction)), inputs
    )

    step = 1e-5
    finite_differences = []
    for sign in (1.0, -1.0):
        shifted = tuple(
            (value.detach() + sign * step * vector).requires_grad_()
            for value, vector in zip(inputs, direction)
        )
        shifted_score = joint_energy_no_self(*shifted, available).sum()
        finite_differences.append(torch.autograd.grad(shifted_score, shifted))
    for actual, plus, minus in zip(hvp, *finite_differences):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, (plus - minus) / (2.0 * step), rtol=2e-6, atol=2e-8)


def test_full_trajectory_width_has_2300_finite_hvp_elements():
    generator = torch.Generator().manual_seed(778)
    encoded = torch.randn(1, 4, 25, 23, generator=generator, dtype=torch.float64)
    target = torch.randn(1, 25, 23, generator=generator, dtype=torch.float64)
    direction = torch.randn(encoded.shape, generator=generator, dtype=encoded.dtype)
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    available = torch.ones(1, 25, 23, dtype=torch.bool)
    original_encoded = encoded.clone().requires_grad_()
    new_encoded = encoded.clone().requires_grad_()

    original_score = joint_energy_score(original_encoded, probabilities, target, available)
    new_score = joint_energy_no_self(new_encoded, probabilities, target, available)
    original_gradient = torch.autograd.grad(original_score.sum(), original_encoded)[0]
    new_gradient = torch.autograd.grad(new_score.sum(), new_encoded, create_graph=True)[0]
    hvp = torch.autograd.grad((new_gradient * direction).sum(), new_encoded)[0]

    torch.testing.assert_close(new_score, original_score, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new_gradient, original_gradient, rtol=1e-12, atol=1e-12)
    assert hvp.numel() == 2300
    assert torch.isfinite(hvp).all()


class _SyntheticTemporalTrajectory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = torch.nn.Linear(3, 4)
        self.trajectory = torch.nn.Linear(4, 3 * 2 * 2)
        self.mixture = torch.nn.Linear(4, 3)
        self.source_head = torch.nn.Linear(4, 2)

    def forward(self, tokens):
        state = torch.tanh(self.temporal(tokens)).mean(dim=1)
        encoded = self.trajectory(state).reshape(-1, 3, 2, 2)
        return encoded, torch.softmax(self.mixture(state), dim=1), self.source_head(state)


def test_synthetic_temporal_trajectory_has_mixed_and_unrolled_outer_gradients():
    torch.manual_seed(419)
    model = _SyntheticTemporalTrajectory().double()
    tokens = torch.randn(2, 3, 3, dtype=torch.float64)
    target = torch.randn(2, 2, 2, dtype=torch.float64)
    available = torch.tensor([[[True, True], [True, False]], [[True, True], [True, True]]])
    encoded, probabilities, _ = model(tokens)
    inner = joint_energy_no_self(encoded, probabilities, target, available).mean()
    temporal_grad, trajectory_grad = torch.autograd.grad(
        inner, (model.temporal.weight, model.trajectory.weight), create_graph=True
    )
    mixed = torch.autograd.grad(
        temporal_grad.square().sum(), model.trajectory.weight, retain_graph=True
    )[0]

    updated = {
        "temporal.weight": model.temporal.weight - 0.2 * temporal_grad,
        "trajectory.weight": model.trajectory.weight - 0.2 * trajectory_grad,
    }
    source_tokens = torch.randn(2, 3, 3, dtype=torch.float64)
    _, _, source_logits = torch.func.functional_call(model, updated, (source_tokens,))
    outer = F.cross_entropy(source_logits, torch.tensor([0, 1]))
    outer_grad = torch.autograd.grad(outer, model.trajectory.weight)[0]

    for gradient in (temporal_grad, trajectory_grad, mixed, outer_grad):
        assert torch.isfinite(gradient).all()
        assert torch.linalg.vector_norm(gradient) > 1e-12


def test_coincident_target_on_observed_entries_raises():
    encoded = torch.tensor([[[[1.0, 900.0], [2.0, 800.0]], [[4.0, 1.0], [5.0, 1.0]]]])
    probabilities = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[[1.0, -900.0], [2.0, -800.0]]])
    available = torch.tensor([[[True, False], [True, False]]])
    with pytest.raises(ValueError, match=r"non-coincident.*target"):
        joint_energy_no_self(encoded, probabilities, target, available)


def test_coincident_distinct_scenarios_raise_even_with_zero_probability():
    encoded = torch.tensor([[[[1.0, 900.0], [2.0, 800.0]], [[1.0, -1.0], [2.0, -1.0]]]])
    probabilities = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[[4.0, 2.0], [5.0, 2.0]]])
    available = torch.tensor([[[True, False], [True, False]]])
    with pytest.raises(ValueError, match=r"non-coincident.*scenarios"):
        joint_energy_no_self(encoded, probabilities, target, available)
