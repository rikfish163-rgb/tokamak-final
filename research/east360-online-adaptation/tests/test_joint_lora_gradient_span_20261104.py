import json

import numpy as np
import pytest
import torch

from joint_lora_gradient_span_20261104 import (
    adam_first_delta,
    box_oracle,
    channel_losses,
    projection_metrics,
)


def test_channel_gradients_sum_to_global_masked_mse_with_uneven_support():
    theta = torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64, requires_grad=True)
    generator = torch.Generator().manual_seed(11)
    design = torch.randn(8, 21, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(8, 21, generator=generator, dtype=torch.float64)
    mask = torch.rand(8, 21, generator=generator) > 0.3
    mask[:, 1] = False
    pred = design @ theta
    losses = channel_losses(pred, target, mask)
    global_mse = (pred - target).square().masked_fill(~mask, 0).sum() / mask.sum()
    torch.testing.assert_close(losses.sum(), global_mse)
    (joint,) = torch.autograd.grad(global_mse, theta, retain_graph=True)
    columns = torch.stack(
        [
            torch.autograd.grad(losses[i], theta, retain_graph=True)[0]
            for i in range(21)
        ],
        dim=1,
    )
    torch.testing.assert_close(columns.sum(1), joint)
    assert torch.count_nonzero(columns[:, 1]) == 0


def test_channel_losses_never_silently_skips_unsupported_case():
    with pytest.raises(ValueError, match="support"):
        channel_losses(
            torch.zeros(8, 21), torch.zeros(8, 21), torch.zeros(8, 21, dtype=torch.bool)
        )


def test_projection_is_unconstrained_column_span_cosine_upper_bound():
    g = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    risk = torch.tensor([3.0, 4.0, 12.0], dtype=torch.float64)
    result = projection_metrics(g, risk)
    assert result["rank64"] == result["rank32"] == 2
    assert result["projection_ratio64"] == pytest.approx(5 / 13)
    assert result["projection_ratio32"] == pytest.approx(5 / 13)
    assert result["projection_energy_fraction64"] == pytest.approx(25 / 169)
    json.dumps(result, allow_nan=False)


def test_projection_handles_exactly_dependent_or_zero_columns():
    g = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    result = projection_metrics(g, torch.tensor([1.0, 1.0, 0.0]))
    assert result["rank64"] == result["rank32"] == 1
    assert result["projection_ratio64"] == pytest.approx(2**-0.5)
    zero = projection_metrics(torch.zeros_like(g), torch.ones(3))
    assert zero["rank64"] == 0 and zero["projection_ratio64"] is None


def test_box_oracle_exact_sum_and_global_optimum_against_permutations():
    scores = torch.arange(21, dtype=torch.float64) - 10
    weights = box_oracle(scores)
    assert torch.equal(weights[:10], torch.full((10,), 0.5, dtype=torch.float64))
    assert weights[10] == 1
    assert torch.equal(weights[11:], torch.full((10,), 1.5, dtype=torch.float64))
    assert weights.sum() == 21
    optimum = float(torch.dot(scores, weights))
    rng = np.random.default_rng(9)
    for _ in range(100):
        trial = weights[torch.as_tensor(rng.permutation(21))]
        assert float(torch.dot(scores, trial)) <= optimum


def test_box_oracle_cannot_fix_all_negative_dots():
    scores = torch.full((21,), -1.0, dtype=torch.float64)
    weights = box_oracle(scores)
    assert weights.min() == 0.5 and weights.max() == 1.5 and weights.sum() == 21
    assert torch.dot(scores, weights) == -21
    with pytest.raises(ValueError):
        box_oracle(torch.zeros(20))


def test_zero_moment_adam_is_a_simulated_copy_not_parameter_update():
    theta = torch.tensor([0.1, -0.4, 0.5, -0.01], dtype=torch.float32)
    original = theta.clone()
    gradient = torch.tensor([0.2, -0.5, 0.0, 0.0002], dtype=torch.float32)
    delta = adam_first_delta(theta, gradient)
    expected = -1e-4 * gradient / (gradient.abs() + 1e-8)
    torch.testing.assert_close(delta, expected, atol=4e-8, rtol=1e-4)
    assert delta[2] == 0 and torch.equal(theta, original)
    assert adam_first_delta(theta, torch.zeros_like(theta)).count_nonzero() == 0
