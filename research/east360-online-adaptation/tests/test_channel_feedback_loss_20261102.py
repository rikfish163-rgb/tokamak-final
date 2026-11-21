from types import SimpleNamespace

import pytest
import torch

from experiments.risk_aligned_online.centered_gate_stream_20261101.model import loss
from experiments.risk_aligned_online.channel_feedback_20261102.loss import (
    channel_losses,
)


def setup():
    torch.manual_seed(77)
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.randn(23), scale=torch.ones(23))
    )
    p = torch.randn(2, 25, 23, requires_grad=True)
    truth = torch.randn_like(p)
    hold = truth[:, :1].expand_as(truth).clone()
    mask = torch.rand(2, 25, 23) > 0.2
    mask[:, :, 10] = False
    mask[:, :, 2] = False
    return model, p, truth, hold, mask


def test_losses_and_gradients_sum_to_the_existing_difference_objective():
    model, p, truth, hold, mask = setup()
    values = channel_losses(model, p, truth, hold, mask)
    original = loss(model, p, truth, mask) - loss(model, p, hold, mask)
    assert values.shape == (23,)
    assert torch.allclose(values.sum(), original, atol=1e-7, rtol=1e-6)
    split = torch.autograd.grad(values.sum(), p, retain_graph=True)[0]
    expected = torch.autograd.grad(original, p)[0]
    assert torch.allclose(split, expected, atol=1e-8, rtol=1e-5)
    assert values[10] == values[2] == 0


def test_pure_hold_and_missing_targets_do_not_create_gradients():
    model, p, truth, hold, mask = setup()
    values = channel_losses(model, p, hold, hold, mask)
    gradient = torch.autograd.grad(values.sum(), p, retain_graph=True)[0]
    assert torch.count_nonzero(gradient) == torch.count_nonzero(values) == 0
    none = channel_losses(model, p, truth, hold, torch.zeros_like(mask))
    missing = torch.autograd.grad(none.sum(), p)[0]
    assert torch.count_nonzero(none) == torch.count_nonzero(missing) == 0


def test_invalid_points_cannot_change_losses_or_gradients():
    model, p, truth, hold, mask = setup()
    expected = channel_losses(model, p, truth, hold, mask)
    truth[~mask] = float("nan")
    hold[~mask] = float("inf")
    actual = channel_losses(model, p, truth, hold, mask)
    assert torch.equal(actual, expected)


def test_observed_nonfinite_target_fails():
    model, p, truth, hold, mask = setup()
    truth[mask] = float("nan")
    with pytest.raises(ValueError):
        channel_losses(model, p, truth, hold, mask)
