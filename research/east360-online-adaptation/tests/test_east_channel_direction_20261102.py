from types import SimpleNamespace

import torch

from experiments.risk_aligned_online.east_decoder_calibration_20261102.channel_direction import (
    channel_losses,
    risk_contributions,
)
from experiments.risk_aligned_online.innovation_metric_20261102.core import (
    innovation_loss,
    metric_direction,
)


def test_channel_losses_reconstruct_the_actual_masked_centered_objective():
    torch.manual_seed(3)
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    )
    predicted = torch.randn(1, 25, 23, requires_grad=True)
    target = torch.randn_like(predicted)
    mask = torch.rand_like(predicted) > 0.2
    mask[:, :, 2] = False
    losses = channel_losses(model, predicted, target, mask)
    assert losses.shape == (23,) and losses[2] == 0
    torch.testing.assert_close(
        losses.sum(), innovation_loss(model, predicted, target, mask)
    )
    g0 = torch.autograd.grad(losses.sum(), predicted, retain_graph=True)[0]
    g1 = torch.autograd.grad(
        innovation_loss(model, predicted, target, mask), predicted
    )[0]
    torch.testing.assert_close(g0, g1)


def test_channel_directions_reconstruct_risk_slope_with_one_shared_total_cap():
    generator = torch.Generator().manual_seed(5)
    gradients = torch.randn(23, 4, 64, generator=generator, dtype=torch.float64) * 0.01
    risk = torch.randn(4, 64, generator=generator, dtype=torch.float64)
    pack = dict(
        phi=torch.randn(1, 16, 4, generator=generator, dtype=torch.float64),
        valid=torch.ones(1, 17, dtype=torch.bool),
    )
    contributions = risk_contributions(pack, gradients, risk)
    total = metric_direction(pack, gradients.sum(0))
    torch.testing.assert_close(contributions.sum(), (risk * total).sum())
    assert contributions.shape == (23,)
