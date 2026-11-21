from types import SimpleNamespace

import torch

from experiments.risk_aligned_online.innovation_metric_20261102.core import (
    innovation_loss,
    metric_direction,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.gradient_trace import (
    channel_terms,
    metric_components,
)


def test_channel_terms_exactly_reconstruct_masked_centered_objective_and_gradient():
    gen = torch.Generator().manual_seed(131)
    prediction = torch.randn(1, 25, 23, generator=gen, requires_grad=True)
    truth = torch.randn(1, 25, 23, generator=gen)
    mask = torch.rand(1, 25, 23, generator=gen) > 0.2
    mask[:, :, 2] = False
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    )
    terms = channel_terms(model, prediction, truth, mask)
    exact = innovation_loss(model, prediction, truth, mask)
    torch.testing.assert_close(terms.sum(), exact)
    assert terms.shape == (23,) and terms[2] == 0
    g1 = torch.autograd.grad(terms.sum(), prediction, retain_graph=True)[0]
    g2 = torch.autograd.grad(exact, prediction)[0]
    torch.testing.assert_close(g1, g2)
    altered = truth.masked_fill(~mask, float("nan"))
    torch.testing.assert_close(channel_terms(model, prediction, altered, mask), terms)


def test_metric_channel_components_sum_to_actual_hidden_metric_step():
    gen = torch.Generator().manual_seed(131)
    pack = {
        "phi": torch.randn(1, 16, 4, generator=gen),
        "valid": torch.ones(1, 17, dtype=torch.bool),
    }
    pack["valid"][:, :2] = False
    gradients = torch.randn(23, 4, 64, generator=gen) * 0.0001
    total = gradients.sum(0)
    total[1, 10] += 1e-6  # Explicit whole-loss versus per-channel reduction residual.
    result = metric_components(pack, gradients, total)
    torch.testing.assert_close(
        result["channels"].sum(0) + result["reduction_residual"],
        metric_direction(pack, total).double(),
        atol=1e-7,
        rtol=1e-5,
    )
    assert torch.count_nonzero(result["reduction_residual"]) > 0
    zero = metric_components(pack, torch.zeros_like(gradients), torch.zeros(4, 64))
    assert torch.count_nonzero(zero["channels"]) == 0
    assert torch.count_nonzero(zero["reduction_residual"]) == 0
