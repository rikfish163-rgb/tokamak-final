from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.operational_input_gain_20261101.core import (
    gain_basis,
    apply_gain,
    backward_bag,
)
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import (
    batch,
)


@pytest.mark.parametrize("scope", ["density", "structured"])
def test_zero_update_exact_prior_and_missing_unchanged(scope):
    w = gain_basis(scope)
    theta = torch.zeros(w.shape[0], dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 128, 23)
    mask = torch.rand_like(x) > 0.1
    prior = torch.ones(23, dtype=torch.float64)
    prior[12] = 0.00024050718742271607
    actual = apply_gain(x, mask, theta, prior, w)
    expected = torch.where(mask, (x.double() * prior).float(), x)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[~mask], x[~mask])


def test_structural_zero_sign_and_statistical_constraints():
    w = gain_basis("structured")
    theta = torch.linspace(-0.5, 0.5, len(w), dtype=torch.float64, requires_grad=True)
    x = torch.ones(2, 128, 23)
    x[:, 0] = 0
    mask = torch.ones_like(x, dtype=torch.bool)
    out = apply_gain(x, mask, theta, torch.ones(23, dtype=torch.float64), w)
    assert torch.equal(out[:, 0], x[:, 0]) and (out >= 0).all()
    unchanged = [0, 2, 5, 6, 9, 10, 13, 14, 16, 18, 19, 20, 21]
    assert torch.equal(out[:, :, unchanged], x[:, :, unchanged])
    assert torch.allclose(out[:, :, 17], out[:, :, 15].square())
    out.sum().backward()
    assert torch.isfinite(theta.grad).all() and (theta.grad != 0).all()


class Toy(torch.nn.Module):
    cfg = SimpleNamespace(window=128)

    def forward(self, x, m):
        return dict(
            logit=(
                torch.where(m, x, 0.0)[:, :, 12] + 0.3 * torch.where(m, x, 0.0)[:, :, 1]
            ).mean(1)
        )


@pytest.mark.parametrize("aggregation", ["mean", "peak"])
@pytest.mark.parametrize("flag", [False, True])
def test_chunked_loss_and_gradient_match_full_batch(aggregation, flag):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(37, 23)).astype("float32")
    s = dict(values=x, available=np.ones_like(x, bool))
    ends = np.arange(37)
    w = gain_basis("structured")
    prior = torch.ones(23, dtype=torch.float64)
    theta = torch.linspace(-0.2, 0.2, len(w), dtype=torch.float64, requires_grad=True)
    model = Toy()
    xx, mm = batch(s, ends, "cpu")
    z = model(apply_gain(xx, mm, theta, prior, w), mm)["logit"]
    selected = z
    if aggregation == "peak":
        selected = z[z.detach().argmax() : z.detach().argmax() + 1]
    expected = F.softplus(0.7 - selected if flag else selected - 0.7).mean() * 0.5
    expected.backward()
    gradient = theta.grad.clone()
    theta.grad = None
    receipt = backward_bag(
        model,
        s,
        ends,
        flag,
        0.7,
        aggregation,
        theta,
        prior,
        w,
        weight=0.5,
        device="cpu",
    )
    assert abs(receipt["loss"] * 0.5 - float(expected.detach())) < 1e-7
    assert torch.allclose(theta.grad, gradient, rtol=2e-6, atol=1e-9)
    assert receipt["candidates"] == 37


def test_safety_peak_gradient_not_diluted_by_many_easy_windows():
    z = torch.tensor([-8.0] * 99 + [2.0], requires_grad=True)
    mean = F.softplus(z).mean()
    peak = F.softplus(z.max())
    a = torch.autograd.grad(mean, z, retain_graph=True)[0][-1]
    b = torch.autograd.grad(peak, z)[0][-1]
    assert torch.allclose(b, 100 * a)


def test_reject_bad_scope_nonfinite_and_wrong_dimension():
    with pytest.raises(ValueError):
        gain_basis("all23")
    x = torch.ones(1, 128, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    w = gain_basis("density")
    with pytest.raises(ValueError):
        apply_gain(x, mask, torch.zeros(2), torch.ones(23), w)
    with pytest.raises(FloatingPointError):
        apply_gain(x, mask, torch.tensor([1000.0]), torch.ones(23), w)
