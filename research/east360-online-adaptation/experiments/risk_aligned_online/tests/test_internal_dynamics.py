"""Function-preserving internal adaptation, not an output-score correction."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.internal_dynamics_20261022.model import (
    InternalDynamics, feedback_loss, temporary_step,
)


@pytest.fixture
def model():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    base = Simple23(Config(window=32, patch=8, dim=16, heads=2, layers=2,
                           horizon=8, fast_site='tokens'), np.zeros(23), np.ones(23))
    return InternalDynamics(base, np.ones(23), np.ones(23, bool), rank=4)


def inputs():
    x = torch.randn(2, 32, 23)
    mask = torch.rand_like(x) > .1
    target = torch.randn(2, 8, 23)
    return x, mask, target, torch.ones_like(target, dtype=torch.bool)


def test_zero_fast_preserves_parent_even_after_auxiliary_changes(model):
    x, m, _, _ = inputs()
    with torch.enable_grad():
        dummy = model.parent.initial_fast(len(x)).detach().requires_grad_(True)
        old = model.parent(x, m, dummy)['logit'].detach()
    actual = model(x, m)['logit']
    torch.testing.assert_close(actual, old, rtol=0, atol=0)
    with torch.no_grad():
        for p in model.adaptation_parameters(): p.add_(torch.randn_like(p))
    torch.testing.assert_close(model(x, m)['logit'], old, rtol=0, atol=0)


def test_internal_step_changes_risk_and_keeps_parent_frozen(model):
    x, m, truth, tm = inputs()
    # Initial persistence forecast is deliberately zero. A fitted readout is
    # needed for nonzero representation gradients; this test simulates it.
    torch.nn.init.normal_(model.decoder.weight, std=.02)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    fast = model.initial_fast(2)
    old = model(x, m, fast)['logit'].detach()
    new, info = temporary_step(model, fast, x, m, truth, tm)
    assert info['gradient_norm'] > 0
    assert (model(x, m, new)['logit'] - old).abs().max() > 1e-7
    assert new.norm(dim=(1, 2)).max() <= 1.00001
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    torch.testing.assert_close(model(x, m)['logit'], old, rtol=0, atol=0)


def test_forecast_loss_masks_unavailable_and_discrete_slots(model):
    x, m, truth, tm = inputs()
    model.active[10] = False
    m[:, -1, 2] = False
    out = model(x, m)
    a, count = feedback_loss(model, out['forecast'], x[:, -1], m[:, -1], truth, tm)
    truth[:, :, [2, 10]] = 1e20
    b, _ = feedback_loss(model, out['forecast'], x[:, -1], m[:, -1], truth, tm)
    torch.testing.assert_close(a, b)
    assert (count > 0).all()
    zero, c = feedback_loss(model, out['forecast'], x[:, -1], m[:, -1], truth, tm & False)
    assert not zero.any() and not c.any()


def test_double_backward_reaches_decoder_and_internal_basis(model):
    x, m, truth, tm = inputs()
    torch.nn.init.normal_(model.decoder.weight, std=.02)
    fast = model.initial_fast(2)
    fast, _ = temporary_step(model, fast, x, m, truth, tm, create_graph=True)
    risk = model(x + .1, m, fast)['logit'].square().mean()
    risk.backward()
    for p in (model.decoder.weight, model.basis.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parent.parameters())


def test_batch_independent_and_empty_feedback_identity(model):
    x, m, truth, tm = inputs()
    torch.nn.init.normal_(model.decoder.weight, std=.02)
    fast = model.initial_fast(2)
    together, _ = temporary_step(model, fast, x, m, truth, tm)
    for i in range(2):
        single, _ = temporary_step(model, model.initial_fast(1), x[i:i+1], m[i:i+1], truth[i:i+1], tm[i:i+1])
        torch.testing.assert_close(together[i:i+1], single, atol=1e-6, rtol=1e-5)
    empty, info = temporary_step(model, fast, x, m, truth, tm & False)
    torch.testing.assert_close(empty, fast, rtol=0, atol=0)
    assert info['observations'] == 0


def test_missing_values_invariant_and_no_hooks_leak(model):
    x, m, _, _ = inputs()
    a = model(x, m)
    x[~m] = float('nan')
    b = model(x, m)
    for k in a: torch.testing.assert_close(a[k], b[k])
    assert not model.parent.encoder.layers[-1]._forward_pre_hooks
    assert not model.parent.forecast_head._forward_pre_hooks
