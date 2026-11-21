"""Regression tests for source-prepared, observed-only masked adaptation."""
import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.masked_state_20261022.model import (
    MaskedState, masked_views, reconstruction_loss, adapt_observed,
)
from experiments.risk_aligned_online.masked_state_20261022.probe import distort


@pytest.fixture
def model():
    torch.manual_seed(19)
    torch.set_num_threads(2)
    base = Simple23(Config(window=32, patch=8, dim=16, heads=2, layers=2,
                           fast_site='tokens'), np.zeros(23), np.ones(23))
    return MaskedState(base, np.ones(23))


def example():
    x = torch.randn(1, 32, 23)
    m = torch.ones_like(x, dtype=torch.bool)
    m[:, :7] = False
    m[:, :, 2] = False
    return x, m


def test_unadapted_risk_is_parent_and_parent_input_unchanged(model):
    x, m = example()
    original = model.base(x, m)['logit']
    torch.testing.assert_close(model(x, m)['logit'], original, rtol=0, atol=0)
    assert not model.base.encoder.layers[0]._forward_hooks


def test_whole_channel_masks_never_expose_heldout_value(model):
    x, m = example()
    xx, mm, target = masked_views(x, m)
    assert xx.shape == (4, 32, 23)
    assert not (mm & target).any()
    assert torch.equal(target.sum(0), m[0].to(torch.int64))
    for i in range(4):
        assert not mm[i, :, i::4].any()
    before = model(xx, mm, mode='reconstruct')['reconstruction']
    xx = xx.clone(); xx[target] = float('nan')
    after = model(xx, mm, mode='reconstruct')['reconstruction']
    torch.testing.assert_close(before, after)


def test_loss_ignores_missing_and_has_no_target_gradient(model):
    x, m = example(); x.requires_grad_(True)
    xx, mm, held = masked_views(x, m)
    pred = torch.randn_like(xx, requires_grad=True)
    a = reconstruction_loss(model, pred, xx, held)
    btruth = xx.detach().clone(); btruth[~held] = float('nan')
    b = reconstruction_loss(model, pred, btruth, held)
    torch.testing.assert_close(a, b)
    a.backward()
    assert x.grad is None
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_temporary_updates_change_internal_tensors_not_persistent_model(model):
    x, m = example()
    before = copy.deepcopy(model.state_dict())
    old = model(x, m)['logit'].detach()
    fast, audit = adapt_observed(model, x, m, learning_rate=.01, steps=2)
    assert audit['gradient_steps'] == 2 and audit['parameter_delta_norm'] > 0
    assert all(k.startswith(('base.embedding.', 'base.encoder.layers.0.', 'base.position', 'base.cls')) for k in fast)
    out = model.predict_fast(x, m, fast)['logit'].detach()
    assert (out-old).abs().max() > 1e-8
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    torch.testing.assert_close(model(x, m)['logit'], old)


def test_empty_observations_do_not_update_and_no_batch_mixing(model):
    x, m = example()
    fast, audit = adapt_observed(model, x, m & False, learning_rate=.01, steps=2)
    assert audit['gradient_steps'] == 0 and audit['parameter_delta_norm'] == 0
    with pytest.raises(ValueError, match='one observed'):
        adapt_observed(model, x.repeat(2, 1, 1), m.repeat(2, 1, 1), learning_rate=.01, steps=2)


def test_parameter_groups_separate_encoder_from_risk_readout(model):
    encoder = set(model.encoder_names())
    risk = set(model.risk_names())
    assert encoder and risk and not encoder & risk
    assert 'base.risk_head.weight' in risk
    assert not any('risk_head' in k for k in encoder)
    model.set_stage('reconstruction')
    assert all(p.requires_grad == (k in encoder or k.startswith(('reconstructor.', 'event_head.')))
               for k, p in model.named_parameters())
    model.set_stage('risk')
    assert all(p.requires_grad == (k in risk) for k, p in model.named_parameters())


def test_known_observation_transform_is_consistent_and_invertible(model):
    x, m = example()
    gain = torch.linspace(.8, 1.2, 23)
    bias = torch.linspace(-.5, .5, 23)
    moved = distort(model, x, m, gain, bias)
    restored = distort(model, moved, m, gain.reciprocal(), -bias/gain)
    torch.testing.assert_close(restored[m], x[m], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(moved[~m], x[~m], atol=0, rtol=0)
    for c in range(23):
        expected = torch.asinh((x[..., c]-model.base.center[c])/model.base.scale[c])*gain[c]+bias[c]
        got = torch.asinh((moved[..., c]-model.base.center[c])/model.base.scale[c])
        torch.testing.assert_close(got[m[..., c]], expected[m[..., c]], atol=3e-6, rtol=3e-6)
