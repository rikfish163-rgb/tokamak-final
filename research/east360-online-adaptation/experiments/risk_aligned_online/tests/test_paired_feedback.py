"""Paired TRAIN teaching must never enter the label-free deployed update."""
import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.paired_feedback_20261022.model import (
    PairedFeedback, initial_encoder, update_observed, representation_loss, risk_state_loss,
)


@pytest.fixture
def model():
    torch.manual_seed(77); torch.set_num_threads(2)
    base = Simple23(Config(window=32, patch=8, dim=16, heads=2,
                           fast_site='tokens'), np.zeros(23), np.ones(23))
    model = PairedFeedback(base, np.ones(23)); model.prepare_decoder()
    return model


def sample():
    x = torch.randn(1, 32, 23); m = torch.ones_like(x, dtype=torch.bool)
    m[:, :3] = False; m[:, :, 6] = False
    return x, m


def test_static_risk_and_persistent_state_are_unchanged(model):
    x, m = sample(); old = copy.deepcopy(model.state_dict())
    expected = model.base(x, m)['logit'].detach()
    torch.testing.assert_close(model(x, m)['logit'], expected, rtol=0, atol=0)
    fast, audit = update_observed(model, x, m, steps=2, learning_rate=.01)
    assert audit['steps'] == 2 and audit['delta'] > 0
    assert set(fast) == set(model.encoder_names())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.items())
    assert (model.predict_fast(x, m, fast)['logit'].detach()-expected).abs().max() > 1e-8
    assert all(not p.requires_grad for p in model.base.parameters())


def test_same_observation_reset_is_deterministic_and_no_cross_shot_batch(model):
    x, m = sample()
    a, _ = update_observed(model, x, m, steps=2, learning_rate=.01)
    b, _ = update_observed(model, x, m, steps=2, learning_rate=.01)
    for k in a: torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
    with pytest.raises(ValueError, match='one observed'):
        update_observed(model, x.repeat(2,1,1), m.repeat(2,1,1), steps=2, learning_rate=.01)


def test_outer_representation_target_is_detached(model):
    h = torch.randn(1, 4, 16, requires_grad=True)
    target = torch.randn_like(h, requires_grad=True)
    valid = torch.tensor([[True, True, False, False]])
    loss = representation_loss(h, target, valid)
    loss.backward()
    assert target.grad is None
    assert h.grad[:, :2].abs().sum() > 0 and h.grad[:, 2:].abs().sum() == 0


def test_meta_gradient_reaches_decoder_through_encoder_update(model):
    x, m = sample(); clean = x*.8
    target = model(clean, m, mode='state')['tokens'].detach()
    fast, _ = update_observed(model, x, m, steps=2, learning_rate=.01, create_graph=True)
    out = model.predict_fast(x, m, fast, mode='state')
    loss = representation_loss(out['tokens'], target, out['valid'])
    gradients = torch.autograd.grad(loss, tuple(model.reconstructor.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 1e-6


def test_meta_gradient_matches_directional_finite_difference(model):
    x, m = sample(); target = model(x*.7, m, mode='state')['tokens'].detach()
    weight = model.reconstructor.weight
    direction = torch.randn_like(weight); direction /= direction.norm()
    def objective(graph):
        fast, _ = update_observed(model, x, m, steps=2, learning_rate=.02, create_graph=graph)
        out = model.predict_fast(x, m, fast, mode='state')
        return representation_loss(out['tokens'], target, out['valid'])
    grad = torch.autograd.grad(objective(True), weight)[0]
    exact = (grad*direction).sum()
    old = weight.detach().clone(); eps = .01
    with torch.no_grad(): weight.copy_(old+eps*direction)
    plus = objective(False).detach()
    with torch.no_grad(): weight.copy_(old-eps*direction)
    minus = objective(False).detach()
    with torch.no_grad(): weight.copy_(old)
    torch.testing.assert_close(exact, (plus-minus)/(2*eps), atol=3e-6, rtol=.08)


def test_missing_truth_and_empty_window_are_not_training_data(model):
    x, m = sample(); x[~m] = float('nan')
    fast, audit = update_observed(model, x, m, steps=2, learning_rate=.01)
    assert audit['observations'] == int(m.sum()) and audit['delta'] > 0
    untouched, audit = update_observed(model, x, m & False, steps=2, learning_rate=.01)
    origin = initial_encoder(model)
    assert audit['steps'] == 0
    assert all(torch.equal(untouched[k], origin[k]) for k in origin)


def test_no_outcome_or_clean_reference_in_inner_api(model):
    x, m = sample()
    with pytest.raises(TypeError):
        update_observed(model, x, m, steps=2, learning_rate=.01, y=torch.ones(1))
    with pytest.raises(TypeError):
        update_observed(model, x, m, steps=2, learning_rate=.01, clean=x)


def test_risk_state_is_exactly_what_the_fixed_risk_head_reads(model):
    x,m = sample(); fast,_ = update_observed(model,x,m,steps=2,learning_rate=.01)
    for state in (initial_encoder(model),fast):
        h = model.predict_fast(x,m,state,mode='risk_state')['risk_state']
        explicit = model.base.risk_head(h).squeeze(-1)
        actual = model.predict_fast(x,m,state)['logit']
        torch.testing.assert_close(explicit,actual,atol=2e-7,rtol=2e-6)


def test_risk_state_recovery_bounds_score_change_and_detaches_reference(model):
    h = torch.randn(3,model.cfg.dim,requires_grad=True)
    ref = torch.randn_like(h,requires_grad=True)
    bound = risk_state_loss(model,h,ref)
    difference = (model.base.risk_head(h)-model.base.risk_head(ref)).square().mean()
    assert bound >= difference
    bound.backward()
    assert ref.grad is None and h.grad.abs().sum() > 0
