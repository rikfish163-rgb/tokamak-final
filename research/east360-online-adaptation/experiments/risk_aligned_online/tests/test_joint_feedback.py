"""The same inner operator must train its recipient, not just the decoder."""
import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.paired_feedback_20261022.model import (
    PairedFeedback, update_observed,
)
from experiments.risk_aligned_online.joint_feedback_20261022.train import (
    enable_joint, objective, sampled_positive_probability,
)


@pytest.fixture
def model():
    torch.manual_seed(131)
    torch.set_num_threads(2)
    base = Simple23(Config(window=32, patch=8, dim=16, heads=2,
                           fast_site='tokens'), np.zeros(23), np.ones(23))
    result = PairedFeedback(base, np.ones(23))
    enable_joint(result)
    return result


def example():
    x = torch.randn(1, 32, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :3] = False
    return x, mask, torch.ones(1), torch.tensor([2])


def test_connected_and_deployed_update_have_equal_values_and_no_mutation(model):
    x, mask, _, _ = example()
    saved = copy.deepcopy(model.state_dict())
    connected, _ = update_observed(model, x, mask, learning_rate=.01, steps=2,
                                   create_graph=True, track_initial=True)
    deployed, _ = update_observed(model, x, mask, learning_rate=.01, steps=2)
    for key in connected:
        torch.testing.assert_close(connected[key], deployed[key], rtol=0, atol=0)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())
    with pytest.raises(ValueError, match='outer'):
        update_observed(model, x, mask, learning_rate=.01, steps=2, track_initial=True)


def test_meta_gradient_reaches_recipient_and_matches_finite_difference(model):
    x, mask, y, _ = example()
    weight = model.base.embedding.weight
    direction = torch.randn_like(weight)
    direction /= direction.norm()
    def risk(graph):
        fast, _ = update_observed(model, x, mask, learning_rate=.02, steps=2,
                                  create_graph=graph, track_initial=graph)
        logit = model.predict_fast(x, mask, fast)['logit']
        return torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    grad = torch.autograd.grad(risk(True), weight)[0]
    exact = (grad * direction).sum()
    old = weight.detach().clone()
    eps = .003
    with torch.no_grad():
        weight.copy_(old + eps * direction)
    plus = risk(False).detach()
    with torch.no_grad():
        weight.copy_(old - eps * direction)
    minus = risk(False).detach()
    with torch.no_grad():
        weight.copy_(old)
    assert grad.abs().sum() > 0
    torch.testing.assert_close(exact, (plus-minus)/(2*eps), rtol=.02, atol=2e-5)


@pytest.mark.parametrize('mode', ['joint', 'meta'])
def test_objective_trains_declared_paths_only(model, mode):
    loss, info = objective(model, *example(), mode=mode)
    loss.backward()
    for weight in (model.base.embedding.weight, model.base.risk_head.weight,
                   model.reconstructor.weight, model.event_head.weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    assert model.base.forecast_head.weight.grad is None
    assert model.base.fast_origin.grad is None
    expected = (info['risk_before'] if mode == 'joint'
                else .5 * (info['risk_before'] + info['risk_after']))
    expected += .2*info['completion'] + .1*info['event']
    assert float(loss.detach()) == pytest.approx(expected)


def test_unknown_event_is_not_a_normal_label(model):
    x, mask, y, _ = example()
    loss, info = objective(model, x, mask, y, torch.tensor([-1]), mode='meta')
    loss.backward()
    assert info['event'] == 0
    assert torch.equal(model.event_head.weight.grad, torch.zeros_like(model.event_head.weight))


def test_source_joint_rejects_cross_shot_inner_batch(model):
    x, mask, y, event = example()
    with pytest.raises(ValueError, match='one'):
        objective(model, x.repeat(2, 1, 1), mask.repeat(2, 1, 1),
                  y.repeat(2), event.repeat(2), mode='meta')


def test_sampling_probability_uses_interval_and_shot_uniform_draws():
    # Risk-positive 1/4; ordinary negative 1/2. Event draws 1/4,
    # with equal disruptive/safe probability, then equal interval probability.
    # Disruptive event means are 1/2 and 1, not pooled-by-length 2/3.
    shots = [dict(device='J-TEXT', y=np.array([0., 1., 1.])),
             dict(device='J-TEXT', y=np.array([0., 0.]))]
    events = {True: [(0, np.array([0, 1])), (0, np.array([2]))],
              False: [(1, np.array([0, 1]))]}
    prior = sampled_positive_probability(shots, events)
    assert prior == pytest.approx(.25 + .125 * .75)
    assert prior * (.5/prior) == pytest.approx(.5)
    assert (1-prior) * (.5/(1-prior)) == pytest.approx(.5)
    shots[0]['device'] = 'EAST'
    with pytest.raises(ValueError, match='TRAIN'):
        sampled_positive_probability(shots, events)


@pytest.mark.parametrize('mode', ['joint', 'meta'])
def test_risk_weight_changes_only_supervised_term(model, mode):
    data = example()
    normal, a = objective(model, *data, mode=mode)
    weighted, b = objective(model, *data, mode=mode, risk_weight=1.7)
    risk = a['risk_before'] if mode == 'joint' else .5*(a['risk_before']+a['risk_after'])
    assert float((weighted-normal).detach()) == pytest.approx(.7*risk, abs=1e-6)
    assert a['event'] == b['event'] and a['completion'] == b['completion']
    for invalid in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='risk weight'):
            objective(model, *data, mode=mode, risk_weight=invalid)
