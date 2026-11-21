from copy import deepcopy

import torch
import numpy as np
import pytest

from experiments.risk_aligned_online.ended_channel import fit_weights
from experiments.risk_aligned_online.sequence_fast import sequence
from experiments.risk_aligned_online.sequence_train import pools, sample
from experiments.risk_aligned_online.tests.test_sequence_fast import fixture
from experiments.risk_aligned_online.forecast_replay import ForecastRetrainer
from experiments.risk_aligned_online.ended_channel import EndedChannelLearner


def test_ended_fit_changes_only_loss_weights():
    model, shot = fixture()
    before = deepcopy(model.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    logits = torch.nn.Parameter(torch.zeros(23))
    result = fit_weights(model, sequence(shot, 0, 2, model.cfg, 'cpu'), logits,
                         torch.optim.Adam([logits], lr=.01))
    assert result['gradient_norm'] > 0
    assert logits.abs().sum() > 0
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    assert [p.requires_grad for p in model.parameters()] == flags


def test_ended_sequence_sampling_uses_only_available_class():
    model, shot = fixture()
    pool = pools([shot], model.cfg, 2, require_both=False)
    seq, manifest = sample([shot], pool, model.cfg, np.random.default_rng(1), 2, 8, 'cpu')
    assert seq[-1]['y'].eq(0).all()
    assert all(sid == 1 for sid, _ in manifest)


def test_ended_meta_preserves_paired_network_and_rolls_back_failure(monkeypatch):
    from experiments.risk_aligned_online import ended_channel
    a, shot = fixture()
    shot.update(device='EAST', is_disrupt=False)
    b = deepcopy(a)
    base_a = ForecastRetrainer(a, supervised_steps=1, forecast_steps=1, batch_size=4, maintenance='aligned')
    base_b = ForecastRetrainer(b, supervised_steps=1, forecast_steps=1, batch_size=4, maintenance='aligned')
    weights = torch.zeros(23)
    learner = EndedChannelLearner(base_b, weights, steps=1, depth=2, batch_size=2)
    plain, adapted = base_a(a, shot), learner(b, shot)
    assert plain['sample_sha256'] == adapted['sample_sha256']
    assert all(torch.equal(v, a.state_dict()[k]) for k,v in b.state_dict().items())
    assert weights.abs().sum() > 0
    before, old_weights = deepcopy(b.state_dict()), weights.clone()
    next_shot = dict(shot, shot=2)
    def fail(*args):
        raise FloatingPointError('injected meta failure')
    monkeypatch.setattr(ended_channel, 'fit_weights', fail)
    with pytest.raises(FloatingPointError, match='injected'):
        learner(b, next_shot)
    assert all(torch.equal(v, before[k]) for k,v in b.state_dict().items())
    assert torch.equal(weights, old_weights)
    assert base_b.supervised.round == 1
    assert list(base_b.supervised.seen) == [1]
