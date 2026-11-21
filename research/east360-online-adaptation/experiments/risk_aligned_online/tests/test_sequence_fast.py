from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.sequence_fast import sequence, unroll
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.simple_online import predict_shot


def fixture():
    torch.manual_seed(7)
    cfg = Config(window=8, patch=4, dim=8, heads=2, layers=1,
                 horizon=25, fast_site='tokens', fast_rule='normalized')
    model = Simple23(cfg, np.zeros(23), np.ones(23)).eval()
    s = dict(values=np.random.default_rng(7).normal(size=(80, 23)).astype('float32'),
             available=np.ones((80, 23), bool), valid=np.ones(80, bool),
             y=np.zeros(80, 'float32'), shot=1)
    return model, s


def test_unroll_matches_continuous_online_and_does_not_mutate_model():
    model, s = fixture()
    before = deepcopy(model.state_dict())
    seq = sequence(s, 0, 3, model.cfg, 'cpu')
    result = unroll(model, seq, create_graph=False)
    scores, _ = predict_shot(model, s, intra=True, device='cpu')
    np.testing.assert_allclose(result['logits'].detach().sigmoid().numpy()[:, 0],
                               scores[[26, 52, 78]], atol=1e-6)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_inner_sequence_does_not_use_risk_labels():
    model, s = fixture()
    seq = sequence(s, 0, 3, model.cfg, 'cpu')
    a = unroll(model, seq, create_graph=False)
    for e in seq:
        e['y'] = 1-e['y']
    b = unroll(model, seq, create_graph=False)
    torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)


def test_outer_risk_backpropagates_through_all_inner_steps():
    model, s = fixture()
    result = unroll(model, sequence(s, 0, 3, model.cfg, 'cpu'), create_graph=True)
    loss = torch.nn.functional.softplus(result['logits']).mean()
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.forecast_head.parameters()
               if p.grad is not None) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_sequence_rejects_unobserved_tail_and_invalid_interval():
    model, s = fixture()
    with pytest.raises(ValueError):
        sequence(s, 3, 3, model.cfg, 'cpu')
    s['valid'][30] = False
    with pytest.raises(ValueError):
        sequence(s, 0, 3, model.cfg, 'cpu')
