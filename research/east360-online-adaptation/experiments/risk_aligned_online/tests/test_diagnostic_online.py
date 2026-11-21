from copy import deepcopy

import numpy as np
import torch

from experiments.risk_aligned_online.diagnostic_adapter import DiagnosticAdapter
from experiments.risk_aligned_online.diagnostic_online import predict_shot
from experiments.risk_aligned_online.diagnostic_training import train_step
from experiments.risk_aligned_online.simple23 import Config, Simple23


def objects():
    torch.manual_seed(131)
    parent = Simple23(Config(window=32, fast_site='tokens'), torch.zeros(23), torch.ones(23))
    parent.eval().requires_grad_(False)
    adapter = DiagnosticAdapter()
    rng = np.random.default_rng(131)
    values = rng.normal(size=(104, 23)).astype(np.float32)
    shot = dict(shot=1, values=values, available=np.ones_like(values, bool))
    return parent, adapter, shot


def test_arriving_signal_cannot_rewrite_past_or_persist_to_next_shot():
    parent, adapter, shot = objects()
    original, events = predict_shot(parent, adapter, shot, intra=True, device='cpu')
    changed = deepcopy(shot)
    changed['values'][52:] += 20
    changed_scores, _ = predict_shot(parent, adapter, changed, intra=True, device='cpu')
    np.testing.assert_array_equal(original[:52], changed_scores[:52])
    repeat, _ = predict_shot(parent, adapter, shot, intra=True, device='cpu')
    np.testing.assert_array_equal(original, repeat)
    assert events and all(event['affects_from'] > event['observed_through'] for event in events)
    assert not parent.encoder.layers[0]._forward_hooks


def test_risk_aligned_training_keeps_all_parent_weights_fixed():
    parent, adapter, _ = objects()
    frozen = deepcopy(parent.state_dict())
    x = torch.randn(2, 32, 23)
    truth = torch.randn(2, 25, 23)
    episode = dict(x=x, a=torch.ones_like(x, dtype=torch.bool), qx=x+1,
                   qa=torch.ones_like(x, dtype=torch.bool), truth=truth,
                   mask=torch.ones_like(truth, dtype=torch.bool), y=torch.tensor([0., 1.]))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    report = train_step(parent, adapter, optimizer, episode, mode='aligned')
    assert report['gradient_norm'] > 0
    assert all(torch.equal(value, frozen[key]) for key, value in parent.state_dict().items())
    assert all(p.grad is None for p in parent.parameters())
