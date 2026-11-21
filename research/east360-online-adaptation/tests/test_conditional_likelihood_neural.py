import numpy as np
import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.conditional_likelihood_20261029.neural import (
    finite_step, proposal,
)


def test_proposal_preserves_confidence_and_zero_evidence():
    z = torch.zeros(5, dtype=torch.float64)
    g = torch.ones(5, 8192, dtype=torch.float64)
    logs = torch.tensor([[0., 4.], [4., 0.], [0., 0.], [0., 1e-7], [0., 4.]])
    state, feedback = proposal(z, g, logs, torch.tensor([1, 1, 1, 1, 0]))
    assert state[0, 0] > 0 and state[1, 0] < 0
    assert not state[2].any() and not state[4].any()
    assert state.norm(dim=1).max() <= .05
    assert 0 < state[3].norm() < 1e-8
    torch.testing.assert_close(state.norm(dim=1), .05*feedback.abs())


def test_zero_gradient_and_nonfinite_contract():
    z = torch.zeros(1)
    g = torch.zeros(1, 8192)
    state, _ = proposal(z, g, torch.tensor([[0., 2.]]), torch.ones(1))
    assert not state.any()
    with pytest.raises(ValueError):
        proposal(z, g, torch.tensor([[0., float('nan')]]), torch.ones(1))
    with pytest.raises(ValueError):
        proposal(z, g, torch.tensor([[0., 2.]]), torch.tensor([-1]))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.register_buffer('center', torch.zeros(23))
        self.base.register_buffer('scale', torch.ones(23))

    def forward(self, x, available, state, slow, mode, *, forecast=False):
        return {'logit': x[:, -1, 0] + state[:, 0]*(1+x[:, -1, 1])}


def signal():
    v = np.zeros((230, 23), np.float32)
    v[:, 0] = np.arange(230)/1000
    return dict(values=v, available=np.ones_like(v, bool))


def test_actual_step_future_cannot_change_update_and_identical_is_exact():
    net = Toy()
    s = signal()
    logs = {'real': np.array([[0., 2.]]), 'identical': np.zeros((1, 2))}
    counts = {k: np.ones(1, int) for k in logs}
    before = finite_step(net, s, np.array([127]), None, logs, counts)
    s['values'][153:] += .1
    after = finite_step(net, s, np.array([127]), None, logs, counts)
    np.testing.assert_array_equal(before['real_state_norm'], after['real_state_norm'])
    np.testing.assert_array_equal(before['real_evidence_gradient'], after['real_evidence_gradient'])
    assert not np.array_equal(before['real_logits'], after['real_logits'])
    np.testing.assert_array_equal(before['identical_logits'], before['baseline_logits'])
    assert before['baseline_logits'].shape == (1, 25)
    assert before['real_changed'][0]
    assert np.isclose(before['baseline_logits'][0, 0], .153)


def test_no_label_fields_or_incomplete_next_window():
    net = Toy()
    s = signal()
    logs = {'real': np.zeros((1, 2))}
    counts = {'real': np.ones(1, int)}
    with pytest.raises(ValueError):
        finite_step(net, dict(s, y=1), np.array([127]), None, logs, counts)
    with pytest.raises(ValueError):
        finite_step(net, s, np.array([180]), None, logs, counts)
