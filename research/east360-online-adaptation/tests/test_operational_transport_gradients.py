"""Actual attention backward path: observed-target timing and channel masks."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
from experiments.risk_aligned_online.operational_transport_20261030.gradients import observed_gradients
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture
def example():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23)))
    rng = np.random.default_rng(42)
    signal = dict(values=rng.normal(size=(200, 23)).astype(np.float32),
                  available=np.ones((200, 23), bool))
    return net, signal, torch.zeros(64, 64)


def test_no_later_signal_or_outcome_used_in_actual_gradient(example):
    net, signal, slow = example
    state = {k: v.clone() for k, v in net.state_dict().items()}
    first = observed_gradients(net, signal, 127, slow)
    altered = {k: v.copy() for k, v in signal.items()}
    altered['values'][153:] = 1e5
    altered['available'][153:] = False
    second = observed_gradients(net, altered, 127, slow)
    shortened = observed_gradients(net, {k: v[:153] for k, v in signal.items()}, 127, slow)
    for i in range(3):
        assert torch.equal(first[i], second[i])
        assert torch.equal(first[i], shortened[i])
        assert first[i].shape == (1, 8192) and torch.isfinite(first[i]).all()
    assert first[3] == second[3] == shortened[3]
    assert first[3]['observed'] == 25 * 22
    assert first[0].norm() > 0 and first[2].norm() > 0
    assert all(torch.equal(v, state[k]) for k, v in net.state_dict().items())
    assert all(p.grad is None for p in net.parameters())
    with pytest.raises(ValueError, match='signal only'):
        observed_gradients(net, dict(signal, is_disrupt=True), 127, slow)


def test_future_mode_and_unobserved_channels_cannot_drive_updates(example):
    net, signal, slow = example
    first = observed_gradients(net, signal, 127, slow)
    signal['values'][128:153, 10] = 1e7
    second = observed_gradients(net, signal, 127, slow)
    assert all(torch.equal(first[i], second[i]) for i in range(3))
    signal['available'][128:153] = False
    signal['values'][128:153] = np.nan
    g, reverse, risk, info = observed_gradients(net, signal, 127, slow)
    assert torch.count_nonzero(g) == torch.count_nonzero(reverse) == 0
    assert torch.equal(risk, first[2])
    assert info['observed'] == info['signal_loss'] == info['reverse_loss'] == 0


def test_reverse_target_changes_signal_gradient_not_saved_risk(example):
    net, signal, slow = example
    g, reverse, risk, _ = observed_gradients(net, signal, 127, slow)
    flipped = {k: v.copy() for k, v in signal.items()}
    for key in flipped:
        flipped[key][128:153] = flipped[key][128:153][::-1]
    g2, reverse2, risk2, _ = observed_gradients(net, flipped, 127, slow)
    assert not torch.equal(g, reverse)
    assert torch.equal(g, reverse2) and torch.equal(reverse, g2)
    assert torch.equal(risk, risk2)


@pytest.mark.parametrize('origin,length', [(126, 200), (127, 152), (177, 200)])
def test_reject_incomplete_feedback(example, origin, length):
    net, signal, slow = example
    with pytest.raises(ValueError, match='complete past'):
        observed_gradients(net, {k: v[:length] for k, v in signal.items()}, origin, slow)
