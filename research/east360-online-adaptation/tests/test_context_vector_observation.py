import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
from experiments.risk_aligned_online.context_vector_20261030.observation import observation
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    torch.manual_seed(131)
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23)))
    signal = dict(values=np.ones((180, 23), np.float32), available=np.ones((180, 23), bool))
    signal['values'][128:153] = np.arange(1, 26)[:, None]
    return net, signal


def test_observation_uses_only_matured_change_and_detaches_context(sample):
    net, signal = sample
    original = {k: v.clone() for k, v in net.state_dict().items()}
    a = observation(net, signal, 127)
    signal['values'][153:] = 1e9
    b = observation(net, signal, 127)
    for key in a:
        assert torch.equal(a[key], b[key])
    assert a['context'].shape == (1, 358) and a['support'].item()
    assert not a['context'].requires_grad
    expected = np.mean(np.arcsinh(np.arange(1, 6))-np.arcsinh(1.))
    assert a['context'][0, 128].item() == pytest.approx(expected, abs=1e-6)
    assert a['context'][0, 128+10].item() == 0
    assert all(torch.equal(v, original[k]) for k, v in net.state_dict().items())


def test_reverse_changes_context_but_not_origin_tokens(sample):
    net, signal = sample
    a = observation(net, signal, 127)
    b = observation(net, signal, 127, reverse=True)
    assert torch.equal(a['context'][:, :128], b['context'][:, :128])
    assert not torch.equal(a['context'][:, 128:243], b['context'][:, 128:243])


def test_missing_future_is_not_a_zero_measurement(sample):
    net, signal = sample
    signal['available'][128:153] = False
    signal['values'][128:153] = np.nan
    a = observation(net, signal, 127)
    assert not a['support'].item()
    assert not torch.count_nonzero(a['context'][:, 128:])
    assert torch.isfinite(a['context']).all()


def test_missing_origin_channel_blocks_change_feedback(sample):
    net, signal = sample
    signal['available'][127, 2] = False
    a = observation(net, signal, 127)
    assert not torch.count_nonzero(a['context'][:, 128:243].reshape(1, 5, 23)[:, :, 2])
    assert not torch.count_nonzero(a['context'][:, 243:].reshape(1, 5, 23)[:, :, 2])


def test_incomplete_or_labeled_observation_rejected(sample):
    net, signal = sample
    with pytest.raises(ValueError):
        observation(net, dict(signal, label=1), 127)
    with pytest.raises(ValueError):
        observation(net, {k: v[:152] for k, v in signal.items()}, 127)
