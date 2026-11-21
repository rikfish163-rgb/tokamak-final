import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.lifecycle_stream_20261025.model import LifecycleAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23


def model(readout='shape'):
    torch.manual_seed(131)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    np.zeros(23), np.ones(23)).eval().requires_grad_(False)
    return LifecycleAdapter(base, readout=readout).eval()


def test_declared_readout_and_default_preserve_legacy_shape():
    m = model()
    assert m.readout == 'shape'
    legacy = LifecycleAdapter(m.base).eval()
    legacy.load_state_dict(m.state_dict())
    x = torch.randn(2, 128, 23)
    h, v = m.tokens(x, torch.ones_like(x, dtype=torch.bool))
    torch.nn.init.normal_(m.memory.output.weight, std=.1)
    legacy.load_state_dict(m.state_dict())
    d = torch.randn(2, 92)
    torch.testing.assert_close(m.risk(h, v, d), legacy.risk(h, v, d), atol=0, rtol=0)
    with pytest.raises(ValueError, match='readout'):
        model('unknown')


def test_full_readout_retains_common_horizon_component_without_training_base():
    m = model('full')
    shape = model('shape')
    torch.nn.init.normal_(m.memory.output.weight, std=.1)
    shape.load_state_dict(m.state_dict())
    x = torch.randn(2, 128, 23)
    h, v = m.tokens(x, torch.ones_like(x, dtype=torch.bool))
    d = torch.randn(2, 1, 23).expand(2, 4, 23).reshape(2, 92)
    zero = torch.zeros_like(d)
    torch.testing.assert_close(shape.risk(h, v, d), shape.risk(h, v, zero), atol=0, rtol=0)
    z = m.risk(h, v, d)
    assert not torch.equal(z, m.risk(h, v, zero))
    z.sum().backward()
    assert m.memory.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.base.parameters())


@pytest.mark.parametrize('readout', ['full', 'shape', 'level'])
def test_readout_does_not_change_causal_memory_or_shot_reset(readout):
    m = model(readout)
    signal = dict(values=np.random.default_rng(1).normal(size=(160, 23)).astype('float32'),
                  available=np.ones((160, 23), bool))
    d, info = m.memory_delta(signal)
    changed = dict(values=signal['values'].copy(), available=signal['available'].copy())
    changed['values'][80:] += 100
    dd, _ = m.memory_delta(changed)
    torch.testing.assert_close(d[:80], dd[:80], atol=0, rtol=0)
    d2, _ = m.memory_delta(signal)
    torch.testing.assert_close(d, d2, atol=0, rtol=0)
    assert not d[:40].any() and d[40:].abs().max() > 0
    assert info['readout'] == readout
    energy = info['readout_energy']
    assert energy['full'] == pytest.approx(energy['shape'] + energy['level'], rel=1e-6)
    assert energy['level'] > 0
