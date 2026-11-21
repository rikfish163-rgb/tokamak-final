import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.fast_direction import episode, measure
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_episode_uses_only_matured_target_before_query():
    cfg = Config(window=8, patch=4, dim=8, heads=2, layers=1, horizon=3)
    shot = dict(values=np.repeat(np.arange(20, dtype=np.float32)[:, None], 23, axis=1),
                available=np.ones((20, 23), bool), valid=np.ones(20, bool), y=np.zeros(20))
    e = episode(shot, 5, cfg, 'cpu')
    assert e['query_end'] == 9
    assert e['truth'][0, :, 0].tolist() == [6, 7, 8]
    assert e['x'][0, -1, 0] == 5
    assert e['qx'][0, -1, 0] == 9
    with pytest.raises(ValueError):
        episode(shot, 17, cfg, 'cpu')


def test_measure_does_not_update_persistent_weights():
    torch.manual_seed(1)
    cfg = Config(window=8, patch=4, dim=8, heads=2, layers=1, horizon=3)
    model = Simple23(cfg, np.zeros(23), np.ones(23))
    rng = np.random.default_rng(0)
    shot = dict(values=rng.normal(size=(20, 23)).astype('float32'),
                available=np.ones((20, 23), bool), valid=np.ones(20, bool), y=np.ones(20))
    before = {k: p.clone() for k, p in model.state_dict().items()}
    result = measure(model, episode(shot, 5, cfg, 'cpu'))
    assert all(np.isfinite(v) for v in result.values())
    assert -1.000001 <= result['cosine'] <= 1.000001
    assert all(torch.equal(p, before[k]) for k, p in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
