from copy import deepcopy

import numpy as np
import torch

from experiments.risk_aligned_online.fast_direction import episode
from experiments.risk_aligned_online.fast_meta import temporary_step, train_step
from experiments.risk_aligned_online.simple23 import Config, Simple23


def fixture():
    torch.manual_seed(31)
    cfg = Config(window=8, patch=4, dim=8, heads=2, layers=1, horizon=3)
    model = Simple23(cfg, np.zeros(23), np.ones(23))
    shot = dict(values=np.random.default_rng(31).normal(size=(20, 23)).astype('float32'),
                available=np.ones((20, 23), bool), valid=np.ones(20, bool), y=np.ones(20))
    return model, episode(shot, 5, cfg, 'cpu')


def test_batch_replication_does_not_dilute_fast_update():
    model, e = fixture()
    f1, _ = temporary_step(model, e, create_graph=False)
    e2 = {k: torch.cat([v, v], 0) if isinstance(v, torch.Tensor) else v for k, v in e.items()}
    f2, _ = temporary_step(model, e2, create_graph=False)
    torch.testing.assert_close(f1, f2[:1])


def test_meta_changes_head_only_and_has_second_order_signal():
    model, e = fixture()
    model.eval()
    before = deepcopy(model.state_dict())
    with torch.no_grad():
        old = model(e['qx'], e['qa'])['logit'].clone()
    optimizer = torch.optim.AdamW(model.forecast_head.parameters(), lr=1e-3, weight_decay=0.)
    result = train_step(model, optimizer, e, mode='meta')
    assert result['head_grad_norm'] > 0
    assert all(np.isfinite(v) for v in result.values())
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all(k.startswith('forecast_head.') for k in changed)
    with torch.no_grad():
        torch.testing.assert_close(old, model(e['qx'], e['qa'])['logit'], rtol=0, atol=0)
    assert all(p.requires_grad for p in model.parameters())


def test_inner_step_ignores_outer_risk_label():
    model, e = fixture()
    f1, _ = temporary_step(model, e, create_graph=False)
    e['y'] = 1-e['y']
    f2, _ = temporary_step(model, e, create_graph=False)
    torch.testing.assert_close(f1, f2, rtol=0, atol=0)
