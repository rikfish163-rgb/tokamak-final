from copy import deepcopy

import numpy as np
import torch

from experiments.risk_aligned_online.sequence_fast import sequence
from experiments.risk_aligned_online.sequence_train import pools, sample, train_step
from experiments.risk_aligned_online.tests.test_sequence_fast import fixture


def test_sequence_pools_and_sampling_preserve_temporal_order():
    model, shot = fixture()
    shot['y'][60:] = 1
    p = pools([shot], model.cfg, 2)
    batch, manifest = sample([shot], p, model.cfg, np.random.default_rng(1), 2, 3, 'cpu')
    assert len(batch) == 2 and len(manifest) == 3
    for i, (sid, start) in enumerate(manifest):
        assert sid == 1
        expected = sequence(shot, start, 2, model.cfg, 'cpu')
        for k in range(2):
            torch.testing.assert_close(batch[k]['truth'][i], expected[k]['truth'][0])


def test_train_updates_forecast_and_encoder_but_not_origin_or_scaler():
    model, shot = fixture()
    before = deepcopy(model.state_dict())
    seq = sequence(shot, 0, 3, model.cfg, 'cpu')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
    log = train_step(model, opt, seq, mode='continuous')
    assert all(np.isfinite(v) for v in log.values())
    assert log['head_grad_norm'] > 0 and log['encoder_grad_norm'] > 0
    for key in ('center', 'scale', 'fast_origin'):
        assert torch.equal(before[key], model.state_dict()[key])
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items()
               if k.startswith('forecast_head.'))


def test_one_step_reset_and_continuous_identical():
    model, shot = fixture()
    other = deepcopy(model)
    seq = sequence(shot, 0, 1, model.cfg, 'cpu')
    for m, mode in ((model, 'continuous'), (other, 'reset')):
        train_step(m, torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.), seq, mode=mode)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)


def test_final_objective_ignores_earlier_risk_labels_but_uses_final_label():
    model, shot = fixture()
    seq = sequence(shot, 0, 3, model.cfg, 'cpu')
    altered = deepcopy(seq)
    for e in altered[:-1]:
        e['y'] = 1-e['y']
    outcomes = []
    for episodes in (seq, altered):
        m = deepcopy(model)
        log = train_step(m, torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.),
                         episodes, mode='continuous', risk_scope='final')
        outcomes.append((log, deepcopy(m.state_dict())))
    assert outcomes[0][0] == outcomes[1][0]
    for key, value in outcomes[0][1].items():
        torch.testing.assert_close(value, outcomes[1][1][key], rtol=0, atol=0)
    altered[-1]['y'] = 1-altered[-1]['y']
    log = train_step(model, torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.),
                     altered, mode='continuous', risk_scope='final')
    assert log['risk_after'] != outcomes[0][0]['risk_after']
