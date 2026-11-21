from copy import deepcopy

import torch

from experiments.risk_aligned_online.block_direction import measure
from experiments.risk_aligned_online.fast_direction import episode
from experiments.risk_aligned_online.tests.test_sequence_fast import fixture


def test_functional_update_is_temporary_and_label_blind():
    model, shot = fixture()
    state = deepcopy(model.state_dict())
    e = episode(shot, 0, model.cfg, 'cpu')
    first = measure(model, e, step=.001)
    e['y'] = 1-e['y']
    second = measure(model, e, step=.001)
    assert first['signal_probability'] == second['signal_probability']
    assert first['oracle_probability'] != second['oracle_probability']
    assert first['parameter_count'] > model.fast_origin.numel()
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_zero_step_leaves_risk_unchanged():
    model, shot = fixture()
    result = measure(model, episode(shot, 0, model.cfg, 'cpu'), step=0.)
    assert abs(result['signal_improvement']) < 1e-7
    assert abs(result['oracle_improvement']) < 1e-7
