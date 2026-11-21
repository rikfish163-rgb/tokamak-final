"""Characterize an existing difference-only feedback blind spot, not a fix."""
import numpy as np
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter, ContextMemory
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.model import patch_signal
from experiments.risk_aligned_online.east_context_operator_20261031.learning import (
    configure_operator_training, objective,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_any_constant_level_produces_zero_state_despite_scheduled_writes():
    torch.manual_seed(131)
    memory = ContextMemory(64)
    for level in (0., 1.5, 8.):
        means = torch.full((32, 23), level)
        complete = torch.ones_like(means, dtype=torch.bool)
        complete[:, 2] = False
        trace = memory.trace(means, complete, mode='memory')
        assert trace['updates'] == 31
        assert not trace['context'].count_nonzero()
        assert not trace['delta'].count_nonzero()


def test_arrived_change_can_activate_the_same_memory():
    memory = ContextMemory(64)
    with torch.no_grad():
        memory.key.weight.zero_()
        memory.key.bias.zero_()
        memory.kernel.fill_(.1)
    means = torch.zeros(32, 23)
    means[16:] = 1.
    trace = memory.trace(means, torch.ones_like(means, dtype=torch.bool), mode='memory')
    assert not trace['context'][:16].count_nonzero()
    assert trace['context'][16:].count_nonzero() > 0


def test_risk_supervision_cannot_train_operator_on_a_constant_trajectory():
    torch.set_num_threads(1)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    model = ContextAdapter(base).eval()
    with torch.no_grad():
        model.memory.output.weight.normal_(0, .1)  # exclude trivial zero-output explanation
    selected = configure_operator_training(model)
    signal = dict(values=np.full((128, 23), 1.5, np.float32), available=np.ones((128, 23), bool))
    with torch.no_grad():
        h, valid = model.tokens(torch.tensor(signal['values'])[None],
                                torch.tensor(signal['available'])[None])
        means, complete = patch_signal(signal, model.base.center, model.base.scale)
    loss, detail = objective(model, dict(h=h, valid=valid, means=means, complete=complete,
                                        ends=[127], flag=True, y=torch.ones(1)))
    assert loss > 0 and detail['writes'] == 15
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.count_nonzero()
               for p in selected)
