import numpy as np
import torch

from experiments.risk_aligned_online.operational_transport_20261030.prepare import choose_origins, task_losses


def test_origins_have_complete_future_and_do_not_enter_late_supervision():
    times = np.arange(1000) / 1000
    ends = choose_origins(times, True, .8)
    assert len(ends) <= 36 and len(ends) == len(set(ends))
    assert np.all((ends-127) % 25 == 0)
    assert np.all(ends+50 < len(times))
    assert np.all(np.round((.8-times[ends+26])*1000, 9) > 30)


def test_early_true_alarm_is_rewarded_only_by_operational_target():
    z = torch.tensor([-.5, .8, .3], requires_grad=True)
    times = np.array([.10, .11, .12])
    a, b, index, label = task_losses(z, times, True, 1., .7)
    ga, = torch.autograd.grad(a, z, retain_graph=True)
    gb, = torch.autograd.grad(b, z)
    assert index == 1 and label == 0
    assert ga[1] < 0 < gb[1]
    assert ga[0] == ga[2] == gb[0] == gb[2] == 0


def test_safe_targets_identical_and_exact_thirty_excluded():
    z = torch.tensor([.2, 10.], requires_grad=True)
    a, b, index, label = task_losses(z, np.array([.169, .170]), True, .2, .7)
    assert index == 0 and label == 1 and torch.equal(a, b)
    a, b, index, label = task_losses(z, np.array([.169, .170]), False, None, .7)
    assert index == 1 and label == 0 and torch.equal(a, b)
