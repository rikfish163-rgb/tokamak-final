import numpy as np
import torch
from copy import deepcopy

from experiments.risk_aligned_online.missing_training import drop_channels, train_step
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_whole_channels_only_and_original_preserved():
    x = torch.ones(32, 16, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, 0] = False
    xx, mm = drop_channels(x, mask, np.random.default_rng(13), max_drop=4)
    assert x.eq(1).all() and mask[:, :, 1:].all()
    assert not mm[:, :, 0].any()
    assert (mm == mm[:, :1]).all()
    assert (~mm[:, 0]).sum(-1).max() <= 5
    assert (mm <= mask).all()
    assert xx[~mm].eq(0).all()
    a, b = drop_channels(x, mask, np.random.default_rng(13), max_drop=4)
    assert torch.equal(xx, a) and torch.equal(mm, b)


def test_training_preserves_forecast_head_buffers_and_origin():
    torch.manual_seed(3)
    model = Simple23(Config(window=16, patch=8, dim=8, heads=2, layers=1),
                     torch.zeros(23), torch.ones(23))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 16, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    y = torch.tensor([0., 1., 0., 1.])
    flags = [p.requires_grad for p in model.parameters()]
    log = train_step(model, opt, x, mask, y, np.random.default_rng(5), augmented=True)
    assert np.isfinite(log['loss'])
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
    for k in ('center', 'scale', 'fast_origin', 'forecast_head.weight', 'forecast_head.bias'):
        assert torch.equal(model.state_dict()[k], before[k])
    assert flags == [p.requires_grad for p in model.parameters()]


def test_teacher_is_fixed_and_protection_penalizes_clean_drift():
    torch.manual_seed(9)
    model = Simple23(Config(window=16, patch=8, dim=8, heads=2, layers=1),
                     torch.zeros(23), torch.ones(23))
    teacher = deepcopy(model)
    initial = {k: v.clone() for k, v in teacher.state_dict().items()}
    with torch.no_grad():
        model.risk_head.bias.add_(1.)
    x = torch.randn(4, 16, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    log = train_step(model, torch.optim.SGD(model.parameters(), lr=.001),
                     x, mask, torch.tensor([0., 1., 0., 1.]), np.random.default_rng(1),
                     augmented=True, teacher=teacher, teacher_weight=1.)
    assert abs(log['protection']-.5) < 1e-5
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v, initial[k]) for k, v in teacher.state_dict().items())
