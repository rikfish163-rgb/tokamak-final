import pytest
import torch

from experiments.risk_aligned_online.cross_density_constant import ConstantDensityModel
from experiments.risk_aligned_online.io23 import RAW23
from .test_cross_density import case, threads


def test_constant_control_has_no_cross_signal_information_but_same_risk():
    parent, model, x, a = case(); control = ConstantDensityModel(parent, -.2)
    fast = control.initial_fast(2).requires_grad_()
    original = model(x, a, fast); out = control(x, a, fast)
    torch.testing.assert_close(original['logit'],out['logit'],rtol=0,atol=0)
    density = RAW23.index('ne_nG')
    torch.testing.assert_close(out['forecast'][:,:,density],torch.full((2,25),-.2),rtol=0,atol=0)
    changed = control(x*5,a,fast)
    torch.testing.assert_close(changed['forecast'][:,:,density],out['forecast'][:,:,density],rtol=0,atol=0)
    assert control.temporary_dim == model.temporary_dim == 1
    assert all(not p.requires_grad for p in control.cross_head.parameters())


def test_constant_must_be_explicit_and_finite():
    parent, _, _, _ = case()
    with pytest.raises(ValueError): ConstantDensityModel(parent,float('nan'))
