import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.prefix_coverage_20261023.learning import (
    prefix_ends,prefix_risk,paired_objective,
)
from experiments.risk_aligned_online.tests.test_increment_forecast import model


def test_prefix_preserves_sample_count_and_real_start_boundary():
    assert prefix_ends(400,5)==[0,63,127,191,255]
    assert len(prefix_ends(400,128))==128
    assert prefix_ends(400,1)==[0]
    for n in (5,7,64,128):
        result=prefix_ends(400,n);assert len(result)==len(set(result))==n
        assert min(result)==0 and max(result)==255
    with pytest.raises(ValueError):prefix_ends(255,5)
    with pytest.raises(ValueError):prefix_ends(400,129)


def test_only_safe_before_loss_is_replaced():
    before=torch.tensor(2.,requires_grad=True);after=torch.tensor(4.,requires_grad=True)
    forecast=torch.tensor(3.,requires_grad=True);early=torch.tensor(1.,requires_grad=True)
    r=dict(before=before,after=after,forecast=forecast)
    assert float(paired_objective(r,early,disruptive=False).detach())==pytest.approx(3.1)
    assert float(paired_objective(r,early,disruptive=True).detach())==pytest.approx(3.6)
    g=torch.autograd.grad(paired_objective(r,early,disruptive=False),(before,after,forecast,early),allow_unused=True)
    assert g[0] is None
    assert [float(x) for x in g[1:]]==pytest.approx([.5,.2,.5])


def test_prefix_loss_and_gradient_ignore_unobserved_suffix():
    _,m=model('increment')
    for k,p in m.named_parameters():p.requires_grad_(k in m.encoder_names())
    s=dict(values=np.random.default_rng(4).normal(size=(500,23)).astype('float32'),available=np.ones((500,23),bool))
    loss,ends=prefix_risk(m,s,16,boundary=1.,device='cpu')
    grads=torch.autograd.grad(loss,tuple(p for p in m.parameters() if p.requires_grad))
    s['values'][256:]=999.
    again,ends2=prefix_risk(m,s,16,boundary=1.,device='cpu')
    other=torch.autograd.grad(again,tuple(p for p in m.parameters() if p.requires_grad))
    torch.testing.assert_close(loss,again,rtol=0,atol=0);assert ends==ends2
    for a,b in zip(grads,other):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError):prefix_risk(m,dict(s,y=0),16,boundary=1.,device='cpu')
