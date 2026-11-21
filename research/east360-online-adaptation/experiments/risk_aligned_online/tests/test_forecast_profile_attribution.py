import torch

from experiments.risk_aligned_online.forecast_profile_20261025.learning import decompose
from experiments.risk_aligned_online.forecast_profile_20261025.diagnose import expand_shape


def test_shape_expansion_matches_autograd_with_masked_nan_targets():
    torch.manual_seed(15)
    p=torch.randn(4,25,23,dtype=torch.float64,requires_grad=True)
    y=torch.randn_like(p);m=torch.rand_like(p)>.2;y[~m]=float('nan')
    terms=expand_shape(p,y,m);shape=decompose(p,y,m)['shape']
    torch.testing.assert_close(sum(terms.values()),shape,atol=1e-14,rtol=1e-12)
    g=torch.autograd.grad(shape,p)[0]
    a=torch.autograd.grad(terms['prediction_energy'],p,retain_graph=True)[0]
    b=torch.autograd.grad(terms['observed_cross'],p)[0]
    torch.testing.assert_close(g,a+b,atol=1e-14,rtol=1e-12)


def test_constant_target_has_no_observed_shape_gradient():
    p=torch.randn(4,25,23,dtype=torch.float64,requires_grad=True);m=torch.ones_like(p,dtype=torch.bool)
    y=torch.ones_like(p)*3;terms=expand_shape(p,y,m)
    assert terms['target_energy']==0 and terms['observed_cross']==0
    g=torch.autograd.grad(terms['observed_cross'],p)[0]
    assert not g.any()
