import numpy as np
import torch

from experiments.risk_aligned_online.temporal_relation_20261029.model import (
    OrderJudge,permutation,paired_views,margin,roughness,mean_view,
)
from experiments.risk_aligned_online.simple23 import Config,Simple23


def setup():
    torch.set_num_threads(1);torch.manual_seed(73);torch.backends.mha.set_fastpath_enabled(False)
    return Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)


def test_permutation_preserves_joint_values_and_endpoints():
    x=torch.arange(2*128*23,dtype=torch.float32).reshape(2,128,23);a=torch.ones_like(x,dtype=torch.bool)
    a[:,:,2]=False;p=permutation(2,np.random.default_rng(71))
    xx,aa,eligible=paired_views(x,a,p)
    assert eligible.all() and torch.equal(aa,a)
    assert torch.equal(xx[:,:8],x[:,:8]) and torch.equal(xx[:,-8:],x[:,-8:])
    for i in range(2):
        assert torch.equal(torch.sort(xx[i],dim=0).values,torch.sort(x[i],dim=0).values)
        assert {tuple(r.tolist()) for r in xx[i]}=={tuple(r.tolist()) for r in x[i]}


def test_time_varying_masks_and_constant_signals_do_not_make_fake_pairs():
    x=torch.randn(3,128,23);a=torch.ones_like(x,dtype=torch.bool);a[0,0,0]=False;x[1]=1
    xx,aa,ok=paired_views(x,a,permutation(3,np.random.default_rng(2)))
    assert ok.tolist()==[False,False,True]
    assert torch.equal(xx[0],x[0]) and torch.equal(aa[0],a[0])


def test_mean_only_is_identical_for_permuted_pair_and_does_not_supply_order_labels():
    base=setup();net=OrderJudge().eval();x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool)
    mean=mean_view(x,a);xx,aa,_=paired_views(mean,a,permutation(2,np.random.default_rng(7)))
    assert torch.equal(xx,mean)
    delta,_,_=margin(base,net,mean,a,permutation(2,np.random.default_rng(7)),torch.zeros(128))
    assert torch.equal(delta,torch.zeros_like(delta))


def test_auxiliary_head_gets_gradient_but_frozen_classifier_does_not():
    base=setup();net=OrderJudge();x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool)
    d,valid,_=margin(base,net,x,a,permutation(2,np.random.default_rng(3)),torch.zeros(128))
    torch.nn.functional.softplus(-d[valid]).mean().backward()
    assert sum(p.numel() for p in net.parameters())==4161
    assert any(p.grad is not None and p.grad.norm()>0 for p in net.parameters())
    assert all(p.grad is None for p in base.parameters())


def test_auxiliary_loss_can_reach_temporary_internal_state():
    base=setup();net=OrderJudge().eval().requires_grad_(False)
    x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool);s=torch.zeros(128,requires_grad=True)
    d,valid,tv=margin(base,net,x,a,permutation(1,np.random.default_rng(4)),s)
    g,=torch.autograd.grad(torch.nn.functional.softplus(-d[valid]).mean(),s)
    assert torch.isfinite(g).all() and g.norm()>0 and torch.isfinite(tv).all()
    h=torch.ones(2,17,64)
    assert torch.equal(roughness(h),torch.zeros(2))
