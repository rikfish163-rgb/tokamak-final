from dataclasses import replace

import torch

from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.fast_meta import temporary_step
from experiments.risk_aligned_online.tests.test_fast_meta import fixture


def test_direction_normalization_invariant_to_positive_loss_rescaling():
    initial=torch.zeros(2,4)
    g=torch.tensor([[.001,.002,-.003,.004],[1.,2.,3.,4.]])
    a=normalized_step(initial,initial,g)
    b=normalized_step(initial,initial,g*100.)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(a.norm(dim=-1),torch.ones(2)*.05)


def test_zero_gradient_and_cumulative_radius_are_safe():
    initial=torch.zeros(1,4)
    fast=initial.clone()
    torch.testing.assert_close(normalized_step(fast,initial,fast),fast)
    for _ in range(100):
        fast=normalized_step(fast,initial,torch.ones_like(fast))
    assert float(fast.norm())<=1.000001


def test_normalized_meta_step_matches_deployment_rule_and_batch_independence():
    net,e=fixture()
    net.cfg=replace(net.cfg,fast_site='tokens',fast_rule='normalized')
    net.eval()
    fast=net.initial_fast(1).detach().requires_grad_(True)
    loss=net.forecast_loss(net(e['x'],e['a'],fast)['forecast'],e['truth'],e['mask'])
    g,=torch.autograd.grad(loss,fast)
    expected=normalized_step(fast,fast.detach(),g)
    actual,_=temporary_step(net,e,create_graph=True)
    torch.testing.assert_close(actual,expected)
    double={k:torch.cat([v,v]) if isinstance(v,torch.Tensor) else v for k,v in e.items()}
    batched,_=temporary_step(net,double,create_graph=True)
    torch.testing.assert_close(actual,batched[:1])


def test_normalized_update_has_finite_second_derivatives():
    g=torch.tensor([[.01,.02,.03,.04]],requires_grad=True)
    origin=torch.zeros_like(g)
    updated=normalized_step(origin,origin,g)
    first,=torch.autograd.grad(updated[0,0],g,create_graph=True)
    second,=torch.autograd.grad(first.sum(),g)
    assert torch.isfinite(second).all()
