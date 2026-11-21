from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.risk_aligned_online import commissioning_cold_embedding_prior_20261114 as run


def test_positive_bag_is_only_the_existing_legal_warning_window():
    ticks = np.arange(360)*.001
    scores = np.linspace(-1.,1.,len(ticks))
    idx = run.bag_indices(ticks,scores,True,.300)
    expected = np.arange(175,270)
    np.testing.assert_array_equal(idx,expected)
    neg = run.bag_indices(ticks,scores,False,None)
    assert neg[-1]==len(ticks)-1
    assert (np.diff(neg)>=run.MIN_SEPARATION).all()


def test_supervised_prior_span_only_uses_supplied_gradient_rows():
    a = torch.zeros((6,4,4),dtype=torch.float32)
    a[0,0,0],a[1,0,1],a[2,1,0],a[3,1,1] = 1.,2.,3.,4.
    a[4:] = a[:2]
    p,s = run.gradient_span(a)
    assert p.shape == (4,4,4)
    torch.testing.assert_close((p.flatten(1)@p.flatten(1).T),torch.eye(4))
    assert not p[:,2:].any() and len(s)==6


def test_value_hook_affects_only_numeric_lanes_and_removes_its_effect_on_exit():
    module = nn.Linear(46*8,3,bias=False)
    with torch.no_grad():
        module.weight.zero_()
    base = SimpleNamespace(embedding=module)
    d = torch.ones((3,8*len(run.VALUES)))
    p = torch.zeros((1,2,8,46))
    p[...,run.VALUES[0]] = 2.
    x = p.flatten(2)
    with run.value_prior(base,d):
        assert not module(x).any()
        p[...,23+run.VALUES[0]] = 1.
        torch.testing.assert_close(module(x), torch.full((1,2,3),16.))
    assert not module(x).any()


def test_reference_loss_does_not_silence_confident_class_and_keeps_the_alarm_floor():
    raw = torch.tensor([5.,-2.,-5.,3.])
    positive = torch.tensor([True,True,False,False])
    mark = .1
    target = run.reference_targets(raw,positive,mark)
    torch.testing.assert_close(target,torch.tensor([5.,.1,-5.,.1]))
    change = raw.detach().requires_grad_(True)
    a,b = run.margin_loss(change,positive,target)
    grad, = torch.autograd.grad(a+b,change)
    assert (grad[positive]<-.9).all()
    assert (grad[~positive]>.9).all()
