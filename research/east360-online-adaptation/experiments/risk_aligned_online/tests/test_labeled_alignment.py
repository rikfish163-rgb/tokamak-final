import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.labeled_alignment_20261023.learning import (
    moment_loss, conditional_moment_loss, choose_budget,
)


def test_conditional_alignment_detects_class_mean_exchange():
    source = torch.tensor([[-2.],[-1.],[1.],[2.]])
    target = -source
    labels = torch.tensor([0.,0.,1.,1.])
    assert moment_loss(source,target).item() == 0
    assert conditional_moment_loss(source,target,labels,labels).item() > 0


def test_zero_match_and_finite_encoder_gradient():
    s = torch.randn(8,4)
    assert moment_loss(s,s).item() == 0
    t = (s+1.).requires_grad_()
    loss = moment_loss(s,t)
    grad, = torch.autograd.grad(loss,t)
    assert torch.isfinite(grad).all() and grad.norm()>0


def test_conditional_requires_both_actual_classes():
    x = torch.ones(4,2)
    with pytest.raises(ValueError,match='at least two'):
        conditional_moment_loss(x,x,torch.zeros(4),torch.tensor([0.,0.,1.,1.]))


def test_nested_budget_uses_only_disjoint_agreeing_training_records():
    records = [dict(shot=i,is_disrupt=i<5,shotlist_is_disrupt=i<5) for i in range(1,25)]
    a = choose_budget(records,10,test_ids={99})
    b = choose_budget(records,20,test_ids={99})
    assert len(a)==10 and sum(r['is_disrupt'] for r in a)==2
    assert len(b)==20 and sum(r['is_disrupt'] for r in b)==4
    assert {r['shot'] for r in a} < {r['shot'] for r in b}
    with pytest.raises(ValueError,match='overlap'):
        choose_budget(records,20,test_ids={2})
    records[0]['shotlist_is_disrupt']=False
    with pytest.raises(ValueError,match='disagree'):
        choose_budget(records,20,test_ids={99})
