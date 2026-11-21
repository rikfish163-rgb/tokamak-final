import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_observation_inverse_20261114 as trial
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_later_forecast_is_disjoint_and_issued_before_its_truth():
    q = np.array([127,159,191,223,287,319,351])
    legal = np.array([True,True,True,True,True,False,True])
    # The original forecast at q+32 is issued during the handoff. The next
    # original issue with the complete action is q+64. Only that exact origin
    # qualifies; a missing validation block cannot be replaced by a later one.
    a,b = trial.follow_indices(q,legal,410)
    np.testing.assert_array_equal(a,[0,1,3,4])
    np.testing.assert_array_equal(b,[2,3,4,6])
    assert (q[b]>q[a]+25).all()


def test_real_jacobian_matches_central_difference_on_the_same_frozen_encoder():
    torch.manual_seed(38)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2,horizon=25),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2,16,23)
    mask = torch.ones_like(x,dtype=torch.bool)
    u = torch.zeros(2,3,4,requires_grad=True)
    w = torch.randn(2,8)

    def emit(z):
        h = trial.group.forward_hidden(base,x,mask,lambda p:p,state=z)
        return h@w.T

    f=emit(u)
    j=trial.jacobian(f,u)
    assert j.shape == (2,2,12)
    d=torch.randn_like(u)*.02
    finite=(emit(u+.02*d)-emit(u-.02*d))/.04
    torch.testing.assert_close(torch.einsum('bjd,bd->bj',j,d.flatten(1)),finite,atol=.001,rtol=.03)
    assert all(p.grad is None for p in base.parameters())


def test_history_does_not_reuse_unreleased_or_ineligible_rows():
    grams=torch.tensor([[[1.]],[[4.]],[[16.]],[[64.]]])
    rhs=torch.tensor([[1.],[4.],[16.],[64.]])
    index=np.array([[-1,-1,0],[-1,0,1],[0,1,2],[-1,2,-1]])
    g,r=trial.aggregate(grams,rhs,index)
    torch.testing.assert_close(g.flatten(),torch.tensor([1.,5.,21.,16.]))
    torch.testing.assert_close(r.flatten(),g.flatten())


def test_factorial_keeps_the_same_gram_or_rhs_when_comparing_cause():
    # Two successive observations make different parameter directions
    # identifiable. First order and inverse must use the same support.
    gram=torch.diag_embed(torch.tensor([[2.,8.],[8.,2.]]))
    real=torch.tensor([[6.,8.],[24.,2.]])
    sham=real.flip(-1)
    chosen=np.array([1])
    history=np.array([[0,1]])
    answers=trial.candidate_steps(gram,real,sham,chosen,history,factorial=True)
    linear,inverse,_,_,linear_history,inverse_single=answers
    torch.testing.assert_close(linear,real[1:])
    torch.testing.assert_close(linear_history,real.sum(0,keepdim=True))
    torch.testing.assert_close(inverse,torch.tensor([[30./11.,10./11.]]))
    torch.testing.assert_close(inverse_single,torch.tensor([[24./8.5,2./2.5]]))
