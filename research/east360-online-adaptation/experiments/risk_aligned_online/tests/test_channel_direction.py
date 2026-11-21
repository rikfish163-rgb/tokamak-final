import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Simple23,Config
from experiments.risk_aligned_online.attention_lowrank_20261029.channel_direction import channel_losses,choose_channel,feedback_clock


def test_channel_terms_recover_the_exact_masked_loss_and_gradient():
    torch.manual_seed(19)
    base=Simple23(Config(),np.zeros(23),np.ones(23))
    prediction=torch.randn(4,25,23,requires_grad=True)
    truth=torch.randn_like(prediction);mask=torch.rand_like(prediction)>.3
    mask[:,:,2]=False;mask[:,:,10]=False;truth[0,0,0]=float('nan')
    terms=channel_losses(base,prediction,truth,mask)
    original=base.forecast_loss(prediction,truth,mask)
    torch.testing.assert_close(terms.sum(),original)
    g1,=torch.autograd.grad(terms.sum(),prediction,retain_graph=True)
    g2,=torch.autograd.grad(original,prediction)
    torch.testing.assert_close(g1,g2)
    assert terms[2]==0 and terms[10]==0
    assert channel_losses(base,prediction,truth,torch.zeros_like(mask)).sum()==0


def test_channel_selection_requires_positive_alignment_not_largest_loss():
    gradients=torch.tensor([[10.,0.],[-2.,2.],[0.,0.]])
    risk=torch.tensor([-1.,1.])
    idx,cos=choose_channel(gradients,risk)
    assert idx==1 and cos[1]>.99 and cos[0]<0 and cos[2]==0
    assert choose_channel(gradients,torch.tensor([0.,-1.]))[0]==-1
    assert choose_channel(gradients,torch.zeros(2))[0]==-1
    with pytest.raises(ValueError):choose_channel(gradients,torch.tensor([float('nan'),0.]))


def test_feedback_clock_cannot_affect_an_already_sealed_endpoint():
    assert feedback_clock(128) is None
    assert feedback_clock(255) is None
    assert feedback_clock(256)==256
    assert feedback_clock(383)==256
    assert feedback_clock(384)==384
    with pytest.raises(ValueError):feedback_clock(-1)


def test_alignment_sign_predicts_the_first_order_loss_change():
    gradients=torch.tensor([[2.,-1.],[-3.,-1.]],dtype=torch.float64)
    point=torch.tensor([1.,-2.],dtype=torch.float64)
    target=torch.tensor([-1.,3.],dtype=torch.float64)
    risk=point-target;idx,cos=choose_channel(gradients,risk)
    assert idx==0 and cos[idx]>0
    direction=gradients[idx]/gradients[idx].norm();eps=1e-5
    before=((point-target)**2).sum()/2
    after=((point-eps*direction-target)**2).sum()/2
    torch.testing.assert_close((after-before)/eps,-torch.dot(risk,direction),atol=1e-5,rtol=0)
