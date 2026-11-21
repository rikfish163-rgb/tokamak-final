import pytest
import torch
from torch import nn

from cta_copy_backtrack_20261107 import scaled_parameters,select_scale


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(1.))

    def forward(self,x,a):
        return {'forecast':self.weight*x}

    def forecast_loss(self,prediction,truth,mask):
        return ((prediction-truth).square()*mask).sum()/mask.sum().clamp_min(1)


def test_first_acceptable_scale_uses_actual_nonlinear_feedback_loss():
    model=Quadratic()
    p=dict(delta={'weight':torch.tensor(-4.)},loss=1.,gradient_norm=2.,update_norm=4.)
    result=select_scale(model,torch.ones(1),torch.ones(1,dtype=torch.bool),torch.zeros(1),torch.ones(1,dtype=torch.bool),p)
    assert result['factor']==.25 and result['loss_after']==0 and result['attempts']==3
    assert model.weight.item()==1.


def test_zero_update_is_exact_noop_without_trials():
    model=Quadratic()
    p=dict(delta={'weight':torch.tensor(0.)},loss=0.,gradient_norm=0.,update_norm=0.)
    result=select_scale(model,torch.ones(1),torch.ones(1,dtype=torch.bool),torch.zeros(1),torch.zeros(1,dtype=torch.bool),p)
    assert result['factor']==0 and result['attempts']==0
    assert torch.equal(scaled_parameters(model,p,0)['weight'],model.weight)


def test_non_descent_direction_is_rejected_without_mutation():
    model=Quadratic()
    p=dict(delta={'weight':torch.tensor(1.)},loss=1.,gradient_norm=2.,update_norm=1.)
    result=select_scale(model,torch.ones(1),torch.ones(1,dtype=torch.bool),torch.zeros(1),torch.ones(1,dtype=torch.bool),p)
    assert result['factor']==0 and not result['accepted'] and result['attempts']==9
    assert model.weight.item()==1.


def test_factor_is_bounded():
    with pytest.raises(ValueError):
        scaled_parameters(Quadratic(),{'delta':{}},2.)
