import pytest
import torch

from risk_meta_step_20261107 import first_adam_step


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('magnitude',[0.,.001,10.])
def test_first_step_and_moments_match_existing_adamw(dtype,magnitude):
    torch.manual_seed(17)
    original=torch.randn(4,7,dtype=dtype)
    gradient=torch.randn_like(original)*magnitude
    actual=first_adam_step(original,gradient,lr=1e-4)
    reference=torch.nn.Parameter(original.clone())
    opt=torch.optim.AdamW([reference],lr=1e-4,weight_decay=0.,foreach=False)
    reference.grad=gradient.clone()
    torch.nn.utils.clip_grad_norm_([reference],1.,error_if_nonfinite=True)
    opt.step()
    tolerance=2e-7 if dtype==torch.float32 else 2e-14
    torch.testing.assert_close(actual['weight'],reference,atol=tolerance,rtol=0)
    torch.testing.assert_close(actual['exp_avg'],opt.state[reference]['exp_avg'],atol=tolerance,rtol=1e-6)
    torch.testing.assert_close(actual['exp_avg_sq'],opt.state[reference]['exp_avg_sq'],atol=tolerance,rtol=1e-6)


def test_zero_gradient_step_has_finite_correct_derivative():
    weight=torch.ones(2,3,dtype=torch.float64)
    gradient=torch.zeros_like(weight,requires_grad=True)
    actual=first_adam_step(weight,gradient,lr=1e-4)['weight']
    derivative,=torch.autograd.grad(actual.sum(),gradient)
    torch.testing.assert_close(derivative,torch.full_like(gradient,-1e4))


def test_nonzero_first_step_derivative_matches_center_difference():
    weight=torch.zeros(2,3,dtype=torch.float64)
    gradient=torch.tensor([[1e-4,-2e-4,3e-4],[-4e-4,5e-4,-6e-4]],dtype=torch.float64,requires_grad=True)
    actual=first_adam_step(weight,gradient,lr=1e-4)['weight'].sum()
    derivative,=torch.autograd.grad(actual,gradient)
    direction=torch.linspace(-.2,.2,6,dtype=torch.float64).reshape(2,3)
    step=1e-8
    high=first_adam_step(weight,gradient.detach()+step*direction,lr=1e-4)['weight'].sum()
    low=first_adam_step(weight,gradient.detach()-step*direction,lr=1e-4)['weight'].sum()
    torch.testing.assert_close((derivative*direction).sum(),(high-low)/(2*step),atol=2e-10,rtol=2e-4)


def teacher_case():
    from risk_meta_step_20261107 import MetaOrderedTrajectory
    from risk_feature_intra_20261107 import FeatureIntra
    from test_risk_feature_intra_20261107 import setup

    base,old,e=setup()
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=8,scenarios=4)
    decoder.load_state_dict(old.state_dict())
    model=FeatureIntra(base,decoder)
    model.decoder.requires_grad_(True)
    return model,e


def test_source_teacher_preserves_the_deployed_signal_objective_and_first_gradient():
    from joint_motion_20261107 import OrderedTrajectory
    from risk_feature_intra_20261107 import FeatureIntra

    model,e=teacher_case()
    ordinary=OrderedTrajectory(model.risk.center,model.risk.scale,dim=8,scenarios=4)
    ordinary.load_state_dict(model.decoder.state_dict())
    original=FeatureIntra(model.risk,ordinary)
    a=model.signal_loss(e)
    b=original.signal_loss(e)
    ga,=torch.autograd.grad(a,model.weight)
    gb,=torch.autograd.grad(b,original.weight)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(ga,gb,atol=1e-6,rtol=1e-5)


def test_risk_teacher_reaches_decoder_only_through_the_parameter_update():
    from copy import deepcopy
    from risk_meta_step_20261107 import teacher_step
    from torch.nn.attention import sdpa_kernel,SDPBackend

    model,e=teacher_case()
    state=deepcopy(model.state_dict())
    labels=torch.tensor([0.,1.])
    with sdpa_kernel(SDPBackend.MATH):
        full=teacher_step(model,e,e['x'],e['a'],labels)
        parameters=tuple(model.decoder.parameters())
        gradients=torch.autograd.grad(full['risk_loss'],parameters,allow_unused=True)
        stopped=teacher_step(model,e,e['x'],e['a'],labels,stop_update=True)
        control=torch.autograd.grad(stopped['risk_loss'],parameters,allow_unused=True)
    torch.testing.assert_close(full['logit'],stopped['logit'],atol=0,rtol=0)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients if g is not None)>0
    assert all(g is None or torch.count_nonzero(g)==0 for g in control)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,state[key],atol=0,rtol=0)


def test_different_outer_labels_cannot_change_the_signal_step_itself():
    from risk_meta_step_20261107 import teacher_step
    from torch.nn.attention import sdpa_kernel,SDPBackend

    model,e=teacher_case()
    with sdpa_kernel(SDPBackend.MATH):
        zero=teacher_step(model,e,e['x'],e['a'],torch.zeros(2))
        one=teacher_step(model,e,e['x'],e['a'],torch.ones(2))
    torch.testing.assert_close(zero['gradient'],one['gradient'],atol=0,rtol=0)
    torch.testing.assert_close(zero['weight'],one['weight'],atol=0,rtol=0)
    torch.testing.assert_close(zero['logit'],one['logit'],atol=0,rtol=0)
    assert not torch.equal(zero['risk_loss'],one['risk_loss'])
