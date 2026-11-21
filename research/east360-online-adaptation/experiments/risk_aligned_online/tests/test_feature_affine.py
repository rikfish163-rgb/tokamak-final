from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.feature_affine import FeatureAffineModel
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case():
    torch.manual_seed(20261009)
    base=Simple23(Config(fast_site='tokens',fast_rule='normalized'),
                  torch.linspace(-1,1,23),torch.linspace(.5,2,23)).eval()
    x=torch.randn(2,128,23); a=torch.ones_like(x,dtype=torch.bool)
    return base,x,a


def test_zero_affine_is_exact_source_and_source_tensors_are_not_mutated():
    base,x,a=case(); model=FeatureAffineModel(base)
    original={k:v.clone() for k,v in base.state_dict().items()}
    f=model.initial_fast(2).requires_grad_()
    # Match the differentiable attention path, not the separate inference kernel.
    expected=base(x.requires_grad_(),a,base.initial_fast(2))
    out=model(x,a,f)
    for k in out:torch.testing.assert_close(out[k],expected[k],rtol=0,atol=0)
    for k,v in original.items():torch.testing.assert_close(base.state_dict()[k],v,rtol=0,atol=0)
    assert all(not p.requires_grad for p in base.parameters())


def test_parameter_scope_is_23_scales_and_23_offsets_not_a_risk_head():
    base,x,a=case(); model=FeatureAffineModel(base)
    assert model.temporary_dim==46
    assert model.initial_fast(2).shape==(2,46)
    p=list(model.persistent_parameters())
    assert sum(v.numel() for v in p)==46 and p==[model.affine]
    assert set(dict(model.named_children()))=={'base'}
    assert not {id(v) for v in p}&{id(v) for v in base.parameters()}
    assert not model.affine.any()


def test_affine_moves_input_tokens_and_both_tasks_have_fast_gradients():
    base,x,a=case(); model=FeatureAffineModel(base)
    seen=[]
    h=base.encoder.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
    f=model.initial_fast(2).requires_grad_(); out=model(x,a,f)
    risk,=torch.autograd.grad(out['logit'].square().mean(),f,retain_graph=True)
    pred,=torch.autograd.grad(out['forecast'].square().mean(),f)
    assert risk.norm()>0 and pred.norm()>0 and torch.isfinite(risk).all() and torch.isfinite(pred).all()
    f=f.detach(); f[:,3]=.5; f[:,26]=.4
    moved=model(x,a,f); h.remove()
    assert not torch.equal(seen[0],seen[1])
    assert not torch.equal(out['logit'],moved['logit'])
    assert not torch.equal(out['forecast'],moved['forecast'])


def test_missing_values_do_not_receive_an_affine_signal_or_gradient():
    base,x,a=case(); model=FeatureAffineModel(base)
    a[:,:,7]=False; x[:,:,7]=float('nan')
    f=model.initial_fast(2).requires_grad_(); out=model(x,a,f)
    g,=torch.autograd.grad(out['forecast'].square().mean(),f)
    assert not g[:,[7,30]].any() and torch.isfinite(g).all()
    changed=x.clone(); changed[:,:,7]=1e20
    result=model(changed,a,f)
    for k in out:torch.testing.assert_close(out[k],result[k],rtol=0,atol=0)


def test_affine_coefficients_are_bounded_and_persistent_fast_share_same_site():
    base,x,a=case(); model=FeatureAffineModel(base)
    f=model.initial_fast(2); f[:,0]=.1; f[:,23]=.2
    reference=model(x,a,f)
    with torch.no_grad():model.affine.copy_(f[0])
    out=model(x,a,model.initial_fast(2))
    for k in out:torch.testing.assert_close(out[k],reference[k],rtol=0,atol=0)
    gain,bias=model.coefficients(torch.tensor([[1e5]*23+[-1e5]*23],dtype=x.dtype))
    assert torch.allclose(gain,torch.full_like(gain,2.))
    assert torch.allclose(bias,torch.full_like(bias,-2.))


def test_source_forecast_targets_keep_original_coordinates():
    base,x,a=case(); model=FeatureAffineModel(base)
    truth=x[:,:25]; available=a[:,:25]
    p=model(x,a)['forecast']
    torch.testing.assert_close(model.forecast_loss(p,truth,available),base.forecast_loss(p,truth,available),rtol=0,atol=0)


def test_generic_causal_stream_uses_declared_46_parameters_and_discards_them():
    base,x,a=case(); model=FeatureAffineModel(base)
    values=x[0,:61].numpy().copy(); observed=a[0,:61].numpy().copy()
    shot=dict(device='EAST',shot=1,values=values,available=observed,times=np.arange(61)/1000.)
    state={k:v.clone() for k,v in model.state_dict().items()}
    s,audit=predict_shot(model,shot,True,stride=100,step=.05,radius=1.,data_api=simple_data)
    assert s.shape==(61,) and np.isfinite(s).all()
    assert audit['fast_parameter_count']==46 and audit['fast_site']=='normalized flat23 before embedding'
    assert len(audit['attempts'][0]['fast_at_origin'])==46
    assert audit['attempts'][0]['maturity_index']==25
    if audit['attempts'][0]['accepted']:assert audit['attempts'][0]['effective_index']==26
    assert audit['fast_discarded']
    for k,v in state.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    again,_=predict_shot(model,shot,True,stride=100,step=.05,radius=1.,data_api=simple_data)
    np.testing.assert_array_equal(s,again)


def test_invalid_declared_dimension_is_a_contract_error():
    base,x,a=case(); model=FeatureAffineModel(base); model.temporary_dim=0
    shot=dict(device='EAST',shot=1,values=x[0,:40].numpy(),available=a[0,:40].numpy(),times=np.arange(40)/1000.)
    with pytest.raises(ValueError,match='dimension'):predict_shot(model,shot,True)


def test_reversible_forecast_uses_the_exact_input_inverse_without_changing_risk():
    base,x,a=case()
    direct=FeatureAffineModel(base)
    reversible=FeatureAffineModel(base,forecast_coordinates='observed')
    f=torch.linspace(-.25,.25,46)[None].expand(2,-1).clone().requires_grad_()
    original=direct(x,a,f); out=reversible(x,a,f)
    gain,bias=reversible.coefficients(f)
    torch.testing.assert_close(out['logit'],original['logit'],rtol=0,atol=0)
    torch.testing.assert_close(out['forecast'],
        (original['forecast']-bias[:,None])/gain[:,None],rtol=0,atol=0)
    g,=torch.autograd.grad(out['forecast'].square().mean(),f)
    assert g.norm()>0 and torch.isfinite(g).all()


def test_forecast_coordinate_choice_is_explicit_and_checked():
    base,_,_=case()
    assert FeatureAffineModel(base).forecast_coordinates=='source'
    with pytest.raises(ValueError,match='coordinates'):
        FeatureAffineModel(base,forecast_coordinates='automatic')


def test_coordinate_roundtrip_is_observed_truth_not_a_synthetic_target(monkeypatch):
    base,x,a=case()
    # A source forecast with the exactly known dynamics F(u)=1.2u+0.3.
    # A coordinate change u=g*v+b means the observed future is
    # (1.2*(g*v+b)+0.3-b)/g, not the unreversed source forecast.
    def source_forward(values,available,fast):
        level=torch.asinh((values-base.center)/base.scale)
        future=(1.2*level[:,-1:]+.3).expand(-1,25,-1)
        return dict(logit=level[:,-1].sum(-1),forecast=future)
    monkeypatch.setattr(base,'forward',source_forward)
    direct=FeatureAffineModel(base)
    reversible=FeatureAffineModel(base,forecast_coordinates='observed')
    f=torch.full((2,46),.2,requires_grad=True)
    gain,bias=reversible.coefficients(f)
    level=torch.asinh((x-base.center)/base.scale)[:,-1:]
    target=(1.2*(level*gain[:,None]+bias[:,None])+.3-bias[:,None])/gain[:,None]
    # Freeze a genuinely observed target; it cannot move with the adapter.
    truth=(base.center+base.scale*target.sinh()).expand(-1,25,-1).detach()
    valid=torch.ones_like(truth,dtype=torch.bool)
    r=reversible(x,a,f); d=direct(x,a,f)
    assert float(reversible.forecast_loss(r['forecast'],truth,valid).detach())<1e-12
    assert float(direct.forecast_loss(d['forecast'],truth,valid).detach())>1e-3
    assert not truth.requires_grad


def test_reversible_zero_identity_and_causal_reset():
    base,x,a=case(); model=FeatureAffineModel(base,forecast_coordinates='observed')
    direct=FeatureAffineModel(base)
    f=model.initial_fast(2).requires_grad_()
    out=model(x,a,f); original=direct(x,a,f)
    for k in out:torch.testing.assert_close(out[k],original[k],rtol=0,atol=0)
    shot=dict(device='EAST',shot=7,values=x[0,:61].numpy(),available=a[0,:61].numpy(),times=np.arange(61)/1000.)
    state={k:v.clone() for k,v in model.state_dict().items()}
    scores,audit=predict_shot(model,shot,True)
    again,_=predict_shot(model,shot,True)
    np.testing.assert_array_equal(scores,again)
    assert audit['fast_discarded'] and audit['fast_parameter_count']==46
    assert audit['attempts'][0]['maturity_index']==25
    for k,v in state.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def window_case():
    base,x,a=case()
    model=FeatureAffineModel(base,forecast_coordinates='observed',temporary_mode='window_gain')
    return base,model,x,a


def test_window_gain_scope_has_no_temporary_offset_but_retains_persistent_experience():
    base,model,x,a=window_case()
    assert model.temporary_dim==23 and model.initial_fast(2).shape==(2,23)
    assert sum(p.numel() for p in model.persistent_parameters())==46
    assert all(not p.requires_grad for p in base.parameters())
    with pytest.raises(ValueError,match='temporary'):
        FeatureAffineModel(base,temporary_mode='guess')
    with pytest.raises(ValueError,match='observed'):
        FeatureAffineModel(base,temporary_mode='window_gain',forecast_coordinates='source')


def test_window_gain_zero_is_exact_reversible_parent_including_persistent_delta():
    base,model,x,a=window_case()
    parent=FeatureAffineModel(base,forecast_coordinates='observed')
    for persistent in (0.,.13):
        with torch.no_grad():
            model.affine.fill_(persistent); parent.affine.fill_(persistent)
        out=model(x,a,model.initial_fast(2).requires_grad_())
        expected=parent(x,a,parent.initial_fast(2).requires_grad_())
        for k in out:torch.testing.assert_close(out[k],expected[k],rtol=0,atol=0)


def test_window_gain_keeps_observed_mean_and_rescales_only_deviations(monkeypatch):
    base,model,x,a=window_case(); seen=[]
    a[:,:17,6]=False
    def capture(values,available,fast):
        seen.append(torch.asinh((values-base.center)/base.scale))
        return dict(logit=values[:,-1].sum(-1),forecast=values.new_zeros(len(values),25,23))
    monkeypatch.setattr(base,'forward',capture)
    f=torch.full((2,23),.31,requires_grad=True)
    with torch.no_grad():model.affine.fill_(.11)
    model(x,a,f)
    level=torch.asinh((x-base.center)/base.scale)
    mean=level.masked_fill(~a,0.).sum(1,keepdim=True)/a.sum(1,keepdim=True).clamp_min(1)
    g=(np.log(2)*f.tanh()).exp()
    gp,bp=model.coefficients(model.affine[None].expand(2,-1))
    expected=gp[:,None]*(mean+g[:,None]*(level-mean))+bp[:,None]
    torch.testing.assert_close(seen[0][a],expected[a],rtol=1e-5,atol=1e-6)
    mapped_mean=seen[0].masked_fill(~a,0.).sum(1,keepdim=True)/a.sum(1,keepdim=True).clamp_min(1)
    torch.testing.assert_close(mapped_mean,gp[:,None]*mean+bp[:,None],rtol=1e-5,atol=1e-6)


def test_window_gain_forecast_inverse_uses_the_same_past_window_mean(monkeypatch):
    base,model,x,a=window_case(); seen=[]
    def forecast(values,available,fast):
        level=torch.asinh((values-base.center)/base.scale)
        predicted=(1.2*level[:,-1:]+.3).expand(-1,25,-1)
        seen.append(predicted)
        return dict(logit=level[:,-1].sum(-1),forecast=predicted)
    monkeypatch.setattr(base,'forward',forecast)
    f=torch.full((2,23),.2,requires_grad=True)
    with torch.no_grad():model.affine.fill_(.15)
    out=model(x,a,f)
    level=torch.asinh((x-base.center)/base.scale); mean=level.mean(1,keepdim=True)
    g=(np.log(2)*f.tanh()).exp(); gp,bp=model.coefficients(model.affine[None].expand(2,-1))
    expected=(((seen[0]-bp[:,None])/gp[:,None])-mean)/g[:,None]+mean
    torch.testing.assert_close(out['forecast'],expected,rtol=1e-5,atol=1e-6)
    grad,=torch.autograd.grad(out['forecast'].square().mean(),f)
    assert grad.norm()>0 and torch.isfinite(grad).all()


def test_constant_levels_cannot_be_shifted_by_window_gain():
    base,model,x,a=window_case()
    x=x[:,:1].expand(-1,128,-1).clone()
    left=model(x,a,model.initial_fast(2).requires_grad_())
    right=model(x,a,torch.ones(2,23,requires_grad=True))
    # Signal forecasts may change after inverse mapping. The risk pathway
    # must not turn a constant observed level into a different operating level.
    torch.testing.assert_close(left['logit'],right['logit'],rtol=0,atol=0)


def test_window_gain_unseen_channels_get_no_input_or_inverse_forecast_gradient():
    base,model,x,a=window_case(); a[:,:,7]=False; x[:,:,7]=float('nan')
    f=torch.full((2,23),.3,requires_grad=True)
    out=model(x,a,f); grad,=torch.autograd.grad(out['forecast'].square().mean(),f)
    assert not grad[:,7].any() and torch.isfinite(grad).all()
    changed=x.clone();changed[:,:,7]=1e25
    again=model(changed,a,f)
    for k in out:torch.testing.assert_close(out[k],again[k],rtol=0,atol=0)


def test_window_gain_has_neural_risk_gradient_and_preserves_source_tensors():
    base,model,x,a=window_case(); source={k:v.clone() for k,v in base.state_dict().items()}
    f=model.initial_fast(2).requires_grad_(); out=model(x,a,f)
    risk,=torch.autograd.grad(out['logit'].square().mean(),f)
    assert risk.norm()>0 and torch.isfinite(risk).all()
    for k,v in source.items():torch.testing.assert_close(base.state_dict()[k],v,rtol=0,atol=0)


def test_window_gain_stream_is_causal_and_discards_23_temporary_parameters():
    base,model,x,a=window_case()
    s=dict(device='EAST',shot=8,values=x[0,:61].numpy(),available=a[0,:61].numpy(),times=np.arange(61)/1000.)
    right={**s,'values':s['values'].copy()};right['values'][26:]=999.
    state={k:v.clone() for k,v in model.state_dict().items()}
    scores,audit=predict_shot(model,s,True)
    changed,_=predict_shot(model,right,True)
    np.testing.assert_array_equal(scores[:26],changed[:26])
    again,_=predict_shot(model,s,True);np.testing.assert_array_equal(scores,again)
    assert audit['fast_parameter_count']==23 and audit['fast_discarded']
    assert len(audit['attempts'][0]['fast_at_origin'])==23
    assert audit['attempts'][0]['maturity_index']==25
    for k,v in state.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
