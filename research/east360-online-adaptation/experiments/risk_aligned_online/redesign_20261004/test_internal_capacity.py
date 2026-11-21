"""Contracts for an explicitly retrospective parameter-location diagnostic."""
import pytest
import torch

from ..simple23 import Config, Simple23
from . import internal_capacity as probe


@pytest.fixture
def source():
    torch.set_num_threads(2)
    with torch.random.fork_rng():
        torch.manual_seed(131)
        model = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                         torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
        x = torch.randn(3, 128, 23)
    return model, x, torch.ones_like(x, dtype=torch.bool)


@pytest.mark.parametrize('site,count', [('fast128',128), ('embedding',23616),
                                       ('encoder_norms',512), ('last_ffn',16576)])
def test_zero_delta_identity_gradient_and_scope(source, site, count):
    model,x,a=source
    spec=probe.parameter_spec(model,site)
    assert spec['count']==count
    assert not any('risk_head' in n or 'forecast_head' in n for n in spec['names'])
    delta=torch.zeros(count,requires_grad=True)
    result=probe.forward_delta(model,spec,delta,x,a)
    expected=model(x,a,model.initial_fast(3).detach().requires_grad_(True))
    assert torch.allclose(result['logit'],expected['logit'],atol=1e-7,rtol=1e-6)
    gradient,=torch.autograd.grad(result['logit'].sum(),delta)
    assert gradient.shape==delta.shape and torch.isfinite(gradient).all()
    assert gradient.abs().sum()>0


@pytest.mark.parametrize('site', ['fast128','embedding','encoder_norms','last_ffn'])
def test_probe_is_disposable_bounded_and_records_reference_damage(source,site):
    model,x,a=source
    original={k:v.clone() for k,v in model.state_dict().items()}
    out=probe.probe_location(model,x[:1],a[:1],1,.999,site,
                             reference=(x,a,torch.tensor([0.,1.,0.])),steps=3,step=.02,radius=.03)
    assert out['diagnostic']=='POSTHOC_ORACLE_INTERNAL_LOCATION'
    assert out['online_performance'] is False
    assert out['parameter_count']==probe.parameter_spec(model,site)['count']
    assert out['trace'][0]['delta_norm']==0
    assert len(out['trace'])==4
    assert all(r['delta_norm']<=.030001 for r in out['trace'])
    assert out['best_probability']>=out['initial_probability']
    assert out['best_reference']['n_endpoints']==3
    assert out['best_reference']['lost_correct_negative']>=0
    assert out['best_reference']['lost_correct_positive']>=0
    first=next((r['step'] for r in out['trace'] if r['threshold_reached']),None)
    assert out['first_crossing_step']==first
    assert (out['first_crossing_reference'] is not None)==(first is not None)
    assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('field,value', [('site','head'),('label',2),('label',True),
    ('threshold',float('nan')),('steps',-1),('steps',True),('step',0.),('radius',-1.)])
def test_bad_controls_rejected(source,field,value):
    model,x,a=source
    kw=dict(label=1,threshold=.7,site='embedding',steps=2,step=.1,radius=1.)
    kw[field]=value
    with pytest.raises(ValueError):
        probe.probe_location(model,x[:1],a[:1],reference=(x,a,torch.zeros(3)),**kw)


def test_reference_changes_are_endpoint_counts_not_shot_metrics():
    result=probe.reference_change(torch.tensor([.2,.8,.1,.9]),
        torch.tensor([.8,.2,.1,.9]),torch.tensor([0.,1.,1.,0.]),.7)
    assert result['lost_correct_negative']==1
    assert result['lost_correct_positive']==1
    assert result['new_correct_negative']==result['new_correct_positive']==0
    assert result['unit']=='endpoint_not_shot'


def test_reference_requires_binary_labels_and_nonempty(source):
    model,x,a=source
    with pytest.raises(ValueError):
        probe.probe_location(model,x[:1],a[:1],1,.7,'embedding',
                             reference=(x,a,torch.tensor([0.,.5,1.])),steps=1)


def test_late_update_site_keeps_original_differentiable_attention_dispatch(source):
    model,x,a=source
    observed=[]
    hook=model.encoder.layers[0].self_attn.register_forward_pre_hook(
        lambda module,args: observed.append(args[0].requires_grad))
    try:
        spec=probe.parameter_spec(model,'last_ffn')
        probe.forward_delta(model,spec,torch.zeros(spec['count'],requires_grad=True),x,a)
    finally:
        hook.remove()
    assert observed==[True]
