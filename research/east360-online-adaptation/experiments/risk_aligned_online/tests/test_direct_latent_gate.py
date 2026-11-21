import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.direct_latent_gate_20261023.learning import (
    DirectLatent, fit_predictor, origins, arrived_step,
)


def test_direct_latent_predictor_fits_nonconstant_change_and_keeps_identity_baseline():
    rng=np.random.default_rng(7);h=rng.normal(size=(256,4));prev=rng.normal(size=(256,4))
    future=h+.2*(h-prev)+.1
    predictor=fit_predictor(h,prev,future)
    x=torch.tensor(h,dtype=torch.float32,requires_grad=True);p=torch.tensor(prev,dtype=torch.float32)
    predicted=predictor(x,p)
    assert torch.mean((predicted-torch.tensor(future)).square())<1e-5
    predicted.square().mean().backward();assert x.grad is not None and x.grad.abs().sum()>0
    assert not list(predictor.parameters())
    identity=DirectLatent(np.zeros(8),np.ones(8),np.zeros((9,4)))
    torch.testing.assert_close(identity(x,p),x,rtol=0,atol=0)


def test_origins_have_past_context_future_feedback_and_later_query():
    assert origins(202)==[]
    assert origins(203)==[152]
    ends=origins(400)
    assert len(ends)==128 and ends==sorted(set(ends)) and ends[0]==152 and ends[-1]==349
    with pytest.raises(ValueError):origins(0)


def test_bad_predictor_data_rejected():
    x=np.zeros((3,4))
    with pytest.raises(ValueError):fit_predictor(x,x[:2],x)
    with pytest.raises(ValueError):fit_predictor(x,x,x,penalty=-1)
    x[0,0]=np.nan
    with pytest.raises(ValueError):fit_predictor(x,x,x)


def make_model_signal():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
    torch.set_num_threads(1);torch.manual_seed(7)
    model=PairedFeedback(Simple23(Config(dim=8,heads=2,fast_site='tokens'),np.zeros(23),np.ones(23)),np.ones(23))
    model.set_stage('frozen')
    x=np.random.default_rng(3).normal(size=(300,23)).astype(np.float32)
    return model,dict(values=x,available=np.ones_like(x,dtype=bool))


def test_arrived_update_cannot_access_unarrived_values_and_does_not_mutate_parent():
    model,signal=make_model_signal()
    predictor=DirectLatent(np.zeros(16),np.ones(16),np.zeros((17,8)))
    state={k:v.clone() for k,v in model.state_dict().items()}
    fast,r=arrived_step(model,predictor,signal,152,device='cpu')
    changed={k:v.copy() for k,v in signal.items()};changed['values'][178:]+=100
    later,s=arrived_step(model,predictor,changed,152,device='cpu')
    assert r==s and r['target_end']==177 and r['first_affected_prediction']==178
    assert r['parameter_delta_norm']>0
    for k,v in fast.items():torch.testing.assert_close(v,later[k],rtol=0,atol=0)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
    assert not any(p.grad is not None for p in model.parameters())
    with pytest.raises(ValueError):arrived_step(model,predictor,dict(signal,outcome=1),152,device='cpu')


def test_missing_new_target_produces_no_update_not_latent_pseudotruth():
    model,signal=make_model_signal();signal['available'][153:178]=False
    predictor=DirectLatent(np.zeros(16),np.ones(16),np.zeros((17,8)))
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    before=initial_encoder(model);fast,r=arrived_step(model,predictor,signal,152,device='cpu')
    assert r['gradient_steps']==0 and r['observed_new_values']==0 and r['parameter_delta_norm']==0
    for k,v in fast.items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_screening_gate_does_not_hide_positive_class_harm():
    from experiments.risk_aligned_online.direct_latent_gate_20261023.learning import risk_summary
    records=[dict(label=1,logits=dict(static=0.,learned=-.01,persistence=0.)),
             dict(label=0,logits=dict(static=0.,learned=-1.,persistence=0.))]
    result=risk_summary(records)
    assert result['learned']['balanced_bce']<result['static']['balanced_bce']
    assert not result['passes_direction_gate']
    assert risk_summary(records[1:])['static']['balanced_bce'] is None
