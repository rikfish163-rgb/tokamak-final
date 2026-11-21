import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.feedback_clock_20261023.learning import (
    feedback_step, predict_clock, update_counts)
from experiments.risk_aligned_online.tests.test_matched_forecast_meta import fixture
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.deepcoral_prefix_20261022.learning import adapt_arrived
from experiments.risk_aligned_online.target_feedback_meta_20261023.stream import predict_five


def test_clock_and_step_budget_are_explicit():
    assert update_counts(600,128)==[256,384,512]
    assert update_counts(310,25)==[256,281,306]
    assert update_counts(256,25)==[]
    with pytest.raises(ValueError): update_counts(600,0)
    with pytest.raises(ValueError): update_counts(600,True)


def test_four_target_update_is_exact_old_operator():
    model,signal=fixture();state=initial_encoder(model)
    a,ra=adapt_arrived(model,signal,256,state,device='cpu')
    b,rb=feedback_step(model,signal,256,state,targets=4,interval=128,device='cpu')
    for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for k,v in ra.items(): assert rb[k]==v
    assert rb['learning_rate']==.001


def test_latest_target_is_arrived_and_future_independent():
    model,signal=fixture();state=initial_encoder(model)
    original={k:v.clone() for k,v in model.state_dict().items()}
    a,ra=feedback_step(model,signal,281,state,targets=1,interval=25,device='cpu')
    changed=dict(signal,values=signal['values'].copy());changed['values'][281:]=999
    b,rb=feedback_step(model,changed,281,state,targets=1,interval=25,device='cpu')
    assert ra==rb and ra['forecast_origins']==[255]
    assert ra['last_observation']==280 and ra['first_affected_prediction']==281
    assert ra['learning_rate']==.001*25/128
    assert 0<ra['parameter_delta_norm']<=ra['learning_rate']*1.001
    for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    with pytest.raises(ValueError): feedback_step(model,dict(signal,y=1),281,state,targets=1,interval=25,device='cpu')
    with pytest.raises(ValueError): feedback_step(model,signal,281,state,targets=2,interval=25,device='cpu')


def test_no_observations_are_not_zero_targets():
    model,signal=fixture();signal=dict(signal,available=np.zeros_like(signal['available']))
    state=initial_encoder(model)
    new,r=feedback_step(model,signal,281,state,targets=1,interval=25,device='cpu')
    assert r['gradient_steps']==0 and r['observed_targets']==0
    for k in state: torch.testing.assert_close(state[k],new[k],rtol=0,atol=0)


def test_stream_control_reset_and_causal_predictions():
    model,signal=fixture();expected,updates=predict_five({'warm':model},signal,device='cpu')
    pred,records=predict_clock(model,signal,interval=128,device='cpu')
    np.testing.assert_array_equal(pred['four'],expected['warm_intra'])
    for a,b in zip(records['four'],updates['warm']):
        for k,v in b.items(): assert a[k]==v
    pred,r=predict_clock(model,signal,interval=25,device='cpu')
    changed=dict(signal,values=signal['values'].copy());changed['values'][310:]=88
    other,rr=predict_clock(model,changed,interval=25,device='cpu')
    again,_=predict_clock(model,signal,interval=25,device='cpu')
    for arm in pred:
        np.testing.assert_array_equal(pred[arm][:310],other[arm][:310])
        np.testing.assert_array_equal(pred[arm],again[arm])
        assert [x['first_affected_prediction'] for x in r[arm]]==update_counts(len(signal['values']),25)
        assert all(max(x['forecast_origins'])+25<=x['last_observation'] for x in r[arm])
