"""Real signal feedback must reach the temporary risk-encoder weights."""

from copy import deepcopy

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory
from risk_feature_intra_20261107 import FeatureIntra, update
from test_joint_stream_20261107 import base_model, signal


def setup():
    base = base_model()
    torch.manual_seed(37)
    decoder = OrderedTrajectory(base.center,base.scale,dim=8,scenarios=4,motion_weight=.472)
    s = signal(190)
    e = j.sample_signals([s],j.eligible_origins([s]),np.random.default_rng(41),2,'cpu')
    return base,decoder,e


def test_initial_risk_is_unchanged_and_only_internal_temporal_weight_is_trainable():
    base,decoder,e = setup()
    model = FeatureIntra(base,decoder)
    trainable = [name for name,p in model.named_parameters() if p.requires_grad]
    assert trainable == ['risk.encoder.layers.0.linear2.weight']
    with torch.no_grad():
        torch.testing.assert_close(model(e['x'],e['a'])['logit'],base(e['x'],e['a'])['logit'],atol=0,rtol=0)


def test_signal_update_reaches_current_risk_representation_not_just_forecast_decoder():
    base,decoder,e = setup()
    original_base = deepcopy(base.state_dict())
    original_decoder = deepcopy(decoder.state_dict())
    model = FeatureIntra(base,decoder)
    before = deepcopy(model.state_dict())
    with torch.no_grad():
        risk_before = model.risk(e['x'],e['a'])['logit']
    opt = torch.optim.AdamW([model.weight],lr=1e-4,weight_decay=0.)
    info = update(model,opt,e)
    assert info['gradient_norm'] > 0 and np.isfinite(info['objective'])
    assert not torch.equal(model.weight,before['risk.encoder.layers.0.linear2.weight'])
    with torch.no_grad():
        risk_after = model.risk(e['x'],e['a'])['logit']
    assert (risk_after-risk_before).abs().max() > 0
    for name,value in model.state_dict().items():
        if name != 'risk.encoder.layers.0.linear2.weight':
            torch.testing.assert_close(value,before[name],atol=0,rtol=0)
    for name,value in base.state_dict().items():
        torch.testing.assert_close(value,original_base[name],atol=0,rtol=0)
    for name,value in decoder.state_dict().items():
        torch.testing.assert_close(value,original_decoder[name],atol=0,rtol=0)


def test_matured_values_affect_real_feedback_but_not_hold_value_control():
    base,decoder,e = setup()
    changed = dict(e,truth=e['truth']+1.)
    model = FeatureIntra(base,decoder)
    real = model.signal_loss(e,feedback='real')
    other = model.signal_loss(changed,feedback='real')
    assert not torch.equal(real,other)
    torch.testing.assert_close(model.signal_loss(e,feedback='hold'),
                               model.signal_loss(changed,feedback='hold'),atol=0,rtol=0)


def test_new_shot_starts_at_the_original_risk_model_after_an_adapted_shot():
    base,decoder,e = setup()
    first = FeatureIntra(base,decoder)
    opt = torch.optim.AdamW([first.weight],lr=1e-4,weight_decay=0.)
    update(first,opt,e)
    fresh = FeatureIntra(base,decoder)
    with torch.no_grad():
        torch.testing.assert_close(fresh.risk(e['x'],e['a'])['logit'],base(e['x'],e['a'])['logit'],atol=0,rtol=0)


def test_current_risk_stream_keeps_the_actual_feedback_clock_and_prefix():
    from run_risk_feature_intra_20261107 import feature_curves

    base,decoder,_ = setup()
    s = signal(164)
    traces,info = feature_curves(base,decoder,s)
    assert info['releases'] == [153]
    for name in ('real','wrong_time','hold'):
        np.testing.assert_allclose(traces[name][:153],traces['static'][:153],atol=3e-6,rtol=0)
    assert max(abs(traces['real'][153:]-traces['static'][153:])) > 0
    partial,_ = feature_curves(base,decoder,{k:v[:154] for k,v in s.items()})
    for name in traces:
        np.testing.assert_allclose(partial[name],traces[name][:154],atol=3e-6,rtol=0)
    cold,info = feature_curves(base,decoder,s,warm=True)
    assert info['releases'] == []
    for value in cold.values():
        np.testing.assert_array_equal(value,cold['static'])
