import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.structure_fixed_20261023.learning import update_pair,predict
from experiments.risk_aligned_online.tests.test_increment_forecast import model
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def setup():
    _,m=model('increment')
    with torch.no_grad():m.base.forecast_head.weight.normal_(0,.01)
    s=dict(values=np.random.default_rng(3).normal(size=(400,23)).astype('float32'),
           available=np.ones((400,23),bool))
    return m,s


def test_structure_is_frozen_and_control_has_same_requested_step():
    m,s=setup();states={a:initial_encoder(m) for a in ('observation','matched_all')}
    new,r=update_pair(m,{k:v[:256] for k,v in s.items()},256,states,device='cpu')
    for k in ('base.cls','base.position'):
        torch.testing.assert_close(new['observation'][k],states['observation'][k],rtol=0,atol=0)
    a,b=r['observation'],r['matched_all']
    assert a['requested_norm']>0 and a['requested_norm']==b['requested_norm']
    assert a['summary_delta_norm']==0 and b['summary_delta_norm']>0
    assert abs(a['parameter_delta_norm']-b['parameter_delta_norm'])<1e-6
    with pytest.raises(ValueError): update_pair(m,s,256,states,device='cpu')


def test_missing_data_does_not_create_gradient():
    m,s=setup();s['available'][:]=False
    states={a:initial_encoder(m) for a in ('observation','matched_all')}
    new,r=update_pair(m,{k:v[:256] for k,v in s.items()},256,states,device='cpu')
    assert all(v['gradient_steps']==0 for v in r.values())
    for a in states:
        for k in states[a]:torch.testing.assert_close(new[a][k],states[a][k],rtol=0,atol=0)


def test_causality_repetition_and_persistent_weights():
    m,s=setup();before={k:v.clone() for k,v in m.state_dict().items()}
    p,r=predict(m,s,device='cpu');p2,r2=predict(m,s,device='cpu');assert r==r2
    for a in p:np.testing.assert_array_equal(p[a],p2[a])
    s['values'][256:]=99.;p3,r3=predict(m,s,device='cpu')
    for a in p:
        assert r[a][0]==r3[a][0]
        np.testing.assert_array_equal(p[a][:256],p3[a][:256])
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
