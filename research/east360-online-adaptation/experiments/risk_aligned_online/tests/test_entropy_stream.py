import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.entropy_stream_20261023.learning import adapt,predict
from experiments.risk_aligned_online.learned_update_loss_20261023.learning import step
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.tests.test_arrived_teacher import signal
from experiments.risk_aligned_online.warmstart_alignment_20261022.run import predict_many


def test_first_update_matches_verified_entropy_screen_and_no_future_access():
    model=small_model();s=signal();initial=initial_encoder(model)
    a,ra=adapt(model,s,256,initial,device='cpu')
    b,rb=step(model,None,s,256,device='cpu',mode='entropy')
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert ra['parameter_delta_norm']==rb['delta_norm'] and ra['loss_before']==rb['loss']
    s['values'][256:]=999
    c,rc=adapt(model,s,256,initial,device='cpu')
    assert ra==rc
    for k in a:torch.testing.assert_close(a[k],c[k],rtol=0,atol=0)
    with pytest.raises(ValueError):adapt(model,dict(s,y=0),256,initial,device='cpu')


def test_shot_reset_parent_frozen_and_prediction_order():
    model=small_model();s=signal(n=520);original={k:v.clone() for k,v in model.state_dict().items()}
    p,u=predict(model,s,device='cpu')
    ref=predict_many(model,{'static':s},device='cpu')['static']
    for a in p:
        np.testing.assert_array_equal(p[a][:256],ref[:256])
        assert [r['first_affected_prediction'] for r in u[a]]==[256,384,512]
        assert all(r['last_observation']+1==r['first_affected_prediction'] for r in u[a])
        assert all(r['parameter_delta_norm']<=.00101 for r in u[a])
    np.testing.assert_array_equal(p['local_reset'][:384],p['cumulative'][:384])
    assert np.any(p['local_reset'][384:]!=p['cumulative'][384:])
    again,uu=predict(model,s,device='cpu')
    assert uu==u
    for a in p:np.testing.assert_array_equal(p[a],again[a])
    for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    s['values'][384:]=999
    altered,au=predict(model,s,device='cpu')
    for a in p:
        np.testing.assert_array_equal(p[a][:384],altered[a][:384])
        assert u[a][:2]==au[a][:2]


def test_missing_window_skips_update_without_inventing_observations():
    model=small_model();s=signal();s['available'][:]=False
    origin=initial_encoder(model);fast,r=adapt(model,s,256,origin,device='cpu')
    assert r['gradient_steps']==0 and r['parameter_delta_norm']==0
    for k in fast:torch.testing.assert_close(fast[k],origin[k],rtol=0,atol=0)
