import numpy as np
import pytest
import torch


def test_signal_parts_ignore_unarrived_values_and_hold_differs():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.stream import signal_parts
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    model=small_model();fast=initial_encoder(model)
    signal=dict(values=np.random.default_rng(51).normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    parts,r=signal_parts(model,signal,256,fast,hold=False,device='cpu')
    signal['values'][256:]=1234
    after,_=signal_parts(model,signal,256,fast,hold=False,device='cpu')
    torch.testing.assert_close(parts,after,rtol=0,atol=0)
    fake,_=signal_parts(model,signal,256,fast,hold=True,device='cpu')
    assert not torch.equal(fake,parts) and r['last_observation']==255
    with pytest.raises(ValueError):signal_parts(model,dict(signal,label=1),256,fast,hold=False,device='cpu')


def test_stream_reset_parent_and_preupdate_predictions_match():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.stream import predict
    model=small_model();original={k:v.clone() for k,v in model.state_dict().items()}
    signal=dict(values=np.random.default_rng(61).normal(size=(280,23)).astype('float32'),available=np.ones((280,23),bool))
    curves,receipts=predict(model,signal,np.zeros(23),device='cpu')
    for arm in curves:
        np.testing.assert_array_equal(curves[arm][:256],curves['off'][:256])
        if arm.startswith(('mean','hold')):np.testing.assert_array_equal(curves[arm],curves['off'])
    assert all(v[0]['last_observation']==255 for v in receipts.values())
    again,_=predict(model,signal,np.zeros(23),device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm],again[arm])
    for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
