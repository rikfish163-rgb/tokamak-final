from copy import deepcopy

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory
from probe_risk_masked_utility_20261107 import masked_view,measure
from test_joint_stream_20261107 import base_model,signal


def example():
    base=base_model();s=signal(190)
    e=j.sample_signals([s],j.eligible_origins([s]),np.random.default_rng(7),1,'cpu')
    e['a'].fill_(True);e['mask'].fill_(True)
    qa=torch.ones_like(e['a'])
    return base,e,qa


def test_missing_sensor_changes_availability_only_and_does_not_mutate_native():
    _,e,qa=example();before={k:v.clone() for k,v in e.items()};qa_before=qa.clone()
    masked,query_mask=masked_view(e,qa)
    assert not masked['a'][...,2].any() and not masked['mask'][...,2].any()
    assert not query_mask[...,2].any()
    for key in ('x','truth','origin'):
        torch.testing.assert_close(masked[key],e[key],atol=0,rtol=0)
    columns=[c for c in range(23) if c!=2]
    for key in ('a','mask'):
        torch.testing.assert_close(masked[key][...,columns],e[key][...,columns])
    for key in e:
        torch.testing.assert_close(e[key],before[key],atol=0,rtol=0)
    torch.testing.assert_close(qa,qa_before,atol=0,rtol=0)


def test_measure_keeps_source_templates_and_distinguishes_current_future_targets():
    base,e,qa=example()
    decoder=OrderedTrajectory(base.center,base.scale,dim=8,scenarios=4,motion_weight=.472)
    frozen_base=deepcopy(base.state_dict());frozen_decoder=deepcopy(decoder.state_dict())
    measured=measure(base,decoder,e,e['x'],qa,0.,1.)
    assert np.isfinite(list(v for v in measured.values() if v is not None)).all()
    assert (measured['future_risk_after']-measured['future_risk_before'])*(measured['current_risk_after']-measured['current_risk_before'])<=0
    if measured['cosine'] is not None:
        assert -1.00001<=measured['cosine']<=1.00001
    for model,before in ((base,frozen_base),(decoder,frozen_decoder)):
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
