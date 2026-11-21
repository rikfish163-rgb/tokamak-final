from copy import deepcopy

import numpy as np
import torch
import pytest

from experiments.risk_aligned_online.forecast_replay import ForecastRetrainer, forecast_batch
from experiments.risk_aligned_online.replay_retraining import ReplayRetrainer
from experiments.risk_aligned_online.tests.test_ended_adaptation import model, example
from experiments.risk_aligned_online.simple_online import predict_shot, run_arm
from experiments.risk_aligned_online.tests.test_simple_online import Toy


def test_forecast_refit_does_not_modify_classifier_or_supervision_samples():
    a,b=model(),model()
    s=example()
    base=ReplayRetrainer(a,mode='continue',steps=2,scope='full')
    paired=ForecastRetrainer(b,supervised_steps=2,forecast_steps=2)
    x,y=base(a,s),paired(b,s)
    assert x['sample_sha256']==y['sample_sha256']
    for name,p in a.named_parameters():
        if not name.startswith('forecast_head.'):
            assert torch.equal(p,dict(b.named_parameters())[name])
    assert not torch.equal(a.forecast_head.weight,b.forecast_head.weight)
    assert y['forecast_risk_max_delta']==0


def test_forecast_batches_never_include_post_event_signal():
    s=example()
    s['values'][40:]=123456
    s['valid'][40:]=False
    x,m,f,fm,_=forecast_batch([s],np.random.default_rng(7),64,model().cfg,'cpu')
    assert not (x==123456).any()
    assert not (f==123456).any()
    assert fm.all()
    assert m.dtype==torch.bool


def test_temporary_token_update_does_not_change_persistent_parameters():
    from dataclasses import replace
    net=model()
    net.cfg=replace(net.cfg,fast_site='tokens')
    before=deepcopy(net.state_dict())
    scores,count=predict_shot(net,example(),intra=True,device='cpu')
    assert len(scores)==60 and count==2
    assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())


def test_custom_both_rejects_unmaintained_forecast_head(tmp_path):
    with pytest.raises(ValueError):
        run_arm(Toy(),order=[],arm='both',threshold=.5,output_dir=tmp_path/'x',
                persistent_updater=lambda *a,**k: {})


def test_forecast_failure_restores_classifier_head_and_history(monkeypatch):
    from experiments.risk_aligned_online import forecast_replay
    net=model()
    before=deepcopy(net.state_dict())
    learner=ForecastRetrainer(net,supervised_steps=1,forecast_steps=1)
    def fail(*args):
        with torch.no_grad():
            net.forecast_head.weight.fill_(float('nan'))
        raise FloatingPointError('test forecast failure')
    monkeypatch.setattr(forecast_replay,'fit_forecaster',fail)
    with pytest.raises(FloatingPointError,match='test forecast failure'):
        learner(net,example())
    assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())
    assert len(learner.supervised.history)==0
    assert learner.supervised.round==0
    assert not learner.supervised.seen


def test_aligned_head_maintenance_preserves_supervised_classifier_and_is_paired():
    from dataclasses import replace
    a,b=model(),model()
    for net in (a,b):
        net.cfg=replace(net.cfg,fast_site='tokens',fast_rule='normalized')
    plain=ReplayRetrainer(a,mode='continue',steps=2,scope='full',batch_size=8)
    aligned=ForecastRetrainer(b,supervised_steps=2,forecast_steps=2,batch_size=8,maintenance='aligned')
    x,y=plain(a,example()),aligned(b,example())
    assert x['sample_sha256']==y['sample_sha256']
    assert y['maintenance']=='aligned'
    assert y['forecast_risk_max_delta']==0
    assert y['forecast_losses'][0]['head_grad_norm']>0
    for name,value in a.state_dict().items():
        if not name.startswith('forecast_head.'):
            assert torch.equal(value,b.state_dict()[name])


def test_ended_alignment_batch_respects_validity_and_single_class():
    from experiments.risk_aligned_online.forecast_replay import aligned_batch
    s=example()
    s['values'][40:]=123456
    s['valid'][40:]=False
    s['y'][:]=0
    e,_=aligned_batch([s],np.random.default_rng(7),8,model().cfg,'cpu')
    assert not (e['x']==123456).any()
    assert not (e['qx']==123456).any()
    assert not (e['truth']==123456).any()
    assert e['y'].eq(0).all()
