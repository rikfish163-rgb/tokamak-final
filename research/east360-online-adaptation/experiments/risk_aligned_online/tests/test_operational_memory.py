"""Operational full-trajectory protection, including the actual intra path."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.operational_memory import (
    OperationalMemory, operational_state, fullcurve_guard, task_gradient)


def shot(sid=1, label=True, size=230):
    rng = np.random.default_rng(sid)
    return dict(shot=sid, device='EAST', horizon=125., is_disrupt=label,
        down=(size-1)/1000. if label else None, times=np.arange(size)/1000.,
        valid=np.ones(size, bool), values=rng.normal(size=(size,23)).astype(np.float32),
        available=np.ones((size,23), bool), _sealed_scores=np.full(size,.4,np.float32))


def test_strict_30ms_not_a_125ms_training_cap():
    s=shot(size=4200); p=np.zeros(4200); p[0]=.8; p[-31]=.99
    state=operational_state(s,p,.7)
    assert state['query']==0 and state['timely']
    assert state['lead_ms']>3000 and state['early_over_3000']


def test_30ms_equal_is_not_timely_and_invalid_endpoints_do_not_alarm():
    s=shot(); p=np.zeros(230); p[199]=.99; p[0]=.9; s['valid'][0]=False
    state=operational_state(s,p,.7)
    assert not state['timely'] and state['query']!=199
    assert state['lead_ms']==30.


def test_missing_eligible_disruptive_time_is_explicit_not_removed():
    s=shot(size=20); state=operational_state(s,np.full(20,.9),.7)
    assert state['query'] is None and not state['timely']


def test_guard_catches_new_peak_at_a_previously_ignored_time():
    s=shot(label=False); old=np.full(230,.1); old[0]=.6
    new=old.copy();new[0]=.5;new[180]=.8
    result=fullcurve_guard([s], {1:old}, {1:new}, .7)
    assert not result['passed'] and result['lost_tn']==[1]


def test_guard_catches_lost_timely_alarm_but_does_not_ban_rescue():
    s=shot();old=np.zeros(230);old[100]=.8;new=np.zeros(230);new[210]=.9
    assert fullcurve_guard([s],{1:old},{1:new},.7)['lost_tp']==[1]
    assert fullcurve_guard([s],{1:new},{1:old},.7)['passed']


@pytest.mark.parametrize('bad',['nan','length','threshold'])
def test_invalid_curve_is_rejected(bad):
    s=shot();p=np.full(230,.4);threshold=.7
    if bad=='nan':p[0]=np.nan
    elif bad=='length':p=p[:-1]
    else:threshold=1.
    with pytest.raises(ValueError):operational_state(s,p,threshold)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.memory=torch.nn.Parameter(torch.zeros(2))
        self.decoder=torch.nn.Linear(1,1);self.requires_grad_(False)
        self.center=torch.zeros(23)


def make_learner(monkeypatch, fail=False, reject=False):
    m=ToyModel();rule=OperationalMemory(m,bank={},intra=True)
    monkeypatch.setattr(rule,'_predict',lambda m,s:np.full(len(s['times']),.4,np.float32))
    monkeypatch.setattr(rule,'_source_guard',lambda m,t:dict(passed=not reject))
    def proposal(*a,**kw):
        with torch.no_grad():m.memory[0]=.03
        if fail:raise FloatingPointError('test failure after mutation')
        return dict(gradient_norm=1.,task_records=[],source={})
    monkeypatch.setattr(rule,'_proposal',proposal)
    # Unchanged toy curves intentionally fail the strict operational descent.
    return m,rule


def context():
    rows=[shot(i,bool(i%2)) for i in range(1,32)]
    p=SimpleNamespace(pending=None,ended_ids=[s['shot'] for s in rows],reencode=lambda s:s)
    return rows,p


@pytest.mark.parametrize('bad',['pending','future','duplicate','unsealed'])
def test_invalid_history_rejected_before_any_mutation(monkeypatch,bad):
    m,rule=make_learner(monkeypatch);rows,p=context()
    if bad=='pending':p.pending=True
    elif bad=='future':p.ended_ids=p.ended_ids[:-1]
    elif bad=='duplicate':rows[-1]['shot']=rows[-2]['shot']
    else:del rows[-1]['_sealed_scores']
    with pytest.raises(ValueError):rule(m,rows[-1],rows[:-1],provider=p,threshold=.7)
    assert not m.memory.any() and rule.last_index==0


def test_rejected_candidate_rolls_back_and_uses_one_not_generic_four_steps(monkeypatch):
    import json
    m,rule=make_learner(monkeypatch);rows,p=context()
    row=rule(m,rows[-1],rows[:-1],provider=p,threshold=.7,steps=4)
    assert not row['accepted'] and row['actual_steps']==1
    assert not m.memory.any() and not any(p.requires_grad for p in m.parameters())
    assert len(rule.cache)==31 and rule.last_index==31
    json.dumps(row,allow_nan=False)  # The durable stream audit must accept it.
    with pytest.raises(ValueError):rule(m,rows[-1],rows[:-1],provider=p,threshold=.7)


def test_exception_restores_memory_history_rng_and_raises(monkeypatch):
    m,rule=make_learner(monkeypatch,fail=True);rows,p=context()
    rng=repr(rule.rng.bit_generator.state)
    with pytest.raises(FloatingPointError):rule(m,rows[-1],rows[:-1],provider=p,threshold=.7)
    assert not m.memory.any() and rule.cache=={} and rule.last_index==0
    assert repr(rule.rng.bit_generator.state)==rng
    assert not any(p.requires_grad for p in m.parameters())


def test_actual_causal_intra_gradient_reaches_only_internal_memory():
    from experiments.risk_aligned_online.simple23 import Config,Simple23
    from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel
    from experiments.risk_aligned_online.ended_context_rule import causal_peak_logit
    from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        torch.manual_seed(131)
        m=DecoderFfnModel(Simple23(Config(fast_site='tokens',fast_rule='normalized'),
                                  torch.zeros(23),torch.ones(23))).eval().requires_grad_(False)
        s=shot();curves=[predict_shot(m,s,intra=True)[0]]
        q=180  # Two genuinely mature updates, not an unadapted early maximum.
        z,info=causal_peak_logit(m,s,q)
        assert float(z.sigmoid().detach())==pytest.approx(float(curves[0][q]),abs=2e-6)
        m.memory.requires_grad_(True)
        gradient,records=task_gradient(m,[(s,q)],threshold=.7,intra=True)
        assert torch.isfinite(gradient).all() and gradient.norm()>0
        assert len(records)==1 and records[0]['query']==q
        assert records[0]['differentiated_updates']>0
        assert not any(p.grad is not None for p in m.decoder.parameters())
        assert not any(p.grad is not None for p in m.base.parameters())
    finally:torch.set_num_threads(previous)


def test_inner_update_memory_gradient_matches_local_finite_difference():
    from experiments.risk_aligned_online.simple23 import Config,Simple23
    from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel
    from experiments.risk_aligned_online.ended_context_rule import causal_peak_logit
    import math
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        torch.manual_seed(131)
        m=DecoderFfnModel(Simple23(Config(fast_site='tokens',fast_rule='normalized'),
                                  torch.zeros(23),torch.ones(23))).double().eval().requires_grad_(False)
        m.memory.requires_grad_(True);s=shot();threshold=.7;q=180
        g,records=task_gradient(m,[(s,q)],threshold=threshold,intra=True)
        assert records[0]['differentiated_updates']==records[0]['accepted_updates']>0
        direction=g/g.norm();epsilon=1e-4;risks=[]
        # All two accepted steps are within the unroll, so this toy trajectory
        # has no truncated burn-in. This tests the actual second-order route,
        # not a claim that the long-shot truncated gradient is exact.
        for sign in (-1,1):
            with torch.no_grad():m.memory.copy_(sign*epsilon*direction)
            z,_=causal_peak_logit(m,s,q,create_graph=False)
            risks.append(float(torch.nn.functional.softplus(
                -(z-math.log(threshold/(1-threshold)))).detach()))
        derivative=(risks[1]-risks[0])/(2*epsilon)
        assert derivative==pytest.approx(float(g.norm()),rel=1e-4,abs=1e-5)
    finally:torch.set_num_threads(previous)


def test_accepted_candidate_is_retained_and_cached_at_new_memory(monkeypatch):
    m,rule=make_learner(monkeypatch);rows,p=context()
    def predict(m,s):
        # Candidate improves the D maximum without moving any safety curve.
        value=.4+float(m.memory[0]) if s['is_disrupt'] else .4
        return np.full(len(s['times']),value,np.float32)
    monkeypatch.setattr(rule,'_predict',predict)
    row=rule(m,rows[-1],rows[:-1],provider=p,threshold=.7)
    assert row['accepted'] and row['task_gain']>row['task_tolerance']
    assert float(m.memory[0])==pytest.approx(.03)
    assert rule.cache[31]['scores'][0]==pytest.approx(.43)
    assert row['memory_before_sha256']!=row['memory_after_sha256']


@pytest.mark.parametrize('change',['head','memory','input','threshold'])
def test_stale_cache_is_never_silently_reused(monkeypatch,change):
    m,rule=make_learner(monkeypatch);rows,p=context()
    rule(m,rows[-1],rows[:-1],provider=p,threshold=.7)
    rows.append(shot(32,False));p.ended_ids.append(32)
    threshold=.7
    if change=='head':
        with torch.no_grad():m.decoder.bias.add_(1.)
    elif change=='memory':
        with torch.no_grad():m.memory.add_(1.)
    elif change=='input':rows[2]['values'][0,0]+=1.
    else:threshold=.8
    with pytest.raises(ValueError,match='cache|coordinates'):
        rule(m,rows[-1],rows[1:-1],provider=p,threshold=threshold)
    assert rule.last_index==31
