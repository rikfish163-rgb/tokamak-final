import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_prefix_probe import Event
from experiments.risk_aligned_online.event_teaching_20261026.learning import (
    event_targets, shuffle_intervals, family_counts, event_objective, risk_and_features,
)


def test_event_targets_half_open_unknown_and_whole_event_weight():
    es=[Event(1,1,0,.1,.2),Event(1,2,1,.3,.4)]
    t=np.array([.099,.1,.15,.2,.3,.399,.4])
    r=event_targets(t,es,counts=np.array([2,1,1,1,1,1]))
    assert r['event_y'].tolist()==[-1,0,0,-1,1,1,-1]
    assert r['event_weight'][[0,3,6]].sum()==0
    assert float(r['event_weight'][1:3].sum())==pytest.approx(1/12)
    assert float(r['event_weight'][4:6].sum())==pytest.approx(1/6)


def test_shuffle_only_train_whole_intervals_and_preserve_outcome_strata():
    es={s:[Event(s,1,s%6,0.,1.),Event(s,2,(s+1)%6,1.,2.)] for s in range(20)}
    flags={s:bool(s%2) for s in range(12)}
    out=shuffle_intervals(es,flags,seed=131)
    assert set(out)==set(flags)
    assert out==shuffle_intervals(es,flags,seed=131)
    assert any(a.family!=b.family for s in flags for a,b in zip(es[s],out[s]))
    for flag in (False,True):
        a=sorted(e.family for s in flags if flags[s]==flag for e in es[s])
        b=sorted(e.family for s in flags if flags[s]==flag for e in out[s])
        assert a==b
    for s in out:
        assert [(e.shot,e.index,e.start,e.end) for e in es[s]]==[(e.shot,e.index,e.start,e.end) for e in out[s]]


def test_counts_use_only_intervals_present_in_sampled_endpoints():
    es={1:[Event(1,1,0,.1,.2),Event(1,2,1,.3,.4)],2:[Event(2,1,5,.1,.2)]}
    result=family_counts({1:np.array([.1,.15,.2]),2:np.array([.12])},es)
    assert result.tolist()==[1,0,0,0,0,1]
    with pytest.raises(ValueError,match='represented'):
        event_targets(np.array([.1]),es[1],counts=np.zeros(6,int))


def test_event_loss_unknown_points_no_grad_and_empty_is_zero():
    logits=torch.zeros(3,6,requires_grad=True)
    y=torch.tensor([0,-1,2]);w=torch.tensor([.5,0.,.5])
    loss=event_objective(logits,y,w)
    loss.backward()
    assert logits.grad[1].abs().sum()==0 and logits.grad[0].abs().sum()>0
    assert event_objective(logits,torch.full((3,),-1),torch.zeros(3))==0


def test_feature_tap_is_exact_risk_forward_and_preserves_gradient():
    from experiments.risk_aligned_online.tests.test_source_task_memory import setup
    from experiments.risk_aligned_online.direction_teaching_20261026.learning import make_adapter
    old,e=setup();m=make_adapter(old.base)
    m.memory.load_state_dict(old.memory.state_dict())
    d=torch.randn(4,92,requires_grad=True)
    before=m.risk(e['h'],e['valid'],d)
    z,h=risk_and_features(m,e['h'],e['valid'],d)
    assert torch.equal(before,z) and h.shape==(4,64)
    torch.testing.assert_close(m.base.risk_head(h).squeeze(-1),z,atol=0,rtol=0)
    head=torch.nn.Linear(64,6)
    event_objective(head(h),torch.tensor([0,1,2,3]),torch.ones(4)).backward()
    assert m.memory.output.weight.grad.abs().sum()>0 and d.grad.abs().sum()>0
    assert all(p.grad is None for p in m.base.parameters())
    assert len(m.base.risk_head._forward_pre_hooks)==0


def test_event_gradient_traverses_causal_write_and_base_stays_frozen():
    from experiments.risk_aligned_online.tests.test_source_task_memory import setup
    from experiments.risk_aligned_online.direction_teaching_20261026.learning import make_adapter
    from experiments.risk_aligned_online.event_teaching_20261026.learning import loss_from_example
    old,e=setup();m=make_adapter(old.base);m.memory.load_state_dict(old.memory.state_dict())
    e.update(labels=torch.ones(4),local_mask=torch.ones(4,dtype=torch.bool),
             direction_mask=torch.ones(4,dtype=torch.bool),event_y=torch.tensor([0,1,2,3]),event_weight=torch.ones(4)/24)
    _,event,_=loss_from_example(m,torch.nn.Linear(64,6),e,.7)
    event.backward()
    for n in ('key.weight','prior','rate','output.weight'):
        g=dict(m.memory.named_parameters())[n].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max()>0
    assert m.slow.weight.grad is None and all(p.grad is None for p in m.base.parameters())


def test_unknown_event_label_cannot_get_positive_weight():
    with pytest.raises(ValueError,match='unknown'):
        event_objective(torch.zeros(1,6),torch.tensor([-1]),torch.ones(1))


def test_training_audit_requires_coverage_and_actual_event_gradients():
    from copy import deepcopy
    from experiments.risk_aligned_online.event_teaching_20261026.audit import validate_training
    arms={k:dict(gradients={n:1. for n in ('key.weight','prior','rate','output.weight')},
                 head_gradient=None if k=='none' else 1., examples=[dict(shot=1),dict(shot=2)])
          for k in ('none','true','shuffled')}
    rows=[dict(epoch=e,ids=[1,2],arms=deepcopy(arms)) for e in (1,2)]
    assert validate_training(rows,[1,2])['complete_epochs']==2
    bad=deepcopy(rows);bad[0]['ids']=[1,1]
    with pytest.raises(ValueError,match='coverage'):
        validate_training(bad,[1,2])
    bad=deepcopy(rows);bad[0]['arms']['none']['head_gradient']=1.
    with pytest.raises(ValueError,match='no-event'):
        validate_training(bad,[1,2])
    bad=deepcopy(rows)
    for r in bad:r['arms']['true']['head_gradient']=0.
    with pytest.raises(ValueError,match='event head'):
        validate_training(bad,[1,2])


def test_event_checkpoint_contract_rejects_mismatches():
    from types import SimpleNamespace
    from experiments.risk_aligned_online.event_teaching_20261026.checkpoint import validate_metadata
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
    from experiments.risk_aligned_online.event_prefix_probe import LABEL_SHA256
    roles={'train':list(range(1353)),'val':list(range(1353,1513)),
           'test':list(range(1513,1733))}
    a=dict(profile='event_teaching_v1',kind='true',feedback='progressive',epoch=2,
           readout='full',source_sha256=PARENT_SHA256,train_ids=roles['train'],
           val_ids=roles['val'],event_sha256=LABEL_SHA256)
    m=SimpleNamespace(feedback='progressive',readout='full')
    validate_metadata(a,m,roles)
    for k,v in [('profile','legacy'),('kind','bag'),('feedback','delayed'),('epoch',1),
                ('readout','shape'),('source_sha256','bad'),('event_sha256','bad')]:
        with pytest.raises(ValueError):validate_metadata(dict(a,**{k:v}),m,roles)


def test_three_arm_protocol_rejects_alarm_or_checkpoint_confounds():
    from copy import deepcopy
    from experiments.risk_aligned_online.event_teaching_20261026.compare import validate_protocols
    protocols=[dict(threshold=.7,feedback='progressive',source_deployment_feedback_match=True,
                    source_memory=dict(kind=k,profile='event_teaching_v1',epoch=2,
                                       source_protocol_sha256='a',audit_sha256='b'))
               for k in ('none','true','shuffled')]
    validate_protocols(protocols)
    bad=deepcopy(protocols);bad[1]['threshold']=.8
    with pytest.raises(ValueError,match='threshold'):validate_protocols(bad)
    bad=deepcopy(protocols);bad[2]['source_memory']['audit_sha256']='changed'
    with pytest.raises(ValueError,match='matched'):validate_protocols(bad)
    with pytest.raises(ValueError,match='three'):validate_protocols(protocols[:2])
