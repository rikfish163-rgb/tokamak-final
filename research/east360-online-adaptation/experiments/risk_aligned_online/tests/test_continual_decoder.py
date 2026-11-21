"""Timing, causal history and retained optimizer state for the rolling head."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel
from experiments.risk_aligned_online.continual_decoder import RollingDecoder


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(131)
    return DecoderFfnModel(Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                                   torch.zeros(23), torch.ones(23))).eval()


def shot(sid, label):
    rng = np.random.default_rng(sid)
    return dict(shot=sid, device='EAST', is_disrupt=label, horizon=125.,
        down=.229 if label else None, values=rng.normal(size=(230,23)).astype(np.float32),
        available=np.ones((230,23),bool), valid=np.ones(230,bool),
        times=np.arange(230)/1000., _sealed_scores=np.linspace(.1,.9,230))


def provider(records):
    return SimpleNamespace(pending=None, ended_ids=[s['shot'] for s in records],
                           reencode=lambda s:s)


def fake_pair(m, pair, optimizer, *, boundary, mode):
    optimizer.zero_grad(set_to_none=True)
    loss = m.decoder.bias.square().mean()
    loss.backward()
    optimizer.step()
    return dict(records=[dict(shot=s['shot'],query=q,label=int(s['is_disrupt'])) for s,q in pair],
                objective=float(loss.detach()), risk_gradient_norm=1.)


def learner(m, monkeypatch, mode='early_continual'):
    import experiments.risk_aligned_online.continual_decoder as module
    monkeypatch.setattr(module, 'train_pair', fake_pair)
    rule = RollingDecoder(m, mode=mode)
    monkeypatch.setattr(rule, '_prepare_shot', lambda s, **kw:(s,180))
    return rule


def test_waits_for_two_naturally_revealed_classes_without_updating(monkeypatch):
    m = model(); rule = learner(m,monkeypatch)
    s = shot(1,True)
    before = {k:v.clone() for k,v in m.decoder.state_dict().items()}
    row = rule(m,s,[],provider=provider([s]),threshold=.7)
    assert not row['accepted'] and row['reason']=='waiting_for_both_ended_classes'
    for k,v in m.decoder.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_optimizer_retained_between_ended_shots_and_current_always_in_pair(monkeypatch):
    m = model(); rule = learner(m,monkeypatch)
    history = [shot(1,True),shot(2,False)]
    a = rule(m,history[-1],history[:-1],provider=provider(history),threshold=.7,steps=99)
    optimizer = rule.optimizer
    assert a['actual_steps']==4 and a['effective_from_index']==3
    history.append(shot(3,True))
    b = rule(m,history[-1],history[:-1],provider=provider(history),threshold=.7)
    assert rule.optimizer is optimizer and b['optimizer_steps_after']==8
    assert all(3 in [r['shot'] for r in row['records']] for row in b['records'])
    assert all(len({r['label'] for r in row['records']})==2 for row in b['records'])


def test_coordinate_transition_resets_head_and_optimizer_then_exact40(monkeypatch):
    m = model(); rule = learner(m,monkeypatch)
    first = [shot(1,True),shot(2,False)]
    rule(m,first[-1],first[:-1],provider=provider(first),threshold=.7)
    old = rule.optimizer
    history = [shot(i,bool(i%2)) for i in range(1,31)]
    with torch.no_grad():m.decoder.bias.add_(10.)
    row = rule(m,history[-1],history[:-1],provider=provider(history),threshold=.7)
    assert row['coordinate_transition_reset'] and rule.optimizer is not old
    assert row['actual_steps']==40 and row['optimizer_steps_after']==40
    assert m.decoder.bias.abs().max()<1.


@pytest.mark.parametrize('mode,expected', [('early_fixed',False),('late_continual',True),('early_continual',True)])
def test_declared_post30_learning_schedule(monkeypatch,mode,expected):
    m=model();rule=learner(m,monkeypatch,mode)
    history=[shot(i,bool(i%2)) for i in range(1,32)]
    bootstrap=history[:30]
    rule(m,bootstrap[-1],bootstrap[:-1],provider=provider(bootstrap),threshold=.7)
    row=rule(m,history[-1],history[:-1],provider=provider(history),threshold=.7)
    assert row['accepted']==expected


@pytest.mark.parametrize('violation',['pending','future','not_sealed'])
def test_rejects_unended_or_future_history(monkeypatch,violation):
    m=model();rule=learner(m,monkeypatch)
    history=[shot(1,True),shot(2,False)];p=provider(history)
    if violation=='pending':p.pending=object()
    elif violation=='future':p.ended_ids=[1]
    else:del history[-1]['_sealed_scores']
    with pytest.raises(ValueError,match='sealed'):
        rule(m,history[-1],history[:-1],provider=p,threshold=.7)


def test_failure_restores_head_optimizer_and_freezes_then_raises(monkeypatch):
    import experiments.risk_aligned_online.continual_decoder as module
    m=model();rule=learner(m,monkeypatch)
    history=[shot(1,True),shot(2,False)]
    rule(m,history[-1],history[:-1],provider=provider(history),threshold=.7)
    before={k:v.clone() for k,v in m.decoder.state_dict().items()}
    def fail(*args,**kwargs):
        with torch.no_grad():m.decoder.bias.add_(100.)
        raise FloatingPointError('tested nonfinite proposal')
    monkeypatch.setattr(module,'train_pair',fail)
    history.append(shot(3,True))
    with pytest.raises(FloatingPointError):
        rule(m,history[-1],history[:-1],provider=provider(history),threshold=.7)
    for k,v in m.decoder.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert not any(p.requires_grad for p in m.parameters())
    assert rule.optimizer_steps==4


def test_replay_keeps_current_and_only_last32_ended_shots(monkeypatch):
    m=model();rule=learner(m,monkeypatch)
    first=[shot(i,bool(i%2)) for i in range(1,31)]
    rule(m,first[-1],first[:-1],provider=provider(first),threshold=.7)
    all_history=[shot(i,bool(i%2)) for i in range(1,41)]
    row=rule(m,all_history[-1],all_history[-33:-1],provider=provider(all_history),threshold=.7)
    assert row['history_ids']==list(range(9,41))
    assert all(40 in [r['shot'] for r in step['records']] for step in row['records'])
