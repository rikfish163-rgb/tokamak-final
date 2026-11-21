import numpy as np
import pytest

from experiments.risk_aligned_online.ffn_intra_20261005.source import make_episode, split_shots


def fake(sid=1,disrupt=True,n=1000):
    return dict(shot=sid,device='J-TEXT',values=np.zeros((n,23),np.float32),
        available=np.ones((n,23),bool),times=np.arange(n)/1000.,
        is_disrupt=disrupt,down=.99 if disrupt else None, valid=np.ones(n,bool))


def test_four_matured_events_precede_timely_source_query():
    shot=fake()
    e=make_episode(shot,np.random.default_rng(1))
    assert len(e['origins'])==4
    assert np.diff(e['origins']).tolist()==[26]*3
    assert e['queries'][0]==e['origins'][-1]+26
    assert e['queries'][-1] <= e['origins'][-1]+51
    assert all(x%26==0 for x in e['origins'])
    assert e['burn_in_origins']==list(range(0,e['origins'][0],26))
    leads=(shot['down']-shot['times'][e['queries']])*1000
    assert np.all(np.round(leads,9)>30) and np.all(np.round(leads,9)<=125)
    assert e['label']==1


def test_short_shot_not_padded_with_future_and_target_not_allowed():
    assert make_episode(fake(n=200),np.random.default_rng(1)) is None
    s=fake(); s['device']='EAST'
    with pytest.raises(ValueError):
        make_episode(s,np.random.default_rng(1))


def test_train_only_disjoint_split_is_reproducible():
    shots=[fake(i,i%2==0) for i in range(40)]
    first=split_shots(shots,seed=131)
    second=split_shots(shots,seed=131)
    assert [[s['shot'] for s in part] for part in first] == [[s['shot'] for s in part] for part in second]
    assert set(s['shot'] for s in first[0]).isdisjoint(s['shot'] for s in first[1])
    assert all(any(s['is_disrupt']==label for s in part) for part in first for label in (0,1))
