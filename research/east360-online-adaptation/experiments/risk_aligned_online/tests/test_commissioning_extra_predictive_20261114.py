import numpy as np

from experiments.risk_aligned_online import commissioning_extra_predictive_20261114 as run


def test_current_extra_packet_can_be_issued_without_its_later_target():
    x = np.arange(600*23,dtype=float).reshape(-1,23)
    ok = np.ones_like(x,dtype=bool)
    dose = np.ones(len(x))
    q = np.arange(127,len(x)-26,32)
    e = np.ones(len(q),bool)
    p = np.full(len(x),.4)
    full = run.make_issues(x,ok,p,q,e,dose)
    n = int(full['release'][3])+1
    cut = q+26<n
    past = run.make_issues(x[:n],ok[:n],p[:n],q[cut],e[cut],dose[:n])
    for k in ('q','release','extra','earlier','core','available'):
        np.testing.assert_array_equal(past[k],full[k][:len(past['q'])])
    assert past['release'][-1]==n-1
    y, mature = run.targets(x[:n],ok[:n],past['release'])
    assert len(y)==len(mature)==len(past['release'])
    assert not mature[-1]


def test_extra_control_requires_both_packets_and_do_not_count_future():
    x = np.tile(np.arange(500,dtype=float)[:,None],(1,23))
    ok = np.ones_like(x,dtype=bool)
    d = np.ones(len(x))
    q = np.arange(127,len(x)-26,32)
    p = np.full(len(x),.4)
    ok[q[0]+4,run.EXTRA[1]] = False
    u = run.make_issues(x,ok,p,q,np.ones(len(q),bool),d)
    assert not u['available'][0,1]
    assert u['available'][0,0]
    assert u['available'][1].all()
    np.testing.assert_array_equal(u['extra'][1,0],[3.,8.,13.,18.,23.])
    np.testing.assert_array_equal(u['extra'][1,0],u['earlier'][1,0])


def test_conditional_fit_only_adds_path_information_not_held_out_target():
    r = np.random.default_rng(13)
    x = r.normal(size=(40,4))
    e = r.normal(size=(40,5))
    y = x[:,0,None]-2*e[:,1,None]
    groups = np.repeat(np.arange(4),10)
    a,b,c = run.fit_pair(x,e,y,groups)
    old,new = run.eval_models(x,e,e[:,::-1],e*0.,a,b,c)[:2]
    assert np.square(new-y).mean()<np.square(old-y).mean()/10


def test_stable_response_fit_cannot_learn_from_its_own_forecast_target():
    fit = dict(feature_energy=np.ones(1),local_prior=6.)
    r = np.array([166,198,230,262])
    y = np.ones((4,15))
    z = np.zeros_like(y)
    u = np.arange(4.,dtype=float)[:,None]
    m = np.ones(4,bool)
    a,b = run.local_update(u,u*0.,z,z,y,r,m,m,fit)
    altered = y.copy()
    altered[2:] = 1.e5
    c,d = run.local_update(u,u*0.,z,z,altered,r,m,m,fit)
    np.testing.assert_array_equal(a[:3],c[:3])
    np.testing.assert_array_equal(b[:3],d[:3])
    assert (a[-1]!=c[-1]).any()


def test_selection_waits_for_scored_packet_and_expires_old_evidence():
    release = np.array([10,42,74,106,138,500])
    available = np.array([41,75,105,137,169,526])
    hold = np.ones(len(release))
    latest = np.array([0.,0.,0.,4.,4.,4.])
    out = run.prequential_gate(release,available,hold,latest)
    assert out.tolist()==[False,False,False,True,False,False]
    # The third forecast's result has not matured at the third issue.
    latest[2] = 30.
    changed = run.prequential_gate(release,available,hold,latest)
    np.testing.assert_array_equal(changed[:3],out[:3])
    assert not changed[3]
