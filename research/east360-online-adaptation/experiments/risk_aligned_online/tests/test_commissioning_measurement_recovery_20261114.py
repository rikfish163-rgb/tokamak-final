import numpy as np

from experiments.risk_aligned_online import commissioning_measurement_recovery_20261114 as run


def example(scale=1.):
    n = 600
    x = np.ones((n,23),np.float32)
    x[:,run.IP] = np.linspace(1e5,3e5,n)
    x[:,run.BT] = 1.9
    x[:,run.QA] = .68775*x[:,run.BT]/x[:,run.IP]
    return dict(values=x,available=np.ones_like(x,dtype=bool)),np.full(n,scale)


def test_isolated_gain_can_be_recovered_using_only_arrived_pairs():
    signal,g = example()
    g[128:320]=1.3
    g[320:]=.8
    damage = run.inject(signal,g)
    estimate = run.estimate(damage,.68775)
    assert estimate[128:144].tolist()==[1.]*16
    np.testing.assert_allclose(estimate[160:320],1.3,rtol=1e-6,atol=0.)
    np.testing.assert_allclose(estimate[352:],.8,rtol=1e-6,atol=0.)
    clean = run.repair(damage,estimate)
    np.testing.assert_allclose(clean['values'][160:320],signal['values'][160:320],rtol=1e-6,atol=0.)
    np.testing.assert_array_equal(clean['available'],signal['available'])
    for n in (21,142,203,329,350):
        sub = {k:v[:n] for k,v in damage.items()}
        np.testing.assert_array_equal(estimate[:n],run.estimate(sub,.68775))


def test_irreducible_common_gain_cannot_be_seen_from_one_identity():
    signal,g = example()
    g[128:]=1.8
    shared = run.inject(signal,g,common_mode=True)
    estimate = run.estimate(shared,.68775)
    np.testing.assert_array_equal(estimate,1.)
    np.testing.assert_allclose(shared['values'][128:,run.BT]/signal['values'][128:,run.BT],g[128:])
    static = run.reconstruct(shared,.68775)
    np.testing.assert_allclose(static['values'][:,run.BT],shared['values'][:,run.BT],rtol=1e-6)


def test_one_gain_control_commits_once_after_an_observed_change():
    signal,g = example()
    g[128:320]=1.3
    g[320:]=.8
    damaged = run.inject(signal,g)
    once = run.estimate(damaged,.68775,fixed=True)
    assert (once[:144]==1.).all()
    assert 1.<once[144]<1.3
    np.testing.assert_array_equal(once[144:],once[144])
    np.testing.assert_array_equal(run.estimate(signal,.68775,fixed=True),1.)


def test_gain_clock_has_two_changes_with_rank_balanced_order():
    g0,g1 = (run.gain_clock(900,i) for i in (20,21))
    np.testing.assert_array_equal(np.flatnonzero(g0[1:]!=g0[:-1])+1,[128,256])
    np.testing.assert_array_equal(np.flatnonzero(g1[1:]!=g1[:-1])+1,[128,256])
    np.testing.assert_array_equal(g0[:128],1.)
    assert g0[128]==g1[256] and g0[256]==g1[128]
