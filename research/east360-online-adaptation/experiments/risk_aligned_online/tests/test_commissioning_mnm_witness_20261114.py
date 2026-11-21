import numpy as np

from experiments.risk_aligned_online import commissioning_mnm_witness_20261114 as m


def traces(n):
    step = np.arange(n)[:, None]
    v = .001 * step + np.arange(1., len(m.COLUMNS)+1.)
    return v, np.ones_like(v, bool), np.linspace(.1, .9, n)


def test_handoff_exposes_features_but_not_the_future_target():
    v, valid, risk = traces(232)
    issue = m.FIRST
    r = issue + m.RELEASE
    short = m.issued(v[:r+1], valid[:r+1], risk[:r+1])
    full = m.issued(v, valid, risk)
    for k in ('q', 'r', 'features', 'true', 'available', 'level', 'reversed'):
        np.testing.assert_array_equal(short[k], full[k][:len(short['q'])])
    assert short['q'].tolist() == [issue]
    np.testing.assert_equal(m.mature(v[:r+26], valid[:r+26], short['r'])[1], [False])
    np.testing.assert_equal(m.mature(v[:r+27], valid[:r+27], short['r'])[1], [True])


def test_mode_value_never_replaces_shared_activity_or_trailing_support():
    v, valid, risk = traces(250)
    start = m.FIRST+1
    v[start, m.MIR[0]] = 0.
    z = m.issued(v, valid, risk)
    assert not z['available'][0]
    assert z['any_live'][0]
    assert z['available'][1]
    assert z['earlier_at'][1] == -1
    open_ = m.issued(*traces(250))
    assert open_['earlier_at'][1] == open_['q'][0]+25
    actual = open_['true'][0]
    np.testing.assert_allclose(actual[::-1], open_['reversed'][0])
    assert np.ptp(actual) > 0.


def test_fit_weights_independent_shots_not_repeated_windows():
    x = np.array([[0.], [1.], [2.]])
    y = np.array([[3.], [1.], [0.]])
    a = m.model(x, y, np.array([1, 1, 2]))
    repeated = [0, 1, 0, 1, 2]
    b = m.model(x[repeated], y[repeated], np.array([1, 1, 1, 1, 2]))
    np.testing.assert_allclose(m.predict(x, a), m.predict(x, b), atol=1e-12)
