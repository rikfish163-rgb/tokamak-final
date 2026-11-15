import numpy as np

from experiments.risk_aligned_online.commissioning_lag_prequential_20261111 import (
    prequential_views,
)


def test_prediction_of_first_window_is_scored_before_any_update():
    q = np.array([127, 159])
    basis = np.eye(70, dtype=np.float32)
    aligned = np.stack((basis[1], basis[1]))
    shifted = -aligned
    a, audit = prequential_views(199, q, aligned, shifted, np.ones(2, bool), .1*basis[0])
    for arm, v in a.items():
        assert v.shape == (199, 18)
        np.testing.assert_array_equal(v[:185], a['fixed'][:185])
        np.testing.assert_array_equal(v[:153], 0.)
        assert audit['updates'].get(arm, 2) == 2
    assert a['real'][185, :6].sum() > a['fixed'][185, :6].sum() > a['shifted'][185, :6].sum()
    assert a['lag'][185, :6].sum() == a['real'][185, :6].sum()
    assert audit['true_order_wins']['fixed'] == 0
    assert audit['true_order_wins']['real'] == 1


def test_common_prefix_and_missing_breaks_persistence_without_update():
    q = 127 + np.arange(4)*32
    prior = np.eye(70, dtype=np.float32)[0]
    r = np.stack((prior, -prior, -prior, -prior))
    s = -r
    ok = np.array([True, True, False, True])
    a, audit = prequential_views(256, q, r, s, ok, prior)
    prefix, _ = prequential_views(213, q[:2], r[:2], s[:2], ok[:2], prior)
    for k in a:
        np.testing.assert_array_equal(a[k][:213], prefix[k])
        np.testing.assert_array_equal(a[k][217:249], 0.)
        np.testing.assert_array_equal(a[k][249:, 12:], 0.)
    assert audit['observed_blocks'] == 3
    assert audit['updates'] == {'real': 3, 'shifted': 3, 'lag': 3}
