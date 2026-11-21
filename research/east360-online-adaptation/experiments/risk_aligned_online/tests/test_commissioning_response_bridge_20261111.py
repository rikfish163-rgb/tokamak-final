import numpy as np

from experiments.risk_aligned_online import commissioning_response_bridge_20261111 as bridge


def test_issued_view_includes_incomplete_future_and_is_prefix_invariant():
    rng = np.random.default_rng(41)
    z = rng.normal(size=(230, 10))
    m = np.ones_like(z, dtype=bool)
    phys = rng.normal(size=(len(z), 2))
    pm = np.ones_like(phys, dtype=bool)
    full = bridge.prefix_views(z, m, phys, pm)
    short = bridge.prefix_views(z[:187], m[:187], phys[:187], pm[:187])
    assert short['issue'].tolist() == [153, 185]
    selected = full['issue'] < 187
    for k in short:
        np.testing.assert_array_equal(short[k], full[k][selected])
    take, target = bridge.realized(z[:187], m[:187], short)
    assert take.tolist() == [0]
    assert target.shape == (1, 25, 4)


def test_response_is_radial_not_a_claim_about_disruption():
    ref = np.zeros((2, 10))
    tail = np.ones((2, 10))
    future = np.broadcast_to(np.array([.5, 2.])[:, None, None], (2, 25, 10))
    response = bridge.radial_response(ref, tail, future)
    assert np.all(response[0] < 0.) and np.all(response[1] > 0.)
    np.testing.assert_array_equal(response, bridge.radial_response(ref, tail, -future))


def test_reversing_the_two_bins_leaves_same_view_except_for_order_terms():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(155, 10))
    m = np.ones_like(z, dtype=bool)
    c = np.ones((len(z), 2))
    a = bridge.prefix_views(z, m, c, c.astype(bool))
    swapped = z.copy()
    swapped[128:144] = np.r_[z[136:144], z[128:136]]
    b = bridge.prefix_views(swapped, m, c, c.astype(bool))
    for k in ('same', 'match', 'ref', 'tail', 'issue'):
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(a['order'], -b['order'])


def test_duplicating_one_shots_rows_does_not_increase_its_fit_weight():
    x = np.array([[0.], [1.], [2.], [3.]])
    y = np.array([0., 1., 5., 3.])[:, None, None]
    y = np.broadcast_to(y, (4, 25, 4))
    first = bridge.fit_ridge(x, y, np.array([1, 1, 2, 2]))
    j = [0, 1, 0, 1, 2, 3]
    repeat = bridge.fit_ridge(x[j], y[j], np.array([1, 1, 1, 1, 2, 2]))
    np.testing.assert_allclose(bridge.predict_ridge(x, first), bridge.predict_ridge(x, repeat))
