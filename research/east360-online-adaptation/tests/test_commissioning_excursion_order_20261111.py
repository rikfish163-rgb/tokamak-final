import numpy as np

from experiments.risk_aligned_online.commissioning_excursion_order_20261111 import (
    describe_excursion, disjoint_pairs, mature_views, pair_fit,
)


def test_exact_order_control_preserves_endpoints_and_unordered_evidence():
    ref = np.zeros((1, 10))
    start = np.ones_like(ref)
    following = np.stack((.95*start, .60*start, .55*start), axis=1)
    baseline, temporal, amplitude = describe_excursion(ref, start, following)
    b, s, a = describe_excursion(ref, start, following[:, [1, 0, 2]])
    np.testing.assert_array_equal(baseline, b)
    np.testing.assert_array_equal(amplitude, a)
    np.testing.assert_array_equal(temporal, -s)
    assert (temporal < 0).all()


def test_maturing_an_extra_window_does_not_change_previous_views():
    rng = np.random.default_rng(131)
    x = rng.normal(size=(220, 10))
    m = np.ones_like(x, bool)
    prefix = mature_views(x[:185], m[:185])
    full = mature_views(x, m)
    assert prefix[0].tolist() == [153]
    for a, b in zip(prefix, full, strict=True):
        np.testing.assert_array_equal(a, b[:len(a)])
    assert len(mature_views(x[:153], m[:153])[0]) == 0
    assert mature_views(x[:154], m[:154])[0].tolist() == [153]


def test_invalid_future_samples_are_never_used_by_either_order():
    x = np.arange(1800, dtype=float).reshape(180, 10)
    mask = np.ones_like(x, bool)
    mask[138, 2] = False  # second 8-point follow-up block of q=127
    assert len(mature_views(x, mask)[0]) == 0


def test_match_is_disjoint_and_prefers_cardinality_to_lowest_pair_distance():
    pos = np.array([[.05], [-.45], [.2]])
    safe = np.array([[0.], [.48], [.1]])
    m = disjoint_pairs(pos, safe, np.array([1, 2, 1]), np.array([3, 4, 3]), np.array([.50]))
    assert len(m) == 2
    assert set(np.array([1, 2, 1])[m[:, 0]]) == {1, 2}
    assert set(np.array([3, 4, 3])[m[:, 1]]) == {3, 4}
    assert (np.abs(pos[m[:, 0]]-safe[m[:, 1]]) <= .50).all()


def test_fit_supports_positive_order_without_fabricating_energy_effect():
    a = np.array([[0., .4], [0., .7], [0., .6], [0., .5]])
    b = np.array([[0., -.4], [0., -.7], [0., -.6], [0., -.5]])
    w = pair_fit(a, b)
    assert w[0] == 0. and w[1] > 0.
    assert ((a-b)@w > 0).all() and (-(a-b)@w < 0).all()
