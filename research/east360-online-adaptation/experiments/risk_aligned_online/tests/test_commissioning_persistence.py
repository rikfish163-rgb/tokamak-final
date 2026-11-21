import numpy as np

from experiments.risk_aligned_online.commissioning_persistence_20261111 import (
    FIRST_EFFECT, location_scale, local_reference, replay,
)


def episode(xs):
    level = np.broadcast_to(np.asarray(xs, dtype=np.float32)[:, None], (len(xs), 10)).copy()
    valid = np.ones_like(level, bool)
    q = 127 + 32*np.arange(len(xs))
    return level, valid, q, np.zeros((len(xs), 3)), np.ones(len(xs), bool)


def test_adjacent_agreement_not_just_a_large_current_error():
    reference = (np.zeros(10), np.ones(10))
    a = episode([1, -1, 1, 1])
    out, _ = replay(*a, a[2][-1]+32, reference)
    assert not out['sustained'][:FIRST_EFFECT].any()
    assert not out['sustained'][FIRST_EFFECT].any()
    assert out['spaced'][FIRST_EFFECT].min() > 0
    assert out['sustained'][FIRST_EFFECT+32].min() > 0
    assert not out['spaced'][FIRST_EFFECT+32].any()
    np.testing.assert_allclose(np.linalg.norm(out['sustained'], axis=1),
                               np.linalg.norm(out['current_equal_norm'], axis=1), rtol=2e-7)


def test_causal_replay_is_prefix_invariant_and_absent_channel_breaks_continuity():
    reference = (np.zeros(10), np.ones(10))
    level, valid, q, context, usable = episode([.8]*6)
    valid[3, 0] = False
    args = level, valid, q, context, usable
    a, _ = replay(*args, q[-1]+31, reference)
    level[-1] = -9
    prefix = q[-1]+8
    b, _ = replay(*args, prefix, reference)
    for k in a:
        np.testing.assert_array_equal(a[k][:prefix], b[k])
        start = FIRST_EFFECT + (64 if k == 'lagged' else 32)
        assert not a[k][start:start+64, 0].any()
    assert a['sustained'][FIRST_EFFECT+32, 1] > 0.


def test_reference_does_not_use_current_or_either_recent_window():
    x, m, q, u, ok = episode([.1, .2, 2., 3., 9.])
    p = (np.zeros(10), np.ones(10))
    a = local_reference(x, m, u, ok, 4, p)
    x[2:] *= -4.
    b = local_reference(x, m, u, ok, 4, p)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert 0. < a[0][0] < .2


def test_normal_prior_ignores_missing_values():
    x, m, *_ = episode([.1, .1, 4.])
    m[-1] = False
    loc, scale = location_scale(x, m)
    np.testing.assert_allclose(loc, .1)
    assert (scale > 0).all()
