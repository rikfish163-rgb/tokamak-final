import numpy as np

from experiments.risk_aligned_online import commissioning_growth_signature_20261112 as growth


def test_directional_growth_changes_sign_for_damping_not_for_global_sign():
    direction = np.ones((2, len(growth.GROUPS)))
    step = np.array([[.3, 1., 2.], [2., 1., .3]])[:, :, None] * direction[:, None, :]
    common, signed, _ = growth.signature(direction, step, np.array([1, 1]))
    neg_common, neg_signed, _ = growth.signature(-direction, -step, np.array([1, 1]))
    np.testing.assert_allclose(signed[0], -signed[1])
    assert np.all(signed[0] > 0.)
    np.testing.assert_allclose(neg_signed, signed)
    np.testing.assert_allclose(neg_common, common)


def test_wrong_order_keeps_endpoint_and_all_non_ordered_features():
    rng = np.random.default_rng(17)
    before = rng.normal(size=(9, len(growth.GROUPS)))
    step = rng.normal(size=(9, 3, len(growth.GROUPS)))
    shift = np.resize([1, 2], len(step))
    moved = growth.permute_steps(step, shift)
    np.testing.assert_allclose(moved.sum(1), step.sum(1), atol=1e-12)
    np.testing.assert_array_equal(np.sort(step, axis=1), np.sort(moved, axis=1))
    common, true, sham = growth.signature(before, step, shift)
    common2, reversed_true, _ = growth.signature(before, moved, shift)
    np.testing.assert_allclose(common2, common, atol=1e-12)
    np.testing.assert_allclose(reversed_true, sham, atol=1e-12)
    assert not np.allclose(true, sham)


def test_straight_ramp_is_not_growth_or_decay():
    before = np.broadcast_to(np.arange(1., len(growth.GROUPS)+1.), (5, len(growth.GROUPS)))
    step = np.repeat(before[:, None], 3, axis=1)
    _, real, sham = growth.signature(before, step, np.ones(5, dtype=int))
    np.testing.assert_allclose(real, 0.)
    np.testing.assert_allclose(sham, 0.)


def test_release_excludes_unavailable_or_unmatured_points():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(290, len(growth.GROUPS)))
    mask = np.ones_like(z, dtype=bool)
    r = np.arange(growth.FIRST, len(z), growth.STRIDE)
    full = growth.observed(z, mask, r, rank=25)
    end = r[2]
    prefix = growth.observed(z[:end], mask[:end], r[r < end], rank=25)
    for key in prefix:
        np.testing.assert_array_equal(prefix[key], full[key][r < end])
    missing = mask.copy()
    missing[r[1]-1, 0] = False
    got = growth.observed(z, missing, r, rank=25)
    np.testing.assert_array_equal(got['usable'], np.arange(len(r)) != 1)
    assert not got['true'][1].any()


def test_next_block_has_no_current_or_cross_gap_access():
    r = growth.FIRST + growth.STRIDE*np.array([0, 1, 3, 4, 5, 6])
    true = np.broadcast_to(np.arange(6.)[:, None], (6, 4))
    null = 2.*true
    usable = np.array([True, True, True, False, True, True])
    ready, p = growth.next_estimate(r, usable, true, null)
    np.testing.assert_array_equal(ready, [False, True, False, False, False, True])
    np.testing.assert_array_equal(p['past_true'][ready, 0], [0., 4.])
    np.testing.assert_array_equal(p['past_wrong'][ready, 0], [0., 8.])


def test_shot_memory_predicts_before_assimilating_current_observation():
    prior = np.array([.2, -.2])
    x = np.ones((6, 2))
    y = np.c_[np.full(6, .7), np.full(6, -.5)]
    valid = np.array([False, True, True, False, True, True])
    p = growth.regressed(x, y, valid, prior)
    np.testing.assert_allclose(p[1], prior)
    expected = (y[1]+growth.MEMORY_PRIOR*prior)/(1.+growth.MEMORY_PRIOR)
    np.testing.assert_allclose(p[2], expected)
    altered = y.copy()
    altered[2:] = -2.
    np.testing.assert_array_equal(growth.regressed(x, altered, valid, prior)[:3], p[:3])
    # Pair 3 is unavailable and must not enter the state at step 4.
    expected = (y[1]+y[2]+growth.MEMORY_PRIOR*prior)/(2.+growth.MEMORY_PRIOR)
    np.testing.assert_allclose(p[4], expected)
