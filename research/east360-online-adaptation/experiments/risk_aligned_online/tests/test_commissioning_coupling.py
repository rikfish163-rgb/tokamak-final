import numpy as np

from experiments.risk_aligned_online.commissioning_coupling_20261111 import (
    activity, alignment, observed_features, outcome_endpoints,
)


def test_only_alignment_direction_changes_under_reflection():
    path = np.zeros((1, 8, 4))
    path[0, 2, 0] = 1.
    path[0, 3, 1] = 1.
    valid = np.ones_like(path, bool)
    valid[0, 0, 2] = False
    sym, order = alignment(path, valid)
    reflection = alignment(path[:, ::-1], valid[:, ::-1])
    assert order[0, 0] > .1  # family 0 moves before family 1 at lag one
    np.testing.assert_allclose(sym, reflection[0], atol=1e-12)
    np.testing.assert_allclose(order, -reflection[1], atol=1e-12)
    empty = alignment(path, np.zeros_like(valid))
    assert not any(x.any() for x in empty)


def test_masked_values_cannot_change_activity_and_future_prefix_is_unused():
    rng = np.random.default_rng(19)
    z = rng.normal(size=(210, 10))
    valid = rng.uniform(size=z.shape) > .15
    ctx = rng.normal(size=(210, 2))
    cm = np.ones_like(ctx, bool)
    ends = np.array([97, 160])
    original = observed_features(z, valid, ctx, cm, ends)
    altered = np.where(valid, z, 1e20)
    changed = observed_features(altered[:161], valid[:161], ctx[:161], cm[:161], ends)
    for k in original:
        np.testing.assert_array_equal(original[k], changed[k])


def test_family_activity_uses_valid_channel_count_not_zero_fill():
    z = np.zeros((72, 10))
    valid = np.ones_like(z, bool)
    valid[:, 1:3] = False
    z[8:, 0] = 3.
    a, available = activity(z, valid, np.array([71]))
    assert available.all()
    assert a[0, 0, 0] == np.log1p(3.)
    assert np.count_nonzero(a) == 1


def test_outcome_anchor_selects_strictly_before_30ms_and_not_outside_fixed_bins():
    times = np.arange(300) * .001
    ends = np.arange(153, 246)
    found = outcome_endpoints(times, ends, np.ones_like(ends, bool), .250)
    np.testing.assert_array_equal(found, [205, 175, 153])
    no_options = outcome_endpoints(times, np.array([220, 222]), [True, True], .250)
    assert not len(no_options)
