import numpy as np

from experiments.risk_aligned_online.commissioning_source_order_probe_20261110 import trajectory_shape


def test_shape_distinguishes_growth_from_return_at_same_total_activity():
    y = np.linspace(-.05, -.5, 25)[:, None]
    m = np.ones_like(y, bool)
    forward, reversed_ = trajectory_shape(y, m), trajectory_shape(y[::-1], m)
    np.testing.assert_allclose(forward[0], reversed_[0])
    assert forward[1] < 0. < reversed_[1]
    assert forward[2] > 0. > reversed_[2]


def test_mask_matched_order_null_never_changes_amplitude_or_reads_missing_values():
    y = np.arange(50, dtype=float).reshape(25, 2) / 50
    m = np.ones_like(y, bool)
    m[[1, 8, 17], 0] = False
    m[-9:, 1] = False
    y[~m] = np.nan
    a = trajectory_shape(y, m)
    b = trajectory_shape(y, m, np.random.default_rng(1).permutation(len(y)))
    np.testing.assert_allclose(a[:2], b[:2])
    assert a[2] != b[2] and a[4] != b[4]
    np.testing.assert_array_equal([a[3], a[5], b[3], b[5]], 0.)
