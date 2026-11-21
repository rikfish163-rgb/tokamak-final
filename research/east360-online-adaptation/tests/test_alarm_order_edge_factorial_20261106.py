import numpy as np

from alarm_order_edge_factorial_20261106 import decomposition, outer_swap


def test_only_swaps_band_edges_and_preserves_terminal_and_values():
    v = np.broadcast_to(np.arange(128)[:, None], (128, 23)).copy()
    a = np.ones_like(v, dtype=bool)
    x = outer_swap(v, a, 11, 14)
    np.testing.assert_array_equal(x[88:96], v[112:120])
    np.testing.assert_array_equal(x[96:112], v[96:112])
    np.testing.assert_array_equal(x[120:], v[120:])
    np.testing.assert_array_equal(np.sort(x[:, 0]), np.arange(128))
    a[88, 0] = False
    assert outer_swap(v, a, 11, 14) is None


def test_factorial_reports_direct_effect_and_interaction_not_double_counted():
    out = decomposition([(2.0, 1.6, 1.9, 1.7), (0.0, 0.4, -0.1, 0.3)])
    np.testing.assert_allclose(out["median_abs"]["full"], 0.4)
    np.testing.assert_allclose(out["median_abs"]["outer"], 0.3)
    np.testing.assert_allclose(out["median_abs"]["inner"], 0.1)
    np.testing.assert_allclose(out["median_abs"]["interaction"], 0.1)
    np.testing.assert_allclose(out["abs_residual_outer_over_full"], 0.25)
