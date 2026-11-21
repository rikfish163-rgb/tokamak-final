import numpy as np

from alarm_order_margin_audit_20261106 import measure


def test_dense_peak_can_fall_between_grid_points_and_support_is_not_uplift():
    t = np.arange(200) / 1000
    p = np.full(200, 0.05)
    p[61] = 0.6
    p[100] = 0.99  # Too late given the true DownTime.
    anchor = dict(
        times=t[[32, 64, 96]],
        index=np.array([32, 64, 96]),
        first=np.zeros(3, dtype=bool),
        logits=np.array([[-2.0, -1.8], [-2.0, -1.85], [-2.0, np.nan]]),
        supported=np.array([[True], [True], [False]]),
    )
    a = measure(t, p, 0.128, 0.725, anchor)
    assert a["dense_peak"] == 0.6 and a["dense_best_index"] == 61
    assert a["grid_peak"] == 0.05
    assert a["dense_near"] and not a["grid_near"]
    assert np.isclose(a["largest_sampled_uplift"], 0.2)
    assert a["measured_uplift_below_dense_gap"]
