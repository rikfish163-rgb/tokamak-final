import numpy as np

from analyze_alarm_order_attribution_20261106 import independent_inputs, point_report


def test_independent_input_support_excludes_mask_mismatch_only_for_touched_blocks():
    v = np.broadcast_to(np.arange(220, dtype=np.float32)[:, None], (220, 23))
    a = np.ones_like(v, dtype=bool)
    a[88, 3] = False
    z = independent_inputs(v, a, np.array([126, 127]), np.zeros(23), np.ones(23))
    assert not z["supported"][0].any()
    # Point 88 belongs to patch 11 at index 127; full recent reverse needs equality.
    np.testing.assert_array_equal(z["supported"][1], [False, True, True, True])
    assert np.isnan(z["motion"][1, 0])
    assert np.all(z["motion"][1, 1:] > 0)


def test_point_coverage_distinguishes_strict_reasonable_and_inside125():
    t = np.array([0, 0.898, 0.97, 0.977])
    q = dict(
        times=t,
        first=np.zeros(4, dtype=bool),
        near=np.array([False, True, True, True]),
        logits=np.full((4, 5), -2.0),
        supported=np.ones((4, 4), dtype=bool),
    )
    q["logits"][[0, 2], 1] = 2.0
    q["logits"][1, 2] = 2.0
    d = point_report(q, True, 1.0, tau=1.0)
    assert (
        d["strict"]["up"]["recent_reverse"] and d["reasonable"]["up"]["recent_reverse"]
    )
    assert not d["inside125"]["up"]["recent_reverse"]
    assert d["inside125"]["up"]["old_reverse"]
    assert d["inside125"]["n"] == 1  # Exactly 30 ms is excluded.
    assert d["reasonable_near"]["n"] == 1
    assert point_report(q, False, None, 1.0)["all"]["up"]["recent_reverse"]
