import numpy as np

from prealarm_history_support_20261106 import available_next, risk_roles


def test_updates_only_affect_endpoints_after_all_required_data_arrives():
    m = np.ones((1200, 23), dtype=bool)
    a = available_next(m)
    assert {k: int(np.flatnonzero(v)[0]) for k, v in a.items()} == {
        "native_padded": 1,
        "native128": 128,
        "extra1x16": 144,
        "extra4x16": 192,
        "extra4x256": 1152,
        "pair2x16": 32,
        "pair2x32": 64,
    }
    m[70:] = False
    b = available_next(m[:90])
    for k in a:
        np.testing.assert_array_equal(a[k][:71], b[k][:71])


def test_two_half_windows_require_two_identically_available_diagnostic_families():
    m = np.zeros((34, 23), dtype=bool)
    m[:16, [3, 15]] = True
    m[16:32, [15, 20]] = True
    assert not available_next(m)["pair2x16"][32]
    m[16:24, 3] = True
    a = available_next(m)["pair2x16"]
    assert a[32] and not a[:32].any()


def test_source_window_is_not_relabelled_to_target_cutoff():
    t = np.array([0.874, 0.875, 0.94, 0.970, 0.975, 0.980, 0.990, 1.0, 1.001])
    x = risk_roles(t, True, 1.0, 25, 10)
    np.testing.assert_array_equal(np.flatnonzero(x["timely_positive"]), [4, 5])
    np.testing.assert_array_equal(np.flatnonzero(x["east_like_negative"]), [1, 2])
    assert not x["east_like_positive"].any()
    y = risk_roles(t, True, 1.0, 125, 30)
    np.testing.assert_array_equal(np.flatnonzero(y["east_like_positive"]), [1, 2])
    assert not y["safe"].any()
