import numpy as np

from inspect_peak_contrast_first_alarm_20261106 import effect_decomposition, first


def test_ten_ms_is_late_and_post_disruption_crossings_are_ignored():
    times = np.array([0.89, 0.99, 1.00002])
    assert first(times, [-1, 1, 2], True, 1.0, 0.5) == {
        "first": 1,
        "lead_ms": 10.0,
        "timely": False,
        "false_alarm": False,
    }
    assert first(times, [-1, -1, 2], True, 1.0, 0.5)["first"] is None
    assert first(times, [-1, -1, 2], False, None, 0.5)["false_alarm"]


def test_relative_improvement_can_hide_an_absolute_drop():
    native = np.ones((1, 2, 32))
    z = np.broadcast_to(native[:, :, None, :], (1, 2, 3, 32)).copy()
    z[:, 0] -= 0.5
    delta = np.zeros((1, 2, 3))
    delta[:, 0] = -1.0
    out = effect_decomposition(native, z, delta, np.ones((1, 32), dtype=bool))
    assert out["raw_danger_safe_mean"] == [-0.5, 0.0]
    assert out["shape_danger_safe_mean"] == [0.5, 0.0]
    assert out["positive_paired_contrast"] == 1
