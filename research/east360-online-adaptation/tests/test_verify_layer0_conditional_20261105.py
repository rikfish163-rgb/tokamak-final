import numpy as np

from verify_layer0_conditional_20261105 import failure_headroom


def test_actual_headroom_distinguishes_magnitude_and_wrong_time():
    clock = np.arange(300) / 1000
    ref = np.full(300, 0.05)
    ref[180] = 0.6
    candidate = ref.copy()
    candidate[200] = 0.25
    result = failure_headroom(clock, ref, candidate, 0.29, 0.7)
    assert result["mechanism"] == "shift_at_other_endpoint"
    assert abs(result["gap"] - 0.1) < 1e-9
    assert (
        failure_headroom(clock, ref, ref + 0.03, 0.29, 0.7)["mechanism"]
        == "insufficient_shift"
    )
    candidate[180] = 0.8
    assert failure_headroom(clock, ref, candidate, 0.29, 0.7)["mechanism"] == "crossed"


def test_early_cannot_be_counted_as_endpoint_rescue():
    clock = np.arange(300) / 1000
    a = np.zeros_like(clock)
    b = a.copy()
    b[133], b[270] = 0.8, 0.8
    out = failure_headroom(clock, a, b, 0.29, 0.7)
    assert out["mechanism"] == "insufficient_shift"
    assert out["max_shift"] == 0.0
