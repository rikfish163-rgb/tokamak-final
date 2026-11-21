import numpy as np

from experiments.risk_aligned_online.centered_gate_stream_20261101.timing_bounds import (
    bound_row,
)


def test_safe_alarm_before_first_mutable_index_is_locked_but_at160_is_not():
    times = np.arange(300) / 1000
    curve = np.zeros(300)
    curve[159] = 0.9
    assert bound_row(times, curve, False, None, 0.7, fixed=False)["locked_FP"]
    curve[159], curve[160] = 0, 0.9
    assert not bound_row(times, curve, False, None, 0.7, fixed=False)["locked_FP"]
    assert bound_row(times, curve, False, None, 0.7, fixed=True)["locked_FP"]


def test_disruption_with_remaining_timely_endpoints_can_only_be_potentially_rescued():
    times = np.arange(300) / 1000
    curve = np.zeros(300)
    curve[172] = 0.9
    row = bound_row(times, curve, True, 0.2, 0.7, fixed=False)
    assert (
        row["can_be_TP"]
        and not row["actual_TP"]
        and row["future_timely_endpoints"] == 10
    )
    assert not bound_row(times, curve, True, 0.2, 0.7, fixed=True)["can_be_TP"]


def test_prefix_late_alarm_and_no_future_timely_endpoint_cannot_be_rescued():
    times = np.arange(300) / 1000
    curve = np.zeros(300)
    curve[155] = 0.9
    row = bound_row(times, curve, True, 0.18, 0.7, fixed=False)
    assert not row["actual_TP"] and not row["can_be_TP"]
    assert row["future_timely_endpoints"] == 0


def test_already_timely_prefix_is_protected_by_immutable_past_even_without_future():
    times = np.arange(300) / 1000
    curve = np.zeros(300)
    curve[140] = 0.9
    row = bound_row(times, curve, True, 0.18, 0.7, fixed=False)
    assert row["actual_TP"] and row["can_be_TP"] and row["locked_TP"]
