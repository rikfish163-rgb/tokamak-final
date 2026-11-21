import numpy as np
import pytest

from probe_pre_first_training_windows_20261106 import candidate_rows


def test_positives_precede_native_first_and_respect_clock_and_horizon():
    t = np.arange(200) / 1000
    z = np.full(200, -2.0)
    z[[129, 157]] = 1.0
    slot = np.maximum(-1, np.arange(200) // 32 - 2)
    s = np.array([True, False, True, True, True])
    first, timely, rows = candidate_rows(t, z, slot, s, 0.177, 92, 0.5)
    assert (first, timely) == (129, True)
    assert [(i, q.tolist()) for i, q in rows] == [(0, list(range(85, 96))), (2, [128])]


def test_miss_and_late_do_not_enable_out_of_window_or_unchanged_points():
    t = np.arange(128) / 1000
    z = np.full(128, -1.0)
    z[[104, 119]] = 0.6
    slot = np.maximum(-1, np.arange(128) // 32 - 2)
    first, timely, rows = candidate_rows(t, z, slot, np.ones(3, bool), 0.110, 60, 0.5)
    assert (first, timely) == (104, False)
    assert [q.tolist() for _, q in rows] == [list(range(64, 96)), list(range(96, 100))]
    z[104] = -1
    a, b, c = candidate_rows(t, z, slot, np.ones(3, bool), 0.110, 60, 0.5)
    assert a is None and not b
    assert all(np.array_equal(p, q) for (_, p), (_, q) in zip(rows, c, strict=True))


def test_earlier_native_alarm_outside_canonical_window_remains_timely():
    t = np.arange(128) / 1000
    z = np.full(128, -1.0)
    z[5] = 0.5
    slot = np.maximum(-1, np.arange(128) // 32 - 2)
    assert candidate_rows(t, z, slot, np.ones(3, bool), 0.120, 30, 0.5) == (5, True, [])


def test_probes_reject_unbound_or_misaligned_slots():
    with pytest.raises(ValueError, match="slots"):
        candidate_rows(
            np.arange(3) / 1000,
            np.zeros(3),
            np.array([-1, 0, 1]),
            np.ones(1, bool),
            0.120,
            30,
            0.5,
        )
