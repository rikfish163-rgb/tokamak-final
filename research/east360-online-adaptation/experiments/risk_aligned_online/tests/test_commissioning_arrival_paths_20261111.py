import numpy as np

from experiments.risk_aligned_online import commissioning_arrival_paths_20261111 as paths


def test_confirmation_is_on_last_of_three_points_not_on_first():
    high = np.array([False, True, True, True, True, False, True, True, True])[:, None]
    np.testing.assert_array_equal(np.flatnonzero(paths.confirmed_edges(high)[:, 0]), [3, 8])
    assert not paths.confirmed_edges(high[:3]).any()


def test_direction_and_action_clock_are_distinct_from_presence():
    edges = [np.array([150]), np.array([156]), np.array([190]), np.array([190])]
    found = paths.path_finishes(edges)
    assert found['magnetic_then_position'].tolist() == [156]
    assert not len(found['position_then_magnetic'])
    assert not len(found['radiation_then_voltage']) and not len(found['voltage_then_radiation'])
    assert found['radiation_voltage_same_tick'].tolist() == [190]
    assert paths.active_at(found['magnetic_then_position'], np.array([156,157,188,189])).tolist() == [False,True,True,False]


def test_events_are_invariant_to_unseen_suffix_and_absent_on_flat_signal():
    x = np.zeros((280, 10))
    available = np.ones_like(x, dtype=bool)
    x[180:, 0] = 1.
    x[195:, 3] = 1.
    a = paths.arrivals(x, available)
    b = paths.arrivals(x[:205], available[:205])
    assert a[:, 0].any() and a[:, 1].any() and not a[:, 2:].any()
    np.testing.assert_array_equal(a[:205], b)
    assert not a[:180].any()


def test_recent_predecessor_cannot_be_more_than_one_release_interval_old():
    edges = [np.array([153]), np.array([186]), np.array([], dtype=int), np.array([], dtype=int)]
    assert not len(paths.path_finishes(edges)['magnetic_then_position'])
