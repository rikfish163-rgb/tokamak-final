import numpy as np

from experiments.risk_aligned_online.event_prefix_probe import Event
from experiments.risk_aligned_online.event_temporal_teacher_20261103.pairs import (
    select_pairs,
)


def test_pairs_are_same_event_but_matched_control_is_not_and_future_query_is_later():
    times = np.arange(500) / 1000
    events = [Event(11, 1, 2, 0.150, 0.300)]
    rows, all_rows = select_pairs(times, events, 11, False, None, 25.0, {1: 7})
    assert 0 < len(rows) <= len(all_rows) <= 4
    for r in rows:
        o, j = r["origin"], r["other_origin"]
        assert 0.150 <= times[o] < times[o + 25] < 0.300
        assert not (0.150 <= times[j] and times[j + 25] < 0.300)
        assert abs(j - o) <= 50 and j >= 127
        assert r["query"] == o + 26 and r["maturity"] == o + 25
        assert r["event_id"] == 7 and r["disruptive"] is False
    assert select_pairs(times, events, 11, False, None, 25.0, {1: 7}) == (
        rows,
        all_rows,
    )


def test_unannotated_gap_is_not_a_new_class_and_hdf_timing_restricts_pairs():
    times = np.arange(350) / 1000
    events = [Event(11, 1, 3, 0.150, 0.320)]
    rows, all_rows = select_pairs(times, events, 11, True, 0.330, 40.0, {1: 7})
    for r in all_rows:
        lead = (0.330 - times[r["query"]]) * 1000
        assert round(lead, 9) > 10
        assert r["group"] == ("danger" if lead <= 40 + 1e-6 else "early")
        assert r["disruptive"] is True
    assert all(r["family"] == 3 for r in rows)
