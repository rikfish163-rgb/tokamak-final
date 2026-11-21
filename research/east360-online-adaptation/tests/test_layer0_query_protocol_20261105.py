import numpy as np
import pytest

from layer0_query_protocol_20261105 import (
    choose_query,
    guard_violations,
    query_progress,
    reveal_block,
    teaching_indices,
)


def make_peak(value, pre=0.01):
    a = np.full(200, 0.01)
    a[2], a[150] = pre, value
    return a


def test_query_chooses_actionable_disagreement_then_nearness_then_rank():
    shots = [
        (21, 1, make_peak(0.71), make_peak(0.72)),
        (22, 2, make_peak(0.70), make_peak(0.79)),
        (23, 3, make_peak(0.90, pre=0.8), make_peak(0.71, pre=0.8)),
        (24, 4, make_peak(0.72), make_peak(0.76)),
        (25, 5, make_peak(0.72), make_peak(0.76)),
    ]
    decision, scores = choose_query(shots, 0.725)
    assert decision == 4
    assert [s["shot"] for s in scores] == [1, 2, 3, 4, 5]
    assert scores[2]["prefix_alarm"]


def test_reveal_needs_durable_commitment_and_reads_only_committed_ids(tmp_path):
    p = tmp_path / "commit.json"
    calls = []

    def oracle(domain, sid):
        calls.append((domain, sid))
        return False, None

    with pytest.raises(FileNotFoundError):
        reveal_block(p, [7], oracle)
    p.write_text('{"selected":[7],"committed_ns":0}')
    with pytest.raises(ValueError, match="different selection"):
        reveal_block(p, [8], oracle)
    assert calls == []
    receipt = reveal_block(p, [7], oracle)
    assert receipt["outcomes"][0]["shot"] == 7
    assert calls == [("EAST", 7)]


def test_guard_preserves_capped_positive_even_when_still_strictly_timely():
    t = np.arange(4000) / 1000
    old, new = np.zeros_like(t), np.zeros_like(t)
    old[3600], new[150] = 0.8, 0.8
    assert guard_violations(t, old, new, True, 3.8, 0.72) == ["lost_30_to_3000"]
    assert guard_violations(t, old, new, False, None, 0.72) == []
    assert guard_violations(t, new * 0, new, False, None, 0.72) == ["new_false_alarm"]


def test_query_progress_needs_help_for_a_current_error_without_harming_other_arm():
    t = np.arange(250) / 1000
    old = {"inter": np.full(250, 0.76), "both": np.full(250, 0.15)}
    new = {"inter": np.full(250, 0.735), "both": np.full(250, 0.16)}
    ok, _ = query_progress(t, old, new, False, None, 0.725)
    assert ok
    new["both"][170] = 0.8
    assert not query_progress(t, old, new, False, None, 0.725)[0]


def test_teaching_points_obey_causal_write_and_strict_positive_clock():
    t = np.arange(300) / 1000
    a = np.linspace(0.9, 0.1, len(t))
    b = a[::-1].copy()
    q = teaching_indices(t, {"inter": a, "both": b}, True, 0.29, 0.725)
    assert len(q) == 8
    leads = np.round((0.29 - t[q]) * 1000, 9)
    assert (q >= 135).all() and (leads > 30).all() and (leads <= 125).all()
    q = teaching_indices(t, {"inter": a, "both": b}, False, None, 0.725)
    assert len(q) >= 8 and (q >= 135).all()
