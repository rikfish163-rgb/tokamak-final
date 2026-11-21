import json

import numpy as np

from analyze_deep_miss_history_20261106 import (
    describe,
    independent_support,
    matched_safety,
    point_roles,
)


def test_history_windows_are_before_short_window_and_stale_loses_a_full_block():
    a = np.ones((1300, 23), dtype=bool)
    i = np.array([127, 382, 383, 639, 895, 1151])
    r, s, coverage = independent_support(a, i)
    np.testing.assert_array_equal(r.sum(1), [0, 0, 1, 2, 3, 3])
    np.testing.assert_array_equal(s.sum(1), [0, 0, 0, 1, 2, 3])
    np.testing.assert_array_equal(coverage, np.ones(len(i)))
    a[:256] = False
    r1, s1, _ = independent_support(a, i)
    assert r1[-1].all() and not s1[-1, -1]


def test_label_and_warning_windows_are_not_collapsed_into_one_target():
    t = np.array([0.001, 0.874, 0.875, 0.9, 0.97, 0.977, 1.02])
    p = np.array([0.2, 0.5, 0.6, 0.7, 0.9, 0.95, 0.2])
    d = point_roles(t, p, True, 1.0, 125.0, 30.0, 0.725)
    np.testing.assert_array_equal(d["positive"], [0, 0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(d["reasonable"], [1, 1, 1, 1, 0, 0, 0])
    assert not d["safe_high"].any()
    s = point_roles(t, p, False, None, 125.0, 30.0, 0.725)
    np.testing.assert_array_equal(s["safe_high"], [0, 0, 1, 1, 1, 1, 0])


def test_summary_keeps_equal_shot_and_equal_seed_weight():
    sample = [
        (4, np.tile([1.0, 2.0, 3.0, 4.0], (3, 1)), 5.0),
        (1, np.tile([3.0, 2.0, 5.0, 4.0], (3, 1)), 1.0),
    ]
    out = describe(sample)
    assert out["shots"] == 2 and out["anchors"] == 5
    assert out["mean"] == dict(short=2.0, real=2.0, stale=4.0, scrambled=4.0)
    assert out["native"] == 3.0


def test_matched_safety_counter_is_serializable_by_strict_receipt_writer():
    positive = [(1, np.array([0.1, 5.0, 0.8]), np.array([0.5, 0.2, 0.1]))]
    negative = [(2, np.array([[0.1, 5.0, 0.8]]), np.array([[0.2, 0.1, 0.0]]))]
    payload = json.loads(json.dumps(matched_safety(positive, negative), allow_nan=False))
    assert payload["n_pairs"] == payload["real_higher"] == 1
