import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_recurrence_core_20261114 import (
    find_precedents,
)


GROUPS = np.array([-1, 0, 1, 2, 3])


def search(q, key, ok, **kwargs):
    return find_precedents(
        np.asarray(q, dtype=np.int64),
        np.asarray(key, dtype=np.float64),
        GROUPS,
        np.asarray(ok, dtype=bool),
        stride=kwargs.pop('stride', 1),
        horizon=kwargs.pop('horizon', 0),
        recent_slots=kwargs.pop('recent_slots', 100),
        **kwargs,
    )


@pytest.mark.parametrize('n', [0, 1, 2])
def test_empty_and_short_shots_have_no_records(n):
    result = search(np.arange(n), np.zeros((n, 5)), np.ones(n, dtype=bool))
    for values in result[:3]:
        assert values.shape == (4, n)
        assert values.dtype == np.int64
        assert np.all(values == -1)
    for values in result[3:]:
        assert values.shape == (4, n)
        assert values.dtype == np.float64
        assert np.all(np.isposinf(values))


def test_weighted_leave_group_out_and_condition_only_match():
    q = [0, 1, 2]
    key = np.array([[0, 10, 0, 0, 0], [2, 0, 4, 0, 0], [0, 0, 0, 0, 0]])
    weights = np.array([2., 1., 3., 1., 1.])
    matched, blind, displaced, distance, own = search(q, key, [1, 1, 1], weights=weights)

    assert matched[0, 2] == 0
    assert matched[1, 2] == 1
    assert np.all(blind[:, 2] == 0)
    assert displaced[0, 2] == 1
    assert displaced[1, 2] == 0
    assert distance[0, 2] == 0.
    assert distance[1, 2] == pytest.approx(8. / 5.)
    assert own[0, 2] == pytest.approx(100.)
    assert own[1, 2] == pytest.approx(16.)
    assert search(q, key, [1, 1, 1])[3][1, 2] == pytest.approx(1.)


def test_maturity_recent_window_and_73_point_separation_are_inclusive():
    q = np.array([0, 27, 28, 29, 72, 73, 74, 100, 101, 128])
    key = np.ones((q.size, 5))
    key[[0, 2, 6, 8, 9], :] = 0.
    ok = np.ones(q.size, dtype=bool)
    ok[5] = False
    result = search(q, key, ok, horizon=24, recent_slots=100, min_separation=73)
    matched, blind, displaced = result[:3]

    # q=28 is exactly 73 points before q=101; q=29 is one point too close.
    # q=0 is outside the inclusive 100-point recent window.
    assert np.all(matched[:, 8] == 2)
    assert np.all(blind[:, 8] == 2)
    assert np.all(displaced[:, 8] == 1)
    # At q=128, q=28 is exactly on the recent-window boundary.
    assert np.all(matched[:, 9] == 2)
    assert np.all(displaced[:, 9] == 3)
    # The original default permits q=74: its target matured at q=99.
    assert np.all(search(q, key, ok, horizon=24, recent_slots=100)[0][:, 8] == 6)

    for i in range(q.size):
        candidates = [j for j in range(q.size) if ok[j]
                      and q[j] + 25 <= q[i]
                      and q[j] >= q[i] - 100
                      and q[i] - q[j] >= 73]
        if not ok[i] or len(candidates) < 2:
            assert all(np.all(part[:, i] == -1) for part in result[:3])
            assert all(np.all(np.isposinf(part[:, i])) for part in result[3:])
        else:
            for group in range(4):
                a, b, c = (part[group, i] for part in result[:3])
                assert a in candidates and b in candidates and c in candidates
                assert a != c
                assert c == candidates[(candidates.index(a) + len(candidates) // 2)
                                       % len(candidates)]


def test_target_maturity_is_horizon_plus_one_and_requires_two_candidates():
    q = [0, 1, 25, 26]
    matched, blind, displaced, _, _ = search(q, np.zeros((4, 5)), [1] * 4, horizon=24)
    assert np.all(matched[:, :3] == -1)
    assert np.all(matched[:, 3] == 1)
    assert np.all(blind[:, 3] == 1)
    assert np.all(displaced[:, 3] == 0)


@pytest.mark.parametrize('count, selected, rotated', [
    (2, 0, 1), (3, 0, 1), (3, 2, 0),
    (4, 0, 2), (4, 3, 1), (5, 4, 1),
])
def test_displacement_rotates_the_selected_position_for_odd_and_even_counts(
        count, selected, rotated):
    key = np.ones((count + 1, 5))
    key[[selected, count], :] = 0.
    matched, blind, displaced, _, _ = search(np.arange(count + 1), key, [True] * (count + 1))
    assert np.all(matched[:, count] == selected)
    assert np.all(blind[:, count] == selected)
    assert np.all(displaced[:, count] == rotated)


@pytest.mark.parametrize('extra, expected', [(5e-13, 1), (2e-12, 0)])
def test_relative_distance_ties_prefer_the_latest(extra, expected):
    key = np.zeros((3, 5))
    key[0, 0] = 1.
    key[1, 0] = np.sqrt(1. + extra)
    matched, blind, _, _, _ = search([0, 1, 2], key, [1, 1, 1])
    assert np.all(matched[:, 2] == expected)
    assert np.all(blind[:, 2] == expected)


def test_a_small_nonzero_distance_is_not_tied_with_zero():
    key = np.zeros((3, 5))
    key[1, 0] = 1e-12
    matched, blind, _, _, _ = search([0, 1, 2], key, [1, 1, 1])
    assert np.all(matched[:, 2] == 0)
    assert np.all(blind[:, 2] == 0)


def test_held_out_group_cannot_influence_its_match_or_match_distance():
    key = np.array([[0., 10., 0., 0., 0.], [2., 0., 4., 0., 0.],
                    [0., 0., 0., 0., 0.]])
    before = search([0, 1, 2], key, [1, 1, 1])
    key[:, 1] = [200., -70., -2.]
    after = search([0, 1, 2], key, [1, 1, 1])
    for index in (0, 1, 2, 3):
        np.testing.assert_array_equal(before[index][0], after[index][0])
    assert before[4][0, 2] != after[4][0, 2]


def test_future_prefix_and_query_ok_are_causally_independent_of_donor_ok():
    rng = np.random.default_rng(1411)
    q = np.cumsum(rng.integers(1, 6, size=30)).astype(np.int64)
    key = rng.normal(size=(q.size, 5))
    donor_ok = rng.random(q.size) > .2
    query_ok = np.ones(q.size, dtype=bool)
    weights = np.array([1., 2., 3., 4., 5.])
    args = dict(stride=5, horizon=4, recent_slots=20, min_separation=7,
                weights=weights)
    before = search(q, key, donor_ok, query_ok=query_ok, **args)
    i = 18
    assert np.all(before[0][:, i] >= 0)

    changed_ok = donor_ok.copy()
    changed_ok[i:] = ~changed_ok[i:]
    changed_q = q.copy()
    changed_q[i + 1:] += 1000
    changed_key = key.copy()
    changed_key[i + 1:] = rng.normal(size=changed_key[i + 1:].shape) * 100
    after = search(changed_q, changed_key, changed_ok, query_ok=query_ok, **args)
    prefix = search(q[:i + 1], key[:i + 1], donor_ok[:i + 1],
                    query_ok=query_ok[:i + 1], **args)
    for a, b, c in zip(before, after, prefix):
        np.testing.assert_array_equal(a[:, :i + 1], b[:, :i + 1])
        np.testing.assert_array_equal(a[:, :i + 1], c)

    # Explicit query validity permits emission even when this row is not a donor.
    no_current_donor = donor_ok.copy()
    no_current_donor[i] = False
    emitted = search(q, key, no_current_donor, query_ok=query_ok, **args)
    assert np.all(emitted[0][:, i] >= 0)
    assert np.all(search(q, key, no_current_donor, **args)[0][:, i] == -1)
    query_ok[i] = False
    suppressed = search(q, key, donor_ok, query_ok=query_ok, **args)
    assert np.all(suppressed[0][:, i] == -1)


def test_explicit_query_validity_never_makes_incomplete_donors_eligible():
    key = np.zeros((5, 5))
    donor_ok = [True, False, True, True, False]
    result = search([0, 1, 2, 3, 4], key, donor_ok, query_ok=np.ones(5, bool))
    assert np.all(result[0][:, 4] == 3)
    assert np.all(result[2][:, 4] == 0)
    # Disabling the newest mature donor changes later queries, not its own.
    changed = donor_ok.copy()
    changed[3] = False
    later = search([0, 1, 2, 3, 4], key, changed, query_ok=np.ones(5, bool))
    for old, new in zip(result, later):
        np.testing.assert_array_equal(old[:, 3], new[:, 3])
    assert np.all(later[0][:, 4] == 2)


def test_changed_donor_becomes_relevant_only_when_its_target_matures():
    q = np.arange(10)
    key = np.zeros((q.size, 5))
    query_ok = np.ones(q.size, dtype=bool)
    donor_ok = np.ones(q.size, dtype=bool)
    before = search(q, key, donor_ok, horizon=2, query_ok=query_ok)
    donor_ok[5] = False
    after = search(q, key, donor_ok, horizon=2, query_ok=query_ok)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a[:, :8], b[:, :8])
    assert np.all(before[0][:, 8] == 5)
    assert np.all(after[0][:, 8] == 4)


def test_group_without_columns_reports_undefined_own_distance():
    q = np.arange(3, dtype=np.int64)
    result = find_precedents(q, np.zeros((3, 1)), np.array([-1]), np.ones(3, bool),
                             stride=1, horizon=0, recent_slots=10)
    assert np.all(result[0][:, 2] == 1)
    assert np.all(result[3][:, 2] == 0.)
    assert np.all(np.isposinf(result[4][:, 2]))


def test_rejects_negative_min_separation_and_mismatched_query_mask():
    with pytest.raises(ValueError, match='min_separation'):
        search([0, 1, 2], np.zeros((3, 5)), [1, 1, 1], min_separation=-1)
    with pytest.raises(ValueError, match='query_ok'):
        search([0, 1, 2], np.zeros((3, 5)), [1, 1, 1], query_ok=np.ones(2, bool))
