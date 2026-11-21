import copy

import pytest

from experiments.risk_aligned_online.matched_preparation_20261023.selection import matched_controls


def records():
    rows = []
    for shot in (100, 200, 300, 400):
        rows.append(dict(shot=shot, is_disrupt=True, shotlist_is_disrupt=True))
        for delta in (-2, -1, 1, 2):
            rows.append(dict(shot=shot+delta, is_disrupt=False, shotlist_is_disrupt=False))
    return rows


def test_preserves_cases_and_selects_two_controls_each_side_without_replacement():
    pool = records(); original = copy.deepcopy(pool)
    selected, pairs = matched_controls(pool, [100, 200, 300, 400], test_ids=[900])
    assert pool == original and len(selected) == len({r['shot'] for r in selected}) == 20
    assert [r['shot'] for r in selected if r['is_disrupt']] == [100, 200, 300, 400]
    assert len(pairs) == 16
    for shot in (100, 200, 300, 400):
        match = [p['safe'] for p in pairs if p['disruptive'] == shot]
        assert sorted(match) == [shot-2, shot-1, shot+1, shot+2]
    assert matched_controls(pool[::-1], [400, 100, 300, 200], test_ids=[900]) == (selected, pairs)


def test_excludes_test_metadata_even_if_not_selected_and_rejects_conflicts():
    with pytest.raises(ValueError, match='overlap'):
        matched_controls(records(), [100, 200, 300, 400], test_ids=[98])
    bad = records(); bad[0]['shotlist_is_disrupt'] = False
    with pytest.raises(ValueError, match='disagree'):
        matched_controls(bad, [100, 200, 300, 400], test_ids=[])


def test_no_replacement_when_local_controls_unavailable():
    pool = [r for r in records() if r['shot'] != 98]
    with pytest.raises(ValueError, match='before'):
        matched_controls(pool, [100, 200, 300, 400], test_ids=[])


def test_duplicate_ids_or_wrong_case_identity_fail():
    pool = records()
    with pytest.raises(ValueError, match='distinct'):
        matched_controls(pool+[pool[0]], [100, 200, 300, 400], test_ids=[])
    with pytest.raises(ValueError, match='four disruptive'):
        matched_controls(pool, [98, 200, 300, 400], test_ids=[])
