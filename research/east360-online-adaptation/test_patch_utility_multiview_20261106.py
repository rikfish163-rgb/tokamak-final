"""Tests of grouped weighting, physical-shot split, and joint safety."""

import numpy as np
import pytest

import run_patch_utility_multiview_20261106 as m


def test_class_weight_does_not_multiply_a_shot_with_more_views():
    y = np.array([
        [0, 1, 1, -1, 0], [0, 1, 0, -1, 0], [0, 1, 0, -1, 0],
        [0, 1, 0, -1, 0], [0, 1, 0, -1, 0],
    ])
    owner = np.array([5, 5, 5, 5, 6])
    view = np.array([0, 0, 1, 2, 1])
    allow = np.ones_like(y, dtype=bool)
    w = m.paired_weights(y, allow, owner, view)
    for klass in (-1, 0, 1):
        assert w[y == klass].sum() == pytest.approx(1 / 3)
    win = np.where(y == 1, w, 0.0)
    assert win[owner == 5].sum() == pytest.approx(1 / 6)
    assert win[owner == 6].sum() == pytest.approx(1 / 6)
    assert win[(owner == 5) & (view == 0)].sum() == pytest.approx(1 / 18)
    assert win[0, 1] == pytest.approx(1 / 72)
    assert not w[:, 0].any()


def test_group_split_slices_all_views_by_one_physical_id_assignment():
    fold = np.array([0, 1, 2, 3, 4, 0])
    owner = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    tr, cal, held = m.rows_for_fold(fold, owner, 0)
    assert set(owner[tr]) == {2, 3, 4}
    assert set(owner[cal]) == {1}
    assert set(owner[held]) == {0, 5}
    assert np.all(tr.astype(int) + cal + held == 1)


def result(recovered=(), lost=(), fp_added=(), fp_removed=()):
    return dict(recovered=list(recovered), lost=list(lost), fp_added=list(fp_added),
                fp_removed=list(fp_removed))


def test_calibration_cannot_hide_one_views_harm_with_other_views_improvement():
    choices = {
        -2.0: {m.VIEWS[0]: result(fp_added=[9]),
               m.VIEWS[1]: result(recovered=[1], fp_removed=[9]),
               m.VIEWS[2]: result(recovered=[2])},
        -1.0: {m.VIEWS[0]: result(),
               m.VIEWS[1]: result(recovered=[1]),
               m.VIEWS[2]: result(recovered=[1])},
        0.0: {m.VIEWS[0]: result(),
              m.VIEWS[1]: result(recovered=[1]),
              m.VIEWS[2]: result()},
        np.inf: {v: result() for v in m.VIEWS},
    }
    margin, receipt = m.calibrate(choices.__getitem__, list(choices))
    assert margin == 0.0
    assert not receipt[0]["eligible"]
    assert receipt[1]["recovered"] == [1]


def test_promotion_distinguishes_physical_shot_not_number_of_views():
    result_rows = {
        key: {v: result() for v in m.VIEWS}
        for key in ("true", "risk", "reverse", "scalar")
    }
    result_rows["true"][m.VIEWS[1]] = result(recovered=[1])
    result_rows["reverse"][m.VIEWS[2]] = result(recovered=[1])
    assert m.source_gate(result_rows) == (False, [])
    result_rows["true"][m.VIEWS[2]] = result(recovered=[2])
    assert m.source_gate(result_rows) == (True, [2])
    result_rows["true"][m.VIEWS[0]] = result(fp_added=[9])
    assert m.source_gate(result_rows) == (False, [2])


def test_single_view_matches_the_existing_weighted_model():
    rng = np.random.default_rng(50)
    x = rng.normal(size=(9, 4, 3))
    hard = np.tile([0, -1, 0, 1], (9, 1))
    allowed = np.ones_like(hard, dtype=bool)
    owner = np.repeat(np.arange(3), 3)
    result_a = m.sparse.fit_events(x, hard, allowed, owner)
    result_b = m.fit(x, hard, allowed, owner, np.zeros_like(owner))
    assert m.sparse.event_scores(result_a, x) == pytest.approx(m.sparse.event_scores(result_b, x))
    with pytest.raises(ValueError, match="class"):
        m.paired_weights(np.abs(hard), allowed, owner, np.zeros_like(owner))
