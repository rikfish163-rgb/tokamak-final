from experiments.risk_aligned_online.risk_bridge_forecaster_20261102.prepare import (
    folds,
    summarize,
)


def test_fold_plan_holds_whole_shots_with_both_outcomes():
    ids = list(range(20))
    rows = [dict(shot=sid, disruptive=sid < 8) for sid in ids]
    plan = folds(rows, ids)
    assert len(plan) == 4
    assert sorted(sid for f in plan for sid in f["held"]) == ids
    for f in plan:
        assert len(f["fit"]) == 15 and len(f["held"]) == 5
        assert not set(f["fit"]) & set(f["held"])
        assert sum(sid < 8 for sid in f["held"]) == 2


def test_summary_balances_labels_then_shots_not_raw_points():
    rows = [
        dict(shot=1, label=0, disruptive=False, delta_bce=2.0),
        dict(shot=1, label=0, disruptive=False, delta_bce=4.0),
        dict(shot=2, label=0, disruptive=True, delta_bce=-1.0),
        dict(shot=2, label=1, disruptive=True, delta_bce=-3.0),
    ]
    result = summarize(rows)
    assert result["class0"]["mean_delta_bce"] == 1.0
    assert result["class1"]["mean_delta_bce"] == -3.0
    assert result["safe"]["mean_delta_bce"] == 3.0
    assert result["balanced_delta_bce"] == -1.0
