import copy

import numpy as np

from experiments.risk_aligned_online.history_gradient_metric_20261102.run import (
    leave_one_shot_out,
)


def row(shot, label, signal, task):
    return dict(
        shot=shot,
        query=1,
        label=label,
        disruptive=bool(label),
        task_gradient256=task,
        results={"difference_real": {"receipt": {"gradient256": signal}}},
    )


def test_held_risk_gradient_cannot_change_its_own_fitted_metric():
    rows = [
        row(1, 0, [1, 0], [1, 1]),
        row(2, 0, [1, 1], [1, 0]),
        row(3, 1, [0, 1], [1, 1]),
        row(4, 1, [1, 1], [0, 1]),
    ]
    folds, held = leave_one_shot_out(rows)
    changed = copy.deepcopy(rows)
    changed[0]["task_gradient256"] = [-7, 12]
    repeat, _ = leave_one_shot_out(changed)
    first = next(r for r in folds if r["held_shot"] == 1)
    updated = next(r for r in repeat if r["held_shot"] == 1)
    assert 1 not in first["fit"]["training_ids"]
    assert first["fit"]["training_ids"] == [2, 3, 4]
    assert np.array_equal(first["fit"]["weights"], updated["fit"]["weights"])
    assert len(held) == len(rows)
