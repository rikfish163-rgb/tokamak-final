import numpy as np

from experiments.risk_aligned_online.risk_conditioned_feedback_20261102.run import (
    leave_one_shot_out,
)


def test_held_risk_condition_and_observation_do_not_fit_own_tables():
    rows = [
        dict(shot=i + 1, query=160, label=int(i >= 2), disruptive=i >= 2)
        for i in range(4)
    ]
    g = np.array(
        [
            [[1, 0], [0, 1], [1, 2]],
            [[1, 1], [1, 0], [0, 0]],
            [[0, 1], [1, 1], [0, 0]],
            [[2, 1], [1, 0], [0, 0]],
        ],
        float,
    )
    r = np.array([[1, 2], [2, 1], [1, -1], [2, -1]], float)
    obs = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    u = np.array([0.1, 0.2, 0.7, 0.9])
    folds, actual, control, constant = leave_one_shot_out(g, r, rows, obs, u, -g)
    f = next(x for x in folds if x["held_shot"] == 1)
    assert f["fit"]["training_ids"] == [2, 3, 4]
    assert f["fit"]["active_indices"] == [0, 1]
    assert f["unsupported_held_channels"] == [2]
    r2, u2 = r.copy(), u.copy()
    r2[0], u2[0] = [-5, 6], 0.95
    again, _, _, _ = leave_one_shot_out(g, r2, rows, obs, u2, -g)
    updated = next(x for x in again if x["held_shot"] == 1)
    assert f["fit"] == updated["fit"]
    assert f["constant_fit"] == updated["constant_fit"]
    assert len(actual) == len(control) == len(constant) == len(rows)
    for real, reverse in zip(actual, control, strict=True):
        assert (
            real["shot"] == reverse["shot"]
            and real["condition"] == reverse["condition"]
        )
        np.testing.assert_allclose(real["weights"], reverse["weights"], rtol=0, atol=0)
        np.testing.assert_allclose(
            real["after_cosine"], -reverse["after_cosine"], atol=1e-12
        )
