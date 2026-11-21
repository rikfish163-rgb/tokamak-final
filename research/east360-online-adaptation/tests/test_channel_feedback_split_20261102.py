import numpy as np

from experiments.risk_aligned_online.channel_feedback_20261102.run import (
    leave_one_shot_out,
)


def test_held_risk_and_observations_cannot_fit_their_own_weights():
    rows = [
        dict(shot=i + 1, query=10, label=int(i >= 2), disruptive=i >= 2)
        for i in range(4)
    ]
    g = np.array(
        [
            [[1, 0], [0, 1], [2, 1]],
            [[1, 1], [1, 0], [0, 0]],
            [[0, 1], [1, 1], [0, 0]],
            [[2, 1], [1, 0], [0, 0]],
        ],
        dtype=float,
    )
    r = np.array([[1, 2], [2, 1], [1, -1], [2, -1]], dtype=float)
    obs = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    folds, real, reverse = leave_one_shot_out(g, r, rows, obs, -g)
    first = next(f for f in folds if f["held_shot"] == 1)
    assert first["fit"]["training_ids"] == [2, 3, 4]
    assert first["fit"]["active_indices"] == [0, 1]
    assert first["fit"]["weights"][2] == 0
    assert first["unsupported_held_channels"] == [2]
    changed_r = r.copy()
    changed_r[0] = [-20, 30]
    changed_obs = obs.copy()
    changed_obs[0, 2] = 0
    changed_g = g.copy()
    changed_g[0, 2] = 0
    again, _, _ = leave_one_shot_out(
        changed_g, changed_r, rows, changed_obs, -changed_g
    )
    updated = next(f for f in again if f["held_shot"] == 1)
    assert first["fit"] == updated["fit"]
    assert len(real) == len(reverse) == len(rows)
    # The control reuses the real-data fit. It is not independently optimized.
    for actual, control in zip(real, reverse, strict=True):
        assert actual["shot"] == control["shot"]
        assert np.isclose(actual["after_cosine"], -control["after_cosine"])
