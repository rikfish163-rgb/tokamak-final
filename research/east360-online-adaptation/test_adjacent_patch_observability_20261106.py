import numpy as np

from run_adjacent_patch_observability_20261106 import energy, remove_history, sample


def test_guard_projection_and_stable_forecast_rank_are_distinct():
    e = np.eye(5)
    p = remove_history(e[:1])
    np.testing.assert_array_equal(e[0] @ p, np.zeros(5))
    direction = p @ (e[0] + e[3])
    assert energy(e[:2].T, direction) == 0
    assert energy(e[3:4].T, direction) == 1


def test_sampling_is_per_role_per_shot_and_includes_all_late_windows():
    rows = [
        dict(role=role, shot=i, maturity=m, late=(role == "positive" and i == 1))
        for role in ("positive", "safe", "protect")
        for i in range(1, 30)
        for m in (63, 95)
    ]
    out = sample(list(reversed(rows)))
    assert len(out) == 38
    assert len({(x["shot"], x["maturity"]) for x in out if x["late"]}) == 2
    assert all(x["maturity"] == 63 for x in out if not x["late"])
