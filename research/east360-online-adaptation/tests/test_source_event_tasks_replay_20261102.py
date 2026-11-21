import pytest

from experiments.risk_aligned_online.source_event_tasks_20261102.replay import select


def test_fixed_replay_covers_safe_and_danger_known_and_unknown_without_scores():
    records = [
        dict(shot=i, origin=127, group=g, origin_family=f)
        for i, (g, f) in enumerate(
            (("safe", -1), ("safe", 2), ("danger", -1), ("danger", 3)), 1
        )
    ]
    assert select(records) == [0, 1, 2, 3]
    assert select([dict(r, delta_bce=999) for r in records]) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        select(records[:3])
