import numpy as np
import pytest

from experiments.risk_aligned_online.cold20_compare_20261104.data import (
    build_pools,
    make_schedule,
)


def shots():
    times = np.array([0.0, 0.874, 0.875, 0.9, 1.0, 1.001])
    return {
        sid: dict(
            shot=sid,
            device="EAST",
            times=times,
            values=np.ones((6, 23), np.float32),
            available=np.ones((6, 23), bool),
            disruption_label=0,
        )
        for sid in (1, 2)
    }


def test_pools_use_only_revealed_hdf_truth_and_fixed125():
    records, pools = build_pools(shots(), {1: (True, 1.0), 2: (False, None)}, [1, 2])
    disrupted = [r for r in records if r["shot"] == 1]
    assert [(r["query"], r["label"], r["group"]) for r in disrupted] == [
        (0, 0, "earlier"),
        (1, 0, "earlier"),
        (2, 1, "positive"),
        (3, 1, "positive"),
        (4, 1, "positive"),
    ]
    assert 1 in pools["earlier"] and 1 in pools["positive"] and 2 in pools["safe"]
    assert all(r["label"] == 0 for r in records if r["shot"] == 2)


def test_unrevealed_or_future_shot_cannot_enter_training():
    for truth, allowed in [
        ({1: (True, 1.0)}, [1, 2]),
        ({1: (True, 1.0), 2: (False, None)}, [1]),
        ({1: (True, 1.0), 2: (False, None), 3: (False, None)}, [1, 2]),
    ]:
        with pytest.raises(ValueError, match="prefix|revealed"):
            build_pools(shots(), truth, allowed)


def test_missing_stratum_is_reported_not_reweighted_silently():
    with pytest.raises(ValueError, match="strat"):
        build_pools(shots(), {1: (False, None), 2: (False, None)}, [1, 2])


def test_schedule_is_fixed_and_balanced_by_shot_then_endpoint():
    records, pools = build_pools(shots(), {1: (True, 1.0), 2: (False, None)}, [1, 2])
    a = make_schedule(pools, steps=4, batch_size=8)
    assert a == make_schedule(pools, steps=4, batch_size=8)
    for batch in a:
        groups = [records[i]["group"] for i in batch["indices"]]
        assert (
            groups.count("positive") == 4
            and groups.count("safe") == 2
            and groups.count("earlier") == 2
        )


@pytest.mark.parametrize(
    "flag,down", [(1, 1.0), (True, None), (True, np.nan), (False, 1.0)]
)
def test_outcome_contract_is_explicit(flag, down):
    with pytest.raises(ValueError, match="truth"):
        build_pools(shots(), {1: (flag, down), 2: (False, None)}, [1, 2])
