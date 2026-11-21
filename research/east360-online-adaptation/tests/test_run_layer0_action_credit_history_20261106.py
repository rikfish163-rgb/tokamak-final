import numpy as np
import pytest

from run_layer0_action_credit_history_20261106 import (
    anchor_inputs,
    calibration_record,
    release_rows,
)


def sample_release():
    seal = [{"shot": 10, "seal_sha256": "a", "seal_ns": 11}]
    commit = dict(
        block=1,
        rank_end=1,
        selected=[10],
        committed_ns=12,
        seals=[{"shot": 10, "seal_sha256": "a"}],
    )
    outcome = dict(
        shot=10, disruptive=True, down=0.75, read_start_ns=13, read_end_ns=14
    )
    return seal, commit, {"commit_sha256": "c", "outcomes": [outcome]}


def test_label_is_bound_to_the_block_that_committed_after_prediction():
    seals, commit, reveal = sample_release()
    rows = release_rows(1, seals, commit, reveal, "c", block_size=1)
    assert rows[0]["available_after_rank"] == 1
    assert rows[0]["disruptive"] is True


@pytest.mark.parametrize("key", ["commit_hash", "start", "shot", "seal"])
def test_reveal_before_commit_or_other_shot_is_rejected(key):
    seals, commit, reveal = sample_release()
    if key == "commit_hash":
        reveal["commit_sha256"] = "d"
    if key == "start":
        reveal["outcomes"][0]["read_start_ns"] = 10
    if key == "shot":
        reveal["outcomes"][0]["shot"] = 11
    if key == "seal":
        commit["seals"][0]["seal_sha256"] = "b"
    with pytest.raises(ValueError):
        release_rows(1, seals, commit, reveal, "c", block_size=1)


def test_anchor_features_never_use_later_scores_or_shot_duration():
    times = 2 + np.arange(11, dtype=float) / 100
    p = np.linspace(0.1, 0.6, 11)
    z = np.log(p / (1 - p))
    adapted = p.copy()
    adapted[3:] += 0.1
    history, norms = np.ones((3, 128)), np.array([0.1, 0.2, 0.3])
    before = anchor_inputs(times, p, z, adapted, history, norms, first=3, stride=3)
    adapted[4:] = 0.95
    after = anchor_inputs(
        times[:4], p[:4], z[:4], adapted[:4], history[:1], norms[:1], first=3, stride=3
    )
    np.testing.assert_array_equal(before[:1], after)
    np.testing.assert_allclose(before[0, :4], [z[3], 0.03, 0.1, 0.1])


def test_map_statistics_use_same_sentinel_mask_as_frozen_map():
    x = np.zeros((2, 23), dtype=np.float32)
    x[0, 1] = -10
    m = np.ones_like(x, dtype=bool)
    got = calibration_record(
        dict(shot=10, device="EAST", values=x, available=m, times=np.arange(2))
    )
    assert set(got) == {"shot", "device", "values", "available"}
    assert not got["available"][0, 1] and got["available"][1, 1]
    assert m.all()
