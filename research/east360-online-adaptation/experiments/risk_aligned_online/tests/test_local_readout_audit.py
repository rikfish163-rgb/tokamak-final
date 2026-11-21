import copy

import numpy as np
import pytest

from experiments.risk_aligned_online.local_readout_20261029.audit import (
    auxiliary,
    check_feedback,
)
from experiments.risk_aligned_online.local_readout_20261029.learning import MODES


def test_auxiliary_recount_excludes_missing_and_post_end_targets():
    n = 300
    signal = dict(values=np.zeros((n, 23)), available=np.ones((n, 23), bool))
    active = np.ones(23, bool)
    active[[2, 10]] = False
    z = dict(aux_clock=np.array([256]), aux_last=np.zeros((1, 23)))
    z["aux_available"] = active[None]
    for arm in MODES:
        for part in ("before", "after"):
            z[f"aux_{arm}_{part}"] = np.ones((1, 25, 23))
    signal["available"][256:258, 0] = False
    valid = np.arange(n) < 260
    rows = auxiliary(z, signal, valid, np.zeros(23), np.ones(23), active)
    r = rows[0]
    assert r["elements"] == 4 * 21 - 2
    assert r["channel_counts"][0] == 2 and r["channel_counts"][2] == 0
    assert r["losses"]["hold"]["sum"] == 0
    assert r["losses"]["local_readout_after"]["sum"] == r["elements"] / 2
    bad = copy.deepcopy(z)
    bad["aux_available"][0, 2] = True
    with pytest.raises(ValueError):
        auxiliary(bad, signal, valid, np.zeros(23), np.ones(23), active)


def test_feedback_recount_rejects_overlap_and_invented_observations():
    n = 400
    signal = dict(values=np.zeros((n, 23)), available=np.ones((n, 23), bool))
    active = np.ones(23, bool)
    active[[2, 10]] = False
    receipt = []
    for j, c in enumerate((256, 384)):
        receipt.append(
            dict(
                clock=c,
                origins=[c - v for v in (101, 76, 51, 26)],
                calibration_last_target=c - 51,
                gradient_first_target=c - 50,
                gradient_last_target=c - 1,
                history_fit_rows=2 + 4 * j,
                fit_counts_max=2 + 4 * j,
                history_rows_after=4 + 4 * j,
                readout_delta_norm=0.1,
            )
        )
    updates = {
        k: dict(
            opportunities=2,
            observed=2 * 2 * 25 * 21,
            updates=2,
            decreased=2,
            max_norm=0.05,
            loss_before_sum=2.0,
            loss_after_sum=1.0,
        )
        for k in MODES
    }
    updates["feedback"] = receipt
    check_feedback(updates, signal, active)
    wrong = copy.deepcopy(updates)
    wrong["feedback"][0]["calibration_last_target"] = 206
    with pytest.raises(ValueError):
        check_feedback(wrong, signal, active)
    wrong = copy.deepcopy(updates)
    wrong["local_readout"]["observed"] += 1
    with pytest.raises(ValueError):
        check_feedback(wrong, signal, active)


def test_missing_targets_never_count_as_zero_labels():
    signal = dict(values=np.zeros((300, 23)), available=np.zeros((300, 23), bool))
    active = np.ones(23, bool)
    z = dict(
        aux_clock=np.array([256]),
        aux_last=np.zeros((1, 23)),
        aux_available=np.zeros((1, 23), bool),
    )
    for arm in MODES:
        for part in ("before", "after"):
            z[f"aux_{arm}_{part}"] = np.full((1, 25, 23), np.nan)
    result = auxiliary(z, signal, np.ones(300, bool), np.zeros(23), np.ones(23), active)
    assert result[0]["elements"] == 0
    assert all(v["sum"] == 0 for v in result[0]["losses"].values())
