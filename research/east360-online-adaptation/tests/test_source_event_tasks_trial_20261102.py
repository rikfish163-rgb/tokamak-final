import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.source_event_tasks_20261102 import trial


def records(role="train"):
    return [
        dict(shot=1, group="safe", label=0.0, role=role, origin_family=-1),
        dict(shot=2, group="danger", label=1.0, role=role, origin_family=2),
        dict(shot=2, group="early", label=0.0, role=role, origin_family=2),
    ]


def test_schedule_is_reproducible_paired_and_does_not_train_early():
    a = trial.make_schedule(records(), steps=12)
    assert a == trial.make_schedule(records(), steps=12)
    assert [r["task"] for r in a[:6]] == list(trial.TASKS) * 2
    assert all(r["indices"] == [0, 1] for r in a)
    assert len(a) == 12
    with pytest.raises(ValueError):
        trial.make_schedule(records("val"), steps=12)


def test_pack_index_is_single_example_not_a_batch_averaged_update():
    pack = dict(
        h=torch.arange(12).reshape(3, 4), valid=torch.ones(3, 2, dtype=torch.bool)
    )
    out = trial.one(pack, 1)
    assert out["h"].shape == (1, 4)
    torch.testing.assert_close(out["h"], pack["h"][1:2])


def test_summary_equal_weights_shots_not_number_of_windows():
    rows = [
        dict(shot=1, group="safe", delta_bce=1.0),
        dict(shot=1, group="safe", delta_bce=1.0),
        dict(shot=2, group="safe", delta_bce=-3.0),
        dict(shot=3, group="danger", delta_bce=-2.0),
        dict(shot=3, group="early", delta_bce=10.0),
    ]
    s = trial.summarize(rows)
    assert s["safe"]["mean_delta_bce"] == -1.0
    assert s["balanced"] == -1.5
    assert s["early"]["mean_delta_bce"] == 10.0


def test_gate_requires_masked_usefulness_semantics_and_native_protection():
    base = dict(
        safe=dict(mean_delta_bce=-0.01),
        danger=dict(mean_delta_bce=-0.01),
        balanced=-0.01,
    )
    s = {
        task: {arm: copy.deepcopy(base) for arm in trial.EVALUATION_ARMS}
        for task in trial.TASKS
    }
    assert not trial.gates(s)["passed"]
    s["no_MNM_CIII"]["true_real"].update(
        safe=dict(mean_delta_bce=-0.02),
        danger=dict(mean_delta_bce=-0.02),
        balanced=-0.02,
    )
    assert trial.gates(s)["passed"]
    s["native"]["true_real"]["safe"]["mean_delta_bce"] = 0.001
    assert not trial.gates(s)["passed"]


@pytest.mark.parametrize("mode", ["real", "reverse", "hold"])
def test_hidden_channels_cannot_supply_inner_targets(mode):
    signal = dict(
        values=np.ones((190, 23), np.float32), available=np.ones((190, 23), bool)
    )
    truths, mask = trial.task_targets(signal, origin=127, task="no_MNM_CIII")
    assert truths[mode].shape == (1, 25, 23)
    assert not mask[:, :, [1, 2, 10]].any()
    assert mask[:, :, 0].all()
    assert signal["available"].all()


def test_packed_windows_do_not_include_other_shots(monkeypatch):
    from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import (
        batch,
    )

    values = np.stack([np.full((128, 23), i, np.float32) for i in (7.0, 19.0, -5.0)])
    mask = np.ones_like(values, bool)
    mask[1, :10] = False

    def captured(_model, signal, ends, _basis):
        x, a = batch(signal, ends, "cpu")
        np.testing.assert_array_equal(x.numpy(), values)
        np.testing.assert_array_equal(a.numpy(), mask)
        return dict(h=x, valid=a)

    monkeypatch.setattr(trial, "encode", captured)
    trial.windows_pack(None, values, mask, None)
