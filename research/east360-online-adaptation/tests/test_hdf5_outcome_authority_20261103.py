"""HDF metadata is the only hard-outcome authority; never coerce invalid truth."""

import h5py
import numpy as np
import pytest

from experiments.risk_aligned_online import io23


def make_file(tmp_path, monkeypatch, flag, down):
    path = tmp_path / "69854.hdf5"
    with h5py.File(path, "w") as f:
        f["meta/IsDisrupt"] = flag
        f["meta/DownTime"] = down
    monkeypatch.setattr(io23, "path_for", lambda *_: path)
    return path


@pytest.mark.parametrize("flag", [1, True, np.array([1.0])])
def test_hdf_disruption_and_timestamp_are_returned(tmp_path, monkeypatch, flag):
    make_file(tmp_path, monkeypatch, flag, np.array([6.9536]))
    assert io23.read_outcome("EAST", 69854) == (True, 6.9536)


def test_safe_hdf_does_not_turn_terminal_time_into_disruption(tmp_path, monkeypatch):
    make_file(tmp_path, monkeypatch, 0, 7.25)
    assert io23.read_outcome("EAST", 69854) == (False, None)


@pytest.mark.parametrize("flag", [2, -1, np.nan, np.array([0, 1]), np.array([]), b"0"])
def test_malformed_hdf_flag_is_not_silently_coerced(tmp_path, monkeypatch, flag):
    make_file(tmp_path, monkeypatch, flag, 6.9536)
    with pytest.raises(ValueError, match="IsDisrupt"):
        io23.read_outcome("EAST", 69854)


@pytest.mark.parametrize("down", [np.nan, np.array([]), np.array([1.0, 2.0])])
def test_disruption_needs_one_finite_timestamp(tmp_path, monkeypatch, down):
    make_file(tmp_path, monkeypatch, 1, down)
    with pytest.raises(ValueError, match="timestamp"):
        io23.read_outcome("EAST", 69854)


def test_east_point_labels_use_hdf_truth_not_supplied_legacy_fields(
    tmp_path, monkeypatch
):
    from experiments.risk_aligned_online import simple_data

    make_file(tmp_path, monkeypatch, 1, 2.0)
    signal = dict(
        device="EAST",
        shot=69854,
        times=np.array([1.874, 1.875, 1.970, 2.0]),
        is_disrupt=False,
        disruption_label=0,
        down=99.0,
    )
    result = simple_data.attach_outcome(signal, "EAST", {69854: 999.0})
    assert result["is_disrupt"] is True and result["down"] == 2.0
    assert result["horizon"] == 125.0
    np.testing.assert_array_equal(result["y"], [0, 1, 1, 1])


@pytest.mark.parametrize("advanced,horizon", [({}, 25.0), ({69854: 40.0}, 40.0)])
def test_jtext_keeps_advanced_window_but_hdf_outcome(
    tmp_path, monkeypatch, advanced, horizon
):
    from experiments.risk_aligned_online import simple_data

    make_file(tmp_path, monkeypatch, 1, 2.0)
    signal = dict(device="J-TEXT", shot=69854, times=np.array([1.959, 1.960, 1.975]))
    result = simple_data.attach_outcome(signal, "J-TEXT", advanced)
    assert result["is_disrupt"] is True and result["down"] == 2.0
    assert result["horizon"] == horizon


def test_score_recheck_uses_strict30_and_first_alarm():
    from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve

    assert score_curve(np.array([0.970]), np.array([0.8]), True, 1.0, 0.7) == dict(
        alarm=True, lead_ms=30.0, timely=False
    )
    assert score_curve(
        np.array([0.969, 0.970]), np.array([0.8, 0.9]), True, 1.0, 0.7
    ) == dict(alarm=True, lead_ms=31.0, timely=True)
    assert score_curve(np.array([0.969]), np.array([0.8]), False, None, 0.7) == dict(
        alarm=True, lead_ms=None, timely=False
    )
