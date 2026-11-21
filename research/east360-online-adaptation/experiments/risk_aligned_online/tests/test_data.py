from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from experiments.fast_online_idea import pipeline as P
from experiments.risk_aligned_online import data


def _raw_signal(*, length: int = 4, ip: float = 2.0, mnm: float = 0.0) -> dict:
    values = np.zeros((length, len(data.CHANNELS)), dtype=np.float32)
    available = np.ones_like(values, dtype=bool)
    values[:, data.CHANNELS.index("ip")] = ip
    values[:, data.CHANNELS.index("MNM")] = mnm
    values[:, data.CHANNELS.index("ne_nG")] = 0.5
    values[:, data.CHANNELS.index("qa_proxy")] = 3.0
    return {
        "shot": 7,
        "v": values,
        "a": available,
        "times": 0.8 + np.arange(length, dtype=np.float64) / 1000.0,
        "rate": 1000.0,
        "planning_ip": np.full(length, ip, dtype=np.float32),
        "planning_Bt": np.full(length, 0.5, dtype=np.float32),
    }


def test_read_shot_is_prefix_causal_and_harmonizes_jtext_ip() -> None:
    base = _raw_signal(length=5, ip=2.0, mnm=0.0)
    future_changed = _raw_signal(length=5, ip=2.0, mnm=0.0)
    future_changed["v"][3:, :] = 1_000_000.0

    with patch.object(P, "read_signals", side_effect=[base, future_changed]):
        first = data.read_shot("J-TEXT", 7)
        second = data.read_shot("J-TEXT", 7)

    assert first["shot"] == 7
    assert first["values"].dtype == np.float32
    assert first["available"].dtype == bool
    assert first["times"].dtype == np.float64
    assert first["rate"] == 1000
    assert first["values"].shape == (5, 23)
    ip = data.CHANNELS.index("ip")
    np.testing.assert_array_equal(first["values"][:3], second["values"][:3])
    np.testing.assert_allclose(first["values"][:, ip], 2_000.0)


def test_east_masks_mnm_but_keeps_ne_ng_and_qa_proxy() -> None:
    raw = _raw_signal(length=3, ip=100_000.0, mnm=4.0)

    with patch.object(P, "read_signals", return_value=raw):
        shot = data.read_shot("EAST", 7)

    mnm = data.CHANNELS.index("MNM")
    ne_ng = data.CHANNELS.index("ne_nG")
    qa_proxy = data.CHANNELS.index("qa_proxy")
    np.testing.assert_array_equal(shot["values"][:, mnm], 0.0)
    assert not shot["available"][:, mnm].any()
    np.testing.assert_array_equal(shot["values"][:, ne_ng], 0.5)
    np.testing.assert_array_equal(shot["values"][:, qa_proxy], 3.0)
    assert shot["available"][:, ne_ng].all()
    assert shot["available"][:, qa_proxy].all()


def test_jtext_real_zero_is_an_observed_value() -> None:
    raw = _raw_signal(length=2, ip=0.0, mnm=0.0)

    with patch.object(P, "read_signals", return_value=raw):
        shot = data.read_shot("J-TEXT", 7)

    mnm = data.CHANNELS.index("MNM")
    ip = data.CHANNELS.index("ip")
    np.testing.assert_array_equal(shot["values"][:, mnm], 0.0)
    np.testing.assert_array_equal(shot["values"][:, ip], 0.0)
    assert shot["available"][:, mnm].all()
    assert shot["available"][:, ip].all()


def test_read_shot_does_not_read_outcome_metadata() -> None:
    with patch.object(P, "read_signals", return_value=_raw_signal()), patch.object(
        P, "read_outcome", side_effect=AssertionError("read_shot read metadata")
    ):
        data.read_shot("EAST", 7)


def test_attach_outcome_uses_jtext_floor_and_advanced_horizon() -> None:
    raw = _raw_signal(length=6)
    raw["times"] = np.array([0.870, 0.875, 0.900, 0.960, 0.975, 1.001], dtype=np.float64)
    with patch.object(P, "read_signals", return_value=raw):
        shot = data.read_shot("J-TEXT", 7)
    shot["times"] = raw["times"]

    with patch.object(P, "read_outcome", return_value=(True, 1.0)):
        data.attach_outcome(shot, "J-TEXT", {7: 10.0})

    assert shot["is_disrupt"] is True
    assert shot["down"] == 1.0
    assert shot["horizon"] == 25.0
    np.testing.assert_array_equal(shot["y"], [0, 0, 0, 0, 1, 0])
    # 0.870 is 130 ms before DownTime, outside the fixed 125-ms auxiliary head.
    np.testing.assert_array_equal(shot["y125"], [0, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(shot["valid"], [True, True, True, True, True, False])

    with patch.object(P, "read_signals", return_value=raw):
        shot = data.read_shot("J-TEXT", 7)
    shot["times"] = raw["times"]
    with patch.object(P, "read_outcome", return_value=(True, 1.0)):
        data.attach_outcome(shot, "J-TEXT", {7: 40.0})
    assert shot["horizon"] == 40.0
    assert shot["y"][3] == 1.0


def test_attach_outcome_east_uses_fixed_125_ms_and_invalidates_post_event() -> None:
    with patch.object(P, "read_signals", return_value=_raw_signal(length=5)):
        shot = data.read_shot("EAST", 7)
    shot["times"] = np.array([0.875, 0.900, 0.975, 1.000, 1.001], dtype=np.float64)

    with patch.object(P, "read_outcome", return_value=(True, 1.0)):
        data.attach_outcome(shot, "EAST", {7: 4.0})

    assert shot["horizon"] == 125.0
    np.testing.assert_array_equal(shot["y"], shot["y125"])
    np.testing.assert_array_equal(shot["y"], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(shot["valid"], [True, True, True, True, False])


def test_fit_source_scaler_requires_jtext_and_excludes_invalid_points() -> None:
    values = np.zeros((4, 23), dtype=np.float32)
    values[:, 0] = [1.0, 3.0, 5.0, 10_000.0]
    available = np.ones_like(values, dtype=bool)
    valid = np.array([True, True, False, False], dtype=bool)
    shot = dict(device="J-TEXT", values=values, available=available, valid=valid)

    scaler = data.fit_source_scaler([shot])
    assert scaler["center"].dtype == np.float32
    assert scaler["scale"].dtype == np.float32
    assert scaler["center"][0] == 2.0
    assert scaler["scale"][0] == 1.0

    east = dict(device="EAST", values=values, available=available, valid=valid)
    with pytest.raises(ValueError, match="J-TEXT"):
        data.fit_source_scaler([east])


def test_fit_source_scaler_rejects_unobserved_columns_but_accepts_real_zero() -> None:
    values = np.zeros((3, 23), dtype=np.float32)
    available = np.ones_like(values, dtype=bool)
    valid = np.ones(3, dtype=bool)
    available[:, 1] = False
    shot = dict(device="J-TEXT", values=values, available=available, valid=valid)

    with pytest.raises(ValueError, match="channel 1"):
        data.fit_source_scaler([shot])

    available[:, 1] = True
    scaler = data.fit_source_scaler([shot])
    assert scaler["center"][0] == 0.0
    assert scaler["scale"][0] > 0.0


def test_window_has_left_padding_and_never_reads_future() -> None:
    values = np.arange(5 * 23, dtype=np.float32).reshape(5, 23)
    available = np.ones_like(values, dtype=bool)
    shot = {"values": values, "available": available}

    before_values, before_available = data.window(shot, 1, length=4)
    values[2:] = 999_999.0
    after_values, after_available = data.window(shot, 1, length=4)

    expected_values = np.zeros((4, 23), dtype=np.float32)
    expected_values[-2:] = np.arange(2 * 23, dtype=np.float32).reshape(2, 23)
    np.testing.assert_array_equal(before_values, expected_values)
    np.testing.assert_array_equal(before_values, after_values)
    np.testing.assert_array_equal(before_available, after_available)
    assert not before_available[:2].any()

    right_values, right_available = data.window(shot, 4, length=3)
    np.testing.assert_array_equal(right_values, values[2:5])
    assert right_available.all()


def test_build_order_uses_only_signals_rejects_short_east_and_does_not_read_outcome() -> None:
    def fake_read_shot(device: str, shot: int) -> dict:
        length = 99 if shot == 2 else 100
        values = np.zeros((length, 23), dtype=np.float32)
        values[:, data.CHANNELS.index("ip")] = {1: 100_000.0, 2: 300_000.0, 3: 200_000.0}[shot]
        values[:, data.CHANNELS.index("Bt")] = {1: 1.0, 2: 1.0, 3: 0.5}[shot]
        return {
            "shot": shot,
            "values": values,
            "available": np.ones_like(values, dtype=bool),
            "times": np.arange(length, dtype=np.float64) / 1000.0,
            "rate": 1000,
            "planning_ip": values[:, data.CHANNELS.index("ip")],
            "planning_Bt": values[:, data.CHANNELS.index("Bt")],
        }

    with patch.object(data, "read_shot", side_effect=fake_read_shot), patch.object(
        P, "read_outcome", side_effect=AssertionError("ordering read outcome")
    ):
        rows = data.build_order("EAST", [3, 1])
        with pytest.raises(ValueError, match='100'):
            data.build_order("EAST", [3, 2, 1])

    assert [row["shot"] for row in rows] == [1, 3]
    assert rows[0]["initial_ip_A"] == 100_000.0
    assert rows[0]["initial_abs_Bt_raw"] == 1.0
    assert "initial_abs_Bt_T" not in rows[0]
    assert rows[0]["proxy"] < rows[1]["proxy"]


def test_build_order_keeps_jtext_chronology_without_reading_signals_or_outcome() -> None:
    with patch.object(data, "read_shot", side_effect=AssertionError("no J-TEXT planning read")), patch.object(
        P, "read_outcome", side_effect=AssertionError("ordering read outcome")
    ):
        rows = data.build_order("J-TEXT", [20, 10, 15])

    assert rows == [{"shot": 10}, {"shot": 15}, {"shot": 20}]


def test_build_order_uses_prefix_medians_and_never_silently_drops_shots() -> None:
    def fake_read(_device, sid):
        n = 99 if sid == 4 else 100
        v = np.zeros((n, 23), np.float32)
        ip = np.full(n, {1: 100_000., 2: 200_000., 4: 100_000.}[sid], np.float32)
        bt = np.full(n, {1: 2., 2: 1., 4: 1.}[sid], np.float32)
        ip[-1] = 1.e9  # a final-point outlier cannot define the order
        return dict(shot=sid, values=v, available=np.ones_like(v, bool),
                    planning_ip=ip, planning_Bt=bt)

    with patch.object(data, 'read_shot', side_effect=fake_read):
        rows = data.build_order('EAST', [1, 2])
        assert [row['shot'] for row in rows] == [1, 2]
        assert rows[1]['initial_ip_A'] == 200_000.
        assert rows[1]['proxy'] == pytest.approx(2.**1.2)
        with pytest.raises(ValueError, match='100'):
            data.build_order('EAST', [1, 4])


def test_source_roles_and_advanced_times_use_existing_canonical_files() -> None:
    roles = data.source_roles()
    assert set(roles) == {"train", "val", "test"}
    assert {name: len(ids) for name, ids in roles.items()} == {
        "train": 1353,
        "val": 160,
        "test": 220,
    }
    assert not (set(roles["train"]) & set(roles["val"]))
    assert not (set(roles["train"]) & set(roles["test"]))
    assert not (set(roles["val"]) & set(roles["test"]))
    advanced = data.advanced_times()
    assert isinstance(advanced, dict)
    assert advanced
