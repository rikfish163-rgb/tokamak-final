from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

selected = importlib.import_module("selected_mainline_contract")
jio = importlib.import_module("jtext_online_io")


def _shot_path(data_root: Path, shot: int) -> Path:
    return (
        data_root / "J-TEXT/unified_hdf5" / str(shot // 100 * 100)
        / f"{shot}.hdf5"
    )


def _write_jtext_hdf5(
    data_root: Path, shot: int, *, disruptive: bool,
    down_time_s: float = 0.018, points: int = 24,
) -> Path:
    path = _shot_path(data_root, shot)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index, name in enumerate(selected.SOURCE_RAW_SIGNAL_NAMES):
            values = np.linspace(0.1, 1.0, points, dtype=np.float32) + index
            if name in {"Mir_avg_fre", "mode_number_n"}:
                values[:6] = 0.0
            signal = data.create_dataset(name, data=values)
            if name == "ip":
                signal.attrs["SampleRate"] = 1000.0
                signal.attrs["StartTime"] = 0.0
        meta = handle.create_group("meta")
        meta.create_dataset("IsDisrupt", data=np.asarray([disruptive], np.bool_))
        meta.create_dataset(
            "DownTime", data=np.asarray([down_time_s], np.float64),
        )
    return path


def test_signal_only_loader_is_outcome_free_and_meta_mutation_cannot_change_input(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    shot = 1051500
    path = _write_jtext_hdf5(data_root, shot, disruptive=False)
    advanced = tmp_path / "AdvancedTime.json"
    advanced.write_text(json.dumps({str(shot): 73.0}), encoding="utf-8")

    before = jio.load_jtext_signal_only(data_root, shot)
    with h5py.File(path, "r+") as handle:
        handle["meta/IsDisrupt"][...] = np.asarray([True], np.bool_)
        handle["meta/DownTime"][...] = np.asarray([0.012], np.float64)
    advanced.write_text(json.dumps({str(shot): 999.0}), encoding="utf-8")
    after = jio.load_jtext_signal_only(data_root, shot)

    forbidden = {
        "is_disrupt", "outcome", "down_time_s", "advanced_time_ms",
        "positive_horizon_ms", "label", "target",
    }
    assert forbidden.isdisjoint(vars(before))
    np.testing.assert_array_equal(before.values, after.values)
    np.testing.assert_array_equal(before.availability, after.availability)
    np.testing.assert_array_equal(before.times_s, after.times_s)
    assert before.sample_rate_hz == after.sample_rate_hz == 1000.0
    assert before.start_time_s == after.start_time_s == 0.0


def test_future_suffix_mutation_cannot_change_prefix_features_or_window(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    shot = 1051501
    path = _write_jtext_hdf5(data_root, shot, disruptive=True)
    endpoint = 9
    before = jio.load_jtext_signal_only(data_root, shot)
    before_window, before_valid = before.causal_window(endpoint, window_points=128)
    with h5py.File(path, "r+") as handle:
        for name in ("Mir_avg_fre", "mode_number_n", "n=1 amplitude"):
            values = handle[f"data/{name}"][:]
            values[endpoint + 1 :] = values[endpoint + 1 :] * -99.0 + 7.0
            handle[f"data/{name}"][...] = values
    after = jio.load_jtext_signal_only(data_root, shot)
    after_window, after_valid = after.causal_window(endpoint, window_points=128)

    np.testing.assert_array_equal(
        before.values[: endpoint + 1], after.values[: endpoint + 1],
    )
    np.testing.assert_array_equal(
        before.availability[: endpoint + 1],
        after.availability[: endpoint + 1],
    )
    np.testing.assert_array_equal(before_window, after_window)
    assert before_valid == after_valid == endpoint + 1


def test_lazy_oracle_requires_seal_and_query_and_preserves_hdf5_authority(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    safe, disruptive = 1051502, 1051503
    _write_jtext_hdf5(data_root, safe, disruptive=False)
    _write_jtext_hdf5(data_root, disruptive, disruptive=True, down_time_s=0.018)
    advanced = tmp_path / "AdvancedTime.json"
    # A safe shot deliberately has an entry; the disruptive shot deliberately
    # does not, so neither case can use AdvancedTime as outcome authority.
    advanced.write_text(json.dumps({str(safe): 80.0}), encoding="utf-8")
    oracle = jio.JTextLazyOutcomeOracle(
        data_root=data_root,
        ordered_roster=[safe, disruptive],
        advanced_time_path=advanced,
    )

    with pytest.raises(RuntimeError, match="seal"):
        oracle.reveal_for_query(safe, selected=True)
    oracle.seal_current(safe, prediction_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="selected"):
        oracle.reveal_for_query(safe, selected=False)
    safe_outcome = oracle.reveal_for_query(safe, selected=True)
    assert safe_outcome.is_disrupt is False
    assert safe_outcome.down_time_s is None
    assert safe_outcome.advanced_time_ms is None
    assert safe_outcome.positive_horizon_ms is None
    with pytest.raises(RuntimeError, match="duplicate"):
        oracle.reveal_for_query(safe, selected=True)

    oracle.seal_current(disruptive, prediction_sha256="b" * 64)
    disruptive_outcome = oracle.reveal_for_query(disruptive, selected=True)
    assert disruptive_outcome.is_disrupt is True
    assert disruptive_outcome.down_time_s == pytest.approx(0.018)
    assert disruptive_outcome.advanced_time_ms is None
    assert disruptive_outcome.positive_horizon_ms == pytest.approx(25.0)


def test_advanced_time_is_not_opened_until_postseal_selected_reveal(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    shot = 1051504
    _write_jtext_hdf5(data_root, shot, disruptive=True)
    advanced = tmp_path / "not-created-yet.json"
    oracle = jio.JTextLazyOutcomeOracle(
        data_root=data_root,
        ordered_roster=[shot],
        advanced_time_path=advanced,
    )
    oracle.seal_current(shot, prediction_sha256="c" * 64)
    advanced.write_text(json.dumps({str(shot): 61.0}), encoding="utf-8")

    revealed = oracle.reveal_for_query(shot, selected=True)

    assert revealed.advanced_time_ms == pytest.approx(61.0)
    assert revealed.positive_horizon_ms == pytest.approx(61.0)


def test_postseal_target_uses_revealed_horizon_only() -> None:
    revealed = jio.JTextRevealedOutcome(
        shot_id=1,
        is_disrupt=True,
        down_time_s=0.100,
        advanced_time_ms=40.0,
        positive_horizon_ms=40.0,
    )
    times = np.asarray([0.050, 0.060, 0.070, 0.080, 0.090, 0.100])

    target = jio.build_postseal_jtext_target(revealed, times)

    np.testing.assert_array_equal(
        target,
        np.asarray([0, 1, 1, 1, 1, 1], dtype=np.float32),
    )


def test_ordered_test220_is_exact_unique_disjoint_and_diagnostic() -> None:
    contract = jio.load_ordered_test220(ROOT)

    assert contract.schema_version == "jtext-test220-ordered-v1"
    assert contract.claim_status == "TEST_DRIVEN_EXPLORATION"
    assert contract.order_rule == "ascending_shot_id"
    assert len(contract.shots) == len(set(contract.shots)) == 220
    assert list(contract.shots) == sorted(contract.shots)
    roles = jio.load_canonical_jtext_roles(ROOT)
    assert set(contract.shots).isdisjoint(roles["train"])
    assert set(contract.shots).isdisjoint(roles["val"])
    assert list(contract.shots) == roles["test"]
    assert selected.sha256_json(list(contract.shots)) == contract.roster_sha256
