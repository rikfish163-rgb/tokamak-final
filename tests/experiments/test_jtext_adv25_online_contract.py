from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.transfer_methods_2026.jtext_online import controller, io, labels


def test_advanced_floor25_contract_and_safe_shot_is_ignored() -> None:
    mapping = {10: 12.0, 11: 38.0}

    assert labels.resolve_positive_horizon_ms(
        shot_id=10, is_disrupt=True, advanced_time_ms=mapping,
    ) == pytest.approx(25.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=11, is_disrupt=True, advanced_time_ms=mapping,
    ) == pytest.approx(38.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=12, is_disrupt=True, advanced_time_ms=mapping,
    ) == pytest.approx(25.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=10, is_disrupt=False, advanced_time_ms=mapping,
    ) is None


def test_prefix_feature_availability_does_not_use_later_mhd_samples() -> None:
    raw = {
        name: np.ones(5, dtype=np.float32) for name in io.RAW_SIGNAL_NAMES
    }
    raw["ip"] = np.full(5, 0.01, dtype=np.float32)
    raw["Mir_avg_fre"] = np.asarray([0.0, 0.0, 2.0, 2.0, 0.0], dtype=np.float32)
    raw["mode_number_n"] = np.zeros(5, dtype=np.float32)
    values, available = io.prefix_causal_rebuild_features(raw)

    mir = io.FEATURE_NAMES.index("Mir_avg_fre")
    mode = io.FEATURE_NAMES.index("mode_number_n")
    assert available[:2, mir].tolist() == [False, False]
    # Once the first valid MHD observation has arrived, later finite samples
    # (including a physically zero value) are observable and therefore remain
    # available; the key contract is that the first two prefixes stay masked.
    assert available[2:, mir].tolist() == [True, True, True]
    assert not available[:, mode].any()
    assert np.all(values[:2, mir] == 0.0)


def test_causal_window_cannot_contain_future_samples() -> None:
    values = np.arange(6 * io.FEATURE_COUNT, dtype=np.float32).reshape(
        6, io.FEATURE_COUNT,
    )
    window, valid = io.causal_prefix_window(values, endpoint=2, width=8)

    assert valid == 3
    np.testing.assert_array_equal(window[:5], 0.0)
    np.testing.assert_array_equal(window[-3:], values[:3])
    assert not np.any(window == values[3])


def _write_one_shot(root: Path, shot: int = 1) -> None:
    path = root / "J-TEXT/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for name in io.RAW_SIGNAL_NAMES:
            dataset = data.create_dataset(name, data=np.ones(4, dtype=np.float32))
            if name == "ip":
                dataset.attrs["SampleRate"] = 1000.0
                dataset.attrs["StartTime"] = 0.0
        meta = handle.create_group("meta")
        meta.create_dataset("IsDisrupt", data=np.asarray([1], dtype=np.uint8))
        meta.create_dataset("DownTime", data=np.asarray([1.0], dtype=np.float64))


def test_outcome_oracle_requires_sealed_selected_query(tmp_path: Path) -> None:
    _write_one_shot(tmp_path)
    advanced = tmp_path / "advanced.json"
    advanced.write_text('{"1": 40.0}\n', encoding="utf-8")
    oracle = io.OrderedOutcomeOracle(
        data_root=tmp_path, roster=[1], advanced_time_path=advanced,
    )
    oracle.begin_block([1])
    oracle.seal_prediction(1, "a" * 64)
    with pytest.raises(RuntimeError, match="sealed prediction block"):
        oracle.reveal_for_query(1, selected=True)
    oracle.finish_prediction_block()
    with pytest.raises(RuntimeError, match="explicit selected query"):
        oracle.reveal_for_query(1, selected=False)
    revealed = oracle.reveal_for_query(1, selected=True)
    assert revealed.positive_horizon_ms == pytest.approx(40.0)
    assert oracle.outcome_reads == 1
    oracle.complete_block([1])
    assert oracle.read_all_after_stream()[1] == revealed
    assert oracle.outcome_reads == 1


class _RecordingOracle:
    def __init__(self, roster: tuple[int, ...]) -> None:
        self.roster = roster
        self.events: list[tuple[str, int | None]] = []
        self.revealed: dict[int, io.RevealedOutcome] = {}

    @property
    def outcome_reads(self) -> int:
        return len(self.revealed)

    def begin_block(self, shots) -> None:
        self.events.append(("begin", int(shots[0])))

    def seal_prediction(self, shot_id: int, prediction_sha256: str) -> None:
        self.events.append(("seal", int(shot_id)))

    def finish_prediction_block(self) -> None:
        self.events.append(("finish", None))

    def reveal_for_query(self, shot_id: int, *, selected: bool):
        assert selected
        shot = int(shot_id)
        self.events.append(("reveal", shot))
        outcome = io.RevealedOutcome(
            shot_id=shot, is_disrupt=shot < 110, down_time_s=1.0,
            advanced_time_ms=25.0 if shot < 110 else None,
            positive_horizon_ms=25.0 if shot < 110 else None,
        )
        self.revealed[shot] = outcome
        return outcome

    def complete_block(self, selected_shots) -> None:
        self.events.append(("complete", None))

    def read_all_after_stream(self):
        return {
            shot: self.revealed.get(
                shot,
                io.RevealedOutcome(
                    shot_id=shot, is_disrupt=shot < 110, down_time_s=1.0,
                    advanced_time_ms=25.0 if shot < 110 else None,
                    positive_horizon_ms=25.0 if shot < 110 else None,
                ),
            )
            for shot in self.roster
        }


def test_active_stream_uses_exact_budget_and_reveals_after_seal() -> None:
    roster = tuple(range(220))
    signals = {
        shot: io.SignalShot(
            shot_id=shot,
            values=np.zeros((1, io.FEATURE_COUNT), dtype=np.float32),
            availability=np.ones((1, io.FEATURE_COUNT), dtype=bool),
            times_s=np.asarray([0.0]), sample_rate_hz=1000.0,
            start_time_s=0.0,
        )
        for shot in roster
    }
    traces = {
        seed: {shot: np.asarray([0.1], dtype=np.float32) for shot in roster}
        for seed in controller.SEEDS
    }
    oracle = _RecordingOracle(roster)
    result = controller.run_ordered_stream(
        roster=roster, expert_traces=traces,
        thresholds={seed: 0.5 for seed in controller.SEEDS},
        signals=signals, oracle=oracle, budget_percent=10,
        block_size=20, query_policy="active",
    )

    assert result["query_count"] == 22
    assert result["outcome_reads_during_stream"] == 22
    assert all(
        row["outcome_accessed_for_score"] is False
        for row in result["query_ledger"]
    )
    first_reveal = next(i for i, event in enumerate(oracle.events) if event[0] == "reveal")
    first_finish = next(i for i, event in enumerate(oracle.events) if event[0] == "finish")
    assert first_reveal > first_finish


def test_adv25_source_sampler_uses_per_shot_horizon_and_allows_safe_shots() -> None:
    import experiments.transfer_methods_2026.train_jtext_source_transformer as source

    disruptive = {
        "shot": 1, "v": np.zeros((101, 17), dtype=np.float32),
        "start": 0.0, "rate": 1000.0, "down": 0.1, "is_disrupt": True,
        "positive_horizon_ms": 25.0,
    }
    safe = {
        "shot": 2, "v": np.zeros((8, 17), dtype=np.float32),
        "start": 0.0, "rate": 1000.0, "down": None, "is_disrupt": False,
    }
    sampler = source.BalancedSourceSampler(
        [disruptive, safe], seed=131, horizon_ms=None,
    )

    np.testing.assert_array_equal(
        sampler.positive[0], np.arange(76, 101, dtype=np.int64),
    )
    assert 1 in sampler.negative and len(sampler.negative[1]) == 8
