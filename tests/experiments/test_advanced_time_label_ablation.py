from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _labels():
    return importlib.import_module("advanced_time_labels")


def _disruptive_shot(shot_id: int = 101, *, length: int = 201) -> dict:
    return {
        "shot": int(shot_id),
        "v": np.zeros((length, 2), dtype=np.float32),
        "a": np.ones((length, 2), dtype=bool),
        "is_disrupt": True,
        "down": 0.200,
        "start": 0.0,
        "rate": 1000.0,
    }


def test_advanced_time_json_is_interpreted_directly_as_milliseconds(
    tmp_path: Path,
) -> None:
    labels = _labels()
    path = tmp_path / "advanced.json"
    path.write_text(json.dumps({"101": 34.5, "102": 7}), encoding="utf-8")

    values = labels.load_advanced_time_ms(path)

    assert values == {101: 34.5, 102: 7.0}
    assert labels.resolve_positive_horizon_ms(
        shot_id=101, device="J-TEXT", policy="advanced_floor", advanced_time_ms=values,
    ) == pytest.approx(34.5)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True])
def test_advanced_time_json_rejects_invalid_durations(
    tmp_path: Path, bad: object,
) -> None:
    labels = _labels()
    path = tmp_path / "advanced.json"
    path.write_text(json.dumps({"101": bad}), encoding="utf-8")

    with pytest.raises(ValueError, match="finite positive milliseconds"):
        labels.load_advanced_time_ms(path)


def test_jtext_uses_max_25ms_and_missing_falls_back_to_25ms() -> None:
    labels = _labels()
    values = {101: 10.0, 102: 40.0}

    assert labels.resolve_positive_horizon_ms(
        shot_id=101, device="J-TEXT", policy="advanced_floor", advanced_time_ms=values,
    ) == pytest.approx(25.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=102, device="J-TEXT", policy="advanced_floor", advanced_time_ms=values,
    ) == pytest.approx(40.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=103, device="J-TEXT", policy="advanced_floor", advanced_time_ms=values,
    ) == pytest.approx(25.0)


def test_east_uses_max_125ms_and_missing_falls_back_to_125ms() -> None:
    labels = _labels()
    values = {101: 100.0, 102: 171.0}

    assert labels.resolve_positive_horizon_ms(
        shot_id=101, device="EAST", policy="advanced_floor", advanced_time_ms=values,
    ) == pytest.approx(125.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=102, device="EAST", policy="advanced_floor", advanced_time_ms=values,
    ) == pytest.approx(171.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=103, device="EAST", policy="advanced_floor", advanced_time_ms=values,
    ) == pytest.approx(125.0)


def test_fixed_policy_remains_the_exact_125ms_control() -> None:
    labels = _labels()

    assert labels.resolve_positive_horizon_ms(
        shot_id=101, device="J-TEXT", policy="fixed125", advanced_time_ms={101: 80.0},
    ) == pytest.approx(125.0)
    assert labels.resolve_positive_horizon_ms(
        shot_id=101, device="EAST", policy="fixed125", advanced_time_ms={101: 171.0},
    ) == pytest.approx(125.0)


def test_attaching_horizons_never_changes_hdf5_outcome() -> None:
    labels = _labels()
    disruptive = _disruptive_shot(101)
    safe = dict(_disruptive_shot(102), is_disrupt=False, down=None)

    attached, audit = labels.attach_positive_horizons(
        [disruptive, safe], device="J-TEXT", policy="advanced_floor",
        advanced_time_ms={101: 40.0, 102: 200.0},
    )

    assert [shot["is_disrupt"] for shot in attached] == [True, False]
    assert attached[0]["positive_horizon_ms"] == pytest.approx(40.0)
    assert "positive_horizon_ms" not in attached[1]
    assert audit["disruptive_shots"] == 1
    assert audit["safe_shots_ignored"] == 1


def test_source_sampler_uses_the_per_shot_horizon() -> None:
    source = importlib.import_module("train_jtext_source_transformer")
    short = _disruptive_shot(101)
    long = _disruptive_shot(102)
    short["positive_horizon_ms"] = 25.0
    long["positive_horizon_ms"] = 40.0

    sampler = source.BalancedSourceSampler([short, long], seed=131)

    # Keep the production sampler's original direct floating-point comparison:
    # synthetic 25/40 ms boundaries evaluate infinitesimally above the limit.
    assert sampler.positive[0].tolist() == list(range(176, 201))
    assert sampler.positive[1].tolist() == list(range(161, 201))
    assert 175 in sampler.negative[0]
    assert 160 in sampler.negative[1]


def test_east_sampler_uses_the_per_shot_horizon() -> None:
    grid = importlib.import_module("grid_search")
    shot = _disruptive_shot(101, length=401)
    shot["down"] = 0.400
    shot["positive_horizon_ms"] = 171.0

    pools = grid.build_endpoint_pools(
        [shot], pos_ms=125.0, gray_ms=0.0, allow_left_padding=False,
    )

    assert pools[0]["positive"][0] == 229
    # Preserve the registered EAST sampler's strict lower bound ttd > 0 ms.
    assert pools[0]["positive"][-1] == 399
    assert pools[0]["negative"][-1] == 228


def test_comparison_matrix_is_factorial_not_a_conflated_two_arm_test() -> None:
    runner = importlib.import_module("run_advanced_time_label_ablation")

    assert runner.comparison_arms() == (
        ("source_fixed__east_fixed", "fixed125", "fixed125"),
        ("source_dynamic__east_fixed", "advanced_floor", "fixed125"),
        ("source_fixed__east_dynamic", "fixed125", "advanced_floor"),
        ("source_dynamic__east_dynamic", "advanced_floor", "advanced_floor"),
    )


def test_operational_metrics_count_very_early_alarm_as_timely() -> None:
    runner = importlib.import_module("run_advanced_time_label_ablation")
    rows = [
        {
            "shot": 1, "score": 0.9, "start_time_s": 0.0,
            "sample_rate_hz": 1000.0,
        },
        {
            "shot": 2, "score": 0.8, "start_time_s": 0.0,
            "sample_rate_hz": 1000.0,
        },
        {
            "shot": 3, "score": 0.7, "start_time_s": 0.0,
            "sample_rate_hz": 1000.0,
        },
    ]
    traces = {
        1: np.asarray([0.9], np.float32),
        2: np.asarray([0.8], np.float32),
        3: np.asarray([0.7], np.float32),
    }
    truth = {
        1: {"is_disrupt": True, "down_time_s": 4.0},
        2: {"is_disrupt": True, "down_time_s": 0.200},
        3: {"is_disrupt": False, "down_time_s": None},
    }

    result = runner.evaluate_threshold(
        rows, traces, truth, threshold=0.75, minimum_lead_ms=30.0,
    )

    assert result["raw_tpr"] == pytest.approx(1.0)
    assert result["operational_tpr"] == pytest.approx(1.0)
    assert result["fpr"] == pytest.approx(0.0)
    assert result["over_3000ms_count"] == 1

    scores, labels = runner.operational_selection_scores(
        rows, traces, truth, minimum_lead_ms=30.0,
    )
    np.testing.assert_array_equal(labels, [True, True, False])
    np.testing.assert_allclose(scores, [0.9, 0.8, 0.7])
