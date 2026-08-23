from __future__ import annotations

import importlib
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def test_historical_parent_sampler_uses_the_per_shot_horizon() -> None:
    """The fidelity sampler, not only the newer generic sampler, must change."""

    frontload = importlib.import_module("run_equal_label_budget_ablation")
    short = _disruptive_shot(101, length=401)
    long = _disruptive_shot(102, length=401)
    short["down"] = long["down"] = 0.400
    short["positive_horizon_ms"] = 25.0
    long["positive_horizon_ms"] = 171.0

    pools = frontload.historical_target_endpoint_pools(
        [short, long], pos_ms=125.0, gray_ms=0.0,
    )

    assert len(pools[0]["positive"]) < len(pools[1]["positive"])
    # Preserve the historical parent's direct floating-point comparison.
    assert pools[0]["positive"][0] == 376
    assert pools[1]["positive"][0] == 229


def test_shared_hazard_supervision_uses_the_per_shot_horizon() -> None:
    shared = importlib.import_module("run_dynamic_hazard_adapter")
    times = np.asarray([0.300, 0.360, 0.390], dtype=np.float64)
    shot = {
        "shot": 101, "is_disrupt": True, "down": 0.400,
        "positive_horizon_ms": 25.0,
    }

    labels = shared.support_window_labels(shot, times)

    np.testing.assert_array_equal(labels, [False, False, True])


def test_private_raw8_supervision_uses_the_per_shot_horizon() -> None:
    private = importlib.import_module("run_east_private23_online")
    shot = _disruptive_shot(101, length=201)
    shot["positive_horizon_ms"] = 25.0

    binary, soft = private.temporal_targets(shot, risk_length=74)

    # The first risk endpoint is 127 ms. At 73 ms TTD it must be negative
    # under the 25 ms label, although it was positive under fixed 125 ms.
    assert not bool(binary[0])
    assert soft[0] == pytest.approx(0.0)
    assert bool(binary[-1])
    assert soft[-1] > 0.0


def test_comparison_matrix_is_factorial_not_a_conflated_two_arm_test() -> None:
    runner = importlib.import_module("run_advanced_time_label_ablation")

    assert runner.comparison_arms() == (
        ("source_fixed__east_fixed", "fixed125", "fixed125"),
        ("source_dynamic__east_fixed", "advanced_floor", "fixed125"),
        ("source_fixed__east_dynamic", "fixed125", "advanced_floor"),
        ("source_dynamic__east_dynamic", "advanced_floor", "advanced_floor"),
    )


def test_corrected_runner_requires_the_full_static_pipeline() -> None:
    runner = importlib.import_module("run_advanced_time_label_ablation")

    assert runner.FULL_STATIC_PIPELINE == (
        "encoder-head+shared-hazard+east-private-raw8+dual-threshold-v1"
    )
    runner.assert_fixed_control(
        {"tpr_at_30ms": 148 / 180, "fpr": 28 / 180}, seed=131,
    )
    runner.assert_fixed_control(
        {"tpr_at_30ms": 151 / 180, "fpr": 26 / 180}, seed=133,
    )
    runner.assert_fixed_control(
        {"tpr_at_30ms": 155 / 180, "fpr": 36 / 180}, seed=137,
    )
    with pytest.raises(RuntimeError, match="fixed/fixed control"):
        runner.assert_fixed_control(
            {"tpr_at_30ms": 0.5277777778, "fpr": 0.0611111111}, seed=131,
        )


def test_dynamic_source_command_uses_the_trainer_default_test_evaluation(
    tmp_path: Path,
) -> None:
    runner = importlib.import_module("run_advanced_time_label_ablation")

    command = runner._source_command(
        seed=131,
        policy="advanced_floor",
        output=tmp_path / "source",
        advanced_time=tmp_path / "advanced.json",
    )

    # The source trainer evaluates TEST220 by default.  It exposes only the
    # inverse switch (--skip-test), so the orchestration command must contain
    # neither an invented positive switch nor the skip switch.
    assert "--evaluate-test" not in command
    assert "--skip-test" not in command


def test_operational_counts_do_not_relabel_late_raw_alarms_as_timely() -> None:
    runner = importlib.import_module("run_advanced_time_label_ablation")
    metrics = {
        "TP": 171,
        "FN": 9,
        "tpr_at_30ms": 148 / 180,
    }

    runner.add_operational_confusion_counts(metrics, disruptions=180)

    assert metrics["raw_alarm_TP"] == 171
    assert metrics["raw_alarm_FN"] == 9
    assert metrics["timely_TP_at_30ms"] == 148
    assert metrics["timely_FN_at_30ms"] == 32
    assert metrics["confusion_count_semantics"] == (
        "TP/FN and raw_alarm_TP/FN count any alarm occurrence; "
        "timely_TP_at_30ms/timely_FN_at_30ms enforce lead>=30ms"
    )


def test_operational_metrics_count_very_early_alarm_as_timely() -> None:
    unified = importlib.import_module("run_unified_online_active")
    rows = [
        {"shot": 1, "alarmed": True, "alarm_time_s": 0.5},
        {"shot": 2, "alarmed": True, "alarm_time_s": 0.1},
        {"shot": 3, "alarmed": False, "alarm_time_s": None},
    ]
    truth = {
        1: {"is_disrupt": True, "down_time_s": 4.0},
        2: {"is_disrupt": True, "down_time_s": 0.200},
        3: {"is_disrupt": False, "down_time_s": None},
    }

    result = unified.alarm_metrics(rows, truth)

    assert result["tpr"] == pytest.approx(1.0)
    assert result["tpr_at_30ms"] == pytest.approx(1.0)
    assert result["fpr"] == pytest.approx(0.0)
    assert result["over_3000ms_count"] == 1


def test_released_v2_evidence_matches_every_prediction_seal() -> None:
    evidence = (
        HERE / "paper_grade_package/evidence/advanced_time_label_ablation_v2"
    )
    report = json.loads((evidence / "report.json").read_text())
    arms = (
        "source_fixed__east_fixed",
        "source_dynamic__east_fixed",
        "source_fixed__east_dynamic",
        "source_dynamic__east_dynamic",
    )
    metric_map = {
        "tpr_at_30ms": "tpr_at_30ms",
        "fpr": "fpr",
        "utility": "utility_tpr30_minus_fpr",
        "auc": "auc",
        "median_lead_ms": "median_lead_ms",
    }

    assert report["correction"]["invalid_v1_metrics_must_not_be_used"] is True
    for seed in report["seeds"]:
        source_metrics = json.loads((
            evidence / f"seed{seed}/source_dynamic/test220_metrics.json"
        ).read_text())["metrics"]
        source_report = report["source_test220"]["seeds"][str(seed)][
            "advanced_floor"
        ]
        assert source_report["tpr_at_10ms"] == pytest.approx(
            source_metrics["timely_tpr"]
        )
        assert source_report["fpr"] == pytest.approx(source_metrics["fpr"])
        assert source_report["auc"] == pytest.approx(source_metrics["auc"])

        for arm in arms:
            arm_dir = evidence / f"seed{seed}/targets/{arm}"
            summary = json.loads((arm_dir / "summary.json").read_text())
            seal_path = arm_dir / "balanced360_prediction_seal.jsonl"
            rows = [
                json.loads(line) for line in seal_path.read_text().splitlines()
                if line.strip()
            ]
            released = report["east_balanced360"][arm]["seeds"][str(seed)]

            assert len(rows) == 360
            assert [row["stream_index"] for row in rows] == list(range(360))
            assert len({int(row["shot"]) for row in rows}) == 360
            assert _sha256(seal_path) == summary["prediction_seal_sha256"]
            assert released["prediction_seal_sha256"] == _sha256(seal_path)
            assert summary["balanced360_outcomes_read_after_complete_prediction_seal"]
            assert not summary["balanced360_used_for_training_or_threshold"]
            assert released["threshold"] == pytest.approx(
                summary["threshold"]["value"]
            )
            for report_key, summary_key in metric_map.items():
                assert released[report_key] == pytest.approx(
                    summary["metrics"][summary_key]
                )
            expected_timely_tp = round(180 * summary["metrics"]["tpr_at_30ms"])
            assert released["timely_TP_at_30ms"] == expected_timely_tp
            assert released["timely_FN_at_30ms"] == 180 - expected_timely_tp
            if arm == "source_fixed__east_fixed":
                assert summary["prediction_fidelity"]["passed"] is True

    for arm in arms:
        for report_key in metric_map:
            values = [
                report["east_balanced360"][arm]["seeds"][str(seed)][report_key]
                for seed in report["seeds"]
            ]
            assert report["east_balanced360"][arm]["mean"][report_key] == pytest.approx(
                float(np.mean(values))
            )
