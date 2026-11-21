"""TDD contract tests for the sealed, post-hoc representation report."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.risk_aligned_online import representation_analysis as analysis


def _row(shot, *, disruptive, alarm, lead_ms):
    return {
        "shot": shot,
        "is_disrupt": disruptive,
        "alarm": alarm,
        "lead_ms": lead_ms,
        "timely": bool(disruptive and alarm and lead_ms is not None and lead_ms > 30.0),
    }


def test_strict_30ms_boundary_is_not_timely_after_nine_digit_rounding():
    assert analysis.classify_prediction(True, True, 30.0) == "TARDY"
    # The unrounded binary subtraction must not turn this exact 30 ms case into
    # a timely detection.
    assert analysis.rounded_lead_ms(1.0, 0.97) == 30.0
    assert analysis.classify_prediction(True, True, analysis.rounded_lead_ms(1.0, 0.97)) == "TARDY"
    assert analysis.classify_prediction(True, True, 30.000000001) == "TP"


def test_safe_score_equal_to_threshold_is_an_alarm():
    rows = [
        {"shot": 1, "is_disrupt": False, "down": None,
         "times": np.array([0.0]), "scores": np.array([0.5])},
    ]
    result = analysis.metrics_for_rows(rows, threshold=0.5)
    assert result["predictions"][0]["alarm"] is True
    assert result["fp"] == 1
    assert result["class_counts"] == {"TP": 0, "FN": 0, "TARDY": 0, "TN": 0, "FP": 1}


@pytest.mark.parametrize(
    ("rows", "missing"),
    [
        ([{"shot": 1, "is_disrupt": True, "down": 1.0,
           "times": np.array([0.9]), "scores": np.array([0.9])}], "safe"),
        ([{"shot": 1, "is_disrupt": False, "down": None,
           "times": np.array([0.0]), "scores": np.array([0.1])}], "disrupt"),
    ],
)
def test_metrics_use_none_for_missing_class(rows, missing):
    result = analysis.metrics_for_rows(rows, threshold=0.5)
    if missing == "safe":
        assert result["fpr"] is None
    else:
        assert result["tpr"] is None
    assert result["J"] is None


def test_paired_harmful_flip_is_classified_from_tp_to_fn():
    before = [_row(1, disruptive=True, alarm=True, lead_ms=40.0)]
    after = [_row(1, disruptive=True, alarm=False, lead_ms=None)]
    result = analysis.paired_flips(before, after, before_arm="base", after_arm="intra")
    assert result["harmful_count"] == 1
    assert result["helpful_count"] == 0
    assert result["fixed_threshold"]["timely_tp_lost"] == 1
    assert result["fixed_threshold"]["tpr_declined"] is True
    assert {tuple(item[k] for k in ("from", "to", "classification", "count"))
            for item in result["flips"]} == {("TP", "FN", "harmful", 1)}
    assert result["flips"][0]["shots"] == [1]


def _write_arm(run, arm, *, shot=1, scores=(0.5,), threshold=0.5):
    arm_dir = Path(run) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    stem = f"0001_{shot}"
    npz = arm_dir / f"{stem}_sealed.npz"
    np.savez(npz, shot=shot, times=np.arange(len(scores), dtype=float),
             scores=np.asarray(scores, dtype=float))
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    (arm_dir / f"{stem}_seal.json").write_text(json.dumps(
        {"shot": shot, "arm": arm, "threshold": threshold,
         "npz": npz.name, "sha256": digest}))
    (arm_dir / f"{stem}_audit.json").write_text(json.dumps({"shot": shot}))
    return arm_dir


def _write_multi_arm_run(tmp_path):
    run = tmp_path / "complete-run"
    threshold = 0.5
    traces = {
        "stats": [(1, True, (0.6,)), (2, True, (0.4,)), (3, False, (0.6,))],
        "intra": [(1, True, (0.6,)), (2, True, (0.6,)), (3, False, (0.1,))],
        "inter": [(1, True, (0.6,)), (2, True, (0.4,)), (3, False, (0.1,))],
        "both": [(1, True, (0.4,)), (2, True, (0.6,)), (3, False, (0.6,))],
    }
    outcomes = {1: (True, 1.0), 2: (True, 1.0), 3: (False, None)}
    summary = {}
    for arm, entries in traces.items():
        arm_dir = run / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, (shot, disruptive, scores) in enumerate(entries, 1):
            stem = f"{index:04d}_{shot}"
            npz = arm_dir / f"{stem}_sealed.npz"
            np.savez(npz, shot=shot, times=np.array([0.9]), scores=np.array(scores))
            digest = hashlib.sha256(npz.read_bytes()).hexdigest()
            (arm_dir / f"{stem}_seal.json").write_text(json.dumps(
                {"shot": shot, "arm": arm, "threshold": threshold,
                 "npz": npz.name, "sha256": digest}))
            (arm_dir / f"{stem}_audit.json").write_text(json.dumps({"shot": shot}))
            down = outcomes[shot][1]
            rows.append({"shot": shot, "is_disrupt": disruptive, "down": down,
                         "times": np.array([0.9]), "scores": np.array(scores)})
        measured = analysis.metrics_for_rows(rows, threshold)
        summary[arm] = {"observed_count": len(entries), "metrics": {
            key: [dict(item, **{}) for item in value] if key == "predictions" else value
            for key, value in measured.items()
        }}
        for prediction in summary[arm]["metrics"]["predictions"]:
            prediction.pop("class", None)
    (run / "summary.json").write_text(json.dumps(summary))
    return run, outcomes


def test_seal_sha_mismatch_is_rejected_before_outcome_read(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _write_arm(run, "contract")
    summary = {"contract": {"metrics": {"threshold": 0.5, "predictions": []}}}
    (run / "summary.json").write_text(json.dumps(summary))
    npz = run / "contract" / "0001_1_sealed.npz"
    npz.write_bytes(npz.read_bytes() + b"tampered")

    def outcome_must_not_be_read(*args, **kwargs):
        raise AssertionError("outcome read happened before SHA verification")

    monkeypatch.setattr(analysis.simple_data, "attach_outcome", outcome_must_not_be_read)
    with pytest.raises(ValueError, match="SHA|sha"):
        analysis.analyze_run(run, tmp_path / "report.json")


def test_prediction_count_must_match_summary_after_sha_verification(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _write_arm(run, "contract")
    (run / "summary.json").write_text(json.dumps({
        "contract": {"metrics": {"threshold": 0.5, "predictions": []}},
    }))
    monkeypatch.setattr(
        analysis.simple_data,
        "attach_outcome",
        lambda s, device, advanced: dict(s, valid=np.ones(len(s["times"]), bool),
                                         is_disrupt=False, down=None),
    )
    with pytest.raises(ValueError, match="prediction count"):
        analysis.analyze_run(run, tmp_path / "report.json")


def test_oracle_threshold_is_explicitly_diagnostic_and_caps_are_inclusive():
    rows = [
        {"shot": 1, "is_disrupt": True, "down": 1.0,
         "times": np.array([0.9]), "scores": np.array([0.6])},
        {"shot": 2, "is_disrupt": False, "down": None,
         "times": np.array([0.0]), "scores": np.array([0.6])},
        {"shot": 3, "is_disrupt": False, "down": None,
         "times": np.array([0.0]), "scores": np.array([0.1])},
    ]
    result = analysis.threshold_diagnostic(rows, lead=30.0)
    assert result["kind"] == "TEST_DRIVEN_ORACLE_DIAGNOSTIC"
    assert result["not_online_predeployed"] is True
    assert result["caps"]["0.20"]["fpr"] <= 0.20
    assert result["caps"]["0.15"]["fpr"] <= 0.15
    assert result["caps"]["0.10"]["fpr"] <= 0.10


def test_complete_report_uses_stats_base_stages_pairs_and_no_overwrite(tmp_path):
    run, outcomes = _write_multi_arm_run(tmp_path)

    def attach(s, device, advanced):
        disruptive, down = outcomes[int(s["shot"])]
        return dict(s, valid=np.ones(len(s["times"]), bool),
                    is_disrupt=disruptive, down=down)

    output = tmp_path / "report.json"
    result = analysis.analyze_run(run, output, outcome_loader=attach)
    assert result["base_arm"] == "stats"
    assert set(result["paired_flips"]) == {"intra_vs_base", "inter_vs_base", "both_vs_inter"}
    assert result["paired_flips"]["intra_vs_base"]["helpful_count"] == 2
    assert result["paired_flips"]["inter_vs_base"]["helpful_count"] == 1
    assert result["paired_flips"]["inter_vs_base"]["flips"][0]["shots"] == [3]
    assert result["paired_flips"]["both_vs_inter"]["harmful_count"] == 2
    assert result["paired_flips"]["both_vs_inter"]["fixed_threshold"] == {
        "before_tp": 1,
        "after_tp": 1,
        "timely_tp_gained": 1,
        "timely_tp_lost": 1,
        "tpr_before": 0.5,
        "tpr_after": 0.5,
        "tpr_delta": 0.0,
        "tpr_declined": False,
    }
    assert result["arms"]["intra"]["feedback_stages"][0]["cumulative"]["shot_count"] == 1
    assert result["verification"]["sealed_sha256_verified"] is True
    assert json.loads(output.read_text())["format"] == "representation-analysis-v1"
    with pytest.raises(FileExistsError, match="overwrite"):
        analysis.analyze_run(run, output, outcome_loader=attach)
