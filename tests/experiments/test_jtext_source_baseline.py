from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load_subject():
    path = ROOT / "experiments/transfer_methods_2026/jtext_source_baseline.py"
    spec = importlib.util.spec_from_file_location("jtext_source_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(*, risk, times, disruptive, down=None, shot=1):
    return {
        "shot": shot,
        "risk": np.asarray(risk, dtype=np.float64),
        "times": np.asarray(times, dtype=np.float64),
        "is_disrupt": disruptive,
        "down_time_s": down,
    }


def test_jtext_alarm_requires_strictly_more_than_ten_ms_lead() -> None:
    subject = _load_subject()
    rows = [
        _row(risk=[0.9], times=[0.0], disruptive=True, down=0.010, shot=1),
        _row(risk=[0.9], times=[0.0], disruptive=True, down=0.010001, shot=2),
        _row(risk=[0.6], times=[0.0], disruptive=False, shot=3),
        _row(risk=[0.4], times=[0.0], disruptive=False, shot=4),
    ]

    result = subject.evaluate_threshold(rows, threshold=0.5, minimum_lead_ms=10.0)

    assert result["TP"] == 1
    assert result["FN"] == 1
    assert result["FP"] == 1
    assert result["TN"] == 1
    assert result["tpr"] == 0.5
    assert result["fpr"] == 0.5
    assert result["median_lead_ms"] > 10.0


def test_jtext_alarm_uses_first_causal_crossing_for_lead() -> None:
    subject = _load_subject()
    rows = [
        _row(
            risk=[0.1, 0.8, 0.95],
            times=[0.00, 0.40, 0.49],
            disruptive=True,
            down=0.50,
        )
    ]

    result = subject.evaluate_threshold(rows, threshold=0.75, minimum_lead_ms=10.0)

    assert result["TP"] == 1
    assert result["median_lead_ms"] == 100.0 - 0.0
    assert result["shots"][0]["alarm_index"] == 1


def test_threshold_is_selected_on_validation_for_target_tpr_then_minimum_fpr() -> None:
    subject = _load_subject()
    rows = [
        _row(risk=[0.90], times=[0.0], disruptive=True, down=0.020, shot=1),
        _row(risk=[0.80], times=[0.0], disruptive=True, down=0.020, shot=2),
        _row(risk=[0.85], times=[0.0], disruptive=False, shot=3),
        _row(risk=[0.20], times=[0.0], disruptive=False, shot=4),
    ]

    selected = subject.select_validation_threshold(
        rows, target_tpr=1.0, minimum_lead_ms=10.0,
    )

    assert selected["threshold"] == 0.8
    assert selected["metrics"]["tpr"] == 1.0
    assert selected["metrics"]["fpr"] == 0.5


def test_unreachable_target_reports_context_ceiling_without_using_test() -> None:
    subject = _load_subject()
    rows = [
        _row(risk=[0.90], times=[0.0], disruptive=True, down=0.020, shot=1),
        _row(risk=[], times=[], disruptive=True, down=0.005, shot=2),
        _row(risk=[0.80], times=[0.0], disruptive=False, shot=3),
        _row(risk=[0.20], times=[0.0], disruptive=False, shot=4),
    ]

    selected = subject.select_validation_threshold(
        rows, target_tpr=1.0, minimum_lead_ms=10.0,
    )

    assert selected["nominal_target_reached"] is False
    assert selected["maximum_structural_tpr"] == 0.5
    assert selected["effective_target_tpr"] == 0.5
    assert selected["threshold"] == 0.9
    assert selected["metrics"]["tpr"] == 0.5
    assert selected["metrics"]["fpr"] == 0.0


def test_published_jtext_roles_are_1353_160_220_and_disjoint() -> None:
    roles = {
        name: {
            int(value)
            for value in (ROOT / "splits" / f"{name}.txt").read_text().split()
        }
        for name in ("train", "val", "test")
    }

    assert {name: len(values) for name, values in roles.items()} == {
        "train": 1353,
        "val": 160,
        "test": 220,
    }
    assert roles["train"].isdisjoint(roles["val"])
    assert roles["train"].isdisjoint(roles["test"])
    assert roles["val"].isdisjoint(roles["test"])


def test_source_sampler_balances_window_classes_not_raw_shot_prevalence() -> None:
    subject = _load_subject()
    width = subject.GS.W
    disruptive = {
        "shot": 1,
        "v": np.ones((width + 80, 2), dtype=np.float32),
        "a": np.ones((width + 80, 2), dtype=bool),
        "is_disrupt": True,
        "down": (width + 80) / 1000.0,
        "start": 0.0,
        "rate": 1000.0,
    }
    safe = [
        {
            **disruptive,
            "shot": shot,
            "is_disrupt": False,
            "down": None,
        }
        for shot in range(2, 11)
    ]
    sampler = subject.GS.build_sampler(
        [disruptive, *safe], None, pos_ms=125.0, gray_ms=0.0, seed=131,
    )

    labels = np.asarray([sampler(0.5)[1] for _ in range(4000)])

    assert 0.47 <= float(labels.mean()) <= 0.53


def test_causal_left_padding_uses_no_future_value() -> None:
    subject = _load_subject()
    values = np.asarray([[1.0], [2.0], [99.0]], dtype=np.float32)

    window = subject.GS.causal_window(values, endpoint=1, width=4)

    np.testing.assert_array_equal(window[:, 0], [1.0, 1.0, 1.0, 2.0])
    assert 99.0 not in window


def test_padded_sampler_recovers_short_shot_but_excludes_tardy_band() -> None:
    subject = _load_subject()
    values = np.arange(50, dtype=np.float32)[:, None]
    disruptive = {
        "shot": 1,
        "v": values,
        "a": np.ones_like(values, dtype=bool),
        "is_disrupt": True,
        "down": 0.050,
        "start": 0.0,
        "rate": 1000.0,
    }
    sampler = subject.GS.build_sampler(
        [disruptive], None, pos_ms=125.0, gray_ms=0.0, seed=131,
        minimum_pos_ms=10.0, allow_left_padding=True,
        context_lengths=(subject.GS.W,),
    )

    endpoints = [int(sampler(1.0)[0][-1, 0]) for _ in range(200)]

    assert min(endpoints) >= 0
    assert max(endpoints) <= 39  # endpoint 40 has exactly 10 ms lead and is excluded


def test_committed_source_reference_is_internally_consistent() -> None:
    import json

    reference = json.loads(
        (ROOT / "results/reference/jtext_source_multiseed_summary.json").read_text()
    )

    assert reference["protocol"] == (
        "jtext-rebuilt17-source-test220-lead-gt10-padfix-v2"
    )
    assert [row["seed"] for row in reference["seeds"]] == [131, 133, 137]
    for row in reference["seeds"]:
        assert row["TP"] + row["FN"] == 110
        assert row["FP"] + row["TN"] == 110
        assert row["tpr_at_gt10ms"] == row["TP"] / 110
        assert row["fpr"] == row["FP"] / 110
    assert reference["admission"]["passed"] is False
