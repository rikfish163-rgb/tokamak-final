import json

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.alarm_label_audit import audit_rows, main


torch.set_num_threads(2)


def _row(
    probabilities,
    times,
    labels,
    *,
    shot=1001,
    is_disrupt=True,
    down=0.1,
    observed=None,
):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    base_logits = np.log(probabilities / (1.0 - probabilities)).astype(np.float32)
    times = np.asarray(times, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float32)
    if observed is None:
        observed = np.ones(len(times), dtype=bool)
    return {
        "shot": shot,
        "features": np.zeros((len(times), 64), dtype=np.float32),
        "base_logits": base_logits,
        "observed": np.asarray(observed, dtype=bool),
        "y": labels,
        "times": times,
        "is_disrupt": is_disrupt,
        "down": down,
    }


def test_first_alarm_is_used_instead_of_maximum_risk_point():
    report = audit_rows(
        [_row([0.80, 0.99, 0.20], [0.00, 0.01, 0.02], [0, 0, 1], down=0.04)],
        threshold=0.70,
    )

    record = report["rows"][0]
    assert record["first_alarm_index"] == 0
    assert record["maximum_risk_index"] == 1
    assert record["first_alarm_is_maximum_risk"] is False
    assert record["first_alarm_label"] == 0
    assert report["timely_TP"]["count"] == 1
    assert report["timely_TP"]["first_alarm_label0"] == 1
    assert report["timely_TP"]["first_alarm_label1"] == 0
    assert report["lead_distribution_ms"]["first_alarm"] == [40.0]


def test_cached_y_is_authoritative_and_not_rebuilt_with_a_25_ms_default():
    report = audit_rows(
        [_row([0.90], [0.00], [1], down=0.10)],
        threshold=0.50,
    )

    record = report["rows"][0]
    assert record["first_alarm_label"] == 1
    assert report["timely_TP"]["first_alarm_label1"] == 1
    assert report["positive_label_windows"] == {
        "total": 1,
        "lead_gt_10": 1,
        "lead_le_10": 0,
    }


def test_lead_exactly_10_ms_is_not_strictly_actionable():
    report = audit_rows(
        [_row([0.90], [0.09], [1], down=0.10)],
        threshold=0.50,
    )

    record = report["rows"][0]
    assert record["lead_ms"] == pytest.approx(10.0)
    assert record["first_alarm_label"] == 1
    assert record["timely_tp"] is False
    assert report["timely_TP"]["count"] == 0
    assert report["positive_label_windows"] == {
        "total": 1,
        "lead_gt_10": 0,
        "lead_le_10": 1,
    }


def test_positive_label_windows_count_actionable_and_non_actionable():
    report = audit_rows(
        [
            _row(
                [0.10, 0.10, 0.10, 0.10],
                [0.00, 0.08, 0.09, 0.095],
                [0, 1, 1, 1],
                down=0.10,
            )
        ],
        threshold=0.50,
    )

    assert report["positive_label_windows"] == {
        "total": 3,
        "lead_gt_10": 1,
        "lead_le_10": 2,
    }


def test_no_alarm_has_no_first_alarm_or_timely_tp():
    report = audit_rows(
        [_row([0.10, 0.20], [0.00, 0.01], [1, 1], down=0.10)],
        threshold=0.50,
    )

    record = report["rows"][0]
    assert record["first_alarm_index"] is None
    assert record["first_alarm_label"] is None
    assert report["timely_TP"]["count"] == 0
    assert report["lead_distribution_ms"]["first_alarm"] == []
    assert report["safeFP"] == 0


def test_safe_alarm_is_counted_as_safe_fp():
    report = audit_rows(
        [
            _row(
                [0.90, 0.10],
                [0.00, 0.01],
                [0, 0],
                shot=2001,
                is_disrupt=False,
                down=None,
            )
        ],
        threshold=0.50,
    )

    record = report["rows"][0]
    assert record["first_alarm_index"] == 0
    assert record["safe_fp"] is True
    assert report["safeFP"] == 1
    assert report["timely_TP"]["count"] == 0
    assert report["positive_label_windows"]["total"] == 0


def test_observed_false_does_not_suppress_the_model_alarm():
    report = audit_rows(
        [_row([0.90], [0.00], [1], observed=[False], down=0.10)],
        threshold=0.50,
    )

    assert report["rows"][0]["first_alarm_index"] == 0
    assert report["timely_TP"]["count"] == 1


def test_cli_reads_real_cache_shape_and_rejects_test_partition(tmp_path):
    cache = tmp_path / "source_features.pt"
    output = tmp_path / "audit.json"
    torch.save(
        {
            "parent_sha256": "parent",
            "train": [],
            "val": [_row([0.90], [0.00], [1], shot=3001)],
        },
        cache,
    )

    assert main(
        [
            "--cache",
            str(cache),
            "--threshold",
            "0.5",
            "--output",
            str(output),
        ]
    ) == 0
    saved = json.loads(output.read_text())
    assert saved["threshold"] == 0.5
    assert saved["n_rows"] == 1

    test_cache = tmp_path / "test_features.pt"
    torch.save({"parent_sha256": "parent", "train": [], "test": []}, test_cache)
    with pytest.raises(ValueError, match="VAL"):
        main(
            [
                "--cache",
                str(test_cache),
                "--threshold",
                "0.5",
                "--output",
                str(tmp_path / "test-audit.json"),
            ]
        )
