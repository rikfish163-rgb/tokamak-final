from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts"
    / "build_dynamic_mainline_evidence.py"
)


def module():
    spec = importlib.util.spec_from_file_location(
        "build_dynamic_mainline_evidence", SCRIPT,
    )
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prediction_seal_rejects_any_outcome_bearing_key() -> None:
    evidence = module()

    with pytest.raises(
        evidence.EvidenceContractError,
        match=r"outcome-bearing key.*is_disrupt",
    ):
        evidence.validate_prediction_rows([
            {
                "stream_index": 0,
                "shot": 101,
                "score": 0.9,
                "is_disrupt": True,
            },
        ])


def test_prediction_seal_requires_a_complete_hash_bound_manifest(
    tmp_path: Path,
) -> None:
    evidence = module()
    predictions = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "static_seal_manifest.json"
    _write_jsonl(predictions, [
        {"stream_index": 0, "shot": 101, "score": 0.9},
    ])
    manifest.write_text(json.dumps({
        "schema_version": "selected-static-seal-v1",
        "row_count": 1,
    }), encoding="utf-8")

    with pytest.raises(
        evidence.EvidenceContractError,
        match="complete prediction SHA",
    ):
        evidence.load_prediction_seal(predictions, manifest)


def test_postseal_join_binds_outcomes_to_the_validated_prediction_seal(
    tmp_path: Path,
) -> None:
    evidence = module()
    predictions = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "static_seal_manifest.json"
    rows = [
        {"stream_index": 0, "shot": 101, "score": 0.9, "alarm_time_s": 4.8},
        {"stream_index": 1, "shot": 102, "score": 0.1, "alarm_time_s": None},
    ]
    _write_jsonl(predictions, rows)
    manifest.write_text(json.dumps({
        "schema_version": "selected-static-seal-v1",
        "row_count": 2,
        "prediction_sha256": _sha256_file(predictions),
    }, sort_keys=True), encoding="utf-8")
    seal = evidence.load_prediction_seal(predictions, manifest)

    joined = evidence.build_postseal_outcome_join(seal, [
        {"shot": 102, "is_disrupt": False, "down_time_s": None},
        {"shot": 101, "is_disrupt": True, "down_time_s": 5.0},
    ])

    assert joined["status"] == "available"
    assert [row["shot"] for row in joined["rows"]] == [101, 102]
    assert joined["rows"][0]["is_disrupt"] is True
    assert joined["rows"][1]["is_disrupt"] is False
    assert joined["prediction_seal_sha256"] == seal["seal_manifest_sha256"]
    assert len(joined["truth_join_sha256"]) == 64
    assert len(joined["shot_order_sha256"]) == 64


def test_jtext_val_threshold_recomputes_raw_and_operational_metrics() -> None:
    evidence = module()
    rows = [
        {
            "shot": 1, "score": 0.9, "alarmed": False,
            "alarm_time_s": 4.9, "is_disrupt": True, "down_time_s": 5.0,
        },
        {
            "shot": 2, "score": 0.8, "alarm_time_s": 4.99,
            "is_disrupt": True, "down_time_s": 5.0,
        },
        {
            "shot": 3, "score": 0.4, "alarm_time_s": None,
            "is_disrupt": True, "down_time_s": 5.0,
        },
        {
            "shot": 4, "score": 0.7, "alarm_time_s": 2.0,
            "is_disrupt": False, "down_time_s": None,
        },
        {
            "shot": 5, "score": 0.1, "alarm_time_s": None,
            "is_disrupt": False, "down_time_s": None,
        },
    ]

    result = evidence.compute_operational_metrics(
        rows,
        device="jtext",
        threshold_receipt={
            "value": 0.5,
            "source": "J-TEXT VAL160",
            "sha256": "a" * 64,
        },
    )

    assert result["status"] == "available"
    assert result["counts"] == {
        "n_shots": 5,
        "n_disruptions": 3,
        "n_safe": 2,
        "raw_TP": 2,
        "raw_FN": 1,
        "TARDY": 1,
        "operational_TP": 1,
        "operational_FN": 2,
        "FP": 1,
        "TN": 1,
    }
    assert result["rates"]["raw_tpr"]["value"] == pytest.approx(2 / 3)
    assert result["rates"]["operational_tpr"]["value"] == pytest.approx(1 / 3)
    assert result["rates"]["fpr"]["value"] == pytest.approx(0.5)
    assert result["rates"]["j"]["value"] == pytest.approx(-1 / 6)
    assert result["rates"]["auc"]["value"] == pytest.approx(5 / 6)
    assert result["lead_distributions"]["raw_alarm_lead_ms"] == {
        "status": "available",
        "n": 2,
        "median": pytest.approx(55.0),
        "q1": pytest.approx(32.5),
        "q3": pytest.approx(77.5),
    }
    assert result["lead_distributions"]["operational_tp_lead_ms"] == {
        "status": "available",
        "n": 1,
        "median": pytest.approx(100.0),
        "q1": pytest.approx(100.0),
        "q3": pytest.approx(100.0),
    }
    assert result["per_shot"][1]["tardy"] is True
    assert result["per_shot"][1]["operational_tp"] is False


def test_east_nominal_thirty_ms_lead_is_operational() -> None:
    evidence = module()

    result = evidence.compute_operational_metrics(
        [{
            "shot": 1,
            "score": 0.9,
            "alarm_time_s": 4.97,
            "is_disrupt": True,
            "down_time_s": 5.0,
        }],
        device="east",
        threshold_receipt={
            "value": 0.5,
            "source": "EAST support110",
            "sha256": "b" * 64,
        },
    )

    assert result["counts"]["raw_TP"] == 1
    assert result["counts"]["TARDY"] == 0
    assert result["counts"]["operational_TP"] == 1


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        (0, 10, (0.0, 0.2775327998628892)),
        (10, 10, (0.7224672001371107, 1.0)),
        (138, 180, (0.6997021803760345, 0.8224868873991052)),
    ],
)
def test_wilson_interval_matches_fixed_physical_shot_examples(
    successes: int,
    total: int,
    expected: tuple[float, float],
) -> None:
    evidence = module()

    result = evidence.wilson_interval(successes, total)

    assert result["method"] == "wilson_score"
    assert result["alpha"] == pytest.approx(0.05)
    assert result["numerator"] == successes
    assert result["denominator"] == total
    assert result["interval"] == pytest.approx(expected)


def test_seed_summary_does_not_pool_seed_times_shot_as_independent() -> None:
    evidence = module()
    per_seed = {
        "131": {"rates": {"operational_tpr": {
            "status": "available", "value": 0.50,
            "numerator": 2, "denominator": 4,
        }}},
        "133": {"rates": {"operational_tpr": {
            "status": "available", "value": 0.75,
            "numerator": 3, "denominator": 4,
        }}},
        "137": {"rates": {"operational_tpr": {
            "status": "available", "value": 1.00,
            "numerator": 4, "denominator": 4,
        }}},
    }

    result = evidence.summarize_seed_rate(per_seed, "operational_tpr")

    assert result["aggregation"] == "equal_weight_over_model_seeds"
    assert result["model_seed_count"] == 3
    assert result["mean"] == pytest.approx(0.75)
    assert result["sample_sd"] == pytest.approx(0.25)
    assert result["per_seed_denominators"] == {"131": 4, "133": 4, "137": 4}
    assert result["uncertainty"] == "per_seed_wilson_only"
    assert "denominator" not in result
    assert "wilson_95" not in result


def _paired_metric_fixture(evidence):
    threshold = {
        "value": 0.5,
        "source": "J-TEXT VAL160",
        "sha256": "c" * 64,
    }
    truth = {
        1: (True, 5.0),
        2: (True, 5.0),
        3: (True, 5.0),
        4: (True, 5.0),
        5: (False, None),
        6: (False, None),
        7: (False, None),
    }

    def rows(scores_and_times):
        return [
            {
                "shot": shot,
                "score": score,
                "alarm_time_s": alarm_time,
                "is_disrupt": truth[shot][0],
                "down_time_s": truth[shot][1],
            }
            for shot, score, alarm_time in scores_and_times
        ]

    static = evidence.compute_operational_metrics(
        rows([
            (1, 0.9, 4.9),
            (2, 0.9, 4.95),
            (3, 0.9, 4.995),
            (4, 0.1, None),
            (5, 0.9, 2.0),
            (6, 0.1, None),
            (7, 0.1, None),
        ]),
        device="jtext",
        threshold_receipt=threshold,
    )
    online = evidence.compute_operational_metrics(
        rows([
            (1, 0.9, 4.88),
            (2, 0.1, None),
            (3, 0.9, 4.92),
            (4, 0.9, 4.96),
            (5, 0.1, None),
            (6, 0.9, 2.0),
            (7, 0.1, None),
        ]),
        device="jtext",
        threshold_receipt=threshold,
    )
    hashes = {
        "shot_order_sha256": "1" * 64,
        "truth_join_sha256": "2" * 64,
        "evaluation_mask_sha256": "3" * 64,
    }
    return static, online, hashes


def test_static_online_delta_is_recomputed_from_shot_paired_indicators() -> None:
    evidence = module()
    static, online, hashes = _paired_metric_fixture(evidence)

    result = evidence.compute_shot_paired_delta(
        static,
        online,
        static_hashes=hashes,
        online_hashes=hashes,
        baseline_name="static",
        candidate_name="online",
    )

    assert result["status"] == "available"
    assert result["binary_deltas"]["operational_tpr"]["value"] == pytest.approx(0.25)
    assert result["binary_deltas"]["operational_tpr"]["discordance"] == {
        "static_only": 1,
        "online_only": 2,
        "both": 1,
        "neither": 0,
    }
    assert result["binary_deltas"]["fpr"]["value"] == pytest.approx(0.0)
    assert result["binary_deltas"]["fpr"]["discordance"] == {
        "static_only": 1,
        "online_only": 1,
        "both": 0,
        "neither": 1,
    }
    assert result["binary_deltas"]["j"]["value"] == pytest.approx(0.25)
    timing = result["timing_deltas"]
    assert timing["armwise_operational_median_difference_ms"] == {
        "status": "available",
        "value": pytest.approx(5.0),
        "paired": False,
        "composition_sensitive": True,
    }
    assert timing["common_operational_tp_lead_delta_ms"] == {
        "status": "available",
        "n_both": 1,
        "median": pytest.approx(20.0),
        "q1": pytest.approx(20.0),
        "q3": pytest.approx(20.0),
        "paired": True,
        "direction": "online_minus_static",
    }


@pytest.mark.parametrize(
    "mismatched_key",
    ["shot_order_sha256", "truth_join_sha256", "evaluation_mask_sha256"],
)
def test_static_online_delta_fails_closed_on_any_pairing_hash_mismatch(
    mismatched_key: str,
) -> None:
    evidence = module()
    static, online, hashes = _paired_metric_fixture(evidence)
    changed = dict(hashes)
    changed[mismatched_key] = "f" * 64

    with pytest.raises(evidence.EvidenceContractError, match=mismatched_key):
        evidence.compute_shot_paired_delta(
            static,
            online,
            static_hashes=hashes,
            online_hashes=changed,
            baseline_name="static",
            candidate_name="online",
        )


def _resource_fixture() -> tuple[list[dict], list[dict]]:
    queries = [
        {"stream_index": 0, "shot": 100, "block": 0, "selected": False},
        {"stream_index": 1, "shot": 101, "block": 0, "selected": True},
        {"stream_index": 2, "shot": 102, "block": 1, "selected": False},
        {"stream_index": 3, "shot": 103, "block": 1, "selected": True},
    ]
    updates = []
    for attempt_index, (stream_index, shot, replay_windows) in enumerate([
        (1, 101, 0),
        (3, 103, 8),
    ]):
        flop_inputs = {
            "current_windows": 8,
            "replay_windows": replay_windows,
            "guard_windows": 8,
            "optimizer_attempts": 2,
            "online_parameter_count": 10,
        }
        updates.append({
            "stream_index": stream_index,
            "shot": shot,
            "accepted": True,
            "steps_attempted": 2,
            "steps_committed": 2,
            "compute": {
                "attempt_index": attempt_index,
                "current_windows": 8,
                "replay_windows": replay_windows,
                "guard_windows": 8,
                "optimizer_attempts": 2,
                "replay_size": attempt_index,
                "flop_inputs": flop_inputs,
                "reported_flops": 2 * 10 * (16 + replay_windows) * 2,
            },
        })
    return queries, updates


def test_query_update_resource_audit_recomputes_every_receipt() -> None:
    evidence = module()
    queries, updates = _resource_fixture()

    result = evidence.audit_query_update_resources(
        queries,
        updates,
        expected_shots=4,
        block_size=2,
    )

    assert result["status"] == "available"
    assert result["queries"] == {
        "decision_count": 4,
        "selected_count": 2,
        "selected_per_block": {"0": 1, "1": 1},
    }
    assert result["updates"] == {
        "receipt_count": 2,
        "optimizer_attempts": 4,
        "accepted_updates": 2,
        "rollbacks": 0,
        "steps_committed": 4,
    }
    assert result["resources"]["window_totals"] == {
        "current": 16,
        "replay": 8,
        "guard": 16,
    }
    assert result["resources"]["replay_size_sequence"] == [0, 1]
    assert result["resources"]["reported_flops"] == 1600
    assert len(result["resources"]["flop_inputs_sha256"]) == 64


def test_active_random_resource_mismatch_suppresses_selector_estimand() -> None:
    evidence = module()
    active_queries, active_updates = _resource_fixture()
    random_queries, random_updates = _resource_fixture()
    random_updates[1]["compute"]["replay_size"] = 2
    active = evidence.audit_query_update_resources(
        active_queries, active_updates, expected_shots=4, block_size=2,
    )
    random = evidence.audit_query_update_resources(
        random_queries, random_updates, expected_shots=4, block_size=2,
    )

    result = evidence.compare_matched_resources(active, random)

    assert result["status"] == "mismatch"
    assert result["selector_estimand_status"] == "suppressed"
    assert "replay_size_sequence" in result["mismatched_fields"]
