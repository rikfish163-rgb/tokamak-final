from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _audit():
    return importlib.import_module("audit_online_physics_attribution")


def test_factorial_effects_separate_intra_inter_and_interaction() -> None:
    audit = _audit()
    results = {
        "static": {"tpr_at_30ms": 0.80, "fpr": 0.20, "auc": 0.90},
        "intra_only": {"tpr_at_30ms": 0.82, "fpr": 0.20, "auc": 0.91},
        "inter_only": {"tpr_at_30ms": 0.80, "fpr": 0.15, "auc": 0.92},
        "intra_inter": {"tpr_at_30ms": 0.84, "fpr": 0.14, "auc": 0.94},
    }

    effects = audit.factorial_effects(results, tpr_key="tpr_at_30ms")

    assert effects["objective_j"]["intra_main"] == pytest.approx(0.02)
    assert effects["objective_j"]["inter_main"] == pytest.approx(0.05)
    assert effects["objective_j"]["joint_total"] == pytest.approx(0.10)
    assert effects["objective_j"]["interaction"] == pytest.approx(0.03)


def test_query_signature_is_ordered_and_label_free() -> None:
    audit = _audit()
    rows = [
        {"stream_index": 0, "shot": 10, "selected": False},
        {"stream_index": 1, "shot": 11, "selected": True},
    ]

    assert audit.query_signature(rows) == ((0, 10, False), (1, 11, True))

    with pytest.raises(ValueError, match="strictly ordered"):
        audit.query_signature([rows[1], rows[0]])
    with pytest.raises(ValueError, match="outcome field"):
        audit.query_signature([{**rows[0], "is_disrupt": True}])
    with pytest.raises(ValueError, match="outcome field"):
        audit.query_signature([{
            **rows[0],
            "components": {"nested": [{"revealed_is_disrupt": False}]},
        }])


def test_physics_attribution_requires_a_measurable_real_advantage() -> None:
    audit = _audit()
    on = {
        "static": {"tpr_at_30ms": 0.80, "fpr": 0.20, "auc": 0.90},
        "intra_only": {"tpr_at_30ms": 0.80, "fpr": 0.20, "auc": 0.90},
        "inter_only": {"tpr_at_30ms": 0.80, "fpr": 0.20, "auc": 0.90},
        "intra_inter": {"tpr_at_30ms": 0.80, "fpr": 0.20, "auc": 0.90},
    }
    off = {name: dict(values) for name, values in on.items()}

    result = audit.physics_ablation_verdict(on, off, tpr_key="tpr_at_30ms")

    assert result["supported"] is False
    assert result["joint_delta_j"] == pytest.approx(0.0)
    assert result["joint_delta_auc"] == pytest.approx(0.0)


def test_query_targeting_reports_error_enrichment() -> None:
    audit = _audit()
    result = audit.query_targeting(
        {"TP": 5, "TN": 12, "FP": 4, "FN": 0, "TARDY": 1},
        stream_error_count=19,
        stream_count=220,
    )

    assert result["queries"] == 22
    assert result["queried_errors"] == 5
    assert result["query_error_rate"] == pytest.approx(5 / 22)
    assert result["stream_error_rate"] == pytest.approx(19 / 220)
    assert result["enrichment"] == pytest.approx((5 / 22) / (19 / 220))


def test_neural_effect_evidence_handles_one_shot_float_grid_without_overclaiming() -> None:
    audit = _audit()
    results = {
        "static": {"tpr_at_30ms": 0.80, "fpr": 0.20, "auc": 0.90},
        "intra_only": {"tpr_at_30ms": 0.80, "fpr": 0.20, "auc": 0.90},
        "inter_only": {"tpr_at_30ms": 0.8055555555555555, "fpr": 0.20, "auc": 0.90},
        "intra_inter": {"tpr_at_30ms": 0.8055555555555555, "fpr": 0.20, "auc": 0.90},
    }
    effects = audit.factorial_effects(results, tpr_key="tpr_at_30ms")

    evidence = audit.neural_effect_evidence(
        effects,
        effects,
        east_shots=180,
        jtext_shots=110,
        factorial_seed_count=1,
    )

    assert evidence["single_seed_effect_observed"] == {"intra": False, "inter": True}
    assert evidence["formal_supported_across_seeds"] == {"intra": False, "inter": False}

    replicated = audit.neural_effect_evidence(
        effects,
        effects,
        east_shots=180,
        jtext_shots=110,
        factorial_seed_count=3,
    )
    assert replicated["formal_supported_across_seeds"] == {"intra": False, "inter": True}


def test_frozen_lineage_is_loaded_as_separate_evidence(tmp_path: Path) -> None:
    audit = _audit()
    table = tmp_path / "candidate_table.json"
    table.write_text(json.dumps({
        "claim_boundary": "development_only",
        "aggregate": {"frozen_winner": {"static": {"tpr": 0.8}}},
        "winner": {"candidate": "frozen"},
    }))

    result = audit._frozen_architecture_evidence(table)

    assert result is not None
    assert result["available"] is True
    assert result["aggregate"]["static"]["tpr"] == pytest.approx(0.8)


def test_fixed_query_manifest_and_physics_controls_are_paired(tmp_path: Path) -> None:
    runner = importlib.import_module("run_two_timescale_online")
    rows = [
        {"stream_index": index, "shot": 100 + index, "selected": index < 2}
        for index in range(4)
    ]
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    loaded = runner.load_fixed_query_manifest(
        path, [100, 101, 102, 103], expected_queries=2,
    )
    assert len(loaded) == 4
    values = __import__("numpy").arange(20, dtype=float).reshape(5, 4)
    available = __import__("numpy").ones_like(values, dtype=bool)
    zero, zero_mask = runner.transform_physics_trace(
        values, available, transform="zero", seed=131, stream_index=0,
    )
    assert not zero.any()
    assert not zero_mask.any()

    with pytest.raises(ValueError, match="outcome field"):
        leaked = tmp_path / "leaked.jsonl"
        leaked.write_text(json.dumps({
            **rows[0], "nested": {"revealed_is_disrupt": True},
        }) + "\n")
        runner.load_fixed_query_manifest(leaked, [100], expected_queries=0)
