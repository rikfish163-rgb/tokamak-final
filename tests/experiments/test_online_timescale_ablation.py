from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "online_timescale_ablation.py"
    spec = importlib.util.spec_from_file_location("online_timescale_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_timescale_modes_have_non_overlapping_authority() -> None:
    module = load_module()

    intra = module.timescale_contract("intra_only")
    inter = module.timescale_contract("inter_only")
    joint = module.timescale_contract("intra_inter")

    assert intra.enable_intra
    assert not intra.enable_query
    assert not intra.enable_reveal
    assert not intra.enable_inter

    assert not inter.enable_intra
    assert inter.enable_query
    assert inter.enable_reveal
    assert inter.enable_inter

    assert joint.enable_intra
    assert joint.enable_query
    assert joint.enable_reveal
    assert joint.enable_inter

    with pytest.raises(ValueError, match="timescale mode"):
        module.timescale_contract("almost_joint")


def test_gradient_blend_query_score_uses_only_sealed_label_free_values() -> None:
    module = load_module()
    features = {"expected_gradient": 0.8, "decision_changeability": 0.5}
    assert module.sealed_query_score(
        "current", features=features, base_score=0.4,
    ) == pytest.approx(0.4)
    assert module.sealed_query_score(
        "gradient_blend", features=features, base_score=0.4,
    ) == pytest.approx(0.65 * 0.8 + 0.20 * 0.5 + 0.15 * 0.4)
    with pytest.raises(ValueError, match="query score mode"):
        module.sealed_query_score("oracle", features=features, base_score=0.4)


def test_reordered_roster_requires_explicit_diagnostic_flag(tmp_path: Path) -> None:
    module = load_module()
    base = [10, 20, 30, 40]
    manifest = tmp_path / "order.json"
    manifest.write_text(json.dumps([20, 10, 40, 30]))

    with pytest.raises(ValueError, match="diagnostic reorder"):
        module.resolve_online_roster(
            base, order_manifest=manifest, allow_diagnostic_reorder=False,
        )

    resolved = module.resolve_online_roster(
        base, order_manifest=manifest, allow_diagnostic_reorder=True,
    )
    assert resolved.roster == [20, 10, 40, 30]
    assert resolved.is_reordered
    assert resolved.claim_status == "ORDER_ORACLE_DIAGNOSTIC"


def test_order_manifest_must_be_an_exact_permutation(tmp_path: Path) -> None:
    module = load_module()
    base = [10, 20, 30, 40]
    manifest = tmp_path / "order.json"
    manifest.write_text(json.dumps([10, 20, 20, 40]))

    with pytest.raises(ValueError, match="exact permutation"):
        module.resolve_online_roster(
            base, order_manifest=manifest, allow_diagnostic_reorder=True,
        )


def test_query_manifest_reports_explicit_label_oracle_metadata(tmp_path: Path) -> None:
    module = load_module()
    manifest = tmp_path / "queries.json"
    manifest.write_text(json.dumps({
        "shots": [10, 20], "label_oracle_used": True,
    }))
    assert module.query_manifest_uses_label_oracle(manifest)
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"shots": [10, 20]}))
    assert not module.query_manifest_uses_label_oracle(plain)


def test_fixed_query_manifest_is_built_only_from_sealed_label_free_decisions() -> None:
    module = load_module()
    roster = [10, 20, 30, 40]
    rows = [
        {"stream_index": index, "shot": shot, "selected": shot in {20, 40},
         "query_score": 0.1 * index}
        for index, shot in enumerate(roster)
    ]

    manifest = module.fixed_query_manifest_from_decisions(rows, roster=roster)

    assert manifest["shots"] == [20, 40]
    assert manifest["positions_zero_based"] == [1, 3]
    assert manifest["label_oracle_used"] is False

    rows[1]["revealed_is_disrupt"] = True
    with pytest.raises(ValueError, match="outcome field"):
        module.fixed_query_manifest_from_decisions(rows, roster=roster)


def test_fixed_query_manifest_rejects_roster_or_index_drift() -> None:
    module = load_module()
    rows = [
        {"stream_index": 0, "shot": 10, "selected": False},
        {"stream_index": 2, "shot": 20, "selected": True},
    ]
    with pytest.raises(ValueError, match="stream order"):
        module.fixed_query_manifest_from_decisions(rows, roster=[10, 20])


def test_oracle_curriculum_fails_closed_without_label_oracle_permission() -> None:
    module = load_module()
    records = [
        {"shot": 1, "static_group": "TN", "phenotype": [0.0, 0.0]},
        {"shot": 2, "static_group": "FP", "phenotype": [1.0, 0.0]},
        {"shot": 3, "static_group": "TN", "phenotype": [0.9, 0.0]},
        {"shot": 4, "static_group": "FN", "phenotype": [0.0, 1.0]},
        {"shot": 5, "static_group": "TP", "phenotype": [0.0, 0.9]},
        {"shot": 6, "static_group": "TN", "phenotype": [0.2, 0.2]},
        {"shot": 7, "static_group": "FP", "phenotype": [0.8, 0.1]},
        {"shot": 8, "static_group": "TP", "phenotype": [0.1, 0.8]},
    ]

    with pytest.raises(PermissionError, match="label oracle"):
        module.build_oracle_error_curriculum(
            records, query_rate=0.25, allow_label_oracle=False,
        )

    result = module.build_oracle_error_curriculum(
        records, query_rate=0.25, allow_label_oracle=True,
    )
    assert sorted(result.roster) == list(range(1, 9))
    assert len(result.query_shots) <= 2
    assert set(result.query_shots).issubset({2, 4, 7})
    assert result.label_oracle_used


def test_fixed_query_schedule_obeys_causal_token_budget() -> None:
    module = load_module()
    roster = list(range(1, 21))
    selector = module.CausalFixedQuerySelector(
        roster=roster, query_shots=[10, 20], query_rate=0.10,
        capacity=2.0, block_size=20, quota=2,
    )
    selected = []
    for index, shot in enumerate(roster):
        receipt = selector.decide(score=0.0, stream_index=index, shot_id=shot)
        if receipt["selected"]:
            selected.append(shot)
    assert selected == [10, 20]
    assert selector.total_selected == 2

    impossible = module.CausalFixedQuerySelector(
        roster=roster, query_shots=[1], query_rate=0.10,
        capacity=2.0, block_size=20, quota=2,
    )
    with pytest.raises(RuntimeError, match="insufficient causal query tokens"):
        impossible.decide(score=1.0, stream_index=0, shot_id=1)


def test_fixed_query_schedule_can_replay_a_rolling_budget_burst() -> None:
    module = load_module()
    roster = list(range(1, 21))
    selector = module.CausalFixedQuerySelector(
        roster=roster, query_shots=[2, 4, 6], query_rate=0.50,
        capacity=4.0, block_size=20, quota=2,
        enforce_block_quota=False,
    )
    selected = []
    for index, shot in enumerate(roster):
        receipt = selector.decide(score=0.0, stream_index=index, shot_id=shot)
        if receipt["selected"]:
            selected.append(shot)
    assert selected == [2, 4, 6]
    assert selector.state_dict()["budget_semantics"] == "rolling_token_only"


def test_random_query_manifest_matches_active_block_budget_and_is_causal() -> None:
    module = load_module()
    roster = list(range(100, 140))
    reference_positions = [3, 6, 10, 15, 22, 27, 31, 38]
    reference = {
        "shots": [roster[index] for index in reference_positions],
        "positions_zero_based": reference_positions,
        "label_oracle_used": False,
    }

    first = module.build_causal_random_query_manifest(
        roster=roster,
        reference_manifest=reference,
        block_size=20,
        query_rate=0.25,
        token_capacity=4.0,
        seed=131,
        replicate=0,
    )
    second = module.build_causal_random_query_manifest(
        roster=roster,
        reference_manifest=reference,
        block_size=20,
        query_rate=0.25,
        token_capacity=4.0,
        seed=131,
        replicate=0,
    )

    assert first == second
    assert first["label_oracle_used"] is False
    assert first["kind"] == "causal_block_matched_random_queries"
    assert len(first["shots"]) == len(reference["shots"])
    assert [sum(start <= p < start + 20 for p in first["positions_zero_based"])
            for start in (0, 20)] == [4, 4]

    selector = module.CausalFixedQuerySelector(
        roster=roster,
        query_shots=first["shots"],
        query_rate=0.25,
        capacity=4.0,
        block_size=20,
        quota=4,
    )
    for index, shot in enumerate(roster):
        selector.decide(score=0.0, stream_index=index, shot_id=shot)
    assert selector.total_selected == len(reference["shots"])

    changed = module.build_causal_random_query_manifest(
        roster=roster,
        reference_manifest=reference,
        block_size=20,
        query_rate=0.25,
        token_capacity=4.0,
        seed=131,
        replicate=1,
    )
    assert changed["shots"] != first["shots"]


def test_transition_report_counts_rescues_harms_and_query_efficiency() -> None:
    module = load_module()
    static = {1: "FP", 2: "FN", 3: "TP", 4: "TN", 5: "TARDY"}
    online = {1: "TN", 2: "TP", 3: "FN", 4: "FP", 5: "TP"}

    report = module.rescue_transition_report(
        static, online, queried_shots=[1, 2], stream_order=[1, 2, 3, 4, 5],
    )

    assert report["repaired_fp_to_tn"] == 1
    assert report["rescued_fn_to_tp"] == 1
    assert report["rescued_tardy_to_tp"] == 1
    assert report["harmed_tp_to_error"] == 1
    assert report["harmed_tn_to_fp"] == 1
    assert report["gross_rescues"] == 3
    assert report["gross_harms"] == 2
    assert report["net_rescues"] == 1
    assert report["net_rescues_per_query"] == pytest.approx(0.5)


def test_operational_group_counts_any_alarm_at_least_30ms_early_as_tp() -> None:
    module = load_module()
    assert module.operational_group(
        {"alarmed": True, "alarm_time_s": 1.0},
        {"is_disrupt": True, "down_time_s": 5.0},
    ) == "TP"
    assert module.operational_group(
        {"alarmed": True, "alarm_time_s": 4.99},
        {"is_disrupt": True, "down_time_s": 5.0},
    ) == "TARDY"
    assert module.operational_group(
        {"alarmed": True, "alarm_time_s": 1.0},
        {"is_disrupt": False, "down_time_s": None},
    ) == "FP"
    # A 1 kHz, exactly-30-sample lead must not become tardy due to binary
    # floating-point subtraction (3.559 - 3.529 is just below 0.030 in float).
    assert module.operational_group(
        {"alarmed": True, "alarm_time_s": 3.529},
        {"is_disrupt": True, "down_time_s": 3.5589999999999997},
    ) == "TP"


def test_oracle_curriculum_is_deterministic() -> None:
    module = load_module()
    records = [
        {
            "shot": shot,
            "static_group": "FP" if shot in (3, 7) else "TN",
            "phenotype": np.array([shot / 10.0, (shot % 3) / 3.0]),
        }
        for shot in range(1, 11)
    ]
    first = module.build_oracle_error_curriculum(
        records, query_rate=0.20, allow_label_oracle=True,
    )
    second = module.build_oracle_error_curriculum(
        records, query_rate=0.20, allow_label_oracle=True,
    )
    assert first.roster == second.roster
    assert first.query_shots == second.query_shots


def test_chronological_oracle_queries_keep_order_and_obey_rolling_tokens() -> None:
    module = load_module()
    records = [
        {
            "shot": shot,
            "static_group": "FP" if shot in {1, 2, 10, 20} else "TN",
            "phenotype": [float(shot)],
        }
        for shot in range(1, 21)
    ]
    with pytest.raises(PermissionError, match="label oracle"):
        module.build_chronological_error_queries(
            records, query_rate=0.10, allow_label_oracle=False,
        )
    result = module.build_chronological_error_queries(
        records, query_rate=0.10, allow_label_oracle=True,
    )
    assert result.roster == list(range(1, 21))
    assert result.query_shots == [10, 20]
    assert result.query_positions == [9, 19]
    assert result.label_oracle_used


def test_fixed_query_reorder_preserves_query_set_and_affordable_positions() -> None:
    module = load_module()
    records = [
        {
            "shot": shot,
            "static_group": "FP" if shot in (10, 11, 12) else "TN",
            "phenotype": [float(shot), float(shot % 2)],
        }
        for shot in range(1, 21)
    ]
    result = module.build_fixed_query_reorder(
        records,
        query_shots=[10, 20],
        query_positions=[9, 19],
        strategy="random_nonquery",
        random_seed=131,
        allow_label_oracle=False,
    )

    assert sorted(result.roster) == list(range(1, 21))
    assert [result.roster[position] for position in result.query_positions] == [10, 20]
    assert result.query_shots == [10, 20]
    assert result.query_positions == [9, 19]
    assert not result.label_oracle_used


def test_oracle_fixed_query_reorder_uses_same_budget_but_frontloads_error_learning() -> None:
    module = load_module()
    records = [
        {"shot": 1, "static_group": "TN", "phenotype": [9.0, 9.0]},
        {"shot": 2, "static_group": "FP", "phenotype": [0.0, 0.0]},
        {"shot": 3, "static_group": "FP", "phenotype": [0.1, 0.0]},
        {"shot": 4, "static_group": "TN", "phenotype": [8.0, 8.0]},
        {"shot": 5, "static_group": "FN", "phenotype": [4.0, 4.0]},
        {"shot": 6, "static_group": "FN", "phenotype": [4.1, 4.0]},
        {"shot": 7, "static_group": "TN", "phenotype": [7.0, 7.0]},
        {"shot": 8, "static_group": "TN", "phenotype": [6.0, 6.0]},
    ]

    with pytest.raises(PermissionError, match="label oracle"):
        module.build_fixed_query_reorder(
            records,
            query_shots=[2, 5],
            query_positions=[1, 5],
            strategy="hindsight_error_neighbors",
            random_seed=131,
            allow_label_oracle=False,
        )

    result = module.build_fixed_query_reorder(
        records,
        query_shots=[2, 5],
        query_positions=[1, 5],
        strategy="hindsight_error_neighbors",
        random_seed=131,
        allow_label_oracle=True,
    )
    assert set(result.query_shots) == {2, 5}
    assert [result.roster[position] for position in result.query_positions] == [2, 5]
    assert result.roster[2] == 3  # nearest same-type FP follows the revealed FP
    assert result.roster[6] == 6  # nearest same-type FN follows the revealed FN
    assert result.label_oracle_used
