from __future__ import annotations

import gzip
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts/build_evidence.py"
)
REPRO_SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts/reproduce_primary_chain.py"
)
TABLE_SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts/generate_latex_tables.py"
)
VALIDATOR_SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts/validate_paper_package.py"
)
EQUAL_COMPUTE_REPORT_SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts/build_equal_compute_selector_report.py"
)


def module():
    spec = importlib.util.spec_from_file_location("build_evidence", SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def reproduce_module():
    spec = importlib.util.spec_from_file_location("reproduce_primary_chain", REPRO_SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def table_module():
    spec = importlib.util.spec_from_file_location("generate_latex_tables", TABLE_SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def validator_module():
    spec = importlib.util.spec_from_file_location("validate_paper_package", VALIDATOR_SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def equal_compute_report_module():
    spec = importlib.util.spec_from_file_location(
        "build_equal_compute_selector_report", EQUAL_COMPUTE_REPORT_SCRIPT,
    )
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_alarm_metrics_separate_raw_and_timely_detection() -> None:
    evidence = module()
    truth = {
        1: {"is_disrupt": True, "down_time_s": 5.0, "duration_s": 5.0},
        2: {"is_disrupt": True, "down_time_s": 5.0, "duration_s": 5.0},
        3: {"is_disrupt": False, "down_time_s": None, "duration_s": 5.0},
        4: {"is_disrupt": False, "down_time_s": None, "duration_s": 5.0},
    }
    records = [
        {"shot": 1, "alarmed": True, "alarm_time_s": 4.9},
        {"shot": 2, "alarmed": True, "alarm_time_s": 4.99},
        {"shot": 3, "alarmed": True, "alarm_time_s": 2.0},
        {"shot": 4, "alarmed": False, "alarm_time_s": None},
    ]

    result = evidence.alarm_metrics(records, truth, minimum_lead_ms=30.0)

    assert result["TP"] == 2
    assert result["FP"] == 1
    assert result["tpr"] == pytest.approx(1.0)
    assert result["tpr_at_30ms"] == pytest.approx(0.5)
    assert result["fpr"] == pytest.approx(0.5)
    assert result["median_lead_ms"] == pytest.approx(55.0)
    assert result["operational_TP"] == 1
    assert result["operational_FN"] == 1
    assert result["operational_tpr"] == pytest.approx(0.5)
    assert result["operational_median_lead_ms"] == pytest.approx(100.0)


def test_source_metrics_are_recomputed_from_prediction_ledger() -> None:
    evidence = module()
    summary = {
        "n_shots": 4,
        "n_disruptions": 2,
        "n_non_disruptions": 2,
        "timely_tpr": 0.5,
        "fpr": 0.5,
        "auc": 1.0,
        # Deliberately wrong: the evidence builder must not trust this field.
        "median_lead_ms": 55.0,
        "minimum_lead_ms": 10.0,
        "threshold": 0.5,
        "threshold_origin": "validation",
        "early_over_3000_ms_count": 0,
    }
    records = [
        {"shot": 1, "is_disrupt": True, "alarm": True,
         "alarm_time_s": 4.9, "lead_ms": 100.0, "score": 0.9},
        # The J-TEXT contract is strict >10 ms, so exactly 10 ms is tardy.
        {"shot": 2, "is_disrupt": True, "alarm": True,
         "alarm_time_s": 4.99, "lead_ms": 10.0, "score": 0.8},
        {"shot": 3, "is_disrupt": False, "alarm": True,
         "alarm_time_s": 4.0, "lead_ms": None, "score": 0.2},
        {"shot": 4, "is_disrupt": False, "alarm": False,
         "alarm_time_s": None, "lead_ms": None, "score": 0.1},
    ]
    truth = {
        1: {"is_disrupt": True, "down_time_s": 5.0},
        2: {"is_disrupt": True, "down_time_s": 5.0},
        3: {"is_disrupt": False, "down_time_s": None},
        4: {"is_disrupt": False, "down_time_s": None},
    }

    result = evidence.source_operational_row(summary, records, truth)

    assert result["operational_TP"] == 1
    assert result["operational_FN"] == 1
    assert result["FP"] == 1
    assert result["TN"] == 1
    assert result["operational_tpr"] == pytest.approx(0.5)
    assert result["operational_median_lead_ms"] == pytest.approx(100.0)
    assert result["auc"] == pytest.approx(1.0)
    assert result["lead_recomputation"]["status"] == "PASS"


def test_source_metrics_fail_closed_without_alarm_endpoint() -> None:
    evidence = module()
    summary = {
        "minimum_lead_ms": 10.0, "threshold": 0.5,
        "threshold_origin": "validation",
    }
    records = [{
        "shot": 1, "is_disrupt": True, "alarm": True,
        "alarm_time_s": None, "lead_ms": 100.0, "score": 0.9,
    }]
    truth = {1: {"is_disrupt": True, "down_time_s": 5.0}}

    with pytest.raises(RuntimeError, match="independently recomputed"):
        evidence.source_operational_row(summary, records, truth)


def test_aggregate_evidence_contains_no_workstation_absolute_paths() -> None:
    path = (
        ROOT
        / "experiments/transfer_methods_2026/paper_grade_package/evidence/three_stage_results.json"
    )
    text = path.read_text()

    assert '"/home/' not in text
    assert '"/media/' not in text


def test_circular_moving_blocks_preserve_local_order() -> None:
    evidence = module()
    rng = np.random.default_rng(7)

    indices = evidence._circular_moving_block_indices(
        rng, n_observations=11, block_size=4,
    )

    assert indices.shape == (11,)
    for start in range(0, len(indices), 4):
        block = indices[start:start + 4]
        if len(block) > 1:
            assert np.all(block[1:] == (block[:-1] + 1) % 11)


def test_primary_table_uses_only_lead_qualified_operational_tpr() -> None:
    tables = table_module()
    evidence = tables.json.loads(tables.EVIDENCE.read_text())
    rendered = tables.primary_table(evidence)

    assert "Raw TPR" not in rendered
    assert "Operational TPR" in rendered
    assert "official_results" in evidence
    assert evidence["official_metric_contract"]["raw_alarm_detection"] == "diagnostic_only"


def test_equal_label_budget_table_uses_query_disjoint_comparisons() -> None:
    tables = table_module()
    report = tables.json.loads(tables.EQUAL_BUDGET.read_text())
    rendered = tables.equal_label_budget_table(report)

    assert "front-load" in rendered
    assert "causal online" in rendered
    assert "119/139" in rendered
    assert report["direct_selected_complement"][
        "all_seed_selected_j_worse_than_static"
    ]


def test_new_latex_tables_have_valid_row_terminators_and_no_control_chars() -> None:
    tables = table_module()
    causal = tables.json.loads(tables.CAUSAL_RANDOM.read_text())
    target = tables.json.loads(tables.EAST_TARGET_ONLY.read_text())
    equal_compute = tables.json.loads(tables.EQUAL_COMPUTE.read_text())

    for rendered in (
        tables.causal_random_table(causal),
        tables.target_only_table(target),
        tables.equal_compute_selector_table(equal_compute),
    ):
        assert "\r" not in rendered
        data_rows = [
            line for line in rendered.splitlines()
            if " & " in line and not line.startswith(("Seed ", "Arm "))
        ]
        assert data_rows
        assert all(line.endswith(r"\\") for line in data_rows)

    assert r"J-TEXT $\rightarrow$ EAST" in tables.target_only_table(target)
    equal_compute_rendered = tables.equal_compute_selector_table(equal_compute)
    assert "102 & 1632 & 1632 & 204/204" in equal_compute_rendered
    assert "-0.00185" in equal_compute_rendered


def test_compact_archives_and_portable_hdf5_manifest_are_hash_valid() -> None:
    validator = validator_module()
    package = ROOT / "experiments/transfer_methods_2026/paper_grade_package"

    for manifest in (
        package / "evidence/causal_random_query_online_v1/artifact_manifest.json",
        package / "evidence/east_target_only_v1/artifact_manifest.json",
        package / "evidence/equal_compute_selector_control_v1/artifact_manifest.json",
    ):
        passed, failures = validator.validate_compact_artifact_manifest(manifest)
        assert passed, failures

    passed, failures = validator.validate_hdf5_content_manifest(
        package / "evidence/required_hdf5_content_manifest.json"
    )
    assert passed, failures


def test_equal_compute_metric_recomputation_uses_alarm_times_and_truth() -> None:
    report = equal_compute_report_module()
    rows = [
        {"shot": 1, "alarmed": True, "alarm_time_s": 4.90},
        {"shot": 2, "alarmed": True, "alarm_time_s": 4.99},
        {"shot": 3, "alarmed": True, "alarm_time_s": 2.0},
        {"shot": 4, "alarmed": False, "alarm_time_s": None},
    ]
    # The production helper requires 360 unique rows; repeat the four semantic
    # cases with unique shot ids so this test exercises its fail-closed roster gate.
    expanded = []
    truth = {}
    for offset in range(90):
        for original in rows:
            shot = offset * 4 + original["shot"]
            expanded.append({**original, "shot": shot})
            disruptive = original["shot"] in (1, 2)
            truth[shot] = {
                "is_disrupt": disruptive,
                "down_time_s": 5.0 if disruptive else None,
            }

    metrics = report.recompute_alarm_metrics(expanded, truth)

    assert metrics["TP"] == 180
    assert metrics["FN"] == 0
    assert metrics["FP"] == 90
    assert metrics["TN"] == 90
    assert metrics["tpr_at_30ms"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [([0.01, 0.02, 0.03], True), ([0.01, 0.0, 0.03], False),
     ([-0.01, 0.02, 0.03], False)],
)
def test_equal_compute_roster_screen_is_derived_from_deltas(
    deltas: list[float], expected: bool,
) -> None:
    adjudication = equal_compute_report_module().roster_screen_adjudication(deltas)

    assert adjudication["passed"] is expected
    assert ("passes" in adjudication["adjudication"]) is expected


def test_q30_manifest_and_primary_causal_ledgers_are_fail_closed() -> None:
    validator = validator_module()
    package = ROOT / "experiments/transfer_methods_2026/paper_grade_package"

    passed, failures = validator.validate_q30_manifest(
        package / "evidence/fixed_query_low_budget30_seed131.json",
        package / "evidence/primary_online_artifacts/seed131/static_predictions.jsonl",
    )
    assert passed, failures

    for seed in (131, 133, 137):
        passed, failures = validator.validate_primary_causal_ledgers(
            package / f"evidence/primary_online_artifacts/seed{seed}"
        )
        assert passed, failures

    passed, failures = validator.validate_text_artifact_portability(package)
    assert passed, failures


def test_alarm_state_machine_audit_records_real_parameter_updates() -> None:
    evidence = module().read_json(
        ROOT / "experiments/transfer_methods_2026/paper_grade_package/evidence/three_stage_results.json"
    )
    audit = evidence["primary_lineage"]["alarm_state_machine"]

    assert audit["adaptive_parameters"] == [
        "high_threshold", "physics_weight", "trend_weight", "early_threshold"
    ]
    assert audit["fixed_causal_widths"] == {"fast": 5, "slow": 100}
    assert [audit["seeds"][str(seed)]["changed_update_count"] for seed in (131, 133, 137)] == [38, 30, 40]


def test_exact_paired_test_counts_rescues_and_harms() -> None:
    evidence = module()
    before = [False, False, True, True, False]
    after = [True, False, True, False, True]

    result = evidence.exact_paired_change(before, after)

    assert result["rescued"] == 2
    assert result["harmed"] == 1
    assert 0.0 <= result["two_sided_exact_p"] <= 1.0


def test_phase_arrays_use_same_inclusive_30ms_contract_as_main_metrics() -> None:
    evidence = module()
    truth = {
        1: {"is_disrupt": True, "down_time_s": 5.0, "duration_s": 5.0},
        2: {"is_disrupt": False, "down_time_s": None, "duration_s": 5.0},
    }
    # Binary floating point represents this subtraction slightly below 30 ms
    # on some platforms, so the registered evaluator uses a 1e-9 ms tolerance.
    records = {
        str(seed): [
            {"shot": 1, "alarmed": True, "alarm_time_s": 4.9700000000005},
            {"shot": 2, "alarmed": False, "alarm_time_s": None},
        ]
        for seed in evidence.SEEDS
    }

    arrays = evidence._phase_arrays(records, truth)

    assert np.all(arrays["timely"][:, 0])


def test_east_support_roster_relabels_to_hdf5_authoritative_15d_95n() -> None:
    evidence = module()
    manifest = evidence.read_json(
        ROOT / "config/splits/east_paper131_split_all_in.json"
    )
    legacy_disruptions = {
        int(value) for value in manifest["train"]["disruption_shots"]
    }
    legacy_safe = {
        int(value) for value in manifest["train"]["non_disruption_shots"]
    }
    roster = legacy_disruptions | legacy_safe

    truth = evidence.load_east_truth(roster)

    assert len(roster) == 110
    assert sum(item["is_disrupt"] for item in truth.values()) == 15
    assert sum(not item["is_disrupt"] for item in truth.values()) == 95
    assert (len(legacy_disruptions), len(legacy_safe)) == (10, 100)


def test_compressed_ledger_is_read_and_hashed_as_original_bytes(tmp_path: Path) -> None:
    evidence = module()
    original = b'{"shot": 1, "alarmed": false}\n{"shot": 2, "alarmed": true}\n'
    compressed = tmp_path / "updates.jsonl.gz"
    with gzip.open(compressed, "wb") as handle:
        handle.write(original)

    logical_path = tmp_path / "updates.jsonl"
    assert evidence.read_jsonl(logical_path) == [
        {"shot": 1, "alarmed": False},
        {"shot": 2, "alarmed": True},
    ]
    assert evidence.sha256_artifact(logical_path) == hashlib.sha256(original).hexdigest()


def test_primary_chain_plan_freezes_v3c_causal_flags(tmp_path: Path) -> None:
    reproduce = reproduce_module()
    plan = reproduce.command_plan(seed=131, output_root=tmp_path, retrain_source=False)
    online = plan["online"]

    for flag in (
        "--development-search", "--strict-window-active",
        "--typed-endpoint-memory", "--episode-value-v3",
        "--episode-protected-sequence", "--episode-isolate-learning-feedback",
    ):
        assert flag in online
    assert online[online.index("--episode-query-mode") + 1] == "shadow"
    assert online[online.index("--episode-gradient-mode") + 1] == "legacy"
    assert online[online.index("--query-quota") + 1] == "6"
    assert online[online.index("--query-token-capacity") + 1] == "12.0"
    source_checkpoint = Path(online[online.index("--source-checkpoint") + 1])
    assert source_checkpoint == (
        reproduce.PACKAGE / "evidence/source_lineage/seed131/source_checkpoint.pt"
    )
