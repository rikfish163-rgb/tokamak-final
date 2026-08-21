"""Contract tests for the EAST-only target baseline.

These tests intentionally exercise only CPU-side contract helpers.  They must
not import or execute a training entry point that allocates CUDA state.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "experiments/transfer_methods_2026/run_east_target_only_baseline.py"
CONFIG_PATH = ROOT / (
    "experiments/transfer_methods_2026/paper_grade_package/configs/"
    "east_target_only_v1.json"
)
SUITE_PATH = ROOT / (
    "experiments/transfer_methods_2026/paper_grade_package/scripts/"
    "run_east_target_only_suite.py"
)
REPORT_PATH = ROOT / (
    "experiments/transfer_methods_2026/paper_grade_package/scripts/"
    "build_east_target_only_report.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location("east_target_only_baseline", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shot(shot_id: int, *, disruptive: bool, down: float | None = None) -> dict:
    n = 220
    return {
        "shot": int(shot_id),
        "v": np.zeros((n, 17), dtype=np.float32),
        "a": np.ones((n, 17), dtype=bool),
        "is_disrupt": bool(disruptive),
        "down": down,
        "start": 0.0,
        "rate": 1000.0,
    }


def test_contract_constants_are_frozen_and_native23_is_future_only() -> None:
    subject = _load_subject()

    assert subject.ARM_NAMES == ("target_only_local17", "target_only_matched17")
    assert subject.FEATURE_NAMES == (
        "Bt", "ip_A", "Bt_over_Ip", "Ip_diff", "v_loop", "Z_proxy",
        "sxr_kurt", "sxr_skew", "sxr_var", "xuv_kurt", "xuv_skew", "xuv_var",
        "Mir_avg_fre", "n=1 amplitude", "mode_number_n", "dx_a", "dy_a",
    )
    assert subject.WINDOW == 128
    assert subject.POSITIVE_TTD_MS == (0.0, 125.0)
    assert subject.TARGET_STEPS == 384
    assert subject.BATCH_SIZE == 128
    assert subject.SUPPORT_TARGET_TPR == 0.80
    assert subject.MINIMUM_LEAD_MS == 30.0
    assert subject.NATIVE23_STATUS == "future_not_implemented"


def test_historical_sampler_is_shot_first_then_within_shot() -> None:
    subject = _load_subject()
    shots = [_shot(1, disruptive=True, down=0.300)]
    shots.extend(_shot(index, disruptive=False) for index in range(2, 11))
    sampler = subject.historical_target_sampler(shots, seed=17)

    labels = [float(sampler(0.5)[1]) for _ in range(6000)]
    observed = float(np.mean(labels))
    # One of ten shots has both positive and negative windows; the other nine
    # can only contribute negatives.  Class-first sampling would be near 0.5.
    assert 0.035 < observed < 0.080


def test_ea125_window_contract_excludes_gray_and_out_of_range_endpoints() -> None:
    subject = _load_subject()
    shots = [_shot(1, disruptive=True, down=0.300)]
    windows = subject.historical_window_indices(shots, pos_ms=125.0, gray_ms=0.0)
    positive = windows[0]["positive"]
    negative = windows[0]["negative"]
    assert positive
    assert negative
    times = shots[0]["start"] + np.arange(len(shots[0]["v"])) / shots[0]["rate"]
    positive_ttd = (shots[0]["down"] - times[positive]) * 1000.0
    negative_ttd = (shots[0]["down"] - times[negative]) * 1000.0
    assert np.all((positive_ttd >= -1e-9) & (positive_ttd <= 125.0 + 1e-9))
    assert np.all(negative_ttd > 125.0)


def test_support_threshold_uses_only_support_and_target_tpr_point() -> None:
    subject = _load_subject()
    support_scores = np.asarray([0.90, 0.70, 0.20, 0.10], dtype=np.float64)
    support_labels = np.asarray([True, True, False, False])
    selected = subject.select_support_threshold(
        support_scores, support_labels, target_tpr=0.80,
    )
    assert selected["threshold"] == pytest.approx(0.70)
    assert selected["tpr"] == pytest.approx(1.0)
    assert selected["fpr"] == pytest.approx(0.0)
    with pytest.raises(TypeError):
        subject.select_support_threshold(  # type: ignore[call-arg]
            support_scores, support_labels, target_tpr=0.80,
            evaluation_scores=np.asarray([0.99]),
        )


def test_roster_contract_uses_hdf5_authority_and_disjoint_roles() -> None:
    subject = _load_subject()
    contract = subject.validate_roster_contract(
        support_ids=list(range(110)),
        balanced_ids=list(range(110, 470)),
        support_labels=[True] * 15 + [False] * 95,
        balanced_labels=[True] * 180 + [False] * 180,
    )
    assert contract["support_counts"] == {"disruptive": 15, "safe": 95}
    assert contract["balanced360_counts"] == {"disruptive": 180, "safe": 180}
    with pytest.raises(ValueError, match="overlap"):
        subject.validate_roster_contract(
            support_ids=[1, 2], balanced_ids=[2, 3],
            support_labels=[True, False], balanced_labels=[True, False],
        )


def test_prediction_seal_requires_complete_label_free_balanced_roster() -> None:
    subject = _load_subject()
    roster = [11, 12]
    good = [
        {"shot": 11, "alarmed": False, "alarm_index": None, "sealed": True},
        {"shot": 12, "alarmed": True, "alarm_index": 4, "sealed": True},
    ]
    subject.assert_prediction_seal(good, roster)
    with pytest.raises(ValueError, match="outcome"):
        subject.assert_prediction_seal(
            [{**good[0], "is_disrupt": True}, good[1]], roster,
        )
    with pytest.raises(ValueError, match="complete"):
        subject.assert_prediction_seal(good[:1], roster)


def test_lineage_hash_and_training_ledger_are_deterministic() -> None:
    subject = _load_subject()
    assert subject.sha256_ids([3, 1, 2]) == subject.sha256_ids([1, 2, 3])
    ledger = subject.training_ledger(
        arm="target_only_local17", seed=131,
        step_records=[{"step": 1, "batch_size": 128, "positive": 64}],
    )
    assert ledger["optimizer_steps"] == 1
    assert ledger["batch_size"] == 128
    assert ledger["arm"] == "target_only_local17"
    assert ledger["balanced360_labels_used"] is False
    assert ledger["threshold_source"] == "support110_hdf5_authoritative_only"
    assert subject.portable_path(ROOT / "relative.pt") == "relative.pt"
    assert subject.portable_path(Path("/tmp/external.pt")) == "<external-checkpoint>"


def test_config_records_the_frozen_contract_and_forbids_posthoc_selection() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["schema_version"] == "east-target-only-v1"
    assert config["arms"] == ["target_only_local17", "target_only_matched17"]
    assert config["target_training"]["sampler"] == (
        "historical-shot-first-then-within-shot-window-v1"
    )
    assert config["target_training"]["positive_ttd_ms"] == [0.0, 125.0]
    assert config["target_training"]["optimizer_steps"] == 384
    assert config["threshold"]["support_target_tpr"] == 0.8
    assert config["threshold"]["balanced360_selection"] is False
    assert config["evaluation"]["minimum_lead_ms"] == 30.0
    assert config["evaluation"]["seal_all_predictions_before_outcome"] is True
    assert config["native23"]["status"] == "future_not_implemented"


def test_suite_plan_is_two_arms_by_three_seeds_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("east_target_only_suite", SUITE_PATH)
    assert spec is not None and spec.loader is not None
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    output = tmp_path / "suite"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_east_target_only_suite.py",
            "--output-root", str(output),
            "--data-root", str(tmp_path / "external-data"),
            "--stage", "plan",
        ],
    )
    suite.main()
    plan = json.loads((output / "run_plan.json").read_text())
    assert len(plan["commands"]) == 6
    assert plan["data_root"] == "<external-data-root>"
    assert plan["data_root_serialized"] is False
    assert str(tmp_path / "external-data") not in (output / "run_plan.json").read_text()
    assert {
        command[command.index("--arm") + 1] for command in plan["commands"]
    } == {"target_only_local17", "target_only_matched17"}
    assert {
        int(command[command.index("--seed") + 1]) for command in plan["commands"]
    } == {131, 133, 137}


def test_target_only_report_uses_equal_seed_means_and_marks_development_stream(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("east_target_only_report", REPORT_PATH)
    assert spec is not None and spec.loader is not None
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)
    package = tmp_path / "package"
    evidence_dir = package / "evidence"
    evidence_dir.mkdir(parents=True)
    static = {
        str(seed): {
            "operational_tpr": 0.84,
            "fpr": 0.16,
            "operational_median_lead_ms": 240.0,
        }
        for seed in (131, 133, 137)
    }
    (evidence_dir / "three_stage_results.json").write_text(json.dumps({
        "official_results": {"east_static360": {"seeds": static}},
    }))
    source = tmp_path / "target-only"
    for arm, tpr, fpr in (
        ("target_only_local17", 0.70, 0.20),
        ("target_only_matched17", 0.75, 0.18),
    ):
        for seed in (131, 133, 137):
            run = source / arm / f"seed{seed}"
            run.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({
                "arm": arm,
                "seed": seed,
                "metrics": {
                    "operational_tpr": tpr,
                    "fpr": fpr,
                    "operational_median_lead_ms": 200.0,
                },
            }))
    report = report_module.build(source_root=source, package=package)
    assert report["claim_boundary"] == "TEST_DRIVEN_EXPLORATION"
    local = report["target_only_arms"]["target_only_local17"]["equal_seed_mean"]
    assert local["utility_tpr30_minus_fpr"] == pytest.approx(0.50)
    delta = report["target_only_arms"]["target_only_local17"][
        "transfer_static_minus_target_only"
    ]
    assert delta["utility_tpr30_minus_fpr"] == pytest.approx(0.18)
