from __future__ import annotations

import importlib
import gzip
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _runner():
    return importlib.import_module("run_equal_label_budget_ablation")


def test_active_query_ledger_is_label_authoritative_and_unique(tmp_path: Path) -> None:
    runner = _runner()
    ledger = tmp_path / "reveals.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"shot": 11, "revealed_is_disrupt": True, "revealed_down_time_s": 1.5},
                {"shot": 14, "revealed_is_disrupt": False, "revealed_down_time_s": None},
            )
        )
        + "\n"
    )
    outcomes = {
        11: {"is_disrupt": True, "down_time_s": 1.5},
        14: {"is_disrupt": False, "down_time_s": None},
    }

    selected = runner.load_active_query_ids(
        ledger, roster=[10, 11, 12, 13, 14], outcome_reader=outcomes.__getitem__,
    )

    assert selected == [11, 14]


def test_active_query_ledger_rejects_hdf5_label_mismatch(tmp_path: Path) -> None:
    runner = _runner()
    ledger = tmp_path / "reveals.jsonl"
    ledger.write_text(
        json.dumps({
            "shot": 11,
            "revealed_is_disrupt": True,
            "revealed_down_time_s": 1.5,
        })
        + "\n"
    )

    try:
        runner.load_active_query_ids(
            ledger,
            roster=[10, 11],
            outcome_reader=lambda _shot: {
                "is_disrupt": False, "down_time_s": None,
            },
        )
    except RuntimeError as exc:
        assert "HDF5-authoritative" in str(exc)
    else:
        raise AssertionError("label mismatch must fail closed")


def test_label_free_query_identity_loader_needs_no_outcomes(tmp_path: Path) -> None:
    runner = _runner()
    ledger = tmp_path / "query_decisions.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps({"shot": shot, "selected": selected})
            for shot, selected in ((10, False), (11, True), (12, False), (13, True))
        )
        + "\n"
    )

    assert runner.load_label_free_query_ids(
        ledger, roster=[10, 11, 12, 13],
    ) == [11, 13]


def test_block_matched_random_control_is_label_agnostic_and_exact() -> None:
    runner = _runner()
    roster = list(range(40))
    active = [0, 1, 8, 20, 22]

    random_ids = runner.block_matched_random_ids(
        roster=roster,
        excluded=[],
        reference_ids=active,
        block_size=20,
        seed=131,
        replicate=0,
    )

    assert len(random_ids) == 5
    assert sum(shot < 20 for shot in random_ids) == 3
    assert sum(shot >= 20 for shot in random_ids) == 2


def test_frontload_partition_never_scores_training_shots() -> None:
    runner = _runner()
    partition = runner.frontload_partition(
        roster=[1, 2, 3, 4, 5], extra_train_ids=[2, 5], support_ids=[100, 101],
    )

    assert partition["extra_train_ids"] == [2, 5]
    assert partition["evaluation_ids"] == [1, 3, 4]
    assert set(partition["extra_train_ids"]).isdisjoint(partition["evaluation_ids"])
    assert set(partition["support_ids"]).isdisjoint(partition["evaluation_ids"])


def test_online_update_budget_reads_plain_and_gzip_ledgers(tmp_path: Path) -> None:
    runner = _runner()
    rows = [
        {"event": "unified_revealed_finalize", "group": "FP",
         "shared": {"steps": 2, "updated": True},
         "private": {"steps": 2, "updated": True}},
        {"event": "unified_revealed_finalize", "group": "TN",
         "shared": {"steps": 0, "updated": False},
         "private": {"steps": 0, "updated": False}},
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    plain = tmp_path / "updates.jsonl"
    compressed = tmp_path / "updates.jsonl.gz"
    plain.write_text(payload)
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write(payload)

    expected = {
        "revealed_shots": 2,
        "model_update_shots": 1,
        "shared_optimizer_steps": 2,
        "private_optimizer_steps": 2,
        "shared_accepted_updates": 1,
        "private_accepted_updates": 1,
    }
    assert runner.read_online_update_budget(plain) == expected
    assert runner.read_online_update_budget(compressed) == expected


def test_common_evaluation_excludes_both_frontload_training_sets() -> None:
    runner = _runner()
    common = runner.common_evaluation_ids(
        roster=[1, 2, 3, 4, 5, 6],
        selected_train_ids=[2, 4],
        random_train_ids=[1, 6],
    )

    assert common == [3, 5]


def test_primary_objective_uses_lead_qualified_tpr() -> None:
    report = importlib.import_module("build_equal_label_budget_report")

    assert report.objective({"tpr_at_30ms": 0.85, "fpr": 0.12}) == 0.73


def test_historical_target_sampler_preserves_shot_first_weighting() -> None:
    runner = _runner()
    window = runner.grid.W
    shots = []
    for index in range(10):
        disruptive = index == 0
        shots.append({
            "shot": index,
            "v": np.full((window + 1, 2), index, dtype=np.float32),
            "start": 0.0,
            "rate": 1000.0,
            "is_disrupt": disruptive,
            "down": window / 1000.0 if disruptive else None,
        })

    sampler = runner.historical_target_sampler(
        shots, pos_ms=125.0, gray_ms=0.0, seed=131,
    )
    labels = np.asarray([sampler(1.0)[1] for _ in range(4000)])

    # The registered parent first drew one of ten shots uniformly.  Therefore
    # pp=1 does not turn the 1D/9N cohort into a class-balanced batch.
    assert 0.08 <= float(labels.mean()) <= 0.12


def test_random_replicates_are_averaged_within_seed_before_across_seeds() -> None:
    report = importlib.import_module("build_equal_label_budget_report")
    rows = [
        {"seed": 131, "tpr": 0.8, "tpr_at_30ms": 0.7, "fpr": 0.1,
         "median_lead_ms": 100.0, "tpr30_minus_fpr": 0.6},
        {"seed": 131, "tpr": 1.0, "tpr_at_30ms": 0.9, "fpr": 0.3,
         "median_lead_ms": 200.0, "tpr30_minus_fpr": 0.6},
        {"seed": 133, "tpr": 0.6, "tpr_at_30ms": 0.5, "fpr": 0.1,
         "median_lead_ms": 300.0, "tpr30_minus_fpr": 0.4},
    ]

    aggregate = report.mean_within_seed_then_across(rows)

    assert aggregate["tpr"] == 0.75
    assert aggregate["tpr30_minus_fpr"] == 0.5


def test_adjudication_preserves_label_access_and_compute_boundaries() -> None:
    report_module = importlib.import_module("build_equal_label_budget_report")
    summary_path = (
        ROOT
        / "experiments/transfer_methods_2026/paper_grade_package/evidence"
        / "equal_label_budget_frontload_v2/common_mask_report/summary.json"
    )
    summary = json.loads(summary_path.read_text())

    adjudication = report_module.build_adjudication(summary)

    assert adjudication["parent_fidelity_passed"]
    assert adjudication["primary_comparison"][
        "all_three_seeds_frontload_j_worse_than_static"
    ]
    assert adjudication["primary_comparison"][
        "all_three_seeds_online_j_better_than_frontload"
    ]
    assert any("FLOPs" in item for item in adjudication["limitations"])
