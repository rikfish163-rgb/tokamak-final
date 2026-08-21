#!/usr/bin/env python3
"""Open J-TEXT TEST220 once for the validation-locked PADFIX checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
sys.path.insert(0, str(HERE))

import train_jtext_source_transformer as BASE  # noqa: E402


SEEDS = (131, 133, 137)
EXPECTED_REGULARIZATION = {
    "context_lengths": [32, 64, 96, 128],
    "context_view_weight": 0.5,
    "consistency_weight": 0.2,
    "representation_consistency_weight": 0.5,
    "pad_adversary_weight": 0.005,
    "pad_adversary_condition": "separate outcome-specific heads",
    "pad_adversary_strategy": (
        "alternating detached probe plus uniform-confusion encoder loss"
    ),
    "causal_view_contract": "same endpoint; delete older history only",
    "epoch_selection": "VAL160 preregistered source and context gates first",
}


def main() -> None:
    root = PACKAGE / "source_padfix_confirm_e32"
    results: dict[str, dict] = {}
    locked: list[dict] = []
    for seed in SEEDS:
        output = root / f"seed{seed}"
        checkpoint = output / "source_checkpoint.pt"
        summary = json.loads((output / "training_summary.json").read_text())
        if not summary.get("passes_all_validation_gates"):
            raise RuntimeError(f"seed {seed} did not pass every validation gate")
        if int(summary.get("test_shards_read", -1)) != 0:
            raise RuntimeError(f"seed {seed} read TEST220 before configuration lock")
        if summary.get("training_regularization") != EXPECTED_REGULARIZATION:
            raise RuntimeError(f"seed {seed} regularization differs from the lock")
        locked.append({
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "best_epoch": summary["best_epoch"],
            "validation_context_audit": summary["validation_context_audit"],
        })

    lock_path = root / "locked_before_test220.json"
    lock_path.write_text(json.dumps({
        "status": "LOCKED_BEFORE_TEST220",
        "seeds": list(SEEDS),
        "maximum_epochs": 32,
        "training_regularization": EXPECTED_REGULARIZATION,
        "checkpoints": locked,
        "test_shards_read_before_lock": 0,
    }, indent=2, sort_keys=True) + "\n")

    for seed in SEEDS:
        output = root / f"seed{seed}"
        results[str(seed)] = BASE.evaluate_test(
            output / "source_checkpoint.pt",
            output=output,
            seed=seed,
            device="cuda",
        )

    metric_names = (
        "auc", "raw_tpr", "timely_tpr", "fpr", "median_lead_ms",
    )
    aggregate = {}
    for name in metric_names:
        values = np.asarray([
            results[str(seed)]["metrics"][name] for seed in SEEDS
        ], dtype=np.float64)
        aggregate[name] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    report = {
        "status": "SOURCE_PADFIX_TEST220_EVALUATED_ONCE",
        "lock_manifest": str(lock_path),
        "seeds": results,
        "aggregate": aggregate,
        "selection_boundary": (
            "configuration and checkpoints were locked on TRAIN1353/VAL160 "
            "before TEST220 was opened"
        ),
    }
    (root / "test220_three_seed_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
