"""Conditional controller-only leave-one-shot-out geometric experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .core import evaluate, fit, group_summary

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "probe_outputs/difference_feedback_probe20_v1"
SOURCE_SHA = "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
PLAN = ROOT / "HISTORY_GRADIENT_METRIC_PLAN_20261102.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def leave_one_shot_out(rows):
    folds, held_out = [], []
    for sid in sorted({row["shot"] for row in rows}):
        training = [row for row in rows if row["shot"] != sid]
        fitted = fit(training)
        assert sid not in fitted["training_ids"]
        # No validation gradient is passed to the fitting function above.
        validation = [row for row in rows if row["shot"] == sid]
        evaluation = [evaluate(row, fitted["weights"]) for row in validation]
        folds.append(
            dict(
                held_shot=sid, fit=fitted, held_queries=[r["query"] for r in validation]
            )
        )
        held_out.extend(evaluation)
    return folds, held_out


def main(root):
    if root.exists():
        raise FileExistsError(root)
    start = time.monotonic()
    assert sha(SOURCE / "completed.json") == SOURCE_SHA
    source_done, source_protocol = (
        read(SOURCE / "completed.json"),
        read(SOURCE / "protocol.json"),
    )
    assert read(SOURCE / "verification_records.json")["status"] == "PASS"
    assert read(SOURCE / "verification_neural_v2.json")["status"] == "PASS"
    assert sha(SOURCE / "samples.json") == source_done["samples_sha256"]
    rows = read(SOURCE / "samples.json")
    ids = source_protocol["ids"]
    assert len(ids) == len(set(ids)) == 20
    assert {r["shot"] for r in rows} == set(ids) and len(rows) == 100
    assert all(len(r["task_gradient256"]) == 256 for r in rows)
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    root.mkdir(parents=True)
    protocol = dict(
        scope="CONDITIONAL_CONTROLLER_LOSO_GEOMETRY_NOT_NEURAL_UPDATE",
        source_probe=str(SOURCE),
        source_completed_sha256=SOURCE_SHA,
        source_protocol_sha256=sha(SOURCE / "protocol.json"),
        samples_sha256=sha(SOURCE / "samples.json"),
        source_model_sha256=source_protocol["source_sha256"],
        cold_state_sha256=source_protocol["cold_state_sha256"],
        ids=ids,
        n_samples=len(rows),
        dim=256,
        regularizer=1.0,
        max_coordinate=4.0,
        coordinate_mean=1.0,
        bisection_steps=80,
        numpy_version=np.__version__,
        controller_holdout_only=True,
        cold_parent_previously_saw_ended20=True,
        labels="unchanged_HDF5_EAST125ms",
        plan_sha256=sha(PLAN),
        raw_signal_reads=0,
        raw_outcome_reads=0,
        suffix_raw_reads=0,
        code_sha256={str(f.relative_to(ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(ROOT)))
    folds, held = leave_one_shot_out(rows)
    assert len(folds) == 20 and len(held) == 100
    write(root / "folds.json", folds)
    write(root / "held_out.json", held)
    write(root / "full_fit.json", fit(rows))
    summary = group_summary(held)
    result = dict(
        status="COMPLETED_GEOMETRIC_SCREEN",
        summary=summary,
        folds=20,
        held_rows=100,
        raw_signal_reads=0,
        raw_outcome_reads=0,
        folds_sha256=sha(root / "folds.json"),
        held_out_sha256=sha(root / "held_out.json"),
        full_fit_sha256=sha(root / "full_fit.json"),
        full_fit_used_for_validation=False,
        ready_for_neural_probe=summary["gates"]["passed"],
        method_promoted=False,
        online_gain_measured=False,
        independent_future_confirmation=False,
        scientific_goal_achieved=False,
        seconds=time.monotonic() - start,
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
