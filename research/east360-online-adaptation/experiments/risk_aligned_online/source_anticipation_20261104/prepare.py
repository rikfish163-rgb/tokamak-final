"""Materialize an auditable offline teaching design; no signal arrays or GPU."""

import argparse
from collections import Counter
from pathlib import Path
import time
import zipfile

import numpy as np

from .. import io23
from ..history_gradient_metric_20261102.run import sha, write
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .data import build_records
from .design import make_targets, make_schedule

PLAN = io23.ROOT / "SOURCE_ANTICIPATION_PLAN_20261104.md"


def inventory(rows):
    pairs = Counter(
        f"{r['original_source_point_label']}_to_{r['future_disruption_125ms']}"
        for r in rows
    )
    strata = {}
    for group in sorted({r["anticipation_stratum"] for r in rows}):
        local = [r for r in rows if r["anticipation_stratum"] == group]
        strata[group] = dict(
            records=len(local),
            shots=len({r["shot"] for r in local}),
            hdf_disruptive_shots=len(
                {r["shot"] for r in local if r["hdf5_disruptive"]}
            ),
        )
    return dict(
        records=len(rows),
        shots=len({r["shot"] for r in rows}),
        strata=strata,
        label_pairs={
            f"{a}_to_{b}": pairs[f"{a}_to_{b}"] for a in (0, 1) for b in (0, 1)
        },
    )


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError("qualified source checkpoint changed")
    train, train_receipt = build_records("train")
    val, val_receipt = build_records("val")
    if {r["shot"] for r in train} & {r["shot"] for r in val}:
        raise AssertionError("locked source roles overlap")
    schedule = make_schedule(train)
    labels, permutation = make_targets(train, schedule=schedule)
    if not permutation["exposure_matched"]:
        raise AssertionError(
            "a corpus-only time shuffle is not a matched training control"
        )
    output.mkdir(parents=True)
    code = imported_code_hashes()
    protocol = dict(
        scope="SOURCE_ONLY_OFFLINE_TASK_DESIGN_NOT_TRAINING_OR_JTEXT_ONLINE",
        source_sha256=PARENT_SHA256,
        source_model_loaded=False,
        source_point_label="unchanged max(25ms,AdvancedTime), HDF outcome authority",
        auxiliary_label="separate future_disruption_125ms; never overwrite original label",
        arms=list(labels),
        planned_steps=len(schedule),
        seed=131,
        same_schedule_for_all_arms=True,
        exposure_matched=True,
        sampling="equal shot then row in each of four strata",
        control="within-disruptive-shot x equal exposure-count permutation; preserve corpus and actual training positive counts",
        hdf_receipts=dict(train=train_receipt, val=val_receipt),
        code_sha256=code,
        plan_sha256=sha(PLAN),
        signal_array_reads=0,
        test_raw_reads=0,
        east_raw_reads=0,
        model_training=False,
        raw_metadata="J-TEXT TRAIN/VAL IsDisrupt/DownTime and ip attrs/shape only",
        archived_original_fields="retained provenance; only hdf5_* and recomputed source/aux targets are authoritative",
    )
    write(output / "protocol.json", protocol)
    write(output / "train_records.json", train)
    write(output / "val_records.json", val)
    write(output / "schedule.json", schedule)
    write(output / "training_targets.json", {k: v.tolist() for k, v in labels.items()})
    summary = dict(
        train=inventory(train),
        val=inventory(val),
        permutation=permutation,
        original_point_targets_preserved=True,
        trained_model_count=0,
    )
    write(output / "inventory.json", summary)
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in code:
            archive.write(name, str(Path(name).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    # Exact per-shot control prevalence and unchanged archived truth fields.
    for sid in {r["shot"] for r in train}:
        use = np.array([r["shot"] == sid for r in train])
        if (
            labels["anticipation125"][use].sum()
            != labels["time_shuffled125"][use].sum()
        ):
            raise AssertionError("control prevalence changed")
    if (
        any(sha(Path(p)) != h for p, h in code.items())
        or sha(PLAN) != protocol["plan_sha256"]
    ):
        raise AssertionError("design code or plan changed during preparation")
    write(
        output / "completed.json",
        dict(
            status="COMPLETED",
            seconds=time.monotonic() - started,
            model_training=False,
            source_file_unchanged=sha(DEFAULT_PARENT) == PARENT_SHA256,
            artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    print(
        dict(
            status="COMPLETED",
            train=summary["train"],
            val=summary["val"],
            shuffled_changed_records=permutation["changed_records"],
            seconds=time.monotonic() - started,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
