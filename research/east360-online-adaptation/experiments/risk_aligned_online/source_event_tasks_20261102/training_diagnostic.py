"""Posthoc fixed-final-rule TRAIN check, not selection or another fit.

Checks whether the failed VAL danger direction was at least learned on TRAIN.
Uses all 435 already-selected TRAIN episodes, no new sampling or outcome read.
Only the predeclared primary missingness view and real feedback are evaluated.
"""

import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_vector_20261030.source_run import source_gate
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from .data import SOURCE_SHA
from .trial import CONTEXT, adapt, collect, load_source, one, source_basis, summarize


def main(root):
    root = Path(root)
    started = time.monotonic()
    completed, protocol = read(root / "completed.json"), read(root / "protocol.json")
    if completed["status"] != "COMPLETED" or completed["gate"]["passed"]:
        raise ValueError("this diagnostic is only for the failed fixed source trial")
    if any(
        sha(root / name) != digest
        for name, digest in completed["artifact_sha256"].items()
    ):
        raise ValueError("completed artifacts changed")
    if any(
        sha(Path(p)) != digest for p, digest in protocol["bound_code_sha256"].items()
    ):
        raise ValueError("bound experiment code changed")
    records = read(root / "train_records_selected.json")
    allowed = set(io23.source_roles()["train"])
    if any(
        r["shot"] not in allowed
        or r["role"] != "train"
        or r["group"] not in ("safe", "danger")
        for r in records
    ):
        raise ValueError("selected source TRAIN-only records required")
    configure()
    model, _ = load_source()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    signals = {}
    with source_gate(allowed):
        for sid in sorted({r["shot"] for r in records}):
            if (
                sha(io23.path_for("J-TEXT", sid))
                != protocol["train_provenance"]["input_hashes"][str(sid)]
            ):
                raise ValueError("source TRAIN input changed")
            raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
            signals[sid] = {k: raw[k] for k in ("values", "available")}
    if sha(CONTEXT / "completed.json") != protocol["context_completed_sha256"]:
        raise ValueError("context identity changed")
    tables = torch.load(root / "tables.pt", map_location="cpu", weights_only=True)
    if tables["source_sha256"] != SOURCE_SHA:
        raise ValueError("rule source changed")
    cache = collect(
        model,
        records,
        signals,
        source_basis(model),
        read(CONTEXT / "context_heads.json"),
    )["no_MNM_CIII"]
    rows, summary = [], {}
    for arm, value in tables["tables"].items():
        table, local = value.cuda(), []
        for i, r in enumerate(records):
            state, receipt, _ = adapt(model, cache, i, table, arm, "real")
            with torch.no_grad():
                before = forward(model, one(cache["later"], i), table.new_zeros(4, 64))[
                    "logit"
                ]
                after = forward(model, one(cache["later"], i), state)["logit"]
                label = before.new_full(before.shape, r["label"])
                loss0 = F.binary_cross_entropy_with_logits(before, label)
                loss1 = F.binary_cross_entropy_with_logits(after, label)
            local.append(
                dict(
                    **r,
                    arm=arm,
                    before_bce=float(loss0),
                    after_bce=float(loss1),
                    delta_bce=float(loss1 - loss0),
                    accepted=receipt["accepted"],
                )
            )
        summary[arm] = summarize(local)
        rows.extend(local)
        print(
            dict(stage="posthoc_train_diagnostic", arm=arm, summary=summary[arm]),
            flush=True,
        )
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("source or wrapper changed")
    output = dict(
        status="COMPLETED",
        posthoc=True,
        no_model_selection=True,
        optimizer_steps=0,
        task="no_MNM_CIII",
        feedback="real",
        rows=rows,
        summary=summary,
        completed_sha256=sha(root / "completed.json"),
        own_code_sha256=sha(Path(__file__)),
        source_unchanged=True,
        new_outcome_reads=0,
        east_reads=0,
        seconds=time.monotonic() - started,
    )
    write(root / "training_replay_diagnostic.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
