"""Small actual-neural replay; not a rerun of the source meta-training fit."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_vector_20261030.source_run import source_gate
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from .data import SOURCE_SHA, TASKS, mask_task
from .trial import CONTEXT, adapt, collect, load_source, one, source_basis


def select(records):
    """First archive row per group x known/unknown, without looking at scores."""
    chosen = []
    for group in ("safe", "danger"):
        for known in (False, True):
            candidates = [
                i
                for i, r in enumerate(records)
                if r["group"] == group and (r["origin_family"] >= 0) == known
            ]
            if not candidates:
                raise ValueError("replay requires every predeclared group")
            chosen.append(candidates[0])
    return chosen


def main(root):
    started = time.monotonic()
    root = Path(root)
    complete, protocol = read(root / "completed.json"), read(root / "protocol.json")
    if complete["status"] != "COMPLETED" or not complete["source_unchanged"]:
        raise ValueError("completed source-only run required")
    if any(
        sha(root / name) != digest
        for name, digest in complete["artifact_sha256"].items()
    ):
        raise ValueError("completed artifact changed")
    if any(
        sha(Path(path)) != digest
        for path, digest in protocol["bound_code_sha256"].items()
    ):
        raise ValueError("frozen experiment code changed")
    configure()
    model, _ = load_source()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis, all_records = source_basis(model), read(root / "val_records.json")
    chosen = select(all_records)
    records = [all_records[i] for i in chosen]
    allowed = set(io23.source_roles()["val"])
    if any(r["role"] != "val" or r["shot"] not in allowed for r in records):
        raise ValueError("VAL-only replay required")
    signals = {}
    with source_gate(allowed):
        for sid in sorted({r["shot"] for r in records}):
            if (
                sha(io23.path_for("J-TEXT", sid))
                != protocol["val_provenance"]["input_hashes"][str(sid)]
            ):
                raise ValueError("replay raw source changed")
            raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
            signals[sid] = {k: raw[k] for k in ("values", "available")}
    context_heads = read(CONTEXT / "context_heads.json")
    if sha(CONTEXT / "completed.json") != protocol["context_completed_sha256"]:
        raise ValueError("context identity changed")
    saved = torch.load(root / "tables.pt", map_location="cpu", weights_only=True)
    if (
        saved["source_sha256"] != SOURCE_SHA
        or saved["context_completed_sha256"] != protocol["context_completed_sha256"]
    ):
        raise ValueError("rule provenance changed")
    heads = {a: v.to(model.base.center.device) for a, v in saved["tables"].items()}
    cache = collect(model, records, signals, basis, context_heads)
    old_rows = {
        (r["task"], r["shot"], r["origin"], r["arm"]): r
        for r in read(root / "validation_records.json")
    }
    checks, maximum = [], 0.0
    for task in TASKS:
        for i, r in enumerate(records):
            view = mask_task(signals[r["shot"]], task)
            prefix = {k: v[: r["query"] + 1] for k, v in view.items()}
            with torch.no_grad():
                before = forward(
                    model,
                    one(cache[task]["later"], i),
                    model.base.center.new_zeros(4, 64),
                )["logit"]
            for arm, table in heads.items():
                for mode in ("real", "reverse", "hold"):
                    old = old_rows[task, r["shot"], r["origin"], arm + "_" + mode]
                    state, receipt, q = adapt(model, cache[task], i, table, arm, mode)
                    with torch.no_grad():
                        z = forward(model, one(cache[task]["later"], i), state)["logit"]
                        hooked = direct(model, prefix, [r["query"]], basis, state)[
                            "logit"
                        ]
                        loss = F.binary_cross_entropy_with_logits(
                            z, z.new_full(z.shape, r["label"])
                        )
                    values = dict(
                        before_logit=float(before),
                        after_logit=float(z),
                        after_bce=float(loss),
                        gradient_norm=receipt["gradient_norm"],
                        step_norm=receipt["step_norm"],
                    )
                    for name, value in values.items():
                        error = abs(value - old[name])
                        if error > 2e-6 + 2e-5 * abs(old[name]):
                            raise AssertionError(
                                (task, r["shot"], arm, mode, name, value, old[name])
                            )
                    torch.testing.assert_close(hooked, z, atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(
                        q.cpu().numpy(), old["context"], atol=2e-6, rtol=2e-5
                    )
                    if receipt["accepted"] != old["accepted"]:
                        raise AssertionError("replay acceptance changed")
                    maximum = max(maximum, abs(float(z) - old["after_logit"]))
                    checks.append(
                        dict(
                            task=task,
                            shot=r["shot"],
                            origin=r["origin"],
                            arm=arm + "_" + mode,
                            hooked_error=float((hooked - z).abs().max()),
                        )
                    )
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("source changed during replay")
    result = dict(
        status="PASS",
        source_only=True,
        seconds=time.monotonic() - started,
        selected_val_indices=chosen,
        unique_shots=len(signals),
        episodes=len(records),
        actual_neural_updates_recomputed=len(checks),
        actual_ffn_hook_checks=len(checks),
        max_record_logit_error=maximum,
        new_outcome_reads=0,
        east_reads=0,
        source_unchanged=True,
        meta_training_repeated=False,
        checks=checks,
    )
    write(root / "replay_neural.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
