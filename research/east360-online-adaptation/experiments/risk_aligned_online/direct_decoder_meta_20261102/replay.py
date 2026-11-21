"""Actual neural replay of fixed four VAL episodes, not a meta-training rerun."""

import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_vector_20261030.source_run import source_gate
from ..east_context_operator_20261031.run import configure
from ..group_protected_meta_20261102.trial import bound_previous
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct, encode, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.data import SOURCE_SHA, TASKS, mask_task
from ..source_event_tasks_20261102.replay import select
from ..source_event_tasks_20261102.trial import (
    CONTEXT,
    collect,
    load_source,
    one,
    source_basis,
)
from .core import clone_head
from .trial import at


@torch.no_grad()
def matching_batch_hook(model, prefix, query, basis, temporary, cached_logit):
    """Separate batch-size arithmetic from the actual FFN insertion identity.

    The original VAL cache was encoded in batches of 32; this replay cache has
    four windows. Comparing either cache directly to a full batch-one pass mixes
    encoder GEMM rounding with the insertion test. Re-encode batch-one for BOTH
    insertion paths and require exact equality; record, do not hide, the separate
    multi-batch difference. Saved-record tolerances are unchanged.
    """
    same_pack = encode(model, prefix, [query], basis)
    same = forward(model, same_pack, temporary)["logit"]
    actual = direct(model, prefix, [query], basis, temporary)["logit"]
    torch.testing.assert_close(actual, same, atol=0, rtol=0)
    return dict(
        same_batch_ffn_error=float((actual - same).abs().max()),
        encoding_batch_difference=float((same - cached_logit).abs().max()),
    )


def main(root):
    root = Path(root)
    started = time.monotonic()
    done, p = read(root / "completed.json"), read(root / "protocol.json")
    if done["status"] != "COMPLETED" or not done["source_unchanged"]:
        raise ValueError("completed unchanged source required")
    for name, digest in done["artifact_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("artifact changed")
    for path, digest in p["bound_code_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("frozen code changed")
    previous = bound_previous()
    all_rows = read(root / "val_records.json")
    indices = select(all_rows)
    selected = [all_rows[i] for i in indices]
    configure()
    model, _ = load_source()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    signals = {}
    with source_gate(io23.source_roles()["val"]):
        for sid in sorted({r["shot"] for r in selected}):
            if (
                sha(io23.path_for("J-TEXT", sid))
                != previous["val_provenance"]["input_hashes"][str(sid)]
            ):
                raise ValueError("raw VAL input changed")
            raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
            signals[sid] = {k: raw[k] for k in ("values", "available")}
    basis = source_basis(model)
    cache = collect(
        model, selected, signals, basis, read(CONTEXT / "context_heads.json")
    )
    state = torch.load(root / "decoders.pt", map_location="cpu", weights_only=True)
    if state["source_sha256"] != SOURCE_SHA:
        raise ValueError("decoder source mismatch")
    heads = {}
    for arm, parameters in state["heads"].items():
        head = clone_head(model)
        head.load_state_dict(parameters, strict=True)
        heads[arm] = head.eval().requires_grad_(False)
    old = {
        (r["task"], r["shot"], r["origin"], r["arm"]): r
        for r in read(root / "validation_records.json")
    }
    checks = []
    max_error = 0.0
    for task in TASKS:
        for i, r in enumerate(selected):
            view = mask_task(signals[r["shot"]], task)
            prefix = {k: v[: r["query"] + 1] for k, v in view.items()}
            with torch.no_grad():
                before = forward(
                    model,
                    one(cache[task]["later"], i),
                    model.base.center.new_zeros(4, 64),
                )["logit"]
            for arm, head in heads.items():
                for mode in ("real", "reverse", "hold"):
                    ref = old[task, r["shot"], r["origin"], arm + "_" + mode]
                    temporary, receipt = at(model, cache[task], i, head, mode)
                    with torch.no_grad():
                        z = forward(model, one(cache[task]["later"], i), temporary)[
                            "logit"
                        ]
                        loss = F.binary_cross_entropy_with_logits(
                            z, z.new_full(z.shape, r["label"])
                        )
                    for name, value in dict(
                        before_logit=float(before),
                        after_logit=float(z),
                        after_bce=float(loss),
                        gradient_norm=receipt["gradient_norm"],
                        step_norm=receipt["step_norm"],
                    ).items():
                        if abs(value - ref[name]) > 2e-6 + 2e-5 * abs(ref[name]):
                            raise AssertionError(
                                (task, r["shot"], arm, mode, name, value, ref[name])
                            )
                    insertion = matching_batch_hook(
                        model, prefix, r["query"], basis, temporary, z
                    )
                    if receipt["accepted"] != ref["accepted"]:
                        raise AssertionError("strict acceptance changed")
                    max_error = max(max_error, abs(float(z) - ref["after_logit"]))
                    checks.append(
                        dict(
                            task=task,
                            shot=r["shot"],
                            origin=r["origin"],
                            arm=arm + "_" + mode,
                            **insertion,
                        )
                    )
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("qualified source changed")
    result = dict(
        status="PASS",
        selected_val_indices=indices,
        shots=len(signals),
        episodes=len(selected),
        neural_updates_recomputed=len(checks),
        actual_ffn_hook_checks=len(checks),
        max_record_logit_error=max_error,
        same_batch_ffn_identity="exact float32 equality",
        max_encoding_batch_difference=max(
            r["encoding_batch_difference"] for r in checks
        ),
        first_cross_batch_comparison="FAILED_1e-6_ATOL_PLUS_1e-6_RTOL; see NUMERICAL_REPLAY_BATCH_DIAGNOSIS_20261102.md",
        source_unchanged=True,
        new_outcome_reads=0,
        east_reads=0,
        training_repeated=False,
        completed_sha256=sha(root / "completed.json"),
        seconds=time.monotonic() - started,
        checks=checks,
    )
    write(root / "replay_neural_v2.json", result)
    print({k: v for k, v in result.items() if k != "checks"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
