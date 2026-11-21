"""Isolate source outer-objective tradeoffs with fixed internal update machinery.

Source-only auxiliary teaching, NOT source-backbone training or J-TEXT online.
Two new heads use exactly the last trial's episode/task schedule. The max of
two sampled risk changes is not full group DRO or a no-harm guarantee.
"""

import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_vector_20261030.source_run import source_gate
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import (
    SOURCE_SHA,
    TASKS as TASKS,
    attach_event,
    load_events,
    load_records,
)
from ..source_event_tasks_20261102.trial import (
    CONTEXT,
    adapt,
    collect,
    load_source,
    one,
    preflight,
    source_basis,
    summarize as base_summarize,
)
from .objectives import paired_risk

SOURCE = io23.ROOT / "probe_outputs/source_event_masked_tasks_v1"
SOURCE_COMPLETE_SHA = "ecd4825f7d228ebf005ede5d42cdc757b143b780be33c09cfd5a9c6ce6f6cbcb"
PLAN = io23.ROOT / "GROUP_PROTECTED_META_PLAN_20261102.md"
ARMS = ("mean_risk", "worst_risk")
MODES = ("real", "reverse", "hold")


def summarize(rows):
    result = base_summarize(rows)
    local = [r for r in rows if r["group"] == "danger" and 30 < r["lead_ms"] <= 125]
    ids = sorted({r["shot"] for r in local})
    values = [
        np.mean([r["delta_bce"] for r in local if r["shot"] == sid]) for sid in ids
    ]
    result["compatible_danger"] = dict(
        n_shots=len(ids),
        n_points=len(local),
        mean_delta_bce=float(np.mean(values)) if values else None,
    )
    return result


def gates(summary, old):
    primary = summary["no_MNM_CIII"]
    candidate = primary["worst_risk_real"]
    result = {
        g + "_improved": candidate[g]["mean_delta_bce"] is not None
        and candidate[g]["mean_delta_bce"] < -1e-8
        for g in ("safe", "danger", "compatible_danger")
    }
    result["native_protected"] = all(
        summary["native"]["worst_risk_real"][g]["mean_delta_bce"] <= 1e-6
        for g in ("safe", "danger")
    )
    for control in ("mean_risk_real", "worst_risk_reverse", "worst_risk_hold"):
        result["better_than_" + control] = (
            candidate["balanced"] < primary[control]["balanced"] - 1e-8
        )
    result["better_than_old_mean_aux"] = (
        candidate["balanced"] < old["no_MNM_CIII"]["true_real"]["balanced"] - 1e-8
    )
    result["passed"] = all(result.values())
    return result


def bound_previous():
    if sha(SOURCE / "completed.json") != SOURCE_COMPLETE_SHA:
        raise ValueError("fixed previous source trial changed")
    completed = read(SOURCE / "completed.json")
    if completed["status"] != "COMPLETED" or not completed["source_unchanged"]:
        raise ValueError("completed unchanged source required")
    audit = read(SOURCE / "audit_cpu.json")
    if audit["status"] != "PASS" or audit["completed_sha256"] != SOURCE_COMPLETE_SHA:
        raise ValueError("previous saved-record audit is not bound")
    for name, digest in completed["artifact_sha256"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError("previous artifact changed: " + name)
    protocol = read(SOURCE / "protocol.json")
    if (
        protocol["source_sha256"] != SOURCE_SHA
        or sha(CONTEXT / "completed.json") != protocol["context_completed_sha256"]
    ):
        raise ValueError("source/context binding changed")
    return protocol


def load_fixed_role(role, protocol):
    name = "train_records_selected.json" if role == "train" else "val_records.json"
    rows = read(SOURCE / name)
    originals, ids, _ = load_records(role)
    indexed = {(r["shot"], r["origin"]): r for r in originals}
    for r in rows:
        old = indexed.get((r["shot"], r["origin"]))
        if old is None or any(r.get(k) != v for k, v in old.items()):
            raise ValueError("fixed source episode disagrees with canonical archive")
    if role == "val" and len(rows) != len(originals):
        raise ValueError("all fixed VAL episodes required")
    signals, times = {}, {}
    with source_gate(ids):
        for sid in sorted({r["shot"] for r in rows}):
            if (
                sha(io23.path_for("J-TEXT", sid))
                != protocol[role + "_provenance"]["input_hashes"][str(sid)]
            ):
                raise ValueError("bound source raw input changed")
            raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
            signals[sid] = {k: raw[k] for k in ("values", "available")}
            times[sid] = raw["times"]
    events, _ = load_events()
    if attach_event(rows, times, events) != rows:
        raise ValueError("absolute-clock event association changed")
    return rows, signals


def pair_forward(model, data, take, table):
    before, after, receipts = [], [], []
    for i in take:
        state, receipt, _ = adapt(model, data, i, table, "true", outer=True)
        with torch.no_grad():
            z0 = forward(model, one(data["later"], i), table.new_zeros(4, 64))["logit"]
        z = forward(model, one(data["later"], i), state)["logit"]
        before.append(z0)
        after.append(z)
        receipts.append(receipt)
    labels = table.new_tensor([data["records"][i]["label"] for i in take])
    return torch.cat(after), torch.cat(before), labels, receipts


def fit(model, cache, schedule):
    heads = {
        a: torch.nn.Parameter(model.base.center.new_zeros(6, 25, 23)) for a in ARMS
    }
    optimizers = {a: torch.optim.Adam([p], lr=0.001) for a, p in heads.items()}
    history = []
    for row in schedule:
        item = dict(**row, arms={})
        data = cache[row["task"]]
        for arm, table in heads.items():
            optimizers[arm].zero_grad(set_to_none=True)
            after, before, labels, receipts = pair_forward(
                model, data, row["indices"], table
            )
            paired = paired_risk(
                after, before, labels, mode="mean" if arm == "mean_risk" else "worst"
            )
            objective = paired["risk"] / 0.1 + 1e-4 * table.square().mean()
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite grouped meta objective")
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_([table], 1.0, error_if_nonfinite=True)
            optimizers[arm].step()
            item["arms"][arm] = dict(
                objective=float(objective.detach()),
                group_risk_changes=paired["delta"].detach().cpu().tolist(),
                gradient_norm=float(norm),
                accepted=sum(r["accepted"] for r in receipts),
            )
        history.append(item)
        if row["step"] % 64 == 0:
            print(dict(stage="fit", **item), flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("frozen source acquired gradients")
    return {a: p.detach() for a, p in heads.items()}, history


def evaluate(model, cache, heads, *, modes=MODES):
    all_rows, summary = [], {}
    for task, data in cache.items():
        buckets = {a + "_" + m: [] for a in heads for m in modes}
        for i, record in enumerate(data["records"]):
            with torch.no_grad():
                before = forward(
                    model, one(data["later"], i), model.base.center.new_zeros(4, 64)
                )["logit"]
                label = before.new_full(before.shape, record["label"])
                bce0 = F.binary_cross_entropy_with_logits(before, label)
            for arm, table in heads.items():
                for mode in modes:
                    state, receipt, q = adapt(model, data, i, table, "true", mode)
                    with torch.no_grad():
                        output = forward(model, one(data["later"], i), state)
                        loss = F.binary_cross_entropy_with_logits(
                            output["logit"], label
                        )
                    r = dict(
                        **record,
                        task=task,
                        arm=arm + "_" + mode,
                        before_logit=float(before),
                        after_logit=float(output["logit"]),
                        before_bce=float(bce0),
                        after_bce=float(loss),
                        delta_bce=float(loss - bce0),
                        query_rms=float(output["rms"].max()),
                        context=q.cpu().tolist(),
                        accepted=receipt["accepted"],
                        gradient_norm=receipt["gradient_norm"],
                        step_norm=receipt["step_norm"],
                        loss_before=receipt.get("loss_before"),
                        loss_proposal=receipt.get("loss_proposal"),
                    )
                    buckets[r["arm"]].append(r)
                    all_rows.append(r)
            if (i + 1) % 200 == 0:
                print(dict(stage="evaluate", task=task, rows=i + 1), flush=True)
        summary[task] = {a: summarize(v) for a, v in buckets.items()}
    return all_rows, summary


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    previous = bound_previous()
    model, parent = load_source()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis, context_heads = source_basis(model), read(CONTEXT / "context_heads.json")
    train_rows, train_signals = load_fixed_role("train", previous)
    val_rows, val_signals = load_fixed_role("val", previous)
    schedule = read(SOURCE / "schedule.json")
    if len(schedule) != 512 or any(
        [train_rows[i]["group"] for i in r["indices"]] != ["safe", "danger"]
        for r in schedule
    ):
        raise ValueError("fixed paired source schedule required")
    output.mkdir(parents=True)
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    protocol = dict(
        scope="SOURCE_ONLY_OBJECTIVE_DECOMPOSITION_NOT_EAST_RESULT",
        source_sha256=SOURCE_SHA,
        previous_completed_sha256=SOURCE_COMPLETE_SHA,
        previous_root=str(SOURCE),
        plan_sha256=sha(PLAN),
        bound_code_sha256=bound,
        source_slow_zero=True,
        source_threshold=parent["threshold"],
        source_labels="max(25ms,AdvancedTime), lead>10ms",
        compatible_danger="original danger and 30<lead<=125; no relabeling",
        table_shape=[6, 25, 23],
        zero_initialization=True,
        arms=list(ARMS),
        modes=list(MODES),
        steps=512,
        batch_size=2,
        lr=0.001,
        optimizer="Adam",
        clip=1.0,
        seed=131,
        new_auxiliary_weight=0.0,
        table_regularizer=1e-4,
        fixed_final_iterate=True,
        no_val_selection=True,
        no_new_sampling=True,
        context_completed_sha256=sha(CONTEXT / "completed.json"),
        original_schedule_sha256=sha(SOURCE / "schedule.json"),
        source_parameter_updates=0,
        test_signals_read=0,
        east_reads=0,
        source_train_shots_read=len(train_signals),
        source_val_shots_read=len(val_signals),
        local_episode_not_stream=True,
        torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
    )
    write(output / "protocol.json", protocol)
    write(output / "train_records.json", train_rows)
    write(output / "val_records.json", val_rows)
    write(output / "schedule.json", schedule)
    if sha(output / "schedule.json") != protocol["original_schedule_sha256"]:
        raise AssertionError("serialized paired schedule changed")
    with zipfile.ZipFile(
        output / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sorted(bound):
            archive.write(p, str(Path(p).relative_to(io23.ROOT)))
    write(
        output / "preflight.json",
        preflight(
            model,
            train_signals[train_rows[0]["shot"]],
            train_rows[0],
            basis,
            context_heads,
        ),
    )
    train = collect(model, train_rows, train_signals, basis, context_heads)
    heads, history = fit(model, train, schedule)
    write(output / "training.json", history)
    torch.save(
        dict(tables={a: t.cpu() for a, t in heads.items()}, source_sha256=SOURCE_SHA),
        output / "tables.pt",
    )
    train_records, train_summary = evaluate(
        model, {"no_MNM_CIII": train["no_MNM_CIII"]}, heads, modes=("real",)
    )
    write(output / "training_diagnostic_records.json", train_records)
    write(output / "training_diagnostic_summary.json", train_summary)
    del train, train_signals
    val = collect(model, val_rows, val_signals, basis, context_heads)
    rows, summary = evaluate(model, val, heads)
    write(output / "validation_records.json", rows)
    write(output / "validation_summary.json", summary)
    # Bind common baseline/q to the old run, without rerunning or refitting it.
    old = {
        (r["task"], r["shot"], r["origin"]): r
        for r in read(SOURCE / "validation_records.json")
        if r["arm"] == "true_real"
    }
    for r in rows:
        ref = old[r["task"], r["shot"], r["origin"]]
        if abs(r["before_logit"] - ref["before_logit"]) > 2e-6 or not np.allclose(
            r["context"], ref["context"], rtol=0, atol=1e-6
        ):
            raise AssertionError("comparison baseline or past-only context drift")
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("source or wrapper changed")
    if any(sha(Path(p)) != digest for p, digest in bound.items()):
        raise AssertionError("frozen imported code changed")
    completed = dict(
        status="COMPLETED",
        seconds=time.monotonic() - started,
        source_unchanged=True,
        new_east_performance_measured=False,
        east_reads=0,
        test_signals_read=0,
        gate=gates(summary, read(SOURCE / "validation_summary.json")),
        validation_episodes=len(val_rows),
        validation_rows=len(rows),
        train_episodes=len(train_rows),
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", completed)
    print(completed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
