"""Offline source-only teaching of an arrived-diagnostic update rule.

Never retrains the qualified classifier. The three mask views alter diagnostic
access, not physical values or labels. Validation is a local episode screen,
not a J-TEXT online branch and not evidence of EAST operational improvement.
"""

import argparse
import hashlib
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..context_memory_20261028.model import ContextAdapter
from ..context_vector_20261030.source_run import source_gate
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..event_conditional_feedback_20261102.context import probabilities
from ..event_conditional_feedback_20261102.core import prediction, update
from ..event_prefix_probe import causal_summary
from ..history_gradient_metric_20261102.run import read, sha, write
from ..innovation_metric_20261102.core import innovation_loss
from ..lora_stream_20261102.core import encode, forward, direct
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from .context_pair import imported_code_hashes
from .data import (
    TASKS,
    SOURCE_SHA,
    attach_event,
    load_events,
    load_records,
    mask_task,
    sample_indices,
    summarize_records,
)

PLAN = io23.ROOT / "SOURCE_EVENT_TASKS_PLAN_20261102.md"
CONTEXT = io23.ROOT / "probe_outputs/source_event_context_paired_v1"
ARMS = ("shared", "true", "shuffled")
EVALUATION_ARMS = tuple(a + "_" + m for a in ARMS for m in ("real", "reverse", "hold"))
STEPS = 512


def one(pack, i):
    return {k: value[i : i + 1] for k, value in pack.items()}


def make_schedule(records, steps=STEPS):
    rng = np.random.default_rng(131)
    return [
        dict(
            step=s + 1,
            task=list(TASKS)[s % 3],
            indices=sample_indices(records, rng, per_class=1),
        )
        for s in range(steps)
    ]


def task_targets(signal, origin, task):
    return targets(mask_task(signal, task), origin, origin + 25)


def summarize(rows):
    result = {}
    for group in ("safe", "danger", "early"):
        local = [r for r in rows if r["group"] == group]
        ids = sorted({r["shot"] for r in local})
        means = [
            np.mean([r["delta_bce"] for r in local if r["shot"] == sid]) for sid in ids
        ]
        result[group] = dict(
            n_shots=len(ids),
            n_points=len(local),
            mean_delta_bce=float(np.mean(means)) if means else None,
            helpful_points=sum(r["delta_bce"] < 0 for r in local),
            harmful_points=sum(r["delta_bce"] > 0 for r in local),
        )
    if not all(result[g]["n_shots"] for g in ("safe", "danger")):
        raise ValueError("both declared source validation groups required")
    result["balanced"] = (
        result["safe"]["mean_delta_bce"] + result["danger"]["mean_delta_bce"]
    ) / 2
    return result


def gates(summary):
    selected = summary["no_MNM_CIII"]["true_real"]
    result = {
        g + "_improved": selected[g]["mean_delta_bce"] < -1e-8
        for g in ("safe", "danger")
    }
    for name in ("shared_real", "shuffled_real", "true_reverse", "true_hold"):
        result["better_than_" + name] = (
            selected["balanced"] < summary["no_MNM_CIII"][name]["balanced"] - 1e-8
        )
    result["native_protected"] = all(
        summary["native"]["true_real"][g]["mean_delta_bce"] <= 1e-6
        for g in ("safe", "danger")
    )
    result["passed"] = all(result.values())
    return result


def load_source():
    base, parent = load_parent(DEFAULT_PARENT)
    model = (
        ContextAdapter(base.cuda(), readout="full", feedback="progressive")
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    if model.slow.weight.count_nonzero():
        raise ValueError("source teaching must not load the EAST cold-start state")
    return model, parent


def load_role(role):
    rows, ids, provenance = load_records(role)
    signals, times = {}, {}
    with source_gate(ids):
        for n, sid in enumerate(ids):
            if (
                sha(io23.path_for("J-TEXT", sid))
                != provenance["input_hashes"][str(sid)]
            ):
                raise ValueError("source raw hash changed: " + str(sid))
            raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
            signals[sid] = {k: raw[k] for k in ("values", "available")}
            times[sid] = raw["times"]
            if (n + 1) % 250 == 0:
                print(dict(stage="load_source", role=role, shots=n + 1), flush=True)
    events, _ = load_events()
    rows = attach_event(rows, times, events)
    provenance["raw_inputs_rehashed"] = True
    return rows, signals, provenance


@torch.no_grad()
def windows_pack(model, windows, masks, basis):
    """Batch independent past windows without changing or joining their time axes.

    The flattening is only a layout for the existing fixed-window builder:
    endpoint 128*i+127 has exactly window i as its 128-point past. No context
    from another shot can enter any returned window.
    """
    n = len(windows)
    if windows.shape != (n, 128, 23) or masks.shape != windows.shape:
        raise ValueError("independent source128x23 windows required")
    packed = dict(values=windows.reshape(-1, 23), available=masks.reshape(-1, 23))
    return encode(model, packed, np.arange(n) * 128 + 127, basis)


@torch.no_grad()
def collect(model, records, signals, basis, context_heads):
    device = model.base.center.device
    result = {}
    for task in TASKS:
        views = {sid: mask_task(s, task) for sid, s in signals.items()}
        packs = {"origin": [], "later": []}
        truths, masks, contexts = [], [], {a: [] for a in ("true", "shuffled")}
        for start in range(0, len(records), 32):
            local = records[start : start + 32]
            for key, clock in (("origin", "origin"), ("later", "query")):
                xs, aa = zip(
                    *(
                        simple_data.window(views[r["shot"]], r[clock], 128)
                        for r in local
                    )
                )
                xx, mm = np.stack(xs), np.stack(aa)
                packs[key].append(windows_pack(model, xx, mm, basis))
                if key == "origin":
                    # q is issued from the origin, never reselected by matured targets.
                    qmask = torch.as_tensor(mm, device=device).clone()
                    qmask[:, :, 2] = False
                    raw = (
                        causal_summary(
                            torch.as_tensor(xx, device=device),
                            qmask,
                            model.base.center,
                            model.base.scale,
                        )
                        .cpu()
                        .numpy()
                    )
                    for a, head in context_heads.items():
                        contexts[a].append(
                            torch.as_tensor(probabilities(raw, head), device=device)
                        )
            yy, ma = zip(
                *(targets(views[r["shot"]], r["origin"], r["maturity"]) for r in local)
            )
            truths.append(
                {
                    m: np.concatenate([y[m] for y in yy])
                    for m in ("real", "reverse", "hold")
                }
            )
            masks.append(np.concatenate(ma))
        result[task] = dict(
            records=records,
            origin={
                k: torch.cat([p[k] for p in packs["origin"]])
                for k in packs["origin"][0]
            },
            later={
                k: torch.cat([p[k] for p in packs["later"]]) for k in packs["later"][0]
            },
            truths={
                m: torch.as_tensor(
                    np.concatenate([y[m] for y in truths]), device=device
                )
                for m in ("real", "reverse", "hold")
            },
            mask=torch.as_tensor(np.concatenate(masks), device=device),
            context={a: torch.cat(v) for a, v in contexts.items()},
        )
        if task != "native" and result[task]["mask"][:, :, 2].any():
            raise ValueError("hidden source diagnostics leaked into inner targets")
        if task == "no_MNM_CIII" and result[task]["mask"][:, :, 1].any():
            raise ValueError("hidden CIII leaked into inner targets")
        print(dict(stage="encode", task=task, records=len(records)), flush=True)
    return result


def adapt(model, cache, i, table, arm, mode="real", *, outer=False):
    context = cache["context"]["true" if arm == "shared" else arm][i]
    state, receipt = update(
        model,
        one(cache["origin"], i),
        table.new_zeros(4, 64),
        cache["truths"][mode][i : i + 1],
        cache["mask"][i : i + 1],
        table,
        context,
        create_graph=outer,
    )
    return state, receipt, context


def objective(model, cache, i, table, arm):
    after, receipt, context = adapt(model, cache, i, table, arm, outer=True)
    zero = table.new_zeros(4, 64)
    with torch.no_grad():
        before = forward(model, one(cache["later"], i), zero)["logit"]
    later = forward(model, one(cache["later"], i), after)["logit"]
    label = later.new_full(later.shape, cache["records"][i]["label"])
    risk = (
        F.binary_cross_entropy_with_logits(later, label)
        - F.binary_cross_entropy_with_logits(before, label).detach()
    ) / 0.1
    signal = innovation_loss(
        model,
        prediction(model, one(cache["origin"], i), zero, table, context),
        cache["truths"]["real"][i : i + 1],
        cache["mask"][i : i + 1],
    )
    return risk + 0.20 * signal + 1e-4 * table.square().mean(), receipt


def fit(model, cache, schedule):
    heads = {
        a: torch.nn.Parameter(
            model.base.center.new_zeros((25, 23) if a == "shared" else (6, 25, 23))
        )
        for a in ARMS
    }
    optimizers = {a: torch.optim.Adam([p], lr=0.001) for a, p in heads.items()}
    history = []
    for row in schedule:
        log = dict(**row, objectives={}, gradient_norms={}, accepted={})
        for arm, table in heads.items():
            optimizers[arm].zero_grad(set_to_none=True)
            total, accepted = 0.0, 0
            for i in row["indices"]:
                value, receipt = objective(model, cache[row["task"]], i, table, arm)
                if not torch.isfinite(value):
                    raise FloatingPointError("nonfinite source-only meta objective")
                (value / 2).backward()
                total += float(value.detach()) / 2
                accepted += int(receipt["accepted"])
            norm = torch.nn.utils.clip_grad_norm_([table], 1.0, error_if_nonfinite=True)
            optimizers[arm].step()
            log["objectives"][arm], log["gradient_norms"][arm], log["accepted"][arm] = (
                total,
                float(norm),
                accepted,
            )
        history.append(log)
        if row["step"] % 64 == 0:
            print(dict(stage="fit", **log), flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("frozen source received a parameter gradient")
    return {a: p.detach() for a, p in heads.items()}, history


def evaluate(model, cache, heads):
    all_rows, summary = [], {}
    for task, data in cache.items():
        per_arm = {a: [] for a in EVALUATION_ARMS}
        for i, record in enumerate(data["records"]):
            with torch.no_grad():
                before = forward(
                    model, one(data["later"], i), model.base.center.new_zeros(4, 64)
                )["logit"]
                label = before.new_full(before.shape, record["label"])
                old_loss = F.binary_cross_entropy_with_logits(before, label)
            for arm, table in heads.items():
                for mode in ("real", "reverse", "hold"):
                    state, receipt, q = adapt(model, data, i, table, arm, mode)
                    with torch.no_grad():
                        output = forward(model, one(data["later"], i), state)
                        loss = F.binary_cross_entropy_with_logits(
                            output["logit"], label
                        )
                    r = dict(
                        task=task,
                        arm=arm + "_" + mode,
                        **record,
                        before_logit=float(before),
                        after_logit=float(output["logit"]),
                        before_bce=float(old_loss),
                        after_bce=float(loss),
                        delta_bce=float(loss - old_loss),
                        query_rms=float(output["rms"].max()),
                        context=q.cpu().tolist(),
                        accepted=receipt["accepted"],
                        loss_before=receipt.get("loss_before"),
                        loss_proposal=receipt.get("loss_proposal"),
                        gradient_norm=receipt["gradient_norm"],
                        step_norm=receipt["step_norm"],
                    )
                    per_arm[r["arm"]].append(r)
                    all_rows.append(r)
            if (i + 1) % 200 == 0:
                print(dict(stage="validation", task=task, records=i + 1), flush=True)
        summary[task] = {a: summarize(r) for a, r in per_arm.items()}
    return all_rows, summary


def preflight(model, signal, record, basis, context_heads):
    mini = collect(model, [record], {record["shot"]: signal}, basis, context_heads)
    result = {}
    for task, data in mini.items():
        view = mask_task(signal, task)
        x, a = simple_data.window(view, record["query"], 128)
        with torch.no_grad():
            native = model.base(
                torch.as_tensor(x[None], device="cuda"),
                torch.as_tensor(a[None], device="cuda"),
            )["logit"]
            cached = forward(model, data["later"], model.base.center.new_zeros(4, 64))[
                "logit"
            ]
        torch.testing.assert_close(native, cached, atol=1e-6, rtol=1e-6)
        table = model.base.center.new_full((6, 25, 23), 0.01, requires_grad=True)

        def risk(value):
            state, receipt, _ = adapt(model, data, 0, value, "true", outer=True)
            return forward(model, data["later"], state)["logit"], state, receipt

        z, state, receipt = risk(table)
        (gradient,) = torch.autograd.grad(z.sum(), table)
        if not torch.isfinite(gradient).all() or gradient.norm() == 0:
            raise AssertionError("no later-risk derivative into the auxiliary rule")
        direction = gradient.detach() / gradient.detach().norm()
        finite = []
        for eps in (0.005, 0.0025):
            plus, _, rp = risk(table.detach() + eps * direction)
            minus, _, rm = risk(table.detach() - eps * direction)
            if (
                rp["accepted"] != receipt["accepted"]
                or rm["accepted"] != receipt["accepted"]
            ):
                raise AssertionError("finite difference crossed acceptance branch")
            actual, expected = (
                float((plus.detach() - minus.detach()) / (2 * eps)),
                float(gradient.norm()),
            )
            if abs(actual - expected) > 2e-4 + 0.05 * abs(expected):
                raise AssertionError((actual, expected))
            finite.append(dict(epsilon=eps, analytic=expected, numeric=actual))
        with torch.no_grad():
            actual = direct(model, view, [record["query"]], basis, state.detach())[
                "logit"
            ]
        torch.testing.assert_close(actual, z.detach(), atol=1e-6, rtol=1e-6)
        result[task] = dict(
            native_zero_error=float((native - cached).abs().max()),
            gradient_norm=float(gradient.norm()),
            finite_differences=finite,
            direct_ffn_error=float((actual - z.detach()).abs().max()),
        )
    return dict(status="PASS", checks=result, source_only=True, east_reads=0)


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    completed_context = read(CONTEXT / "completed.json")
    if (
        completed_context["status"] != "COMPLETED"
        or not completed_context["paired_minibatches"]
    ):
        raise ValueError("paired source-only context fit required")
    for name, digest in completed_context["artifact_sha256"].items():
        if sha(CONTEXT / name) != digest:
            raise ValueError("context artifact changed")
    context_heads = read(CONTEXT / "context_heads.json")
    model, parent = load_source()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    train_rows, train_signals, train_provenance = load_role("train")
    val_rows, val_signals, val_provenance = load_role("val")
    schedule = make_schedule(train_rows)
    used = sorted({i for row in schedule for i in row["indices"]})
    index = {old: new for new, old in enumerate(used)}
    compact_schedule = [
        dict(r, indices=[index[i] for i in r["indices"]]) for r in schedule
    ]
    selected = [train_rows[i] for i in used]
    output.mkdir(parents=True)
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    protocol = dict(
        scope="SOURCE_ONLY_OFFLINE_EVENT_MASKED_TASK_TEACHING",
        source_sha256=SOURCE_SHA,
        qualified_source_parameters_updated=0,
        source_slow_zero=True,
        east_reads=0,
        test_signals_read=0,
        train_role="train",
        val_role="val",
        steps=STEPS,
        batch_size=2,
        seed=131,
        lr=0.001,
        optimizer="Adam",
        gradient_clip=1.0,
        tasks=TASKS,
        arms=list(ARMS),
        selection="fixed last iterate",
        source_threshold=parent["threshold"],
        source_labels="max(25ms,AdvancedTime), strict lead>10ms",
        event_use="offline task strata and frozen prefix context, never true current-input family",
        train_provenance=train_provenance,
        val_provenance=val_provenance,
        context_completed_sha256=sha(CONTEXT / "completed.json"),
        bound_code_sha256=bound,
        independent_confirmation=False,
        local_episode_not_continuous_stream=True,
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
    )
    write(output / "protocol.json", protocol)
    write(output / "census.json", summarize_records(train_rows + val_rows))
    write(output / "train_records_selected.json", selected)
    write(output / "val_records.json", val_rows)
    write(output / "schedule.json", compact_schedule)
    with zipfile.ZipFile(
        output / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(bound):
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
    write(
        output / "preflight.json",
        preflight(
            model, train_signals[selected[0]["shot"]], selected[0], basis, context_heads
        ),
    )
    train = collect(model, selected, train_signals, basis, context_heads)
    heads, history = fit(model, train, compact_schedule)
    torch.save(
        dict(
            tables={a: p.cpu() for a, p in heads.items()},
            source_sha256=SOURCE_SHA,
            context_completed_sha256=sha(CONTEXT / "completed.json"),
        ),
        output / "tables.pt",
    )
    write(output / "training.json", history)
    del train, train_signals
    val = collect(model, val_rows, val_signals, basis, context_heads)
    rows, summary = evaluate(model, val, heads)
    write(output / "validation_records.json", rows)
    write(output / "validation_summary.json", summary)
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("qualified source or frozen wrapper changed")
    if any(sha(Path(p)) != digest for p, digest in bound.items()):
        raise AssertionError("bound experiment code changed")
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    completed = dict(
        status="COMPLETED",
        seconds=time.monotonic() - started,
        gate=gates(summary),
        source_unchanged=True,
        source_parameter_updates=0,
        east_reads=0,
        test_signals_read=0,
        source_train_episodes_selected=len(selected),
        validation_episodes=len(val_rows),
        validation_records=len(rows),
        full_east_stream_tested=False,
        schedule_sha256=hashlib.sha256(
            (output / "schedule.json").read_bytes()
        ).hexdigest(),
        artifact_sha256=artifacts,
    )
    write(output / "completed.json", completed)
    print(completed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
