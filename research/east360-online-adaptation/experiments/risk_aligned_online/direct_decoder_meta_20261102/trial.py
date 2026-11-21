"""Source-only matched direct diagnostic decoders, no risk-logit bridge."""

import argparse
from pathlib import Path
import time
import zipfile

import torch
from torch.nn import functional as F

from .. import io23
from ..east_context_operator_20261031.run import configure
from ..group_protected_meta_20261102.objectives import paired_risk
from ..group_protected_meta_20261102.trial import (
    SOURCE,
    bound_previous,
    load_fixed_role,
    summarize,
)
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import forward, direct
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import (
    CONTEXT,
    collect,
    load_source,
    one,
    source_basis,
)
from .core import advance, clone_head, prediction

PLAN = io23.ROOT / "DIRECT_DECODER_META_PLAN_20261102.md"
ARMS = ("forecast_only", "mean_meta", "worst_meta")
MODES = ("real", "reverse", "hold")


def combine(arm, absolute, risk, anchor):
    if arm not in ARMS:
        raise ValueError("fixed direct-decoder arm required")
    value = 0.20 * absolute + 1e-4 * anchor
    if arm != "forecast_only":
        value = value + risk["mean" if arm == "mean_meta" else "worst"] / 0.1
    return value


def gates(summary):
    primary = summary["no_MNM_CIII"]
    candidate = primary["worst_meta_real"]
    result = {
        g + "_improved": candidate[g]["mean_delta_bce"] is not None
        and candidate[g]["mean_delta_bce"] < -1e-8
        for g in ("safe", "danger", "compatible_danger")
    }
    result["native_protected"] = all(
        summary["native"]["worst_meta_real"][g]["mean_delta_bce"] <= 1e-6
        for g in ("safe", "danger")
    )
    for control in (
        "forecast_only_real",
        "mean_meta_real",
        "worst_meta_reverse",
        "worst_meta_hold",
    ):
        result["better_than_" + control] = (
            candidate["balanced"] < primary[control]["balanced"] - 1e-8
        )
    result["passed"] = all(result.values())
    return result


def at(model, data, i, decoder, mode="real", *, outer=False):
    return advance(
        model,
        one(data["origin"], i),
        model.base.center.new_zeros(4, 64),
        data["truths"][mode][i : i + 1],
        data["mask"][i : i + 1],
        decoder,
        create_graph=outer,
    )


def anchor_loss(decoder, initial):
    return sum(
        (p - initial[k]).square().sum() for k, p in decoder.named_parameters()
    ) / sum(p.numel() for p in decoder.parameters())


def pair_terms(model, data, take, decoder, *, meta):
    old, later, absolutes, accepted = [], [], [], 0
    for i in take:
        state, receipt = at(model, data, i, decoder, outer=meta)
        with torch.no_grad():
            z0 = forward(
                model, one(data["later"], i), model.base.center.new_zeros(4, 64)
            )["logit"]
        z = forward(model, one(data["later"], i), state)["logit"]
        physical = prediction(
            model, one(data["origin"], i), model.base.center.new_zeros(4, 64), decoder
        )
        absolutes.append(
            model.base.forecast_loss(
                physical, data["truths"]["real"][i : i + 1], data["mask"][i : i + 1]
            )
        )
        old.append(z0)
        later.append(z)
        accepted += int(receipt["accepted"])
    after, before = torch.cat(later), torch.cat(old)
    label = after.new_tensor([data["records"][i]["label"] for i in take])
    paired = {
        mode: paired_risk(after, before, label, mode=mode) for mode in ("mean", "worst")
    }
    return (
        torch.stack(absolutes).mean(),
        {k: v["risk"] for k, v in paired.items()},
        paired["mean"]["delta"],
        accepted,
    )


def fit(model, cache, schedule):
    heads = {a: clone_head(model).requires_grad_(True) for a in ARMS}
    initial = {k: p.detach().clone() for k, p in heads[ARMS[0]].named_parameters()}
    if any(
        not torch.equal(initial[k], p)
        for h in heads.values()
        for k, p in h.named_parameters()
    ):
        raise AssertionError("direct-head initial parameters must match")
    optimizers = {
        a: torch.optim.Adam(h.parameters(), lr=1e-4) for a, h in heads.items()
    }
    history = []
    for row in schedule:
        item = dict(**row, arms={})
        for arm, decoder in heads.items():
            optimizers[arm].zero_grad(set_to_none=True)
            absolute, risk, delta, accepted = pair_terms(
                model,
                cache[row["task"]],
                row["indices"],
                decoder,
                meta=arm != "forecast_only",
            )
            anchor = anchor_loss(decoder, initial)
            objective = combine(arm, absolute, risk, anchor)
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite direct decoder objective")
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizers[arm].step()
            item["arms"][arm] = dict(
                objective=float(objective.detach()),
                absolute_diagnostic_loss=float(absolute.detach()),
                anchor=float(anchor.detach()),
                group_risk_changes=delta.detach().cpu().tolist(),
                gradient_norm=float(norm),
                accepted=accepted,
            )
        history.append(item)
        if row["step"] % 64 == 0:
            print(dict(stage="fit", **item), flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("qualified source received gradients")
    for head in heads.values():
        head.eval().requires_grad_(False)
    return heads, history


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
                loss0 = F.binary_cross_entropy_with_logits(before, label)
            for arm, head in heads.items():
                with torch.no_grad():
                    physical = prediction(
                        model,
                        one(data["origin"], i),
                        model.base.center.new_zeros(4, 64),
                        head,
                    )
                    absolute = model.base.forecast_loss(
                        physical,
                        data["truths"]["real"][i : i + 1],
                        data["mask"][i : i + 1],
                    )
                for mode in modes:
                    state, receipt = at(model, data, i, head, mode)
                    with torch.no_grad():
                        output = forward(model, one(data["later"], i), state)
                        loss1 = F.binary_cross_entropy_with_logits(
                            output["logit"], label
                        )
                    r = dict(
                        **record,
                        task=task,
                        arm=arm + "_" + mode,
                        before_logit=float(before),
                        after_logit=float(output["logit"]),
                        before_bce=float(loss0),
                        after_bce=float(loss1),
                        delta_bce=float(loss1 - loss0),
                        absolute_diagnostic_before=float(absolute),
                        query_rms=float(output["rms"].max()),
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


def preflight(model, example, signal, basis):
    head = clone_head(model).requires_grad_(True)
    zero = model.base.center.new_zeros(4, 64)
    original = {k: p.detach().clone() for k, p in head.named_parameters()}
    with torch.no_grad():
        actual = prediction(model, example["origin"], zero, head)
        native = forward(model, example["origin"], zero, forecast=True)["forecast"]
    if not torch.equal(actual, native):
        raise AssertionError("initial direct decoder differs from source forecast")

    def risk():
        state, receipt = at(model, example, 0, head, outer=True)
        return forward(model, example["later"], state)["logit"], state, receipt

    z, state, receipt = risk()
    gradients = torch.autograd.grad(z.sum(), tuple(head.parameters()))
    norm = sum(g.double().square().sum() for g in gradients).sqrt()
    if not torch.isfinite(norm) or norm == 0:
        raise AssertionError("no later-risk derivative into physical decoder")
    direction = {
        k: (g.double() / norm).to(g.dtype).detach()
        for (k, _), g in zip(head.named_parameters(), gradients)
    }
    trials = []
    for eps in (0.005, 0.0025):
        values, receipts = [], []
        for sign in (1, -1):
            with torch.no_grad():
                for k, p in head.named_parameters():
                    p.copy_(original[k] + sign * eps * direction[k])
            zz, _, rr = risk()
            values.append(float(zz.detach()))
            receipts.append(rr["accepted"])
        if any(a != receipt["accepted"] for a in receipts):
            raise AssertionError("finite difference crossed strict-acceptance branch")
        actual = (values[0] - values[1]) / (2 * eps)
        if abs(actual - float(norm)) > 2e-4 + 0.05 * float(norm):
            raise AssertionError((actual, float(norm)))
        trials.append(dict(epsilon=eps, analytic=float(norm), numeric=actual))
    with torch.no_grad():
        for k, p in head.named_parameters():
            p.copy_(original[k])
        hooked = direct(
            model, signal, [example["records"][0]["query"]], basis, state.detach()
        )["logit"]
    torch.testing.assert_close(hooked, z.detach(), atol=1e-6, rtol=1e-6)
    return dict(
        status="PASS",
        initial_forecast_equal=True,
        later_risk_only_gradient_norm=float(norm),
        finite_differences=trials,
        ffnhook_logit_error=float((hooked - z.detach()).abs().max()),
        external_decoder_parameters=sum(p.numel() for p in head.parameters()),
        risk_bridge=False,
    )


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    previous = bound_previous()
    model, parent = load_source()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_rows, train_signals = load_fixed_role("train", previous)
    val_rows, val_signals = load_fixed_role("val", previous)
    schedule = read(SOURCE / "schedule.json")
    context_heads, basis = read(CONTEXT / "context_heads.json"), source_basis(model)
    output.mkdir(parents=True)
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    protocol = dict(
        scope="SOURCE_ONLY_DIRECT_DIAGNOSTIC_DECODER_NOT_JTEXT_ONLINE",
        source_sha256=SOURCE_SHA,
        bound_code_sha256=bound,
        previous_root=str(SOURCE),
        previous_completed_sha256=sha(SOURCE / "completed.json"),
        source_labels="max(25ms,AdvancedTime), lead>10ms",
        source_slow_zero=True,
        source_threshold=parent["threshold"],
        source_parameters_updated=0,
        head_shape=[1088, 575],
        head_parameters=626175,
        decoder_initialization="independent exact copies of qualified source forecast_head",
        risk_bridge=False,
        arms=list(ARMS),
        modes=list(MODES),
        steps=512,
        batch_size=2,
        lr=1e-4,
        optimizer="Adam",
        gradient_clip=1.0,
        seed=131,
        fixed_last_iterate=True,
        absolute_maintenance_weight=0.20,
        risk_scale=0.1,
        initial_parameter_anchor_weight=1e-4,
        inner_target="centered diagnostic shape",
        outer_maintenance="absolute diagnostic asinh source coordinates",
        event_use="fixed offline task strata only; cached q not consumed by prediction/update",
        schedule_sha256=sha(SOURCE / "schedule.json"),
        no_new_sampling=True,
        train_shots_read=len(train_signals),
        val_shots_read=len(val_signals),
        east_reads=0,
        test_signals_read=0,
        local_episodes_not_continuous_stream=True,
        independent_confirmation=False,
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
    )
    write(output / "protocol.json", protocol)
    write(output / "train_records.json", train_rows)
    write(output / "val_records.json", val_rows)
    write(output / "schedule.json", schedule)
    if sha(output / "schedule.json") != protocol["schedule_sha256"]:
        raise AssertionError("fixed schedule changed")
    with zipfile.ZipFile(
        output / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sorted(bound):
            archive.write(p, str(Path(p).relative_to(io23.ROOT)))
    train = collect(model, train_rows, train_signals, basis, context_heads)
    tiny = {
        "records": [train_rows[0]],
        "origin": one(train["native"]["origin"], 0),
        "later": one(train["native"]["later"], 0),
        "truths": {m: y[:1] for m, y in train["native"]["truths"].items()},
        "mask": train["native"]["mask"][:1],
    }
    write(
        output / "preflight.json",
        preflight(model, tiny, train_signals[train_rows[0]["shot"]], basis),
    )
    heads, history = fit(model, train, schedule)
    write(output / "training.json", history)
    torch.save(
        dict(
            heads={
                a: {k: v.detach().cpu() for k, v in h.state_dict().items()}
                for a, h in heads.items()
            },
            source_sha256=SOURCE_SHA,
        ),
        output / "decoders.pt",
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
    if sha(DEFAULT_PARENT) != SOURCE_SHA or any(
        not torch.equal(v, model.state_dict()[k].detach().cpu())
        for k, v in frozen.items()
    ):
        raise AssertionError("qualified source changed")
    if any(sha(Path(p)) != h for p, h in bound.items()):
        raise AssertionError("frozen imported code changed")
    completed = dict(
        status="COMPLETED",
        seconds=time.monotonic() - started,
        source_unchanged=True,
        gate=gates(summary),
        east_reads=0,
        test_signals_read=0,
        full_east_stream_tested=False,
        validation_rows=len(rows),
        validation_episodes=len(val_rows),
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", completed)
    print(completed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
