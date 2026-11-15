"""Ended20, four-fold actual neural test of source-event conditional feedback."""

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.objectives import margin_loss
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..innovation_metric_20261102.core import innovation_loss
from ..lora_stream_20261102.core import encode, forward, direct
from ..risk_bridge_forecaster_20261102.prepare import folds
from ..warmstart_alignment_20261022.alignment import transform
from .context import at_origin
from .core import prediction, update

PLAN = io23.ROOT / "EVENT_CONDITIONAL_FEEDBACK_PLAN_20261102.md"
SOURCE = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"
CONTEXT = io23.ROOT / "probe_outputs/event_context_fit_v1"
ARMS = ("shared", "true", "shuffled")
STEPS = 128


def summarize(rows):
    result = {}
    for group in ("disruptive", "safe", "danger_window"):
        selected = [
            r
            for r in rows
            if (
                not r["disruptive"]
                if group == "safe"
                else r["disruptive"]
                and (group == "disruptive" or 30 < r["actual_lead_ms"] <= 125)
            )
        ]
        ids = sorted({r["shot"] for r in selected})
        if not ids:
            raise ValueError("all predeclared shot groups must be covered")
        means = [
            float(np.mean([r["delta_bce"] for r in selected if r["shot"] == sid]))
            for sid in ids
        ]
        result[group] = dict(
            n_points=len(selected),
            n_shots=len(ids),
            mean_delta_bce=float(np.mean(means)),
            helpful_points=sum(r["delta_bce"] < 0 for r in selected),
            harmful_points=sum(r["delta_bce"] > 0 for r in selected),
        )
    result["balanced"] = (
        result["disruptive"]["mean_delta_bce"] + result["safe"]["mean_delta_bce"]
    ) / 2
    return result


def gates(summary):
    true = summary["true_real"]
    result = {
        g: true[g]["mean_delta_bce"] < -1e-8
        for g in ("disruptive", "safe", "danger_window")
    }
    for arm in ("shared_real", "shuffled_real", "true_reverse", "true_hold"):
        result["better_than_" + arm] = (
            true["balanced"] < summary[arm]["balanced"] - 1e-8
        )
    result["passed"] = all(result.values())
    return result


def objective(model, e, table, arm, threshold):
    zero = model.base.center.new_zeros(4, 64)
    context = e["context"]["true" if arm == "shared" else arm]
    state, receipt = update(
        model,
        e["origin_pack"],
        zero,
        e["truth"]["real"],
        e["mask"],
        table,
        context,
        create_graph=True,
    )
    later = forward(model, e["later_pack"], state)["logit"]
    with torch.no_grad():
        before = forward(model, e["later_pack"], zero)["logit"]
    label = int(e["disruptive"])
    risk = (
        margin_loss(later, threshold, label)
        - margin_loss(before, threshold, label).detach()
    ) / 0.1
    signal = innovation_loss(
        model,
        prediction(model, e["origin_pack"], zero, table, context),
        e["truth"]["real"],
        e["mask"],
    )
    return risk + 0.20 * signal + 1e-4 * table.square().mean(), receipt


def fit(model, examples, training_ids, threshold):
    if {e["shot"] for e in examples} != set(training_ids):
        raise ValueError("fit may only see the assigned historical training shots")
    heads = {
        a: torch.nn.Parameter(
            model.base.center.new_zeros((25, 23) if a == "shared" else (6, 25, 23))
        )
        for a in ARMS
    }
    optimizers = {a: torch.optim.Adam([b], lr=0.001) for a, b in heads.items()}
    pools = {
        flag: {
            sid: [e for e in examples if e["shot"] == sid]
            for sid in training_ids
            if any(e["shot"] == sid and e["disruptive"] == flag for e in examples)
        }
        for flag in (True, False)
    }
    rng, history = np.random.default_rng(131), []
    for step in range(STEPS):
        chosen = []
        for flag in (True, False):
            sid = list(pools[flag])[int(rng.integers(len(pools[flag])))]
            chosen.append(pools[flag][sid][int(rng.integers(len(pools[flag][sid])))])
        record = dict(
            step=step + 1,
            samples=[dict(shot=e["shot"], query=e["query"]) for e in chosen],
            losses={},
            gradient_norms={},
        )
        for arm, table in heads.items():
            optimizers[arm].zero_grad(set_to_none=True)
            total = 0.0
            for e in chosen:
                loss, _ = objective(model, e, table, arm, threshold)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "nonfinite event-conditional meta objective"
                    )
                (loss / 2).backward()
                total += float(loss.detach()) / 2
            norm = torch.nn.utils.clip_grad_norm_([table], 1.0, error_if_nonfinite=True)
            optimizers[arm].step()
            record["losses"][arm], record["gradient_norms"][arm] = total, float(norm)
        history.append(record)
    assert all(p.grad is None for p in model.parameters())
    return {a: b.detach() for a, b in heads.items()}, history


def evaluate(model, e, heads, threshold):
    zero = model.base.center.new_zeros(4, 64)
    label = int(e["disruptive"])
    with torch.no_grad():
        before = forward(model, e["later_pack"], zero)["logit"]
        old_loss = margin_loss(before, threshold, label)
    record = {
        k: e[k]
        for k in ("shot", "query", "disruptive", "actual_lead_ms", "point_label125")
    }
    record.update(
        before_logit=float(before),
        before_bce=float(old_loss),
        context={a: q.cpu().tolist() for a, q in e["context"].items()},
        results={},
    )
    for arm, table in heads.items():
        context = e["context"]["true" if arm == "shared" else arm]
        for mode in ("real", "reverse", "hold"):
            after, receipt = update(
                model,
                e["origin_pack"],
                zero,
                e["truth"][mode],
                e["mask"],
                table,
                context,
            )
            with torch.no_grad():
                out = forward(model, e["later_pack"], after)
                loss = margin_loss(out["logit"], threshold, label)
            record["results"][arm + "_" + mode] = dict(
                after_logit=float(out["logit"]),
                after_bce=float(loss),
                delta_bce=float(loss - old_loss),
                query_rms=float(out["rms"].max()),
                receipt=receipt,
            )
    return record


def build_example(
    model, signal, basis, context_heads, *, sid, query, flag, lead, point_label
):
    origin = query - 26
    prefix = {k: v[: origin + 1] for k, v in signal.items()}
    observed, mask = targets(signal, origin, query - 1)
    return dict(
        shot=sid,
        query=query,
        disruptive=flag,
        actual_lead_ms=lead,
        point_label125=point_label,
        origin_pack=encode(model, prefix, [origin], basis),
        later_pack=encode(
            model, {k: v[: query + 1] for k, v in signal.items()}, [query], basis
        ),
        truth={
            k: torch.as_tensor(v, device=model.base.center.device)
            for k, v in observed.items()
        },
        mask=torch.as_tensor(mask, device=model.base.center.device),
        context={
            a: at_origin(model, prefix, origin, head)
            for a, head in context_heads.items()
        },
    )


def preflight(model, example, signal, basis):
    def query_logit(value):
        state, receipt = update(
            model,
            example["origin_pack"],
            value.new_zeros(4, 64),
            example["truth"]["real"],
            example["mask"],
            value,
            example["context"]["true"],
            create_graph=True,
        )
        return forward(model, example["later_pack"], state)["logit"], state, receipt

    table = model.base.center.new_full((6, 25, 23), 0.01, requires_grad=True)
    logit, state, receipt = query_logit(table)
    # Test the risk path itself. A nonzero auxiliary reconstruction gradient
    # would not establish that the learned rule can change a later risk score.
    (gradient,) = torch.autograd.grad(logit, table)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    direction = gradient.detach() / gradient.detach().norm()
    trials = []
    for epsilon in (0.005, 0.0025):
        plus, _, rplus = query_logit(table.detach() + epsilon * direction)
        minus, _, rminus = query_logit(table.detach() - epsilon * direction)
        assert rplus["accepted"] == rminus["accepted"] == receipt["accepted"]
        analytic = float((gradient.detach() * direction).sum())
        numeric = float((plus.detach() - minus.detach()) / (2 * epsilon))
        assert abs(analytic - numeric) <= 2e-4 + 0.05 * abs(analytic), (
            analytic,
            numeric,
        )
        trials.append(
            dict(epsilon=epsilon, analytic=analytic, finite_difference=numeric)
        )
    with torch.no_grad():
        actual = direct(
            model,
            {k: v[: example["query"] + 1] for k, v in signal.items()},
            np.asarray([example["query"]]),
            basis,
            state.detach(),
        )["logit"]
    assert torch.equal(actual, logit.detach())
    return dict(
        status="PASS",
        shot=example["shot"],
        query=example["query"],
        accepted=receipt["accepted"],
        gradient_norm=float(gradient.norm()),
        gradient_scope="later_risk_logit_only_not_total_outer_loss",
        actual_layer_hook_exact=True,
        query_logit=float(logit.detach()),
        receipt=receipt,
        finite_difference=trials,
    )


def main(root, *, preflight_only=False):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    assert (
        sha(SOURCE / "completed.json")
        == "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
    )
    parent, source_done = (
        read(SOURCE / "protocol.json"),
        read(SOURCE / "completed.json"),
    )
    assert sha(SOURCE / "samples.json") == source_done["samples_sha256"]
    context_done = read(CONTEXT / "completed.json")
    assert (
        context_done["status"] == "COMPLETED"
        and context_done["source_sha256"] == parent["source_sha256"]
    )
    assert (
        sha(CONTEXT / "context_heads.json")
        == context_done["artifact_sha256"]["context_heads.json"]
    )
    for path, digest in context_done["bound_files_sha256"].items():
        assert sha(Path(path)) == digest, path
    configure()
    model, configuration, cold = load_model()
    assert sha(cold) == parent["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis, threshold = source_basis(model), float(configuration["threshold"])
    map_path = Path(parent["calibration_reference"]) / "input_map.json"
    assert sha(map_path) == parent["input_map_sha256"]
    mapping, context_heads = read(map_path), read(CONTEXT / "context_heads.json")
    rows = read(SOURCE / "samples.json")
    fold_plan = folds(rows, parent["ids"])
    root.mkdir(parents=True)
    files = {PLAN, Path(__file__)}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    bound = {
        str(p): sha(p)
        for p in (
            SOURCE / "completed.json",
            SOURCE / "samples.json",
            SOURCE / "protocol.json",
            CONTEXT / "completed.json",
            CONTEXT / "context_heads.json",
            map_path,
            cold,
        )
    }
    protocol = dict(
        scope="ENDED20_CONDITIONAL_LOCAL_NEURAL_SCREEN_NOT_ONLINE_GAIN",
        ids=parent["ids"],
        folds=fold_plan,
        steps=STEPS,
        seed=131,
        lr=0.001,
        source_sha256=parent["source_sha256"],
        cold_sha256=sha(cold),
        input_map_sha256=sha(map_path),
        context_source=str(CONTEXT),
        bound_files_sha256=bound,
        threshold=threshold,
        temporary_parameters=256,
        head_parameters={"shared": 575, "true": 3450, "shuffled": 3450},
        context_parameters_frozen=696,
        canonical_point_labels="EAST125ms preserved separately",
        outer_task="revealed shot outcome at queries with lead>30ms",
        context_input="origin prefix only; no current/future event or outcome",
        reset_each_local_sample=True,
        clock_is_local_not_deployed_stride128=True,
        cold_parent_previously_saw_all20=True,
        suffix_raw_reads=0,
        independent_confirmation=False,
        plan_sha256=sha(PLAN),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sorted(files):
            archive.write(p, str(p.relative_to(io23.ROOT)))
    if preflight_only:
        sid = 69854
        assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        flag, down = io23.read_outcome("EAST", sid)
        lead = round((down - raw["times"][160]) * 1000, 9) if flag else None
        example = build_example(
            model,
            signal,
            basis,
            context_heads,
            sid=sid,
            query=160,
            flag=flag,
            lead=lead,
            point_label=int(flag and lead <= 125),
        )
        result = preflight(model, example, signal, basis)
        result["outcome_reads"] = 1
    else:
        prereq = io23.ROOT / "probe_outputs/event_conditional_preflight_v1"
        assert read(prereq / "completed.json")["status"] == "PASS"
        assert read(prereq / "protocol.json")["plan_sha256"] == sha(PLAN)
        examples = []
        for sid in parent["ids"]:
            assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            for row in rows:
                if row["shot"] == sid:
                    examples.append(
                        build_example(
                            model,
                            signal,
                            basis,
                            context_heads,
                            sid=sid,
                            query=row["query"],
                            flag=row["disruptive"],
                            lead=row["actual_lead_ms"],
                            point_label=row["label"],
                        )
                    )
        records = []
        for fold in fold_plan:
            print(json.dumps(dict(stage="fit", fold=fold["fold"])), flush=True)
            fitted, history = fit(
                model,
                [e for e in examples if e["shot"] in fold["fit"]],
                fold["fit"],
                threshold,
            )
            write(root / f"fold{fold['fold']}_history.json", history)
            write(
                root / f"fold{fold['fold']}_heads.json",
                {a: b.cpu().tolist() for a, b in fitted.items()},
            )
            for e in examples:
                if e["shot"] in fold["held"]:
                    records.append(
                        dict(fold=fold["fold"], **evaluate(model, e, fitted, threshold))
                    )
        assert len(records) == len({(r["shot"], r["query"]) for r in records}) == 100
        summary = {
            a: summarize(
                [
                    dict(
                        shot=r["shot"],
                        disruptive=r["disruptive"],
                        actual_lead_ms=r["actual_lead_ms"],
                        **r["results"][a],
                    )
                    for r in records
                ]
            )
            for a in records[0]["results"]
        }
        write(root / "records.json", records)
        result = dict(
            status="COMPLETED_LOCAL_NEURAL_SCREEN",
            summary=summary,
            gates=gates(summary),
            ready_for_full_stream=gates(summary)["passed"],
            records_sha256=sha(root / "records.json"),
            fold_files_sha256={p.name: sha(p) for p in root.glob("fold*.json")},
            online_gain_measured=False,
            method_promoted=False,
            scientific_goal_achieved=False,
            independent_future_confirmation=False,
        )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    assert all(sha(Path(p)) == h for p, h in bound.items())
    assert all(sha(io23.ROOT / p) == h for p, h in protocol["code_sha256"].items())
    result.update(
        source_tensors_unchanged=True,
        source_parameter_updates=0,
        suffix_raw_reads=0,
        seconds=time.monotonic() - started,
        protocol_sha256=sha(root / "protocol.json"),
        code_snapshot_sha256=sha(root / "code_snapshot.zip"),
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight_only=args.preflight_only)
