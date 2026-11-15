"""Fixed four-fold ended20 auxiliary-head training and real neural evaluation."""

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
from ..difference_feedback_probe_20261102.run import risk_loss
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import encode, forward
from ..warmstart_alignment_20261022.alignment import transform
from .core import outer_objective, propose

PLAN = io23.ROOT / "RISK_BRIDGE_FORECASTER_PLAN_20261102.md"
STEPS = 128


def folds(rows, ids):
    roles = {sid: {r["disruptive"] for r in rows if r["shot"] == sid} for sid in ids}
    if len(ids) != len(set(ids)) or any(len(v) != 1 for v in roles.values()):
        raise ValueError("unique shots with consistent revealed outcomes required")
    groups = {
        flag: [sid for sid in ids if next(iter(roles[sid])) == flag]
        for flag in (False, True)
    }
    if len(groups[False]) != 12 or len(groups[True]) != 8:
        raise ValueError("fixed ended20 8D12S contract required")
    result = []
    for i in range(4):
        held = [sid for sid in ids if sid in groups[False][i::4] + groups[True][i::4]]
        result.append(
            dict(fold=i, fit=[sid for sid in ids if sid not in held], held=held)
        )
    return result


def summarize(rows):
    result = {}
    for name in ("class0", "class1", "safe"):
        group = [
            r
            for r in rows
            if (not r["disruptive"] if name == "safe" else r["label"] == int(name[-1]))
        ]
        ids = {r["shot"] for r in group}
        means = [
            np.mean([r["delta_bce"] for r in group if r["shot"] == sid]) for sid in ids
        ]
        if not means:
            raise ValueError("all fixed evaluation groups required")
        result[name] = dict(
            n=len(group),
            n_shots=len(ids),
            mean_delta_bce=float(np.mean(means)),
            helpful_points=sum(r["delta_bce"] < 0 for r in group),
            harmful_points=sum(r["delta_bce"] > 0 for r in group),
        )
    result["balanced_delta_bce"] = (
        result["class0"]["mean_delta_bce"] + result["class1"]["mean_delta_bce"]
    ) / 2
    return result


def fit_heads(model, examples, train_ids):
    if {r["shot"] for r in examples} != set(train_ids):
        raise ValueError("only assigned training shots may enter head fit")
    device = model.base.center.device
    heads = {
        name: torch.nn.Parameter(torch.zeros(25, 23, device=device))
        for name in ("forecast", "meta")
    }
    optimizers = {
        name: torch.optim.Adam([value], lr=0.001) for name, value in heads.items()
    }
    pools = {
        y: {
            sid: [r for r in examples if r["shot"] == sid and r["label"] == y]
            for sid in train_ids
            if any(r["shot"] == sid and r["label"] == y for r in examples)
        }
        for y in (0, 1)
    }
    if not all(pools.values()):
        raise ValueError("both label groups required")
    rng = np.random.default_rng(131)
    history = []
    for step in range(STEPS):
        chosen = []
        for label in (0, 1):
            sid = list(pools[label])[int(rng.integers(len(pools[label])))]
            candidates = pools[label][sid]
            chosen.append(candidates[int(rng.integers(len(candidates)))])
        record = dict(
            step=step + 1,
            samples=[{k: r[k] for k in ("shot", "query", "label")} for r in chosen],
        )
        for name, table in heads.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for example in chosen:
                r = outer_objective(
                    model,
                    example["origin_pack"],
                    example["later_pack"],
                    example["truth"]["real"],
                    example["truth"]["hold"],
                    example["mask"],
                    example["label"],
                    table,
                )
                objective = r[name] / 2
                if not torch.isfinite(objective):
                    raise FloatingPointError("nonfinite auxiliary outer objective")
                objective.backward()
                total += float(objective.detach())
            norm = torch.nn.utils.clip_grad_norm_([table], 1.0, error_if_nonfinite=True)
            optimizer.step()
            record[name] = dict(
                objective=total,
                head_gradient_norm=float(norm),
                head_norm=float(table.detach().double().norm()),
            )
        history.append(record)
        if (step + 1) % 32 == 0:
            print(
                json.dumps(
                    dict(
                        stage="fit",
                        step=step + 1,
                        train_ids=train_ids,
                        objective={name: record[name]["objective"] for name in heads},
                    )
                ),
                flush=True,
            )
    assert all(p.grad is None for p in model.parameters())
    return {k: v.detach() for k, v in heads.items()}, history


def evaluate(model, example, heads):
    z = example["truth"]["real"].new_zeros(4, 64)
    with torch.no_grad():
        before = forward(model, example["later_pack"], z)["logit"]
        before_bce = risk_loss(before, example["label"])
    result = {k: example[k] for k in ("shot", "query", "label", "disruptive")}
    result["before_logit"] = float(before)
    result["before_bce"] = float(before_bce)
    result["results"] = {}
    for name, table in dict(
        original=example["truth"]["real"].new_zeros(25, 23), **heads
    ).items():
        for mode in ("real", "reverse", "hold"):
            after, receipt = propose(
                model,
                example["origin_pack"],
                z,
                example["truth"][mode],
                example["truth"]["hold"],
                example["mask"],
                table,
                create_graph=False,
            )
            with torch.no_grad():
                out = forward(model, example["later_pack"], after)
                bce = risk_loss(out["logit"], example["label"])
            if mode == "hold":
                assert receipt["gradient_norm"] == 0 and receipt["step_norm"] == 0
                assert torch.equal(bce, before_bce)
            result["results"][name + "_" + mode] = dict(
                after_logit=float(out["logit"]),
                after_bce=float(bce),
                delta_bce=float(bce - before_bce),
                query_action_rms=float(out["rms"].max()),
                receipt=receipt,
            )
    return result


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    source = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"
    assert (
        sha(source / "completed.json")
        == "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
    )
    parent, done = read(source / "protocol.json"), read(source / "completed.json")
    assert sha(source / "samples.json") == done["samples_sha256"]
    preflight = io23.ROOT / "probe_outputs/risk_bridge_forecaster_preflight20_v1"
    assert (
        read(preflight / "completed.json")["status"]
        == "COMPLETED_NUMERICAL_META_PREFLIGHT"
    )
    preflight_protocol = read(preflight / "protocol.json")
    assert preflight_protocol["plan_sha256"] == sha(PLAN)
    rows = read(source / "samples.json")
    plan = folds(rows, parent["ids"])
    model, _, cold = load_model()
    assert sha(cold) == parent["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    mapping_path = Path(parent["calibration_reference"]) / "input_map.json"
    assert sha(mapping_path) == parent["input_map_sha256"]
    mapping = read(mapping_path)
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    root.mkdir(parents=True)
    protocol = dict(
        scope="FOUR_FOLD_CONDITIONAL_LOCAL_HEAD_META_TRAINING_NOT_ONLINE",
        ids=parent["ids"],
        folds=plan,
        steps=128,
        batch=2,
        learning_rate=0.001,
        gradient_clip=1.0,
        seed=131,
        head_parameters=575,
        epoch_selection=False,
        source_samples_sha256=sha(source / "samples.json"),
        cold_sha256=sha(cold),
        input_map_sha256=sha(mapping_path),
        source_sha256=parent["source_sha256"],
        preflight_completed_sha256=sha(preflight / "completed.json"),
        cold_parent_saw_all20=True,
        new_outcome_reads=0,
        suffix_raw_reads=0,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    examples = []
    for sid in parent["ids"]:
        assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        for row in rows:
            if row["shot"] != sid:
                continue
            q = row["query"]
            origin = encode(
                model,
                {k: v[: row["origin"] + 1] for k, v in signal.items()},
                [row["origin"]],
                basis,
            )
            later = encode(
                model, {k: v[: q + 1] for k, v in signal.items()}, [q], basis
            )
            truth, mask = targets(signal, row["origin"], q - 1)
            device = model.base.center.device
            examples.append(
                dict(
                    shot=sid,
                    query=q,
                    label=row["label"],
                    disruptive=row["disruptive"],
                    origin_pack=origin,
                    later_pack=later,
                    truth={
                        k: torch.as_tensor(v, device=device) for k, v in truth.items()
                    },
                    mask=torch.as_tensor(mask, device=device),
                )
            )
    records = []
    for fold in plan:
        print(json.dumps(dict(stage="start_fold", **fold)), flush=True)
        train = [r for r in examples if r["shot"] in fold["fit"]]
        heads, history = fit_heads(model, train, fold["fit"])
        write(root / f"fold{fold['fold']}_history.json", history)
        write(
            root / f"fold{fold['fold']}_heads.json",
            {k: v.cpu().tolist() for k, v in heads.items()},
        )
        for example in examples:
            if example["shot"] in fold["held"]:
                record = evaluate(model, example, heads)
                record["fold"] = fold["fold"]
                records.append(record)
        print(
            json.dumps(
                dict(
                    stage="evaluated_fold", fold=fold["fold"], held_records=len(records)
                )
            ),
            flush=True,
        )
    assert (
        len(records) == 100 and len({(r["shot"], r["query"]) for r in records}) == 100
    )
    summaries = {}
    for arm in records[0]["results"]:
        summaries[arm] = summarize(
            [
                dict(
                    shot=r["shot"],
                    label=r["label"],
                    disruptive=r["disruptive"],
                    **r["results"][arm],
                )
                for r in records
            ]
        )
    meta = summaries["meta_real"]
    gates = {
        name: meta[name]["mean_delta_bce"] < 0 for name in ("class0", "class1", "safe")
    }
    for arm in ("original_real", "forecast_real", "meta_reverse"):
        gates["better_than_" + arm] = (
            meta["balanced_delta_bce"] < summaries[arm]["balanced_delta_bce"]
        )
    gates["coverage"] = len(records) == 100
    gates["passed"] = all(gates.values())
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    write(root / "records.json", records)
    result = dict(
        status="COMPLETED_FOUR_FOLD_LOCAL_NEURAL_HEAD_TEST",
        summary=summaries,
        gates=gates,
        ready_for_full_stream=gates["passed"],
        head_fit_shots_per_fold=15,
        head_held_shots_per_fold=5,
        no_final_full20_head_fitted=True,
        source_tensors_unchanged=True,
        head_is_auxiliary_only=True,
        online_gain_measured=False,
        independent_future_confirmation=False,
        scientific_goal_achieved=False,
        records_sha256=sha(root / "records.json"),
        fold_files_sha256={f.name: sha(f) for f in root.glob("fold*.json")},
        seconds=time.monotonic() - started,
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
