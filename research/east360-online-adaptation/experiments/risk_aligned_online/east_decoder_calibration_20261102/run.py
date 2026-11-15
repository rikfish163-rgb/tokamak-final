"""Four-fold ended-EAST20 update-rule calibration; never a future-stream score."""

import argparse
import math
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.objectives import margin_loss
from ..direct_decoder_meta_20261102.core import advance, clone_head, prediction
from ..direct_decoder_meta_20261102.east20 import LOCAL, SOURCE
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..event_conditional_feedback_20261102.pilot import summarize
from ..group_protected_meta_20261102.objectives import paired_risk
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import encode, forward, direct
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..risk_bridge_forecaster_20261102.prepare import folds
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import source_basis
from ..warmstart_alignment_20261022.alignment import transform
from .core import LowRankDecoder, training_eligible

PLAN = io23.ROOT / "EAST_DECODER_CALIBRATION_PLAN_20261102.md"
ARMS = ("forecast_only", "mean_meta", "worst_meta")


def gates(s):
    candidate = s["worst_meta_real"]
    g = {
        name + "_improved": candidate[name]["mean_delta_bce"] < -1e-8
        for name in ("disruptive", "safe", "danger_window")
    }
    for name in (
        "forecast_only_real",
        "mean_meta_real",
        "prior_real",
        "worst_meta_reverse",
        "worst_meta_hold",
    ):
        g["better_than_" + name] = candidate["balanced"] < s[name]["balanced"] - 1e-8
    g["passed"] = all(g.values())
    return g


def collect(model, basis, p):
    rows = read(LOCAL / "samples.json")
    mapping = read(Path(p["calibration_reference"]) / "input_map.json")
    ids = p["ids"]
    readers = (io23.read_signals, io23.read_outcome)

    def checked(reader):
        def call(device, sid, *args, **kw):
            if device != "EAST" or sid not in ids:
                raise ValueError("only previously ended EAST20 allowed")
            return reader(device, sid, *args, **kw)

        return call

    examples, signals = [], {}
    with (
        patch.object(io23, "read_signals", side_effect=checked(readers[0])),
        patch.object(io23, "read_outcome", side_effect=checked(readers[1])),
    ):
        for sid in ids:
            if sha(io23.path_for("EAST", sid)) != p["input_hashes"][str(sid)]:
                raise ValueError("EAST raw hash mismatch")
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            signals[sid] = signal
            flag, down = io23.read_outcome("EAST", sid)
            for r in rows:
                if r["shot"] != sid:
                    continue
                lead = (
                    round((down - raw["times"][r["query"]]) * 1000, 9) if flag else None
                )
                if flag != r["disruptive"] or (
                    flag and (lead <= 30 or abs(lead - r["actual_lead_ms"]) > 1e-6)
                ):
                    raise ValueError("authoritative outcome/timing changed")
                meta = {
                    k: r[k]
                    for k in (
                        "shot",
                        "origin",
                        "query",
                        "label",
                        "disruptive",
                        "actual_lead_ms",
                    )
                }
                if meta["origin"] != meta["query"] - 26:
                    raise ValueError("local feedback clock changed")
                truth, mask = targets(
                    {k: v[: r["query"]] for k, v in signal.items()},
                    r["origin"],
                    r["query"] - 1,
                )
                examples.append(
                    dict(
                        **meta,
                        origin_pack=encode(
                            model,
                            {k: v[: r["origin"] + 1] for k, v in signal.items()},
                            [r["origin"]],
                            basis,
                        ),
                        later_pack=encode(
                            model,
                            {k: v[: r["query"] + 1] for k, v in signal.items()},
                            [r["query"]],
                            basis,
                        ),
                        truth={
                            k: torch.as_tensor(v, device="cuda")
                            for k, v in truth.items()
                        },
                        mask=torch.as_tensor(mask, device="cuda"),
                    )
                )
    if len(examples) != 100 or len({(e["shot"], e["query"]) for e in examples}) != 100:
        raise ValueError("all fixed 100 episodes required")
    return examples, signals


def step(model, e, head, mode="real", outer=False):
    return advance(
        model,
        e["origin_pack"],
        model.base.center.new_zeros(4, 64),
        e["truth"][mode],
        e["mask"],
        head,
        create_graph=outer,
    )


def fit(model, examples, fit_ids, prior, basis, threshold):
    if {e["shot"] for e in examples} != set(fit_ids):
        raise ValueError("head fit may see assigned training shots only")
    pools = {
        flag: {
            sid: [e for e in examples if e["shot"] == sid and training_eligible(e)]
            for sid in fit_ids
            if any(e["shot"] == sid and e["disruptive"] == flag for e in examples)
        }
        for flag in (False, True)
    }
    if any(not entries for group in pools.values() for entries in group.values()):
        raise ValueError("each training shot needs a legal episode")
    heads = {a: LowRankDecoder(prior, basis) for a in ARMS}
    optimizers = {a: torch.optim.Adam([h.delta], lr=0.001) for a, h in heads.items()}
    rng, history = np.random.default_rng(131), []
    threshold_logit = math.log(threshold / (1 - threshold))
    for n in range(128):
        chosen = []
        for flag in (False, True):
            sid = list(pools[flag])[int(rng.integers(len(pools[flag])))]
            chosen.append(pools[flag][sid][int(rng.integers(len(pools[flag][sid])))])
        record = dict(
            step=n + 1,
            samples=[{k: e[k] for k in ("shot", "query", "label")} for e in chosen],
            arms={},
        )
        for arm, head in heads.items():
            optimizers[arm].zero_grad(set_to_none=True)
            abs_losses, before, after, receipts = [], [], [], []
            for e in chosen:
                state, receipt = step(model, e, head, outer=arm != "forecast_only")
                after.append(forward(model, e["later_pack"], state)["logit"])
                with torch.no_grad():
                    before.append(
                        forward(model, e["later_pack"], state.new_zeros(4, 64))["logit"]
                    )
                pred = prediction(model, e["origin_pack"], state.new_zeros(4, 64), head)
                abs_losses.append(
                    model.base.forecast_loss(pred, e["truth"]["real"], e["mask"])
                )
                receipts.append(receipt)
            z0, z = (
                torch.cat(before).double() - threshold_logit,
                torch.cat(after).double() - threshold_logit,
            )
            labels = z.new_tensor([0.0, 1.0])
            risk = {m: paired_risk(z, z0, labels, mode=m) for m in ("mean", "worst")}
            absolute = torch.stack(abs_losses).mean()
            objective = 0.20 * absolute + 1e-4 * head.delta.square().mean()
            if arm != "forecast_only":
                objective = (
                    objective
                    + risk["mean" if arm == "mean_meta" else "worst"]["risk"] / 0.1
                )
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [head.delta], 1.0, error_if_nonfinite=True
            )
            optimizers[arm].step()
            record["arms"][arm] = dict(
                objective=float(objective.detach()),
                absolute_loss=float(absolute.detach()),
                risk_changes=risk["mean"]["delta"].detach().cpu().tolist(),
                gradient_norm=float(norm),
                accepted=sum(r["accepted"] for r in receipts),
            )
        history.append(record)
        if (n + 1) % 32 == 0:
            print(
                dict(stage="fit", step=n + 1, fit_ids=fit_ids, arms=record["arms"]),
                flush=True,
            )
    for h in heads.values():
        h.delta.requires_grad_(False)
    return heads, history


def evaluate(model, examples, heads, threshold, fold):
    rows = []
    for e in examples:
        with torch.no_grad():
            z0 = forward(model, e["later_pack"], model.base.center.new_zeros(4, 64))[
                "logit"
            ]
            l0 = margin_loss(z0, threshold, int(e["disruptive"]))
        for arm, h in heads.items():
            for mode in ("real", "reverse", "hold"):
                state, r = step(model, e, h, mode)
                with torch.no_grad():
                    out = forward(model, e["later_pack"], state)
                    l1 = margin_loss(out["logit"], threshold, int(e["disruptive"]))
                rows.append(
                    dict(
                        fold=fold,
                        arm=arm + "_" + mode,
                        **{
                            k: e[k]
                            for k in (
                                "shot",
                                "query",
                                "label",
                                "disruptive",
                                "actual_lead_ms",
                            )
                        },
                        before_logit=float(z0),
                        after_logit=float(out["logit"]),
                        before_bce=float(l0),
                        after_bce=float(l1),
                        delta_bce=float(l1 - l0),
                        accepted=r["accepted"],
                        query_rms=float(out["rms"].max()),
                        loss_before=r.get("loss_before"),
                        loss_proposal=r.get("loss_proposal"),
                    )
                )
    return rows


def preflight(model, e, prior, basis, signal, ffn_basis):
    h = LowRankDecoder(prior, basis)

    def risk():
        state, receipt = step(model, e, h, outer=True)
        return forward(model, e["later_pack"], state)["logit"], state, receipt

    z, state, receipt = risk()
    (gradient,) = torch.autograd.grad(z.sum(), h.delta)
    norm = gradient.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise AssertionError("no actual later-risk derivative into C")
    direction = gradient / norm
    trials = []
    for epsilon in (0.005, 0.0025):
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                h.delta.copy_(sign * epsilon * direction)
            value, _, r = risk()
            if r["accepted"] != receipt["accepted"]:
                raise AssertionError("finite difference crossed acceptance branch")
            values.append(float(value.detach()))
        numeric = (values[0] - values[1]) / (2 * epsilon)
        if abs(numeric - float(norm)) > 2e-4 + 0.05 * float(norm):
            raise AssertionError((numeric, float(norm)))
        trials.append(dict(epsilon=epsilon, analytic=float(norm), numeric=numeric))
    prefix = {k: v[: e["query"] + 1] for k, v in signal.items()}
    with torch.no_grad():
        actual = direct(model, prefix, [e["query"]], ffn_basis, state.detach())["logit"]
    torch.testing.assert_close(actual, z.detach(), atol=0, rtol=0)
    return dict(
        status="PASS",
        risk_gradient_norm=float(norm),
        finite_difference=trials,
        actual_ffn_exact=True,
    )


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    p = read(LOCAL / "protocol.json")
    if sha(LOCAL / "samples.json") != read(LOCAL / "completed.json")["samples_sha256"]:
        raise ValueError("fixed local samples changed")
    source_done = read(SOURCE / "completed.json")
    if sha(SOURCE / "decoders.pt") != source_done["artifact_sha256"]["decoders.pt"]:
        raise ValueError("source prior changed")
    model, configuration, cold = load_model()
    if (
        sha(cold) != p["cold_state_sha256"]
        or sha(Path(p["calibration_reference"]) / "input_map.json")
        != p["input_map_sha256"]
    ):
        raise ValueError("cold parent/mapping changed")
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prior = clone_head(model)
    saved = torch.load(SOURCE / "decoders.pt", map_location="cpu", weights_only=True)
    if saved["source_sha256"] != SOURCE_SHA:
        raise ValueError("wrong source prior")
    prior.load_state_dict(saved["heads"]["forecast_only"], strict=True)
    prior.eval().requires_grad_(False)
    _, _, vh = torch.linalg.svd(
        prior.weight.detach().cpu().double(), full_matrices=False
    )
    basis = vh[:4].to(prior.weight)
    ffn_basis = source_basis(model)
    examples, signals = collect(model, ffn_basis, p)
    fold_plan = folds(examples, p["ids"])
    threshold = float(configuration["threshold"])
    root.mkdir(parents=True)
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    protocol = dict(
        scope="ENDED_EAST20_LOW_RANK_DECODER_CALIBRATION",
        ids=p["ids"],
        folds=fold_plan,
        source_sha256=SOURCE_SHA,
        cold_sha256=sha(cold),
        input_map_sha256=p["input_map_sha256"],
        threshold=threshold,
        prior_sha256=sha(SOURCE / "decoders.pt"),
        bound_code_sha256=bound,
        steps_per_fold=128,
        lr=0.001,
        seed=131,
        trainable_parameters=2300,
        arms=list(ARMS),
        source_parameters_updated=0,
        suffix_raw_reads=0,
        cold_parent_previously_saw_all20=True,
        independent_confirmation=False,
        full_stream_not_run=True,
        training_queries="safe or disruptive 30<lead<=125ms; canonical labels preserved",
    )
    write(root / "protocol.json", protocol)
    torch.save(basis.cpu(), root / "diagnostic_basis.pt")
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(bound):
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
    write(
        root / "preflight.json",
        preflight(
            model, examples[0], prior, basis, signals[examples[0]["shot"]], ffn_basis
        ),
    )
    rows = []
    for fold in fold_plan:
        training = [e for e in examples if e["shot"] in fold["fit"]]
        held = [e for e in examples if e["shot"] in fold["held"]]
        heads, history = fit(model, training, fold["fit"], prior, basis, threshold)
        write(root / f"fold{fold['fold']}_training.json", history)
        torch.save(
            {a: h.delta.cpu() for a, h in heads.items()},
            root / f"fold{fold['fold']}_heads.pt",
        )
        heads["prior"] = LowRankDecoder(prior, basis).requires_grad_(False)
        rows.extend(evaluate(model, held, heads, threshold, fold["fold"]))
    summary = {
        arm: summarize([r for r in rows if r["arm"] == arm])
        for arm in sorted({r["arm"] for r in rows})
    }
    write(root / "records.json", rows)
    write(root / "summary.json", summary)
    if (
        any(
            not torch.equal(v, model.state_dict()[k].detach().cpu())
            for k, v in frozen.items()
        )
        or sha(DEFAULT_PARENT) != SOURCE_SHA
    ):
        raise AssertionError("source/cold parameters changed")
    if any(sha(Path(path)) != digest for path, digest in bound.items()):
        raise AssertionError("bound code changed")
    done = dict(
        status="COMPLETED",
        seconds=time.monotonic() - start,
        n_rows=len(rows),
        n_samples=100,
        gate=gates(summary),
        source_unchanged=True,
        cold_unchanged=True,
        suffix_raw_reads=0,
        online_gain_measured=False,
        method_promoted=False,
        artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
