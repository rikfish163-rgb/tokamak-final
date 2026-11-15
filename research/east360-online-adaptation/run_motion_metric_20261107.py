"""Source-only offline teaching of an observed-motion neural update rule."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import moving_peak_replay_20261105 as r
import motion_metric_model_20261107 as model
from motion_metric_kernel_20261107 import MotionMetric, descent, magnitude_descent, match_state, potential
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.learned_update_loss_20261023.learning import context_features


DESIGN = r.ROOT / "probe_outputs/source_anticipation_design_20261104_v2"
GROUPS = ("safe", "earlier", "timely")
VIEWS = ("conditional", "global")
ARMS = ("static", "fixed", "global", "conditional", "time", "channel", "reverse", "scalar")
STEPS, BATCH, STEP = 256, 12, .35


def source_rows(role):
    rows = r.read(DESIGN / f"{role}_records.json")
    allowed = set(r.simple_data.source_roles()[role])
    if any(row["role"] != role or row["shot"] not in allowed for row in rows):
        raise ValueError("source role membership must stay locked")
    return rows


def teaching_group(flag, lead, horizon, teaching):
    if teaching not in ("source", "aux125"):
        raise ValueError("declared source or auxiliary teaching contract required")
    if not flag:
        return "safe"
    deadline = 10 if teaching == "source" else 30
    boundary = horizon + 1e-6 if teaching == "source" else 125
    if lead <= deadline:
        return None
    return "timely" if lead <= boundary else "earlier"


def prepare(base, role, *, teaching="aux125", include_past=False, encoder=model.encode_windows):
    """Offline episode sampling; outcomes never enter the update arguments."""
    rows = source_rows(role)
    all_ids = sorted({row["shot"] for row in rows})
    arrays = {k: [] for k in ("x0", "a0", "xr", "ar", "xq", "aq", "truth", "mask")}
    if include_past:
        arrays.update({k: [] for k in ("xp", "ap", "xpp", "app")})
    advanced = r.simple_data.advanced_times() if teaching == "source" else None
    selected, skipped = [], 0
    for number, sid in enumerate(all_ids, 1):
        raw = apply_input_contract(r.simple_data.read_shot("J-TEXT", sid))
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        horizon = r.io23.effective_horizon_ms(sid, advanced)[0] if teaching == "source" else 125.
        raw["available"][:, 2] = False
        candidates = {key: [] for key in GROUPS}
        for origin in sorted({q["origin"] for q in rows if q["shot"] == sid}):
            release, query = origin + 26, origin + 42
            if origin < 127 or query >= len(raw["times"]):
                skipped += 1
                continue
            lead = round(float((down - raw["times"][query]) * 1000), 9) if flag else None
            group = teaching_group(flag, lead, horizon, teaching)
            if group is None:
                skipped += 1
                continue
            candidates[group].append(dict(shot=sid, origin=origin, release=release, query=query,
                group=group, label=int(group == "timely"), disruptive=bool(flag),
                down=down if flag else None, lead_ms=lead, query_time=float(raw["times"][query])))
        for group, pool in candidates.items():
            if not pool:
                continue
            row = pool[len(pool)//2]
            if teaching == "source":
                row["source_horizon_ms"] = horizon
            selected.append(row)
            for name, end in (("0", row["origin"]), ("r", row["release"]), ("q", row["query"])):
                x, a = r.simple_data.window(raw, end, 128)
                arrays[f"x{name}"].append(x)
                arrays[f"a{name}"].append(a)
            if include_past:
                for name, end in (("p", row["release"]-8), ("pp", row["release"]-16)):
                    x, a = r.simple_data.window(raw, end, 128)
                    arrays[f"x{name}"].append(x)
                    arrays[f"a{name}"].append(a)
            o = row["origin"]
            arrays["truth"].append(raw["values"][o+1:o+26])
            arrays["mask"].append(raw["available"][o+1:o+26])
        if number % 200 == 0 or number == len(all_ids):
            print("source", role, number, "of", len(all_ids), "episodes", len(selected), flush=True)
    if not all(any(q["group"] == g for q in selected) for g in GROUPS):
        raise ValueError("all three offline teaching strata required")
    values = {k: torch.as_tensor(np.stack(v), device="cuda") for k, v in arrays.items()}
    basis = model.basis_for(base)
    packs = {}
    for name in (("0", "r", "q", "p", "pp") if include_past else ("0", "r", "q")):
        chunks = [encoder(base, values[f"x{name}"][i:i+128], values[f"a{name}"][i:i+128], basis)
                  for i in range(0, len(selected), 128)]
        packs[name] = {k: torch.cat([chunk[k] for chunk in chunks]) for k in chunks[0]}
    motion, mask = model.motion_target(base, values["x0"], values["a0"], values["truth"], values["mask"])
    context = context_features(values["xr"], values["ar"], base.center, base.scale)
    sensitivity = []
    for i in range(0, len(selected), 128):
        pack = {k: v[i:i+128] for k, v in packs["q"].items()}
        variable = torch.zeros(len(pack["h"]), 4, 64, device="cuda", requires_grad=True)
        z = model.forward_state(base, pack, variable)["logit"]
        gradient, = torch.autograd.grad(z.sum(), variable)
        sensitivity.append(STEP * gradient.double().flatten(1).norm(dim=1).detach())
    return dict(packs=packs, motion=motion, mask=mask, context=context,
                scale=torch.cat(sensitivity).clamp_min(1e-6),
                auxiliary_labels=torch.tensor([int(q["disruptive"] and q["lead_ms"] <= 125) for q in selected], device="cuda"),
                labels=torch.tensor([q["label"] for q in selected], device="cuda")), selected, skipped


def batch(data, indices):
    return dict(packs={s: {k: v[indices] for k, v in p.items()} for s, p in data["packs"].items()},
                **{k: v[indices] for k, v in data.items() if k != "packs"})


def state(base, origin, context, motion, mask, metric, *, mode="real", create_graph=False, global_context=False, update_scale=None):
    if mode in ("time", "channel"):
        motion = model.scramble(motion, mask, axis=mode)
    elif mode == "reverse":
        motion = -motion
    elif mode != "real":
        raise ValueError("declared feedback control required")
    variable = torch.zeros(len(context), 4, 64, device=context.device, requires_grad=True)
    prediction = model.forward_state(base, origin, variable)["forecast"]
    weights = torch.ones_like(context[:, :23]) if metric is None else metric(torch.zeros_like(context) if global_context else context)
    loss = potential(prediction, motion, mask, weights)
    gradient, = torch.autograd.grad(loss.sum(), variable, create_graph=create_graph)
    result = (descent(gradient, step=STEP) if update_scale is None
              else magnitude_descent(gradient, reference_norm=update_scale, step=STEP))
    return result, weights


def calibration(base, data):
    norms = []
    for lo in range(0, len(data["context"]), 128):
        d = batch(data, np.arange(lo, min(lo+128, len(data["context"]))))
        variable = torch.zeros(len(d["context"]), 4, 64, device="cuda", requires_grad=True)
        prediction = model.forward_state(base, d["packs"]["0"], variable)["forecast"]
        loss = potential(prediction, d["motion"], d["mask"], torch.ones(len(variable), 23, device="cuda"))
        gradient, = torch.autograd.grad(loss.sum(), variable)
        norms.append(gradient.detach().flatten(1).norm(dim=1))
    reference = float(torch.quantile(torch.cat(norms), .9))
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("TRAIN motion does not support positive gradient calibration")
    return reference


def risk_delta(before, after, labels, tau):
    y = labels.double()
    return (F.binary_cross_entropy_with_logits(after.double()-tau, y, reduction="none")
            - F.binary_cross_entropy_with_logits(before.detach().double()-tau, y, reduction="none"))


def fit(out, geometry="unit", teaching="aux125"):
    start = time.monotonic()
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    frozen = {k: v.detach().clone() for k, v in base.state_dict().items()}
    data, rows, skipped = prepare(base, "train", teaching=teaching)
    update_scale = calibration(base, data) if geometry == "magnitude" else None
    tau = math.log(parent["threshold"]/(1-parent["threshold"]))
    prototype = MotionMetric().cuda()
    views = (*VIEWS, "auxiliary125") if teaching == "source" else VIEWS
    models = {name: deepcopy(prototype) for name in views}
    optimizers = {k: torch.optim.Adam(v.parameters(), lr=1e-3) for k, v in models.items()}
    rng = np.random.default_rng(131)
    pools = {g: np.array([i for i, q in enumerate(rows) if q["group"] == g]) for g in GROUPS}
    out.mkdir(parents=True, exist_ok=False)
    r.write_new(out / "protocol.json", dict(status="OFFLINE_SOURCE_MOTION_METRIC", steps=STEPS, batch=BATCH,
        source=str(r.DEFAULT_PARENT), source_state=r.state_digest(base), threshold=parent["threshold"],
        step=STEP, fit_rows=rows, skipped=skipped, source_roles={g: len(v) for g, v in pools.items()},
        geometry=geometry, update_scale=update_scale,
        teaching=teaching, views=views,
        validation_read_before_fit=False,
        target="original_source_point_contract" if teaching == "source" else "auxiliary30_to_125_not_relabeling_source",
        files={str(p): p.read_text() for p in (Path(__file__), Path(model.__file__),
            r.ROOT / "motion_metric_kernel_20261107.py", r.ROOT / "MOTION_METRIC_PROTOCOL_20261107.md")}))
    history = []
    exposure = {name: dict(total=0, positive=0) for name in views}
    for step in range(1, STEPS+1):
        indices = np.concatenate([rng.choice(pools[g], BATCH//3, replace=True) for g in GROUPS])
        sample = batch(data, indices)
        before = sample["packs"]["q"]["native"]
        for name, metric in models.items():
            labels = sample["auxiliary_labels"] if name == "auxiliary125" else sample["labels"]
            exposure[name]["total"] += len(labels)
            exposure[name]["positive"] += int(labels.sum())
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            current, weights = state(base, sample["packs"]["0"], sample["context"], sample["motion"],
                                     sample["mask"], metric, create_graph=True, global_context=name == "global", update_scale=update_scale)
            with torch.enable_grad():
                after = model.forward_state(base, sample["packs"]["q"], current)["logit"]
                change = risk_delta(before, after, labels, tau) / sample["scale"]
            other, _ = state(base, sample["packs"]["0"], sample["context"], sample["motion"],
                             sample["mask"], metric, mode="time", global_context=name == "global", update_scale=update_scale)
            other = match_state(other.detach(), current.detach())
            with torch.no_grad():
                wrong_z = model.forward_state(base, sample["packs"]["q"], other)["logit"]
                wrong = risk_delta(before, wrong_z, labels, tau) / sample["scale"]
            loss = change.mean() + .25 * F.softplus(change - wrong).mean() + .001 * (weights-1).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite motion metric meta-objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(metric.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=step, arm=name, loss=float(loss.detach()),
                                actual_delta=float(change.detach().mean()),
                                control_delta=float(wrong.mean()), gradient=float(norm)))
        if step % 64 == 0:
            print("fit", step, history[-2:], "seconds", round(time.monotonic()-start, 1), flush=True)
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()) or any(p.grad is not None for p in base.parameters()):
        raise ValueError("source state or gradients changed during auxiliary training")
    torch.save({k: v.state_dict() for k, v in models.items()}, out / "metrics.pt")
    r.write_new(out / "fit.json", dict(status="FIT_COMPLETE", history=history, exposure=exposure, seconds=time.monotonic()-start))
    print("fit complete", len(rows), flush=True)


def aggregate(rows, *, arms=ARMS):
    result = {}
    for group in GROUPS:
        ids = sorted({q["shot"] for q in rows if q["group"] == group})
        # Sampling already yields one episode per physical shot and stratum.
        selected = [q for q in rows if q["group"] == group]
        result[group] = dict(shots=len(ids), before=float(np.mean([q["before_bce"] for q in selected])),
            **{arm: float(np.mean([q["after_bce"][arm] - q["before_bce"] for q in selected])) for arm in arms})
    result["balanced"] = {arm: (result["safe"][arm] + result["timely"][arm])/2 for arm in arms}
    return result


def evaluate(out):
    start = time.monotonic()
    r.configure()
    cfg, done = (r.read(out / f"{k}.json") for k in ("protocol", "fit"))
    if done["status"] != "FIT_COMPLETE":
        raise ValueError("complete source TRAIN fit required before reading VAL")
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    if r.state_digest(base) != cfg["source_state"]:
        raise ValueError("same qualified source required")
    saved = torch.load(out / "metrics.pt", map_location="cuda", weights_only=True)
    models = {k: MotionMetric().cuda().eval() for k in cfg["views"]}
    for k, metric in models.items():
        metric.load_state_dict(saved[k])
        metric.requires_grad_(False)
    data, meta, skipped = prepare(base, "val", teaching=cfg["teaching"])
    arms = (*ARMS, "auxiliary125") if cfg["teaching"] == "source" else ARMS
    predictions = {k: [] for k in arms}
    norms = {k: [] for k in arms if k not in ("static", "scalar")}
    zero_controls, witnesses = 0, []
    for i in range(0, len(meta), 64):
        d = batch(data, np.arange(i, min(i+64, len(meta))))
        states = {}
        jobs = [
                ("conditional", models["conditional"], "real", False),
                ("fixed", None, "real", False), ("global", models["global"], "real", True),
                ("time", models["conditional"], "time", False),
                ("channel", models["conditional"], "channel", False),
                ("reverse", models["conditional"], "reverse", False)]
        if cfg["teaching"] == "source":
            jobs.append(("auxiliary125", models["auxiliary125"], "real", False))
        for arm, metric, mode, global_context in jobs:
            state_value, _ = state(base, d["packs"]["0"], d["context"], d["motion"], d["mask"],
                                   metric, mode=mode, global_context=global_context, update_scale=cfg["update_scale"])
            states[arm] = state_value.detach()
        for arm in ("time", "channel", "reverse"):
            zero_controls += int(((states[arm].flatten(1).norm(dim=1) == 0) &
                                 (states["conditional"].flatten(1).norm(dim=1) > 0)).sum())
            states[arm] = match_state(states[arm], states["conditional"])
        with torch.no_grad():
            predictions["static"].append(d["packs"]["q"]["native"].cpu().numpy())
            for arm, state_value in states.items():
                z = model.forward_state(base, d["packs"]["q"], state_value)["logit"]
                predictions[arm].append(z.cpu().numpy())
                norms[arm].append(state_value.flatten(1).norm(dim=1).cpu().numpy())
            anchor = model.forward_state(base, d["packs"]["r"], states["conditional"])["logit"]
            scalar = d["packs"]["q"]["native"] + anchor - d["packs"]["r"]["native"]
            predictions["scalar"].append(scalar.cpu().numpy())
            if i == 0:
                native = model.forward_state(base, d["packs"]["q"], torch.zeros_like(states["conditional"]))["logit"]
                witnesses.append(dict(zero_source_error=float((native - d["packs"]["q"]["native"]).abs().max()),
                                      state_norm=states["conditional"].flatten(1).norm(dim=1).cpu().tolist()))
    prediction = {k: np.concatenate(v) for k, v in predictions.items()}
    state_norm = {k: np.concatenate(v) for k, v in norms.items()}
    r.save_arrays(out / "val_predictions.npz", **prediction, **{f"norm_{k}": v for k, v in state_norm.items()})
    r.write_new(out / "val_sealed.json", dict(status="SOURCE_VAL_PREDICTIONS_SAVED", rows=meta,
                                               skipped=skipped, zero_controls=zero_controls, witnesses=witnesses))
    tau = math.log(parent["threshold"]/(1-parent["threshold"]))
    rows = []
    for i, row in enumerate(meta):
        # Independent hard-outcome recount after predictions have been saved.
        flag, down = r.io23.read_outcome("J-TEXT", row["shot"])
        lead = round((down - row["query_time"])*1000, 9) if flag else None
        horizon = row["source_horizon_ms"] if cfg["teaching"] == "source" else 125.
        group = teaching_group(flag, lead, horizon, cfg["teaching"])
        if group != row["group"] or group is None:
            raise ValueError("source VAL HDF5 strata changed after prediction seal")
        label = int(group == "timely")
        losses = {k: float(np.logaddexp(0., (1-2*label)*(float(v[i])-tau))) for k, v in prediction.items()}
        rows.append(dict(row, before_bce=losses["static"], after_bce=losses,
                         state_norm={k: float(v[i]) for k, v in state_norm.items()},
                         logits={k: float(v[i]) for k, v in prediction.items()}))
    table = aggregate(rows, arms=arms)
    passes = (all(table[g]["conditional"] < -1e-6 for g in ("safe", "timely"))
              and table["earlier"]["conditional"] <= 1e-6
              and all(table["balanced"]["conditional"] < table["balanced"][k] - 1e-6
                      for k in ("static", "fixed", "global", "time", "channel", "scalar", *(('auxiliary125',) if cfg["teaching"] == "source" else ()))))
    r.write_new(out / "val_result.json", dict(status="SOURCE_VAL_AUXILIARY_SCREEN", table=table, rows=rows,
                                               promote=passes, seconds=time.monotonic()-start))
    print("VAL", table, "promote", passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--geometry", choices=("unit", "magnitude"), default="unit")
    parser.add_argument("--teaching", choices=("source", "aux125"), default="aux125")
    args = parser.parse_args()
    (fit(args.out, args.geometry, args.teaching) if args.phase == "fit" else evaluate(args.out))
