"""Matched source teaching of level and lagged-change update potentials."""

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
from motion_metric_kernel_20261107 import magnitude_descent, match_state
from shape_motion_kernel_20261107 import ShapeMetric, potential
from run_motion_metric_20261107 import GROUPS, STEPS, BATCH, STEP, aggregate, batch, prepare, risk_delta, teaching_group
from run_motion_metric_east20_20261107 import prepare_history


REFERENCE = r.ROOT / "probe_outputs/motion_metric_source_20261107_v3"
KINDS = ("level", "global", "shape")
CONTROLS = ("time", "channel", "reverse", "ablate")
ARMS = ("static", "fixed_level", "fixed_shape", *KINDS, *CONTROLS, "scalar")


def calibration(base, data):
    """Use only arrived source TRAIN targets to align two gradient scales."""
    values = {k: [] for k in ("level", "shape")}
    for lo in range(0, len(data["context"]), 128):
        d = batch(data, np.arange(lo, min(lo+128, len(data["context"]))))
        n = len(d["context"])
        variable = torch.zeros(n, 4, 64, device=d["context"].device, requires_grad=True)
        prediction = model.forward_state(base, d["packs"]["0"], variable)["forecast"]
        ones = prediction.new_ones(n, 23)
        for kind, weights in (("level", torch.stack((ones, ones*0), 1)),
                              ("shape", torch.stack((ones*0, ones), 1))):
            loss = potential(prediction, d["motion"], d["mask"], weights, shape_scale=1.)
            g, = torch.autograd.grad(loss.sum(), variable, retain_graph=kind == "level")
            values[kind].append(g.detach().flatten(1).norm(dim=1))
    refs = {k: float(torch.quantile(torch.cat(v), .9)) for k, v in values.items()}
    if any(not math.isfinite(v) or v <= 0 for v in refs.values()):
        raise ValueError("both TRAIN potentials must have positive finite gradient reference")
    return dict(level_reference=refs["level"], shape_reference=refs["shape"],
                shape_scale=refs["level"]/refs["shape"])


def state(base, data, metric, *, kind, shape_scale, update_scale, mode="real", create_graph=False):
    context, motion, mask = data["context"], data["motion"], data["mask"]
    if mode in ("time", "channel"):
        motion = model.scramble(motion, mask, axis=mode)
    elif mode == "reverse":
        motion = -motion
    elif mode != "real":
        raise ValueError("declared arrived-feedback mode required")
    if metric is None:
        ones = context.new_ones(len(context), 23)
        weights = torch.stack((ones, ones if kind == "fixed_shape" else ones*0), 1)
    else:
        weights = metric(torch.zeros_like(context) if kind == "global" else context)
        if kind in ("level", "ablate"):
            weights = torch.stack((weights[:, 0], weights[:, 1]*0), 1)
    variable = torch.zeros(len(context), 4, 64, device=context.device, requires_grad=True)
    prediction = model.forward_state(base, data["packs"]["0"], variable)["forecast"]
    loss = potential(prediction, motion, mask, weights, shape_scale=shape_scale)
    gradient, = torch.autograd.grad(loss.sum(), variable, create_graph=create_graph)
    return magnitude_descent(gradient, reference_norm=update_scale, step=STEP), weights


def parent():
    r.configure()
    base, checkpoint = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    old = r.read(REFERENCE / "protocol.json")
    if old["teaching"] != "source" or old["geometry"] != "magnitude" or r.state_digest(base) != old["source_state"]:
        raise ValueError("same qualified original-contract V3 parent required")
    return base, checkpoint, old


def fit(out):
    start = time.monotonic()
    base, checkpoint, old = parent()
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    data, rows, skipped = prepare(base, "train", teaching="source")
    if rows != old["fit_rows"]:
        raise ValueError("same prior 1387 source TRAIN episodes required")
    scales = calibration(base, data)
    prototype = ShapeMetric().cuda()
    models = {k: deepcopy(prototype) for k in KINDS}
    optimizers = {k: torch.optim.Adam(v.parameters(), lr=1e-3) for k, v in models.items()}
    pools = {g: np.array([i for i, row in enumerate(rows) if row["group"] == g]) for g in GROUPS}
    rng = np.random.default_rng(131)
    tau = math.log(checkpoint["threshold"]/(1-checkpoint["threshold"]))
    out.mkdir(parents=True, exist_ok=False)
    r.write_new(out / "protocol.json", dict(status="FIXED_LAGGED_MOTION_TEACHING", kinds=KINDS,
        source_state=old["source_state"], source=str(r.DEFAULT_PARENT), threshold=checkpoint["threshold"],
        scales=scales, steps=STEPS, batch=BATCH, rows=rows, skipped=skipped,
        source_files={str(p): p.read_text() for p in (Path(__file__), Path(model.__file__),
            r.ROOT / "shape_motion_kernel_20261107.py", r.ROOT / "motion_metric_kernel_20261107.py",
            r.ROOT / "run_motion_metric_20261107.py", r.ROOT / "run_motion_metric_east20_20261107.py",
            r.ROOT / "SHAPE_MOTION_PROTOCOL_20261107.md")}))
    history = []
    for step in range(1, STEPS+1):
        indices = np.concatenate([rng.choice(pools[g], BATCH//3, replace=True) for g in GROUPS])
        d = batch(data, indices)
        for kind, metric in models.items():
            optimizer = optimizers[kind]
            optimizer.zero_grad(set_to_none=True)
            updated, weights = state(base, d, metric, kind=kind,
                shape_scale=scales["shape_scale"], update_scale=scales["level_reference"], create_graph=True)
            after = model.forward_state(base, d["packs"]["q"], updated)["logit"]
            change = risk_delta(d["packs"]["q"]["native"], after, d["labels"], tau) / d["scale"]
            other, _ = state(base, d, metric, kind=kind, mode="time",
                shape_scale=scales["shape_scale"], update_scale=scales["level_reference"])
            other = match_state(other.detach(), updated.detach())
            with torch.no_grad():
                wrong = model.forward_state(base, d["packs"]["q"], other)["logit"]
                wrong_change = risk_delta(d["packs"]["q"]["native"], wrong, d["labels"], tau) / d["scale"]
            anchor = (weights[:, 0]-1).square().mean() + weights[:, 1].square().mean()
            loss = change.mean() + .25 * F.softplus(change-wrong_change).mean() + .001 * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite lagged-motion teaching objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(metric.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=step, kind=kind, loss=float(loss.detach()),
                                risk_delta=float(change.detach().mean()), gradient=float(norm)))
        if step % 64 == 0:
            print("step", step, history[-3:], "seconds", round(time.monotonic()-start, 1), flush=True)
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()) or any(p.grad is not None for p in base.parameters()):
        raise ValueError("source changed during update-rule teaching")
    torch.save({k: m.state_dict() for k, m in models.items()}, out / "models.pt")
    r.write_new(out / "fit.json", dict(status="FIT_COMPLETE", seconds=time.monotonic()-start, history=history))
    print("fit complete", len(rows), "scales", scales, flush=True)


def apply(base, models, data, scales):
    states = {}
    for kind, metric in models.items():
        w, _ = state(base, data, metric, kind=kind, shape_scale=scales["shape_scale"],
                     update_scale=scales["level_reference"])
        states[kind] = w.detach()
    for kind in ("fixed_level", "fixed_shape"):
        w, _ = state(base, data, None, kind=kind, shape_scale=scales["shape_scale"],
                     update_scale=scales["level_reference"])
        states[kind] = w.detach()
    for control in CONTROLS:
        w, _ = state(base, data, models["shape"], kind="ablate" if control == "ablate" else "shape",
            mode=control if control != "ablate" else "real", shape_scale=scales["shape_scale"],
            update_scale=scales["level_reference"])
        states[control] = match_state(w.detach(), states["shape"])
    with torch.no_grad():
        logits = {k: model.forward_state(base, data["packs"]["q"], w)["logit"].cpu().numpy()
                  for k, w in states.items()}
        logits["static"] = data["packs"]["q"]["native"].cpu().numpy()
        anchor = model.forward_state(base, data["packs"]["r"], states["shape"])["logit"]
        logits["scalar"] = (data["packs"]["q"]["native"] + anchor-data["packs"]["r"]["native"]).cpu().numpy()
        norms = {k: v.flatten(1).norm(dim=1).cpu().numpy() for k, v in states.items()}
    return logits, norms


def evaluate(out, role):
    start = time.monotonic()
    base, checkpoint, old = parent()
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    cfg = r.read(out / "protocol.json")
    if r.read(out / "fit.json")["status"] != "FIT_COMPLETE" or cfg["source_state"] != old["source_state"]:
        raise ValueError("complete matching source fit required")
    models = {k: ShapeMetric().cuda().eval().requires_grad_(False) for k in KINDS}
    saved = torch.load(out / "models.pt", map_location="cuda", weights_only=True)
    for k, metric in models.items():
        metric.load_state_dict(saved[k])
    if role == "source_val":
        data, rows, skipped = prepare(base, "val", teaching="source")
        original = r.read(REFERENCE / "val_sealed.json")["rows"]
        metadata, original_prediction = dict(skipped=skipped), REFERENCE / "val_predictions.npz"
    else:
        data, rows, metadata = prepare_history(base)
        original_run = r.ROOT / "probe_outputs/motion_metric_east20_20261107_v2"
        original = r.read(original_run / "sealed.json")["rows"]
        original_prediction = original_run / "predictions.npz"
    if rows != original:
        raise ValueError("the same previously scored source or ended EAST20 windows required")
    logits = {k: [] for k in ARMS}
    norms = {k: [] for k in ARMS if k not in ("static", "scalar")}
    for lo in range(0, len(rows), 64):
        d = batch(data, np.arange(lo, min(lo+64, len(rows))))
        z, n = apply(base, models, d, cfg["scales"])
        for k, v in z.items():
            logits[k].append(v)
        for k, v in n.items():
            norms[k].append(v)
    logits = {k: np.concatenate(v) for k, v in logits.items()}
    norms = {k: np.concatenate(v) for k, v in norms.items()}
    with np.load(original_prediction, allow_pickle=False) as f:
        error = float(np.max(np.abs(logits["static"]-f["static"])))
        if error > 5e-6:
            raise ValueError("frozen native predictions changed")
    r.save_arrays(out / f"{role}_predictions.npz", **logits, **{f"norm_{k}": v for k, v in norms.items()})
    r.write_new(out / f"{role}_sealed.json", dict(status="SEALED_LOCAL_DIAGNOSTIC", rows=rows,
        original=str(original_prediction), static_error=error, **metadata))
    tau = math.log(checkpoint["threshold"]/(1-checkpoint["threshold"]))
    scored = []
    for i, row in enumerate(rows):
        flag, down = r.io23.read_outcome("J-TEXT" if role == "source_val" else "EAST", row["shot"])
        lead = round((down-row["query_time"])*1000, 9) if flag else None
        group = (teaching_group(flag, lead, row["source_horizon_ms"], "source") if role == "source_val"
                 else "safe" if not flag else "timely" if 30 < lead <= 125 else "earlier")
        if group != row["group"] or row["label"] != int(group == "timely"):
            raise ValueError("HDF5 stratum or original target changed")
        losses = {k: float(np.logaddexp(0., (1-2*row["label"])*(float(v[i])-tau))) for k, v in logits.items()}
        scored.append(dict(row, before_bce=losses["static"], after_bce=losses,
                           state_norm={k: float(v[i]) for k, v in norms.items()}))
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()):
        raise ValueError("source parameters changed during evaluation")
    table = aggregate(scored, arms=ARMS)
    passes = (all(table[g]["shape"] < -1e-6 for g in ("safe", "timely"))
              and table["earlier"]["shape"] <= 1e-6
              and all(table["balanced"]["shape"] < table["balanced"][k]-1e-6
                      for k in ARMS if k != "shape"))
    r.write_new(out / f"{role}_result.json", dict(status="WINDOW_NOT_STREAM", rows=scored,
        table=table, passes=passes, seconds=time.monotonic()-start))
    print(role, table, "passes", passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "source_val", "east20"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    (fit(args.out) if args.phase == "fit" else evaluate(args.out, args.phase))
