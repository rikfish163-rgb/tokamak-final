"""Same-budget update-rule teaching with frozen, already-observed neural states."""

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
from latent_motion_context_20261107 import assemble
from motion_metric_kernel_20261107 import MotionMetric, match_state
from run_motion_metric_20261107 import GROUPS, STEPS, BATCH, STEP, aggregate, batch, prepare, risk_delta, state, teaching_group
from run_motion_metric_east20_20261107 import prepare_history


REFERENCE = r.ROOT / "probe_outputs/motion_metric_source_20261107_v3"
KINDS = ("summary", "level", "ordered", "global")
CONTROLS = ("time", "channel", "swap", "stale")
ARMS = ("static", "fixed", *KINDS, *CONTROLS, "scalar")


def context(data, kind):
    p = data["packs"]
    return assemble(data["context"], p["r"]["risk_hidden"], p["p"]["risk_hidden"],
                    p["pp"]["risk_hidden"], kind)


def parent():
    r.configure()
    base, checkpoint = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    old = r.read(REFERENCE / "protocol.json")
    if old["teaching"] != "source" or old["geometry"] != "magnitude" or r.state_digest(base) != old["source_state"]:
        raise ValueError("same frozen original-contract V3 parent required")
    return base, checkpoint, old


def fit(out):
    start = time.monotonic()
    base, checkpoint, old = parent()
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    data, rows, skipped = prepare(base, "train", teaching="source", include_past=True)
    if rows != old["fit_rows"]:
        raise ValueError("exactly the prior V3 source training windows required")
    scale = old["update_scale"]
    prototype = MotionMetric(context_dim=261).cuda()
    models = {k: deepcopy(prototype) for k in KINDS}
    optimizers = {k: torch.optim.Adam(v.parameters(), lr=1e-3) for k, v in models.items()}
    pools = {g: np.array([i for i, row in enumerate(rows) if row["group"] == g]) for g in GROUPS}
    rng = np.random.default_rng(131)
    tau = math.log(checkpoint["threshold"]/(1-checkpoint["threshold"]))
    out.mkdir(parents=True, exist_ok=False)
    r.write_new(out / "protocol.json", dict(status="FROZEN_LATENT_CONTEXT_TEACHING", kinds=KINDS,
        source_state=old["source_state"], source=str(r.DEFAULT_PARENT), threshold=r.THRESHOLD,
        update_scale=scale, steps=STEPS, batch=BATCH, rows=rows, skipped=skipped,
        source_files={str(p): p.read_text() for p in (Path(__file__), Path(model.__file__),
            r.ROOT / "latent_motion_context_20261107.py", r.ROOT / "run_motion_metric_20261107.py",
            r.ROOT / "motion_metric_kernel_20261107.py", r.ROOT / "run_motion_metric_east20_20261107.py",
            r.ROOT / "LATENT_MOTION_PROTOCOL_20261107.md")}))
    history = []
    for step in range(1, STEPS+1):
        indices = np.concatenate([rng.choice(pools[g], BATCH//3, replace=True) for g in GROUPS])
        d = batch(data, indices)
        before = d["packs"]["q"]["native"]
        for kind, metric in models.items():
            optimizer = optimizers[kind]
            optimizer.zero_grad(set_to_none=True)
            c = context(d, kind)
            updated, weights = state(base, d["packs"]["0"], c, d["motion"], d["mask"],
                                     metric, create_graph=True, update_scale=scale)
            after = model.forward_state(base, d["packs"]["q"], updated)["logit"]
            change = risk_delta(before, after, d["labels"], tau) / d["scale"]
            other, _ = state(base, d["packs"]["0"], c, d["motion"], d["mask"], metric,
                             mode="time", update_scale=scale)
            other = match_state(other.detach(), updated.detach())
            with torch.no_grad():
                wrong = model.forward_state(base, d["packs"]["q"], other)["logit"]
                wrong_change = risk_delta(before, wrong, d["labels"], tau) / d["scale"]
            loss = change.mean() + .25 * F.softplus(change-wrong_change).mean() + .001 * (weights-1).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite latent-context teaching loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(metric.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=step, kind=kind, loss=float(loss.detach()),
                                risk_delta=float(change.detach().mean()), gradient=float(norm)))
        if step % 64 == 0:
            print("step", step, history[-4:], "seconds", round(time.monotonic()-start, 1), flush=True)
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()) or any(p.grad is not None for p in base.parameters()):
        raise ValueError("source was changed during rule learning")
    torch.save({k: m.state_dict() for k, m in models.items()}, out / "models.pt")
    r.write_new(out / "fit.json", dict(status="FIT_COMPLETE", seconds=time.monotonic()-start, history=history))
    print("fit complete", len(rows), flush=True)


def apply(base, models, data, scale):
    states = {}
    for kind, metric in models.items():
        w, _ = state(base, data["packs"]["0"], context(data, kind), data["motion"], data["mask"],
                     metric, update_scale=scale)
        states[kind] = w.detach()
    w, _ = state(base, data["packs"]["0"], context(data, "ordered"), data["motion"], data["mask"],
                 None, update_scale=scale)
    states["fixed"] = w.detach()
    for mode in CONTROLS:
        c = context(data, mode if mode in ("swap", "stale") else "ordered")
        w, _ = state(base, data["packs"]["0"], c, data["motion"], data["mask"], models["ordered"],
                     mode=mode if mode in ("time", "channel") else "real", update_scale=scale)
        states[mode] = match_state(w.detach(), states["ordered"])
    with torch.no_grad():
        z = {k: model.forward_state(base, data["packs"]["q"], w)["logit"].cpu().numpy() for k, w in states.items()}
        z["static"] = data["packs"]["q"]["native"].cpu().numpy()
        anchor = model.forward_state(base, data["packs"]["r"], states["ordered"])["logit"]
        z["scalar"] = (data["packs"]["q"]["native"] + anchor-data["packs"]["r"]["native"]).cpu().numpy()
        norms = {k: v.flatten(1).norm(dim=1).cpu().numpy() for k, v in states.items()}
    return z, norms


def evaluate(out, role):
    start = time.monotonic()
    base, checkpoint, old = parent()
    cfg = r.read(out / "protocol.json")
    if r.read(out / "fit.json")["status"] != "FIT_COMPLETE" or cfg["source_state"] != old["source_state"]:
        raise ValueError("complete matching frozen source fit required")
    saved = torch.load(out / "models.pt", map_location="cuda", weights_only=True)
    models = {k: MotionMetric(context_dim=261).cuda().eval() for k in KINDS}
    for k, metric in models.items():
        metric.load_state_dict(saved[k])
        metric.requires_grad_(False)
    if role == "source_val":
        data, rows, skipped = prepare(base, "val", teaching="source", include_past=True)
        original = r.read(REFERENCE / "val_sealed.json")["rows"]
        if rows != original:
            raise ValueError("same original-contract source VAL windows required")
        metadata = dict(skipped=skipped)
        original_prediction = REFERENCE / "val_predictions.npz"
    else:
        data, rows, metadata = prepare_history(base, include_past=True)
        original_run = r.ROOT / "probe_outputs/motion_metric_east20_20261107_v2"
        if rows != r.read(original_run / "sealed.json")["rows"]:
            raise ValueError("same already-revealed EAST20 windows required")
        original_prediction = original_run / "predictions.npz"
    logits, norms = {k: [] for k in ARMS}, {k: [] for k in ARMS if k not in ("static", "scalar")}
    for lo in range(0, len(rows), 64):
        d = batch(data, np.arange(lo, min(lo+64, len(rows))))
        z, n = apply(base, models, d, cfg["update_scale"])
        for k, v in z.items():
            logits[k].append(v)
        for k, v in n.items():
            norms[k].append(v)
    logits = {k: np.concatenate(v) for k, v in logits.items()}
    norms = {k: np.concatenate(v) for k, v in norms.items()}
    with np.load(original_prediction, allow_pickle=False) as f:
        error = float(np.max(np.abs(logits["static"]-f["static"])))
        if error > 5e-6:
            raise ValueError(f"source prediction changed: {error:g}")
    r.save_arrays(out / f"{role}_predictions.npz", **logits, **{f"norm_{k}": v for k, v in norms.items()})
    r.write_new(out / f"{role}_sealed.json", dict(status="SEALED_LOCAL_DIAGNOSTIC", rows=rows,
        original=str(original_prediction), static_error=error, **metadata))
    tau = math.log(checkpoint["threshold"]/(1-checkpoint["threshold"]))
    scored = []
    for i, row in enumerate(rows):
        device = "J-TEXT" if role == "source_val" else "EAST"
        flag, down = r.io23.read_outcome(device, row["shot"])
        lead = round((down-row["query_time"])*1000, 9) if flag else None
        if role == "source_val":
            group = teaching_group(flag, lead, row["source_horizon_ms"], "source")
        else:
            group = "safe" if not flag else "timely" if 30 < lead <= 125 else "earlier"
        if group != row["group"] or (flag and lead <= (10 if role == "source_val" else 30)):
            raise ValueError("HDF5 teaching or evaluation stratum changed")
        losses = {k: float(np.logaddexp(0., (1-2*row["label"])*(float(v[i])-tau))) for k, v in logits.items()}
        scored.append(dict(row, before_bce=losses["static"], after_bce=losses,
                           state_norm={k: float(v[i]) for k, v in norms.items()}))
    table = aggregate(scored, arms=ARMS)
    passes = (all(table[g]["ordered"] < -1e-6 for g in ("safe", "timely"))
              and table["earlier"]["ordered"] <= 1e-6
              and all(table["balanced"]["ordered"] < table["balanced"][k]-1e-6
                      for k in ARMS if k != "ordered"))
    r.write_new(out / f"{role}_result.json", dict(status="WINDOW_NOT_STREAM", rows=scored,
        table=table, passes=passes, seconds=time.monotonic()-start))
    print(role, table, "passes", passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "source_val", "east20"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    (fit(args.out) if args.phase == "fit" else evaluate(args.out, args.phase))
