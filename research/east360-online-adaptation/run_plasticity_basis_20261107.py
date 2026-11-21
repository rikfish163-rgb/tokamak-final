"""Source-only teaching of the current first-FFN plasticity row space."""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import moving_peak_replay_20261105 as r
import motion_metric_model_20261107 as native
import plasticity_basis_model_20261107 as model
from motion_metric_kernel_20261107 import magnitude_descent, match_state
from shape_motion_kernel_20261107 import potential
from run_motion_metric_20261107 import GROUPS, STEPS, BATCH, STEP, aggregate, batch, prepare, risk_delta, teaching_group
from run_motion_metric_east20_20261107 import prepare_history


REFERENCE = r.ROOT / "probe_outputs/motion_metric_source_20261107_v3"
SCALES = r.ROOT / "probe_outputs/shape_motion_source_20261107_v1/protocol.json"
KINDS = ("risk", "feedback", "wrong_feedback")
BASES = ("svd", "random", *KINDS)
ARMS = ("static", *BASES, "time", "channel", "scalar")


def state(base, data, basis, scales, *, mode="real", create_graph=False):
    motion, mask = data["motion"], data["mask"]
    if mode in ("time", "channel"):
        motion = native.scramble(motion, mask, axis=mode)
    elif mode != "real":
        raise ValueError("declared causal feedback mode required")
    variable = basis.new_zeros(len(motion), 4, 64, requires_grad=True)
    prediction = model.forward_state(base, data["packs"]["0"], variable, basis)["forecast"]
    weights = prediction.new_ones(len(motion), 2, 23)
    loss = potential(prediction, motion, mask, weights, shape_scale=scales["shape_scale"])
    gradient, = torch.autograd.grad(loss.sum(), variable, create_graph=create_graph)
    return magnitude_descent(gradient, reference_norm=scales["level_reference"], step=STEP)


def objective(base, data, basis, scales, *, kind, tau):
    if kind not in KINDS:
        raise ValueError("declared plasticity teaching arm required")
    mode = "time" if kind == "wrong_feedback" else "real"
    updated = state(base, data, basis, scales, mode=mode, create_graph=True)
    if kind == "feedback":
        after = model.forward_state(base, data["packs"]["0"], updated, basis)["forecast"]
        with torch.no_grad():
            before = model.forward_state(base, data["packs"]["0"], torch.zeros_like(updated), basis)["forecast"]
        weights = after.new_ones(len(after), 2, 23)
        delta = (potential(after, data["motion"], data["mask"], weights, scales["shape_scale"])
                 - potential(before, data["motion"], data["mask"], weights, scales["shape_scale"]))
        loss = delta.mean()
    else:
        after = model.forward_state(base, data["packs"]["q"], updated, basis)["logit"]
        delta = risk_delta(data["packs"]["q"]["native"], after, data["labels"], tau) / data["scale"]
        other = state(base, data, basis, scales, mode="real" if mode == "time" else "time")
        other = match_state(other.detach(), updated.detach())
        with torch.no_grad():
            wrong = model.forward_state(base, data["packs"]["q"], other, basis)["logit"]
            wrong_delta = risk_delta(data["packs"]["q"]["native"], wrong, data["labels"], tau) / data["scale"]
        loss = delta.mean() + .25 * F.softplus(delta-wrong_delta).mean()
    return loss, dict(delta=float(delta.detach().mean()), state_norm_max=float(updated.detach().flatten(1).norm(dim=1).max()))


def parent():
    r.configure()
    base, checkpoint = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    old = r.read(REFERENCE / "protocol.json")
    if r.state_digest(base) != old["source_state"] or old["teaching"] != "source":
        raise ValueError("same qualified original-contract source required")
    return base, checkpoint, old


def fit(out):
    start = time.monotonic()
    base, checkpoint, old = parent()
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    data, rows, skipped = prepare(base, "train", teaching="source", encoder=model.encode_windows)
    if rows != old["fit_rows"]:
        raise ValueError("exact prior 1387 TRAIN episodes required")
    scales = r.read(SCALES)["scales"]
    initial = native.basis_for(base)
    bases = {k: torch.nn.Parameter(initial.clone()) for k in KINDS}
    optimizers = {k: torch.optim.Adam([p], lr=1e-3) for k, p in bases.items()}
    pools = {g: np.array([i for i, row in enumerate(rows) if row["group"] == g]) for g in GROUPS}
    rng = np.random.default_rng(131)
    tau = math.log(checkpoint["threshold"]/(1-checkpoint["threshold"]))
    out.mkdir(parents=True, exist_ok=False)
    r.write_new(out / "protocol.json", dict(status="CURRENT_FFN_PLASTICITY_TEACHING", kinds=KINDS,
        source_state=old["source_state"], source=str(r.DEFAULT_PARENT), threshold=checkpoint["threshold"],
        scales=scales, steps=STEPS, batch=BATCH, rows=rows, skipped=skipped,
        source_files={str(p): p.read_text() for p in (Path(__file__), Path(model.__file__), Path(native.__file__),
            r.ROOT / "run_motion_metric_20261107.py", r.ROOT / "run_motion_metric_east20_20261107.py",
            r.ROOT / "shape_motion_kernel_20261107.py", r.ROOT / "motion_metric_kernel_20261107.py",
            r.ROOT / "experiments/risk_aligned_online/lora_stream_20261102/core.py",
            r.ROOT / "PLASTICITY_BASIS_PROTOCOL_20261107.md")}))
    history = []
    for step in range(1, STEPS+1):
        indices = np.concatenate([rng.choice(pools[g], BATCH//3, replace=True) for g in GROUPS])
        d = batch(data, indices)
        for kind, basis in bases.items():
            optimizer = optimizers[kind]
            optimizer.zero_grad(set_to_none=True)
            loss, info = objective(base, d, basis, scales, kind=kind, tau=tau)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite plasticity objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([basis], 1., error_if_nonfinite=True)
            optimizer.step()
            model.project_rows(basis)
            history.append(dict(step=step, kind=kind, loss=float(loss.detach()), gradient=float(norm), **info))
        if step % 64 == 0:
            print("step", step, history[-3:], "seconds", round(time.monotonic()-start, 1), flush=True)
    fixed = dict(svd=initial, random=model.random_basis(initial))
    saved = {**fixed, **{k: p.detach() for k, p in bases.items()}}
    orthogonal_error = {k: float((p@p.T-torch.eye(4, device=p.device)).abs().max()) for k, p in saved.items()}
    if max(orthogonal_error.values()) > 2e-6:
        raise ValueError("plasticity row norms changed")
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()) or any(p.grad is not None for p in base.parameters()):
        raise ValueError("source changed during plasticity training")
    torch.save({k: p.cpu() for k, p in saved.items()}, out / "bases.pt")
    angles = {k: torch.linalg.svdvals(p.double()@initial.double().T).clamp(0, 1).acos().rad2deg().cpu().tolist()
              for k, p in saved.items()}
    r.write_new(out / "fit.json", dict(status="FIT_COMPLETE", seconds=time.monotonic()-start,
        orthogonal_error=orthogonal_error, principal_angles_degrees=angles, history=history))
    print("fit complete", len(rows), "angles", angles, flush=True)


def apply(base, bases, data, scales):
    weights = {k: state(base, data, p, scales).detach() for k, p in bases.items()}
    for mode in ("time", "channel"):
        wrong = state(base, data, bases["risk"], scales, mode=mode)
        weights[mode] = match_state(wrong.detach(), weights["risk"])
    with torch.no_grad():
        z = {k: model.forward_state(base, data["packs"]["q"], w,
                 bases[k] if k in BASES else bases["risk"])["logit"].cpu().numpy() for k, w in weights.items()}
        z["static"] = data["packs"]["q"]["native"].cpu().numpy()
        anchor = model.forward_state(base, data["packs"]["r"], weights["risk"], bases["risk"])["logit"]
        z["scalar"] = (data["packs"]["q"]["native"]+anchor-data["packs"]["r"]["native"]).cpu().numpy()
        norms = {k: v.flatten(1).norm(dim=1).cpu().numpy() for k, v in weights.items()}
    return z, norms


def load_bases(out, base):
    cfg = r.read(out / "protocol.json")
    if r.read(out / "fit.json")["status"] != "FIT_COMPLETE" or cfg["source_state"] != r.state_digest(base):
        raise ValueError("complete same-source plasticity fit required")
    bases = torch.load(out / "bases.pt", map_location=base.center.device, weights_only=True)
    for p in bases.values():
        p.requires_grad_(False)
    return bases, cfg


def evaluate(out, role):
    start = time.monotonic()
    base, checkpoint, old = parent()
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    bases, cfg = load_bases(out, base)
    if role == "source_val":
        data, rows, skipped = prepare(base, "val", teaching="source", encoder=model.encode_windows)
        original = r.read(REFERENCE / "val_sealed.json")["rows"]
        metadata, original_prediction = dict(skipped=skipped), REFERENCE / "val_predictions.npz"
    else:
        data, rows, metadata = prepare_history(base, encoder=model.encode_windows)
        original_run = r.ROOT / "probe_outputs/motion_metric_east20_20261107_v2"
        original, original_prediction = r.read(original_run / "sealed.json")["rows"], original_run / "predictions.npz"
    if rows != original:
        raise ValueError("same previously specified windows required")
    logits, norms = {k: [] for k in ARMS}, {k: [] for k in ARMS if k not in ("static", "scalar")}
    for lo in range(0, len(rows), 64):
        d = batch(data, np.arange(lo, min(lo+64, len(rows))))
        z, n = apply(base, bases, d, cfg["scales"])
        for k, v in z.items():
            logits[k].append(v)
        for k, v in n.items():
            norms[k].append(v)
    logits = {k: np.concatenate(v) for k, v in logits.items()}
    norms = {k: np.concatenate(v) for k, v in norms.items()}
    with np.load(original_prediction, allow_pickle=False) as f:
        error = float(np.max(np.abs(logits["static"]-f["static"])))
        if error > 5e-6:
            raise ValueError("source prediction changed")
    r.save_arrays(out / f"{role}_predictions.npz", **logits, **{f"norm_{k}": v for k, v in norms.items()})
    r.write_new(out / f"{role}_sealed.json", dict(status="SEALED_PLASTICITY_WINDOWS", rows=rows,
        original=str(original_prediction), static_error=error, **metadata))
    tau = math.log(checkpoint["threshold"]/(1-checkpoint["threshold"]))
    scored = []
    for i, row in enumerate(rows):
        flag, down = r.io23.read_outcome("J-TEXT" if role == "source_val" else "EAST", row["shot"])
        lead = round((down-row["query_time"])*1000, 9) if flag else None
        group = (teaching_group(flag, lead, row["source_horizon_ms"], "source") if role == "source_val"
                 else "safe" if not flag else "timely" if 30 < lead <= 125 else "earlier")
        if group != row["group"] or row["label"] != int(group == "timely"):
            raise ValueError("HDF5 label contract changed")
        losses = {k: float(np.logaddexp(0., (1-2*row["label"])*(float(v[i])-tau))) for k, v in logits.items()}
        scored.append(dict(row, before_bce=losses["static"], after_bce=losses,
                           state_norm={k: float(v[i]) for k, v in norms.items()}))
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()):
        raise ValueError("source changed during evaluation")
    table = aggregate(scored, arms=ARMS)
    screen = (all(table[g]["risk"] < -1e-6 for g in ("safe", "timely"))
              and table["earlier"]["risk"] <= 1e-6
              and all(table["balanced"]["risk"] < table["balanced"][k]-1e-6 for k in ARMS if k != "risk"))
    r.write_new(out / f"{role}_result.json", dict(status="WINDOW_NOT_STREAM", rows=scored,
        table=table, window_screen=screen, seconds=time.monotonic()-start))
    print(role, table, "window_screen", screen, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "source_val", "east20"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    (fit(args.out) if args.phase == "fit" else evaluate(args.out, args.phase))
