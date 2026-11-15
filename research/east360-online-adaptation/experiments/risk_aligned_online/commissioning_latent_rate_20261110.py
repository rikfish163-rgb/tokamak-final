"""Test a direct latent trajectory model and deadline-aware risk on the fixed EAST stream."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import commissioning_latent_dynamics_20261110 as dynamics
from . import io23, simple_data
from .commissioning_calibration_20261108 import held_safe_cutoffs, prefix_mapping
from .commissioning_intra_20261110 import (
    CALIBRATION, CAPS, PARENT_ARM, PARENT_RUN, STARTUP, load_frozen,
)
from .commissioning_transfer_20261108 import contrast, shot_weights
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .latent_features import extract_shot
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


STEPS = 256
TRAIN_VIEWS = ("static", "real")
HEADS = ("peak", "deadline")
VARIANTS = {
    "peak_static": ("peak", "static"),
    "peak_real": ("peak", "real"),
    "deadline_static": ("deadline", "static"),
    "deadline_real": ("deadline", "real"),
    "deadline_reverse": ("deadline", "reverse"),
}


def score_warning(logits, timely_groups, safe_groups, boundary, *, deadline):
    if not timely_groups or not safe_groups:
        raise ValueError("timely disruptive and safe groups are required")
    positive = []
    for g in timely_groups:
        z = logits[:, g]
        peak = z.cummax(dim=-1).values if deadline else z.amax(dim=-1)
        positive.append(F.softplus(boundary - peak).mean())
    safe = [F.softplus(logits[:, g].amax(dim=-1) - boundary).mean() for g in safe_groups]
    return .5 * (torch.stack(positive).mean() + torch.stack(safe).mean())


def supervised_rows(features, times, flag, down):
    y, valid = io23.labels_at(times, flag, down, 125.)
    timely = (np.round((down - times) * 1000., 9) > 30. if flag
              else np.zeros(len(times), bool))
    return (features[:, valid], y[valid].astype(bool), (y.astype(bool) & timely)[valid],
            np.full(int(valid.sum()), not flag, bool))


def initialise_readout(source, size):
    """One nonlinear risk readout, initially equal to the original source risk."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        readout = nn.Sequential(nn.Linear(size, 16), nn.SiLU(), nn.Linear(16, 1))
    with torch.no_grad():
        first, last = readout[0], readout[2]
        first.weight[:2].zero_()
        w = source.weight.detach().cpu()[0]
        b = source.bias.detach().cpu()[0]
        first.weight[0, :len(w)] = w
        first.weight[1, :len(w)] = -w
        first.bias[:2] = torch.stack((b, -b))
        last.weight.zero_()
        last.weight[0, :2] = torch.tensor([1., -1.])
        last.bias.zero_()
    return readout


def encoded(base, signal, device):
    h, native = extract_shot(base, signal, device)
    m = contrast(signal["values"], signal["available"])
    return h, signal["available"].any(axis=1), m, native


def views(record, prior, modes):
    h, observed, motion, _ = record
    rows, stats = {}, {}
    for k in modes:
        drift, stats[k] = dynamics.causal_drift(h, observed, prior, mode=k)
        rows[k] = np.concatenate((h, motion, drift), axis=1)
    return rows, stats


def fit_heads(base, rows, original, device):
    ordered_x, labels, groups, timely, safe = [], [], [], [], []
    offset = 0
    for group, (x, y, action, normal) in enumerate(rows):
        ordered_x.append(x)
        labels.append(y)
        groups.append(np.full(len(y), group))
        for included, target in ((action, timely), (normal, safe)):
            idx = np.flatnonzero(included)
            if len(idx):
                target.append(slice(offset + int(idx[0]), offset + int(idx[-1]) + 1))
        offset += len(y)
    y = np.concatenate(labels)
    x = torch.from_numpy(np.concatenate(ordered_x, axis=1)).to(device)
    w = torch.as_tensor(shot_weights(np.concatenate(groups), y), dtype=x.dtype, device=device)
    target = torch.as_tensor(y, dtype=x.dtype, device=device).expand(len(TRAIN_VIEWS), -1)
    models, audit = {}, {}
    boundary = float(np.log(original / (1. - original)))
    for mode in HEADS:
        model = initialise_readout(base.risk_head, x.shape[-1]).to(device)
        starting = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.Adam(model.parameters(), lr=.008)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            z = model(x).squeeze(-1)
            point = (F.binary_cross_entropy_with_logits(z, target, reduction="none") @ w).mean()
            warning = score_warning(z, timely, safe, boundary, deadline=mode == "deadline")
            anchor = sum((p - q).square().sum() for p, q in zip(model.parameters(), starting, strict=True))
            (point + warning + .01 * anchor).backward()
            opt.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite commissioning risk readout")
        models[mode] = model.eval().requires_grad_(False)
        audit[mode] = dict(point=float(point.detach()), warning=float(warning.detach()), steps=STEPS,
                           parameters=sum(p.numel() for p in model.parameters()))
    return models, audit


def fit(base, raws, outcomes, ids, mapping, original):
    records = {s: encoded(base, prepare_actual(raws[s], STARTUP + 1, mapping), "cuda") for s in ids}
    prior = dynamics.fit_prior([records[s][:2] for s in ids])
    rows = []
    for sid in ids:
        x, _ = views(records[sid], prior, TRAIN_VIEWS)
        rows.append(supervised_rows(np.stack([x[k] for k in TRAIN_VIEWS]), raws[sid]["times"], *outcomes[sid]))
    models, audit = fit_heads(base, rows, original, "cuda")
    return prior, models, audit


@torch.no_grad()
def predict_views(rows, models, variants):
    matrices = {k: torch.from_numpy(v).cuda() for k, v in rows.items()}
    return {name: models[head](matrices[view]).flatten().sigmoid().cpu().numpy()
            for name, (head, view) in variants.items()}


def calibrate(base, raws, outcomes, source_stats, original, out, started):
    folds = []
    for sid in (s for s in raws if not outcomes[s][0]):
        train = [s for s in raws if s != sid]
        mapping = prefix_mapping(source_stats, [raws[s] for s in train])
        prior, models, _ = fit(base, raws, outcomes, train, mapping, original)
        held = encoded(base, prepare_actual(raws[sid], STARTUP + 1, mapping), "cuda")
        x, _ = views(held, prior, ("static",))
        maxima = predict_views(x, models, {h: (h, "static") for h in HEADS})
        folds.append(dict(held_safe=sid, fit_ids=train, maxima={k: float(v.max()) for k, v in maxima.items()}))
        if len(folds) % 3 == 0:
            print(dict(stage="calibration", held_safe=len(folds), seconds=time.monotonic()-started), flush=True)
    thresholds = held_safe_cutoffs([f["maxima"]["peak"] for f in folds])
    write(out / "calibration.json", dict(scope="COLD20_LEAVE_SAFE_SHOT_OUT", folds=folds,
                                        cutoff_source="peak_static", thresholds=thresholds))
    return thresholds


def read_parent_curve(name, parent_seals):
    p = PARENT_RUN / "shots" / name
    if sha(p) != parent_seals[name]:
        raise ValueError("cold-start comparison no longer matches its seal")
    with np.load(p, allow_pickle=False) as f:
        return f["times"], f[PARENT_ARM]


def actual_prefix(base, signal, prior, models, curves, thresholds):
    n = min(len(signal["values"]), 211)
    x = {k: signal[k][:n] for k in ("values", "available")}
    record = encoded(base, x, "cuda")
    rows, _ = views(record, prior, ("static", "real", "reverse"))
    prefix = predict_views(rows, models, VARIANTS)
    deltas = {k: float(np.max(np.abs(curves[k][:n] - v))) for k, v in prefix.items()}
    for k, v in prefix.items():
        np.testing.assert_allclose(curves[k][:n], v, rtol=0, atol=1e-5)
        for threshold in thresholds.values():
            np.testing.assert_array_equal(curves[k][:n] >= threshold, v >= threshold)
    return dict(points=n, max_difference=deltas)


def run(out):
    started = time.monotonic()
    configure()
    inputs, coef, original, parent_cutoffs, bound = load_frozen()
    base, ckpt = load_parent(DEFAULT_PARENT)
    base.cuda().requires_grad_(False)
    source_state = state_digest(base)
    if ckpt["threshold"] != original or source_state != inputs["baseline_protocol"]["source_state_sha256"]:
        raise ValueError("qualified source contract changed")
    ids = inputs["ids"]
    names = ("parent", *VARIANTS)
    parent_seals = read(PARENT_RUN / "result.json")["curve_sha256"]
    out.mkdir(parents=True)
    (out / "shots").mkdir()
    write(out / "protocol.json", dict(
        scope="EAST360_DEVELOPMENT_COLD20_THEN_INTRA_ONLY", source_sha256=PARENT_SHA256, **bound,
        order=ids, fit_shots=ids[:STARTUP], original=original, parent_thresholds=parent_cutoffs,
        variants=VARIANTS, fit_views=TRAIN_VIEWS, steps=STEPS, event_window_ms=125., deadline_ms=30.,
        first_origin=dynamics.START, lag=dynamics.LAG, forecast_points=dynamics.HORIZON, stride=dynamics.STRIDE,
        earliest_update_effect=dynamics.START+dynamics.HORIZON+1, persistent_updates=0,
        new_head_cutoffs_shared=True, cutoff_source="peak_static_leave_safe_shot_out",
        core_sha256=sha(Path(dynamics.__file__)), producer_sha256=sha(Path(__file__)),
    ))
    raws, outcomes = {}, {}
    for i, sid in enumerate(ids[:STARTUP], 1):
        path = io23.path_for("EAST", sid)
        if sha(path) != inputs["hashes"][str(sid)]:
            raise ValueError(f"startup waveform changed: {sid}")
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        times, _ = read_parent_curve(f"{i:04d}_{sid}.npz", parent_seals)
        np.testing.assert_array_equal(times, raw["times"])
        raws[sid] = raw
        outcomes[sid] = io23.read_outcome("EAST", sid)
    if sum(d for d, _ in outcomes.values()) != 8:
        raise ValueError("expected exactly 8 disruptive outcomes in ended20")
    thresholds = calibrate(base, raws, outcomes, inputs["mapping"]["source_statistics"], original, out, started)
    prior, models, training = fit(base, raws, outcomes, ids[:STARTUP], inputs["mapping"], original)
    with (out / "models.npz").open("xb") as f:
        np.savez(f, gain=prior[0], precision=prior[1], **{
            f"{head}_{key}": v.cpu().numpy() for head, m in models.items() for key, v in m.state_dict().items()
        })
    training_sha = sha(out / "models.npz")
    cal_sha = sha(out / "calibration.json")
    states = {h: state_digest(m) for h, m in models.items()}
    print(dict(stage="fit20_complete", seconds=time.monotonic()-started), flush=True)
    seals, audits, witnesses = {}, {}, {}
    with patch.object(io23, "read_outcome", side_effect=RuntimeError("outcome before prediction seal")):
        for i, sid in enumerate(ids, 1):
            name = f"{i:04d}_{sid}.npz"
            times, parent = read_parent_curve(name, parent_seals)
            if i <= STARTUP:
                curves = dict.fromkeys(VARIANTS, parent)
            else:
                path = io23.path_for("EAST", sid)
                if sha(path) != inputs["hashes"][str(sid)]:
                    raise ValueError(f"EAST waveform changed: {sid}")
                raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
                np.testing.assert_array_equal(times, raw["times"])
                signal = prepare_actual(raw, i, inputs["mapping"])
                record = encoded(base, signal, "cuda")
                matrix = torch.from_numpy(np.concatenate((record[0], record[2]), axis=1)).cuda()
                previous = (matrix @ coef[:-1] + coef[-1]).sigmoid().cpu().numpy()
                np.testing.assert_array_equal(parent, previous)
                x, stats = views(record, prior, ("static", "real", "reverse"))
                curves = predict_views(x, models, VARIANTS)
                if not all(np.isfinite(v).all() for v in curves.values()):
                    raise FloatingPointError(f"nonfinite predicted risk: {sid}")
                first = dynamics.START + dynamics.HORIZON + 1
                for name_alt, (h, _) in VARIANTS.items():
                    np.testing.assert_array_equal(curves[f"{h}_static"][:first], curves[name_alt][:first])
                audits[str(sid)] = stats
                if i in (STARTUP + 1, 120, 360):
                    witnesses[str(sid)] = actual_prefix(base, signal, prior, models, curves, thresholds)
            path = out / "shots" / name
            with path.open("xb") as f:
                np.savez_compressed(f, times=times, parent=parent, **curves)
                f.flush()
                os.fsync(f.fileno())
            seals[name] = sha(path)
            if i % 20 == 0:
                print(dict(stage="sealed", shots=i, seconds=time.monotonic()-started), flush=True)
    if state_digest(base) != source_state or states != {h: state_digest(m) for h, m in models.items()}:
        raise AssertionError("persistent parameters changed within the held suffix")
    write(out / "seal.json", dict(curve_sha256=seals, model_sha256=training_sha,
                                  calibration_sha256=cal_sha, prefix_witnesses=witnesses))

    per = {cap: {name: [] for name in names} for cap in CAPS}
    for i, sid in enumerate(ids, 1):
        path = out / "shots" / f"{i:04d}_{sid}.npz"
        if sha(path) != seals[path.name]:
            raise ValueError("post-seal prediction drift")
        flag, down = outcomes[sid] if i <= STARTUP else io23.read_outcome("EAST", sid)
        with np.load(path, allow_pickle=False) as trace:
            for cap, by_mode in per.items():
                for k, result in by_mode.items():
                    t = original if i <= STARTUP else (parent_cutoffs[cap] if k == "parent"
                                                       else thresholds[cap])
                    result.append(dict(shot=sid, disruptive=flag, down_s=down,
                                       **score_curve(trace["times"], trace[k], flag, down, t)))
    parent_reference = read(CALIBRATION / "per_shot.json")
    if any(by_mode["parent"] != parent_reference[c][PARENT_ARM] for c, by_mode in per.items()):
        raise AssertionError("previous calibrated full-stream comparison changed")
    if cal_sha != sha(out / "calibration.json") or training_sha != sha(out / "models.npz"):
        raise AssertionError("training artifact changed after sealing")
    metrics = {c: {k: timing_decomposition(v) for k, v in a.items()} for c, a in per.items()}
    paired = {c: {k: flips(a[f"{h}_static"], a[k]) for k, (h, v) in VARIANTS.items() if v != "static"}
              for c, a in per.items()}
    totals = {v: {k: sum(m[v][k] for m in audits.values()) for k in next(iter(audits.values()))[v]}
              for v in ("static", "real", "reverse")}
    write(out / "per_shot.json", per)
    write(out / "dynamics_by_shot.json", audits)
    write(out / "result.json", dict(status="COMPLETED_LATENT_RATE360_DEVELOPMENT", metrics=metrics,
                                     same_head_paired=paired, training=training, dynamics=totals,
                                     seconds=time.monotonic()-started))
    print({c: {k: (v["tp"], v["fp"]) for k, v in a.items()} for c, a in metrics.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output.resolve())
