"""Cold-start EAST readout transfer with causal motion and supervised CORAL.

This is an initialization experiment. It has no adaptation within a shot and
no further learning after the first 20 ended shots. CORAL recolors labeled
J-TEXT training examples only, never the inputs of a live EAST prediction.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import sha, write
from .latent_features import extract_shot
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import (
    check_reference, configure, load_inputs, prepare_actual,
)
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import (
    DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train,
)
from .source_anticipation_20261104.static_stream import timing_decomposition


STARTUP = 20
STEPS = 256
SOURCE_STRIDE = 4
ARMS = {
    "level_point": (False, "none", False),
    "level_alarm": (False, "none", True),
    "motion_alarm": (True, "none", True),
    "motion_marginal": (True, "marginal", True),
    "motion_conditional": (True, "conditional", True),
    "motion_target_only": (True, "target_only", True),
}


@dataclass(frozen=True)
class Rows:
    x: np.ndarray
    shot: np.ndarray
    y: np.ndarray
    timely: np.ndarray
    safe: np.ndarray


def contrast(values, available, *, lag=32, width=32, minimum_pairs=16):
    """Signed coherence of recent paired differences, invariant to gain/offset.

    At endpoint t, only differences x[u]-x[u-lag], u <= t, are used.
    Fewer than minimum_pairs observed pairs means no motion evidence; the
    original encoder still observes the values and masks at that endpoint.
    """
    x = np.asarray(values, np.float64)
    a = np.asarray(available, bool)
    delta = np.zeros_like(x)
    paired = np.zeros_like(a)
    paired[lag:] = a[lag:] & a[:-lag]
    np.subtract(x[lag:], x[:-lag], out=delta[lag:], where=paired[lag:])
    q = np.arange(1, len(x) + 1)

    def recent(v):
        cumulative = np.pad(np.cumsum(v, axis=0), ((1, 0), (0, 0)))
        return cumulative[q] - cumulative[np.maximum(q - width, 0)]

    count, net, energy = recent(paired), recent(delta), recent(delta ** 2)
    result = np.zeros_like(net)
    np.divide(net, np.sqrt(np.maximum(count * energy, 0.)), out=result,
              where=(count >= minimum_pairs) & (energy > 0.))
    return np.clip(result, -1., 1.).astype(np.float32)


def make_rows(x, times, disruptive, down, horizon, deadline, group, *, stride=1):
    label, valid = io23.labels_at(times, disruptive, down, horizon)
    timely = (np.round((down - times) * 1000., 9) > deadline
              if disruptive else np.zeros(len(times), bool))
    take = np.flatnonzero(valid)[::stride]
    if not len(take):
        raise ValueError("no valid pre-terminal training endpoints")
    y = label[take].astype(bool)
    return Rows(x[take], np.full(len(take), group, np.int64), y, timely[take] & y,
                np.full(len(take), not disruptive, bool))


def combine(shots):
    return Rows(*(np.concatenate([getattr(s, k) for s in shots])
                  for k in ("x", "shot", "y", "timely", "safe")))


def shot_weights(shot, y):
    """Equal mass per label, then per participating shot within that label."""
    w = np.zeros(len(shot), np.float64)
    for label in (False, True):
        m = y == label
        _, inverse, counts = np.unique(shot[m], return_inverse=True, return_counts=True)
        if not len(counts):
            raise ValueError("both training endpoint labels are needed")
        w[m] = .5 / (len(counts) * counts[inverse])
    return w


def _moments(x, w):
    w = w / w.sum()
    center = w @ x
    centered = x - center
    return center, (centered.T * w) @ centered


def _root(covariance, ridge, exponent):
    s, u = np.linalg.eigh(covariance)
    return (u * (np.maximum(s, 0.) + ridge) ** exponent) @ u.T


def transport(source, target, *, conditional):
    """Regularized source -> target affine CORAL, using only training rows."""
    out = np.empty(source.x.shape, dtype=np.float32)
    ws, wt = shot_weights(source.shot, source.y), shot_weights(target.shot, target.y)
    for label in ((False, True) if conditional else (None,)):
        sm = np.ones(len(ws), bool) if label is None else source.y == label
        tm = np.ones(len(wt), bool) if label is None else target.y == label
        sx, tx = source.x[sm].astype(np.float64), target.x[tm].astype(np.float64)
        ms, cs = _moments(sx, ws[sm])
        mt, ct = _moments(tx, wt[tm])
        ridge = .05 * (np.trace(cs) + np.trace(ct)) / (2 * len(ms))
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("CORAL needs nondegenerate training features")
        out[sm] = (sx - ms) @ _root(cs, ridge, -.5) @ _root(ct, ridge, .5) + mt
    return out


def alarm_loss(z, timely_groups, safe_groups, boundary):
    if not timely_groups or not safe_groups:
        raise ValueError("timely positive and safe shot groups required")
    timely = torch.stack([z[g].amax() for g in timely_groups])
    safe = torch.stack([z[g].amax() for g in safe_groups])
    return .5 * (F.softplus(boundary - timely).mean() + F.softplus(safe - boundary).mean())


def features(base, shot, device):
    hidden, logits = extract_shot(base, shot, device)
    return np.concatenate((hidden, contrast(shot["values"], shot["available"])), axis=1), logits


def fit_readouts(base, source, target, threshold, device):
    t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    origin = torch.cat((base.risk_head.weight.detach().flatten(),
                        torch.zeros(23, device=device), base.risk_head.bias.detach()))
    dim = base.cfg.dim
    target_w = t(shot_weights(target.shot, target.y))
    source_w = t(shot_weights(source.shot, source.y))
    sy, ty = t(source.y), t(target.y)
    boundary = float(np.log(threshold / (1. - threshold)))

    def groups(mask):
        return [torch.as_tensor(np.flatnonzero(mask & (target.shot == sid)), device=device)
                for sid in np.unique(target.shot[mask])]

    pos, neg = groups(target.timely), groups(target.safe)
    learned, details = {}, {}
    for arm, (motion, alignment, use_alarm) in ARMS.items():
        size = dim + 23 if motion else dim
        start = torch.cat((origin[:size], origin[-1:]))
        coef = torch.nn.Parameter(start.clone())
        opt = torch.optim.Adam([coef], lr=.02)
        sx = (transport(source, target, conditional=alignment == "conditional")
              if alignment in ("marginal", "conditional") else source.x)
        sx = t(sx[:, :size]) if alignment != "target_only" else None
        tx = t(target.x[:, :size])
        for i in range(STEPS):
            opt.zero_grad(set_to_none=True)
            tz = tx @ coef[:-1] + coef[-1]
            point = target_w @ F.binary_cross_entropy_with_logits(tz, ty, reduction="none")
            if sx is not None:
                sz = sx @ coef[:-1] + coef[-1]
                point = .5 * (point + source_w @ F.binary_cross_entropy_with_logits(
                    sz, sy, reduction="none"
                ))
            alarm = alarm_loss(tz, pos, neg, boundary) if use_alarm else tz.new_zeros(())
            loss = point + alarm + .05 * (coef - start).square().sum()
            loss.backward()
            opt.step()
        if not torch.isfinite(coef).all():
            raise FloatingPointError(f"nonfinite fitted readout: {arm}")
        learned[arm] = coef.detach()
        details[arm] = dict(parameters=int(coef.numel()), point_loss=float(point.detach()),
                            alarm_loss=float(alarm.detach()), steps=i + 1)
    return learned, details


def cold_fit(base, ended, ids, mapping, threshold, device):
    if tuple(ended) != tuple(ids) or len(ids) != STARTUP:
        raise ValueError("only the sealed and ended first 20 shots may enter cold fit")
    target_rows = []
    for group, (sid, (raw, flag, down)) in enumerate(ended.items()):
        prepared = prepare_actual(raw, STARTUP + 1, mapping)
        x, _ = features(base, prepared, device)
        target_rows.append(make_rows(x, raw["times"], flag, down, 125., 30., group))
    target = combine(target_rows)
    source_shots, train_ids = load_source_train()
    source_rows = []
    for group, shot in enumerate(source_shots):
        # Match the currently deployed EAST MNM availability, not its raw value.
        a = shot["available"].copy()
        a[:, io23.RAW23.index("MNM")] = False
        x, _ = features(base, dict(shot, available=a), device)
        source_rows.append(make_rows(x, shot["times"], shot["is_disrupt"], shot["down"],
                                     shot["horizon"], 10., group, stride=SOURCE_STRIDE))
    source = combine(source_rows)
    learned, details = fit_readouts(base, source, target, threshold, device)
    return learned, dict(source_train_ids=train_ids, target_fit_ids=list(ended),
                         source_rows=len(source.x), target_rows=len(target.x),
                         target_timely_bags=int(len(np.unique(target.shot[target.timely]))),
                         target_safe_bags=int(len(np.unique(target.shot[target.safe]))),
                         heads=details)


def capacity(maxima, d, threshold, *, startup):
    """Posthoc capacity with actually-sealed startup cost kept fixed."""
    v = np.asarray(maxima, np.float64)
    d = np.asarray(d, bool)
    fixed = v[:startup] >= threshold
    suffix = v[startup:]
    observed = np.unique(suffix[np.isfinite(suffix)])
    candidates = np.r_[observed, np.nextafter(observed, np.inf), np.inf]
    hit = np.column_stack((np.broadcast_to(fixed, (len(candidates), startup)),
                           suffix >= candidates[:, None]))
    tp, fp = hit[:, d].sum(1), hit[:, ~d].sum(1)
    out = {}
    for cap in (.10, .15, .20):
        feasible = np.flatnonzero(fp <= cap * (~d).sum())
        if not len(feasible):
            out[str(cap)] = None
            continue
        j = max(feasible, key=lambda i: (tp[i], -fp[i], candidates[i]))
        out[str(cap)] = dict(tp=int(tp[j]), fp=int(fp[j]), tpr=float(tp[j] / d.sum()),
                             fpr=float(fp[j] / (~d).sum()))
    return out


def predict(base, signal, learned, device):
    x, logits = features(base, signal, device)
    scores = {"native": torch.from_numpy(logits).to(device).sigmoid().cpu().numpy()}
    if learned is None:
        scores.update({name: scores["native"] for name in ARMS})
    else:
        v = torch.from_numpy(x).to(device)
        for name, coef in learned.items():
            scores[name] = (v[:, :len(coef) - 1] @ coef[:-1] + coef[-1]).sigmoid().cpu().numpy()
    return scores


def run(output, *, device="cuda"):
    started = time.monotonic()
    inputs = load_inputs()
    ids = inputs["ids"]
    base, parent = load_parent(DEFAULT_PARENT)
    base.to(device)
    initial = state_digest(base)
    threshold = float(parent["threshold"])
    if (initial != inputs["baseline_protocol"]["source_state_sha256"]
            or threshold != inputs["baseline_protocol"]["threshold"]):
        raise ValueError("same source model and threshold required")
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(output / "protocol.json", dict(
        scope="DEVELOPMENT_COLD_START_NOT_INTRA_OR_INDEPENDENT_VALIDATION",
        order=ids, threshold=threshold, arms=ARMS, startup=STARTUP,
        startup_remains_scored=True, source_sha256=PARENT_SHA256,
        east_horizon_ms=125., east_strict_lead_ms=30., source_stride=SOURCE_STRIDE,
        model_changes="replacement linear risk readout, source encoder frozen",
        fit_budget=ids[:STARTUP], suffix_fit_labels=0, input_sha256=inputs["bound"],
        producer_sha256=sha(Path(__file__)), steps=STEPS, device=device,
    ))
    learned, ended, seals = None, {}, {}
    for index, sid in enumerate(ids, 1):
        if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
            raise ValueError(f"raw EAST file changed: {sid}")
        with patch.object(io23, "read_outcome", side_effect=RuntimeError("outcome before seal")):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            signal = prepare_actual(raw, index, inputs["mapping"])
            curves = predict(base, signal, learned, device)
        prefix = f"{index:04d}_{sid}"
        ref = inputs["baseline_shots"] / prefix
        check_reference(ref.with_suffix(".npz"), ref.with_name(prefix + "_seal.json"),
                        inputs["seals"][index - 1], raw["times"], curves["native"])
        path = output / "shots" / f"{prefix}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, times=raw["times"], **curves)
            f.flush()
            os.fsync(f.fileno())
        seals[path.name] = sha(path)
        if index <= STARTUP:
            ended[sid] = (raw, *io23.read_outcome("EAST", sid))
        if index == STARTUP:
            print(dict(stage="first20_sealed", n=len(ended)), flush=True)
            learned, detail = cold_fit(base, ended, ids[:STARTUP], inputs["mapping"],
                                       threshold, device)
            write(output / "fit.json", detail)
            with (output / "readouts.npz").open("xb") as f:
                np.savez(f, **{name: w.cpu().numpy() for name, w in learned.items()})
            print(dict(stage="fit_complete", target_rows=detail["target_rows"],
                       source_rows=detail["source_rows"], seconds=time.monotonic() - started),
                  flush=True)
        if index % 40 == 0 or index == len(ids):
            print(dict(stage="sealed", shots=index, seconds=time.monotonic() - started), flush=True)
    if len(seals) != 360 or state_digest(base) != initial or sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError("complete360 and unchanged parent required")

    names = ("native", *ARMS)
    per, maxima = ({name: [] for name in names} for _ in range(2))
    for i, sid in enumerate(ids, 1):
        name = f"{i:04d}_{sid}.npz"
        path = output / "shots" / name
        if sha(path) != seals[name]:
            raise ValueError("curve changed after seal")
        if i <= STARTUP:
            _, flag, down = ended[sid]
        else:
            flag, down = io23.read_outcome("EAST", sid)
        with np.load(path, allow_pickle=False) as trace:
            times = trace["times"]
            eligible = (np.round((down - times) * 1000., 9) > 30.
                        if flag else np.ones(len(times), bool))
            for arm in names:
                z = trace[arm]
                per[arm].append(dict(shot=sid, disruptive=flag, down_s=down,
                                     **score_curve(times, z, flag, down, threshold)))
                maxima[arm].append(float(z[eligible].max()) if eligible.any() else -np.inf)
    d = [r["disruptive"] for r in per["native"]]
    if sum(d) != 180 or any(per[name][:STARTUP] != per["native"][:STARTUP] for name in names):
        raise AssertionError("fixed class balance and unchanged startup predictions required")
    result = dict(
        status="COMPLETED_COLD_START_TRANSFER_DEVELOPMENT",
        seconds=time.monotonic() - started,
        metrics={arm: {
            "full360": timing_decomposition(rows),
            "startup20": timing_decomposition(rows[:STARTUP]),
            "suffix340": timing_decomposition(rows[STARTUP:]),
        } for arm, rows in per.items()},
        posthoc_capacity={arm: capacity(m, d, threshold, startup=STARTUP)
                          for arm, m in maxima.items()},
        posthoc_capacity_is_deployable=False,
        paired_against_native={arm: flips(per["native"], per[arm]) for arm in ARMS},
        curve_sha256=seals,
    )
    write(output / "per_shot.json", per)
    write(output / "result.json", result)
    print({arm: (m["full360"]["tp"], m["full360"]["fp"])
           for arm, m in result["metrics"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    run(args.output.resolve())
