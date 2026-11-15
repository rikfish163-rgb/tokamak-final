"""Fit matched source-only risk probes and seal predictions before held outcomes."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import alarm_order_conditional_20261106 as conditional
from fit_layer0_arrival_motion_20261106 import past_arrays
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import raw_for
from trajectory_replay_20261105 import SOURCE, baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = r.ROOT / "DEEP_MISS_HISTORY_RISK_PLAN_20261106.md"
ARMS = ("short", "real", "stale", "scrambled")
SEEDS, MASS = (131, 317, 563), (0.40, 0.40, 0.20)
BATCH, W = 256, 128


def source_training_anchors(y, valid):
    y, a = np.asarray(y), np.asarray(valid)
    if y.ndim != 1 or a.shape != y.shape or a.dtype != bool:
        raise ValueError("aligned source labels and endpoint validity required")
    q = np.arange(W - 1, len(y), 64, dtype=np.int64)
    positive = np.flatnonzero((y == 1) & a & (np.arange(len(y)) >= W - 1))
    if len(positive):
        q = np.unique(np.r_[q, positive[::4], positive[-1]])
    return q[a[q]]


def evaluation_anchors(n, stride, first):
    if stride not in (8, 32) or n < 0:
        raise ValueError("declared held stride and signal length required")
    q = np.arange(W - 1, n, stride, dtype=np.int64)
    if first is not None:
        if not 0 <= first < n:
            raise ValueError("original first crossing outside the trace")
        if first >= W - 1:
            q = np.unique(np.r_[q, first])
    return q


def group_weights(shot, group):
    ids, g = np.asarray(shot), np.asarray(group)
    if ids.shape != g.shape or g.ndim != 1 or set(g.tolist()) != {0, 1, 2}:
        raise ValueError("all three source training strata required")
    w = np.empty(len(ids), dtype=np.float32)
    for k, mass in enumerate(MASS):
        a = g == k
        _, inverse, count = np.unique(ids[a], return_inverse=True, return_counts=True)
        w[a] = mass / (len(count) * count[inverse])
    return w


@torch.no_grad()
def frozen_short(base, v, m, anchors):
    """Frozen risk head and its input vector; no change to native decision."""
    hidden, logits = [], []
    device = base.center.device
    buffer = []

    def capture(_module, args):
        buffer.append(args[0])

    hook = base.risk_head.register_forward_pre_hook(capture)
    try:
        for i in range(0, len(anchors), BATCH):
            ix = anchors[i : i + BATCH]
            x, a = past_arrays(v, m, ix)
            z = base(
                torch.as_tensor(x, device=device), torch.as_tensor(a, device=device)
            )["logit"]
            if len(buffer) != 1 or buffer[0].shape != (len(ix), 64):
                raise ValueError("expected one frozen CLS projection")
            logits.append(z.cpu().numpy())
            hidden.append(buffer.pop().cpu().numpy())
    finally:
        hook.remove()
    if not logits:
        return np.empty(0, dtype=np.float32), np.empty((0, 64), dtype=np.float32)
    return np.concatenate(logits), np.concatenate(hidden)


def feature_pack(base, data, anchors):
    from deep_miss_history_features_20261106 import build

    v, m, times = data["values"], data["available"], data["times"]
    center, scale = (
        base.center.detach().cpu().numpy(),
        base.scale.detach().cpu().numpy(),
    )
    z = np.arcsinh((np.where(m, v, center) - center) / scale)
    stat = build(z, m, anchors, data["shot"])
    native, cls = frozen_short(base, v, m, anchors)
    clock = np.log1p(times[anchors] - times[0]).astype(np.float32)
    sr, ss = stat["support_real"], stat["support_stale"]
    x = np.column_stack((stat["short"], cls, native, clock, sr, ss)).astype(np.float32)
    hist = {
        k: stat[k].reshape(len(anchors), 3 * stat["short"].shape[1]).astype(np.float32)
        for k in ARMS[1:]
    }
    hist["short"] = np.zeros_like(hist["real"])
    all_x = {k: np.concatenate((x, hist[k]), axis=1) for k in ARMS}
    # Only the current physical availability is used for matching; the arms
    # all also receive identical flags for available older blocks.
    ix = anchors[:, None] + np.arange(1 - W, 1)[None, :]
    coverage = m[ix].mean((1, 2))
    return all_x, native, sr, ss, coverage


def collect_train(base, roles, adv, h):
    parts = {k: [] for k in ARMS}
    ys, zs, ids, strata = [], [], [], []
    inv = []
    for j, sid in enumerate(roles, start=1):
        path = r.io23.path_for("J-TEXT", sid)
        before = h(path)
        shot = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        horizon, _, _ = r.io23.effective_horizon_ms(sid, adv)
        y, a = r.io23.labels_at(shot["times"], flag, down, horizon)
        ix = source_training_anchors(y, a)
        st = np.where(y[ix] == 1, 0, 2 if flag else 1).astype(np.int8)
        counts = [int((st == g).sum()) for g in range(3)]
        supports = [0] * 3
        if len(ix):
            x, z, sr, ss, _ = feature_pack(base, shot, ix)
            common = sr.all(1) & ss.all(1)
            supports = [int(((st == g) & common).sum()) for g in range(3)]
            for k in ARMS:
                parts[k].append(x[k])
            ys.append(y[ix])
            zs.append(z)
            ids.append(np.full(len(ix), sid, dtype=np.int64))
            strata.append(st)
        if before != h(path):
            raise ValueError("source training raw file changed")
        inv.append(
            dict(
                shot=sid,
                sha256=before,
                counts=counts,
                common=supports,
                n_samples=len(shot["times"]),
            )
        )
        if j % 200 == 0 or j == len(roles):
            print(f"TRAIN signals and labels {j}/{len(roles)}", flush=True)
    labels, logits, shots, groups = (np.concatenate(x) for x in (ys, zs, ids, strata))
    return (
        {k: np.concatenate(v) for k, v in parts.items()},
        logits,
        labels,
        group_weights(shots, groups),
        inv,
    )


def readout(states, x, z, device):
    from deep_miss_risk_probe_20261106 import predict

    if not len(z):
        return np.empty((0, len(SEEDS), len(ARMS)), dtype=np.float32)
    out = np.stack(
        [
            np.column_stack([predict(states[s][k], x[k], z, device) for k in ARMS])
            for s in SEEDS
        ],
        axis=1,
    ).astype(np.float32)
    if out.shape != (len(z), len(SEEDS), len(ARMS)) or not np.isfinite(out).all():
        raise ValueError(
            "held risk probes must be finite and use the same source seeds"
        )
    return out


def seal_held(base, states, out, name, ids, ctx, h):
    from deep_miss_risk_probe_20261106 import FEATURE_CLIP

    prev, base_cfg, cond_cfg, cond_prev, mapping = ctx
    rows, witness = [], []
    threshold = r.THRESHOLD
    place = out / name
    place.mkdir()
    for rank, sid in enumerate(ids, start=1):
        first, p = None, None
        path = r.io23.path_for("J-TEXT" if name == "val" else "EAST", sid)
        before = h(path)
        if name == "val":
            shot = r.simple_data.read_shot("J-TEXT", sid)
        else:
            cfg = base_cfg if name == "fixed" else cond_cfg
            raw = raw_for(sid, cfg["raw_sha256"])
            if name == "fixed":
                t, p = baseline_for(rank, sid, prev)
            else:
                t, p = conditional.baseline(
                    h, rank, sid, cond_prev, cfg["raw_sha256"][str(sid)]
                )
            if not np.array_equal(t, raw["times"]):
                raise ValueError("original static signal grid differs")
            hit = np.flatnonzero(p >= threshold)
            first = int(hit[0]) if len(hit) else None
            shot = {
                **r.prepare_actual(raw, rank if name == "fixed" else 21, mapping),
                "times": raw["times"],
                "shot": sid,
            }
        n = len(shot["times"])
        ix = evaluation_anchors(n, 8 if name == "val" else 32, first)
        x, z, sr, ss, cov = feature_pack(base, shot, ix)
        y = readout(states, x, z, base.center.device)
        train_norm = states[SEEDS[0]]["real"]
        real_common = x["real"][sr.all(1) & ss.all(1)]
        clips = np.count_nonzero(
            np.abs((real_common - train_norm["mean"]) / train_norm["scale"])
            > FEATURE_CLIP,
            axis=0,
        )
        if p is not None and not np.allclose(
            torch.from_numpy(z).sigmoid().numpy(), p[ix], atol=2e-6, rtol=0.0
        ):
            raise ValueError(
                "frozen native logit no longer matches original probability"
            )
        candidates = np.flatnonzero(sr.all(1) & ss.all(1) & (ix < n - 1))
        if rank in (2, 40, 120) and len(candidates):
            j = candidates[len(candidates) // 2]
            end = ix[j] + 1
            prefix = dict(shot)
            for k in ("values", "available", "times"):
                prefix[k] = prefix[k][:end]
            x1, z1, sr1, ss1, cov1 = feature_pack(base, prefix, ix[j : j + 1])
            y1 = readout(states, x1, z1, base.center.device)[0]
            err = float(np.max(np.abs(y1 - y[j])))
            if (
                not np.array_equal(sr1[0], sr[j])
                or not np.array_equal(ss1[0], ss[j])
                or cov1[0] != cov[j]
                or err > 5e-4
                or any(
                    not np.allclose(x1[k][0], x[k][j], atol=2e-5, rtol=0.0)
                    for k in ARMS
                )
            ):
                raise ValueError("unarrived suffix changed held feature or risk score")
            witness.append(dict(shot=sid, anchor=int(ix[j]), max_abs_logit=err))
        if before != h(path):
            raise ValueError("held signals changed")
        dest = place / f"{rank:04d}_{sid}.npz"
        r.save_arrays(
            dest,
            index=ix,
            times=shot["times"][ix],
            native=z,
            risk=y,
            support_real=sr,
            support_stale=ss,
            coverage=cov,
            real_clipped_counts=clips,
        )
        rows.append(
            dict(
                shot=sid,
                rank=rank,
                file=dest.name,
                sha256=h(dest),
                raw_sha256=before,
                first=first,
                samples=n,
                anchors=len(ix),
            )
        )
        if rank % 100 == 0 or rank == len(ids):
            print(f"sealed {name}: {rank}/{len(ids)}", flush=True)
    return rows, witness


def run(out):
    import deep_miss_history_features_20261106 as features
    import deep_miss_risk_probe_20261106 as fitting

    if out.exists():
        raise FileExistsError(out)
    start, h = time.monotonic(), Hashes()
    r.configure()
    base, meta = r.load_parent(r.DEFAULT_PARENT)
    order, cfg, prior = reference()
    extra, cp, cb, mpath, mapping = conditional.references(h)
    digest = r.state_digest(base)
    if (
        digest != cfg["source_state_sha256"]
        or digest != cp["source_state_sha256"]
        or meta["threshold"] != cfg["threshold"]
        or mapping["calibration_ids"] != order[:20]
        or not r.io23.JTEXT_INDEX.is_file()
    ):
        raise ValueError("fixed qualified source, train roles and maps required")
    base.cuda()
    roles = r.simple_data.source_roles()
    adv = r.simple_data.advanced_times()
    files = (
        Path(__file__),
        Path(features.__file__),
        Path(fitting.__file__),
        PLAN,
        r.DEFAULT_PARENT,
        r.io23.JTEXT_INDEX,
        r.io23.JTEXT_ADVANCED_TIME_PATH,
        mpath,
        SOURCE / "protocol.json",
        SOURCE / "completed.json",
        conditional.ROSTER,
        conditional.PREVIOUS / "protocol.json",
        conditional.PREVIOUS / "completed.json",
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(conditional.__file__),
        Path(past_arrays.__code__.co_filename),
        Path(raw_for.__code__.co_filename),
    )
    binding = {str(p.resolve()): h(p) for p in files}
    out.mkdir(parents=True)
    (out / "fit").mkdir()
    r.write_new(
        out / "protocol.json",
        dict(
            input_sha256=binding,
            source_state_sha256=digest,
            train=roles["train"],
            val=roles["val"],
            fixed=order,
            conditional=extra,
            seeds=SEEDS,
            arms=ARMS,
            training_mass=MASS,
            threshold=r.THRESHOLD,
            held_outcome_reads=0,
            source_test_reads=0,
            learned_base_parameters=0,
        ),
    )
    x, z, y, w, inv = collect_train(base, roles["train"], adv, h)
    fits, receipt = {}, {}
    for seed in SEEDS:
        fit, log = fitting.train_heads(x, z, y, w, seed, base.center.device)
        p = out / "fit" / f"{seed}.npz"
        r.save_arrays(p, **{f"{k}__{a}": b for k in ARMS for a, b in fit[k].items()})
        fits[seed] = fit
        receipt[str(seed)] = dict(sha256=h(p), training=log)
        print(f"fitted {seed} on {len(z)} J-TEXT TRAIN rows", flush=True)
    del x, z, y, w
    r.write_new(
        out / "fit_completed.json",
        dict(train=inv, fits=receipt, labels="J-TEXT TRAIN only"),
    )
    rows, witness = {}, {}
    ctx = prior, cfg, cp, cb, mapping
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("held outcome unavailable before seal"),
    ):
        for name, ids in (
            ("val", roles["val"]),
            ("fixed", order),
            ("conditional", extra),
        ):
            rows[name], witness[name] = seal_held(base, fits, out, name, ids, ctx, h)
    if digest != r.state_digest(base) or base.risk_head._forward_pre_hooks:
        raise ValueError("frozen model or hook state changed")
    h.check()
    r.write_new(
        out / "completed.json",
        dict(
            status="COMPLETE_SOURCE_ONLY_HISTORY_RISK_BLIND_SEAL",
            protocol_sha256=h(out / "protocol.json"),
            fit_sha256=h(out / "fit_completed.json"),
            rows=rows,
            prefix=witness,
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
