"""Outcome-blind frozen-model interventions on already observed EAST history."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import alarm_order_views_20261106 as order_views
from fit_layer0_arrival_motion_20261106 import past_arrays
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import raw_for
from trajectory_replay_20261105 import SOURCE, baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes


W, P, C, STRIDE, ADVANCE, BATCH = 128, 8, 23, 32, 24, 256
NEAR = 0.15
ARMS = ("native", *order_views.ARMS)
PLAN = r.ROOT / "REAL_ALARM_TRAJECTORY_ATTRIBUTION_PLAN_20261106.md"
ATOL = 2e-5


def anchors_for(probabilities, threshold):
    p = np.asarray(probabilities)
    if (
        p.ndim != 1
        or not len(p)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not 0 < threshold < 1
    ):
        raise ValueError("finite static probability trace and threshold required")
    hit = np.flatnonzero(p >= threshold)
    stop = int(hit[0]) if len(hit) else len(p)
    i = np.arange(W - 1, stop, STRIDE, dtype=np.int64)
    if (p[i] >= threshold).any():
        raise ValueError("fixed-clock anchor is not below the first crossing")
    if len(hit):
        i = np.append(i, stop)
    first = (i == stop) & bool(len(hit))
    near = (p[i] >= threshold - NEAR) & ~first
    return i, first, near


@torch.no_grad()
def joint_relabel(base, x, a, perm):
    """Permute contents AND position embeddings before attention, leaving CLS fixed."""
    n, k = len(x), W // P
    if (
        base.training
        or any(p.requires_grad for p in base.parameters())
        or (base.cfg.window, base.cfg.patch, base.cfg.input_view) != (W, P, "source")
        or x.shape != (n, W, C)
        or a.shape != x.shape
        or a.dtype != torch.bool
        or perm.shape != (n, k)
        or perm.dtype != torch.long
        or not (perm.device == x.device == a.device)
    ):
        raise ValueError("frozen source windows and a patch permutation required")
    if not torch.equal(
        perm.sort(dim=1).values, torch.arange(k, device=x.device).expand(n, k)
    ):
        raise ValueError("invalid position permutation")
    if (perm[:, -1] != k - 1).any():
        raise ValueError("terminal patch must be held fixed")
    support = (a & torch.isfinite(x)).reshape(n, k, P * C)
    if not torch.equal(
        support.gather(1, perm[:, :, None].expand(-1, -1, P * C)), support
    ):
        raise ValueError("joint relabel cannot change an observation mask")
    token_index = torch.cat(
        (perm, torch.full((n, 1), k, device=perm.device, dtype=perm.dtype)), dim=1
    )
    seen = 0

    def relabel(_module, args, kwargs):
        nonlocal seen
        if not args or args[0].shape[:2] != token_index.shape:
            raise RuntimeError("unexpected encoder token layout")
        seen += 1
        inp = args[0].gather(1, token_index[..., None].expand_as(args[0]))
        return (inp, *args[1:]), kwargs

    handle = base.encoder.register_forward_pre_hook(relabel, with_kwargs=True)
    try:
        out = base(x, a)["logit"]
    finally:
        handle.remove()
    if seen != 1:
        raise RuntimeError("joint token relabel was not applied exactly once")
    return out


def normalized(x, a, center, scale):
    return np.arcsinh((np.where(a, x, center) - center) / scale)


@torch.no_grad()
def forward_views(base, v, m, ix, first):
    n, device = len(ix), base.center.device
    center = base.center.detach().cpu().numpy()
    scale = base.scale.detach().cpu().numpy()
    out = dict(
        logits=np.full((n, len(ARMS)), np.nan, dtype=np.float32),
        supported=np.zeros((n, len(ARMS) - 1), dtype=bool),
        motion=np.full((n, len(ARMS) - 1), np.nan, dtype=np.float64),
        coverage=np.empty(n, dtype=np.float64),
        placebo=np.full(n, np.nan, dtype=np.float32),
        future_index=np.full(n, -1, dtype=np.int64),
        future=np.full((n, 3), np.nan, dtype=np.float32),
    )

    def tensor(a):
        return torch.as_tensor(a, device=device)

    def predict(x, a):
        return base(tensor(x), tensor(a))["logit"].cpu().numpy()

    for lo in range(0, n, BATCH):
        sl = slice(lo, lo + BATCH)
        at = ix[sl]
        x, a = past_arrays(v, m, at)
        perms, supported = order_views.plans(a)
        supported &= at[:, None] >= W - 1
        out["supported"][sl] = supported
        out["coverage"][sl] = a.mean((1, 2))
        native = predict(x, a)
        out["logits"][sl, 0] = native
        z = normalized(x, a, center, scale)
        for j in range(len(ARMS) - 1):
            use = supported[:, j]
            if use.any():
                dst = lo + np.flatnonzero(use)
                p = perms[use, j]
                moved = order_views.apply(x[use], a[use], p)
                out["logits"][dst, j + 1] = predict(moved, a[use])
                out["motion"][dst, j] = order_views.motion_rms(z[use], a[use], p)
        relabel = supported[:, 0] & (
            first[sl] | (np.arange(lo, lo + len(at)) % 16 == 0)
        )
        if relabel.any():
            dst = lo + np.flatnonzero(relabel)
            out["placebo"][dst] = (
                joint_relabel(
                    base,
                    tensor(x[relabel]),
                    tensor(a[relabel]),
                    tensor(perms[relabel, 0]),
                )
                .cpu()
                .numpy()
                - native[relabel]
            )
        later = supported[:, 0] & (at + ADVANCE < len(v))
        if later.any():
            dst = lo + np.flatnonzero(later)
            target_ix = at[later] + ADVANCE
            past, obs = past_arrays(v, m, target_ix)
            p = order_views.transport(perms[later, 0], ADVANCE)
            native1 = predict(past, obs)
            moved1 = predict(order_views.apply(past, obs, p), obs)
            shift0 = out["logits"][dst, 1] - native[later]
            out["future"][dst] = np.stack((native1, moved1, native1 + shift0), axis=1)
            out["future_index"][dst] = target_ix
    if (
        not np.isfinite(out["logits"][:, 0]).all()
        or not np.array_equal(np.isfinite(out["logits"][:, 1:]), out["supported"])
        or not np.array_equal(np.isfinite(out["motion"]), out["supported"])
        or not np.array_equal(
            np.isfinite(out["future"]).all(1), out["future_index"] >= 0
        )
        or np.any(np.abs(out["placebo"][np.isfinite(out["placebo"])]) > ATOL)
    ):
        raise FloatingPointError(
            "view support, transport or joint relabel invariant failed"
        )
    return out


def witness(base, v, a, idx, first, output):
    q = np.flatnonzero(output["future_index"] >= 0)
    if not len(q):
        return None
    i = int(q[len(q) // 2])
    end = idx[i] + 1
    new = forward_views(base, v[:end], a[:end], idx[i : i + 1], first[i : i + 1])
    if (
        not np.array_equal(new["supported"][0], output["supported"][i])
        or new["future_index"][0] != -1
        or not np.array_equal(new["motion"][0], output["motion"][i], equal_nan=True)
    ):
        raise ValueError(
            "prefix cut changed earlier support or accessed an unarrived future"
        )
    delta = float(np.nanmax(np.abs(new["logits"][0] - output["logits"][i])))
    if delta > ATOL:
        raise ValueError("prefix cut changed earlier logits")
    return dict(anchor=int(idx[i]), max_abs_logit=delta, future_unavailable=True)


def run(output):
    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    r.configure()
    order, cfg, ref = reference()
    h = Hashes()
    mpath = r.COLD / "input_map.json"
    mapping = r.read(mpath)
    base, meta = r.load_parent(r.DEFAULT_PARENT)
    source = r.state_digest(base)
    if (
        mapping["calibration_ids"] != order[:20]
        or h(mpath) != cfg["input_sha256"][str(mpath)]
        or source != cfg["source_state_sha256"]
        or meta["threshold"] != cfg["threshold"]
        or cfg["threshold"] != r.THRESHOLD
    ):
        raise ValueError("frozen outcome-blind source protocol required")
    base.cuda()
    inputs = {
        str(p.resolve()): h(p)
        for p in (
            Path(__file__),
            Path(order_views.__file__),
            PLAN,
            SOURCE / "protocol.json",
            SOURCE / "completed.json",
            mpath,
            r.DEFAULT_PARENT,
            Path(r.prepare_actual.__code__.co_filename),
            Path(r.io23.__file__),
            Path(r.simple_data.__file__),
            Path(raw_for.__code__.co_filename),
            Path(past_arrays.__code__.co_filename),
            Path(reference.__code__.co_filename),
        )
    }
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            scope="FROZEN_REAL_EAST_CAUSAL_WINDOW_DIAGNOSTIC",
            order=order,
            arms=ARMS,
            threshold=r.THRESHOLD,
            stride=STRIDE,
            advance=ADVANCE,
            near=NEAR,
            window=W,
            source_state_sha256=source,
            input_sha256=inputs,
            labels_read=0,
            new_alarm_curves=0,
            learned_parameters=0,
        ),
    )
    rows, checks = [], []
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("outcome forbidden before seal"),
    ):
        for rank, sid in enumerate(order, start=1):
            raw = raw_for(sid, cfg["raw_sha256"])
            times, prior = baseline_for(rank, sid, ref)
            if not np.array_equal(times, raw["times"]):
                raise ValueError("fixed reference clock differs from raw")
            q, is_first, near = anchors_for(prior, cfg["threshold"])
            data = r.prepare_actual(raw, rank, mapping)
            v, m = data["values"], data["available"]
            d = forward_views(base, v, m, q, is_first)
            p0 = torch.from_numpy(d["logits"][:, 0]).sigmoid().numpy()
            t1 = d["future_index"] >= 0
            p1 = torch.from_numpy(d["future"][t1, 0]).sigmoid().numpy()
            if not (
                np.allclose(p0, prior[q], atol=2e-6, rtol=0.0)
                and np.allclose(p1, prior[d["future_index"][t1]], atol=2e-6, rtol=0.0)
            ):
                raise ValueError("frozen native scores did not reproduce the reference")
            if rank in (21, 165, 320):
                check = witness(base, v, m, q, is_first, d)
                checks.append(dict(rank=rank, shot=sid, check=check))
            p = output / "shots" / f"{rank:03}_{sid}.npz"
            r.save_arrays(p, index=q, times=times[q], first=is_first, near=near, **d)
            rows.append(
                dict(
                    rank=rank,
                    shot=sid,
                    anchors=len(q),
                    sha256=h(p),
                    raw_sha256=cfg["raw_sha256"][str(sid)],
                    name=p.name,
                )
            )
            if rank % 60 == 0:
                print(f"sealed order interventions {rank}/360", flush=True)
    if r.state_digest(base) != source or base.encoder._forward_pre_hooks:
        raise ValueError("model weights or runtime hooks changed")
    h.check()
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_ALARM_ORDER_BLIND_SEAL",
            shots=rows,
            prefix=checks,
            protocol_sha256=h(output / "protocol.json"),
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
