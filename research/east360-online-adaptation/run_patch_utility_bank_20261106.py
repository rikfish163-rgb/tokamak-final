"""Seal the common low-rank action bank and prefix features without outcomes."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

import run_adjacent_patch_intra_20261106 as parent
from verify_confirmation_raw_replay_20261105 import Hashes


r, bank, op, fb = parent.r, parent.bank, parent.op, parent.feedback
old, pair, TAU = parent.old, parent.pair, parent.TAU
PLAN = r.ROOT / "PATCH_UTILITY_SELECTION_PROTOCOL_20261106.md"
PREVIOUS = r.ROOT / "probe_outputs/adjacent_patch_intra_20261106_v1"
NA = 13
ORDER = (
    "abstain",
    *(
        f"{name}{sign}"
        for name in ("recent", "recent_secant", "older_secant", "mir", "sxr", "xuv")
        for sign in ("+", "-")
    ),
)
FIELDS = ("clock", "states", "allowed", "risk", "effect", "ordered", "g", "curves")


def axes(guard, other, feedback, support):
    """Three reached-only risk axes and three orientation-neutral H25 axes."""
    u = np.zeros((len(guard), 6, op.WIDTH), dtype=np.float64)
    for j in np.flatnonzero(support):
        q = old.geo.basis(guard[j])
        recent, mid, older = other[j]
        v = np.stack(
            [
                other[j].mean(0),
                recent - mid,
                mid - older,
                *[
                    -0.5
                    * (
                        old.unit_average(feedback[j, 0, k : k + 3])
                        + old.unit_average(feedback[j, 1, k : k + 3])
                    )
                    for k in (0, 3, 6)
                ],
            ]
        )
        v = v - (v @ q) @ q.T
        norm = np.linalg.norm(v, axis=-1)
        u[j] = np.divide(
            pair.RADIUS * v,
            norm[:, None],
            where=norm[:, None] > 1e-8,
            out=np.zeros_like(v),
        )
    return u


def risk_context(native, m):
    z = native[m[:, None] - np.array([32, 24, 16, 12, 8, 4, 0])]
    return np.column_stack(
        (
            m / 128.0,
            z,
            np.maximum.accumulate(native)[m],
            z[:, -1] - z[:, 2],
        )
    ).astype(np.float32)


def arrival_features(base, p, raw, m):
    origin = m - 32
    values, mask = old.h25.arrived_targets(raw["values"], raw["available"], origin, m)
    po = p.take(origin)
    width = po.tokens.new_zeros((len(origin), op.WIDTH))
    y, supplied = (torch.as_tensor(v, device=width.device) for v in (values, mask))
    out = fb.vjps(
        lambda u: op.forward(base, po, u, 0, forecast=True)["forecast"],
        width,
        y,
        supplied,
        base.center,
        base.scale,
        independent_rows=True,
    )
    usable = supplied & torch.isfinite(y)
    keep = torch.zeros(23, dtype=torch.bool, device=y.device)
    for _, channels in fb.GROUPS:
        keep[list(channels)] = True
    usable &= ((usable.sum(1) >= fb.MIN_OBSERVED) & keep)[:, None]
    signed = []
    for target in (y, fb._reverse_observed(y, usable)):
        z = torch.asinh(
            (torch.where(usable, target, base.center) - base.center) / base.scale
        )
        delta = (z[:, 8:] - z[:, :-8]).masked_fill(~out["mask"], 0.0)
        signed.append(
            torch.stack(
                [
                    delta[:, part, list(ch)].sum((1, 2))
                    / out["mask"][:, part, list(ch)].sum((1, 2)).clamp_min(1)
                    for _, ch in fb.GROUPS
                    for part in fb.SEGMENTS
                ],
                dim=1,
            )
        )
    g = torch.stack((out["true"], out["reverse"]), dim=1)
    observed = out["terms"][:, None, :].expand(-1, 2, -1)
    data = torch.cat(
        (out["loss"], torch.stack(signed, dim=1), observed.to(width.dtype)), dim=2
    )
    return data.cpu().numpy(), g.cpu().numpy()


def reached_gradients(base, p, m):
    idx = m[:, None] - np.array([16, 8, 0, 4, 12, 24])
    packet = p.take(idx.ravel())
    u = packet.tokens.new_zeros((idx.size, op.WIDTH), requires_grad=True)
    z = op.forward(base, packet, u, 0)["logit"]
    (g,) = torch.autograd.grad(z.sum(), u)
    h = g.detach().cpu().numpy().reshape(len(m), 6, op.WIDTH)
    return h[:, :3], h[:, 3:]


@torch.no_grad()
def risk_in_batches(base, p, index, states):
    return (
        torch.cat(
            [
                op.forward(base, p.take(index[k : k + 3072]), states[k : k + 3072], 0)[
                    "logit"
                ]
                for k in range(0, len(index), 3072)
            ]
        )
        .cpu()
        .numpy()
    )


@torch.no_grad()
def safe_states(base, p, m, native, proposals):
    count = len(m)
    u = p.tokens.new_tensor(
        np.stack((proposals, -proposals), axis=2).reshape(count, NA - 1, op.WIDTH)
    )
    g = u.new_tensor(pair.SCALES)
    xp = bank.tr.reached_indices(m)
    index = np.broadcast_to(xp[:, None, None, :], (count, NA - 1, len(g), 3)).ravel()
    v = u[:, :, None, None] * g[None, None, :, None, None]
    z = risk_in_batches(
        base, p, index, v.expand(-1, -1, -1, 3, -1).reshape(-1, op.WIDTH)
    )
    shifts = z.reshape(count, NA - 1, len(g), 3) - native[xp][:, None, None]
    allowed = torch.linalg.vector_norm(u, dim=2) > 1e-7
    factor, _ = bank.tr.select_symmetric(
        u.new_tensor(shifts),
        allowed,
        scales=pair.SCALES,
        limit=pair.LIMIT - pair.GUARD_HEADROOM,
    )
    scale = factor.cpu().numpy()
    alive = scale > 0
    s = (u * factor.to(u.dtype)[:, :, None]).cpu().numpy()
    chosen = (scale[:, :, None] == np.asarray(pair.SCALES)).argmax(axis=2)
    seen = shifts[np.arange(count)[:, None], np.arange(NA - 1)[None], chosen]
    seen = np.where(alive[:, :, None], seen, 0.0)
    indices = np.repeat(m, NA - 1)
    rms = (
        op.delta(p.take(indices), u.new_tensor(s.reshape(-1, op.WIDTH)), 0)
        .square()
        .sum(-1)
        .mean(-1)
        .sqrt()
        .reshape(count, NA - 1)
        .cpu()
        .numpy()
    )
    norm = np.linalg.norm(s, axis=-1)
    if norm.max() > pair.RADIUS + 5e-7 or np.abs(seen).max() > pair.LIMIT:
        raise ValueError("bounded bank produced unsafe historical changes")
    effect = np.concatenate((seen, norm[:, :, None], rms[:, :, None]), axis=2)
    return (
        np.pad(s, ((0, 0), (1, 0), (0, 0))),
        np.pad(alive, ((0, 0), (1, 0)), constant_values=True),
        np.pad(effect, ((0, 0), (1, 0), (0, 0))),
    )


def action_curves(base, p, m, native, states, allowed):
    grid = m[:, None] + 1 + np.arange(32, dtype=np.int64)
    actual = grid < len(native)
    y = np.broadcast_to(
        np.where(actual, native[np.minimum(grid, len(native) - 1)], 0.0)[:, None],
        (len(m), NA, 32),
    ).copy()
    write = allowed.copy()
    write[:, 0] = False
    b, a, k = np.nonzero(write[:, :, None] & actual[:, None, :])
    if len(b):
        y[b, a, k] = risk_in_batches(
            base, p, grid[b, k], p.tokens.new_tensor(states[b, a])
        )
    return y


def generate(base, projection, raw):
    x, mask = bank.prior.views(base, raw)
    p = op.encode(base, x, mask, projection)
    z = bank.prior.scan(base, x, mask).detach().cpu().numpy()
    m = np.arange(63, len(z) - 1, 32, dtype=np.int64)
    if not len(m):
        return dict(
            native=z,
            clock=m,
            states=np.zeros((0, NA, op.WIDTH), np.float32),
            allowed=np.zeros((0, NA), bool),
            risk=np.zeros((0, 10), np.float32),
            effect=np.zeros((0, NA, 5), np.float32),
            ordered=np.zeros((0, 2, 27), np.float32),
            g=np.zeros((0, 2, 9, op.WIDTH), np.float32),
            curves=np.zeros((0, NA, 32), np.float32),
        )
    ordered, g = arrival_features(base, p, raw, m)
    guard, other = reached_gradients(base, p, m)
    proposal = axes(guard, other, g, p.take(m).support[:, 0].cpu().numpy())
    s, allowed, effect = safe_states(base, p, m, z, proposal)
    return dict(
        native=z,
        clock=m,
        states=s,
        allowed=allowed,
        risk=risk_context(z, m),
        effect=effect,
        ordered=ordered,
        g=g,
        curves=action_curves(base, p, m, z, s, allowed),
    )


def prefix_check(base, projection, raw, full, cut):
    small = generate(
        base, projection, {k: raw[k][:cut] for k in ("times", "values", "available")}
    )
    n = len(small["clock"])
    error = {}
    for key in ("native", "states", "risk", "effect", "g", "ordered"):
        before = full[key][: cut if key == "native" else n]
        error[key] = float(np.abs(before - small[key]).max(initial=0.0))
    present = small["clock"][:, None] + 1 + np.arange(32) < cut
    delta = np.abs(small["curves"] - full["curves"][:n])
    error["curves"] = float(np.where(present[:, None], delta, 0.0).max(initial=0.0))
    if max(error.values()) > 8e-5 or not np.array_equal(
        small["allowed"], full["allowed"][:n]
    ):
        raise ValueError(
            "prefix-only generation changed after truncating future samples"
        )
    return error


def run(root):
    if root.exists():
        raise FileExistsError(root)
    h, started = Hashes(), time.monotonic()
    for p in (
        PLAN,
        Path(__file__),
        Path(parent.__file__),
        Path(op.__file__),
        Path(fb.__file__),
    ):
        h(p)
    previous, inventory = bank.src.source_inputs(h)
    old_cfg = r.read(PREVIOUS / "protocol.json")
    old_result = r.read(PREVIOUS / "result.json")
    seal = PREVIOUS / "predictions_seal.json"
    if (
        h(seal) != old_result["seal_sha256"]
        or h(PREVIOUS / "protocol.json") != r.read(seal)["protocol_sha256"]
    ):
        raise ValueError("prior adjacent-patch receipt chain changed")
    h(PREVIOUS / "result.json")
    roles = previous["roles"]
    if len(inventory) != 1353 or set(roles["train"]) & (
        set(roles["val"]) | set(roles["test"])
    ):
        raise ValueError("require the unchanged full TRAIN1353")
    r.configure()
    base, source = r.load_parent(r.DEFAULT_PARENT)
    digest = r.state_digest(base)
    if (
        digest != previous["source_state_sha256"]
        or digest != old_cfg["source_state_sha256"]
        or source["threshold"] != r.THRESHOLD
    ):
        raise ValueError("qualified frozen source did not match")
    base.cuda()
    projection = op.basis(base)
    root.mkdir(parents=True, exist_ok=False)
    protocol = root / "protocol.json"
    r.write_new(
        protocol,
        dict(
            status="WAVEFORM_ONLY_TRAIN_ACTION_BANK",
            shots=roles["train"],
            tau=TAU,
            actions=ORDER,
            radius=pair.RADIUS,
            limit=pair.LIMIT,
            source_state_sha256=digest,
            inputs={str(p): v[1] for p, v in h.cache.items()},
            outcome_reads=0,
        ),
    )
    all_notes, shards = [], []
    for start in range(0, len(inventory), 100):
        tape, notes = defaultdict(list), []
        for j, info in enumerate(inventory[start : start + 100], start + 1):
            sid = info["shot"]
            path = r.io23.path_for("J-TEXT", sid)
            if h(path) != info["sha256"]:
                raise ValueError("raw TRAIN file differs from locked inventory")
            raw = r.simple_data.read_shot("J-TEXT", sid)
            y = generate(base, projection, raw)
            for key in (*FIELDS, "native"):
                tape[key].append(y[key])
            tape["times"].append(raw["times"])
            note = dict(
                shot=sid,
                raw_sha256=h(path),
                points=len(y["native"]),
                blocks=len(y["clock"]),
            )
            if j in (101, 501, 1001) and len(y["native"]) > 119:
                note["prefix_check"] = prefix_check(base, projection, raw, y, 119)
            notes.append(note)
        path = root / f"tape_{start // 100:02d}.npz"
        r.save_arrays(path, **{k: np.concatenate(v) for k, v in tape.items()})
        shards.append(dict(file=path.name, sha256=h(path), shots=notes))
        all_notes.extend(notes)
        print(
            f"Sealed real action bank {len(all_notes)}/{len(inventory)} TRAIN shots",
            flush=True,
        )
    if r.state_digest(base) != digest or any(
        v.grad is not None for v in base.parameters()
    ):
        raise ValueError("the source base was not frozen")
    h.check()
    r.write_new(
        root / "seal.json",
        dict(
            status="SEALED_BEFORE_OUTCOME_REVEAL",
            protocol_sha256=h(protocol),
            shards=shards,
            points=sum(x["points"] for x in all_notes),
            blocks=sum(x["blocks"] for x in all_notes),
            outcome_reads=0,
            elapsed_seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    run(cli.parse_args().root)
