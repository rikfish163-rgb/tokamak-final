"""Matched adjacent-patch real replay with a sealed waveform/outcome boundary."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

import adjacent_patch_compare_20261106 as pair
import adjacent_patch_feedback_20261106 as feedback
import adjacent_patch_operator_20261106 as op
import run_h25_direction_mechanism_20261106 as old
from inspect_peak_contrast_first_alarm_20261106 import first
from run_h25_increment_mechanism_20261106 import paired_shot_effect
from run_h25_late_capacity_20261106 import load_receipts
from verify_confirmation_raw_replay_20261105 import Hashes


r, bank, TAU = old.r, old.bank, old.TAU
PLAN = r.ROOT / "ADJACENT_PATCH_INTRA_PROTOCOL_20261106.md"
PREVIOUS = r.ROOT / "probe_outputs/h25_direction_mechanism_20261106_v1"
MODES = (0, 0, 1, 1)
ARMS = (
    "static",
    "motion",
    "motion_reverse",
    "level",
    "level_reverse",
    "motion_scalar",
    "level_scalar",
)
BLOCK_DATA = ("clock", "states", "rank", "active", "guard", "offset", "edit_rms")


def gradients(base, packed, raw, clock, *, independent_rows=True):
    origin = clock - 32
    reached = bank.tr.reached_indices(clock)
    past = packed.take(reached.ravel())
    targets, valid = old.h25.arrived_targets(
        raw["values"], raw["available"], origin, clock
    )
    po = packed.take(origin)
    b = po.tokens.new_zeros((len(origin), op.WIDTH))
    truth, mask = (torch.as_tensor(q, device=b.device) for q in (targets, valid))
    result, dp = [], []
    for mode in (0, 1):
        g = feedback.vjps(
            lambda u: op.forward(base, po, u, mode, forecast=True)["forecast"],
            b,
            truth,
            mask,
            base.center,
            base.scale,
            independent_rows=independent_rows,
        )
        result.append(g)
        state = past.tokens.new_zeros((len(past.tokens), op.WIDTH), requires_grad=True)
        z = op.forward(base, past, state, mode)["logit"]
        (gradient,) = torch.autograd.grad(z.sum(), state)
        dp.append(gradient.detach().reshape(len(clock), 3, op.WIDTH).cpu().numpy())
    if not torch.equal(result[0]["mask"], result[1]["mask"]) or not torch.equal(
        result[0]["terms"], result[1]["terms"]
    ):
        raise ValueError("both representations must receive the same observed feedback")
    g = np.stack(
        [x[k].cpu().numpy() for x in result for k in ("true", "reverse")], axis=1
    )
    terms = result[0]["terms"].cpu().numpy().reshape(-1, 3, 3)
    groups = terms.any(axis=2).sum(axis=1)
    valid = (
        (groups >= 2)
        & po.support.all(dim=1).cpu().numpy()
        & packed.take(clock).support.all(dim=1).cpu().numpy()
    )
    return g, np.stack(dp, axis=1), valid


def directions(g, past, alive):
    states = np.zeros((len(g), 4, op.WIDTH), dtype=np.float64)
    ranks = np.zeros(len(g), dtype=np.int16)
    for i in np.flatnonzero(alive):
        q, ranks[i] = pair.common_spaces(g[i])
        if ranks[i] < 4:
            continue
        for a, mode in enumerate(MODES):
            descent = -old.unit_average(g[i, a])
            states[i, a] = q[a] @ old.mix.projected(q[a], past[i, mode], descent)
    return states, ranks


@torch.no_grad()
def choose_states(base, packed, clock, states, ranks, native):
    pm = packed.take(clock)
    u = pm.tokens.new_tensor(states)
    before = np.stack(
        [
            op.delta(pm, u[:, k], mode).square().sum(-1).mean(-1).sqrt().cpu().numpy()
            for k, mode in enumerate(MODES)
        ],
        axis=1,
    )
    factor, match = pair.equal_rms_factors(before)
    u = u * u.new_tensor(factor)[:, :, None]
    reached = bank.tr.reached_indices(clock)
    index = np.broadcast_to(
        reached[:, None, None, :], (len(clock), len(pair.SCALES), 2, 3)
    ).ravel()
    p = packed.take(index)
    g, sign = u.new_tensor(pair.SCALES), u.new_tensor([1.0, -1.0])
    shifts = []
    for k, mode in enumerate(MODES):
        v = (
            u[:, k, None, None, None, :]
            * g[None, :, None, None, None]
            * sign[None, None, :, None, None]
        )
        z = op.forward(
            base, p, v.expand(-1, -1, -1, 3, -1).reshape(-1, op.WIDTH), mode
        )["logit"]
        shifts.append(
            z.reshape(len(clock), len(g), 2, 3).cpu().numpy()
            - native[reached][:, None, None]
        )
    shift = np.stack(shifts, axis=1)
    step = pair.common_step(shift, match & (ranks >= 4))
    row = np.arange(len(clock))
    scale = np.where(step >= 0, np.asarray(pair.SCALES)[step], 0.0)
    new = (u * u.new_tensor(scale)[:, None, None]).cpu().numpy()
    received = shift[row, :, step]
    anchor = np.where(step[:, None] >= 0, received[:, :, 0, -1], 0.0)
    guard = np.where(step >= 0, np.max(np.abs(received), axis=(1, 2, 3)), 0.0)
    rms = np.stack(
        [
            op.delta(pm, pm.tokens.new_tensor(new[:, k]), mode)
            .square()
            .sum(-1)
            .mean(-1)
            .sqrt()
            .cpu()
            .numpy()
            for k, mode in enumerate(MODES)
        ],
        axis=1,
    )
    if (
        np.linalg.norm(new, axis=-1).max(initial=0.0) > pair.RADIUS + 5e-7
        or guard.max(initial=0.0) > pair.LIMIT
        or np.ptp(rms, axis=1).max(initial=0.0) > 1e-6
    ):
        raise ValueError(
            "matched physical edit and output trust contracts did not hold"
        )
    return new, anchor, guard, step >= 0, rms


@torch.no_grad()
def written_trace(base, packed, native, clock, states, active, offsets):
    z = np.repeat(native[:, None], len(ARMS), axis=1)
    future = pair.future_blocks(clock, len(native))
    selected = np.flatnonzero(active)
    if not len(selected):
        return z
    j = np.concatenate([np.full(len(future[i]), i, dtype=np.int64) for i in selected])
    t = np.concatenate([future[i] for i in selected])
    p = packed.take(t)
    for k, mode in enumerate(MODES):
        u = p.tokens.new_tensor(states[j, k])
        z[t, 1 + k] = op.forward(base, p, u, mode)["logit"].cpu().numpy()
    z[t, 5] = native[t] + offsets[j, 0]
    z[t, 6] = native[t] + offsets[j, 2]
    return z


def predict(base, projection, raw):
    x, a = bank.prior.views(base, raw)
    p = op.encode(base, x, a, projection)
    native = bank.prior.scan(base, x, a).detach().cpu().numpy()
    with torch.no_grad():
        native_op = (
            op.forward(base, p, p.tokens.new_zeros((len(x), op.WIDTH)), 0)["logit"]
            .cpu()
            .numpy()
        )
    error = float(np.abs(native_op - native).max())
    if error > 5e-5 or not np.array_equal(native_op >= TAU, native >= TAU):
        raise ValueError("zero-adaptation patch path did not reproduce frozen model")
    clock = np.arange(63, len(native) - 1, 32, dtype=np.int64)
    if len(clock):
        g, jp, ok = gradients(base, p, raw, clock)
        proposed, rank = directions(g, jp, ok)
        u, offset, guard, active, rms = choose_states(
            base, p, clock, proposed, rank, native
        )
    else:
        # There is no H25 feedback maturity in a prefix shorter than 65 points.
        u, offset, guard, active, rms, rank = (
            np.zeros((0, 4, op.WIDTH), np.float32),
            np.zeros((0, 4), np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, bool),
            np.zeros((0, 4), np.float32),
            np.zeros(0, np.int16),
        )
    z = written_trace(base, p, native, clock, u, active, offset)
    return dict(
        curve=z,
        clock=clock,
        states=u,
        rank=rank,
        active=active,
        guard=guard,
        offset=offset,
        edit_rms=rms,
        native_error=error,
    )


def check_prefix(base, projection, raw, full):
    cut = min(119, len(raw["times"]) - 1)
    if cut <= 96:
        raise ValueError("verification witness must include two feedback releases")
    short = predict(
        base, projection, {k: raw[k][:cut] for k in ("times", "values", "available")}
    )
    n = len(short["clock"])
    err = float(np.abs(short["curve"] - full["curve"][:cut]).max())
    state_err = float(np.abs(short["states"] - full["states"][:n]).max())
    if (
        err > 8e-5
        or state_err > 8e-5
        or not np.array_equal(short["active"], full["active"][:n])
        or not np.array_equal(short["curve"] >= TAU, full["curve"][:cut] >= TAU)
    ):
        raise ValueError("strict prefix truncation changed decisions")
    return dict(prefix=cut, curve_error=err, state_error=state_err)


def check_row_independence(base, projection, raw):
    """For this exact frozen backend, compare the optimized VJP to its diagonal."""
    x, a = bank.prior.views(base, raw)
    p = op.encode(base, x, a, projection)
    clock = np.array([63, 95], dtype=np.int64)
    g, _, _ = gradients(base, p, raw, clock, independent_rows=True)
    exact, _, _ = gradients(base, p, raw, clock, independent_rows=False)
    error = float(np.abs(g - exact).max())
    if error > 2e-6:
        raise ValueError("batched VJP is not equivalent to the exact diagonal")
    return error


def score_shot(shot, times, pred, flag, down, horizon, roles):
    z, clock = pred["curve"], pred["clock"]
    native = first(times, z[:, 0], flag, down, TAU)
    actions = [first(times, col, flag, down, TAU) for col in z.T]
    out = dict(shot=shot, disruptive=flag, scores=actions)
    roles_out = []
    for q in roles:
        k = np.flatnonzero(clock == q["maturity"])
        if len(k) != 1:
            raise ValueError("frozen cohort release did not replay")
        j = int(k[0])
        use = np.asarray(q["eligible"], dtype=np.int64)
        block = np.arange(q["maturity"] + 1, min(q["maturity"] + 33, len(times)))
        if q["role"] == "positive":
            lead = ((down - times[block]) * 1000).round(9)
            expected = block[(lead > 10) & (lead <= horizon + 1e-6)]
        elif q["role"] == "protect":
            expected = (
                block[block <= native["first"]] if native["timely"] else np.array([])
            )
        else:
            expected = block
        if not np.array_equal(use, expected) or flag != (q["role"] != "safe"):
            raise ValueError(
                "frozen role timestamps disagree with the authoritative outcome"
            )
        v = z[use].max(0).astype(np.float64)
        harm = (
            v < TAU
            if q["role"] == "protect"
            else (v[0] < TAU) & (v >= TAU)
            if q["role"] == "safe"
            else np.zeros(len(v), dtype=bool)
        )
        roles_out.append(
            dict(
                shot=shot,
                maturity=q["maturity"],
                role=q["role"],
                late=q["late"],
                rank=int(pred["rank"][j]),
                active=bool(pred["active"][j]),
                peaks=v.tolist(),
                delta=(v - v[0]).tolist(),
                harmful=harm.tolist(),
                shape_change_max=np.abs(z[use, 1] - z[use, 5]).max().item(),
            )
        )
    return out, roles_out


def summarize(rows, windows):
    rates = {}
    for k, arm in enumerate(ARMS):
        rates[arm] = dict(
            timely=sum(x["scores"][k]["timely"] for x in rows),
            fp=sum(x["scores"][k]["false_alarm"] for x in rows),
            recover=[
                x["shot"]
                for x in rows
                if not x["scores"][0]["timely"] and x["scores"][k]["timely"]
            ],
            lost=[
                x["shot"]
                for x in rows
                if x["scores"][0]["timely"] and not x["scores"][k]["timely"]
            ],
        )
    blocks = {}
    for role in old.ROLES:
        q = [x for x in windows if x["role"] == role]
        d, ids = np.asarray([x["delta"] for x in q]), [x["shot"] for x in q]
        blocks[role] = dict(
            windows=len(q),
            shots=len(set(ids)),
            common_rank_support=sum(x["rank"] >= 4 for x in q),
            active=sum(x["active"] for x in q),
            mean_delta=old.shot_average(d, ids),
            harms=np.asarray([x["harmful"] for x in q]).sum(0).tolist(),
            motion_vs_level=paired_shot_effect(d[:, 1] - d[:, 3], ids),
            motion_vs_reverse=paired_shot_effect(d[:, 1] - d[:, 2], ids),
            motion_vs_scalar=paired_shot_effect(d[:, 1] - d[:, 5], ids),
            median_abs_shape=float(np.median([x["shape_change_max"] for x in q])),
        )
    return dict(
        by_shot=rates,
        windows=blocks,
        disruptive=sum(x["disruptive"] for x in rows),
        safe=sum(not x["disruptive"] for x in rows),
    )


def run(root):
    h, start = Hashes(), time.monotonic()
    if root.exists():
        raise FileExistsError(root)
    for path in (
        PLAN,
        Path(__file__),
        Path(op.__file__),
        Path(feedback.__file__),
        Path(pair.__file__),
        PREVIOUS / "mechanism.json",
        PREVIOUS / "late_capacity.json",
    ):
        h(path)
    cfg, archived, _ = load_receipts(PREVIOUS, h)
    previous, inventory = bank.src.source_inputs(h)
    raw_sha = {x["shot"]: x["sha256"] for x in inventory}
    roles = defaultdict(list)
    for q in archived["cases"]:
        roles[q["shot"]].append(q)
    if len(roles) != 260 or not set(roles) <= set(previous["roles"]["train"]):
        raise ValueError("exact earlier role cohorts required")
    witness = {}
    for role in old.ROLES:
        sid = next(
            s
            for s, entries in roles.items()
            if s not in witness
            and any(q["role"] == role and q["eligible"][-1] >= 105 for q in entries)
        )
        witness[sid] = role
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if (
        r.state_digest(base) != cfg["source_state_sha256"]
        or parent["threshold"] != r.THRESHOLD
    ):
        raise ValueError("same qualified source and fixed threshold required")
    base.cuda()
    projection = op.basis(base)
    root.mkdir(parents=True, exist_ok=False)
    protocol = root / "protocol.json"
    r.write_new(
        protocol,
        dict(
            status="LOCKED_ADJACENT_PATCH_TRAIN",
            arms=ARMS,
            roles=list(roles),
            tau=TAU,
            radius=pair.RADIUS,
            guard=pair.LIMIT,
            no_east_val_test=True,
            source_state_sha256=cfg["source_state_sha256"],
            inputs={str(k): v[1] for k, v in h.cache.items()},
        ),
    )
    rows, tape, grouped = [], defaultdict(list), defaultdict(list)
    for j, sid in enumerate(roles, 1):
        p = r.io23.path_for("J-TEXT", sid)
        if h(p) != raw_sha[sid]:
            raise ValueError("known TRAIN raw bytes changed")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        pred = predict(base, projection, raw)
        for key in BLOCK_DATA:
            grouped[key].append(pred[key])
        tape["times"].append(raw["times"])
        tape["curve"].append(pred["curve"])
        row = dict(
            shot=sid,
            sha256=h(p),
            length=len(raw["times"]),
            blocks=len(pred["clock"]),
            native_error=pred["native_error"],
        )
        if sid in witness:
            row["prefix_check"] = dict(
                role=witness[sid], **check_prefix(base, projection, raw, pred)
            )
        if j == 1:
            row["batched_vjp_error"] = check_row_independence(base, projection, raw)
        rows.append(row)
        if j % 50 == 0 or j == len(roles):
            print(f"Adjacent patch causal replay {j}/{len(roles)}", flush=True)
    if r.state_digest(base) != cfg["source_state_sha256"] or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("frozen model changed")
    h.check()
    raw_path = root / "curves.npz"
    r.save_arrays(
        raw_path, **{k: np.concatenate(v) for k, v in {**tape, **grouped}.items()}
    )
    seal_path = root / "predictions_seal.json"
    r.write_new(
        seal_path,
        dict(protocol_sha256=h(protocol), curves_sha256=h(raw_path), shots=rows),
    )
    scored, window_rows, offset, k = [], [], 0, 0
    advance = r.simple_data.advanced_times()
    for i, q in enumerate(rows):
        sid, n, b = q["shot"], q["length"], q["blocks"]
        times, z = tape["times"][i], tape["curve"][i]
        pred = dict(curve=z, **{f: grouped[f][i] for f in BLOCK_DATA})
        if len(times) != n or len(pred["clock"]) != b:
            raise ValueError("sealed shot indexing failed")
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        horizon = r.io23.effective_horizon_ms(sid, advance)[0]
        x, block = score_shot(sid, times, pred, flag, down, horizon, roles[sid])
        scored.append(x)
        window_rows.extend(block)
        offset += n
        k += b
    stats = summarize(scored, window_rows)
    r.write_new(
        root / "result.json",
        dict(
            status="SCORED_TRAIN_DEVELOPMENT_ONLY",
            seal_sha256=h(seal_path),
            shots=scored,
            windows=window_rows,
            summary=stats,
            released_blocks=k,
            points=offset,
            max_trust_drift=float(np.max(np.concatenate(grouped["guard"]))),
            elapsed_seconds=time.monotonic() - start,
            val_test_east_reads=0,
        ),
    )
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
