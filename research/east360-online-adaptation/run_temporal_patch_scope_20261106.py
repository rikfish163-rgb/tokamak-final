"""Paired source hindsight test of where in time a temporary token write acts."""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

import run_joint_paired_direction_20261106 as prior


b = prior.b
SCOPES = ("full", "recent", "fade")
MODES = (*SCOPES, "recent_energy", "fade_energy")
KEYS = ("danger", "paired", "neural_only", "scalar_any", "safe_reject")


def temporal_weights(end, release, scope):
    if scope == "full":
        return np.ones((len(end), 16))
    if scope not in ("recent", "fade"):
        raise ValueError(f"unknown temporal scope: {scope}")
    age = np.maximum(np.asarray(end) - release, 0) if scope == "fade" else np.zeros(len(end))
    newest = np.clip((32. - age[:, None] - 8. * np.arange(4)) / 8., 0., 1.)
    return np.pad(newest, ((0, 0), (12, 0)))


def energy_factor(features, state, weights):
    """Same state spread over all data tokens, with the local write's L2 norm."""
    v = np.einsum("epr,...rs->...eps", features, state.reshape(*state.shape[:-1], 8, 8))
    patch = np.square(v).sum(-1)
    total = patch.sum(-1)
    local = (patch * np.square(weights)).sum(-1)
    return np.sqrt(np.divide(local, total, out=np.zeros_like(total), where=total > 0.))


def first_spliced(times, native, indices, changed, *, down, tau):
    eligible = np.ones(len(native), dtype=bool) if down is None else (down-times)*1000 >= -1e-6
    outside = eligible & (native.astype(np.float64) >= tau)
    outside[indices] = False
    old = np.flatnonzero(outside)
    cross = (changed.astype(np.float64) >= tau) & eligible[indices]
    first = np.where(cross.any(-1), indices[cross.argmax(-1)], len(native))
    first = np.minimum(first, old[0] if len(old) else len(native))
    first = np.where(first == len(native), -1, first)
    lead = None if down is None else np.where(
        first >= 0, np.round((down - times[np.maximum(first, 0)]) * 1000, 9), np.nan,
    )
    return first, lead


def summarize(records, archived_full):
    by_view = {view: {mode: {key: sorted({
        r["shot"] for r in rows if r["exact"][mode][key]
    }) for key in KEYS} for mode in MODES} for view, rows in records.items()}

    def ids(views, modes, key):
        return {shot for v in views for m in modes for shot in by_view[v][m][key]}

    original = [v for v in records if v in prior.teaching.m.VIEWS]

    def strict(views, mode):
        return ids(views, [mode], "neural_only") - ids(views, MODES, "scalar_any")

    outcome = {}
    for mode in MODES:
        real = strict(original, mode)
        outcome[mode] = dict(original_strict=sorted(real), all_strict=sorted(strict(records, mode)))
        if mode in ("recent", "fade"):
            new = real - archived_full - ids(original, ("full", f"{mode}_energy"), "paired")
            outcome[mode].update(original_local_only=sorted(new), promote=len(real) >= 3 and bool(new))
    return by_view, outcome


def weighted(p, weights):
    return replace(p, features=p.features * p.features.new_tensor(weights)[:, None, :, None])


def packed(base, basis, view, shot, gain, idx):
    x, a = b.bank.prior.views(base, prior.teaching.source_raw(view, shot, gain))
    x, a = x[idx].contiguous(), a[idx].contiguous()
    return x, a, b.op.encode(base, x, a, basis)


def fit(base, d, s, weights, native, canonical, early, timely):
    indices = [np.arange(len(p.tokens)) for p in (d, s)]
    target = np.flatnonzero(canonical) + 4
    states, ub = [], []
    for name in SCOPES:
        pd, ps = (weighted(p, w) for p, w in zip((d, s), weights[name], strict=True))
        (zd, gd), (zs, gs) = (prior.jacobian(base, p, i, "input")
                               for p, i in zip((pd, ps), indices, strict=True))
        if name == "full":
            for z, frozen in zip((zd, zs), native, strict=True):
                if (np.max(np.abs(z - frozen)) > 5e-5 or
                        not np.array_equal(z.astype(float) >= b.TAU, frozen.astype(float) >= b.TAU)):
                    raise ValueError("zero state disagrees with the archived checkpoint")
        a, cap = prior.inequalities(gd, gs, *native, early, timely, b.TAU, paired=True)
        solved = [prior.solve(gd[k], a, cap, prior.RADIUS) for k in target]
        u, upper = map(np.stack, zip(*solved, strict=True))
        gap = upper - np.einsum("kd,kd->k", gd[target], u)
        if gap.min() < -1e-7 or (a @ u.T - cap[:, None]).max() > 1e-7:
            raise ValueError("invalid paired linear solution")
        states.append(u)
        ub.append(native[0][target] + upper - b.TAU)
    return prior.matched_grid(np.stack(states, axis=1)), np.stack(ub, axis=1)


def exact(base, p, grid, factor=None):
    def logits(q, u):
        n = len(q.tokens)
        idx = np.tile(np.arange(n), u.shape[0] * u.shape[1])
        return prior.placement.scan(base, q, idx, q.tokens.new_tensor(u.reshape(-1, b.op.WIDTH)),
                                    "input").reshape(u.shape[:-1])

    u = np.broadcast_to(grid[:, :, None, :], (*grid.shape[:2], len(p.tokens), b.op.WIDTH))
    if factor is not None:
        u = u * factor[..., None]
    return logits(p, u), logits(p.take(slice(0, 4)), -u[:, :, :4])


def score(f, horizon, down, zd, zs, md, ms, grid):
    dn, sn, dt, st, di, si = (f[k] for k in
        ("danger_native", "safe_native", "danger_time", "safe_time", "di", "si"))
    dg = np.maximum(abs(zd[..., :4] - dn[di[:4]]), abs(md - dn[di[:4]])).max(-1)
    sg = np.maximum(abs(zs[..., :4] - sn[si[:4]]), abs(ms - sn[si[:4]])).max(-1)
    shift = zd[..., 3] - dn[di[3]]
    _, lead = first_spliced(dt, dn, di[4:], zd[..., 4:], down=down, tau=b.TAU)
    _, scalar_lead = first_spliced(dt, dn, di[4:], dn[di[4:]] + shift[..., None],
                                  down=down, tau=b.TAU)
    safe_first, _ = first_spliced(st, sn, si[4:], zs[..., 4:], down=None, tau=b.TAU)
    safe = (sg <= prior.BOUND) & (safe_first < 0)
    danger = (dg <= prior.BOUND) & (lead > 10.) & (lead <= horizon + 1e-6)
    scalar = scalar_lead > 10.
    neural = danger & safe & ~scalar
    result = dict(zip(KEYS, map(lambda a: int(a.sum()),
        (danger, danger & safe, neural, (dg <= prior.BOUND) & safe & scalar, danger & ~safe)),
        strict=True))
    result["witness"] = None
    if neural.any():
        ix = np.unravel_index(np.where(neural, lead, -np.inf).argmax(), neural.shape)
        result["witness"] = dict(idx=list(map(int, ix)), lead_ms=float(lead[ix]),
            danger_guard=float(dg[ix]), safe_guard=float(sg[ix]), anchor_delta=float(shift[ix]),
            norm=float(np.linalg.norm(grid[ix])))
    return result


def probe(base, basis, view, gain, pair, f, out, advance):
    did, sid = pair["shot"], pair["safe"]
    disrupted, down = b.r.io23.read_outcome("J-TEXT", did)
    if not disrupted or b.r.io23.read_outcome("J-TEXT", sid)[0]:
        raise ValueError("wrong source pair outcomes")
    horizon = b.r.io23.effective_horizon_ms(did, advance)[0]
    dc, sc = pair["clock"], pair["safe_clock"]
    di, si = f["di"], f["si"]
    idx, canonical, early, timely = prior.future_indices(f["danger_time"], dc, down, horizon)
    np.testing.assert_array_equal(di, np.r_[prior.history(dc), idx])
    np.testing.assert_array_equal(si, np.r_[prior.history(sc), np.arange(sc + 9, sc + 33)])
    np.testing.assert_array_equal(canonical, f["canon"])
    if not canonical.any() or int(canonical.sum()) != pair["canonical_frames"]:
        raise ValueError("canonical source release changed")
    _, _, pd = packed(base, basis, view, did, gain, di)
    _, _, ps = packed(base, basis, view, sid, gain, si)
    if not (bool(pd.support[3, 0]) and bool(ps.support[3, 0])):
        raise ValueError("archived release no longer has observed motion support")
    weights = {m: (temporal_weights(di, dc + 8, m), temporal_weights(si, sc + 8, m))
               for m in SCOPES}
    grid, ub = fit(base, pd, ps, weights, (f["danger_native"][di], f["safe_native"][si]),
                   canonical, early, timely)
    result, arrays = {}, dict(grid=grid, linear_upper_margin=ub)
    for mode in MODES:
        scope = mode.removesuffix("_energy")
        u = grid[:, SCOPES.index(scope)]
        values = []
        for role, p, w in zip(("danger", "safe"), (pd, ps), weights[scope], strict=True):
            if mode == scope:
                z, minus = exact(base, weighted(p, w), u)
            else:
                factor = energy_factor(p.features[:, 0].cpu().numpy().astype(np.float64), u, w)
                z, minus = exact(base, p, u, factor)
                arrays[f"{role}_{mode}_factor"] = factor
            values.append((z, minus))
            arrays[f"{role}_{mode}"] = z
            arrays[f"{role}_{mode}_minus"] = minus
        (zd, md), (zs, ms) = values
        result[mode] = score(f, horizon, down, zd, zs, md, ms, u)
    b.r.save_arrays(out, **arrays)
    zero = np.linalg.norm(grid[:, 0, 0], axis=-1) <= 1e-12
    return dict(pair, exact=result, horizon=horizon, target_count=len(grid), zero_targets=int(zero.sum()))


def run(previous, stress, output):
    start = time.monotonic()
    protocol, old, verified = (b.r.read(previous/f"{name}.json")
                               for name in ("protocol", "result", "verification"))
    if (verified["status"] != "INDEPENDENT_FULL_TRACE_AND_HOOK_MATCH" or verified["records"] != 48
            or sum(map(len, protocol["matched"].values())) != 48 or protocol["tau"] != b.TAU):
        raise ValueError("the independently verified 48-pair source archive is required")
    gains = b.r.read(stress/"donor_waveform_receipt.json")["profiles"]
    advance, train = b.r.simple_data.advanced_times(), set(b.r.simple_data.source_roles()["train"])
    shots = {p[key] for pairs in protocol["matched"].values() for p in pairs for key in ("shot", "safe")}
    if not shots <= train:
        raise ValueError("paired experiment may only read raw TRAIN shots")
    b.r.configure()
    base, parent = b.r.load_parent(b.r.DEFAULT_PARENT)
    if b.r.state_digest(base) != protocol["source_state"] or parent["threshold"] != b.r.THRESHOLD:
        raise ValueError("source checkpoint or threshold changed")
    base.cuda()
    basis = b.op.basis(base)
    output.mkdir(parents=True, exist_ok=False)
    b.r.write_new(output/"protocol.json", dict(
        status="TRAIN_HINDSIGHT_TEMPORAL_SCOPE", previous=str(previous), stress=str(stress),
        source_state=protocol["source_state"], tau=b.TAU, scopes=SCOPES, modes=MODES,
        radius=prior.RADIUS, scales=prior.SCALES.tolist(), bound=prior.BOUND,
        new_val_test_east_signal_or_outcome_reads=0,
    ))
    rows = {}
    for view, pairs in protocol["matched"].items():
        (output/view).mkdir()
        gain = gains[view]["gain"] if view in gains else None
        rows[view] = []
        for pair in pairs:
            name = f"{pair['shot']}_{pair['clock']}.npz"
            with np.load(previous/view/name, allow_pickle=False) as f:
                rows[view].append(probe(base, basis, view, gain, pair, f, output/view/name, advance))
        print(view, len(rows[view]), {m: sum(r["exact"][m]["paired"] for r in rows[view])
                                    for m in MODES}, flush=True)
    archived = {sid for v in prior.teaching.m.VIEWS
                for sid in old["views"][v]["counts"]["input"]["paired"]["paired"]}
    counts, decision = summarize(rows, archived)
    if any(p.grad is not None for p in base.parameters()):
        raise ValueError("source model was not frozen")
    b.r.write_new(output/"result.json", dict(status="TEMPORAL_SCOPE_CAPACITY_NOT_ONLINE",
        rows=rows, counts=counts, decision=decision, archived_full_original_paired=sorted(archived),
        elapsed=time.monotonic()-start))
    print(decision, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "stress", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.previous, a.stress, a.output)
