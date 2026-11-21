"""Offline exact-forward capacity of two matched patch-state teacher objectives."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import run_patch_utility_shadow_response_20261106 as old
from run_patch_utility_target_views_20261106 import mask
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import menu_analysis


b, m = old.b, old.m
PLAN = b.r.ROOT / "PATCH_SUSTAINED_MARGIN_PROTOCOL_20261106.md"
DELAY, WIDTH = old.DELAY, 32
SCALES = np.asarray(b.pair.SCALES)
BOUND = b.pair.LIMIT - b.pair.GUARD_HEADROOM
ARMS = ("peak", "sustained")
STRATA = ("late_disruption", "safe_no_alarm", "safe_alarm", "timely_disruption")


def timely_at(times, down):
    lead = (down - times) * 1000.0
    # Match the independent scorer's 9-decimal lead convention at the boundary.
    close = np.abs(lead - 10.0) <= 2e-9
    lead[close] = [round(float(x), 9) for x in lead[close]]
    return lead > 10.0


def target_frames(times, clock, down):
    grid = clock[:, None] + DELAY + 1 + np.arange(WIDTH - DELAY)
    valid = grid < len(times)
    if down is not None:
        valid &= timely_at(times, down)[np.minimum(grid, len(times) - 1)]
    return grid, valid


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-10)


def optimized_states(margin, gradient, guard, valid, radius):
    """Best affine frame and affine adjacent maximin in the same local ball."""
    _, s, vh = np.linalg.svd(np.asarray(guard, np.float64), full_matrices=False)
    q = vh * (s > s[:, :1] * 1e-6)[..., None]
    g = gradient - (gradient @ q.swapaxes(-1, -2)) @ q
    paired = valid[:, :-1] & valid[:, 1:]
    endpoint = np.pad(paired, ((0, 0), (0, 1))) | np.pad(paired, ((0, 0), (1, 0)))
    peak = radius * unit(g)
    peak_value = np.where(endpoint, margin + (peak * g).sum(axis=-1), -np.inf)

    # At an optimum of min(a+g·u, c+h·u), either one face is active and
    # at its sphere maximum, or both tie on (g-h)·u=c-a. Optimize on that disk.
    g1, g2 = g[:, :-1], g[:, 1:]
    a1, a2 = margin[:, :-1], margin[:, 1:]
    d = g1 - g2
    diff = np.linalg.norm(d, axis=-1)
    axis = unit(d)
    offset = np.divide(a2 - a1, diff, out=np.zeros_like(diff), where=diff > 1e-10)
    line = g1 - (g1 * axis).sum(-1, keepdims=True) * axis
    side = np.sqrt(np.maximum(0.0, radius**2 - offset**2))
    intersection = offset[..., None] * axis + side[..., None] * unit(line)
    pair_choices = np.stack((peak[:, :-1], peak[:, 1:], intersection), axis=2)
    pair_values = np.minimum(
        a1[..., None] + np.einsum("bfkd,bfd->bfk", pair_choices, g1),
        a2[..., None] + np.einsum("bfkd,bfd->bfk", pair_choices, g2),
    )
    pair_values[..., 2] = np.where((diff > 1e-10) & (np.abs(offset) <= radius),
                                  pair_values[..., 2], -np.inf)
    pair_values = np.where(paired[..., None], pair_values, -np.inf)
    rows = np.arange(len(margin))
    i = peak_value.argmax(axis=1)
    j, k = np.divmod(pair_values.reshape(len(margin), -1).argmax(axis=1), 3)
    support = paired.any(axis=1)
    states = np.stack((peak[rows, i], pair_choices[rows, j, k]), axis=1)
    frames = np.column_stack((i, j))
    return np.where(support[:, None, None], states, 0.0), np.where(support[:, None], frames, -1)


def common_scale(shifts):
    """One exact trust-region scale for both objectives, both signs, all clocks."""
    fits = (np.abs(shifts) <= BOUND).all(axis=(1, 2, 4))
    i = fits.argmax(axis=1)
    alive = fits.any(axis=1)
    chosen = shifts[np.arange(len(shifts)), :, :, i]
    return np.where(alive, SCALES[i], 0.0), np.where(alive[:, None, None, None], chosen, 0.0)


def identity_blocks(native, clock):
    ix = clock[:, None] + 1 + np.arange(WIDTH)
    return np.where(ix < len(native), native[np.minimum(ix, len(native) - 1)], 0.0)


def install_suffix(identity, suffix, clock, length, allowed):
    curves = np.broadcast_to(identity[None, :, None], (2, len(clock), 3, WIDTH)).copy()
    grid = clock[:, None] + DELAY + 1 + np.arange(WIDTH - DELAY)
    curves[:, :, 1:, DELAY:] = np.where(
        (allowed[:, None] & (grid < length))[None, :, None],
        suffix.transpose(1, 0, 2, 3),
        identity[None, :, None, DELAY:],
    )
    return curves


def cohort(shots, tau):
    groups = {k: [] for k in STRATA}
    for q in shots:
        before = q.teacher["baseline"]
        if q.disruptive:
            if before["timely"]:
                closest = float(q.native[timely_at(q.times, q.down)].max()) - tau
                groups["timely_disruption"].append((closest, q.sid, q))
            else:
                groups["late_disruption"].append((0.0, q.sid, q))
        else:
            label = "safe_alarm" if before["false_alarm"] else "safe_no_alarm"
            groups[label].append((abs(float(q.native.max()) - tau), q.sid, q))
    for role, n in zip(STRATA, (len(shots), 48, 24, 24), strict=True):
        for _, _, q in sorted(groups[role])[:n]:
            yield role, q


def release_mask(shot, clock):
    grid, valid = target_frames(shot.times, clock, shot.down if shot.disruptive else None)
    enough = (valid[:, :-1] & valid[:, 1:]).any(axis=1)
    peak = np.maximum.accumulate(shot.native).astype(np.float64)
    before = peak[np.minimum(clock + DELAY, len(peak) - 1)] < b.TAU
    return grid, valid, enough & before


def jacobian(base, packed, indices):
    output, gradients = [], []
    for j in range(0, len(indices), 512):
        p = packed.take(indices[j:j + 512])
        u = p.tokens.new_zeros((len(p.tokens), b.op.WIDTH), requires_grad=True)
        z = b.op.forward(base, p, u, 0)["logit"]
        (g,) = torch.autograd.grad(z.sum(), u)
        output.append(z.detach().cpu().numpy())
        gradients.append(g.detach().cpu().numpy())
    return np.concatenate(output), np.concatenate(gradients).astype(np.float64)


def exact_probe(base, projection, raw, q, clock, grid, valid):
    """Generate both hindsight actions and evaluate the actual network."""
    x, a = b.bank.prior.views(base, raw)
    p = b.op.encode(base, x, a, projection)
    anchors = np.column_stack((b.bank.tr.reached_indices(clock), clock + DELAY))
    indices = np.unique(np.concatenate((anchors.ravel(), grid[valid])))
    z, g = jacobian(base, p, indices)
    zero_error = float(np.max(np.abs(z - q.native[indices])))
    if zero_error > 5e-5 or not np.isfinite(g).all():
        raise ValueError(f"source zero-state trace or gradient mismatch on TRAIN {q.sid}")
    future = np.zeros((*grid.shape, b.op.WIDTH), dtype=np.float64)
    future[valid] = g[np.searchsorted(indices, grid[valid])]
    margin = q.native[np.minimum(grid, len(q.native) - 1)].astype(np.float64) - b.TAU
    directions, targets = optimized_states(margin, future, g[np.searchsorted(indices, anchors)],
                                           valid, b.pair.RADIUS)
    supported = (np.linalg.norm(directions, axis=2).min(axis=1) > 1e-7)
    supported &= p.take(clock).support[:, 0].cpu().numpy()
    n = len(clock)
    shift = np.zeros((n, 2, 2, len(SCALES), anchors.shape[1]), dtype=np.float32)
    sub = np.flatnonzero(supported)
    signed = np.stack((directions, -directions), axis=2)
    if len(sub):
        idx = np.broadcast_to(anchors[sub, None, None, None], (len(sub), *shift.shape[1:]))
        state = signed[sub, :, :, None, None, :] * SCALES[None, None, None, :, None, None]
        state = np.broadcast_to(state, (*idx.shape, b.op.WIDTH)).reshape(-1, b.op.WIDTH)
        evaluated = b.risk_in_batches(base, p, idx.ravel(), p.tokens.new_tensor(state))
        shift[sub] = evaluated.reshape(idx.shape) - q.native[idx]
    factor, guard = common_scale(shift)
    factor = np.where(supported, factor, 0.0)
    actual_state = (signed * factor[:, None, None, None]).astype(np.float32)
    allowed = factor > 0
    suffix = np.zeros((n, 2, 2, grid.shape[1]), dtype=np.float32)
    row, frame = np.nonzero(allowed[:, None] & (grid < len(q.native)))
    if len(row):
        idx = np.repeat(grid[row, frame], 4)
        state = actual_state[row].reshape(-1, b.op.WIDTH)
        new = b.risk_in_batches(base, p, idx, p.tokens.new_tensor(state))
        suffix[row, :, :, frame] = new.reshape(-1, 2, 2)
    return dict(curves=install_suffix(identity_blocks(q.native, clock), suffix, clock, len(q.native), allowed),
                states=actual_state[:, :, 0], scale=factor, targets=targets,
                guard=guard, zero_error=np.array(zero_error))


def empty_probe(native, clock):
    return dict(curves=np.broadcast_to(identity_blocks(native, clock)[None, :, None],
                                       (2, 0, 3, WIDTH)).copy(),
                states=np.zeros((0, 2, b.op.WIDTH), np.float32), scale=np.zeros(0),
                targets=np.zeros((0, 2), np.int64), guard=np.zeros((0, 2, 2, 4), np.float32),
                zero_error=np.array(0.0))


def score(q, clock, curves, allowed):
    y = menu_analysis(q, clock, curves, allowed, b.TAU)
    if y["baseline"] != q.teacher["baseline"]:
        raise ValueError("locked first alarm changed")
    timely_pair = np.zeros_like(allowed)
    if q.disruptive and not y["baseline"]["timely"]:
        _, valid = target_frames(q.times, clock, q.down)
        crossing = (curves[:, :, DELAY:].astype(np.float64) >= b.TAU) & valid[:, None, :]
        timely_pair = (crossing[:, :, :-1] & crossing[:, :, 1:]).any(axis=-1) & (y["hard"] > 0)
    return dict(good=bool((y["hard"] > 0).any()), bad=bool((y["hard"] < 0).any()),
                pair=bool(timely_pair.any()), good_actions=int((y["hard"] > 0).sum()),
                bad_actions=int((y["hard"] < 0).sum()), pair_actions=int(timely_pair.sum()))


def original_menu(tape, q):
    c, y, a = (tape.data[key][q.blocks] for key in ("clock", "curves", "allowed"))
    room = c + DELAY < len(q.native) - 1
    previous = np.maximum.accumulate(q.native)[c]
    return c, old.wait(y, DELAY), old.commit_mask(a, y[:, :, :DELAY], room, previous, b.TAU)


def effects(q, original, take, result):
    c, y, avail = original
    allowed = result["scale"] > 0
    support = np.column_stack((np.ones(len(allowed), bool), allowed, allowed))
    same = avail[take].copy()
    same[:, 1:] &= allowed[:, None]
    scores = {"old_full": score(q, c, y, avail), "old_matched": score(q, c[take], y[take], same)}
    for arm, curves in zip(ARMS, result["curves"], strict=True):
        scores[arm] = score(q, c[take], curves[:, :2], support[:, :2])
        scores[arm + "_signed"] = score(q, c[take], curves, support)
    return scores


def summarize(rows):
    keys = ("old_full", "old_matched", "peak", "sustained", "peak_signed", "sustained_signed")
    strata = {role: [q for q in rows if q["role"] == role] for role in STRATA}
    categories = dict(timely_recovered=("late_disruption", "good"), two_frames=("late_disruption", "pair"),
                      new_fp=("safe_no_alarm", "bad"), fp_removed=("safe_alarm", "good"),
                      timely_lost=("timely_disruption", "bad"))
    return dict(n={k: len(v) for k, v in strata.items()}, arms={arm: {
        name: sorted(q["shot"] for q in strata[role] if q["effects"][arm][key])
        for name, (role, key) in categories.items()} for arm in keys})


def decision(reports):
    def union(arm, key):
        return {sid for v in reports.values() for sid in v["arms"][arm][key]}

    novel = union("sustained", "two_frames") - union("peak", "timely_recovered") - union("old_matched", "timely_recovered")
    fp = {sid for v in reports.values() for sid in set(v["arms"]["sustained"]["new_fp"]) - set(v["arms"]["peak"]["new_fp"])}
    lost = {sid for v in reports.values() for sid in set(v["arms"]["sustained"]["timely_lost"]) - set(v["arms"]["peak"]["timely_lost"])}
    return dict(new_sustained_timely=sorted(novel), new_peak_timely=sorted(union("peak", "timely_recovered")
                - union("old_full", "timely_recovered")), additional_fp=sorted(fp), additional_lost=sorted(lost),
                sustained_gate=len(novel) >= 2 and not fp and not lost)


def run(native, masked, prior, out):
    if out.exists():
        raise FileExistsError(out)
    started, h = time.monotonic(), Hashes()
    tapes, ids, _ = m.load(native, masked, prior, h)
    b.r.configure()
    base, source = b.r.load_parent(b.r.DEFAULT_PARENT)
    sha = b.r.state_digest(base)
    if sha != b.r.read(native / "protocol.json")["source_state_sha256"] or source["threshold"] != b.r.THRESHOLD:
        raise ValueError("frozen source checkpoint or threshold mismatch")
    base.cuda()
    projection = b.op.basis(base)
    cases = {name: list(cohort(tape.shots, b.TAU)) for name, tape in tapes.items()}
    out.mkdir(parents=True, exist_ok=False)
    b.r.write_new(out / "protocol.json", dict(status="HINDSIGHT_TRAIN_ONLY", tau=b.TAU,
        arms=ARMS, bound=BOUND, norm_cap=b.pair.RADIUS, delay=DELAY, width=WIDTH, source_state=sha,
        plan_sha256=h(PLAN), implementation_sha256=h(Path(__file__)), old_bank_seal=h(native / "seal.json"),
        mask_bank_receipt=h(masked / "completed.json"), source_count=len(ids),
        cases={k: [{"shot": q.sid, "role": role} for role, q in rows] for k, rows in cases.items()}))
    reports, rows = {}, {}
    for view, tape in tapes.items():
        (out / view).mkdir()
        rows[view] = []
        for i, (role, q) in enumerate(cases[view], 1):
            orig = original_menu(tape, q)
            clock = orig[0]
            grid, valid, take = release_mask(q, clock)
            c = clock[take]
            if len(c):
                raw = b.r.simple_data.read_shot("J-TEXT", q.sid)
                if not np.array_equal(raw["times"], q.times):
                    raise ValueError(f"TRAIN time grid changed for {q.sid}")
                p = exact_probe(base, projection, raw if view == "native" else mask(raw, view),
                                q, c, grid[take], valid[take])
            else:
                p = empty_probe(q.native, c)
            file = out / view / f"{q.sid}.npz"
            b.r.save_arrays(file, clock=c, times=q.times, native=q.native, **p)
            rows[view].append(dict(shot=q.sid, role=role, disruptive=q.disruptive, down=q.down,
                blocks=len(c), allowed=int((p["scale"] > 0).sum()), effects=effects(q, orig, take, p)))
            if i % 20 == 0 or i == len(cases[view]):
                print(f"{view}: {i}/{len(cases[view])} TRAIN shot-views", flush=True)
        reports[view] = summarize(rows[view])
        print(view, {k: {c: len(s) for c, s in a.items()} for k, a in reports[view]["arms"].items()}, flush=True)
    if b.r.state_digest(base) != sha or any(p.grad is not None for p in base.parameters()):
        raise ValueError("frozen source state changed")
    b.r.write_new(out / "result.json", dict(status="SOURCE_CAPACITY_NOT_ONLINE", reports=reports,
        rows=rows, decision=decision(reports), new_val_test_east_reads=0, duration=time.monotonic() - started))
    print(decision(reports), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("native", "masked", "prior", "out"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    run(a.native, a.masked, a.prior, a.out)
