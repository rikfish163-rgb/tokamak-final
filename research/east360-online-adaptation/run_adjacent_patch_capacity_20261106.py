"""Independently replay finite, outcome-aware late-shot capacity on CPU."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import run_adjacent_patch_intra_20261106 as causal
from inspect_peak_contrast_first_alarm_20261106 import first
from run_h25_late_capacity_20261106 import load_receipts
from verify_confirmation_raw_replay_20261105 import Hashes


r, old, op, pair, TAU = causal.r, causal.old, causal.op, causal.pair, causal.TAU
ARMS = (*causal.ARMS[1:5], "motion_full", "level_full")
MODES = (0, 0, 1, 1, 0, 1)


def adjusted_objective(z, jac, native_anchor):
    z, jac = z.copy(), jac.copy()
    z[3] -= z[2] - native_anchor
    jac[3] -= jac[2]
    return z, jac


def evaluator(base, packet, space, mode):
    q = packet.tokens.new_tensor(space)
    sign = q.new_tensor([1, 1, 1, 1, -1, -1, -1])[:, None]
    previous = cached = None

    def f(w):
        nonlocal previous, cached
        if previous is None or not np.array_equal(previous, w):
            v = (sign * (q @ q.new_tensor(w))).requires_grad_()
            z = op.forward(base, packet, v, mode)["logit"]
            (grad,) = torch.autograd.grad(z.sum(), v)
            cached = (
                z.detach().cpu().numpy().astype(np.float64),
                (sign * grad).detach().cpu().numpy().astype(np.float64) @ space,
            )
            previous = w.copy()
        return cached

    return f


@torch.no_grad()
def verify_state(base, packet, mode, state, native, times, down, past, block, eligible):
    pp, pb = packet.take(past), packet.take(block)
    u = pp.tokens.new_tensor(state)
    if not bool(torch.isfinite(u).all()) or float(u.norm()) > pair.RADIUS + 5e-7:
        return None
    z = [
        op.forward(base, pp, (a * u).expand(len(past), -1), mode)["logit"].cpu().numpy()
        for a in (1, -1)
    ]
    drift = float(np.abs(np.stack(z) - native[past]).max())
    if drift > pair.LIMIT:
        return None
    anchor = float(z[0][-1] - native[past[-1]])
    forward = (
        op.forward(base, pb, u.expand(len(block), -1), mode)["logit"].cpu().numpy()
    )
    edited, scalar = native.copy(), native.copy()
    edited[block], scalar[block] = forward, native[block] + anchor
    new = first(times, edited, True, down, TAU)
    counter = first(times, scalar, True, down, TAU)
    peak = float(edited[eligible].max())
    # Both traces are reconstructed only after release; first() uses all times.
    if peak >= TAU and (not new["timely"] or new["first"] not in block):
        raise ValueError("canonical crossing should reproduce a full first alarm")
    return dict(
        peak=peak,
        shape_peak=peak - anchor,
        anchor_shift=anchor,
        drift=drift,
        norm=float(u.norm()),
        state=u.cpu().numpy().tolist(),
        anchor_edit_rms=float(
            op.delta(pp.take([-1]), u[None], mode).square().sum(-1).mean().sqrt()
        ),
        full_first=new,
        equal_scalar_first=counter,
    )


def trial(base, packet, mode, q, g, target, native, times, down, past, block, eligible):
    target = int(target)
    pp = packet.take(np.r_[past, target, past])
    f = evaluator(base, pp, q, mode)
    z0, j0 = f(np.zeros(q.shape[1]))
    if np.abs(z0[:3] - native[past]).max() > 5e-5:
        raise ValueError("zero optimization state differs from the frozen model")
    direction = old.mix.projected(q, g, j0[3] @ q.T)
    initial, initial_score = np.zeros_like(direction), None
    for scale in pair.SCALES:
        v = q @ (scale * direction)
        score = verify_state(
            base, packet, mode, v, native, times, down, past, block, eligible
        )
        if score is not None and score["drift"] <= pair.LIMIT - pair.GUARD_HEADROOM:
            initial, initial_score = scale * direction, score
            break

    def shape(u):
        return adjusted_objective(*f(u), native[past[-1]])

    out = dict(target=target, initial=initial_score)
    for name, objective in (("raw", f), ("shape", shape)):
        coefficient, info = old.mix.optimize(objective, native[past], initial)
        result = verify_state(
            base,
            packet,
            mode,
            q @ coefficient,
            native,
            times,
            down,
            past,
            block,
            eligible,
        )
        out[name] = dict(solver=info, witness=result)
    return out


def replay_shot(base, projection, raw, down, horizon, cases):
    x, a = causal.bank.prior.views(base, raw)
    p = op.encode(base, x, a, projection)
    native = causal.bank.prior.scan(base, x, a).detach().cpu().numpy()
    times = raw["times"]
    if first(times, native, True, down, TAU)["timely"]:
        raise ValueError("baseline case must still be late")
    clock = np.asarray([q["maturity"] for q in cases], dtype=np.int64)
    g, history, support = causal.gradients(base, p, raw, clock)
    rows = []
    for i, q in enumerate(cases):
        t = int(clock[i])
        past = causal.bank.tr.reached_indices(clock[i : i + 1])[0]
        block = np.arange(t + 1, min(t + 33, len(times)))
        lead = ((down - times[block]) * 1000).round(9)
        eligible = block[(lead > 10) & (lead <= horizon + 1e-6)]
        if eligible.tolist() != q["eligible"]:
            raise ValueError("whole HDF5 canonical window differs")
        top = eligible[np.argsort(-native[eligible], kind="stable")[:3]]
        if not len(top) or int(top[0]) != q["target"]:
            raise ValueError("frozen peak ordering changed")
        qspaces, rank = pair.common_spaces(g[i])
        eye = np.eye(op.WIDTH, dtype=np.float64)
        fields = {}
        for k, name in enumerate(ARMS):
            q_basis = qspaces[k] if k < 4 else eye
            good = (
                bool(support[i] and rank >= 4)
                if k < 4
                else bool(p.support[t, MODES[k]])
            )
            fields[name] = (
                [
                    trial(
                        base,
                        p,
                        MODES[k],
                        q_basis,
                        history[i, MODES[k]],
                        x,
                        native,
                        times,
                        down,
                        past,
                        block,
                        eligible,
                    )
                    for x in top
                ]
                if good
                else []
            )
        rows.append(
            dict(
                shot=q["shot"],
                maturity=t,
                rank=rank,
                native_peak=float(native[eligible].max()),
                arms=fields,
            )
        )
    return rows


def summarize(rows):
    reachable = {}
    stats = {}
    for name in ARMS:
        actual, scalar, independent = set(), set(), set()
        success, feasible, count, best = 0, 0, 0, {}
        for q in rows:
            key = q["shot"]
            best[key] = max(best.get(key, -np.inf), q["native_peak"])
            for t in q["arms"][name]:
                for objective in ("raw", "shape"):
                    count += 1
                    info = t[objective]
                    success += info["solver"]["success"]
                    feasible += info["witness"] is not None
                for s in (t["initial"], t["raw"]["witness"], t["shape"]["witness"]):
                    if s is None:
                        continue
                    best[key] = max(best[key], s["peak"])
                    if s["equal_scalar_first"]["timely"]:
                        scalar.add(key)
                    if s["full_first"]["timely"] and s["peak"] >= TAU:
                        actual.add(key)
                        if not s["equal_scalar_first"]["timely"]:
                            independent.add(key)
        reachable[name] = sorted(actual)
        reachable[name + "_scalar"] = sorted(scalar)
        reachable[name + "_shape_only"] = sorted(independent)
        stats[name] = dict(
            count=count, successful=success, feasible=feasible, best=best
        )
    return reachable, stats


def gate(score, reach, old_reach):
    p = score["positive"]
    support = p["windows"] > 0 and p["common_rank_support"] * 10 >= p["windows"] * 7
    controls = (
        set(old_reach)
        | set(reach["level"])
        | set(reach["level_reverse"])
        | set(reach["motion_scalar"])
        | set(reach["motion_reverse"])
    )
    unique = sorted(set(reach["motion"]) & set(reach["motion_shape_only"]) - controls)
    contrasts = (p["motion_vs_level"], p["motion_vs_scalar"])
    no_more_harm = all(
        score[role]["harms"][1] <= min(score[role]["harms"][3], score[role]["harms"][5])
        for role in ("protect", "safe")
    )
    broad = (
        all(v["mean"] >= 0.005 and v["resample_range"][0] > 0 for v in contrasts)
        and no_more_harm
    )
    return dict(
        promote=bool(support and (unique or broad)),
        support=bool(support),
        unique_motion=unique,
        broad=bool(broad),
        no_more_harm=bool(no_more_harm),
    )


def run(out):
    if out.exists():
        raise FileExistsError(out)
    h, t = Hashes(), time.monotonic()
    for p in (
        Path(__file__),
        causal.PLAN,
        Path(causal.__file__),
        Path(op.__file__),
        Path(causal.feedback.__file__),
        Path(pair.__file__),
        causal.PREVIOUS / "mechanism.json",
        causal.PREVIOUS / "late_capacity.json",
        Path(old.mix.__file__),
    ):
        h(p)
    cfg, receipt, _ = load_receipts(causal.PREVIOUS, h)
    previous, inventory = causal.bank.src.source_inputs(h)
    ids = cfg["late"]
    if len(ids) != 6 or not set(ids) <= set(previous["roles"]["train"]):
        raise ValueError("exact earlier six TRAIN late shots required")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if (
        r.state_digest(base) != cfg["source_state_sha256"]
        or parent["threshold"] != r.THRESHOLD
    ):
        raise ValueError("same fixed source and threshold required")
    if old.mix.RADIUS != pair.RADIUS or old.mix.LIMIT != pair.LIMIT:
        raise ValueError("optimizer and nominal experiment limits differ")
    projection = op.basis(base)
    advance = r.simple_data.advanced_times()
    raw_sha = {q["shot"]: q["sha256"] for q in inventory}
    rows = []
    for sid in ids:
        p = r.io23.path_for("J-TEXT", sid)
        if h(p) != raw_sha[sid]:
            raise ValueError("raw case changed")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        if not flag:
            raise ValueError("expected a disruptive TRAIN case")
        h_ms = r.io23.effective_horizon_ms(sid, advance)[0]
        cases = [q for q in receipt["cases"] if q["shot"] == sid and q["late"]]
        rows.extend(replay_shot(base, projection, raw, down, h_ms, cases))
        print(
            f"Adjacent-patch finite capacity: {sid}, {len(cases)} windows", flush=True
        )
    if len(rows) != 9:
        raise ValueError("same nine canonical windows required")
    reach, stats = summarize(rows)
    if r.state_digest(base) != cfg["source_state_sha256"] or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("frozen model changed")
    h.check()
    r.write_new(
        out,
        dict(
            status="MEASURED_FINITE_TRAIN_CAPACITY_NOT_ONLINE",
            rows=rows,
            reachable=reach,
            statistics=stats,
            input_sha256={str(k): v[1] for k, v in h.cache.items()},
            elapsed_seconds=time.monotonic() - t,
            val_test_east_reads=0,
        ),
    )
    print(dict(reachable=reach, statistics=stats), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
