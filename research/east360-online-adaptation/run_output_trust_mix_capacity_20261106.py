"""Exact prospective-output constraints, offline oracle over the six TRAIN misses."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import minimize

import run_output_trust_table_20261106 as bank
from inspect_peak_contrast_first_alarm_20261106 import first
from verify_confirmation_raw_replay_20261105 import Hashes


RADIUS, LIMIT = bank.SITE_NORM, bank.tr.LIMIT
PLAN = bank.r.ROOT / "OUTPUT_TRUST_MIX_PROTOCOL_20261106.md"


def orthogonal_columns(rows):
    _, s, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
    rank = np.count_nonzero(s > max(s[0] * 1e-6, 1e-10)) if len(s) else 0
    return vh[:rank].T


def projected(q, jac, g):
    v = q.T @ g
    active = orthogonal_columns(jac @ q)
    v = v - active @ (active.T @ v)
    length = np.linalg.norm(v)
    return RADIUS * v / length if length > 1e-10 else np.zeros_like(v)


def optimize(evaluate, native, seed):
    ids = np.array([0, 1, 2, 4, 5, 6])
    target = np.tile(native, 2)
    reduced = (
        LIMIT - 2e-5
    )  # keep solver round-off away from the direct-forward acceptance edge

    def constraint(w):
        z, _ = evaluate(w)
        d = z[ids] - target
        return np.r_[reduced - d, reduced + d, RADIUS**2 - w @ w]

    def jacobian(w):
        _, j = evaluate(w)
        return np.vstack((-j[ids], j[ids], -2 * w))

    fit = minimize(
        lambda w: -evaluate(w)[0][3],
        seed,
        jac=lambda w: -evaluate(w)[1][3],
        method="SLSQP",
        bounds=[(-RADIUS, RADIUS)] * len(seed),
        constraints=[dict(type="ineq", fun=constraint, jac=jacobian)],
        options={"maxiter": 35, "ftol": 2e-7},
    )
    return fit.x, dict(
        success=bool(fit.success), iterations=int(fit.nit), status=int(fit.status)
    )


def gradient(base, x, a):
    u = x.new_zeros((len(x), bank.old.WIDTH), requires_grad=True)
    z = bank.old.risk(base, x, a, u)
    (g,) = torch.autograd.grad(z.sum(), u)
    return g.detach().cpu().numpy().astype(np.float64)


def evaluator(base, x, a, q):
    h = bank.fwd.encode(base, x, a)
    q_tensor = torch.as_tensor(q, device=x.device, dtype=x.dtype)
    sign = x.new_tensor([1, 1, 1, 1, -1, -1, -1])[:, None]
    previous = None
    cached = None

    def evaluate(w):
        nonlocal previous, cached
        if previous is None or not np.array_equal(previous, w):
            u = (
                (sign * (q_tensor @ torch.as_tensor(w, device=x.device, dtype=x.dtype)))
                .detach()
                .requires_grad_()
            )
            z = bank.fwd.packed_risk(base, h, u)
            (g,) = torch.autograd.grad(z.sum(), u)
            j = (sign * g).detach().cpu().numpy().astype(np.float64) @ q
            cached = (z.detach().cpu().numpy().astype(np.float64), j)
            previous = w.copy()
        return cached

    return evaluate


@torch.no_grad()
def verify_state(base, xp, ap, xq, aq, before, state):
    state = torch.as_tensor(state, dtype=xp.dtype, device=xp.device)
    if not torch.isfinite(state).all() or float(state.norm()) > RADIUS + 5e-7:
        return None
    plus = bank.old.risk(base, xp, ap, state.expand(len(xp), -1)).cpu().numpy()
    minus = bank.old.risk(base, xp, ap, -state.expand(len(xp), -1)).cpu().numpy()
    drift = float(np.abs(np.stack((plus, minus)) - before).max())
    if drift > LIMIT:
        return None
    future = bank.old.risk(base, xq, aq, state.expand(len(xq), -1)).cpu().numpy()
    return dict(
        peak=float(future.max()),
        reached_drift=drift,
        norm=float(state.norm()),
        state=state.cpu().numpy().tolist(),
    )


def observed_projection(base, x, a, past, positive, b, reached, native):
    """Generate each direction without the future gradient or future risk."""
    q = orthogonal_columns(b)
    if not q.shape[1]:
        return []
    raw = np.stack(
        [q @ projected(q, reached, b[list(axes)].sum(0)) for _, axes in bank.fwd.AXES]
    )
    vector = np.stack((raw, -raw), axis=1).reshape(-1, bank.old.WIDTH)
    u = torch.as_tensor(vector, dtype=x.dtype, device=x.device)
    scale = u.new_tensor(bank.tr.SCALES)
    shape = (len(u), len(scale), bank.old.WIDTH)
    states = (u[:, None] * scale[None, :, None]).reshape(-1, bank.old.WIDTH)
    xp, ap = x[past].contiguous(), a[past].contiguous()
    xq, aq = x[positive].contiguous(), a[positive].contiguous()
    effects = (
        bank.fwd.batched_risk(
            base, bank.fwd.encode(base, xp, ap), states[None].expand(3, -1, -1)
        )
        .reshape(3, *shape[:2])
        .permute(1, 2, 0)
    )
    shift = effects - torch.as_tensor(native[past], dtype=x.dtype, device=x.device)
    radius, _ = bank.tr.select_symmetric(
        shift[None], (u.norm(dim=-1) > 1e-6)[None], limit=LIMIT - 2e-5
    )
    found = []
    for k in np.flatnonzero(radius[0].cpu().numpy()):
        v = verify_state(base, xp, ap, xq, aq, native[past], u[k] * radius[0, k])
        if v is not None:
            found.append(dict(arm=bank.ARMS[k], radius=float(radius[0, k]), **v))
    return found


def candidates(base, x, a, past, positive, q, w, reached_grad, future_grad, native):
    xp, ap = x[past].contiguous(), a[past].contiguous()
    xq, aq = x[positive].contiguous(), a[positive].contiguous()
    xv, av = x[np.r_[past, w, past]].contiguous(), a[np.r_[past, w, past]].contiguous()
    evaluate = evaluator(base, xv, av, q)
    direction = projected(q, reached_grad, future_grad)
    init = np.zeros_like(direction)
    trial = None
    for scale in bank.tr.SCALES:
        vector = q @ (direction * scale)
        score = verify_state(base, xp, ap, xq, aq, native[past], vector)
        if score is not None and score["reached_drift"] <= LIMIT - 2e-5:
            init, trial = direction * scale, score
            break
    coeff, convergence = optimize(evaluate, native[past], init)
    optimum = verify_state(base, xp, ap, xq, aq, native[past], q @ coeff)
    return dict(
        target=int(w),
        rank=q.shape[1],
        projected=trial,
        optimized=optimum,
        solver=convergence,
    )


def run(root, output):
    h, start = Hashes(), time.monotonic()
    r, prior = bank.r, bank.prior
    if output.exists():
        raise FileExistsError(output)
    for p in (
        PLAN,
        Path(__file__),
        root / "train_crossfit.json",
        root / "fit_completed.json",
    ):
        h(p)
    cfg, original = bank.src.source_inputs(h)
    fit = r.read(root / "fit_completed.json")
    if h(root / "train_crossfit.json") != fit["train_audit_sha256"]:
        raise ValueError("locked TRAIN late cases required")
    info = r.read(root / "train_crossfit.json")["geometry"]
    ids = info["non_timely_shots"]
    index = {q["shot"]: q["sha256"] for q in original}
    if len(ids) != 6 or not set(ids) <= set(cfg["roles"]["train"]):
        raise ValueError("expected original six TRAIN late cases")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    fits = bank.frozen_fits(base)
    adv = r.simple_data.advanced_times()
    rows = []
    for sid in ids:
        if h(r.io23.path_for("J-TEXT", sid)) != index[sid]:
            raise ValueError("raw late case changed")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        x, a = prior.views(base, raw)
        native = prior.scan(base, x, a).numpy()
        if not flag or first(raw["times"], native, flag, down, bank.TAU)["timely"]:
            raise ValueError("recomputed native late outcome differs")
        schedule = prior.signal_schedule(base, raw)
        m = schedule["maturity"]
        horizon = r.io23.effective_horizon_ms(sid, adv)[0]
        pos, _, _ = bank.fwd.teaching_masks(
            raw["times"], native, m, True, down, horizon, bank.TAU
        )
        for j in np.flatnonzero(pos.any(1) & schedule["support"]):
            k = int(m[j])
            o, past = int(schedule["origin"][j]), np.array([k - 16, k - 8, k])
            positive = k + 1 + np.flatnonzero(pos[j])
            top = positive[np.argsort(-native[positive], kind="stable")[:3]]
            reached = gradient(base, x[past].contiguous(), a[past].contiguous())
            q_grad = gradient(base, x[top].contiguous(), a[top].contiguous())
            for seed, (dec, mix, norm, ref) in zip(bank.SEEDS, fits, strict=True):
                target = prior.normalize_targets(
                    torch.from_numpy(schedule["delta"][j : j + 1]), norm
                )
                valid = torch.from_numpy(schedule["valid"][j : j + 1])
                _, basis, _, common = bank.rule.reached_bank(
                    base,
                    dec,
                    mix,
                    x[o : o + 1].contiguous(),
                    a[o : o + 1].contiguous(),
                    x[k : k + 1].contiguous(),
                    a[k : k + 1].contiguous(),
                    target,
                    valid,
                    ref,
                )
                if not bool(common[0]):
                    raise ValueError(
                        "one original late block has no common diagnostic support"
                    )
                b = basis[0].cpu().numpy().astype(np.float64)
                trial = {}
                observed = observed_projection(
                    base, x, a, past, positive, b, reached, native
                )
                for name in ("bank_mix", "oracle_enriched"):
                    values = []
                    for w, g in zip(top, q_grad, strict=True):
                        space = b if name == "bank_mix" else np.vstack((b, reached, g))
                        q = orthogonal_columns(space)
                        if q.shape[1]:
                            values.append(
                                candidates(
                                    base,
                                    x,
                                    a,
                                    past,
                                    positive,
                                    q,
                                    int(w),
                                    reached,
                                    g,
                                    native,
                                )
                            )
                    trial[name] = values
                rows.append(
                    dict(
                        shot=sid,
                        seed=seed,
                        maturity=k,
                        native_peak=float(native[positive].max()),
                        trials=trial,
                        observed_projection=observed,
                    )
                )
        print("continuous capacity", sid, flush=True)
    results = {}
    for name in ("bank_mix", "oracle_enriched"):
        best = {}
        for q in rows:
            scores = [
                v["peak"]
                for t in q["trials"][name]
                for v in (t["projected"], t["optimized"])
                if v is not None
            ]
            key = (q["shot"], q["seed"])
            best[key] = max(best.get(key, -np.inf), q["native_peak"], *scores)
        results[name] = [
            dict(
                seed=seed,
                reachable=[
                    sid for sid in ids if best.get((sid, seed), -np.inf) >= bank.TAU
                ],
                peak_by_shot={
                    str(sid): best[(sid, seed)] for sid in ids if (sid, seed) in best
                },
            )
            for seed in bank.SEEDS
        ]
    results["observed_projection"] = []
    for seed in bank.SEEDS:
        peaks = {
            str(sid): max(
                score
                for q in rows
                if q["shot"] == sid and q["seed"] == seed
                for score in (
                    q["native_peak"],
                    *(x["peak"] for x in q["observed_projection"]),
                )
            )
            for sid in ids
        }
        results["observed_projection"].append(
            dict(
                seed=seed,
                reachable=[sid for sid in ids if peaks[str(sid)] >= bank.TAU],
                peak_by_shot=peaks,
            )
        )
    h.check()
    r.write_new(
        output,
        dict(
            status="TRAIN_LATE_CONTINUOUS_ORACLE_NOT_POLICY",
            results=results,
            rows=rows,
            axis_reachable=info["late_reachable_by_mode_seed"],
            val_test_east_reads=0,
            input_sha256={
                str(p.resolve()): h(p)
                for p in (PLAN, Path(__file__), root / "train_crossfit.json")
            },
            seconds=time.monotonic() - start,
        ),
    )
    print(results, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.root, a.output)
