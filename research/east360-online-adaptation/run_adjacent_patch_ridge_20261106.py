"""Causal pointwise-residual directions vs matched aggregate H25 directions."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import run_adjacent_patch_intra_20261106 as a
import run_adjacent_patch_observability_20261106 as observe
from inspect_peak_contrast_first_alarm_20261106 import first
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = a.r.ROOT / "ADJACENT_PATCH_RIDGE_PROTOCOL_20261106.md"
MODES = a.MODES * 2
NAMES = (
    "static",
    "aggregate_motion",
    "aggregate_motion_reverse",
    "aggregate_level",
    "aggregate_level_reverse",
    "ridge_motion",
    "ridge_motion_reverse",
    "ridge_level",
    "ridge_level_reverse",
    "ridge_motion_scalar",
)


def ridge_directions(j, residual, projector):
    u, s, vh = np.linalg.svd(j @ projector, full_matrices=False)
    if not len(s) or s[0] <= 1e-10:
        return np.zeros((len(residual), j.shape[1]), dtype=np.float64)
    coeff = (residual @ u) * (s / (s * s + 0.01 * s[0] ** 2))
    v = coeff @ vh @ projector
    d = np.linalg.norm(v, axis=1)
    return np.divide(
        a.pair.RADIUS * v, d[:, None], where=d[:, None] > 1e-10, out=np.zeros_like(v)
    )


def residuals(base, p, raw, m):
    raw_y, ok = a.old.h25.arrived_targets(
        raw["values"],
        raw["available"],
        np.array([m - 32]),
        np.array([m]),
    )
    ok &= np.isfinite(raw_y)
    channels = np.zeros(23, dtype=bool)
    for _, c in a.feedback.GROUPS:
        channels[list(c)] = True
    ok &= ((ok.sum(1) >= 13) & channels)[:, None, :]
    y, good = (
        torch.as_tensor(raw_y, device=base.center.device),
        torch.as_tensor(ok, device=base.center.device),
    )
    rev = a.feedback._reverse_observed(y, good)
    target = torch.stack(
        [
            torch.asinh(
                (torch.where(good, value, base.center) - base.center) / base.scale
            )[0]
            for value in (y, rev)
        ]
    )
    with torch.no_grad():
        base_f = a.op.forward(
            base, p, p.tokens.new_zeros((1, a.op.WIDTH)), 0, forecast=True
        )["forecast"][0]
    mask = ok[0, 8:] & ok[0, :-8]
    w = np.zeros_like(mask, dtype=np.float64)
    for _, c in a.feedback.GROUPS:
        for sl in a.feedback.SEGMENTS:
            z = mask[sl, list(c)]
            if z.any():
                w[sl, list(c)] = z / np.sqrt(z.sum())
    gap = ((target[:, 8:] - target[:, :-8]) - (base_f[8:] - base_f[:-8])).clamp(-1, 1)
    return gap[:, mask].cpu().numpy(), w[mask], mask


@torch.no_grad()
def match_and_guard(base, p, clock, vectors, native):
    pair, op = a.pair, a.op
    u = p.tokens.new_tensor(vectors)
    pm = p.take(clock)
    rms = np.stack(
        [
            op.delta(pm, u[:, k], mode).square().sum(-1).mean(-1).sqrt().cpu().numpy()
            for k, mode in enumerate(MODES)
        ],
        axis=1,
    )
    support = (rms > 1e-7).all(1)
    matched = np.divide(
        rms.min(1)[:, None], rms, out=np.zeros_like(rms), where=support[:, None]
    )
    u = u * u.new_tensor(matched)[:, :, None]
    past = a.bank.tr.reached_indices(clock)
    grid = u.new_tensor(pair.SCALES)
    sign = u.new_tensor([1.0, -1.0])
    pp = p.take(
        np.broadcast_to(past[:, None, None], (len(clock), len(grid), 2, 3)).ravel()
    )
    shifts = []
    for k, mode in enumerate(MODES):
        v = (
            u[:, k, None, None, None]
            * grid[None, :, None, None, None]
            * sign[None, None, :, None, None]
        )
        z = op.forward(
            base, pp, v.expand(-1, -1, -1, 3, -1).reshape(-1, a.op.WIDTH), mode
        )["logit"]
        shifts.append(
            z.reshape(len(clock), len(grid), 2, 3).cpu().numpy()
            - native[past][:, None, None]
        )
    delta = np.stack(shifts, axis=1)
    if not np.isfinite(delta).all():
        raise ValueError("bounded forward produced a nonfinite risk")
    ok = np.abs(delta).max(axis=(1, 3, 4)) <= pair.LIMIT - pair.GUARD_HEADROOM
    selected = ok.argmax(1)
    keep = support & ok.any(1)
    i = np.arange(len(clock))
    risk = delta[i, :, selected]
    factor = np.where(keep, np.asarray(pair.SCALES)[selected], 0.0)
    norm = (u * u.new_tensor(factor)[:, None, None]).cpu().numpy()
    matches = rms * matched * factor[:, None]
    if (
        np.linalg.norm(norm, axis=2).max() > pair.RADIUS + 5e-7
        or np.ptp(matches, axis=1).max() > 1e-6
    ):
        raise ValueError("matched rank probe exceeded edit budget")
    return (
        norm,
        keep,
        np.where(keep, risk[:, 4, 0, -1], 0.0),
        np.where(keep, np.abs(risk).max(axis=(1, 2, 3)), 0.0),
    )


def predict(base, projection, raw):
    x, available = a.bank.prior.views(base, raw)
    p = a.op.encode(base, x, available, projection)
    z = a.bank.prior.scan(base, x, available).detach().cpu().numpy()
    trace = np.broadcast_to(z[:, None], (len(z), len(NAMES))).copy()
    clock = np.arange(63, len(z) - 1, 32, dtype=np.int64)
    if not len(clock):
        return trace, clock, np.zeros(0, bool), np.zeros(0)
    g, history, valid = a.gradients(base, p, raw, clock)
    previous, ranks = a.directions(g, history, valid)
    current = np.zeros_like(previous)
    for j in np.flatnonzero(valid & (ranks >= 4)):
        m = int(clock[j])
        po = p.take([m - 32])
        residual, weights, mask = residuals(base, po, raw, m)
        for mode in (0, 1):
            jac = observe.forecast_jacobian(base, po, mode, mask)
            current[j, 2 * mode : 2 * mode + 2] = ridge_directions(
                jac * weights[:, None],
                residual * weights[None],
                observe.remove_history(history[j, mode]),
            )
    states, active, scalar, drift = match_and_guard(
        base, p, clock, np.concatenate((previous, current), axis=1), z
    )
    blocks = a.pair.future_blocks(clock, len(z))
    for j in np.flatnonzero(active):
        ix = blocks[j]
        with torch.no_grad():
            for k, mode in enumerate(MODES):
                state = p.tokens.new_tensor(states[j, k]).expand(len(ix), -1)
                trace[ix, k + 1] = (
                    a.op.forward(base, p.take(ix), state, mode)["logit"].cpu().numpy()
                )
        trace[ix, -1] = z[ix] + scalar[j]
    return trace, clock, active, drift


def summarize(scores):
    return {
        arm: dict(
            timely=sum(q["scores"][j]["timely"] for q in scores),
            fp=sum(q["scores"][j]["false_alarm"] for q in scores),
            recovered=[
                q["shot"]
                for q in scores
                if not q["scores"][0]["timely"] and q["scores"][j]["timely"]
            ],
            lost=[
                q["shot"]
                for q in scores
                if q["scores"][0]["timely"] and not q["scores"][j]["timely"]
            ],
            fp_added=[
                q["shot"]
                for q in scores
                if not q["scores"][0]["false_alarm"] and q["scores"][j]["false_alarm"]
            ],
            fp_removed=[
                q["shot"]
                for q in scores
                if q["scores"][0]["false_alarm"] and not q["scores"][j]["false_alarm"]
            ],
        )
        for j, arm in enumerate(NAMES)
    }


def run(observed, output):
    if output.exists():
        raise FileExistsError(output)
    h, start = Hashes(), time.monotonic()
    for path in (PLAN, Path(__file__), Path(observe.__file__), observed):
        h(path)
    data = a.r.read(observed)
    if data["status"] != "MEASURED_TRAIN_OBSERVABILITY_NOT_POLICY" or not all(
        h(p) == sha for p, sha in data["input_sha256"].items()
    ):
        raise ValueError("exact existing 40-shot TRAIN experiment required")
    previous, inventory = a.bank.src.source_inputs(h)
    raw_sha = {q["shot"]: q["sha256"] for q in inventory}
    cohort = sorted({x["shot"] for x in data["rows"]})
    if len(cohort) != 40 or not set(cohort) <= set(previous["roles"]["train"]):
        raise ValueError("same earlier 40 source shots required")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, parent = a.r.load_parent(a.r.DEFAULT_PARENT)
    state = a.r.state_digest(base)
    if state != previous["source_state_sha256"] or parent["threshold"] != a.r.THRESHOLD:
        raise ValueError("same original frozen source required")
    projection = a.op.basis(base)
    output.mkdir(parents=True, exist_ok=False)
    protocol = output / "protocol.json"
    a.r.write_new(
        protocol,
        dict(
            status="LOCKED_TRAIN_RIDGE",
            source_sha256=state,
            cohort=cohort,
            inputs={str(k): v[1] for k, v in h.cache.items()},
            arms=NAMES,
            tau=a.TAU,
        ),
    )
    tape, notes = [], []
    for i, sid in enumerate(cohort, 1):
        p = a.r.io23.path_for("J-TEXT", sid)
        if h(p) != raw_sha[sid]:
            raise ValueError("raw source bytes changed")
        raw = a.r.simple_data.read_shot("J-TEXT", sid)
        z, c, active, drift = predict(base, projection, raw)
        notes.append(
            dict(
                shot=sid,
                sha256=h(p),
                length=len(z),
                updates=int(active.sum()),
                blocks=len(c),
                max_guard=float(drift.max(initial=0.0)),
            )
        )
        tape.append((raw["times"], z))
        if i % 10 == 0 or i == len(cohort):
            print(f"Ridge TRAIN whole-shot {i}/{len(cohort)}", flush=True)
    if a.r.state_digest(base) != state or any(
        q.grad is not None for q in base.parameters()
    ):
        raise ValueError("frozen base was modified")
    h.check()
    raw_path = output / "curves.npz"
    a.r.save_arrays(
        raw_path,
        times=np.concatenate([t for t, _ in tape]),
        curves=np.concatenate([z for _, z in tape]),
    )
    seal = output / "seal.json"
    a.r.write_new(
        seal, dict(protocol_sha256=h(protocol), curves_sha256=h(raw_path), shots=notes)
    )
    rows = []
    for q, (t, z) in zip(notes, tape, strict=True):
        flag, down = a.r.io23.read_outcome("J-TEXT", q["shot"])
        rows.append(
            dict(
                shot=q["shot"],
                disruptive=flag,
                scores=[first(t, c, flag, down, a.TAU) for c in z.T],
            )
        )
    summary = summarize(rows)
    main = summary["ridge_motion"]
    extra = sorted(
        set(main["recovered"])
        - {
            s
            for k, item in summary.items()
            if k != "ridge_motion"
            for s in item["recovered"]
        }
    )
    gate = bool(extra and not main["fp_added"] and not main["lost"])
    a.r.write_new(
        output / "result.json",
        dict(
            status="SCORED_TRAIN_ONLY_RIDGE_DEVELOPMENT",
            seal_sha256=h(seal),
            shots=rows,
            summary=summary,
            gate=gate,
            unique_recovered=extra,
            elapsed_seconds=time.monotonic() - start,
        ),
    )
    print(dict(gate=gate, summary=summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("observability", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(args.observability, args.out)
