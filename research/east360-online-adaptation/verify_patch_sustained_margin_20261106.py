"""Recount every new single edit on the whole shot and witness its neural forward."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import run_patch_sustained_margin_20261106 as trial
from verify_adjacent_patch_intra_20261106 import outcome_and_clock
from verify_patch_utility_selection_20261106 import first_alarm


b = trial.b


def audit_view(view, path, rows, tau):
    watches, late_notes = {}, []
    total, maximum, maxnorm = 0, 0.0, 0.0
    for r in rows:
        sid = r["shot"]
        with np.load(path / f"{sid}.npz", allow_pickle=False) as f:
            z, t, c, curves, scale, u, guards, target = (f[k] for k in
                ("native", "times", "clock", "curves", "scale", "states", "guard", "targets"))
        flag, down, truth_t = outcome_and_clock(b.r.io23.path_for("J-TEXT", sid), len(t))
        if not np.array_equal(t, truth_t) or (flag, down) != (r["disruptive"], r["down"]):
            raise ValueError(f"HDF5 changed on {view}/{sid}")
        before = first_alarm(t, z, flag, down, tau)
        indices = c[:, None] + np.arange(1, 33)
        present = indices < len(t)
        ix = np.minimum(indices, len(t) - 1)
        frozen = np.where(present, z[ix], 0)
        live = scale > 0
        cap = float(np.abs(guards).max(initial=0.0))
        norm = np.linalg.norm(u, axis=-1)
        maximum = max(maximum, cap)
        maxnorm = max(maxnorm, float(norm.max(initial=0.0)))
        if (len(c) != r["blocks"] or int(live.sum()) != r["allowed"]
                or (np.diff(c) < 32).any() or ((c - 63) % 32).any()
                or (c < 63).any() or not present[:, 9].all()
                or not (np.maximum.accumulate(z.astype(float))[c + 8] < tau).all()
                or not np.isin(scale, [0.0, *trial.SCALES]).all()
                or (norm > b.pair.RADIUS * scale[:, None] + 5e-7).any() or cap > trial.BOUND
                or not np.isfinite(curves).all() or not np.isfinite(u).all()
                or not (curves[:, :, 0] == frozen).all()
                or not (curves[:, :, :, :8] == frozen[:, None, :8]).all()
                or not (curves[:, ~live] == frozen[~live, None]).all()
                or np.any(np.where(present[None, :, None], 0, curves))):
            raise ValueError(f"invalid write window or exact guard on {view}/{sid}")
        timely = np.ones(t.shape, bool) if not flag else np.array([
            round(float((down - x) * 1000.0), 9) > 10.0 for x in t
        ])
        if len(c):
            aim = c[:, None] + 9 + target
            if ((target < 0).any() or (aim >= len(z)).any() or (aim[:, 1] + 1 >= len(z)).any()
                    or not timely[aim].all() or not timely[aim[:, 1] + 1].all()):
                raise ValueError(f"teacher used ineligible frame on {view}/{sid}")
        truth = present & timely[ix] & (np.arange(32) >= 8)
        for j, arm in enumerate(trial.ARMS):
            hard, pair = np.zeros((len(c), 2), np.int8), np.zeros((len(c), 2), bool)
            for k in np.flatnonzero(live):
                for a in range(2):
                    end = min(c[k] + 33, len(z))
                    trace = z.copy()
                    trace[c[k] + 9:end] = curves[j, k, a+1, 8:8+end-(c[k]+9)]
                    after = first_alarm(t, trace, flag, down, tau)
                    hard[k, a] = (int(after["timely"]) - int(before["timely"]) if flag
                                  else int(before["false_alarm"]) - int(after["false_alarm"]))
                    reaches = truth[k] & (curves[j, k, a+1].astype(float) >= tau)
                    pair[k, a] = hard[k, a] == 1 and flag and bool(np.any(reaches[:-1] & reaches[1:]))
            total += 2 * int(live.sum())
            for name, sel in ((arm, slice(0, 1)), (arm + "_signed", slice(None))):
                g, d, p = (int(x.sum()) for x in (hard[:, sel] > 0, hard[:, sel] < 0, pair[:, sel]))
                got = dict(good=bool(g), bad=bool(d), pair=bool(p), good_actions=g, bad_actions=d, pair_actions=p)
                if got != r["effects"][name]:
                    raise ValueError(f"whole-shot recount changed {view}/{sid}/{name}")
            for key, wanted in ((f"{arm}_late", hard[:, 0] > 0), (f"{arm}_fp", hard[:, 0] < 0)):
                if r["role"] == ("late_disruption" if key.endswith("late") else "safe_no_alarm") and wanted.any():
                    watches.setdefault(key, (sid, int(np.flatnonzero(wanted)[0])))
            if sid == 1056381 and (hard[:, 0] > 0).any():
                watches["peak_new_case"] = (sid, int(np.flatnonzero(hard[:, 0] > 0)[0]))
        if r["role"] == "late_disruption":
            slots = truth & live[:, None]
            late_notes.append(dict(shot=sid, target_slots=int(slots.sum()),
                margins=None if not slots.any() else [float(a[slots].max()) - tau for a in
                (frozen, curves[0, :, 1], curves[1, :, 1])]))
    return dict(edits=total, max_guard=maximum, max_state_norm=maxnorm,
                late_notes=late_notes), sorted(set(watches.values()))


@torch.no_grad()
def witness(view, path, samples, base, projection):
    top = 0.0
    for sid, k in samples:
        with np.load(path / f"{sid}.npz", allow_pickle=False) as f:
            c, n = int(f["clock"][k]), len(f["native"])
            reached = np.array([c-16, c-8, c, c+8])
            future = np.arange(c+9, min(c+33, n))
            state, reference, native = f["states"][k], f["curves"][:, k], f["native"]
            guard = f["guard"][k]
        x = b.r.simple_data.read_shot("J-TEXT", sid)
        inputs = b.bank.prior.views(base, x if view == "native" else trial.mask(x, view))
        packets = b.op.encode(base, *inputs, projection).take(np.r_[reached, future])
        for j in range(2):
            for a, sign in enumerate((1, -1)):
                param = packets.tokens.new_tensor(sign * state[j]).expand(len(packets.tokens), -1)
                logits = b.op.forward(base, packets, param, 0)["logit"].cpu().numpy()
                past_error = np.max(np.abs(logits[:4] - native[reached] - guard[j, a]))
                future_error = np.max(np.abs(logits[4:] - reference[j, a+1, 8:8+len(future)]))
                if not np.array_equal(logits[4:].astype(float) >= b.TAU,
                                      reference[j, a+1, 8:8+len(future)].astype(float) >= b.TAU):
                    raise ValueError(f"raw threshold crossings changed for {sid}")
                top = max(top, float(past_error), float(future_error))
    if top > 5e-5:
        raise ValueError("archived patch shifts or suffix did not match the raw network")
    return dict(samples=samples, max_abs_error=top)


def run(root):
    start = time.monotonic()
    result, proto = (b.r.read(root / f"{name}.json") for name in ("result", "protocol"))
    b.r.configure()
    base, _ = b.r.load_parent(b.r.DEFAULT_PARENT)
    if b.r.state_digest(base) != proto["source_state"]:
        raise ValueError("raw witness source differs")
    base.cuda()
    projection = b.op.basis(base)
    audits = {}
    for view in trial.m.VIEWS:
        rec, samples = audit_view(view, root / view, result["rows"][view], proto["tau"])
        rec["raw_witness"] = witness(view, root / view, samples, base, projection)
        audits[view] = rec
        print(view, rec["edits"], rec["max_guard"], rec["raw_witness"], flush=True)
    b.r.write_new(root / "verification.json", dict(status="WHOLE_SHOT_AND_RAW_WITNESS_MATCH",
                                                   views=audits, duration=time.monotonic()-start))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
