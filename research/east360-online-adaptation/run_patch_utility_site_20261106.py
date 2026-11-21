"""Matched placement experiment using old, causal, symmetric action directions."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import run_patch_utility_multiview_20261106 as m
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import menu_analysis


b = m.sel.bank
PLAN = b.r.ROOT / "PATCH_UTILITY_SITE_PROTOCOL_20261106.md"
SITES = ("input", "middle")
SCALES = np.asarray(b.pair.SCALES, dtype=np.float64)


def shared_radius(shifts, allowed):
    if shifts.ndim != 5 or shifts.shape[0] != len(SITES):
        raise ValueError("need two position-specific risk shifts")
    step, _ = b.bank.tr.select_symmetric(
        torch.as_tensor(np.max(np.abs(shifts), axis=0)),
        torch.as_tensor(allowed),
        scales=b.pair.SCALES, limit=b.pair.LIMIT - b.pair.GUARD_HEADROOM,
    )
    return step.numpy()


def forward(base, p, u, site):
    if site == "input":
        return b.op.forward(base, p, u, 0)["logit"]
    if site != "middle" or len(base.encoder.layers) != 2:
        raise ValueError("middle placement requires the frozen two-layer encoder")
    change = b.op.delta(p, u, 0)
    with sdpa_kernel(SDPBackend.MATH):
        first = base.encoder.layers[0](p.tokens, src_key_padding_mask=~p.valid)
        h = torch.cat((first[:, :-1] + change, first[:, -1:]), dim=1)
        h = base.encoder.layers[1](h, src_key_padding_mask=~p.valid)
    if base.encoder.norm is not None:
        h = base.encoder.norm(h)
    gain, bias = base.fast_origin.chunk(2)
    cls = base.norm(h[:, -1]) * (1 + .1 * gain.tanh()) + .1 * bias.tanh()
    return base.risk_head(cls).squeeze(-1)


@torch.no_grad()
def scan(base, p, indices, states, site):
    return torch.cat([
        forward(base, p.take(indices[k:k+3072]), states[k:k+3072], site)
        for k in range(0, len(indices), 3072)
    ]).cpu().numpy()


def probe(base, raw, projection, native, clock, states, allowed, archived):
    x, a = b.bank.prior.views(base, raw)
    p = b.op.encode(base, x, a, projection)
    u = x.new_tensor(states[:, 1:])
    n, action_count = u.shape[:2]
    indices = np.arange(len(native))
    reached = b.bank.tr.reached_indices(clock)
    g = x.new_tensor(SCALES)
    anchors = np.broadcast_to(reached[:, None, None], (n, action_count, len(g), 3)).ravel()
    values = (u[:, :, None, None] * g[None, None, :, None, None])
    values = values.expand(-1, -1, -1, 3, -1).reshape(-1, b.op.WIDTH)
    zero, shift = [], []
    for site in SITES:
        z = scan(base, p, indices, x.new_zeros((len(indices), b.op.WIDTH)), site)
        zero.append(float(np.abs(z-native).max()))
        scored = scan(base, p, anchors, values, site).reshape(n, action_count, len(g), 3)
        shift.append(scored - z[reached][:, None, None])
    shift = np.stack(shift)
    common = shared_radius(shift, allowed[:, 1:])
    alone = np.stack([shared_radius(shift[[j, j]], allowed[:, 1:]) for j in range(len(SITES))])
    alive = common > 0
    j = (common[..., None] == SCALES).argmax(axis=-1)
    history = shift[:, np.arange(n)[:, None], np.arange(action_count)[None], j]
    history = np.where(alive[None, :, :, None], history, 0)
    peak = np.max(np.abs(history), initial=0.0)
    if peak > b.pair.LIMIT or max(zero) > 5e-5:
        raise ValueError("matched site exceeded history limit or changed native zero")
    state = x.new_tensor(states) * x.new_tensor(np.pad(common, ((0, 0), (1, 0))))[:, :, None]
    allowed_new = np.pad(alive, ((0, 0), (1, 0)), constant_values=True)
    timeline = clock[:, None] + 1 + np.arange(32)
    live = timeline < len(native)
    z0 = np.where(live, native[np.minimum(timeline, len(native)-1)], 0.0)
    curves = np.broadcast_to(z0[:, None], (n, b.NA, 32)).copy()
    write = allowed_new.copy()
    write[:, 0] = False
    rows, actions, time_idx = np.nonzero(write[:, :, None] & live[:, None, :])
    c = np.stack((curves.copy(), curves.copy()))
    if len(rows):
        for j, site in enumerate(SITES):
            c[j, rows, actions, time_idx] = scan(
                base, p, timeline[rows, time_idx], state[rows, actions], site,
            )
    compare = (np.pad(common == 1, ((0, 0), (1, 0)))[:, :, None] & live[:, None, :])
    exact = float(np.abs(c[0]-archived)[compare].max(initial=0.0))
    if exact > 5e-5 or not np.array_equal(
        c[0][compare].astype(float) >= b.TAU, archived[compare].astype(float) >= b.TAU,
    ):
        raise ValueError("original input site did not replay its unchanged actions")
    return dict(
        curves=c, allowed=allowed_new, common=common, alone=alone,
        history=np.pad(history, ((0, 0), (0, 0), (1, 0), (0, 0))),
        zero=np.asarray(zero), exact=np.array(exact),
    )


def cases(tape):
    for q in tape.shots:
        before = q.teacher["baseline"]
        if q.disruptive and not before["timely"]:
            yield q, "late"
        elif not q.disruptive and tape.hard[q.blocks].any():
            yield q, "safe"


def counts(records, which):
    role, effect = zip(*[(r["role"], r["effects"][which]) for r in records], strict=True)
    positive = [r["shot"] for r, k, c in zip(records, role, effect)
                if k == "late" and c["single_positive"]]
    removed = [r["shot"] for r, k, c in zip(records, role, effect)
               if k == "safe" and c["single_positive"]]
    multi = [r["shot"] for r, k, c in zip(records, role, effect)
             if k == "safe" and c["multi_only_remove"]]
    harm = [r["shot"] for r, k, c in zip(records, role, effect)
            if k == "safe" and c["single_negative"]]
    return dict(timely=positive, fp_removed=removed, multi_only_fp_removed=multi, fp_added=harm)


def run(native, masked, baseline, multiview, out):
    if out.exists():
        raise FileExistsError(out)
    h, start = Hashes(), time.monotonic()
    selection = b.r.read(multiview / "result.json")
    verification = b.r.read(multiview / "verification.json")
    if selection["source_gate"] or verification["result_sha256"] != h(multiview / "result.json"):
        raise ValueError("verified unsuccessful selection required for the position pivot")
    for p in (Path(__file__), Path(b.__file__), Path(b.op.__file__), PLAN, multiview / "verification.json"):
        h(p)
    tapes, _, _ = m.load(native, masked, baseline, h)
    b.r.configure()
    base, parent = b.r.load_parent(b.r.DEFAULT_PARENT)
    digest = b.r.state_digest(base)
    if digest != b.r.read(native / "protocol.json")["source_state_sha256"] or parent["threshold"] != b.r.THRESHOLD:
        raise ValueError("frozen source mismatch")
    base.cuda()
    basis = b.op.basis(base)
    out.mkdir(parents=True, exist_ok=False)
    index = {}
    for name, tape in tapes.items():
        (out/name).mkdir()
        index[name] = []
        for q, role in cases(tape):
            sl = q.blocks
            if sl.start == sl.stop:
                # A pre-clock-only late case has no permissible intervention.
                pack = dict(curves=np.empty((2, 0, b.NA, 32), np.float32),
                            allowed=np.empty((0,b.NA),bool), common=np.empty((0,b.NA-1)),
                            alone=np.empty((2,0,b.NA-1)), history=np.empty((2,0,b.NA,3)),
                            zero=np.zeros(2), exact=np.array(0.0))
            else:
                raw = b.r.simple_data.read_shot("J-TEXT", q.sid)
                if name != "native":
                    from run_patch_utility_target_views_20261106 import mask
                    raw = mask(raw, name)
                pack = probe(base, raw, basis, q.native, *(
                    tape.data[k][sl] for k in ("clock", "states", "allowed", "curves")
                ))
            file = out/name/f"{q.sid}.npz"
            b.r.save_arrays(file, **pack)
            index[name].append(dict(shot=q.sid, role=role, sha256=h(file)))
        print(f"Sealed {name} position probes: {len(index[name])}", flush=True)
    if b.r.state_digest(base) != digest or any(p.grad is not None for p in base.parameters()):
        raise ValueError("original model parameters changed")
    h.check()
    seal = out/"seal.json"
    b.r.write_new(seal, dict(status="KNOWN_ROLE_CAPACITY_PROBE_SEALED", index=index,
                             elapsed_seconds=time.monotonic()-start, outcome_used_for_roles_only=True))
    reports = {}
    for name, tape in tapes.items():
        rows = []
        for (q, role), note in zip(cases(tape), index[name], strict=True):
            file = out/name/f"{q.sid}.npz"
            if h(file) != note["sha256"]:
                raise ValueError("site probe changed after prediction sealing")
            with np.load(file, allow_pickle=False) as f:
                clock = tape.data["clock"][q.blocks]
                record = dict(shot=q.sid, role=role, effects={},
                              source_allowed=int(tape.data["allowed"][q.blocks, 1:].sum()),
                              common_allowed=int((f["common"] > 0).sum()),
                              alone_allowed=(f["alone"] > 0).sum(axis=(1,2)).tolist(),
                              max_guard=float(np.abs(f["history"]).max(initial=0.0)),
                              zero_diff=float(f["zero"].max()), input_archive_diff=float(f["exact"]))
                for j, site in enumerate(SITES):
                    true = f["curves"][j]
                    scalar = true[:, :1] + f["history"][j, ..., 2, None]
                    for label, curves in ((site, true), (site + "_scalar", scalar)):
                        r = menu_analysis(q, clock, curves, f["allowed"], b.TAU)
                        positive = bool((r["hard"] > 0).any())
                        record["effects"][label] = dict(
                            single_positive=positive, single_negative=bool((r["hard"] < 0).any()),
                            multi_only_remove=bool(role == "safe" and r["baseline"]["false_alarm"]
                                                   and r["oracle_no_false_alarm"] and not positive),
                        )
                rows.append(record)
        labels = (*SITES, *(k+"_scalar" for k in SITES))
        c = {k: counts(rows, k) for k in labels}
        unique = sorted(set(c["middle"]["timely"]) -
                        set(c["input"]["timely"] + c["middle_scalar"]["timely"] + c["input_scalar"]["timely"]))
        reports[name] = dict(rows=rows, counts=c, unique_middle=unique)
        print(name, {k: {p: len(v) for p, v in q.items()} for k, q in c.items()}, flush=True)
    h.check()
    unique = sorted({q for view in reports.values() for q in view["unique_middle"]})
    b.r.write_new(out/"result.json", dict(status="MATCHED_SITE_CAPACITY_NOT_ONLINE", views=reports,
        unique_middle=unique, probe_gate=bool(unique), seal_sha256=h(seal),
        input_sha256={str(p): v[1] for p,v in h.cache.items()},
        elapsed_seconds=time.monotonic()-start, new_val_test_east_reads=0))
    print("Distinct matched mid-layer rescues:", unique, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for a in ("native", "masked", "native_result", "multiview", "output"):
        cli.add_argument(a, type=Path)
    q = cli.parse_args()
    run(q.native, q.masked, q.native_result, q.multiview, q.output)
