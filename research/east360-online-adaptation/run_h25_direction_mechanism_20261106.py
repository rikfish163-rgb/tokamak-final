"""Real TRAIN VJPs and guarded forwards for already-arrived H25 supervision."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

import fit_output_trust_20261106 as fit
import h25_arrived_gradient_20261106 as h25
import h25_direction_geometry_20261106 as geo
import run_output_trust_mix_capacity_20261106 as mix
from verify_confirmation_raw_replay_20261105 import Hashes


bank, r, SEEDS, TAU = fit.bank, fit.r, fit.SEEDS, fit.TAU
PRIOR = r.ROOT / "probe_outputs/output_trust_20261106_v1"
MATCH = r.ROOT / "probe_outputs/peak_contrast_capacity_20261106_v1"
PLAN = r.ROOT / "H25_DIRECTION_MECHANISM_PROTOCOL_20261106.md"
MODES = ("old", "h25", "old_true", "old_reverse")
ROLES = ("positive", "protect", "safe")


def records_for(
    shot, disruptive, first, safe_maturity, native, m, positive, protect, late
):
    rows = []
    for k, a in enumerate(m):
        a = int(a)
        clock = np.arange(a + 1, min(a + 33, len(native)))
        if disruptive:
            p = clock[positive[k, : len(clock)]]
            g = clock[protect[k, : len(clock)]]
            if len(p):
                rows.append(
                    dict(
                        shot=shot,
                        maturity=a,
                        role="positive",
                        target=int(p[np.argmax(native[p])]),
                        eligible=p.tolist(),
                        late=late,
                    )
                )
            if len(g):
                if int(g[-1]) != first:
                    raise ValueError("original first-alarm protection moved")
                rows.append(
                    dict(
                        shot=shot,
                        maturity=a,
                        role="protect",
                        target=int(first),
                        eligible=g.tolist(),
                        late=False,
                    )
                )
        elif safe_maturity == a:
            if len(clock) != 32:
                raise ValueError("archived matched safety needs the complete block")
            rows.append(
                dict(
                    shot=shot,
                    maturity=a,
                    role="safe",
                    target=int(clock[np.argmax(native[clock])]),
                    eligible=clock.tolist(),
                    late=False,
                )
            )
    return rows


def catalog(h):
    cfg, _, shards = fit.checked_table(PRIOR, h)
    bound = r.read(MATCH / "completed.json")
    mp = MATCH / "all_pairs.json"
    if h(mp) != bound["pair_sha256"]["all"]:
        raise ValueError("same previously locked safety matching required")
    matched = r.read(mp)["pairs"]
    safe = {p["safe"]["shot"]: p["safe"]["maturity"] for p in matched}
    if len(safe) != len(matched):
        raise ValueError("unsafe matching cohort reuses shots")
    receipt = r.read(PRIOR / "fit_completed.json")
    audit_path = PRIOR / "train_crossfit.json"
    if h(audit_path) != receipt["train_audit_sha256"]:
        raise ValueError("original late cohort receipt changed")
    late = set(r.read(audit_path)["geometry"]["non_timely_shots"])
    selected, sources = defaultdict(list), {}
    advance = r.simple_data.advanced_times()
    for _, meta, path in shards:
        with np.load(path, allow_pickle=False) as d:
            t, z, ms, pos, protect, owner = (
                d[k]
                for k in (
                    "times",
                    "native",
                    "maturity",
                    "positive",
                    "protect",
                    "row_shot",
                )
            )
        offset = 0
        for rec in meta["shots"]:
            sid = rec["shot"]
            end = offset + rec["n_points"]
            time, native = t[offset:end], z[offset:end]
            offset = end
            if not rec["disruptive"] and sid not in safe:
                continue
            row = owner == sid
            if (not rec["disruptive"] and (row & (ms == safe[sid])).sum() != 1) or int(
                row.sum()
            ) != rec["episodes"]:
                raise ValueError("missing source teaching or safety block")
            flag, down = r.io23.read_outcome("J-TEXT", sid)
            horizon = r.io23.effective_horizon_ms(sid, advance)[0]
            if (
                flag != rec["disruptive"]
                or (flag and down != rec["down"])
                or horizon != rec["horizon_ms"]
            ):
                raise ValueError("source outcome and warning time differ")
            p, q, first = bank.fwd.teaching_masks(
                time, native, ms[row], flag, down, horizon, TAU
            )
            if (
                first != rec["first"]
                or not np.array_equal(p, pos[row])
                or not np.array_equal(q, protect[row])
            ):
                raise ValueError(
                    "source masks did not replay from the frozen raw clock"
                )
            cases = records_for(
                sid, flag, first, safe.get(sid), native, ms[row], p, q, sid in late
            )
            if cases:
                selected[sid].extend(cases)
                sources[sid] = dict(rec=rec, times=time, native=native)
    if {q["shot"] for p in selected.values() for q in p if q["late"]} != late:
        raise ValueError("all original TRAIN late cases required")
    if sum(q["role"] == "safe" for p in selected.values() for q in p) != len(safe):
        raise ValueError("all previously matched safety blocks required")
    return cfg, selected, sources, late


def state_for(base, old_fit, x, a, raw, maturity, schedule):
    # Each call is keyed by m; labels and future classification targets
    # do not enter either the origin forecast or its loss.
    j = np.searchsorted(schedule["maturity"], maturity)
    if (
        not np.array_equal(schedule["maturity"][j], maturity)
        or not schedule["support"][j].all()
    ):
        raise ValueError("old maturity and diagnostic support changed")
    o = schedule["origin"][j]
    x_o, a_o = x[o].contiguous(), a[o].contiguous()
    x_m, a_m = x[maturity].contiguous(), a[maturity].contiguous()
    target, mask = h25.arrived_targets(raw["values"], raw["available"], o, maturity)
    p = h25.gradients(
        base,
        x_o,
        a_o,
        bank.src.field(target, base.center.device),
        bank.src.field(mask, base.center.device),
    )
    old, desc, ok = [], [], []
    for dec, mixer, norm, ref in old_fit:
        v = bank.prior.normalize_targets(
            bank.src.field(schedule["delta"][j], base.center.device), norm
        )
        d, b, _, good = bank.rule.reached_bank(
            base,
            dec,
            mixer,
            x_o,
            a_o,
            x_m,
            a_m,
            v,
            bank.src.field(schedule["valid"][j], base.center.device),
            ref,
        )
        old.append(b.cpu().numpy())
        desc.append(d.cpu().numpy())
        ok.append(good.cpu().numpy())
    return dict(
        old=np.stack(old, 1),
        descriptor=np.stack(desc, 1),
        old_ok=np.stack(ok, 1).all(1),
        **{k: v.cpu().numpy() for k, v in p.items()},
    )


def shared_grid(delta, cap):
    if delta.ndim != 4 or not np.isfinite(delta).all() or delta.shape[-2:] != (2, 3):
        raise ValueError(
            "aligned finite grid, arms, signs and three reached times required"
        )
    k = np.flatnonzero(np.abs(delta).max((1, 2, 3)) <= cap)
    return int(k[0]) if len(k) else None


def unit_average(v):
    n = np.linalg.norm(v, axis=1)
    use = n > 1e-8
    return (v[use] / n[use, None]).mean(0) if use.any() else np.zeros(v.shape[-1])


def spaces(old, real, reverse):
    u, qt, qr, rank = geo.matched_augments(old, real, reverse)
    return (u, geo.basis(real), qt, qr), rank


def fixed_directions(q, old, real, reverse, past):
    a, b, c = unit_average(old[:3]), -unit_average(real), -unit_average(reverse)
    vectors = (a, b, (a + b) * 0.5, (a + c) * 0.5)
    return np.stack(
        [
            space @ mix.projected(space, past, v)
            for space, v in zip(q, vectors, strict=True)
        ]
    )


@torch.no_grad()
def causal_effects(base, x, a, m, k, state, native, role):
    """Try exact steps without using labels to choose a scale."""
    p = bank.tr.reached_indices(np.array([m]))[0]
    xp, ap = x[p].contiguous(), a[p].contiguous()
    t = torch.as_tensor(state, dtype=x.dtype, device=x.device)
    g = t.new_tensor(bank.tr.SCALES)
    vv = (g[:, None, None] * t[None]).unsqueeze(2) * t.new_tensor([1.0, -1.0])[
        None, None, :, None
    ]
    n, a_count = len(g), len(t)
    z = (
        bank.fwd.batched_risk(
            base,
            bank.fwd.encode(base, xp, ap),
            vv.reshape(-1, 128)[None].expand(3, -1, -1),
        )
        .reshape(3, n, a_count, 2)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    shift = z - native[p]
    selected = []
    legal = []
    offsets = []
    for columns in ([0], [1], [2, 3]):
        f = shared_grid(shift[:, columns], bank.tr.LIMIT - 2e-5)
        if f is None or (np.linalg.norm(state[columns], axis=-1) <= 1e-7).any():
            selected.extend(np.zeros((len(columns), 128), dtype=np.float32))
            legal.extend([0.0] * len(columns))
            offsets.extend([0.0] * len(columns))
        else:
            selected.extend((state[columns] * float(g[f])).astype(np.float32))
            legal.extend([float(g[f])] * len(columns))
            offsets.extend(shift[f, columns, 0, -1].tolist())
    out = (
        bank.fwd.batched_risk(
            base,
            bank.fwd.encode(base, x[k].contiguous(), a[k].contiguous()),
            bank.src.field(np.asarray(selected), x.device)[None].expand(len(k), -1, -1),
        )
        .cpu()
        .numpy()
    )
    prior = float(native[k].max())
    peak = np.where(np.asarray(legal) > 0, out.max(0), prior)
    harmful = (
        (peak < TAU)
        if role == "protect"
        else ((prior < TAU) & (peak >= TAU))
        if role == "safe"
        else np.zeros(len(peak), dtype=bool)
    )
    return dict(
        delta=(peak - prior).tolist(),
        peak=peak.tolist(),
        radius=legal,
        harmful=harmful.tolist(),
        anchor=offsets,
        relative=(peak - prior - np.asarray(offsets)).tolist(),
    )


def shot_average(values, ids):
    keys = sorted(set(ids))
    if not keys:
        return None
    y, ids = np.asarray(values, dtype=np.float64), np.asarray(ids)
    by_shot = np.stack([y[ids == sid].mean(0) for sid in keys])
    return by_shot.mean(0).tolist()


def summary(rows):
    out = {}
    for role in ROLES:
        items = [q for q in rows if q["role"] == role]
        good = [q for q in items if q["supported"]]
        ids = [q["shot"] for q in good]
        cap = np.asarray([q["fraction"] for q in good])
        v = np.asarray([q["delta"] for q in good])
        harm = np.asarray([q["harmful"] for q in good], dtype=np.int32)
        k = np.asarray([q["radius"] for q in good])
        out[role] = dict(
            n_windows=len(items),
            n_supported=len(good),
            n_shots=len(set(ids)),
            capture=shot_average(cap, ids),
            raw_peak_delta=shot_average(v, ids),
            normalized_shape_delta=shot_average([q["relative"] for q in good], ids),
            raw_harm_by_seed=harm.sum(0).tolist(),
            active_by_seed=(k > 0).sum(0).tolist(),
            common_extra_ranks={
                str(i): sum(x == i for q in good for x in q["rank"]) for i in range(4)
            },
        )
        if good:
            unique = sorted(set(ids))
            diff = cap[..., 2] - cap[..., 3]
            by_shot = np.stack(
                [diff[np.asarray(ids) == sid].mean(0) for sid in unique]
            ).mean(1)
            rng = np.random.default_rng(61106)
            idx = rng.integers(0, len(unique), (2000, len(unique)))
            out[role]["paired_capture_gain"] = dict(
                real_minus_reverse=float(by_shot.mean()),
                percentiles=np.percentile(by_shot[idx].mean(1), [2.5, 97.5]).tolist(),
            )
    return out


def trace_one(base, diag, raw, native, rows, do_witness):
    x, a = bank.prior.views(base, raw)
    full = bank.prior.scan(base, x, a).cpu().numpy()
    if not np.allclose(full, native, atol=5e-5, rtol=0) or not np.array_equal(
        full >= TAU, native >= TAU
    ):
        raise ValueError("new native shot does not reproduce prior frozen curve")
    m = np.asarray([q["maturity"] for q in rows], dtype=np.int64)
    unique, idx = np.unique(m, return_inverse=True)
    schedule = bank.prior.signal_schedule(base, raw)
    s = state_for(base, diag, x, a, raw, unique, schedule)
    pk = bank.tr.reached_indices(unique)
    past = mix.gradient(
        base, x[pk.ravel()].contiguous(), a[pk.ravel()].contiguous()
    ).reshape(len(unique), 3, 128)
    target = np.asarray([q["target"] for q in rows])
    truth = mix.gradient(base, x[target].contiguous(), a[target].contiguous())
    check = None
    if do_witness:
        cut = int(unique[0] + 1)
        short = {k: raw[k][:cut] for k in ("times", "values", "available")}
        xs, ass = bank.prior.views(base, short)
        p = state_for(
            base,
            diag,
            xs,
            ass,
            short,
            unique[:1],
            bank.prior.signal_schedule(base, short),
        )
        err = max(
            float(np.abs(p[k] - s[k][:1]).max()) for k in ("true", "reverse", "old")
        )
        if err > 8e-5 or any(
            not np.array_equal(p[k], s[k][:1]) for k in ("mask", "groups", "old_ok")
        ):
            raise ValueError("cut prefix changed a reached gradient or mask")
        check = dict(shot=rows[0]["shot"], maturity=cut - 1, max_gradient_error=err)
    result = []
    payload = dict(
        old=[],
        true=[],
        reverse=[],
        past=[],
        future=[],
        loss=[],
        descriptor=[],
        support=[],
    )
    for w, (case, j) in enumerate(zip(rows, idx, strict=True)):
        h, v, b = s["true"][j], s["reverse"][j], s["old"][j]
        supported = bool(s["old_ok"][j] and s["groups"][j].sum() >= 2)
        feature = dict(
            old=b,
            true=h,
            reverse=v,
            past=past[j],
            future=truth[w],
            loss=s["loss"][j],
            descriptor=s["descriptor"][j],
            support=supported,
        )
        for k in payload:
            payload[k].append(feature[k])
        rec = {
            **case,
            "supported": supported,
            "h25_groups": s["groups"][j].tolist(),
            "h25_observed_count": int(s["mask"][j].sum()),
            "source_peak": float(native[case["eligible"]].max()),
        }
        if supported:
            f, absolute, geometry_rank, gains = [], [], [], defaultdict(list)
            for old in b:
                q, extra = spaces(old, h, v)
                real = truth[w] * (-1 if case["role"] == "safe" else 1)
                u = fixed_directions(q, old, h, v, past[j])
                c = [geo.capture(d, past[j], real) for d in q]
                f.append([i["fraction"] for i in c])
                absolute.append([i["absolute_norm"] for i in c])
                geometry_rank.append(extra)
                for name, values in causal_effects(
                    base,
                    x,
                    a,
                    case["maturity"],
                    np.asarray(case["eligible"]),
                    u,
                    native,
                    case["role"],
                ).items():
                    gains[name].append(values)
            rec.update(fraction=f, absolute_norm=absolute, rank=geometry_rank, **gains)
        result.append(rec)
    return result, {k: np.asarray(v) for k, v in payload.items()}, check


def run(root):
    h, start = Hashes(), time.monotonic()
    if root.exists():
        raise FileExistsError(root)
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    cfg, chosen, sources, late = catalog(h)
    if (
        r.state_digest(base) != cfg["source_state_sha256"]
        or parent["threshold"] != r.THRESHOLD
    ):
        raise ValueError("same qualified parent, head, and threshold required")
    lock_paths = (
        PLAN,
        Path(__file__),
        Path(h25.__file__),
        Path(geo.__file__),
        Path(mix.__file__),
        Path(fit.__file__),
        Path(bank.__file__),
        PRIOR / "protocol.json",
        PRIOR / "completed.json",
        PRIOR / "train_crossfit.json",
        PRIOR / "continuous_capacity_v2.json",
        MATCH / "completed.json",
        MATCH / "all_pairs.json",
    )
    root.mkdir(parents=True, exist_ok=False)
    protocol = root / "protocol.json"
    r.write_new(
        protocol,
        dict(
            status="LOCKED_H25_MECHANISM",
            inputs={str(p.resolve()): h(p) for p in lock_paths},
            modes=MODES,
            seeds=SEEDS,
            threshold=r.THRESHOLD,
            late=sorted(late),
            source_state_sha256=cfg["source_state_sha256"],
            val_test_east_reads=0,
        ),
    )
    base.cuda()
    diag = bank.frozen_fits(base)
    all_rows, probes, witness = [], defaultdict(list), []
    for rank, (sid, rows) in enumerate(chosen.items(), 1):
        rec = sources[sid]
        if h(r.io23.path_for("J-TEXT", sid)) != rec["rec"]["raw_sha256"]:
            raise ValueError("selected TRAIN raw bytes changed")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        if not np.array_equal(raw["times"], rec["times"]):
            raise ValueError("raw source times changed")
        cases, data, check = trace_one(
            base, diag, raw, rec["native"], rows, rank in (1, 73, 151)
        )
        all_rows.extend(cases)
        for k, v in data.items():
            probes[k].append(v)
        if check is not None:
            witness.append(check)
        if rank % 50 == 0 or rank == len(chosen):
            print(
                f"H25 mechanism {rank}/{len(chosen)} shots, {len(all_rows)} windows",
                flush=True,
            )
    data_path = root / "geometry.npz"
    r.save_arrays(data_path, **{k: np.concatenate(v) for k, v in probes.items()})
    if r.state_digest(base) != cfg["source_state_sha256"] or any(
        x.grad is not None for x in base.parameters()
    ):
        raise ValueError("source head or backbone changed")
    h.check()
    r.write_new(
        root / "mechanism.json",
        dict(
            status="MEASURED_TRAIN_ONLY_H25_GEOMETRY",
            protocol_sha256=h(protocol),
            geometry_sha256=h(data_path),
            cases=all_rows,
            summary=summary(all_rows),
            prefix=witness,
            seconds=time.monotonic() - start,
            val_test_east_reads=0,
        ),
    )
    print(summary(all_rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
