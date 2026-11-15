"""Measure matched TRAIN-only shape capacity, with real internal model forwards."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from scipy.special import logit

import peak_contrast_forward_20261106 as forward
import peak_contrast_match_20261106 as matching
import run_whole_block_update_20261106 as parent_run
from verify_confirmation_raw_replay_20261105 import Hashes


r, prior, old, rule = (parent_run.r, parent_run.prior, parent_run.old, parent_run.rule)
TAU, SEEDS = parent_run.TAU, parent_run.SEEDS
HIGH = float(logit(r.THRESHOLD - 0.15))
PLAN = r.ROOT / "PEAK_CONTRAST_CAPACITY_PROTOCOL_20261106.md"
PRIOR_RESULT = r.ROOT / "probe_outputs/whole_block_update_20261106_v2"
BLOCK = old.BLOCK
PREFIX_ATOL = 8e-5


def collect(base, cfg, source, h):
    """Read TRAIN truth only for offline selection; full signals stay in memory."""
    census = r.read(prior.SOURCE_CENSUS)["source"]
    if [q["shot"] for q in census] != cfg["roles"]["train"]:
        raise ValueError("exact TRAIN-only manifest required")
    adv = r.simple_data.advanced_times()
    tapes, availability, danger, safe, raw_manifest = [], [], [], [], []
    start = 0
    for rank, (ref, rec) in enumerate(zip(source, census, strict=True), 1):
        sid = rec["shot"]
        path = r.io23.path_for("J-TEXT", sid)
        raw_hash = h(path)
        if raw_hash != rec["sha256"] or raw_hash != ref["sha256"]:
            raise ValueError(f"frozen J-TEXT TRAIN shot bytes changed: {sid}")
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        if flag != rec["disruptive"]:
            raise ValueError(f"authoritative J-TEXT TRAIN outcome changed: {sid}")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        t = raw["times"]
        horizon = r.io23.effective_horizon_ms(sid, adv)[0]
        f = prior.signal_schedule(base, raw)
        x, a = prior.views(base, raw)
        z = prior.scan(base, x, a).cpu().numpy()
        eligible = 0
        for j in np.flatnonzero(f["support"]):
            m, o = int(f["maturity"][j]), int(f["origin"][j])
            window = forward.episode_windows(t, z, m, flag, down, horizon, TAU)
            if window is None:
                continue
            positive, protect, timely = window
            k = np.arange(m + 1, m + 1 + BLOCK)
            present = k < len(t)
            use = np.minimum(k, len(t) - 1)
            z_next = np.where(present, z[use], np.float32(0.0))
            row = dict(
                shot=sid,
                maturity=m,
                age_ms=float((t[m] - t[0]) * 1000),
                anchor=float(z[m]),
                trend=float(z[m] - z[o]),
                valid=f["valid"][j].copy(),
                delta=f["delta"][j].copy(),
                mx=start + m + 127,
                ox=start + o + 127,
                future_ix=start + use + 127,
                native=z_next,
                present=present,
                positive=positive,
                protect=protect,
                native_timely=timely,
            )
            (danger if flag else safe).append(row)
            eligible += 1
        raw_manifest.append(
            dict(shot=sid, sha256=raw_hash, disruptive=flag, candidates=eligible)
        )
        tapes.append(np.pad(raw["values"], ((127, 0), (0, 0))))
        availability.append(np.pad(raw["available"], ((127, 0), (0, 0))))
        start += len(t) + 127
        if rank % 300 == 0 or rank == len(source):
            print(f"source TRAIN catalog {rank}/{len(source)}", flush=True)
    tape = {
        "values": parent_run.field(np.concatenate(tapes), base.center.device),
        "available": parent_run.field(np.concatenate(availability), base.center.device),
    }
    return tape, danger, safe, raw_manifest


def paired_rows(danger, safe, result):
    return [(danger[q["danger"]], safe[q["safe"]]) for q in result["pairs"]]


def exported_episode(row):
    return {
        key: row[key].tolist() if isinstance(row[key], np.ndarray) else row[key]
        for key in (
            "shot",
            "maturity",
            "age_ms",
            "anchor",
            "trend",
            "valid",
            "positive",
            "protect",
            "present",
            "native_timely",
        )
    }


def bank_at(base, fit, ox, oa, mx, ma, target, valid):
    dec, mixer, scale, ref = fit
    target = prior.normalize_targets(target, scale)
    descriptor, broad, _, old_ok = rule.reached_bank(
        base, dec, mixer, ox, oa, mx, ma, target, valid, ref
    )
    recent = forward.localized_bank(base, dec, ox, oa, mx, ma, target, valid, ref)
    states, available = forward.fixed_states(broad, recent)
    return states, available, old_ok, descriptor


def banks(base, fits, tape, rows):
    dev = base.center.device
    field = lambda key: parent_run.field(np.asarray([q[key] for q in rows]), dev)  # noqa: E731
    origin, mature = field("ox"), field("mx")
    target, valid = field("delta"), field("valid")
    states, allowed, old_support, descriptions = [], [], [], []
    for fit in fits:
        values = []
        for ix in torch.arange(len(rows), device=dev).split(96):
            ox, oa = parent_run.tape_views(tape, origin[ix])
            mx, ma = parent_run.tape_views(tape, mature[ix])
            values.append(bank_at(base, fit, ox, oa, mx, ma, target[ix], valid[ix]))
        u, k, ok, desc = (torch.cat([q[i] for q in values], dim=0) for i in range(4))
        states.append(u)
        allowed.append(k)
        old_support.append(ok)
        descriptions.append(desc)
    # One action index must be usable at both placements and on both sides.
    ok = torch.stack(old_support, 1).reshape(-1, 2, len(SEEDS)).all(dim=(1, 2))
    avail = torch.stack(allowed, 1).reshape(-1, 2, len(SEEDS), len(forward.ARMS))
    common = avail.all(1) & ok[:, None, None]
    u = torch.stack(states, 2)
    u.masked_fill_(~common.repeat_interleave(2, dim=0)[:, None, :, :, None], 0.0)
    return u, common, torch.stack(descriptions, 1)


@torch.no_grad()
def routed_batch(base, tape, endpoints, u, site):
    x, a = parent_run.tape_views(tape, endpoints)
    m, q = u.shape[:2]
    xx = x[:, None].expand(-1, q, -1, -1).reshape(-1, 128, 23)
    aa = a[:, None].expand(-1, q, -1, -1).reshape_as(xx)
    v = forward.risk(base, xx, aa, u.reshape(-1, old.WIDTH), site)
    return v.reshape(m, q)


@torch.no_grad()
def effects(base, tape, rows, states, common, desc):
    dev = base.center.device
    m, sites, seeds, arms, _ = states.shape
    field = lambda key: parent_run.field(np.asarray([q[key] for q in rows]), dev)  # noqa: E731
    ix, present = field("future_ix"), field("present")
    anchor, mature, native = field("anchor").float(), field("mx"), field("native")
    shape = (m, sites, seeds, arms)
    updated = native[:, None, None, None, :].expand(*shape, BLOCK).clone()
    offsets = updated.new_zeros(shape)
    used = common.repeat_interleave(2, 0)
    row, t = present.nonzero(as_tuple=True)
    for i, site in enumerate(forward.SITES):
        for j in range(seeds):
            state = states[:, i, j]
            for k in range(0, m, 48):
                ids = slice(k, k + 48)
                v = routed_batch(base, tape, mature[ids], state[ids], site)
                offsets[ids, i, j] = (v - anchor[ids, None]).masked_fill(
                    ~used[ids, j], 0
                )
            for k in range(0, len(t), 48):
                q, lag = row[k : k + 48], t[k : k + 48]
                v = routed_batch(base, tape, ix[q, lag], state[q], site)
                updated[q, i, j, :, lag] = torch.where(
                    used[q, j], v, native[q, lag, None]
                )
    norms = states.norm(dim=-1)
    expected = used[:, None].expand_as(norms).to(norms) * forward.SITE_NORM
    if (
        not torch.allclose(norms, expected, rtol=0, atol=1e-6)
        or not torch.equal(states[..., 1::2, :], -states[..., ::2, :])
        or not all(torch.isfinite(q).all() for q in (updated, offsets, norms, desc))
    ):
        raise ValueError("finite, same-norm real site and reverse actions required")
    pairs = m // 2
    return {
        key: x.reshape(pairs, 2, *x.shape[1:]).cpu().numpy()
        for key, x in (
            ("native", native),
            ("future", updated),
            ("anchor_delta", offsets),
            ("norms", norms),
            ("descriptor", desc),
        )
    } | {"available": common.cpu().numpy()}


def prefix_witness(base, fits, tape, pair):
    """Remove every sample after the update was determined, without labels."""
    dev = base.center.device
    checks = []
    for row in pair:
        raw = r.simple_data.read_shot("J-TEXT", row["shot"])
        stop = row["maturity"] + 1
        short = {k: raw[k][:stop] for k in ("values", "available", "times")}
        f = prior.signal_schedule(base, short)
        i = len(f["maturity"]) - 1
        if (
            f["maturity"][i] != stop - 1
            or not np.array_equal(f["valid"][i], row["valid"])
            or not np.array_equal(f["delta"][i], row["delta"])
        ):
            raise ValueError("unarrived suffix altered diagnostic feedback")
        x, a = prior.views(base, short)
        o, m = f["origin"][i], f["maturity"][i]
        ox, oa = parent_run.tape_views(tape, torch.tensor([row["ox"]], device=dev))
        mx, ma = parent_run.tape_views(tape, torch.tensor([row["mx"]], device=dev))
        target = parent_run.field(row["delta"][None], dev)
        valid = parent_run.field(row["valid"][None], dev)
        error = 0.0
        for fit in fits:
            full = bank_at(base, fit, ox, oa, mx, ma, target, valid)
            part = bank_at(
                base,
                fit,
                x[o : o + 1].contiguous(),
                a[o : o + 1].contiguous(),
                x[m : m + 1].contiguous(),
                a[m : m + 1].contiguous(),
                target,
                valid,
            )
            if not torch.equal(full[1], part[1]) or not torch.equal(full[2], part[2]):
                raise ValueError("unarrived suffix changed direction support")
            error = max(error, float((full[0] - part[0]).abs().max()))
        if error > PREFIX_ATOL:
            raise ValueError("unarrived suffix changed temporary parameter")
        checks.append(dict(shot=row["shot"], maturity=stop - 1, max_state_error=error))
    return checks


def empty_payload():
    shape = (0, 2, len(forward.SITES), len(SEEDS), len(forward.ARMS))
    return dict(
        native=np.empty((0, 2, BLOCK), dtype=np.float32),
        future=np.empty((*shape, BLOCK), dtype=np.float32),
        anchor_delta=np.empty(shape, dtype=np.float32),
        norms=np.empty(shape, dtype=np.float32),
        descriptor=np.empty((0, 2, len(SEEDS), rule.FEATURES), dtype=np.float32),
        available=np.empty((0, len(SEEDS), len(forward.ARMS)), dtype=bool),
    )


def seal_cohort(base, root, name, pairs, match, tape, fits, h):
    rows = [q for pair in pairs for q in pair]
    if rows:
        u, common, desc = banks(base, fits, tape, rows)
        payload = effects(base, tape, rows, u, common, desc)
        witness = prefix_witness(base, fits, tape, pairs[0])
    else:
        payload, witness = empty_payload(), []
    dest = root / f"{name}.npz"
    r.save_arrays(dest, **payload)
    detail = root / f"{name}_pairs.json"
    r.write_new(
        detail,
        dict(
            cohort=name,
            pairs=[
                dict(
                    danger=exported_episode(pair[0]),
                    safe=exported_episode(pair[1]),
                    cost=q["cost"],
                )
                for pair, q in zip(pairs, match["pairs"], strict=True)
            ],
            diagnostics=match["diagnostics"],
            prefix=witness,
            archive_sha256=h(dest),
            n_supported=int(payload["available"].any((1, 2)).sum()),
        ),
    )
    print(
        name,
        "matched",
        len(pairs),
        "supported",
        int(payload["available"].any((1, 2)).sum()),
        flush=True,
    )
    return h(detail)


def run(root):
    h, start = Hashes(), time.monotonic()
    r.configure()
    cfg, inventory = parent_run.source_inputs(h)
    base, raw_parent = r.load_parent(r.DEFAULT_PARENT)
    state = r.state_digest(base)
    if raw_parent["threshold"] != r.THRESHOLD or state != cfg["source_state_sha256"]:
        raise ValueError("same qualified J-TEXT source and risk threshold required")
    base.cuda()
    root.mkdir(parents=True, exist_ok=False)
    source_done = PRIOR_RESULT / "source_done.json"
    reference = r.read(source_done)
    if reference["status"] != "COMPLETE_WHOLE_BLOCK_SOURCE":
        raise ValueError("reference whole-block experiment must be complete")
    protocol = root / "protocol.json"
    r.write_new(
        protocol,
        dict(
            status="LOCKED_PEAK_CONTRAST",
            seed_order=SEEDS,
            sites=forward.SITES,
            arms=forward.ARMS,
            action_norm=forward.SITE_NORM,
            logit_threshold=TAU,
            threshold=r.THRESHOLD,
            high_floor=HIGH,
            state_sha256=state,
            train_ids=cfg["roles"]["train"],
            held_signal_reads=0,
            held_outcome_reads=0,
            input_sha256={
                **cfg["input_sha256"],
                **{
                    str(p.resolve()): h(p)
                    for p in (
                        PLAN,
                        Path(__file__),
                        Path(forward.__file__),
                        Path(matching.__file__),
                        Path(rule.__file__),
                        Path(parent_run.__file__),
                        source_done,
                        parent_run.PREVIOUS / "protocol.json",
                        parent_run.PREVIOUS / "fit_completed.json",
                        parent_run.PREVIOUS / "train_coverage.json",
                        *(parent_run.PREVIOUS / f"fit_{s}.npz" for s in SEEDS),
                    )
                },
            },
        ),
    )
    tape, danger, safe, raw = collect(base, cfg, inventory, h)
    raw_path = root / "train_raw.json"
    r.write_new(
        raw_path,
        dict(
            rows=raw,
            danger_episodes=len(danger),
            safe_episodes=len(safe),
            danger_shots=len({q["shot"] for q in danger}),
            safe_shots=len({q["shot"] for q in safe}),
        ),
    )
    fits = [
        prior.load_fit(parent_run.PREVIOUS / f"fit_{seed}.npz", base) for seed in SEEDS
    ]
    receipts = {}
    for name, floor in (("all", None), ("high", HIGH)):
        match = matching.match_pairs(danger, safe, high_floor=floor)
        pairs = paired_rows(danger, safe, match)
        receipts[name] = seal_cohort(base, root, name, pairs, match, tape, fits, h)
    if r.state_digest(base) != state or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("intervention probe mutated the qualified source")
    h.check()
    r.write_new(
        root / "completed.json",
        dict(
            status="SEALED_TRAIN_ONLY_PEAK_CONTRAST",
            protocol_sha256=h(protocol),
            raw_manifest_sha256=h(raw_path),
            pair_sha256=receipts,
            no_fit=True,
            held_signal_reads=0,
            held_outcome_reads=0,
            cuda_device=torch.cuda.get_device_name(),
            peak_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.monotonic() - start,
        ),
    )
    print("capacity effects sealed", time.monotonic() - start, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
