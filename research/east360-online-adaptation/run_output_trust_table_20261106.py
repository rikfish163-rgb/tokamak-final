"""Freeze real J-TEXT TRAIN action effects before fitting any selector."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import output_trust_forward_20261106 as fwd
import output_trust_step_20261106 as tr
import run_whole_block_update_20261106 as src
from peak_contrast_forward_20261106 import ARMS, SITE_NORM
from verify_confirmation_raw_replay_20261105 import Hashes


r, prior, old, rule, SEEDS, TAU = (
    src.r,
    src.prior,
    src.old,
    src.rule,
    src.SEEDS,
    src.TAU,
)
PLAN = r.ROOT / "OUTPUT_TRUST_PROTOCOL_20261106.md"
REFERENCE = r.ROOT / "probe_outputs/peak_contrast_first_alarm_20261106_v2.json"
MODES = ("free", "trusted")
SHOTS_PER_SHARD, EPISODES_PER_BATCH = 96, 96


def frozen_fits(base):
    return [prior.load_fit(src.PREVIOUS / f"fit_{s}.npz", base) for s in SEEDS]


def protocol_inputs(h, base):
    cfg, ref = src.source_inputs(h)
    paths = (
        PLAN,
        REFERENCE,
        Path(__file__),
        Path(fwd.__file__),
        Path(tr.__file__),
        Path(src.__file__),
        src.PREVIOUS / "protocol.json",
        src.PREVIOUS / "fit_completed.json",
        src.PREVIOUS / "train_coverage.json",
        *(src.PREVIOUS / f"fit_{s}.npz" for s in SEEDS),
    )
    if r.state_digest(base) != cfg["source_state_sha256"]:
        raise ValueError("same qualified J-TEXT source required")
    inputs = {**cfg["input_sha256"], **{str(p.resolve()): h(p) for p in paths}}
    return cfg, ref, inputs


def rows_for_shot(base, raw, native, t, start):
    """Causal action inventory, intentionally independent of any truth."""
    f = prior.signal_schedule(base, raw)
    js = np.flatnonzero(f["support"] & (f["maturity"] < len(t) - 1))
    m = f["maturity"][js]
    grid = m[:, None] + np.arange(1, 33)
    present = grid < len(t)
    q = np.minimum(grid, len(t) - 1)
    end = start + 127
    past = tr.reached_indices(m)
    return dict(
        maturity=m,
        ox=end + f["origin"][js],
        mx=end + m,
        reached=end + past,
        future=end + q,
        present=present,
        delta=f["delta"][js],
        valid=f["valid"][js],
        reference=native[past],
        frozen=np.where(present, native[q], np.float32(0)),
    )


def proposal(base, fits, tape, data):
    """Only reached diagnostic feedback, and only reached risk windows for trust."""
    ox, oa = src.tape_views(tape, data["ox"])
    mx, ma = src.tape_views(tape, data["mx"])
    states, descriptors, support, old_common = [], [], [], []
    for decoder, mixer, norm, grad_scale in fits:
        y = prior.normalize_targets(data["delta"], norm)
        x, basis, _, ok = rule.reached_bank(
            base, decoder, mixer, ox, oa, mx, ma, y, data["valid"], grad_scale
        )
        u, valid = fwd.action_directions(basis)
        states.append(u)
        descriptors.append(x)
        support.append(valid)
        old_common.append(ok)
    u, x, live = (
        torch.stack(states, 1),
        torch.stack(descriptors, 1),
        torch.stack(support, 1),
    )
    live &= torch.stack(old_common, 1).all(1)[:, None, None]
    u = u.masked_fill(~live[..., None], 0)
    n, seeds, arms, width = u.shape
    scales = u.new_tensor(tr.SCALES)
    v = (u[..., None, :] * scales[None, None, None, :, None]).reshape(n, -1, width)
    effects = []
    with torch.no_grad():
        for k in range(3):
            xx, aa = src.tape_views(tape, data["reached"][:, k])
            value = fwd.batched_risk(base, fwd.encode(base, xx, aa), v)
            delta = (
                value.reshape(n, seeds, arms, len(scales))
                - data["reference"][:, k, None, None, None]
            )
            effects.append(delta.masked_fill(~live[..., None], 0))
    shift = torch.stack(effects, -1)
    radius, largest = tr.select_symmetric(shift.flatten(0, 1), live.flatten(0, 1))
    radius = radius.reshape(n, seeds, arms).to(u.dtype)
    largest = largest.reshape(n, seeds, arms)
    ix = (radius[..., None] == scales).to(torch.long).argmax(-1)
    anchor = (
        shift[..., -1].gather(-1, ix[..., None]).squeeze(-1).masked_fill(radius == 0, 0)
    )
    return dict(
        state=torch.stack((u, u * radius[..., None]), 1),
        available=torch.stack((live, radius > 0), 1),
        descriptor=x,
        anchor=torch.stack((shift[..., 0, -1], anchor), 1),
        radius=radius,
        largest=largest,
        free_largest=shift[..., 0, :].abs().amax(-1),
    )


@torch.no_grad()
def dense_effects(base, tape, data, props):
    """Every present endpoint; compact fixed blocks, not selected peaks."""
    w = props["state"]
    n, modes, seeds, arms, width = w.shape
    shape = (n, modes, seeds, arms, 32)
    z = data["frozen"][:, None, None, None].expand(shape).clone()
    w = w.reshape(n, modes * seeds * arms, width)
    active = props["available"].reshape(n, -1)
    rows, clock = data["present"].nonzero(as_tuple=True)
    for start in range(0, len(rows), 160):
        stop = min(start + 160, len(rows))
        i, t = rows[start:stop], clock[start:stop]
        x, a = src.tape_views(tape, data["future"][i, t])
        value = fwd.batched_risk(base, fwd.encode(base, x, a), w[i])
        value = torch.where(active[i], value, data["frozen"][i, t, None])
        z[i, :, :, :, t] = value.reshape(-1, modes, seeds, arms)
    return z


def gather(base, rows):
    return {k: src.field(v, base.center.device) for k, v in rows.items()}


def measured_shard(base, fits, tape, rows):
    keys = (
        "available",
        "descriptor",
        "anchor",
        "radius",
        "largest",
        "free_largest",
        "future",
    )
    all_data = {k: [] for k in keys}
    parity = None
    for begin in range(0, len(rows["maturity"]), EPISODES_PER_BATCH):
        stop = min(begin + EPISODES_PER_BATCH, len(rows["maturity"]))
        data = gather(base, {k: v[begin:stop] for k, v in rows.items()})
        p = proposal(base, fits, tape, data)
        if parity is None:
            # On qualified weights verify an actual finite action against
            # the direct source-model hook, not just the frozen origin.
            n = min(12, len(data["mx"]))
            x, a = src.tape_views(tape, data["mx"][:n])
            u = p["state"][:n, 0, 0, -2]
            with torch.no_grad():
                direct = old.risk(base, x, a, u)
                cached = fwd.packed_risk(base, fwd.encode(base, x, a), u)
            parity = float((direct - cached).abs().max())
            if parity > 4e-5:
                raise ValueError(
                    "cached intervention differs from actual source model hook"
                )
        p["future"] = dense_effects(base, tape, data, p)
        for k in keys:
            all_data[k].append(p[k].cpu().numpy())
    if not all_data["future"]:
        raise ValueError("no reached state in the TRAIN shard")
    return {k: np.concatenate(v) for k, v in all_data.items()}, parity


def collect_shard(base, block, truth, advance, h):
    """Original TRAIN signals only; canonical labels are for offline teaching masks."""
    tapes, masks, records, episodes, clocks, curves = [], [], [], [], [], []
    start = 0
    for j, (ref, item) in enumerate(zip(block, truth, strict=True)):
        sid = ref["shot"]
        sha = h(r.io23.path_for("J-TEXT", sid))
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        if (
            sha != ref["sha256"]
            or sha != item["sha256"]
            or flag != item["disruptive"]
            or sid != item["shot"]
        ):
            raise ValueError(f"source identity or authoritative outcome changed: {sid}")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        t = raw["times"]
        x, a = prior.views(base, raw)
        z = prior.scan(base, x, a).cpu().numpy()
        p = rows_for_shot(base, raw, z, t, start)
        horizon = r.io23.effective_horizon_ms(sid, advance)[0]
        pos, protect, k = fwd.teaching_masks(
            t, z, p["maturity"], flag, down, horizon, TAU
        )
        episodes.append(
            {
                **p,
                "positive": pos,
                "protect": protect,
                "owner": np.full(len(pos), j, dtype=np.int64),
            }
        )
        records.append(
            dict(
                shot=sid,
                raw_sha256=sha,
                disruptive=flag,
                down=down if flag else None,
                horizon_ms=horizon,
                n_points=len(t),
                first=k,
                episodes=len(pos),
            )
        )
        clocks.append(t)
        curves.append(z)
        tapes.append(np.pad(raw["values"], ((127, 0), (0, 0))))
        masks.append(np.pad(raw["available"], ((127, 0), (0, 0))))
        start += len(t) + 127
    tape = gather(
        base, dict(values=np.concatenate(tapes), available=np.concatenate(masks))
    )
    data = {k: np.concatenate([p[k] for p in episodes]) for k in episodes[0]}
    return tape, data, records, np.concatenate(clocks), np.concatenate(curves)


def run(root):
    h, start = Hashes(), time.monotonic()
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if parent["threshold"] != r.THRESHOLD:
        raise ValueError("same qualified threshold required")
    cfg, inventory, inputs = protocol_inputs(h, base)
    census = r.read(prior.SOURCE_CENSUS)["source"]
    if [q["shot"] for q in census] != cfg["roles"]["train"]:
        raise ValueError("exact locked TRAIN shot set required")
    root.mkdir(parents=True, exist_ok=False)
    base.cuda()
    fits = frozen_fits(base)
    advance = r.simple_data.advanced_times()
    pp = root / "protocol.json"
    r.write_new(
        pp,
        dict(
            status="LOCKED_OUTPUT_TRUST_TABLE",
            input_sha256=inputs,
            roles=cfg["roles"],
            source_state_sha256=cfg["source_state_sha256"],
            seed_order=SEEDS,
            modes=MODES,
            actions=ARMS,
            limit=tr.LIMIT,
            scales=tr.SCALES,
            state_norm=SITE_NORM,
            threshold=r.THRESHOLD,
            logit_threshold=TAU,
            val_test_east_reads=0,
        ),
    )
    shards = []
    for i in range(0, len(inventory), SHOTS_PER_SHARD):
        j = min(i + SHOTS_PER_SHARD, len(inventory))
        tape, rows, shots, times, native = collect_shard(
            base, inventory[i:j], census[i:j], advance, h
        )
        effects, error = measured_shard(base, fits, tape, rows)
        shot_ids = np.asarray([q["shot"] for q in shots], dtype=np.int64)
        name = f"shard_{i // SHOTS_PER_SHARD:03d}"
        path, meta = root / f"{name}.npz", root / f"{name}.json"
        r.save_arrays(
            path,
            **effects,
            **{
                k: rows[k]
                for k in (
                    "maturity",
                    "present",
                    "positive",
                    "protect",
                    "reference",
                    "frozen",
                )
            },
            row_shot=shot_ids[rows["owner"]],
            times=times,
            native=native,
        )
        r.write_new(
            meta, dict(shots=shots, predictions_sha256=h(path), direct_parity=error)
        )
        shards.append(
            dict(
                archive=path.name,
                meta=meta.name,
                sha256=h(meta),
                n_windows=len(rows["maturity"]),
            )
        )
        h.check()
        print(
            f"TRAIN shots {j}/{len(inventory)}; windows {sum(q['n_windows'] for q in shards)}; {time.monotonic() - start:.1f}s",
            flush=True,
        )
    if r.state_digest(base) != cfg["source_state_sha256"] or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("base is not frozen")
    h.check()
    r.write_new(
        root / "completed.json",
        dict(
            status="SEALED_OUTPUT_TRUST_TRAIN",
            protocol_sha256=h(pp),
            shards=shards,
            val_test_east_reads=0,
            cuda_device=torch.cuda.get_device_name(),
            peak_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
