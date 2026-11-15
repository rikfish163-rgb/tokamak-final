"""Run TRAIN-only whole-block alignment, then seal label-blind full-shot forecasts."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import logit

import moving_peak_replay_20261105 as r
import paired_diagnostic_step_20261106 as old
import run_paired_diagnostic_update_20261106 as prior
import whole_block_loss_20261106 as loss
import whole_block_update_20261106 as rule
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = r.ROOT / "WHOLE_BLOCK_UPDATE_PROTOCOL_20261106.md"
PREVIOUS = r.ROOT / "probe_outputs/paired_diagnostic_update_20261106_v1"
ARMS = ("static", "old", "point", "block", "reverse", "score")
SEEDS = prior.SEEDS
TAU = float(logit(r.THRESHOLD))
CONTEXT, STEPS, BATCHES = 128, 512, (8, 4, 4)
INFER_BATCH = 128
PREFIX_ATOL = 8e-5


def source_inputs(h):
    previous, fit = prior.checked_protocol(PREVIOUS, h), prior.checked_fit(PREVIOUS, h)
    inventory_path = PREVIOUS / "train_coverage.json"
    inventory = r.read(inventory_path)["inventory"]
    if (
        h(inventory_path) != fit["train_coverage_sha256"]
        or [q["shot"] for q in inventory] != previous["roles"]["train"]
        or len(inventory) != 1353
    ):
        raise ValueError(
            "exact earlier TRAIN-only diagnostic fit and inventory required"
        )
    return previous, inventory


def field(v, device):
    return torch.as_tensor(v, device=device)


def tape_views(tape, endpoints):
    ix = endpoints[..., None] + torch.arange(1 - CONTEXT, 1, device=endpoints.device)
    return tape["values"][ix], tape["available"][ix]


def collect(base, previous, inventory, h):
    """Keep one small signal tape per shot, never materialize all future windows."""
    advance = r.simple_data.advanced_times()
    census = r.read(prior.SOURCE_CENSUS)["source"]
    if [q["shot"] for q in census] != previous["roles"]["train"]:
        raise ValueError("original TRAIN-only signal census required")
    tapes, masks, rows, provenance = [], [], [], []
    start = 0
    for k, (rec, original) in enumerate(zip(inventory, census, strict=True), 1):
        sid = rec["shot"]
        sha = h(r.io23.path_for("J-TEXT", sid))
        if sha != rec["sha256"] or sha != original["sha256"]:
            raise ValueError(f"original J-TEXT TRAIN bytes changed: {sid}")
        raw = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        if flag != original["disruptive"]:
            raise ValueError(f"authoritative TRAIN outcome changed: {sid}")
        horizon = r.io23.effective_horizon_ms(sid, advance)[0]
        f = prior.signal_schedule(base, raw)
        x, a = prior.views(base, raw)
        z = prior.scan(base, x, a).cpu().numpy()
        slots = old.write_slots(len(z), f["maturity"])
        teaching = loss.role_masks(
            raw["times"], z, slots, f["support"], flag, down, horizon, TAU
        )
        for slot, label, ix in teaching:
            query = np.full(old.BLOCK, start + int(ix[0]), dtype=np.int64)
            mask = np.arange(old.BLOCK) < len(ix)
            query[mask] = start + ix
            native = np.zeros(old.BLOCK, dtype=np.float32)
            native[mask] = z[ix]
            rows.append(
                dict(
                    shot=sid,
                    role=label,
                    ox=start + f["origin"][slot],
                    mx=start + f["maturity"][slot],
                    query=query,
                    mask=mask,
                    native=native,
                    delta=f["delta"][slot],
                    valid=f["valid"][slot],
                )
            )
        provenance.append(
            dict(
                shot=sid,
                raw_sha256=sha,
                counts=np.bincount([q[1] for q in teaching], minlength=3).tolist(),
            )
        )
        # Last 127 points from a different shot cannot enter this shot's input.
        tapes.append(np.pad(raw["values"], ((CONTEXT - 1, 0), (0, 0))))
        masks.append(np.pad(raw["available"], ((CONTEXT - 1, 0), (0, 0))))
        start += len(raw["times"]) + CONTEXT - 1
        if k == len(inventory) or k % 250 == 0:
            print(f"TRAIN full block inventory {k}/{len(inventory)}", flush=True)
    dev = base.center.device
    if not rows:
        raise ValueError("no supported source TRAIN teaching rows")
    data = {key: field(np.asarray([v[key] for v in rows]), dev) for key in rows[0]}
    # Metadata uses global indices with the 127-point causal left padding.
    for key in ("ox", "mx", "query"):
        data[key] += CONTEXT - 1
    tape = dict(
        values=field(np.concatenate(tapes), dev),
        available=field(np.concatenate(masks), dev),
    )
    return tape, data, provenance


def bank_on_tape(base, tape, data, fit):
    decoder, mixer, scale, g_ref = fit
    description, basis, common = [], [], []
    for ix in torch.arange(len(data["ox"]), device=scale.device).split(INFER_BATCH):
        ox, oa = tape_views(tape, data["ox"][ix])
        mx, ma = tape_views(tape, data["mx"][ix])
        target = prior.normalize_targets(data["delta"][ix], scale)
        desc, bank, _, ok = rule.reached_bank(
            base, decoder, mixer, ox, oa, mx, ma, target, data["valid"][ix], g_ref
        )
        description.append(desc)
        basis.append(bank)
        common.append(ok)
    return torch.cat(description), torch.cat(basis), torch.cat(common)


def draw_batches(data, common, seed, steps=STEPS):
    ids, roles, eligible = (
        q.detach().cpu().numpy() for q in (data["shot"], data["role"], common)
    )
    rng, groups = np.random.default_rng(seed), []
    for label in range(len(BATCHES)):
        k = np.flatnonzero((roles == label) & eligible)
        shots = np.unique(ids[k])
        if not len(shots):
            raise ValueError(f"no independently supported TRAIN shots for role {label}")
        groups.append([k[ids[k] == sid] for sid in shots])
    return np.asarray(
        [
            [
                rng.choice(g[j])
                for g, count in zip(groups, BATCHES, strict=True)
                for j in rng.integers(len(g), size=count)
            ]
            for _ in range(steps)
        ],
        dtype=np.int64,
    )


def future_logits(base, tape, data, ix, action, *, point=False):
    mask, native = data["mask"][ix], data["native"][ix]
    if point:
        # The point control needs exactly one forward per episode, mathematically
        # identical to taking the frozen maximum's index after a dense forward.
        j = native.masked_fill(~mask, -torch.inf).argmax(dim=1, keepdim=True)
        q = data["query"][ix].gather(1, j).squeeze(1)
        x, a = tape_views(tape, q)
        return native.scatter(1, j, old.risk(base, x, a, action)[:, None])
    row = mask.nonzero(as_tuple=True)[0]
    x, a = tape_views(tape, data["query"][ix][mask])
    z = old.risk(base, x, a, action[row])
    return native.masked_scatter(mask, z)


def train_feasibility(base, tape, data, basis, common):
    """TRAIN-only reachable examples under a small, declared finite direction set."""
    recover = ((data["role"] == loss.RECOVER) & common).nonzero().flatten()
    if not recover.numel():
        raise ValueError("no canonical timely support on native TRAIN misses")
    rows = []
    for ix in recover.split(24):
        q, mask = data["native"][ix], data["mask"][ix]
        at = q.masked_fill(~mask, -torch.inf).argmax(1)
        x, a = tape_views(tape, data["query"][ix, at])
        v = x.new_zeros((len(ix), old.WIDTH), requires_grad=True)
        (grad,) = torch.autograd.grad(old.risk(base, x, a, v).sum(), v)
        bank = basis[ix]
        signs = torch.bmm(bank, grad[..., None]).squeeze(-1).sign()
        with torch.no_grad():
            native = q.masked_fill(~mask, -torch.inf).amax(1)
            exact = [
                future_logits(base, tape, data, ix, rule.compose(bank, signs * c))
                .masked_fill(~mask, -torch.inf)
                .amax(1)
                for c in (0.475, 0.95, 1.9)
            ]
            best = torch.stack((native, *exact), 1).amax(1)
            rows.extend(
                dict(shot=int(sid), before=float(before), best=float(after))
                for sid, before, after in zip(
                    data["shot"][ix].cpu(), native.cpu(), best.cpu(), strict=True
                )
            )
    return dict(
        probes=rows,
        n_shots=len({q["shot"] for q in rows}),
        crossing_shots=sorted(
            {q["shot"] for q in rows if q["before"] < TAU <= q["best"]}
        ),
    )


def fit_arm(base, tape, data, descriptor, bank, batches, seed, *, point):
    torch.manual_seed(seed)
    net = rule.Combiner().to(base.center.device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.001)
    trace, interval = [], []
    for k, ix in enumerate(field(batches, base.center.device), 1):
        action = net(descriptor[ix], bank[ix])
        z = future_logits(base, tape, data, ix, action, point=point)
        objective = loss.peak_loss(
            z,
            data["native"][ix],
            data["mask"][ix],
            data["role"][ix],
            TAU,
            point=point,
        )
        objective = (objective + 0.02 * action.square().sum(1)).mean()
        opt.zero_grad(set_to_none=True)
        objective.backward()
        opt.step()
        interval.append(objective.detach())
        if k % 128 == 0:
            value = float(torch.stack(interval).mean())
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite source alignment loss")
            trace.append(value)
            interval.clear()
            print(
                f"seed {seed} {'point' if point else 'block'} {k}/{len(batches)}: {value:.6f}",
                flush=True,
            )
    if any(not torch.isfinite(v).all() for v in net.parameters()):
        raise FloatingPointError("nonfinite source combiner")
    return net.eval().requires_grad_(False), trace


def fit_source(base, root, cfg, inventory, h):
    tape, data, provenance = collect(base, cfg, inventory, h)
    r.write_new(root / "train_inventory.json", dict(rows=provenance))
    legacy = [prior.load_fit(PREVIOUS / f"fit_{s}.npz", base) for s in SEEDS]
    features = [bank_on_tape(base, tape, data, fit) for fit in legacy]
    common = torch.stack([v[2] for v in features]).all(0)
    caps = {
        str(s): train_feasibility(base, tape, data, features[i][1], common)
        for i, s in enumerate(SEEDS)
    }
    support = [
        int(data["shot"][common & (data["role"] == role)].unique().numel())
        for role in range(len(BATCHES))
    ]
    r.write_new(root / "train_feasibility.json", dict(roles=support, seeds=caps))
    print(
        "TRAIN unique shots by role",
        support,
        "reachable",
        [len(q["crossing_shots"]) for q in caps.values()],
        flush=True,
    )
    fits, receipts = [], {}
    for k, s in enumerate(SEEDS):
        desc, bank, _ = features[k]
        batches = draw_batches(data, common, s)
        opt = [
            fit_arm(base, tape, data, desc, bank, batches, s, point=point)
            for point in (True, False)
        ]
        nets = [v[0] for v in opt]
        path = root / f"fit_{s}.npz"
        r.save_arrays(
            path,
            **{
                f"{kind}_{key}": v.detach().cpu().numpy()
                for kind, net in zip(("point", "block"), nets, strict=True)
                for key, v in net.state_dict().items()
            },
        )
        fits.append((legacy[k], *nets))
        receipts[str(s)] = dict(
            sha256=h(path), point_loss=opt[0][1], block_loss=opt[1][1]
        )
    return fits, receipts


def inference_bank(base, fits, x, a, f, enabled):
    """Produce candidate states using only each reached origin and maturity."""
    m, dev = len(f["maturity"]), x.device
    states = x.new_zeros((m, len(SEEDS), 3, old.WIDTH))
    common = f["support"] & enabled
    eligible = np.flatnonzero(common)
    for j, ((decoder, mixer, scale, g_ref), point, block) in enumerate(fits):
        for ix in np.array_split(eligible, max(1, (len(eligible) + 127) // 128)):
            if not len(ix):
                continue
            o, t = f["origin"][ix], f["maturity"][ix]
            q = prior.normalize_targets(field(f["delta"][ix], dev), scale)
            v = field(f["valid"][ix], dev)
            desc, bank, original, ok = rule.reached_bank(
                base,
                decoder,
                mixer,
                x[o].contiguous(),
                a[o].contiguous(),
                x[t].contiguous(),
                a[t].contiguous(),
                q,
                v,
                g_ref,
            )
            with torch.no_grad():
                states[ix, j] = torch.stack(
                    (original, point(desc, bank), block(desc, bank)), 1
                )
            common[ix] &= ok.cpu().numpy()
    states[~common] = 0
    return states, common


@torch.no_grad()
def evaluate_paths(base, x, a, frozen, states, common, maturity, slots):
    n = len(x)
    z = frozen.cpu().numpy()
    logits = np.broadcast_to(z[:, None, None], (n, len(SEEDS), len(ARMS))).copy()
    active = np.zeros(n, dtype=bool)
    active[slots >= 0] = common[slots[slots >= 0]]
    idx, mature = np.flatnonzero(active), np.flatnonzero(common)
    shifts = frozen.new_zeros((len(common), len(SEEDS)))
    for j in range(len(SEEDS)):
        for kk in np.array_split(mature, max(1, (len(mature) + 127) // 128)):
            if len(kk):
                t = maturity[kk]
                shifts[kk, j] = (
                    old.risk(
                        base, x[t].contiguous(), a[t].contiguous(), states[kk, j, 2]
                    )
                    - frozen[t]
                )
        for k in range(0, len(idx), INFER_BATCH):
            t = idx[k : k + INFER_BATCH]
            xx = x[t, None].expand(-1, 4, -1, -1).reshape(-1, CONTEXT, 23)
            aa = a[t, None].expand(-1, 4, -1, -1).reshape_as(xx)
            u = states[slots[t], j]
            u = torch.cat((u, -u[:, 2:3]), dim=1)
            z_new = old.risk(base, xx, aa, u.reshape(-1, old.WIDTH))
            logits[t, j, 1:5] = z_new.reshape(-1, 4).cpu().numpy()
            logits[t, j, 5] = (frozen[t] + shifts[slots[t], j]).cpu().numpy()
    if not np.isfinite(logits).all() or not torch.isfinite(shifts).all():
        raise FloatingPointError("sealed full-shot logits must be finite")
    return logits, active, shifts.cpu().numpy()


def predict(base, fits, raw, *, enabled=True):
    x, a = prior.views(base, raw)
    frozen = prior.scan(base, x, a)
    f = prior.signal_schedule(base, raw)
    slots = old.write_slots(len(x), f["maturity"])
    states, common = inference_bank(base, fits, x, a, f, enabled)
    logits, active, shifts = evaluate_paths(
        base, x, a, frozen, states, common, f["maturity"], slots
    )
    norm = states.norm(dim=-1)
    if not torch.isfinite(norm).all() or (norm > old.RADIUS + 2e-6).any():
        raise ValueError("bounded temporary weights required")
    unit = states / norm.clamp_min(1e-8)[..., None]
    return dict(
        time=raw["times"],
        logits=logits,
        active=active,
        maturity=f["maturity"],
        common=common,
        norms=norm.cpu().numpy(),
        offsets=shifts,
        cos=(unit[:, :, 1] * unit[:, :, 2]).sum(-1).cpu().numpy(),
    )


def seal_source(base, fits, root, ids, protocol_sha, fit_sha, h):
    place = root / "val"
    place.mkdir()
    rows, prefix = [], []
    with patch.object(
        r.io23, "read_outcome", side_effect=RuntimeError("VAL outcomes are sealed")
    ):
        for rank, sid in enumerate(ids, 1):
            sha = h(r.io23.path_for("J-TEXT", sid))
            raw = r.simple_data.read_shot("J-TEXT", sid)
            pred = predict(base, fits, raw)
            if rank in (2, 40, 120):
                stop = min(103, len(raw["times"]) - 1)
                if stop >= 65:
                    trunc = {k: raw[k][:stop] for k in ("times", "values", "available")}
                    short = predict(base, fits, trunc)
                    err = float(np.abs(pred["logits"][:stop] - short["logits"]).max())
                    if err > PREFIX_ATOL or not np.array_equal(
                        pred["active"][:stop], short["active"]
                    ):
                        raise ValueError(
                            "unarrived data altered a prior score or write"
                        )
                    prefix.append(
                        dict(
                            shot=sid,
                            points=stop,
                            error=err,
                            active=int(short["active"].sum()),
                        )
                    )
            path = place / f"{rank:04d}_{sid}.npz"
            r.save_arrays(path, **pred)
            rows.append(
                dict(
                    shot=sid, rank=rank, raw_sha256=sha, sha256=h(path), file=path.name
                )
            )
            if rank % 40 == 0 or rank == len(ids):
                print(f"blind VAL sealed {rank}/{len(ids)}", flush=True)
    if not any(q["active"] > 0 for q in prefix):
        raise ValueError("at least one label-blind active prefix witness required")
    h.check()
    r.write_new(
        place / "completed.json",
        dict(
            status="SEALED_WHOLE_BLOCK_VAL",
            protocol_sha256=protocol_sha,
            fit_sha256=fit_sha,
            rows=rows,
            prefix=prefix,
            held_outcome_reads=0,
        ),
    )


def load_nets(root, base):
    fits = []
    for s in SEEDS:
        leg = prior.load_fit(PREVIOUS / f"fit_{s}.npz", base)
        with np.load(root / f"fit_{s}.npz", allow_pickle=False) as data:
            nets = [rule.Combiner().to(base.center.device) for _ in range(2)]
            for name, net in zip(("point", "block"), nets, strict=True):
                net.load_state_dict(
                    {
                        k: field(data[f"{name}_{k}"], base.center.device)
                        for k in net.state_dict()
                    },
                    strict=True,
                )
                net.eval().requires_grad_(False)
        fits.append((leg, *nets))
    return fits


def run_source(root):
    h, start = Hashes(), time.monotonic()
    r.configure()
    cfg, inventory = source_inputs(h)
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    state = r.state_digest(base)
    if state != cfg["source_state_sha256"] or parent["threshold"] != r.THRESHOLD:
        raise ValueError("qualified frozen source and unchanged threshold required")
    base.cuda()
    root.mkdir(parents=True, exist_ok=False)
    pp, fp = root / "protocol.json", root / "fit_completed.json"
    inputs = {
        **cfg["input_sha256"],
        **{
            str(p.resolve()): h(p)
            for p in (
                PLAN,
                Path(__file__),
                Path(loss.__file__),
                Path(rule.__file__),
                PREVIOUS / "protocol.json",
                PREVIOUS / "fit_completed.json",
                PREVIOUS / "train_coverage.json",
                *(PREVIOUS / f"fit_{s}.npz" for s in SEEDS),
            )
        },
    }
    r.write_new(
        pp,
        dict(
            status="LOCKED_WHOLE_BLOCK_PROTOCOL",
            input_sha256=inputs,
            roles=cfg["roles"],
            seeds=SEEDS,
            arms=ARMS,
            threshold=r.THRESHOLD,
            logit_threshold=TAU,
            state_sha256=state,
            source_test_reads=0,
        ),
    )
    fits, results = fit_source(base, root, cfg, inventory, h)
    pp_sha = h(pp)
    r.write_new(
        fp,
        dict(
            status="WHOLE_BLOCK_TRAIN_ONLY_FIT",
            protocol_sha256=pp_sha,
            train_inventory_sha256=h(root / "train_inventory.json"),
            train_feasibility_sha256=h(root / "train_feasibility.json"),
            fits=results,
            fit_steps=STEPS,
            current_labels_used=False,
        ),
    )
    ff_sha = h(fp)
    seal_source(base, fits, root, cfg["roles"]["val"], pp_sha, ff_sha, h)
    if r.state_digest(base) != state or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("frozen base was not preserved")
    h.check()
    r.write_new(
        root / "source_done.json",
        dict(
            status="COMPLETE_WHOLE_BLOCK_SOURCE",
            protocol_sha256=pp_sha,
            fit_sha256=ff_sha,
            seal_sha256=h(root / "val" / "completed.json"),
            cuda_device=torch.cuda.get_device_name(),
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.monotonic() - start,
            east_performance_replay=False,
        ),
    )
    print("source completed", time.monotonic() - start, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run_source(p.parse_args().root)
