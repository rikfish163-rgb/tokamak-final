"""Fit on source TRAIN, then seal actual short-block parameter-write replays."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import alarm_order_conditional_20261106 as extra
import moving_peak_replay_20261105 as r
import paired_diagnostic_features_20261106 as features
import paired_diagnostic_step_20261106 as step
from fit_layer0_arrival_motion_20261106 import past_arrays
from run_deep_miss_history_20261106 import group_weights
from trajectory_replay_20261105 import baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes

PLAN = r.ROOT / "PAIRED_DIAGNOSTIC_UPDATE_PROTOCOL_20261106.md"
SOURCE_CENSUS = r.ROOT / "probe_outputs/prealarm_history_support_20261106_v1.json"
SEEDS = (131, 317, 563)
ARMS = ("static", "real", "reverse", "unsigned", "zero_target", "score")
FIT_BATCH, INFER_BATCH = 128, 256
SUPPORT_FLOORS = (151, 699, 113)
PREFIT_EPOCHS, META_EPOCHS = 20, 8


def train_counts(rows):
    by_shot = np.asarray([r["counts"] for r in rows], dtype=np.int64)
    shots = (by_shot > 0).sum(0).tolist()
    return dict(
        shots=shots,
        windows=by_shot.sum(0).tolist(),
        passes=all(a >= b for a, b in zip(shots, SUPPORT_FLOORS, strict=True)),
    )


def commit_trace(z, ix, new, score):
    if (
        z.ndim != 1
        or ix.ndim != 1
        or ix.dtype.kind not in "iu"
        or (ix < 64).any()
        or (ix >= len(z)).any()
        or (np.diff(ix) <= 0).any()
        or new.shape != (len(ix), len(SEEDS), 4)
        or score.shape != (len(ix), len(SEEDS))
        or not all(np.isfinite(a).all() for a in (z, new, score))
    ):
        raise ValueError("aligned finite causal rows after maturity required")
    out = np.broadcast_to(z[:, None, None], (len(z), len(SEEDS), len(ARMS))).copy()
    out[ix, :, 1:5], out[ix, :, 5] = new, score
    return out


def to_device(x, base):
    return torch.as_tensor(x, device=base.center.device)


def signal_schedule(base, shot):
    return features.schedule(
        shot["values"],
        shot["available"],
        base.center.detach().cpu().numpy(),
        base.scale.detach().cpu().numpy(),
    )


def views(base, shot):
    return r.policy.window_views(
        to_device(shot["values"], base), to_device(shot["available"], base)
    )


@torch.no_grad()
def scan(base, x, a):
    return torch.cat(
        [
            step.risk(
                base,
                x[i : i + INFER_BATCH].contiguous(),
                a[i : i + INFER_BATCH].contiguous(),
            )
            for i in range(0, len(x), INFER_BATCH)
        ]
    )


def collect(base, roles, h):
    census = r.read(SOURCE_CENSUS)
    if [q["shot"] for q in census["source"]] != roles["train"]:
        raise ValueError("source fit must match the exact original TRAIN census")
    advance, rows = r.simple_data.advanced_times(), []
    values = {
        k: []
        for k in (
            "ox",
            "oa",
            "qx",
            "qa",
            "delta",
            "valid",
            "cls",
            "native",
            "group",
            "shot",
        )
    }
    for j, receipt in enumerate(census["source"], 1):
        sid = receipt["shot"]
        path = r.io23.path_for("J-TEXT", sid)
        before = h(path)
        raw = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        if before != receipt["sha256"] or flag != receipt["disruptive"]:
            raise ValueError(
                "source bytes or outcome changed since the coverage census"
            )
        horizon, _, _ = r.io23.effective_horizon_ms(sid, advance)
        f = signal_schedule(base, raw)
        x, a = views(base, raw)
        z = scan(base, x, a).cpu().numpy()
        slots, q, g = step.training_queries(
            raw["times"], z, f["maturity"], f["support"], flag, down, horizon
        )
        if len(q):
            ox, oa = past_arrays(raw["values"], raw["available"], f["origin"][slots])
            qx, qa = past_arrays(raw["values"], raw["available"], q)
            with torch.no_grad():
                hidden, _ = step.native(base, to_device(ox, base), to_device(oa, base))
            for k, data in dict(
                ox=ox,
                oa=oa,
                qx=qx,
                qa=qa,
                delta=f["delta"][slots],
                valid=f["valid"][slots],
                cls=hidden.cpu().numpy(),
                native=z[q],
                group=g,
                shot=np.full(len(q), sid),
            ).items():
                values[k].append(data)
        rows.append(
            dict(
                shot=sid,
                sha256=before,
                counts=np.bincount(g, minlength=3).tolist(),
                supported_maturities=int(f["support"].sum()),
                length=len(z),
            )
        )
        if j % 200 == 0 or j == len(roles["train"]):
            print(f"TRAIN episode inventory {j}/{len(roles['train'])}", flush=True)
    return values, rows


def normalize_targets(delta, scale):
    return (delta / scale).clamp(-4.0, 4.0)


def data_tensor(parts, device):
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    mass = group_weights(arrays["shot"], arrays["group"])
    if not arrays["valid"].any(axis=0).all():
        raise ValueError(
            "each diagnostic channel requires source normalization support"
        )
    scale = np.array(
        [
            max(0.05, np.median(np.abs(arrays["delta"][arrays["valid"][:, i], i])))
            for i in range(arrays["delta"].shape[1])
        ],
        dtype=np.float32,
    )
    if not np.isfinite(scale).all():
        raise ValueError(
            "each diagnostic channel requires source normalization support"
        )
    # Episode tensors stay on CPU; only the active optimizer mini-batch is on GPU.
    return {
        k: torch.from_numpy(a) for k, a in (arrays | {"mass": mass}).items()
    }, torch.as_tensor(scale, device=device)


def fit_one(base, d, scale, seed):
    device, n = scale.device, len(d["group"])
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    decoder = step.Decoder().to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=0.003, weight_decay=0.001)
    aux_log, meta_log = [], []

    def batch(key, i):
        return d[key][i].to(device)

    def objective(per, i):
        return (per * batch("mass", i)).sum() * (n / len(i))

    for _ in range(PREFIT_EPOCHS):
        running = []
        for ix in torch.randperm(n, generator=g).split(FIT_BATCH):
            y = normalize_targets(batch("delta", ix), scale)
            prediction = decoder(batch("cls", ix))
            family, support = step.family_loss(prediction, y, batch("valid", ix))
            loss = objective(family.sum(1) / support.sum(1).clamp_min(1), ix)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running.append(loss.detach())
        aux_log.append(float(torch.stack(running).mean()))
    decoder.eval().requires_grad_(False)

    gradient_bank, descriptors, families = [], [], []
    for ix in torch.arange(n).split(FIT_BATCH):
        y = normalize_targets(batch("delta", ix), scale)
        valid = batch("valid", ix)
        _, basis, present = step.bank(
            base, decoder, batch("ox", ix), batch("oa", ix), y, valid
        )
        gradient_bank.append(basis)
        descriptors.append(step.mix_inputs(y, valid))
        families.append(present)
    basis, desc, present = (
        torch.cat(v) for v in (gradient_bank, descriptors, families)
    )
    if not torch.isfinite(basis).all() or (present.sum(1) < 2).any():
        raise ValueError("invalid source gradient cache or group support")
    g_ref = float(
        (basis.sum(1) / present.sum(1, keepdim=True))
        .norm(dim=1)
        .median()
        .clamp_min(1e-4)
    )
    mixer = step.Mixer().to(device)
    opt = torch.optim.AdamW(mixer.parameters(), lr=0.01, weight_decay=0.001)
    for epoch in range(META_EPOCHS):
        running = []
        for ix in torch.randperm(n, generator=g).split(FIT_BATCH):
            w = mixer(desc[ix], present[ix])
            u = step.effects((basis[ix] * w[..., None]).sum(1), g_ref)
            z = step.risk(base, batch("qx", ix), batch("qa", ix), u)
            role, native = batch("group", ix), batch("native", ix)
            per = F.binary_cross_entropy_with_logits(
                z, (role == 0).float(), reduction="none"
            )
            per = per + 0.25 * (z - native).relu().square() * (role == 1)
            loss = objective(per, ix)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running.append(loss.detach())
        meta_log.append(float(torch.stack(running).mean()))
        print(
            f"seed {seed} aligned epoch {epoch + 1}/{META_EPOCHS}: {meta_log[-1]:.6f}",
            flush=True,
        )
    mixer.eval().requires_grad_(False)
    if not np.isfinite([*aux_log, *meta_log, g_ref]).all() or any(
        not torch.isfinite(v).all()
        for model in (decoder, mixer)
        for v in model.parameters()
    ):
        raise FloatingPointError("source auxiliary fit is not finite")
    return (decoder, mixer, scale, g_ref), dict(
        auxiliary=aux_log, aligned=meta_log, g_ref=g_ref
    )


def save_fit(path, fit):
    decoder, mixer, scale, g_ref = fit
    arrays = {
        f"{tag}_{k}": v.detach().cpu().numpy()
        for tag, net in (("decoder", decoder), ("mixer", mixer))
        for k, v in net.state_dict().items()
    }
    r.save_arrays(path, **arrays, scale=scale.cpu().numpy(), g_ref=np.asarray(g_ref))


def load_fit(path, base):
    with np.load(path, allow_pickle=False) as a:
        decoder, mixer = (
            step.Decoder().to(base.center.device),
            step.Mixer().to(base.center.device),
        )
        for tag, net in (("decoder", decoder), ("mixer", mixer)):
            net.load_state_dict(
                {k: to_device(a[f"{tag}_{k}"], base) for k in net.state_dict()},
                strict=True,
            )
            net.eval().requires_grad_(False)
        scale, g_ref = to_device(a["scale"], base), float(a["g_ref"])
    if (
        scale.shape != (11,)
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.isfinite(g_ref)
        or g_ref < float(np.float32(1e-4))
    ):
        raise ValueError("frozen transition normalizer and reference gradient required")
    return decoder, mixer, scale, g_ref


def stack_actions(base, fits, x, a, f, eligible):
    """Generate all matched directions without reading any later signal."""
    m = len(f["maturity"])
    device = x.device
    u = torch.zeros((m, len(SEEDS), 4, step.WIDTH), dtype=x.dtype, device=device)
    common = eligible.copy()
    decay = np.zeros((m, len(SEEDS), 2), dtype=np.float32)
    for seed_ix, fit in enumerate(fits):
        decoder, mixer, scale, g_ref = fit
        for i in range(0, int(eligible.sum()), FIT_BATCH):
            k = np.flatnonzero(eligible)[i : i + FIT_BATCH]
            o = f["origin"][k]
            target = normalize_targets(to_device(f["delta"][k], base), scale)
            valid = to_device(f["valid"][k], base)
            v, ok, p, _w = step.interventions(
                base,
                decoder,
                mixer,
                x[o].contiguous(),
                a[o].contiguous(),
                target,
                valid,
                g_ref,
            )
            u[k, seed_ix] = torch.stack(v, 1)
            common[k] &= ok.cpu().numpy()
            with torch.no_grad():
                family, present = step.family_loss(p, target, valid)
                z, _ = step.family_loss(torch.zeros_like(p), target, valid)
                decay[k, seed_ix, 0] = (family.sum(1) / present.sum(1)).cpu().numpy()
                decay[k, seed_ix, 1] = (z.sum(1) / present.sum(1)).cpu().numpy()
    u[~common] = 0.0
    return u, common, decay


@torch.no_grad()
def trace_active(base, x, a, z, u, maturities, slots, active_ix):
    n, device = len(active_ix), x.device
    curves = np.empty((n, len(SEEDS), 4), np.float32)
    scores = np.empty((n, len(SEEDS)), np.float32)
    anchor_shift = torch.zeros(
        (len(maturities), len(SEEDS)), dtype=x.dtype, device=device
    )
    active_slots = np.unique(slots[active_ix])
    for j in range(len(SEEDS)):
        for b in range(0, len(active_slots), INFER_BATCH):
            k = active_slots[b : b + INFER_BATCH]
            t = maturities[k]
            anchor_shift[k, j] = (
                step.risk(base, x[t].contiguous(), a[t].contiguous(), u[k, j, 0]) - z[t]
            )
        for b in range(0, n, INFER_BATCH):
            t = active_ix[b : b + INFER_BATCH]
            xq, aq, state = x[t].contiguous(), a[t].contiguous(), u[slots[t], j]
            vv = xq[:, None].expand(-1, 4, -1, -1).reshape(-1, 128, 23)
            aa = aq[:, None].expand(-1, 4, -1, -1).reshape_as(vv)
            values = step.risk(base, vv, aa, state.reshape(-1, step.WIDTH))
            curves[b : b + len(t), j] = values.reshape(-1, 4).cpu().numpy()
            scores[b : b + len(t), j] = (z[t] + anchor_shift[slots[t], j]).cpu().numpy()
    return curves, scores, anchor_shift.cpu().numpy()


def predict(base, fits, raw, enabled):
    x, a = views(base, raw)
    z = scan(base, x, a)
    f = signal_schedule(base, raw)
    slots = step.write_slots(len(x), f["maturity"])
    # Calculate a release at an exact prefix boundary even if no later sample
    # exists in this call; only existing later endpoints can consume the state.
    u, common, aux = stack_actions(base, fits, x, a, f, f["support"] & enabled)
    active = np.zeros(len(x), dtype=bool)
    active[slots >= 0] = common[slots[slots >= 0]]
    ix = np.flatnonzero(active)
    other, scores, offsets = trace_active(base, x, a, z, u, f["maturity"], slots, ix)
    norms = u.norm(dim=-1)
    if (norms > step.RADIUS + 2e-6).any() or not torch.allclose(
        norms[..., :1].expand_as(norms), norms, atol=2e-6, rtol=0
    ):
        raise ValueError("bounded equal-norm controls did not match")
    unit = u / norms.clamp_min(1e-8)[..., None]
    cosine = (unit[:, :, :1] * unit[:, :, 1:]).sum(-1)
    return dict(
        time=raw["times"],
        logits=commit_trace(z.cpu().numpy(), ix, other, scores),
        active=active,
        maturity=f["maturity"],
        visible=f["support"],
        common=common,
        norms=norms.cpu().numpy(),
        cosine=cosine.cpu().numpy(),
        offsets=offsets,
        auxiliary=aux,
    )


def dataset_context(h):
    fixed, fp, fb = reference()
    conditional, cp, cb, map_path, mapping = extra.references(h)
    return dict(
        order=fixed,
        config=fp,
        baseline=fb,
        conditional=conditional,
        extra_config=cp,
        extra_baseline=cb,
        map_path=map_path,
        mapping=mapping,
    )


def seal_role(base, fits, root, role, ids, context, protocol, fit_receipt, h):
    place = root / role
    place.mkdir()
    rows, witness = [], []
    with patch.object(
        r.io23, "read_outcome", side_effect=RuntimeError("held outcomes are sealed")
    ):
        for rank, sid in enumerate(ids, 1):
            source = role == "val"
            dev = "J-TEXT" if source else "EAST"
            path = r.io23.path_for(dev, sid)
            before = h(path)
            if source:
                raw = r.simple_data.read_shot(dev, sid)
                p = None
            else:
                fixed = role == "fixed"
                cfg = context["config"] if fixed else context["extra_config"]
                if before != cfg["raw_sha256"][str(sid)]:
                    raise ValueError("held EAST source bytes changed")
                raw = r.simple_data.read_shot(dev, sid, rate_contract="nominal")
                t, p = (
                    baseline_for(rank, sid, context["baseline"])
                    if fixed
                    else extra.baseline(h, rank, sid, context["extra_baseline"], before)
                )
                if not np.array_equal(t, raw["times"]):
                    raise ValueError("original EAST grid changed")
                raw = {
                    **raw,
                    **r.prepare_actual(raw, rank if fixed else 21, context["mapping"]),
                }
            enabled = role != "fixed" or rank > 20
            pred = predict(base, fits, raw, enabled)
            if p is not None and not np.allclose(
                torch.from_numpy(pred["logits"][:, 0, 0]).sigmoid().numpy(),
                p,
                atol=2e-6,
                rtol=0,
            ):
                raise ValueError(
                    "original static probabilities differ from archived EAST curve"
                )
            if rank in (2, 40, 120):
                # Truncate inside the available observation prefix, after an
                # early block is already active. Never include the cut suffix.
                stop = min(103, len(raw["times"]) - 1)
                if stop >= 65:
                    trunc = {
                        **raw,
                        **{k: raw[k][:stop] for k in ("times", "values", "available")},
                    }
                    part = predict(base, fits, trunc, enabled)
                    error = float(
                        np.max(np.abs(part["logits"] - pred["logits"][:stop]))
                    )
                    if error > 5e-5 or not np.array_equal(
                        part["active"], pred["active"][:stop]
                    ):
                        raise ValueError(
                            "unarrived observations changed earlier predictions"
                        )
                    witness.append(
                        dict(
                            shot=sid,
                            points=stop,
                            error=error,
                            active=int(part["active"].sum()),
                        )
                    )
            dest = place / f"{rank:04d}_{sid}.npz"
            r.save_arrays(dest, **pred)
            rows.append(
                dict(
                    shot=sid,
                    rank=rank,
                    raw_sha256=before,
                    file=dest.name,
                    sha256=h(dest),
                )
            )
            if rank % 100 == 0 or rank == len(ids):
                print(f"sealed {role} {rank}/{len(ids)}", flush=True)
    h.check()
    r.write_new(
        place / "completed.json",
        dict(
            status="SEALED_PAIRED_DIAGNOSTIC_UPDATE",
            role=role,
            rows=rows,
            prefix=witness,
            protocol_sha256=h(protocol),
            fit_sha256=h(fit_receipt),
            held_outcome_reads=0,
        ),
    )


def checked_protocol(root, h):
    p = r.read(root / "protocol.json")
    if (
        p["status"] != "PAIRED_DIAGNOSTIC_UPDATE_PROTOCOL"
        or tuple(p["seeds"]) != SEEDS
        or tuple(p["arms"]) != ARMS
        or p["threshold"] != r.THRESHOLD
        or p["roles"] != r.simple_data.source_roles()
        or not all(h(k) == v for k, v in p["input_sha256"].items())
    ):
        raise ValueError("unchanged source-only signed transition experiment required")
    return p


def checked_fit(root, h):
    p = r.read(root / "fit_completed.json")
    if (
        p["status"] != "SOURCE_ONLY_SIGNED_DIAGNOSTIC_FIT"
        or h(root / "protocol.json") != p["protocol_sha256"]
        or not all(
            h(root / f"fit_{s}.npz") == p["fits"][str(s)]["sha256"] for s in SEEDS
        )
    ):
        raise ValueError("unchanged source fits required")
    return p


def run(root, stage):
    h, start = Hashes(), time.monotonic()
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    context = dataset_context(h)
    state = r.state_digest(base)
    if (
        state != context["config"]["source_state_sha256"]
        or state != context["extra_config"]["source_state_sha256"]
        or parent["threshold"] != r.THRESHOLD
    ):
        raise ValueError("qualified frozen source and alarm threshold required")
    base.cuda()
    pp, ff = root / "protocol.json", root / "fit_completed.json"
    if stage == "source":
        if not r.io23.JTEXT_INDEX.is_file():
            raise ValueError("canonical source TRAIN split required")
        root.mkdir(parents=True, exist_ok=False)
        roles = r.simple_data.source_roles()
        paths = (
            PLAN,
            SOURCE_CENSUS,
            r.DEFAULT_PARENT,
            Path(__file__),
            Path(features.__file__),
            Path(step.__file__),
            Path(r.policy.__file__),
            Path(r.latent.__file__),
            Path(r.io23.__file__),
            Path(r.simple_data.__file__),
            Path(r.prepare_actual.__code__.co_filename),
            r.io23.JTEXT_INDEX,
            r.io23.JTEXT_ADVANCED_TIME_PATH,
            context["map_path"],
            r.SOURCE / "completed.json",
            r.SOURCE / "protocol.json",
            extra.ROSTER,
            extra.PREVIOUS / "completed.json",
            extra.PREVIOUS / "protocol.json",
            Path(past_arrays.__code__.co_filename),
            Path(group_weights.__code__.co_filename),
        )
        r.write_new(
            pp,
            dict(
                status="PAIRED_DIAGNOSTIC_UPDATE_PROTOCOL",
                roles=roles,
                fixed=context["order"],
                conditional=context["conditional"],
                threshold=r.THRESHOLD,
                seeds=SEEDS,
                arms=ARMS,
                input_sha256={str(p.resolve()): h(p) for p in paths},
                source_state_sha256=state,
                current_labels_used=False,
                source_test_signal_reads=0,
            ),
        )
        parts, inventory = collect(base, roles, h)
        coverage = train_counts(inventory)
        r.write_new(root / "train_coverage.json", dict(**coverage, inventory=inventory))
        if not coverage["passes"]:
            raise ValueError(
                f"declared block schedule cannot support source classes: {coverage}"
            )
        print("source coverage", coverage, flush=True)
        d, scale = data_tensor(parts, base.center.device)
        del parts
        fits, receipts = [], {}
        for seed in SEEDS:
            fit, diag = fit_one(base, d, scale, seed)
            dest = root / f"fit_{seed}.npz"
            save_fit(dest, fit)
            fits.append(fit)
            receipts[str(seed)] = dict(sha256=h(dest), **diag)
        r.write_new(
            ff,
            dict(
                status="SOURCE_ONLY_SIGNED_DIAGNOSTIC_FIT",
                protocol_sha256=h(pp),
                fits=receipts,
                train_coverage_sha256=h(root / "train_coverage.json"),
            ),
        )
        seal_role(base, fits, root, "val", roles["val"], context, pp, ff, h)
    else:
        p, _ = checked_protocol(root, h), checked_fit(root, h)
        gate = r.read(root / "val_audit.json")
        if (
            gate["status"] != "RECOUNT_PAIRED_DIAGNOSTIC_UPDATE"
            or gate["role"] != "val"
            or gate["seal_sha256"] != h(root / "val" / "completed.json")
            or gate["gate"]["passed"] is not True
        ):
            raise ValueError("source VAL gate did not authorize a new EAST replay")
        if p["fixed"] != context["order"] or p["conditional"] != context["conditional"]:
            raise ValueError("locked target roster changed")
        fits = [load_fit(root / f"fit_{s}.npz", base) for s in SEEDS]
        for role in ("fixed", "conditional"):
            seal_role(base, fits, root, role, p[role], context, pp, ff, h)
    if r.state_digest(base) != state or base.risk_head._forward_pre_hooks:
        raise ValueError("source state or hook ownership changed during replay")
    r.write_new(
        root / f"{stage}_done.json",
        dict(
            status="COMPLETE_PAIRED_DIAGNOSTIC_STAGE",
            stage=stage,
            protocol_sha256=h(pp),
            fit_sha256=h(ff),
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "east"))
    parser.add_argument("root", type=Path)
    a = parser.parse_args()
    run(a.root, a.stage)
