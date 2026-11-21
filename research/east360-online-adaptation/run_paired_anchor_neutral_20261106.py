"""Separate anchor score motion from real finite writes on sealed source VAL."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import paired_anchor_neutral_20261106 as solver
import paired_diagnostic_metrics_20261106 as metrics
import run_paired_diagnostic_update_20261106 as prior
from verify_paired_diagnostic_update_20261106 import verify_prediction

r, step, SEEDS = prior.r, prior.step, prior.SEEDS
PLAN = r.ROOT / "PAIRED_ANCHOR_NEUTRAL_PLAN_20261106.md"
ARMS = (
    "static",
    "neutral_real",
    "neutral_reverse",
    "neutral_unsigned",
    "raw_real_matched",
    "neutral_anchor_score",
)
COLMAP = dict(zip(metrics.ARMS, ARMS, strict=True))


def input_rows(root, h):
    protocol, _ = prior.checked_protocol(root, h), prior.checked_fit(root, h)
    seal = root / "val" / "completed.json"
    audit = root / "val_audit.json"
    a, s = r.read(audit), r.read(seal)
    if (
        a["status"] != "RECOUNT_PAIRED_DIAGNOSTIC_UPDATE"
        or a["role"] != "val"
        or a["seal_sha256"] != h(seal)
        or s["held_outcome_reads"] != 0
        or [q["shot"] for q in s["rows"]] != protocol["roles"]["val"]
    ):
        raise ValueError("verified source VAL archive required")
    for i, q in enumerate(s["rows"], 1):
        sid = q["shot"]
        if (
            (q["rank"], q["file"]) != (i, f"{i:04d}_{sid}.npz")
            or h(root / "val" / q["file"]) != q["sha256"]
            or h(r.io23.path_for("J-TEXT", sid)) != q["raw_sha256"]
        ):
            raise ValueError("source VAL seal changed")
    return protocol, s["rows"], seal, audit


def predict(base, fits, raw, archived=None):
    x, a = prior.views(base, raw)
    f = prior.signal_schedule(base, raw)
    z = prior.scan(base, x, a)
    raw_u, old_common, _ = prior.stack_actions(base, fits, x, a, f, f["support"])
    maturity = f["maturity"]
    dtype, device = x.dtype, x.device
    final = torch.zeros_like(raw_u)
    common = np.zeros(len(maturity), dtype=bool)
    root_ok = np.zeros((len(maturity), len(SEEDS), 3), dtype=bool)
    angles = np.zeros(root_ok.shape, dtype=np.float32)
    residuals = np.zeros_like(angles)
    candidates = np.flatnonzero(old_common)
    if len(candidates):
        ts = maturity[candidates]
        xx, aa = x[ts].contiguous(), a[ts].contiguous()
        with torch.enable_grad():
            zero = torch.zeros(
                (len(ts), step.WIDTH), dtype=dtype, device=device, requires_grad=True
            )
            original_score = step.risk(base, xx, aa, zero)
            (normal,) = torch.autograd.grad(original_score.sum(), zero)
        if not torch.allclose(original_score.detach(), z[ts], rtol=0, atol=5e-5):
            raise ValueError("native anchor value differs under a zero perturbation")
        size = len(SEEDS) * 3
        which = torch.arange(len(ts), device=device).repeat_interleave(size)

        def forward(ix, u):
            return step.risk(base, xx[which[ix]], aa[which[ix]], u)

        action, ok, err, angle = solver.neutralize(
            raw_u[candidates, :, :3].reshape(-1, step.WIDTH),
            normal[which],
            z[ts][which],
            forward,
        )
        root_ok[candidates] = ok.reshape(-1, len(SEEDS), 3).cpu().numpy()
        select = root_ok[candidates].all(axis=(1, 2))
        accepted = candidates[select]
        common[accepted] = True
        final[accepted, :, :3] = action.reshape(-1, len(SEEDS), 3, step.WIDTH)[select]
        final[accepted, :, 3] = raw_u[accepted, :, 0]
        angles[candidates] = angle.reshape(-1, len(SEEDS), 3).cpu().numpy()
        residuals[candidates] = err.reshape(-1, len(SEEDS), 3).cpu().numpy()
    slots = step.write_slots(len(x), maturity)
    eligible = slots >= 0
    active = np.zeros(len(x), dtype=bool)
    active[eligible] = common[slots[eligible]]
    ix = np.flatnonzero(active)
    values, equal_score, offsets = prior.trace_active(
        base, x, a, z, final, maturity, slots, ix
    )
    dense = prior.commit_trace(z.cpu().numpy(), ix, values, equal_score)
    norms, raw_norms = (
        final.norm(dim=-1).cpu().numpy(),
        raw_u.norm(dim=-1).cpu().numpy(),
    )
    if (
        not np.allclose(norms[common], raw_norms[common, :, :1], atol=2e-6, rtol=0)
        or (np.abs(offsets) > solver.TOL + 2e-5).any()
    ):
        raise ValueError("neutral comparison lost the finite constraint or the norm")
    if archived is not None and (
        not np.array_equal(old_common, archived["common"])
        or not np.allclose(raw_norms, archived["norms"], atol=2e-6, rtol=0)
        or not np.allclose(
            dense[:, :, 0], archived["logits"][:, :, 0], atol=5e-5, rtol=0
        )
        or not np.allclose(
            dense[active, :, 4], archived["logits"][active, :, 1], atol=5e-5, rtol=0
        )
    ):
        raise ValueError("matched original actions do not reproduce the sealed replay")
    return dict(
        time=raw["times"],
        logits=dense,
        maturity=maturity,
        active=active,
        common=common,
        old_common=old_common,
        root_ok=root_ok,
        angles=angles,
        solver_residual=residuals,
        offsets=offsets,
        norms=norms,
    )


def produce(root, out):
    if out.exists():
        raise FileExistsError(out)
    start, h = time.monotonic(), prior.Hashes()
    protocol, rows, old_seal, audit = input_rows(root, h)
    files = (
        Path(__file__),
        Path(solver.__file__),
        Path(metrics.__file__),
        PLAN,
        old_seal,
        audit,
        root / "protocol.json",
        root / "fit_completed.json",
    )
    binding = {str(k.resolve()): h(k) for k in files}
    r.configure()
    base, meta = r.load_parent(r.DEFAULT_PARENT)
    if (
        r.state_digest(base) != protocol["source_state_sha256"]
        or meta["threshold"] != r.THRESHOLD
    ):
        raise ValueError("unchanged source and threshold required")
    base.cuda()
    fits = [prior.load_fit(root / f"fit_{i}.npz", base) for i in SEEDS]
    out.mkdir(parents=True)
    r.write_new(
        out / "protocol.json",
        dict(
            status="EXPLORATORY_PAIRED_ANCHOR_NEUTRAL_NO_FIT",
            paths=binding,
            arms=ARMS,
            seeds=SEEDS,
            source_sha256=protocol["source_state_sha256"],
            source_val=protocol["roles"]["val"],
            cutoff=10,
            threshold=r.THRESHOLD,
            fitted_new_parameters=0,
        ),
    )
    ledger, prefixes = [], []
    with patch.object(
        r.io23, "read_outcome", side_effect=RuntimeError("held source labels sealed")
    ):
        for q in rows:
            sid, rank = q["shot"], q["rank"]
            raw = r.simple_data.read_shot("J-TEXT", sid)
            with np.load(root / "val" / q["file"], allow_pickle=False) as a:
                new = predict(base, fits, raw, a)
            if rank in (2, 40, 120) and len(raw["times"]) > 103:
                sub = {
                    **raw,
                    **{k: raw[k][:103] for k in ("values", "available", "times")},
                }
                z = predict(base, fits, sub)
                diff = float(np.max(np.abs(z["logits"] - new["logits"][:103])))
                if not np.array_equal(z["active"], new["active"][:103]) or diff > 2e-4:
                    raise ValueError("new neutral causal prefix does not reproduce")
                prefixes.append(
                    dict(shot=sid, n=103, max_diff=diff, active=int(z["active"].sum()))
                )
            file = out / q["file"]
            r.save_arrays(file, **new)
            ledger.append(
                dict(**q, parent_curve_sha256=q["sha256"], neutral_sha256=h(file))
            )
            if rank % 40 == 0:
                print(f"sealed source neutral {rank}/160", flush=True)
    if (
        r.state_digest(base) != protocol["source_state_sha256"]
        or base.risk_head._forward_pre_hooks
    ):
        raise ValueError("model or hook state changed")
    h.check()
    r.write_new(
        out / "completed.json",
        dict(
            status="SEALED_PAIRED_ANCHOR_NEUTRAL",
            protocol_sha256=h(out / "protocol.json"),
            rows=ledger,
            prefix=prefixes,
            held_outcome_reads=0,
            seconds=time.monotonic() - start,
        ),
    )


def verify_new(a, old):
    time = a["time"]
    z, mat, common, active, ok = (
        a[k] for k in ("logits", "maturity", "common", "active", "root_ok")
    )
    slots = np.arange(len(time)) // 32 - 2
    expected = np.zeros(len(time), dtype=bool)
    expected[slots >= 0] = common[slots[slots >= 0]]
    if (
        z.shape != old["logits"].shape
        or not np.isfinite(z).all()
        or not np.array_equal(time, old["time"])
        or not np.array_equal(mat, old["maturity"])
        or not np.array_equal(a["old_common"], old["common"])
        or not np.array_equal(common, ok.all(axis=(1, 2)))
        or not np.array_equal(active, expected)
        or (active & ~old["active"]).any()
        or not np.array_equal(
            z[~active], np.broadcast_to(z[~active, :, :1], z[~active].shape)
        )
        or not np.allclose(z[:, :, 0], old["logits"][:, :, 0], atol=5e-5, rtol=0)
        or not np.allclose(
            z[active, :, 4], old["logits"][active, :, 1], atol=5e-5, rtol=0
        )
        or not np.allclose(
            z[active, :, 5],
            z[active, :, 0] + a["offsets"][slots[active]],
            atol=8e-7,
            rtol=0,
        )
        or not np.allclose(
            a["norms"][common], old["norms"][common, :, :1], atol=2e-6, rtol=0
        )
        or (np.abs(a["solver_residual"][ok]) > solver.TOL).any()
        or (np.abs(a["offsets"]) > solver.TOL + 2e-5).any()
    ):
        raise ValueError("sealed neutral rows violate causal, score or norm matching")


def audit(root, out):
    start, h = time.monotonic(), prior.Hashes()
    cfg, rows, _, _ = input_rows(root, h)
    pp, done = out / "protocol.json", out / "completed.json"
    p, d = r.read(pp), r.read(done)
    if (
        d["status"] != "SEALED_PAIRED_ANCHOR_NEUTRAL"
        or d["held_outcome_reads"] != 0
        or d["protocol_sha256"] != h(pp)
        or tuple(p["arms"]) != ARMS
        or not all(h(k) == v for k, v in p["paths"].items())
        or [q["shot"] for q in d["rows"]] != cfg["roles"]["val"]
        or not any(w["active"] > 0 for w in d["prefix"])
    ):
        raise ValueError("complete blind neutral archive required")
    for q, old in zip(d["rows"], rows, strict=True):
        if (
            q["parent_curve_sha256"] != old["sha256"]
            or h(out / old["file"]) != q["neutral_sha256"]
        ):
            raise ValueError("neutral or original curve bytes changed")
    adv, scored = r.simple_data.advanced_times(), []
    num, denominator, angles, residual, diffs = 0, 0, [], [], []
    for q in rows:
        sid = q["shot"]
        raw = r.simple_data.read_shot("J-TEXT", sid)
        with (
            np.load(out / q["file"], allow_pickle=False) as a,
            np.load(
                root / "val" / q["file"],
                allow_pickle=False,
            ) as old,
        ):
            verify_prediction(old, raw["times"], raw["available"], enabled=True)
            verify_new(a, old)
            flag, down = r.io23.read_outcome("J-TEXT", sid)
            horizon, _, _ = r.io23.effective_horizon_ms(sid, adv)
            z = metrics.shot_metrics(
                raw["times"],
                a["logits"],
                a["active"],
                flag,
                down,
                horizon,
                10,
                cfg["threshold"],
            )
            scored.append(dict(shot=sid, **z))
            num += int(a["common"].sum())
            denominator += int(a["old_common"].sum())
            angles.extend(np.abs(a["angles"][a["common"]]).ravel().tolist())
            residual.extend(np.abs(a["solver_residual"][a["common"]]).ravel().tolist())
            active = a["active"]
            if active.any():
                z = a["logits"][active]
                diffs.append(
                    np.abs(z[:, :, 1] - z[:, :, 5]).mean(axis=0, dtype=np.float64)
                )
    aggregate = metrics.aggregate(scored, cfg["threshold"], 10)
    aggregate["per_seed"] = {
        s: {COLMAP[k]: v for k, v in arm.items()}
        for s, arm in aggregate["per_seed"].items()
    }
    aggregate["pooled_strata"] = {
        g: {COLMAP[k]: v for k, v in arm.items()}
        for g, arm in aggregate["pooled_strata"].items()
    }
    aggregate["mean_counts"] = {
        COLMAP[k]: v for k, v in aggregate["mean_counts"].items()
    }
    h.check()
    result = dict(
        status="ANCHOR_NEUTRAL_SOURCE_VAL_EXPLORATORY_RECOUNT",
        seal_sha256=h(done),
        common=num,
        original_common=denominator,
        arms=ARMS,
        max_anchor_residual=max(residual, default=None),
        median_abs_angle=float(np.median(angles)) if angles else None,
        mean_abs_later_real_vs_equal_score=np.mean(diffs, axis=0).tolist()
        if diffs
        else None,
        aggregate=aggregate,
        bootstrap_colmap=COLMAP,
        promotes_method=False,
        seconds=time.monotonic() - start,
    )
    r.write_new(out / "audit.json", result)
    print({k: v for k, v in result.items() if k != "aggregate"}, flush=True)
    print(aggregate["mean_counts"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("produce", "audit"))
    p.add_argument("parent", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    (produce if a.mode == "produce" else audit)(a.parent, a.out)
