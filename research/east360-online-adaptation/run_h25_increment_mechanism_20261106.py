"""Matched TRAIN-only test of level vs lag-8 arrived H25 targets."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

import h25_increment_gradient_20261106 as inc
import run_h25_direction_mechanism_20261106 as m
from run_h25_late_capacity_20261106 import load_receipts
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = m.r.ROOT / "H25_RELEASE_AND_INCREMENT_PROTOCOL_20261106.md"
ARMS = ("level_true", "level_reverse", "increment_true", "increment_reverse")


def four_spaces(old, level, level_reverse, delta, delta_reverse):
    old_q, level_q, level_r, x = m.geo.matched_augments(old, level, level_reverse)
    _, delta_q, delta_r, y = m.geo.matched_augments(old, delta, delta_reverse)
    extra = min(x, y)
    dim = old_q.shape[1] + extra
    return tuple(q[:, :dim] for q in (level_q, level_r, delta_q, delta_r)), extra


def paired_shot_effect(values, ids):
    a, ids = np.asarray(values, dtype=np.float64), np.asarray(ids)
    v = np.asarray([a[ids == sid].mean() for sid in np.unique(ids)])
    if not len(v) or not np.isfinite(v).all():
        raise ValueError("nonempty finite paired shot effects required")
    rng = np.random.default_rng(61106)
    i = rng.integers(0, len(v), (2000, len(v)))
    return dict(
        mean=float(v.mean()),
        median=float(np.median(v)),
        n_shots=len(v),
        resample_range=np.percentile(v[i].mean(1), [2.5, 97.5]).tolist(),
    )


def cosines(level, lr, delta, dr, keep):
    v = np.stack((level, lr, delta, dr)).astype(np.float64)
    norm = np.linalg.norm(v, axis=-1)
    same = keep & (norm > 1e-10).all(0)
    n = v[:, same] / norm[:, same, None]
    return np.stack(((n[0] * n[1]).sum(-1), (n[2] * n[3]).sum(-1)), axis=1)


def cosine_summary(level, lr, delta, dr, keep):
    return cosine_totals(cosines(level, lr, delta, dr, keep))


def cosine_totals(s):
    return dict(
        n_vectors=len(s),
        median_level=float(np.median(s[:, 0])) if len(s) else None,
        median_increment=float(np.median(s[:, 1])) if len(s) else None,
    )


def actions(q, old, gradients, past):
    stable = m.unit_average(old[:3])
    return np.stack(
        [
            space @ m.mix.projected(space, past, 0.5 * (stable - m.unit_average(g)))
            for space, g in zip(q, gradients, strict=True)
        ]
    )


@torch.no_grad()
def same_radius(base, x, a, past, eligible, vectors, native, witness):
    u = torch.as_tensor(vectors, device=x.device, dtype=x.dtype)
    g = u.new_tensor(m.bank.tr.SCALES)
    signs = u.new_tensor([1.0, -1.0])
    grid = g[:, None, None, None] * u[None, :, None, :] * signs[None, None, :, None]
    xp, ap = x[past].contiguous(), a[past].contiguous()
    xq, aq = x[eligible].contiguous(), a[eligible].contiguous()
    z = (
        m.bank.fwd.batched_risk(
            base,
            m.bank.fwd.encode(base, xp, ap),
            grid.reshape(-1, 128)[None].expand(3, -1, -1),
        )
        .reshape(3, len(g), len(u), 2)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    shift = z - native[past]
    j = m.shared_grid(shift, m.bank.tr.LIMIT - 2e-5)
    frozen = float(native[eligible].max())
    if j is None or (np.linalg.norm(vectors, axis=-1) <= 1e-7).any():
        return dict(
            peaks=[frozen] * len(u),
            radius=0.0,
            drift=0.0,
            direct_error=None,
        )
    actual = u * g[j]
    if bool((actual.norm(dim=-1) > m.mix.RADIUS + 5e-7).any()):
        raise ValueError("matched trust grid exceeded the parameter budget")
    out = m.bank.fwd.batched_risk(
        base,
        m.bank.fwd.encode(base, xq, aq),
        actual[None].expand(len(eligible), -1, -1),
    )
    direct = None
    if witness:
        by_arm = torch.stack(
            [m.bank.old.risk(base, xq, aq, v.expand(len(xq), -1)) for v in actual],
            dim=1,
        )
        direct = float((by_arm - out).abs().max())
        if direct > 5e-5 or not torch.equal(by_arm >= m.TAU, out >= m.TAU):
            raise ValueError(
                "optimized forward did not reproduce direct first-layer writes"
            )
    return dict(
        peaks=out.max(dim=0).values.cpu().tolist(),
        radius=float(g[j]),
        drift=float(np.abs(shift[j]).max()),
        direct_error=direct,
    )


def grad_at(base, x, a, raw, maturity):
    o = maturity - 32
    y, ok = m.h25.arrived_targets(raw["values"], raw["available"], o, maturity)
    return {
        k: v.cpu().numpy()
        for k, v in inc.gradients(
            base, x[o].contiguous(), a[o].contiguous(), y, ok
        ).items()
    }


def shot_rows(base, raw, earlier, cases, saved, witness):
    x, a = m.bank.prior.views(base, raw)
    z = m.bank.prior.scan(base, x, a).cpu().numpy()
    if not np.allclose(z, earlier, atol=5e-5, rtol=0) or not np.array_equal(
        z >= m.TAU, earlier >= m.TAU
    ):
        raise ValueError("same native curve required")
    clock, inverse = np.unique(
        [case["maturity"] for _, case in cases], return_inverse=True
    )
    fresh = grad_at(base, x, a, raw, clock)
    prefix = None
    if witness:
        cut = int(clock[0]) + 1
        prefix_raw = {k: raw[k][:cut] for k in ("values", "available", "times")}
        px, pa = m.bank.prior.views(base, prefix_raw)
        short = grad_at(base, px, pa, prefix_raw, clock[:1])
        prefix = max(
            float(np.abs(short[k] - fresh[k][:1]).max())
            for k in ("true", "reverse", "loss")
        )
        if prefix > 8e-5 or not np.array_equal(short["mask"], fresh["mask"][:1]):
            raise ValueError("future truncation changed the increment signal")
    rows = []
    for slot, ((i, row), j) in enumerate(zip(cases, inverse, strict=True)):
        item = {k: row[k] for k in ("shot", "maturity", "role", "late")}
        alive = fresh["groups"][j] & row["h25_groups"]
        modes = (
            saved["true"][i],
            saved["reverse"][i],
            fresh["true"][j],
            fresh["reverse"][j],
        )
        bases, ranks = zip(
            *(four_spaces(b, *modes) for b in saved["old"][i]), strict=True
        )
        if row["supported"] and alive.sum() >= 2 and min(ranks) > 0:
            ix = np.asarray(row["eligible"])
            past = m.bank.tr.reached_indices(np.array([row["maturity"]]))[0]
            jp, gf = saved["past"][i], saved["future"][i]
            if row["role"] == "safe":
                gf = -gf
            effects, cover = [], []
            for b, q in zip(saved["old"][i], bases, strict=True):
                u = actions(q, b, modes, jp)
                effects.append(
                    same_radius(base, x, a, past, ix, u, z, witness and slot == 0)
                )
                cover.append([m.geo.capture(v, jp, gf)["fraction"] for v in q])
            peaks = np.asarray([f["peaks"] for f in effects])
            frozen = float(z[ix].max())
            harm = (
                peaks < m.TAU
                if row["role"] == "protect"
                else ((frozen < m.TAU) & (peaks >= m.TAU))
                if row["role"] == "safe"
                else np.zeros_like(peaks, dtype=bool)
            )
            item.update(
                supported=True,
                cosines=cosines(*modes, alive).tolist(),
                capture=cover,
                radius=[f["radius"] for f in effects],
                shared_extra=list(ranks),
                peaks=peaks.tolist(),
                delta=(peaks - frozen).tolist(),
                harmful=harm.tolist(),
                drift=[f["drift"] for f in effects],
                direct_error=[f["direct_error"] for f in effects],
                inc_observed_pairs=int(fresh["mask"][j].sum()),
                frozen=frozen,
            )
        else:
            item["supported"] = False
        rows.append(item)
    return rows, prefix


def summary(rows):
    out = {}
    for role in m.ROLES:
        items = [q for q in rows if q["role"] == role]
        good = [q for q in items if q["supported"]]
        val = np.asarray([q["delta"] for q in good])
        ids = [q["shot"] for q in good]
        angles = np.asarray([c for q in good for c in q["cosines"]]).reshape(-1, 2)
        out[role] = dict(
            n_total=len(items),
            n_supported=len(good),
            n_shots=len(set(ids)),
            angles=cosine_totals(angles),
            peak=m.shot_average(val, ids),
            capture=m.shot_average([q["capture"] for q in good], ids),
            harm=np.asarray([q["harmful"] for q in good], dtype=int).sum(0).tolist()
            if good
            else None,
            active=(np.asarray([q["radius"] for q in good]) > 0).sum(0).tolist()
            if good
            else None,
            increment_minus_reverse=paired_shot_effect(val[..., 2] - val[..., 3], ids)
            if good
            else None,
            increment_minus_level=paired_shot_effect(val[..., 2] - val[..., 0], ids)
            if good
            else None,
        )
    return out


def passes(s):
    p = s["positive"]
    angles = p["angles"]
    a, b = angles["median_level"], angles["median_increment"]
    pair = p["increment_minus_reverse"]
    safe = []
    for role in ("protect", "safe"):
        h = s[role]["harm"]
        if h is not None:
            total = np.asarray(h).sum(0)
            safe.append(int(total[2]) <= min(int(total[0]), int(total[3])))
        else:
            safe.append(False)
    return bool(
        p["n_total"] > 0
        and p["n_supported"] * 10 >= p["n_total"] * 7
        and a is not None
        and b is not None
        and a - b >= 0.15
        and np.asarray(p["peak"])[:, 2].mean() > 0
        and pair["mean"] >= 0.005
        and pair["resample_range"][0] > 0
        and all(safe)
    )


def run(primary, out):
    h, t = Hashes(), time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    for path in (PLAN, Path(__file__), Path(inc.__file__), primary / "mechanism.json"):
        h(path)
    cfg, receipt, saved = load_receipts(primary, h)
    _, cohort, sources, _ = m.catalog(h)
    by_shot = defaultdict(list)
    for i, row in enumerate(receipt["cases"]):
        by_shot[row["shot"]].append((i, row))
    if set(by_shot) != set(cohort) or not all(
        a == {k: b[k] for k in a}
        for s in cohort
        for a, (_, b) in zip(cohort[s], by_shot[s], strict=True)
    ):
        raise ValueError("this test must reuse the same frozen role windows")
    m.r.configure()
    base, parent = m.r.load_parent(m.r.DEFAULT_PARENT)
    if (
        m.r.state_digest(base) != cfg["source_state_sha256"]
        or parent["threshold"] != m.r.THRESHOLD
    ):
        raise ValueError("same base and threshold required")
    base.cuda()
    rows, prefix = [], []
    for n, (sid, cases) in enumerate(by_shot.items(), 1):
        if h(m.r.io23.path_for("J-TEXT", sid)) != sources[sid]["rec"]["raw_sha256"]:
            raise ValueError("raw TRAIN bytes differ")
        raw = m.r.simple_data.read_shot("J-TEXT", sid)
        if not np.array_equal(raw["times"], sources[sid]["times"]):
            raise ValueError("raw TRAIN clock differs")
        items, error = shot_rows(
            base, raw, sources[sid]["native"], cases, saved, n in (1, 73, 151)
        )
        rows.extend(items)
        if error is not None:
            prefix.append(dict(shot=sid, max_gradient_or_loss_error=error))
        if n % 50 == 0 or n == len(by_shot):
            print(f"H25 lag8 {n}/{len(by_shot)} source shots", flush=True)
    if m.r.state_digest(base) != cfg["source_state_sha256"] or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("frozen base has changed")
    h.check()
    stats = summary(rows)
    m.r.write_new(
        out,
        dict(
            status="MEASURED_TRAIN_LEVEL_VS_LAG8",
            arms=ARMS,
            seeds=m.SEEDS,
            protocol_sha256=h(PLAN),
            code_sha256=h(Path(__file__)),
            inc_sha256=h(Path(inc.__file__)),
            primary_sha256=h(primary / "mechanism.json"),
            source_state_sha256=cfg["source_state_sha256"],
            prefix=prefix,
            rows=rows,
            summary=stats,
            pass_training_gate=passes(stats),
            seconds=time.monotonic() - t,
            val_test_east_reads=0,
        ),
    )
    print(dict(summary=stats, pass_training_gate=passes(stats)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.primary, a.out)
