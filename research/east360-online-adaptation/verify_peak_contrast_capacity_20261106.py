"""Recount matched effects and verify the selected TRAIN clocks independently."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import peak_contrast_scores_20261106 as scores
import run_peak_contrast_capacity_20261106 as run
from verify_confirmation_raw_replay_20261105 import Hashes


def preflight(root, h):
    pp, seal_path = root / "protocol.json", root / "completed.json"
    cfg, done = run.r.read(pp), run.r.read(seal_path)
    inv_path = root / "train_raw.json"
    raw = run.r.read(inv_path)["rows"]
    source, source_rows = run.parent_run.source_inputs(h)
    if (
        cfg["status"] != "LOCKED_PEAK_CONTRAST"
        or done["status"] != "SEALED_TRAIN_ONLY_PEAK_CONTRAST"
        or not done["no_fit"]
        or cfg["train_ids"] != source["roles"]["train"]
        or cfg["state_sha256"] != source["source_state_sha256"]
        or cfg["threshold"] != source["threshold"]
        or cfg["high_floor"] != run.HIGH
        or cfg["logit_threshold"] != run.TAU
        or cfg["action_norm"] != run.forward.SITE_NORM
        or tuple(cfg["seed_order"]) != run.SEEDS
        or tuple(cfg["sites"]) != run.forward.SITES
        or tuple(cfg["arms"]) != run.forward.ARMS
        or any(
            p[k]
            for p in (cfg, done)
            for k in ("held_signal_reads", "held_outcome_reads")
        )
        or [q["shot"] for q in raw] != cfg["train_ids"]
        or h(pp) != done["protocol_sha256"]
        or h(inv_path) != done["raw_manifest_sha256"]
        or not all(h(p) == v for p, v in cfg["input_sha256"].items())
        or not all(
            q["sha256"] == src["sha256"] == h(run.r.io23.path_for("J-TEXT", q["shot"]))
            for q, src in zip(raw, source_rows, strict=True)
        )
    ):
        raise ValueError("unbound source scope, base, or completed paired archive")
    parts = {}
    for name in ("all", "high"):
        mp, data = root / f"{name}_pairs.json", root / f"{name}.npz"
        manifest = run.r.read(mp)
        if (
            manifest["cohort"] != name
            or done["pair_sha256"][name] != h(mp)
            or manifest["archive_sha256"] != h(data)
        ):
            raise ValueError(f"{name} predictions or pairing changed after seal")
        parts[name] = (manifest, data)
    h.check()
    return cfg, parts, {q["shot"]: q["disruptive"] for q in raw}


def check_match(d, s, cost, *, high):
    delta = np.array([d[k] - s[k] for k in ("age_ms", "anchor", "trend")])
    valid_d, valid_s = (np.asarray(q["valid"]) for q in (d, s))
    if (
        valid_d.shape != (11,)
        or valid_d.dtype != bool
        or valid_s.shape != (11,)
        or valid_s.dtype != bool
        or d["shot"] == s["shot"]
        or not np.isfinite(delta).all()
    ):
        raise ValueError("pair identities or availability differ")
    family = [slice(0, 3), slice(3, 7), slice(7, 11)]
    fam_d = [valid_d[k].any() for k in family]
    fam_s = [valid_s[k].any() for k in family]
    mismatch = int(np.count_nonzero(valid_d != valid_s))
    ratio = delta / np.array([64.0, 0.25, 0.35])
    expected = float(ratio @ ratio) + (mismatch / 3.0) ** 2
    if (
        fam_d != fam_s
        or sum(fam_d) < 2
        or mismatch > 3
        or np.max(np.abs(ratio)) > 1
        or not np.isclose(expected, cost, atol=1e-12, rtol=0)
        or (high and min(d["anchor"], s["anchor"]) < run.HIGH)
    ):
        raise ValueError("a pair violates the predeclared risk or clock calipers")


def masks_from_native(t, z, m, flag, down, horizon):
    """Independent whole-shot first crossing and canonical positive mask."""
    n = len(t)
    idx = m + np.arange(1, 33)
    exists = idx < n
    if not flag:
        if not exists.all():
            raise ValueError("safe comparison has a partial next block")
        return exists, np.zeros(32, bool), False, exists
    lead = (down - t) * 1000.0
    early = np.flatnonzero((z.astype(np.float64) >= run.TAU) & (lead >= -1e-6))
    timely = bool(early.size and round(float(lead[early[0]]), 9) > 10)
    protect = np.zeros(32, bool)
    if timely and m < early[0] <= m + 32:
        protect = exists & (idx <= early[0])
    pos = np.zeros(32, bool)
    delta = lead[idx[exists]].round(9)
    pos[exists] = (delta > 10) & (delta <= horizon + 1e-6)
    if not pos.any():
        raise ValueError("danger cohort contains no canonical timely endpoint")
    return pos, protect, timely, exists


def verify_native(base, meta, archived, flag, cache, advance):
    sid, m = meta["shot"], meta["maturity"]
    if sid not in cache:
        raw = run.r.simple_data.read_shot("J-TEXT", sid)
        truth, down = run.r.io23.read_outcome("J-TEXT", sid)
        x, a = run.prior.views(base, raw)
        z = run.prior.scan(base, x, a).numpy()
        cache[sid] = (raw, truth, down, z)
    raw, truth, down, z = cache[sid]
    if truth != flag:
        raise ValueError(f"wrong authoritative outcome side: {sid}")
    t = raw["times"]
    horizon = run.r.io23.effective_horizon_ms(sid, advance)[0]
    positive, protect, timely, present = masks_from_native(
        t, z, m, truth, down, horizon
    )
    used = [3, 4, 5, 14, 15, 16, 17, 19, 20, 21, 22]
    avail = raw["available"][m - 63 : m + 1, used].reshape(2, 32, 11).sum(1)
    valid = (avail >= 16).all(0)
    ix = m + np.arange(1, 33)[present]
    err = max(
        float(np.abs(archived[present] - z[ix]).max()),
        abs(meta["anchor"] - float(z[m])),
        abs(meta["trend"] - float(z[m] - z[m - 32])),
    )
    if (
        not 63 <= m < len(t)
        or (m - 63) % 32
        or abs((t[m] - t[0]) * 1000.0 - meta["age_ms"]) > 1e-9
        or not np.array_equal(positive, meta["positive"])
        or not np.array_equal(protect, meta["protect"])
        or not np.array_equal(present, meta["present"])
        or not np.array_equal(valid, meta["valid"])
        or timely != meta["native_timely"]
        or err > 8e-5
    ):
        raise ValueError(
            f"actual source clock, role, availability or logits differ: {sid}"
        )
    return err


def as_arrays(manifest, raw_truth):
    pairs = manifest["pairs"]
    p = len(pairs)
    ids = np.asarray(
        [[q[k]["shot"] for k in ("danger", "safe")] for q in pairs], dtype=np.int64
    ).reshape(p, 2)
    if len(np.unique(ids)) != 2 * p or any(
        not raw_truth.get(d, False) or raw_truth.get(s, True) for d, s in ids
    ):
        raise ValueError(
            "TRAIN sides must be outcome-disjoint and used without replacement"
        )
    positive, protect = (
        np.asarray([q["danger"][key] for q in pairs], dtype=bool).reshape(p, 32)
        for key in ("positive", "protect")
    )
    timely = np.asarray([q["danger"]["native_timely"] for q in pairs], dtype=bool)
    return ids, positive, protect, timely


def check_payload(a, p, manifest):
    dims = (p, 2, 2, len(run.SEEDS), len(run.forward.ARMS))
    native, z, shifts, norm, avail, features = (
        a[k]
        for k in (
            "native",
            "future",
            "anchor_delta",
            "norms",
            "available",
            "descriptor",
        )
    )
    if (
        native.shape != (p, 2, 32)
        or z.shape != (*dims, 32)
        or norm.shape != dims
        or shifts.shape != dims
        or avail.shape != (p, len(run.SEEDS), len(run.forward.ARMS))
        or avail.dtype != bool
        or features.shape != (p, 2, len(run.SEEDS), run.rule.FEATURES)
        or not all(np.isfinite(v).all() for v in (native, z, shifts, norm, features))
    ):
        raise ValueError("sealed predictions or action geometry misaligned")
    common = np.broadcast_to(avail[:, None, None], dims)
    frozen = np.broadcast_to(native[:, :, None, None, None, :], z.shape)
    if (
        not np.array_equal(avail[..., ::2], avail[..., 1::2])
        or int(avail.any((1, 2)).sum()) != manifest["n_supported"]
        or not np.allclose(norm, common * run.forward.SITE_NORM, atol=1e-6, rtol=0)
        or (shifts[~common] != 0).any()
        or not np.array_equal(z[~common], frozen[~common])
    ):
        raise ValueError("matched finite-radius reverse and no-write invariants failed")
    return native, z, shifts, norm, avail


def own_counts(native, z, shifts, positive, protect, timely):
    """Recalculate the decisive availability gate without the scorer module."""
    native, z, shifts = (np.asarray(q, dtype=np.float64) for q in (native, z, shifts))
    n = len(native)
    p = positive[:, None, None, :]
    protected = protect[:, None, None, :]
    native_peak = np.where(positive[:, None], native, -np.inf).max(-1)
    future = z - shifts[..., None]
    dp = np.where(p, future[:, 0], -np.inf).max(-1) - native_peak[:, 0, None, None]
    sp = np.where(p, future[:, 1], -np.inf).max(-1) - native_peak[:, 1, None, None]
    safe_cross = (native[:, 1].max(1)[:, None, None] < run.TAU) & (
        z[:, 1].max(-1) >= run.TAU
    )
    keep = ~protect.any(-1)[:, None, None] | (
        np.where(protected, z[:, 0], -np.inf).max(-1) >= run.TAU
    )
    feasible = (dp >= 0.05) & (sp <= 1e-6) & ~safe_cross & keep
    raw_gain = (
        (~timely[:, None, None])
        & (native_peak[:, 0, None, None] < run.TAU)
        & (np.where(p, z[:, 0], -np.inf).max(-1) >= run.TAU)
    )
    shape_gain = (
        (~timely[:, None, None])
        & (native_peak[:, 0, None, None] < run.TAU)
        & (np.where(p, future[:, 0], -np.inf).max(-1) >= run.TAU)
    )
    return dict(
        pairs=n,
        feasible_any_arm_by_seed=feasible.any(2).sum(0).tolist(),
        feasible_at_least_two_seeds=int((feasible.any(2).sum(1) >= 2).sum()),
        raw_reachable_by_seed=raw_gain.any(2).sum(0).tolist(),
        shape_reachable_by_seed=shape_gain.any(2).sum(0).tolist(),
        raw_new_safe_cross=safe_cross.sum((0, 2)).tolist(),
        mean_shape_difference_by_action=(dp - sp).mean(0).tolist() if n else [],
    )


def cpu_intervention_witness(base, metadata, cache, a):
    """Recompute one full pair's state, dense trajectories and crossing bits on CPU."""
    rows, values, valid, start = [], [], [], 0
    for side in ("danger", "safe"):
        row = metadata[side]
        raw, _, _, z = cache[row["shot"]]
        f = run.prior.signal_schedule(base, raw)
        at = np.flatnonzero(f["maturity"] == row["maturity"])
        if len(at) != 1:
            raise ValueError("CPU replay maturity changed")
        j, m = int(at[0]), row["maturity"]
        k = np.arange(m + 1, m + 33)
        present = k < len(z)
        use = np.minimum(k, len(z) - 1)
        rows.append(
            dict(
                **row,
                delta=f["delta"][j],
                ox=start + m - 32 + 127,
                mx=start + m + 127,
                future_ix=start + use + 127,
                native=np.where(present, z[use], np.float32(0.0)),
            )
        )
        values.append(np.pad(raw["values"], ((127, 0), (0, 0))))
        valid.append(np.pad(raw["available"], ((127, 0), (0, 0))))
        start += len(z) + 127
    tape = {
        "values": torch.from_numpy(np.concatenate(values)),
        "available": torch.from_numpy(np.concatenate(valid)),
    }
    fits = [
        run.prior.load_fit(run.parent_run.PREVIOUS / f"fit_{s}.npz", base)
        for s in run.SEEDS
    ]
    u, common, desc = run.banks(base, fits, tape, rows)
    replay = run.effects(base, tape, rows, u, common, desc)
    diffs = {
        key: float(np.abs(replay[key] - a[key][:1]).max())
        for key in ("future", "native", "norms", "anchor_delta", "descriptor")
    }
    same = np.array_equal(replay["future"] >= run.TAU, a["future"][:1] >= run.TAU)
    if (
        not same
        or not np.array_equal(replay["available"], a["available"][:1])
        or max(diffs.values()) > 1e-4
        or diffs["norms"] > 1e-6
    ):
        raise ValueError("CPU vs sealed CUDA state or threshold bits differ")
    return dict(
        pair=[metadata[side]["shot"] for side in ("danger", "safe")],
        max_errors=diffs,
        threshold_bits_equal=same,
    )


def verify(root):
    h, start = Hashes(), time.monotonic()
    out = root / "audit.json"
    if out.exists():
        raise FileExistsError(out)
    verifier_sha, metrics_sha = h(Path(__file__)), h(Path(scores.__file__))
    cfg, parts, raw_truth = preflight(root, h)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, parent = run.r.load_parent(run.r.DEFAULT_PARENT)
    if (
        run.r.state_digest(base) != cfg["state_sha256"]
        or parent["threshold"] != run.r.THRESHOLD
    ):
        raise ValueError("independent verifier needs the same frozen base")
    advance, cache, summaries, separate = run.r.simple_data.advanced_times(), {}, {}, {}
    witness, maximum = {}, 0.0
    for name, (manifest, path) in parts.items():
        p, positive, protect, timely = as_arrays(manifest, raw_truth)
        separate[name], summaries[name] = {}, {}
        with np.load(path, allow_pickle=False) as a:
            native, z, shift, norm, available = check_payload(a, len(p), manifest)
            for i, q in enumerate(manifest["pairs"]):
                d, s = q["danger"], q["safe"]
                check_match(d, s, q["cost"], high=name == "high")
                for j, row in enumerate((d, s)):
                    maximum = max(
                        maximum,
                        verify_native(base, row, native[i, j], j == 0, cache, advance),
                    )
            if len(p):
                witness[name] = cpu_intervention_witness(
                    base, manifest["pairs"][0], cache, a
                )
            for k, site in enumerate(run.forward.SITES):
                calc = own_counts(
                    native, z[:, :, k], shift[:, :, k], positive, protect, timely
                )
                res = scores.score_pairs(
                    native,
                    z[:, :, k],
                    shift[:, :, k],
                    norm[:, :, k],
                    positive,
                    protect,
                    timely,
                    run.TAU,
                    run.forward.ARMS,
                    p,
                )
                if (
                    res["capacity"]["candidate_pairs_per_seed"]
                    != calc["feasible_any_arm_by_seed"]
                    or res["capacity"]["candidate_pairs_at_least_two_seeds"]
                    != calc["feasible_at_least_two_seeds"]
                    or (
                        len(p)
                        and not np.allclose(
                            np.asarray(calc["mean_shape_difference_by_action"]).T,
                            [q["shape_contrast"]["per_seed_mean"] for q in res["arms"]],
                            atol=2e-9,
                            rtol=0,
                        )
                    )
                ):
                    raise ValueError("independent shape or capacity recount disagrees")
                res_path = root / f"{name}_{site}_effects.json"
                run.r.write_new(res_path, res)
                separate[name][site] = h(res_path)
                summaries[name][site] = calc | {
                    "retrospective": True,
                    "native_not_timely_pairs": int((~timely).sum()),
                    "matched_action_supported_pairs": int(available.any((1, 2)).sum()),
                    "fixed_action_summary": res["arms"],
                }
        print(
            name,
            len(p),
            {s: x["feasible_at_least_two_seeds"] for s, x in summaries[name].items()},
            flush=True,
        )
    gate = {
        site: all(
            (
                summaries["all"][site]["matched_action_supported_pairs"] >= 40,
                summaries["high"][site]["matched_action_supported_pairs"] >= 12,
                summaries["high"][site]["feasible_at_least_two_seeds"] >= 5,
            )
        )
        for site in run.forward.SITES
    }
    h.check()
    run.r.write_new(
        out,
        dict(
            status="INDEPENDENT_TRAIN_ONLY_CAPACITY_RECOUNT",
            gate=gate,
            cohorts=summaries,
            input_seal_sha256=h(root / "completed.json"),
            detailed_effects_sha256=separate,
            verifier_sha256=verifier_sha,
            metric_sha256=metrics_sha,
            cpu_witness=witness,
            max_cpu_native_error=maximum,
            held_signal_reads=0,
            held_outcome_reads=0,
            seconds=time.monotonic() - start,
        ),
    )
    print("capacity gate", gate, "CPU native max error", maximum, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    verify(p.parse_args().root)
