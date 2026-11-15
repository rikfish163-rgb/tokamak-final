"""Fit separate source-only value and harm heads and replay every held TRAIN shot."""

import argparse
from pathlib import Path
import time

import numpy as np

import output_trust_policy_20261106 as policy
import run_output_trust_table_20261106 as bank
from inspect_peak_contrast_first_alarm_20261106 import first
from verify_confirmation_raw_replay_20261105 import Hashes


r, SEEDS, TAU = bank.r, bank.SEEDS, bank.TAU
ARMS = (
    "static",
    "free_joint",
    "trusted_raw",
    "trusted_joint",
    "guarded",
    "stale_guarded",
    "reversed_guarded",
    "score_guarded",
)
MODE = (0, 1, 1, 1, 1, 1, 1)  # nonstatic arms


def checked_table(root, h):
    pp, cp = root / "protocol.json", root / "completed.json"
    cfg, done = r.read(pp), r.read(cp)
    roles, original = bank.src.source_inputs(h)
    if (
        cfg["status"] != "LOCKED_OUTPUT_TRUST_TABLE"
        or done["status"] != "SEALED_OUTPUT_TRUST_TRAIN"
        or h(pp) != done["protocol_sha256"]
        or cfg["roles"] != roles["roles"]
        or cfg["source_state_sha256"] != roles["source_state_sha256"]
        or cfg["threshold"] != r.THRESHOLD
        or cfg["logit_threshold"] != TAU
        or cfg["limit"] != bank.tr.LIMIT
        or tuple(cfg["scales"]) != bank.tr.SCALES
        or tuple(cfg["seed_order"]) != SEEDS
        or tuple(cfg["modes"]) != bank.MODES
        or tuple(cfg["actions"]) != bank.ARMS
        or any(q["val_test_east_reads"] for q in (cfg, done))
        or not all(h(p) == digest for p, digest in cfg["input_sha256"].items())
    ):
        raise ValueError("unchanged full TRAIN table and frozen source required")
    shards, ordered = [], []
    for q in done["shards"]:
        meta, path = root / q["meta"], root / q["archive"]
        data = r.read(meta)
        if (
            h(meta) != q["sha256"]
            or h(path) != data["predictions_sha256"]
            or data["direct_parity"] > 4e-5
        ):
            raise ValueError("TRAIN forecast shard or parity changed")
        ordered.extend(data["shots"])
        shards.append((q, data, path))
    if [q["shot"] for q in ordered] != roles["roles"]["train"] or not all(
        q["raw_sha256"] == p["sha256"] == h(r.io23.path_for("J-TEXT", q["shot"]))
        for q, p in zip(ordered, original, strict=True)
    ):
        raise ValueError("source TRAIN identity or raw files changed")
    return cfg, done, shards


def check_shard(data, meta, n):
    shape = (n, 2, len(SEEDS), len(bank.ARMS))
    z, avail, anchor, radius, largest, free_largest = (
        data[k]
        for k in ("future", "available", "anchor", "radius", "largest", "free_largest")
    )
    desc = data["descriptor"]
    if (
        z.shape != (*shape, 32)
        or avail.shape != shape
        or avail.dtype != bool
        or anchor.shape != shape
        or radius.shape != shape[:1] + shape[2:]
        or largest.shape != radius.shape
        or free_largest.shape != radius.shape
        or desc.shape != (n, len(SEEDS), bank.rule.FEATURES)
        or not all(
            np.isfinite(v).all()
            for v in (z, anchor, radius, largest, free_largest, desc)
        )
    ):
        raise ValueError("unaligned finite action table")
    frozen = np.broadcast_to(data["frozen"][:, None, None, None], z.shape)
    if (
        not np.array_equal(avail[..., ::2], avail[..., 1::2])
        or not np.array_equal(radius[..., ::2], radius[..., 1::2])
        or not np.array_equal(avail[:, 1], radius > 0)
        or not np.isin(radius, (0.0, *bank.tr.SCALES)).all()
        or (avail[:, 1] & ~avail[:, 0]).any()
        or (largest > bank.tr.LIMIT + 1e-7).any()
        or (np.abs(anchor[:, 1]) > largest + 1e-7).any()
        or (anchor[~avail] != 0).any()
        or not np.array_equal(z[~avail], frozen[~avail])
    ):
        raise ValueError(
            "matched reverse, no-write or finite trust constraint violated"
        )
    lengths = [q["n_points"] for q in meta["shots"]]
    offsets = np.r_[0, np.cumsum(lengths)]
    if len(data["times"]) != offsets[-1] or len(data["native"]) != offsets[-1]:
        raise ValueError("incomplete original TRAIN trajectories")
    records, start = [], 0
    for j, q in enumerate(meta["shots"]):
        ix = slice(offsets[j], offsets[j + 1])
        k = slice(start, start + q["episodes"])
        t, native = data["times"][ix], data["native"][ix]
        m = data["maturity"][k]
        pos, prot, native_first = bank.fwd.teaching_masks(
            t, native, m, q["disruptive"], q["down"], q["horizon_ms"], TAU
        )
        target = m[:, None] + np.arange(1, 33)
        present = target < len(t)
        block = np.where(present, native[np.minimum(target, len(t) - 1)], np.float32(0))
        if (
            (data["row_shot"][k] != q["shot"]).any()
            or native_first != q["first"]
            or not np.array_equal(pos, data["positive"][k])
            or not np.array_equal(prot, data["protect"][k])
            or not np.array_equal(present, data["present"][k])
            or not np.array_equal(block, data["frozen"][k])
            or not np.array_equal(
                native[bank.tr.reached_indices(m)], data["reference"][k]
            )
        ):
            raise ValueError("wrong native data, reached times or teaching labels")
        records.append((q, t.copy(), native.copy()))
        start = k.stop
    if start != n:
        raise ValueError("some action windows were not attributed to a source shot")
    return records


def load_table(shards):
    fields = (
        "future",
        "available",
        "anchor",
        "radius",
        "largest",
        "free_largest",
        "descriptor",
        "positive",
        "protect",
        "present",
        "frozen",
        "row_shot",
        "maturity",
    )
    recs, parts = [], {k: [] for k in fields}
    advance = r.simple_data.advanced_times()
    for q, m, p in shards:
        for shot in m["shots"]:
            sid = shot["shot"]
            truth, down = r.io23.read_outcome("J-TEXT", sid)
            horizon = r.io23.effective_horizon_ms(sid, advance)[0]
            if (
                truth != shot["disruptive"]
                or (truth and down != shot["down"])
                or horizon != shot["horizon_ms"]
            ):
                raise ValueError(
                    f"authoritative source label or horizon differs: {sid}"
                )
        with np.load(p, allow_pickle=False) as a:
            recs.extend(check_shard(a, m, q["n_windows"]))
            for k in fields:
                parts[k].append(a[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    return data, recs


def targets(data, records):
    flag = {q["shot"]: q["disruptive"] for q, _, _ in records}
    disruptive = np.array([flag[int(i)] for i in data["row_shot"]], dtype=bool)
    pos, protect, present = (data[k] for k in ("positive", "protect", "present"))
    chosen = np.where(disruptive[:, None], pos, present)
    utility_role = np.where(disruptive, np.where(pos.any(1), 1, -1), 0)
    guard_role = np.where(disruptive, np.where(protect.any(1), 1, -1), 0)
    # Define dummy peaks on unlabeled windows without using them for fit.
    chosen = np.where(chosen.any(1)[:, None], chosen, present)
    n, z = data["frozen"], data["future"]
    old_peak = np.where(chosen, n, -np.inf).max(1)
    new_peak = np.where(chosen[:, None, None, None], z, -np.inf).max(-1)
    sign = np.where(disruptive, 1, -1)[:, None, None, None]
    abs_change = sign * (new_peak - old_peak[:, None, None, None])
    joint = np.minimum(abs_change, abs_change - sign * data["anchor"])
    p = present[:, None, None, None]
    new_s = (np.where(p, n[:, None, None, None], -np.inf).max(-1) < TAU) & (
        np.where(p, z, -np.inf).max(-1) >= TAU
    )
    lost_p = np.where(protect[:, None, None, None], z, -np.inf).max(-1) < TAU
    harm = np.where(disruptive[:, None, None, None], lost_p, new_s)
    return (
        utility_role,
        guard_role,
        abs_change.clip(-0.25, 0.25),
        joint.clip(-0.25, 0.25),
        harm.astype(np.float32),
    )


def previous_descriptor(x, ids, maturity):
    """Counterfactual: replace only the reached diagnostic inputs when possible."""
    prev = np.r_[0, np.arange(len(x) - 1)]
    matching = (ids == ids[prev]) & (maturity == maturity[prev] + 32)
    mask = (x[..., 22:33] > 0) & (x[prev, ..., 22:33] > 0)
    group = np.stack(
        [mask[..., lo:hi].any(-1) for lo, hi in ((0, 3), (3, 7), (7, 11))], -1
    )
    enabled = matching[:, None] & (group.sum(-1) >= 2)
    result = x.copy()
    lag = x[prev, ..., :22] * np.concatenate((mask, mask), -1)
    result[..., :22] = np.where(enabled[..., None], lag, result[..., :22])
    return result, enabled


def heads_for(data, teaching, indices):
    roles, guard, raw, joint, harm = teaching
    x, sids, allowed = (data[k] for k in ("descriptor", "row_shot", "available"))
    return [
        [
            policy.fit_heads(
                x[indices, j],
                sids[indices],
                roles[indices],
                guard[indices],
                raw[indices, mode, j],
                joint[indices, mode, j],
                harm[indices, mode, j],
                allowed[indices, mode, j],
            )
            for j in range(len(SEEDS))
        ]
        for mode in range(len(bank.MODES))
    ]


def choose_all(heads, desc, stale, allowed):
    result = []
    for j in range(len(SEEDS)):
        x, st = desc[:, j], stale[:, j]
        free, trust = heads[0][j], heads[1][j]
        before, after = allowed[:, 0, j], allowed[:, 1, j]
        main = policy.choose(trust, x, after, objective="joint", guard=True)
        result.append(
            np.stack(
                (
                    policy.choose(free, x, before, objective="joint", guard=False),
                    policy.choose(trust, x, after, objective="raw", guard=False),
                    policy.choose(trust, x, after, objective="joint", guard=False),
                    main,
                    policy.choose(trust, st, after, objective="joint", guard=True),
                    np.where(main < 0, -1, main ^ 1),
                    main,
                ),
                1,
            )
        )
    return np.stack(result, 1)


def shot_groups(records):
    d, s = (
        sorted(q["shot"] for q, _, _ in records if q["disruptive"] is flag)
        for flag in (True, False)
    )
    return [
        np.sort(np.r_[a, b])
        for a, b in zip(np.array_split(d, 5), np.array_split(s, 5), strict=True)
    ]


def trace_from_table(data, picks, slots, native):
    z = np.broadcast_to(
        native[:, None, None], (len(native), len(SEEDS), len(ARMS))
    ).copy()
    for e in slots:
        m, length = int(data["maturity"][e]), int(data["present"][e].sum())
        t = slice(m + 1, m + 1 + length)
        for j in range(len(SEEDS)):
            for k, mode in enumerate(MODE):
                a = int(picks[e, j, k])
                if a < 0:
                    continue
                if not data["available"][e, mode, j, a]:
                    raise ValueError("choice exceeds reached action support")
                z[t, j, k + 1] = (
                    native[t] + data["anchor"][e, mode, j, a]
                    if ARMS[k + 1] == "score_guarded"
                    else data["future"][e, mode, j, a, :length]
                )
    return z


def score_shot(rec, t, logits):
    flag, down, horizon = rec["disruptive"], rec["down"], rec["horizon_ms"]
    lead = ((down - t) * 1000).round(9) if flag else None
    pos = (
        (lead > 10) & (lead <= horizon + 1e-6) if flag else np.zeros(len(t), dtype=bool)
    )
    return dict(
        shot=rec["shot"],
        disruptive=flag,
        scores=[
            [
                first(t, logits[:, j, k], flag, down, TAU)
                | {
                    "positive_peak": float(logits[pos, j, k].max())
                    if pos.any()
                    else None,
                }
                for k in range(len(ARMS))
            ]
            for j in range(len(SEEDS))
        ],
    )


def summarize(rows):
    n_d = sum(q["disruptive"] for q in rows)
    n_s = len(rows) - n_d
    metrics = {}
    for j, s in enumerate(SEEDS):
        values = {}
        for k, name in enumerate(ARMS):
            x = [(q["scores"][j][0], q["scores"][j][k]) for q in rows]
            leads = [
                a["lead_ms"] - b["lead_ms"] for a, b in x if a["timely"] and b["timely"]
            ]
            peaks = [
                b["positive_peak"] - a["positive_peak"]
                for a, b in x
                if a["positive_peak"] is not None
            ]
            n_t, n_f = (
                sum(b["timely"] for a, b in x),
                sum(b["false_alarm"] for a, b in x),
            )
            values[name] = dict(
                timely=n_t,
                fp=n_f,
                lost=sum(a["timely"] and not b["timely"] for a, b in x),
                gained=sum(not a["timely"] and b["timely"] for a, b in x),
                false_added=sum(
                    not a["false_alarm"] and b["false_alarm"] for a, b in x
                ),
                false_removed=sum(
                    a["false_alarm"] and not b["false_alarm"] for a, b in x
                ),
                delay_gt_10=sum(x > 10 for x in leads),
                maximum_delay=max(leads) if leads else None,
                absolute_positive_peak_mean=float(np.mean(peaks)) if peaks else None,
            )
        metrics[str(s)] = values
    return dict(n_disruptive=n_d, n_safe=n_s, by_seed=metrics)


def serialize_heads(heads):
    return {
        f"{mode}_{seed}_{key}": value
        for i, mode in enumerate(bank.MODES)
        for j, seed in enumerate(SEEDS)
        for key, value in heads[i][j].items()
    }


def load_heads(path):
    with np.load(path, allow_pickle=False) as a:
        return [
            [
                {
                    k.removeprefix(f"{mode}_{s}_"): a[k]
                    for k in a.files
                    if k.startswith(f"{mode}_{s}_")
                }
                for s in SEEDS
            ]
            for mode in bank.MODES
        ]


def geometry(data, records):
    u = data["available"]
    pos = data["positive"].any(1)
    old = u[:, 0].any((1, 2)) & pos
    legal = u[:, 1].any((1, 2)) & pos
    late = [
        q["shot"]
        for q, t, z in records
        if q["disruptive"] and not first(t, z, True, q["down"], TAU)["timely"]
    ]
    z = data["future"]
    good = np.where(data["positive"][:, None, None, None], z, -np.inf).max(-1) >= TAU
    reach = [
        [
            int(
                len(
                    np.unique(
                        data["row_shot"][
                            np.isin(data["row_shot"], late)
                            & (good[:, m, j] & u[:, m, j]).any(1)
                        ]
                    )
                )
            )
            for j in range(len(SEEDS))
        ]
        for m in range(2)
    ]
    radii = data["radius"][u[:, 0]]
    return dict(
        active_windows=len(pos),
        available_free=int(u[:, 0].sum()),
        available_trusted=int(u[:, 1].sum()),
        zero_fraction=float(np.mean(radii == 0)),
        radius_hist={str(k): int((radii == k).sum()) for k in (0.0, *bank.tr.SCALES)},
        max_free_past_change=float(data["free_largest"].max()),
        max_trusted_past_change=float(data["largest"].max()),
        positive_supported_free=int(old.sum()),
        positive_supported_trust=int(legal.sum()),
        non_timely_shots=late,
        late_reachable_by_mode_seed=reach,
        val_geometry_pass=bool(old.any() and legal.sum() >= 0.10 * old.sum()),
    )


def fit(root):
    h, start = Hashes(), time.monotonic()
    out = root / "fit_completed.json"
    if out.exists() or (root / "heads.npz").exists():
        raise FileExistsError("the TRAIN selector receipt or heads already exist")
    cfg, done, shards = checked_table(root, h)
    paths = (Path(__file__), Path(policy.__file__), root / "completed.json")
    inputs = {str(p.resolve()): h(p) for p in paths}
    data, records = load_table(shards)
    g = geometry(data, records)
    teaching = targets(data, records)
    desc, sid, avail = (data[k] for k in ("descriptor", "row_shot", "available"))
    stale, stale_active = previous_descriptor(desc, sid, data["maturity"])
    ids = shot_groups(records)
    action = np.full((len(desc), len(SEEDS), len(ARMS) - 1), -2, dtype=np.int16)
    for k, held in enumerate(ids, 1):
        take = np.isin(sid, held)
        heads = heads_for(data, teaching, np.flatnonzero(~take))
        action[take] = choose_all(heads, desc[take], stale[take], avail[take])
        print(f"TRAIN whole-shot crossfit {k}/{len(ids)}", flush=True)
    if (action < -1).any():
        raise ValueError("some reached source windows were not crossfit")
    rows = [
        score_shot(
            q, t, trace_from_table(data, action, np.flatnonzero(sid == q["shot"]), z)
        )
        for q, t, z in records
    ]
    aggregate = summarize(rows)
    heads = heads_for(data, teaching, np.arange(len(desc)))
    hp, audit, choices = (
        root / "heads.npz",
        root / "train_crossfit.json",
        root / "train_choices.npz",
    )
    r.save_arrays(hp, **serialize_heads(heads))
    r.save_arrays(
        choices,
        actions=action,
        shot=sid,
        maturity=data["maturity"],
        stale_supported=stale_active,
    )
    h.check()
    r.write_new(
        audit,
        dict(
            status="COMPLETE_TRAIN_WHOLE_SHOT_CROSSFIT",
            geometry=g,
            aggregate=aggregate,
            folds=[q.tolist() for q in ids],
            role_shots=[int(len(np.unique(sid[teaching[0] == k]))) for k in (0, 1)],
            protection_shots=int(len(np.unique(sid[teaching[1] == 1]))),
            selected_by_seed_arm=(action >= 0).sum(0).tolist(),
            stale_supported=int(stale_active.sum()),
            rows=rows,
            elapsed=time.monotonic() - start,
        ),
    )
    r.write_new(
        out,
        dict(
            status="FIT_OUTPUT_TRUST_SOURCE_ONLY",
            train_table_sha256=h(root / "completed.json"),
            train_audit_sha256=h(audit),
            heads_sha256=h(hp),
            choices_sha256=h(choices),
            input_sha256=inputs,
            seed_order=SEEDS,
            arms=ARMS,
            source_state_sha256=cfg["source_state_sha256"],
            val_geometry_pass=g["val_geometry_pass"],
            val_test_east_reads=0,
            seconds=time.monotonic() - start,
        ),
    )
    print("geometry", g, flush=True)
    print(
        "TRAIN scores",
        {
            s: {k: (a["timely"], a["fp"]) for k, a in v.items()}
            for s, v in aggregate["by_seed"].items()
        },
        flush=True,
    )


def checked_fit(root, h):
    ff = root / "fit_completed.json"
    f = r.read(ff)
    if (
        f["status"] != "FIT_OUTPUT_TRUST_SOURCE_ONLY"
        or not f["val_geometry_pass"]
        or f["val_test_east_reads"] != 0
        or tuple(f["seed_order"]) != SEEDS
        or tuple(f["arms"]) != ARMS
        or not all(h(p) == v for p, v in f["input_sha256"].items())
        or h(root / "completed.json") != f["train_table_sha256"]
        or h(root / "train_crossfit.json") != f["train_audit_sha256"]
        or h(root / "heads.npz") != f["heads_sha256"]
        or h(root / "train_choices.npz") != f["choices_sha256"]
    ):
        raise ValueError(
            "unchanged full source fit and geometry gate required before held replay"
        )
    return f


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    fit(p.parse_args().root)
