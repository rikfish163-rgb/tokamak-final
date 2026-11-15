"""Seal reached intergroup phase cues and test held-shot action rankings."""

import argparse
from pathlib import Path
import time

import numpy as np

import intergroup_phase_features_20261106 as phase
import run_patch_utility_multiview_20261106 as m
from verify_confirmation_raw_replay_20261105 import Hashes


b, sel, sparse = m.sel.bank, m.sel, m.sparse
PLAN = b.r.ROOT / "PATCH_UTILITY_INTERGROUP_TIME_PLAN_20261106.md"
ARMS = ("risk", "within", "real", "shifted")
FIELDS = ("within", "real", "shifted", "supported", "common")
SIGNS = np.pad(np.repeat(np.eye(6, dtype=np.float32), 2, axis=0) *
               np.tile((1.0, -1.0), 6)[:, None], ((1, 0), (0, 0)))


def add_features(x, cues, arm):
    if arm not in ARMS or x.ndim != 3 or x.shape[1] != len(SIGNS):
        raise ValueError("fixed control name and 13 actions required")
    n, a = x.shape[:2]
    group = np.concatenate((cues["within"], cues["supported"]), axis=1)
    signal = cues[arm] if arm in ("real", "shifted") else np.zeros((n, 9))
    if arm == "risk":
        group = np.zeros_like(group)
    static = np.broadcast_to(np.concatenate((group, signal), axis=1)[:, None], (n, a, 27))
    cross = (signal[:, None, :, None] * SIGNS[None, :, None, :]).reshape(n, a, 54)
    return np.concatenate((x, static, cross), axis=2).astype(np.float32)


def raw_parent(native, h):
    cfg_path, seal_path = native / "protocol.json", native / "seal.json"
    cfg, root = b.r.read(cfg_path), b.r.read(seal_path)
    previous, inventory = b.bank.src.source_inputs(h)
    if (
        cfg["outcome_reads"] or root["outcome_reads"]
        or h(cfg_path) != root["protocol_sha256"]
        or cfg["shots"] != previous["roles"]["train"]
        or cfg["source_state_sha256"] != previous["source_state_sha256"]
        or not all(h(p) == d for p, d in cfg["inputs"].items())
    ):
        raise ValueError("unrecognized unchanged TRAIN source/action-bank protocol")
    base, checkpoint = b.r.load_parent(b.r.DEFAULT_PARENT)
    if (b.r.state_digest(base) != cfg["source_state_sha256"]
            or checkpoint["threshold"] != b.r.THRESHOLD):
        raise ValueError("the qualified source scaler changed")
    h(seal_path)
    return cfg, root, {r["shot"]: r["sha256"] for r in inventory}, base


def seal(native, out):
    if out.exists():
        raise FileExistsError(out)
    h, start = Hashes(), time.monotonic()
    for p in (Path(__file__), Path(phase.__file__), PLAN):
        h(p)
    cfg, prior, inv, base = raw_parent(native, h)
    center, scale = base.center.cpu().numpy(), base.scale.cpu().numpy()
    excluded = {b.r.io23.RAW23.index(k) for k in ("MNM", "CIII")}
    if excluded.intersection(j for g in phase.CHANNELS for j in g):
        raise ValueError("cannot share cues if the observed groups are altered")
    data = {k: [] for k in (*FIELDS, "clock", "owner")}
    ids, prefixes = [], []
    for shard in prior["shards"]:
        p = native / shard["file"]
        if h(p) != shard["sha256"]:
            raise ValueError("action clock shard changed")
        with np.load(p, allow_pickle=False) as arrays:
            clock, t = arrays["clock"], arrays["times"]
        off = row = 0
        for rec in shard["shots"]:
            sid, count, n = rec["shot"], rec["blocks"], rec["points"]
            path = b.r.io23.path_for("J-TEXT", sid)
            if h(path) != inv[sid] or inv[sid] != rec["raw_sha256"]:
                raise ValueError(f"raw source input changed: {sid}")
            raw = b.r.simple_data.read_shot("J-TEXT", sid)
            values, mask = raw["values"], raw["available"]
            if (not np.array_equal(raw["times"], t[off:off+n])
                    or not np.array_equal(clock[row:row+count], np.arange(63, n-1, 32))):
                raise ValueError("reached schedule and raw source grid differ")
            for q in clock[row:row+count]:
                cues = phase.prefix(values, mask, center, scale, q)
                for k in FIELDS:
                    data[k].append(cues[k])
                data["clock"].append(q)
                data["owner"].append(len(ids))
                if len(ids) in (100, 500, 1000) and q == 63:
                    shorter = phase.prefix(values[:q+1], mask[:q+1], center, scale, q)
                    if not all(np.array_equal(cues[k], shorter[k]) for k in FIELDS):
                        raise ValueError("future truncation changed the reached descriptor")
                    prefixes.append([sid, int(q)])
            off, row = off+n, row+count
            ids.append(sid)
        if off != len(t) or row != len(clock):
            raise ValueError("sealed archive contained unbound points or blocks")
        print(f"Physical group prefixes sealed in memory for {len(ids)} shots", flush=True)
    if ids != cfg["shots"] or len(data["clock"]) != prior["blocks"]:
        raise ValueError("TRAIN coverage is incomplete")
    h.check()
    out.mkdir(parents=True, exist_ok=False)
    archive = out / "features.npz"
    b.r.save_arrays(archive, ids=np.asarray(ids), **{k: np.asarray(v) for k,v in data.items()})
    b.r.write_new(out / "seal.json", dict(
        status="SEALED_REACHED_PHASE_BEFORE_OUTCOME_READ", outcome_reads=0,
        native_seal_sha256=h(native / "seal.json"), feature_sha256=h(archive),
        n_shots=len(ids), blocks=prior["blocks"], prefix_witnesses=prefixes,
        groups=phase.CHANNELS, hashes={str(p): v[1] for p,v in h.cache.items()},
        elapsed_seconds=time.monotonic()-start,
    ))


def describe(tape, score, cues, ids, flags):
    owner, allow = tape.owner, tape.allowed.copy()
    allow[:, 0] = False
    ranks = np.argsort(-np.where(allow, score, -np.inf), axis=1, kind="stable")
    row = np.arange(len(score))
    a = ranks[:, 0]
    has = allow.any(axis=1)
    top = np.where(has, a, 0)
    covered = cues["supported"].any(axis=1)
    change = tape.hard[row, top]
    gain = (tape.hard > 0) & allow
    possible = gain.any(axis=1)
    late = flags[owner] & possible
    ordered = gain[row[:, None], ranks]
    best = np.where(possible, ordered.argmax(axis=1)+1, 0)
    out = {
        "late_top": np.unique(ids[owner[late & covered & (change > 0)]]).tolist(),
        "late_possible": np.unique(ids[owner[late]]).tolist(),
        "late_supported": np.unique(ids[owner[late & covered]]).tolist(),
        "safe_top_help": np.unique(ids[owner[(~flags[owner]) & has & (change > 0)]]).tolist(),
        "safe_top_harm": np.unique(ids[owner[(~flags[owner]) & has & (change < 0)]]).tolist(),
        "safe_help_blocks": int(((~flags[owner]) & has & (change > 0)).sum()),
        "safe_harm_blocks": int(((~flags[owner]) & has & (change < 0)).sum()),
        "late_blocks": [
            dict(shot=int(ids[owner[i]]), clock=int(tape.data["clock"][i]), action=int(top[i]),
                 helpful=np.flatnonzero(gain[i]).tolist(), rank=int(best[i]),
                 supported=int(cues["supported"][i].sum()), risk=float(tape.data["risk"][i, 7]))
            for i in np.flatnonzero(late)
        ],
    }
    return out, top, best


def distinct_late(reports):
    by_arm = {k: {i for v in m.VIEWS for i in reports[k][v]["late_top"]} for k in ARMS}
    return sorted(by_arm["real"] - set().union(*(by_arm[k] for k in ARMS if k != "real")))


def rank(native, masked, native_result, multiview, bank, out):
    if out.exists():
        raise FileExistsError(out)
    h, start = Hashes(), time.monotonic()
    seal_path, feature_path = bank / "seal.json", bank / "features.npz"
    receipt = b.r.read(seal_path)
    result = b.r.read(multiview / "result.json")
    verified = b.r.read(multiview / "verification.json")
    if (
        receipt["status"] != "SEALED_REACHED_PHASE_BEFORE_OUTCOME_READ" or receipt["outcome_reads"]
        or receipt["native_seal_sha256"] != h(native / "seal.json")
        or receipt["feature_sha256"] != h(feature_path)
        or verified["result_sha256"] != h(multiview / "result.json") or result["source_gate"]
        or not all(h(p) == d for p,d in receipt["hashes"].items())
    ):
        raise ValueError("phase features or preceding verified result changed")
    for p in (seal_path, PLAN, Path(__file__), Path(m.__file__), Path(sparse.__file__),
              Path(sel.__file__), Path(sel.fit.__file__), multiview / "verification.json"):
        h(p)
    tapes, ids, flags = m.load(native, masked, native_result, h)
    owner = tapes["native"].owner
    folds = sel.fit.shot_folds(ids, sel.N_FOLDS)
    with np.load(feature_path, allow_pickle=False) as f:
        if not all(np.array_equal(f[k], v) for k,v in (
            ("ids",ids), ("clock",tapes["native"].data["clock"]), ("owner",owner),
        )):
            raise ValueError("phase features and action rows use different reached data")
        cues = {k: f[k] for k in FIELDS}
    common = cues["supported"].any(axis=1)
    if len(common) != receipt["blocks"] or not np.isfinite(cues["real"]).all():
        raise ValueError("unusable sealed support")
    if (not np.array_equal(cues["real"][~cues["supported"]], np.zeros((~cues["supported"]).sum()))
            or np.any(cues["shifted"][~cues["supported"]])):
        raise ValueError("unsupported phase observations must be zero with explicit support flag")
    base = {v: sel.features(tapes[v].data, "risk") for v in m.VIEWS}
    reports, records, archived = {}, {}, {}
    for arm in ARMS:
        features = {v: add_features(base[v], cues, arm) for v in m.VIEWS}
        fitted = []
        held_score = {v: np.empty(tapes[v].hard.shape, dtype=np.float64) for v in m.VIEWS}
        for k in range(sel.N_FOLDS):
            train, _, held = m.rows_for_fold(folds, owner, k)
            model = m.train_model(tapes, features, train, m.VIEWS)
            fitted.append(model)
            for v in m.VIEWS:
                held_score[v][held] = sparse.event_scores(model, features[v][held])
        records[arm] = fitted
        reports[arm] = {}
        for v in m.VIEWS:
            reports[arm][v], a, r = describe(tapes[v], held_score[v], cues, ids, flags)
            archived[f"{arm}__{v}"] = np.stack((a,r),axis=1)
        print(arm,{v:reports[arm][v]["late_top"] for v in m.VIEWS},flush=True)
    h.check()
    distinct = distinct_late(reports)
    out.mkdir(parents=True, exist_ok=False)
    model_path, ranks_path = out / "models.json", out / "rankings.npz"
    b.r.write_new(model_path, dict(
        fits=records,
        splits=[dict(held=ids[folds==k].tolist(), calibrate=ids[folds==(k+1)%5].tolist(),
                     fit=ids[(folds!=k)&(folds!=(k+1)%5)].tolist())
                for k in range(sel.N_FOLDS)],
    ))
    b.r.save_arrays(ranks_path, ids=ids, folds=folds, owner=owner, **archived)
    b.r.write_new(out / "result.json", dict(
        status="SCORED_HELD_PHASE_RANK_ONLY_NOT_ONLINE", supports_blocks=int(common.sum()),
        support_shots=int(len(np.unique(owner[common]))),
        all_terms_blocks=int(cues["supported"].all(axis=1).sum()),
        n_shots=len(ids), blocks=len(owner), reports=reports, distinct_late=distinct,
        rank_gate=bool(distinct), first_alarm_replays=0, new_val_test_east_reads=0,
        models_sha256=h(model_path), rankings_sha256=h(ranks_path),
        input_sha256={str(p): v[1] for p,v in h.cache.items()},
        elapsed_seconds=time.monotonic()-start,
    ))
    print(dict(common=int(common.sum()), distinct=distinct, rank_gate=bool(distinct)),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    command = sub.add_parser("seal")
    for k in ("native", "output"):
        command.add_argument(k, type=Path)
    command = sub.add_parser("rank")
    for k in ("native", "masked", "native_result", "multiview", "bank", "output"):
        command.add_argument(k, type=Path)
    args = parser.parse_args()
    if args.mode == "seal":
        seal(args.native, args.output)
    else:
        rank(args.native,args.masked,args.native_result,args.multiview,args.bank,args.output)
