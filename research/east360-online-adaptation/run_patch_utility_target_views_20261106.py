"""Measure unchanged action-menu capacity under two real EAST mask patterns."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

import run_patch_utility_bank_20261106 as b
import run_patch_utility_selection_20261106 as sel
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import menu_analysis


PLAN = b.r.ROOT / "PATCH_UTILITY_TARGET_VIEWS_PROTOCOL_20261106.md"
VIEWS = {
    name: tuple(b.r.io23.RAW23.index(channel) for channel in channels)
    for name, channels in (
        ("no_MNM", ("MNM",)),
        ("no_MNM_CIII", ("MNM", "CIII")),
    )
}


def mask(raw, name):
    columns = VIEWS[name]
    v, a = raw["values"].copy(), raw["available"].copy()
    v[:, columns], a[:, columns] = 0.0, False
    return dict(raw, values=v, available=a)


def seal(native, root):
    if root.exists():
        raise FileExistsError(root)
    h, start = Hashes(), time.monotonic()
    np_, ns = native / "protocol.json", native / "seal.json"
    config, archived = b.r.read(np_), b.r.read(ns)
    prior, inv = b.bank.src.source_inputs(h)
    roles = prior["roles"]
    if (
        len(inv) != 1353
        or [q["shot"] for q in inv] != roles["train"]
        or set(roles["train"]) & (set(roles["val"]) | set(roles["test"]))
        or config["shots"] != roles["train"]
        or config["outcome_reads"]
        or archived["outcome_reads"]
        or config["tau"] != b.TAU
        or tuple(config["actions"]) != b.ORDER
        or h(np_) != archived["protocol_sha256"]
        or not all(h(p) == sha for p, sha in config["inputs"].items())
    ):
        raise ValueError("fixed native TRAIN comparator changed")
    for p in (ns, PLAN, Path(__file__), Path(b.__file__)):
        h(p)
    b.r.configure()
    base, parent = b.r.load_parent(b.r.DEFAULT_PARENT)
    digest = b.r.state_digest(base)
    if (
        digest != config["source_state_sha256"]
        or digest != prior["source_state_sha256"]
        or parent["threshold"] != b.r.THRESHOLD
    ):
        raise ValueError("qualified frozen source changed")
    base.cuda()
    basis = b.op.basis(base)
    inputs = {str(k): v[1] for k, v in h.cache.items()}
    root.mkdir(parents=True, exist_ok=False)
    shards = {name: [] for name in VIEWS}
    total_b, total_n = 0, 0
    for name in VIEWS:
        (root / name).mkdir()
        b.r.write_new(
            root / name / "protocol.json",
            dict(config, inputs=inputs, observed_view=name,
                 removed_columns=VIEWS[name], native_seal_sha256=h(ns)),
        )
    for offset in range(0, len(inv), 100):
        tapes = {name: defaultdict(list) for name in VIEWS}
        notes = {name: [] for name in VIEWS}
        for j, q in enumerate(inv[offset:offset + 100], offset + 1):
            sid = q["shot"]
            path = b.r.io23.path_for("J-TEXT", sid)
            if h(path) != q["sha256"]:
                raise ValueError("fixed TRAIN waveform changed")
            raw = b.r.simple_data.read_shot("J-TEXT", sid)
            n = len(raw["times"])
            nb = len(range(63, n - 1, 32))
            total_b += nb
            total_n += n
            for name in VIEWS:
                changed = mask(raw, name)
                y = b.generate(base, basis, changed)
                for k in (*b.FIELDS, "native"):
                    tapes[name][k].append(y[k])
                tapes[name]["times"].append(changed["times"])
                meta = dict(shot=sid, points=n, blocks=nb, raw_sha256=h(path))
                if j in (101, 501, 1001) and n > 119:
                    meta["prefix_check"] = b.prefix_check(base, basis, changed, y, 119)
                notes[name].append(meta)
        for name in VIEWS:
            fn = root / name / f"tape_{offset//100:02d}.npz"
            b.r.save_arrays(fn, **{k: np.concatenate(v) for k, v in tapes[name].items()})
            shards[name].append(dict(file=fn.name, sha256=h(fn), shots=notes[name]))
        print(f"Sealed masked candidates for {min(offset + 100, len(inv))}/{len(inv)} shots", flush=True)
    if b.r.state_digest(base) != digest or any(p.grad is not None for p in base.parameters()):
        raise ValueError("frozen source parameters changed")
    h.check()
    if total_n != archived["points"] or total_b != archived["blocks"]:
        raise ValueError("masked views changed the time grid")
    for name in VIEWS:
        p = root / name
        b.r.write_new(p / "seal.json", dict(
            status="SEALED_BEFORE_OUTCOME_REVEAL", protocol_sha256=h(p/"protocol.json"),
            shards=shards[name], points=total_n, blocks=total_b, outcome_reads=0,
            elapsed_seconds=time.monotonic()-start,
        ))
    b.r.write_new(root / "completed.json", dict(
        status="BOTH_VIEWS_SEALED_BEFORE_OUTCOME_REVEAL", native_seal_sha256=h(ns),
        seals={name: h(root / name / "seal.json") for name in VIEWS},
        elapsed_seconds=time.monotonic()-start, outcome_reads=0,
    ))


def score(root, native_result, output):
    h, start = Hashes(), time.monotonic()
    done, original = b.r.read(root / "completed.json"), b.r.read(native_result)
    if (
        done["status"] != "BOTH_VIEWS_SEALED_BEFORE_OUTCOME_REVEAL"
        or done["outcome_reads"]
        or done["native_seal_sha256"] != original["bank_seal_sha256"]
        or set(done["seals"]) != set(VIEWS)
        or any(h(root / k / "seal.json") != digest for k, digest in done["seals"].items())
    ):
        raise ValueError("both views must be complete before HDF5 is opened")
    hashes = {str(p): h(p) for p in (Path(__file__), PLAN, native_result, root / "completed.json")}
    source = original["capacity"]
    benefit = set(source["single_edit_timely"]) | set(source["single_edit_fp_removed"])
    extra, rows = set(), {}
    for name in VIEWS:
        shots, data, hard, _, _ = sel.load_bank(root / name, h)
        ids = np.array([q.sid for q in shots])
        folds = dict(zip(ids.tolist(), sel.fit.shot_folds(ids, sel.N_FOLDS).tolist()))
        base = sel.summarize([(q.sid, q.teacher["baseline"], q.teacher["baseline"]) for q in shots])
        capacity = sel.support(shots, hard, data["allowed"])
        positive = capacity["single_edit_timely"] + capacity["single_edit_fp_removed"]
        benefit.update(positive)
        scalar_t, scalar_fp = [], []
        for q in shots:
            if q.teacher["baseline"]["timely"] or (not q.disruptive and not q.teacher["baseline"]["false_alarm"]):
                continue
            ix = q.blocks
            c, e, m, a = (data[k][ix] for k in ("curves", "effect", "clock", "allowed"))
            scalar = sel.scalar_curves(c, e, m, len(q.native))
            result = menu_analysis(q, m, scalar, a, b.TAU)
            if (result["hard"] > 0).any():
                (scalar_t if q.disruptive else scalar_fp).append(q.sid)
        internal_only = set(capacity["single_edit_timely"]) - set(scalar_t)
        extra.update(internal_only)
        rows[name] = dict(
            baseline=base, capacity=capacity, scalar_timely=scalar_t, scalar_fp_removed=scalar_fp,
            unique_benefit_shots=sorted(positive),
            unique_harm_shots=[q.sid for q in shots if (hard[q.blocks] < 0).any()],
            internal_only_timely=sorted(internal_only),
        )
        print(name, base["timely"], base["fp"], len(positive), sorted(internal_only), flush=True)
    h.check()
    covering = sorted({folds[i] for i in extra})
    passed = len(benefit) >= 20 and len(extra) >= 4 and len(covering) >= 3
    b.r.write_new(output, dict(
        status="SCORED_TARGET_MASK_TRAIN_CAPACITY_NOT_ONLINE",
        native=dict(baseline=original["baseline"], capacity=source),
        views=rows, pooled_benefit_shots=sorted(benefit),
        mask_internal_only_timely=sorted(extra), mask_internal_only_folds=covering,
        support_gate=passed, hashes=hashes, elapsed_seconds=time.monotonic()-start,
        new_val_test_east_reads=0,
    ))
    print(dict(pooled_benefit=len(benefit), internal_only_timely=len(extra),
               held_folds=covering, support_gate=passed), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    seal_p = modes.add_parser("seal")
    seal_p.add_argument("native", type=Path)
    seal_p.add_argument("output", type=Path)
    score_p = modes.add_parser("score")
    for arg in ("bank", "native_result", "output"):
        score_p.add_argument(arg, type=Path)
    arg = cli.parse_args()
    if arg.mode == "seal":
        seal(arg.native, arg.output)
    else:
        score(arg.bank, arg.native_result, arg.output)
