"""CPU-only, post-completion audit of the raw confirmation replay."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "probe_outputs/qualified_source_static360_20261104_v1"
PARENT = SOURCE / "protocol.json"
POINT = ROOT / "probe_outputs/trajectory_point_attribution_20261105_v1"
DONOR = ROOT / "probe_outputs/point_donor_controls_20261105_v1"
GUARD = ROOT / "probe_outputs/point_confirmation_20261105_v1"
SOURCE_DONE_SHA = "ef41bc2db11c19d12f7fd2f6b2fea24a5208d0daa4cf5b818c996bf80d66d78f"
COHORT = ROOT / "config/splits/east_hdf5_authoritative_balanced360_v1.json"
ROSTER = ROOT / "probe_outputs/conditional_confirmation_roster_20261105_v1.json"
RAW_ROOT = ROOT / "data/EAST/unified_hdf5"
FIXED_COUNT, FIRST = 360, 135
ARMS = [
    "static",
    "point_real",
    "point_lagged",
    "previous_clock",
    "first_full",
    "intra_confirm_past32",
    "donor_confirm_past32",
]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class Hashes:
    """Hash each canonical path once and detect changes through the audit."""

    def __init__(self):
        self.cache = {}

    def __call__(self, path):
        path = Path(path).resolve(strict=True)
        before = signature(path)
        if path not in self.cache:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            need(signature(path) == before, f"file changed while hashing: {path}")
            self.cache[path] = (before, digest.hexdigest())
        need(self.cache[path][0] == before, f"file changed after hashing: {path}")
        return self.cache[path][1]

    def check(self):
        for path, (old, _) in self.cache.items():
            need(signature(path) == old, f"file changed during audit: {path}")


def score(times, p, truth, threshold):
    flag, down = truth["disruptive"], truth["down"]
    hits = np.flatnonzero(
        (p >= threshold)
        & (((down - times) * 1000 >= -1e-6) if flag else np.ones(len(times), bool))
    )
    first = int(hits[0]) if len(hits) else None
    lead = (
        round(float((down - times[first]) * 1000), 9)
        if flag and first is not None
        else None
    )
    status = (
        ("miss" if first is None else "tp" if lead > 30 else "tardy")
        if flag
        else ("fp" if first is not None else "tn")
    )
    return dict(
        shot=truth["shot"],
        disruptive=flag,
        down=down,
        status=status,
        first_index=first,
        lead_ms=lead,
    )


def counts(rows):
    d = sum(r["disruptive"] for r in rows)
    s = len(rows) - d
    tp = sum(r["status"] == "tp" for r in rows)
    fp = sum(r["status"] == "fp" for r in rows)
    early = sum(r["status"] == "tp" and r["lead_ms"] > 3000 for r in rows)
    tpr, fpr = (tp / d if d else None), (fp / s if s else None)
    return dict(
        n_disruptive=d,
        n_safe=s,
        tp=tp,
        fp=fp,
        tpr=tpr,
        fpr=fpr,
        j=tpr - fpr if tpr is not None and fpr is not None else None,
        early_over3000=early,
        tp_30_to_3000=tp - early,
    )


def pair(a, b, capped=False):
    pairs = list(zip(a, b, strict=True))
    need(
        all(
            (x["shot"], x["disruptive"], x["down"])
            == (y["shot"], y["disruptive"], y["down"])
            for x, y in pairs
        ),
        "pair identity",
    )

    def good(r):
        return r["status"] == "tp" and (not capped or r["lead_ms"] <= 3000)

    return dict(
        tp_gained=[x["shot"] for x, y in pairs if good(x) and not good(y)],
        tp_lost=[x["shot"] for x, y in pairs if good(y) and not good(x)],
        fp_added=[
            x["shot"] for x, y in pairs if x["status"] == "fp" and y["status"] != "fp"
        ],
        fp_removed=[
            x["shot"] for x, y in pairs if y["status"] == "fp" and x["status"] != "fp"
        ],
    )


def confirmed(base, real, shadow, threshold):
    candidates = np.flatnonzero((base < threshold) & (real >= threshold))
    support = np.flatnonzero((base < threshold) & (shadow >= threshold))
    right = np.searchsorted(support, candidates, side="right")
    left = np.searchsorted(support, candidates - 32, side="left")
    expected = base.copy()
    selected = candidates[right > left]
    expected[selected] = real[selected]
    return expected


def raw_truth(path, sid):
    # Parse the original HDF5 directly; safe DownTime is never a disruption time.
    with h5py.File(path, "r") as f:
        flag = np.asarray(f["meta/IsDisrupt"][()])
        need(
            flag.size == 1 and flag.dtype.kind in "biuf" and flag.item() in (0, 1),
            "invalid HDF IsDisrupt",
        )
        disruptive = bool(flag.item())
        down = None
        if disruptive:
            value = np.asarray(f["meta/DownTime"][()])
            need(
                value.size == 1
                and value.dtype.kind in "biuf"
                and np.isfinite(value.item()),
                "invalid HDF DownTime",
            )
            down = float(value.item())
        ip = f["data/ip"]
        n = len(ip)
        start = float(np.asarray(ip.attrs["StartTime"]).reshape(-1)[0])
        rate = float(np.asarray(ip.attrs["SampleRate"]).reshape(-1)[0])
    need(
        n > 0 and np.isfinite([start, rate]).all() and rate > 0, "invalid signal clock"
    )
    times = start + np.arange(n, dtype=float) / rate
    need(
        np.isfinite(times).all() and (np.diff(times) > 0).all(), "invalid signal clock"
    )
    return dict(shot=sid, disruptive=disruptive, down=down), times


def verify(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    done = read(run / "completed.json")
    need(done["status"] == "COMPLETE_RAW_CONFIRMATION_REPLAY", "run is not COMPLETE")
    hashes = Hashes()
    hashes(run / "completed.json")
    required = ("protocol.json", "per_shot.json", "summary.json")
    need(set(done["artifact_sha256"]) == set(required), "artifact seal keys")
    for name in required:
        need(
            hashes(run / name) == done["artifact_sha256"][name],
            f"artifact hash: {name}",
        )
    cfg, saved, summary = (read(run / n) for n in required)
    mode, order, arms, threshold = (
        cfg["mode"],
        cfg["order"],
        cfg["arms"],
        cfg["threshold"],
    )
    need(mode in ("fixed", "conditional") and arms == ARMS, "mode or arms")
    need(
        order
        and all(type(s) is int and s > 0 for s in order)
        and len(set(order)) == len(order),
        "order IDs",
    )
    need(
        type(threshold) in (int, float)
        and np.isfinite(threshold)
        and 0 < threshold < 1,
        "threshold",
    )
    need(
        cfg["independent_validation"] is False
        and cfg["fixed_prefix_shots"] == (100 if mode == "fixed" else 0),
        "protocol scope",
    )
    need(
        done["n_shots"] == len(order)
        and done["n_curves"] == len(order) * len(arms)
        and len(done["sealed"]) == len(order),
        "completion counts",
    )
    inputs = cfg["input_sha256"]
    for name, expected in inputs.items():
        need(
            Path(name).is_absolute() and hashes(name) == expected, f"input hash: {name}"
        )
    bound = {Path(name).resolve() for name in inputs}
    need(
        PARENT.resolve() in bound and (mode == "fixed" or ROSTER.resolve() in bound),
        "locked order or roster not bound",
    )
    parent = read(PARENT)
    need(threshold == parent["threshold"], "locked threshold")
    parent_ids = [r["shot"] for r in parent["order"]]
    source_done = read(SOURCE / "completed.json")
    need(
        hashes(SOURCE / "completed.json") == SOURCE_DONE_SHA
        and source_done["artifact_sha256"]["protocol.json"] == hashes(PARENT)
        and len(source_done["seals"]) == len(parent_ids),
        "locked source identity",
    )
    need(
        len(parent_ids) == FIXED_COUNT and len(set(parent_ids)) == len(parent_ids),
        "locked source order",
    )
    later = cfg["later_proxy_ids"]
    need(len(later) == len(set(later)) and set(later) <= set(order), "later proxy IDs")
    raw_hashes = cfg["raw_sha256"]
    need(set(raw_hashes) == {str(s) for s in order}, "raw hash roster")
    archives = {}
    if mode == "fixed":
        hashes(COHORT)
        cohort = read(COHORT)["test"]
        ids = cohort["disruption_shots"] + cohort["non_disruption_shots"]
        need(
            len(order) == FIXED_COUNT
            and order == parent_ids
            and len(ids) == len(set(ids))
            and set(ids) == set(order)
            and not later,
            "locked fixed order or roster",
        )
        for root in (POINT, DONOR, GUARD):
            need(
                (root / "completed.json").resolve() in bound,
                "fixed reference receipt not bound",
            )
            archives[root] = read(root / "completed.json")["sealed"]
            need(len(archives[root]) == len(order) - 100, "fixed reference count")
    else:
        roster = read(ROSTER)
        need(
            roster["status"] == "LOCKED_CONDITIONAL_SIGNAL_ONLY_ROSTER"
            and roster["independent_validation"] is False
            and order == roster["ids"] == sorted(order)
            and later == roster["later_proxy_ids"],
            "locked conditional order or roster",
        )
        need(
            all(raw_hashes[str(s)] == roster["raw_sha256"][str(s)] for s in order),
            "locked roster raw hash",
        )
        need(
            not set(order) & set(parent_ids)
            and later == [s for s in order if s > max(parent_ids[:100])],
            "conditional roster or later proxy",
        )
    need(
        set(saved) == set(arms) and all(len(saved[a]) == len(order) for a in arms),
        "per-shot roster",
    )
    observed = {a: [] for a in arms}
    previous_n = parity_curves = 0
    for rank, (sid, binding) in enumerate(zip(order, done["sealed"], strict=True), 1):
        stem = run / "shots" / f"{rank:04d}_{sid}"
        curve = stem.with_suffix(".npz")
        seal_path, reveal_path = (
            stem.with_name(stem.name + suffix)
            for suffix in ("_seal.json", "_revealed.json")
        )
        need(
            binding
            == dict(
                shot=sid,
                curve_sha256=hashes(curve),
                seal_sha256=hashes(seal_path),
                revealed_sha256=hashes(reveal_path),
            ),
            "shot seal hash",
        )
        seal, reveal = read(seal_path), read(reveal_path)
        need(
            (seal["rank"], seal["shot"], reveal["shot"]) == (rank, sid, sid)
            and seal["curve_sha256"] == binding["curve_sha256"]
            and reveal["after_seal_sha256"] == binding["seal_sha256"]
            and seal["current_outcome_used_in_prediction"] is False,
            "seal/reveal identity",
        )
        need(
            curve.stat().st_mtime_ns
            <= seal_path.stat().st_mtime_ns
            <= reveal_path.stat().st_mtime_ns
            <= (run / "completed.json").stat().st_mtime_ns,
            "seal/reveal order",
        )
        donor = (
            order[rank - 2]
            if rank > 1 and rank > cfg["fixed_prefix_shots"] and previous_n > FIRST
            else None
        )
        need(seal["donor"] == donor, "donor identity or maturity")
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        need(hashes(raw) == raw_hashes[str(sid)] == seal["raw_sha256"], "raw hash")
        truth, expected_times = raw_truth(raw, sid)
        hashes(raw)  # Same digest, with a stability check after the HDF read.
        with np.load(curve, allow_pickle=False) as z:
            need(set(z.files) == {"times", *arms}, "curve keys")
            times = z["times"]
            need(np.array_equal(times, expected_times), "signal clock")
            for arm in arms:
                p = z[arm]
                need(
                    p.shape == times.shape
                    and np.isfinite(p).all()
                    and ((p >= 0) & (p <= 1)).all(),
                    f"invalid curve: {arm}",
                )
                prefix = (
                    len(times)
                    if mode == "fixed" and rank <= 100
                    else min(FIRST, len(times))
                )
                need(np.array_equal(p[:prefix], z["static"][:prefix]), "static prefix")
                row = score(times, p, truth, threshold)
                need(
                    row == reveal["rows"][arm] == saved[arm][rank - 1],
                    f"row recount: {arm}",
                )
                observed[arm].append(row)
            need(
                donor is not None or np.array_equal(z["previous_clock"], z["static"]),
                "disabled donor curve",
            )
            for target, shadow in (
                ("intra_confirm_past32", "point_lagged"),
                ("donor_confirm_past32", "previous_clock"),
            ):
                need(
                    np.array_equal(
                        z[target],
                        confirmed(z["static"], z["point_real"], z[shadow], threshold),
                    ),
                    f"confirmation gate: {target}",
                )
            if mode == "fixed":
                ref = SOURCE / "shots" / curve.name
                source_seal = SOURCE / "shots" / seal_path.name
                hashes(source_seal)
                source_binding = source_done["seals"][rank - 1]
                need(
                    read(source_seal) == source_binding
                    and source_binding["shot"] == sid
                    and source_binding["current_outcome_used_in_prediction"] is False
                    and hashes(ref) == source_binding["curve_sha256"],
                    "source curve seal",
                )
                with np.load(ref, allow_pickle=False) as old:
                    need(np.array_equal(old["times"], times), "source signal clock")
                    base = old["map_after20"]
                    need(
                        base.shape == times.shape and np.isfinite(base).all(),
                        "source curve",
                    )
                    for arm in arms if rank <= 100 else ["static"]:
                        need(
                            np.array_equal(z[arm] >= threshold, base >= threshold),
                            f"fixed crossing parity: {arm}",
                        )
                        parity_curves += 1
                if rank > 100:
                    for root, names in (
                        (POINT, ("static", "point_real", "point_lagged")),
                        (DONOR, ("previous_clock", "first_full")),
                        (GUARD, ("intra_confirm_past32", "donor_confirm_past32")),
                    ):
                        ref = root / "shots" / curve.name
                        receipt = archives[root][rank - 101]
                        need(
                            receipt["shot"] == sid
                            and hashes(ref) == receipt["curve_sha256"],
                            "fixed reference seal",
                        )
                        with np.load(ref, allow_pickle=False) as old:
                            need(
                                np.array_equal(old["times"], times),
                                "fixed reference signal clock",
                            )
                            for arm in names:
                                reference = old[arm]
                                need(
                                    reference.shape == times.shape
                                    and np.isfinite(reference).all()
                                    and np.array_equal(
                                        z[arm] >= threshold, reference >= threshold
                                    ),
                                    f"fixed crossing parity: {arm}",
                                )
                                parity_curves += arm != "static"
        previous_n = len(expected_times)
        if rank % 25 == 0:
            print("verified", mode, rank, "of", len(order), flush=True)
    need(
        mode != "fixed" or parity_curves == len(order) * len(arms),
        "fixed crossing parity count",
    )
    need(
        set(summary)
        >= {"full", "later_proxy", "paired_to_static", "paired_capped_to_static"},
        "summary keys",
    )
    groups = {a: counts(rows) for a, rows in observed.items()}
    subset = (
        {
            a: counts([r for r in rows if r["shot"] in set(later)])
            for a, rows in observed.items()
        }
        if later
        else None
    )
    need(
        summary["full"] == groups and summary["later_proxy"] == subset, "summary counts"
    )
    for key, capped in (("paired_to_static", False), ("paired_capped_to_static", True)):
        need(
            summary[key]
            == {
                a: pair(rows, observed["static"], capped)
                for a, rows in observed.items()
                if a != "static"
            },
            "summary paired counts",
        )
    hashes.check()
    result = dict(
        status="PASS_RAW_CONFIRMATION_RECOUNT",
        mode=mode,
        n_shots=len(order),
        n_curves=done["n_curves"],
        fixed_crossing_parity_curves=parity_curves if mode == "fixed" else None,
        independent_validation=False,
        summary={
            k: summary[k]
            for k in (
                "full",
                "later_proxy",
                "paired_to_static",
                "paired_capped_to_static",
            )
        },
        producer_completion_sha256=hashes(run / "completed.json"),
        verifier_sha256=hashes(Path(__file__)),
        limitation="Seals and timestamps do not prove internal prediction causality.",
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(result["status"], mode, len(order), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.run, args.output)
