"""Independently recount the sealed six-arm, fixed-threshold replay."""

import argparse
from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import io23
from joint_lora_hindsight_capacity_20261104 import (
    SOURCE,
    V5_METADATA_SHA,
    read,
    sha,
    write_new,
)


FRESH = (
    io23.ROOT / "probe_outputs/hdf5_outcome_authority_20261105_v1/hdf5_metadata.json"
)


def _score(times, p, row, threshold):
    disruptive, down = row["disruptive"], row["down"]
    eligible = (
        (down - times) * 1000 >= -1e-6
        if disruptive
        else np.ones(len(times), dtype=bool)
    )
    hits = np.flatnonzero(eligible & (p >= threshold))
    first = int(hits[0]) if len(hits) else None
    lead = (
        round(float(1000 * (down - times[first])), 9)
        if disruptive and first is not None
        else None
    )
    if disruptive:
        status = "miss" if first is None else ("tp" if lead > 30 else "tardy")
    else:
        status = "fp" if first is not None else "tn"
    return dict(
        shot=row["shot"],
        disruptive=disruptive,
        down=down,
        status=status,
        first_index=first,
        lead_ms=lead,
    )


def _recount(rows):
    d = sum(r["disruptive"] for r in rows)
    s = len(rows) - d
    tp = sum(r["status"] == "tp" for r in rows)
    fp = sum(r["status"] == "fp" for r in rows)
    over = sum(r["status"] == "tp" and r["lead_ms"] > 3000 for r in rows)
    return dict(
        n_disruptive=d,
        n_safe=s,
        tp=tp,
        fp=fp,
        tpr=tp / d,
        fpr=fp / s,
        j=tp / d - fp / s,
        early_over3000=over,
        tp_30_to_3000=tp - over,
    )


def _fit_cost(z, lead, flag, tau, mode):
    if mode == "point":
        labels = flag & (lead >= 0) & (lead <= 125)
        return np.logaddexp(0, (1 - 2 * labels.astype(float)) * (z - tau)).mean()
    if not flag:
        k = (len(z) + 3) // 4
        top = np.sort(z)[-k:]
        return (
            np.logaddexp(0, 2 * (z.max() - tau))
            + 0.5 * np.logaddexp(0, 2 * (top - tau)).mean()
        )
    total = np.logaddexp(0, 2 * (tau - z[lead > 30].max()))
    bins = []
    for lo, hi in zip((30, 125, 500, 1500), (125, 500, 1500, 3000), strict=True):
        band = z[(lead > lo) & (lead <= hi)]
        if len(band):
            bins.append(band.max())
    if bins:
        aggregate = 0.5 * (np.logaddexp.reduce(2 * np.array(bins)) - np.log(len(bins)))
        total += 0.25 * np.logaddexp(0, 2 * (tau - aggregate))
    if (lead > 3000).any():
        total += 0.25 * np.exp(-np.logaddexp(0, -2 * (z[lead > 3000].max() - tau)))
    return total


def _fit_audit(bank, fit, cfg, historical):
    tau = np.log(cfg["threshold"] / (1 - cfg["threshold"]))
    z = bank["z"].astype(float)
    g = bank["g"].astype(float)
    n = bank["valid"].sum(1)
    slices = [
        slice(a, b) for a, b in zip(np.r_[0, n.cumsum()[:-1]], n.cumsum(), strict=True)
    ]
    w = bank["shot_weight"]
    flags = bank["disruptive"]
    assert np.isclose(w.sum(), 1) and len(z) == n.sum()
    classes = [
        "d_good"
        if r["status"] == "tp"
        else "d_bad"
        if r["disruptive"]
        else "s_bad"
        if r["status"] == "fp"
        else "s_good"
        for r in historical
    ]
    for label in set(classes):
        mask = np.array([x == label for x in classes])
        np.testing.assert_allclose(w[mask], 0.25 / mask.sum(), rtol=0, atol=1e-15)
    report = {}
    for arm, m in fit["fit_results"].items():
        matrix = bank[arm + "_matrix"]
        x = bank[m["feature"]].astype(float) / bank[arm + "_scale"]
        u = x @ matrix
        d = cfg["temporary_radius"] * u / np.sqrt(1 + (u * u).sum(1, keepdims=True))
        moved = z + (g * d).sum(1)
        unregularized = {}
        for name, values in (("before", z), ("after", moved)):
            row = np.array(
                [
                    _fit_cost(
                        values[s], bank["lead"][i, : n[i]], flags[i], tau, m["mode"]
                    )
                    for i, s in enumerate(slices)
                ]
            )
            total = w @ row + (cfg["ridge"] / 2) * (matrix * matrix).sum() * (
                name == "after"
            )
            assert abs(total - m[name]) < 2e-10
            unregularized[name] = row
        report[arm] = {
            label: float(
                np.mean(
                    (unregularized["after"] - unregularized["before"])[
                        np.array(classes) == label
                    ]
                )
            )
            for label in sorted(set(classes))
        }
    # Effective credit coordinates of the main+auxiliary formula at the zero
    # state (unique representatives; exact float ties may share gradients).
    credit = dict(d_good=[], d_bad=[], s_good=[], s_bad=[])
    for i, (s, label) in enumerate(zip(slices, classes, strict=True)):
        v, lead = z[s], bank["lead"][i, : n[i]]
        choose = set()
        if flags[i]:
            masks = [lead > 30, lead > 3000] + [
                (lead > lo) & (lead <= hi)
                for lo, hi in zip(
                    (30, 125, 500, 1500), (125, 500, 1500, 3000), strict=True
                )
            ]
            for mask in masks:
                if mask.any():
                    choose.update(np.flatnonzero(mask & (v == v[mask].max())))
        else:
            choose.update(np.argsort(v)[-((len(v) + 3) // 4) :])
        can = (g[s] != 0).any(1)
        credit[label].append(sum(can[j] for j in choose))
    return dict(
        recomputed_objectives=report,
        trajectory_effective_direct_credit_points_median={
            k: float(np.median(v)) for k, v in credit.items()
        },
    )


def verify(root, output):
    if output.exists():
        raise FileExistsError(output)
    done = read(root / "completed.json")
    assert done["status"] == "COMPLETE_DELAYED_TRAJECTORY_TEACHING360"
    assert (done["n_shots"], done["n_curves"]) == (360, 2160)
    assert all(sha(root / f) == h for f, h in done["artifact_sha256"].items())
    cfg, fit, claimed = [
        read(root / f) for f in ("protocol.json", "fit_receipt.json", "per_shot.json")
    ]
    assert all(sha(Path(f)) == h for f, h in cfg["input_sha256"].items())
    assert sha(FRESH) == V5_METADATA_SHA
    metadata = {r["shot"]: r for r in read(FRESH)["EAST"]}
    order, arms = cfg["order"], cfg["arms"]
    assert len(order) == len(set(order)) == 360 and set(order) == set(
        io23.cohort_ids("EAST")
    )
    assert len(arms) == len(set(arms)) == 6
    first, stride, threshold = cfg["first"], cfg["stride"], cfg["threshold"]
    observed = {k: [] for k in arms}
    update_counts, norm_mismatches, abs_peak = {}, {}, {k: [] for k in arms}
    for rank, (sid, binding) in enumerate(zip(order, done["sealed"], strict=True), 1):
        stem = root / "shots" / f"{rank:04d}_{sid}"
        curve = stem.with_suffix(".npz")
        seal_path, reveal_path = [
            stem.with_name(stem.name + s) for s in ("_seal.json", "_revealed.json")
        ]
        seal, reveal = read(seal_path), read(reveal_path)
        assert binding == dict(
            shot=sid,
            curve_sha256=sha(curve),
            seal_sha256=sha(seal_path),
            revealed_sha256=sha(reveal_path),
        )
        assert seal["curve_sha256"] == binding["curve_sha256"]
        assert (seal["rank"], seal["shot"], reveal["shot"]) == (rank, sid, sid)
        assert reveal["after_seal_sha256"] == binding["seal_sha256"]
        assert (
            curve.stat().st_mtime_ns
            <= seal_path.stat().st_mtime_ns
            <= reveal_path.stat().st_mtime_ns
        )
        assert seal["current_outcome_used_in_prediction"] is False
        assert seal["fit_sha256"] == (fit["fit_sha256"] if rank > 100 else None)
        truth = metadata[sid]
        raw = Path(truth["path"])
        assert sha(raw) == seal["raw_sha256"]
        assert io23.read_outcome("EAST", sid) == (truth["disruptive"], truth["down"])
        with (
            np.load(curve, allow_pickle=False) as trace,
            np.load(
                SOURCE / "shots" / f"{rank:04d}_{sid}.npz", allow_pickle=False
            ) as ref,
        ):
            times = trace["times"]
            expected = (
                truth["start"]
                + np.arange(truth["samples"], dtype=float) / truth["rate"]
            )
            assert np.array_equal(times, expected) and np.array_equal(
                times, ref["times"]
            )
            baseline = trace["static"]
            assert np.array_equal(baseline, ref["map_after20"])
            assert set(trace.files) == {"times", *arms}
            for arm in arms:
                p = trace[arm]
                assert (
                    np.isfinite(p).all()
                    and ((0 <= p) & (p <= 1)).all()
                    and p.shape == times.shape
                )
                prefix = len(times) if rank <= 100 else min(first, len(times))
                assert np.array_equal(p[:prefix], baseline[:prefix])
                row = _score(times, p, truth, threshold)
                assert row == reveal["rows"][arm] == claimed[arm][rank - 1]
                observed[arm].append(row)
                if rank > 100:
                    abs_peak[arm].append(float(np.max(abs(p - baseline))))
        if rank > 100:
            info = seal["action_stats"]
            arrivals = len(range(first, truth["samples"], stride))
            assert info["n_maturities"] == arrivals
            for arm, stats in info["updates"].items():
                assert 0 <= stats["nonzero"] <= arrivals
                assert 0 <= stats["maximum"] <= cfg["temporary_radius"] + 1e-6
                update_counts[arm] = update_counts.get(arm, 0) + stats["nonzero"]
            for arm, value in info["norm_mismatches"].items():
                norm_mismatches[arm] = norm_mismatches.get(arm, 0) + value
    calculated = {
        group: {a: _recount(r[section]) for a, r in observed.items()}
        for group, section in (
            ("full360", slice(None)),
            ("prefix100", slice(100)),
            ("suffix260", slice(100, None)),
        )
    }
    assert calculated == read(root / "summary.json")
    grouped = {}
    for arm in arms:
        if arm == "trajectory_real":
            continue
        real, cmp = observed["trajectory_real"], observed[arm]
        grouped[arm] = {}
        for pos, add, remove in (
            ("tp", "tp_gained", "tp_lost"),
            ("fp", "fp_added", "fp_removed"),
        ):
            grouped[arm][add] = [
                a["shot"]
                for a, b in zip(real, cmp, strict=True)
                if a["status"] == pos != b["status"]
            ]
            grouped[arm][remove] = [
                b["shot"]
                for a, b in zip(real, cmp, strict=True)
                if b["status"] == pos != a["status"]
            ]
    assert grouped == read(root / "paired.json")
    with np.load(root / "fit.npz", allow_pickle=False) as bank:
        assert bank["ids"].tolist() == order[20:100]
        assert bank["disruptive"].tolist() == [
            metadata[s]["disruptive"] for s in order[20:100]
        ]
        k = 0
        for i, sid in enumerate(order[20:100]):
            n = int(bank["valid"][i].sum())
            truth = metadata[sid]
            query = bank["query"][k : k + n]
            assert (
                n
                and (np.diff(query) > 0).all()
                and query[0] >= 0
                and query[-1] < truth["samples"]
            )
            if truth["disruptive"]:
                times = truth["start"] + query / truth["rate"]
                assert np.array_equal(
                    bank["lead"][i, :n], np.round((truth["down"] - times) * 1000, 9)
                )
            assert not bank["g"][k : k + n][query < first].any()
            k += n
        assert k == fit["n_endpoints"]
        fit_profile = _fit_audit(bank, fit, cfg, observed["static"][20:100])
    write_new(
        output,
        dict(
            status="PASS_RECOUNT_COMPLETE_TEACHING360",
            n_shots=360,
            n_curves=2160,
            summary=calculated,
            paired=grouped,
            nonzero_arrival_writes=update_counts,
            norm_mismatches=norm_mismatches,
            hdf5_authority_sha256=sha(FRESH),
            median_shot_peak_absolute_curve_change={
                k: float(np.median(v)) for k, v in abs_peak.items()
            },
            fit_profile=fit_profile,
            producer_completion_sha256=sha(root / "completed.json"),
            auditor_sha256=sha(Path(__file__)),
            independent_future_confirmation=False,
        ),
    )
    print("PASS", calculated["full360"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    arg = parser.parse_args()
    verify(arg.root, arg.output)
