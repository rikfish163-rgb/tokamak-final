"""Independently recount donor and fixed-state controls against HDF truth."""

import argparse
from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import io23
from joint_lora_hindsight_capacity_20261104 import V5_METADATA_SHA, read, sha, write_new
from verify_point_attribution_20261105 import pair
from verify_trajectory_replay_20261105 import FRESH, _recount, _score


def verify(run, parent, output):
    if output.exists():
        raise FileExistsError(output)
    done, cfg, saved, summary = [
        read(run / name)
        for name in ("completed.json", "protocol.json", "per_shot.json", "summary.json")
    ]
    assert done["status"] == "COMPLETE_POINT_DONOR_CONTROLS"
    assert (done["reference_prefix_shots"], done["reexecuted_suffix_shots"]) == (
        100,
        260,
    )
    assert all(sha(run / k) == h for k, h in done["artifact_sha256"].items())
    assert all(sha(Path(k)) == h for k, h in cfg["input_sha256"].items())
    assert cfg["input_sha256"][str((parent / "completed.json").resolve())] == sha(
        parent / "completed.json"
    )
    old, old_cfg, old_rows = [
        read(parent / name)
        for name in ("completed.json", "protocol.json", "per_shot.json")
    ]
    assert cfg["order"] == old_cfg["order"] and cfg["threshold"] == old_cfg["threshold"]
    assert sha(FRESH) == V5_METADATA_SHA
    truth = {r["shot"]: r for r in read(FRESH)["EAST"]}
    ids, arms, start, stride = (
        cfg["order"],
        cfg["arms"],
        cfg["first_anchor"],
        cfg["stride"],
    )
    assert len(ids) == len(set(ids)) == 360 and set(ids) == set(io23.cohort_ids("EAST"))
    assert len(arms) == len(set(arms)) == 6
    verified = {k: list(old_rows["static"][:100]) for k in arms}
    tails = 0
    writes = {k: 0 for k in arms if k != "static"}
    mismatches = {k: 0 for k in cfg["matched"]}
    donor_offsets = []
    first_norm_roundoff = 0.0
    for rank, (sid, receipt) in enumerate(
        zip(ids[100:], done["sealed"], strict=True), 101
    ):
        stem = run / "shots" / f"{rank:04d}_{sid}"
        path = stem.with_suffix(".npz")
        act, seal, reveal = [
            stem.with_name(stem.name + s)
            for s in ("_controls.npz", "_seal.json", "_revealed.json")
        ]
        note, label = read(seal), read(reveal)
        assert receipt == dict(
            shot=sid,
            curve_sha256=sha(path),
            controls_sha256=sha(act),
            seal_sha256=sha(seal),
            revealed_sha256=sha(reveal),
        )
        assert note["curve_sha256"] == receipt["curve_sha256"]
        assert note["controls_sha256"] == receipt["controls_sha256"]
        assert label["after_seal_sha256"] == receipt["seal_sha256"]
        assert (note["rank"], note["shot"], label["shot"]) == (rank, sid, sid)
        assert (
            max(path.stat().st_mtime_ns, act.stat().st_mtime_ns)
            <= seal.stat().st_mtime_ns
            <= reveal.stat().st_mtime_ns
        )
        assert note["current_outcome_used_in_prediction"] is False
        previous = ids[rank - 2]
        assert (note["donor"], note["donor_rank"]) == (previous, rank - 1)
        n_prev = len(range(start, truth[previous]["samples"], stride))
        r = truth[sid]
        assert io23.read_outcome("EAST", sid) == (r["disruptive"], r["down"])
        assert sha(Path(r["path"])) == note["raw_sha256"]
        n = len(range(start, r["samples"], stride))
        assert (note["donor_n_arrivals"], note["stats"]["n_arrivals"]) == (n_prev, n)
        with np.load(act, allow_pickle=False) as f:
            assert set(f.files) == {"clock", "nearest", *writes}
            assert all(
                v.shape == (n,) and np.isfinite(v).all()
                for v in (f[k] for k in f.files)
            )
            clock, near = f["clock"], f["nearest"]
            assert clock.dtype.kind == near.dtype.kind == "i"
            assert np.array_equal(clock, np.minimum(np.arange(n), n_prev - 1))
            assert ((0 <= near) & (near < n_prev)).all()
            tails += max(0, n - n_prev)
            donor_offsets.append(abs(near - np.arange(n)) * stride)
            for k in writes:
                norm = f[k]
                assert ((0 <= norm) & (norm <= cfg["radius"] + 1e-6)).all()
                q = int((norm > 0).sum()) if k != "first_full" else int(norm[0] > 0)
                assert q == note["stats"]["nonzero_optimizer_writes"][k]
                writes[k] += q
            frozen = f["first_full"]
            assert np.array_equal(frozen, np.full(n, frozen[0]))
            # CUDA reductions on a contiguous matrix and a stride-zero
            # expanded vector need not round to the identical float32 norm.
            # The hold itself must still be exactly constant at every row.
            error = float(abs(frozen[0] - f["point_real"][0]))
            assert error <= 2 * np.finfo(frozen.dtype).eps * cfg["radius"]
            first_norm_roundoff = max(first_norm_roundoff, error)
            for k in mismatches:
                q = int((abs(f[k] - f["point_real"]) > 1e-6).sum())
                assert q == note["stats"]["norm_mismatches"][k]
                mismatches[k] += q
        ref = parent / "shots" / f"{rank:04d}_{sid}.npz"
        assert sha(ref) == old["sealed"][rank - 101]["curve_sha256"]
        with (
            np.load(path, allow_pickle=False) as p,
            np.load(ref, allow_pickle=False) as b,
        ):
            t = p["times"]
            assert set(p.files) == {"times", *arms}
            assert np.array_equal(t, b["times"]) and np.array_equal(
                t, r["start"] + np.arange(r["samples"], dtype=float) / r["rate"]
            )
            assert all(np.array_equal(p[k], b[k]) for k in ("static", "point_real"))
            for k in arms:
                v = p[k]
                assert (
                    v.shape == t.shape
                    and np.isfinite(v).all()
                    and ((0 <= v) & (v <= 1)).all()
                )
                assert np.array_equal(v[:start], p["static"][:start])
                row = _score(t, v, r, cfg["threshold"])
                assert row == label["rows"][k] == saved[k][rank - 1]
                verified[k].append(row)
    assert verified == saved and verified["point_real"] == old_rows["point_real"]
    for group, section in (("full360", slice(None)), ("suffix260", slice(100, None))):
        assert summary[group] == {k: _recount(v[section]) for k, v in verified.items()}
    assert summary["paired"] == {
        k: pair(verified["point_real"], v)
        for k, v in verified.items()
        if k != "point_real"
    }
    out = dict(
        status="PASS_RECOUNT_POINT_DONOR_CONTROLS",
        recomputed_suffix_curves=260 * len(arms),
        all_locked_real_curves_exact=True,
        summary=summary,
        paired_capped={
            k: pair(verified["point_real"], v, capped=True)
            for k, v in verified.items()
            if k != "point_real"
        },
        first_full_nonzero_writes=writes["first_full"],
        nonzero_writes=writes,
        norm_mismatches=mismatches,
        maximum_frozen_vs_first_real_norm_roundoff=first_norm_roundoff,
        previous_clock_held_arrivals=tails,
        nearest_state_sample_offset_quantiles=np.quantile(
            np.concatenate(donor_offsets), [0.25, 0.5, 0.75]
        ).tolist(),
        hdf5_metadata_sha256=sha(FRESH),
        producer_completion_sha256=sha(run / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        independent_validation=False,
    )
    write_new(output, out)
    print(out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "parent", "output"):
        ap.add_argument(arg, type=Path)
    opt = ap.parse_args()
    verify(opt.run, opt.parent, opt.output)
