"""Recount point-feedback controls against raw HDF truth and the locked parent."""

import argparse
from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import io23
from joint_lora_hindsight_capacity_20261104 import read, sha, write_new, V5_METADATA_SHA
from verify_trajectory_replay_20261105 import FRESH, _recount, _score


def pair(a, b, *, capped=False):
    def is_tp(row):
        return row["status"] == "tp" and (not capped or row["lead_ms"] <= 3000.0)

    pairs = list(zip(a, b, strict=True))
    assert all(
        (x["shot"], x["disruptive"], x["down"])
        == (y["shot"], y["disruptive"], y["down"])
        for x, y in pairs
    )
    return dict(
        tp_gained=[x["shot"] for x, y in pairs if is_tp(x) and not is_tp(y)],
        tp_lost=[x["shot"] for x, y in pairs if is_tp(y) and not is_tp(x)],
        fp_added=[
            x["shot"] for x, y in pairs if x["status"] == "fp" and y["status"] != "fp"
        ],
        fp_removed=[
            x["shot"] for x, y in pairs if y["status"] == "fp" and x["status"] != "fp"
        ],
    )


def verify(root, parent, out):
    if out.exists():
        raise FileExistsError(out)
    done, cfg, saved, output = [
        read(root / f)
        for f in ("completed.json", "protocol.json", "per_shot.json", "summary.json")
    ]
    assert done["status"] == "COMPLETE_LOCKED_POINT_ATTRIBUTION"
    assert (done["reexecuted_suffix_shots"], done["reference_prefix_shots"]) == (
        260,
        100,
    )
    assert all(sha(root / k) == h for k, h in done["artifact_sha256"].items())
    assert all(sha(Path(k)) == h for k, h in cfg["input_sha256"].items())
    old_cfg, old_done, old_rows = [
        read(parent / f) for f in ("protocol.json", "completed.json", "per_shot.json")
    ]
    assert cfg["input_sha256"][str((parent / "completed.json").resolve())] == sha(
        parent / "completed.json"
    )
    assert cfg["order"] == old_cfg["order"] and cfg["threshold"] == old_cfg["threshold"]
    assert sha(FRESH) == V5_METADATA_SHA
    meta = {r["shot"]: r for r in read(FRESH)["EAST"]}
    order, arms = cfg["order"], list(cfg["arms"])
    assert len(order) == 360 and len(set(order)) == 360 and len(arms) == 5
    prefix = old_rows["static"][:100]
    all_rows = {k: prefix.copy() for k in arms}
    norm_mismatches = {}
    nonzero = {}
    for rank, (sid, binding) in enumerate(
        zip(order[100:], done["sealed"], strict=True), 101
    ):
        stem = root / "shots" / f"{rank:04d}_{sid}"
        p, s, q = (
            stem.with_suffix(".npz"),
            stem.with_name(stem.name + "_seal.json"),
            stem.with_name(stem.name + "_revealed.json"),
        )
        seal, reveal = read(s), read(q)
        assert binding == dict(
            shot=sid, curve_sha256=sha(p), seal_sha256=sha(s), revealed_sha256=sha(q)
        )
        assert (seal["rank"], seal["shot"], reveal["shot"]) == (rank, sid, sid)
        assert seal["current_outcome_used_in_prediction"] is False
        assert (
            seal["curve_sha256"] == binding["curve_sha256"]
            and reveal["after_seal_sha256"] == binding["seal_sha256"]
        )
        assert p.stat().st_mtime_ns <= s.stat().st_mtime_ns <= q.stat().st_mtime_ns
        r = meta[sid]
        assert io23.read_outcome("EAST", sid) == (r["disruptive"], r["down"])
        assert sha(Path(r["path"])) == seal["raw_sha256"]
        reference = parent / "shots" / f"{rank:04d}_{sid}.npz"
        assert (
            sha(reference)
            == seal["reference_curve_sha256"]
            == old_done["sealed"][rank - 1]["curve_sha256"]
        )
        with (
            np.load(p, allow_pickle=False) as new,
            np.load(reference, allow_pickle=False) as old,
        ):
            t = new["times"]
            assert set(new.files) == {"times", *arms}
            assert np.array_equal(t, old["times"]) and np.array_equal(
                t, r["start"] + np.arange(r["samples"], dtype=float) / r["rate"]
            )
            for k in ("static", "point_real"):
                assert np.array_equal(new[k], old[k])
            for k in arms:
                v = new[k]
                assert (
                    v.shape == t.shape
                    and np.isfinite(v).all()
                    and ((v >= 0) & (v <= 1)).all()
                )
                assert np.array_equal(v[:135], new["static"][:135])
                row = _score(t, v, r, cfg["threshold"])
                assert row == saved[k][rank - 1] == reveal["rows"][k]
                all_rows[k].append(row)
        for k, v in seal["stats"]["norm_mismatches"].items():
            norm_mismatches[k] = norm_mismatches.get(k, 0) + v
        for k, v in seal["stats"]["updates"].items():
            assert v["maximum"] <= cfg["max_radius"] + 1e-6
            nonzero[k] = nonzero.get(k, 0) + v["nonzero"]
    assert all_rows == saved
    assert all_rows["point_real"] == old_rows["point_real"]
    for group, part in (("full360", slice(None)), ("suffix260", slice(100, None))):
        assert {k: _recount(v[part]) for k, v in all_rows.items()} == output[group]
    assert {
        k: pair(all_rows["point_real"], v)
        for k, v in all_rows.items()
        if k != "point_real"
    } == output["paired"]
    result = dict(
        status="PASS_RECOUNT_LOCKED_POINT_ATTRIBUTION",
        recomputed_suffix_curves=260 * len(arms),
        reference_prefix_shots=100,
        all_locked_real_curves_exact=True,
        full360=output["full360"],
        suffix260=output["suffix260"],
        paired=output["paired"],
        paired_capped={
            k: pair(all_rows["point_real"], v, capped=True)
            for k, v in all_rows.items()
            if k != "point_real"
        },
        norm_mismatches=norm_mismatches,
        temporary_nonzero_arrivals=nonzero,
        hdf5_metadata_sha256=sha(FRESH),
        auditor_sha256=sha(Path(__file__)),
        producer_completion_sha256=sha(root / "completed.json"),
    )
    write_new(out, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "parent", "output"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    verify(a.root, a.parent, a.output)
