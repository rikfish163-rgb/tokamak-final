"""Read-back check of the four already-sealed confirmation arms."""

from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import io23
from joint_lora_hindsight_capacity_20261104 import V5_METADATA_SHA, read, sha, write_new
from verify_point_attribution_20261105 import pair
from verify_trajectory_replay_20261105 import FRESH, _recount, _score

ROOT = io23.ROOT / "probe_outputs"
RUN = ROOT / "point_confirmation_20261105_v1"
POINT = ROOT / "trajectory_point_attribution_20261105_v1"
DONOR = ROOT / "point_donor_controls_20261105_v1"
OUT = ROOT / "point_confirmation_verification_20261105_v1.json"


def verify():
    if OUT.exists():
        raise FileExistsError(OUT)
    done, cfg, result, saved = [
        read(RUN / k)
        for k in ("completed.json", "protocol.json", "summary.json", "per_shot.json")
    ]
    assert done["status"] == "COMPLETE_CONFIRMATION_DEVELOPMENT_DIAGNOSTIC"
    assert done["suffix_shots"] == 260 and done["guard_curves"] == 1040
    assert all(sha(RUN / f) == h for f, h in done["artifact_sha256"].items())
    assert all(sha(Path(f)) == h for f, h in cfg["input_sha256"].items())
    assert sha(FRESH) == V5_METADATA_SHA
    truth = {r["shot"]: r for r in read(FRESH)["EAST"]}
    old = read(POINT / "per_shot.json")
    order, tau, rules = cfg["order"], cfg["threshold"], cfg["rules"]
    assert (
        order
        == read(POINT / "protocol.json")["order"]
        == read(DONOR / "protocol.json")["order"]
    )
    assert set(order) == set(io23.cohort_ids("EAST"))
    recomputed = {k: list(old["static"][:100]) for k in saved}
    changed = set(
        result["paired_to_static"]["point_real"]["tp_gained"]
        + result["paired_to_static"]["point_real"]["fp_added"]
    )
    case = {}
    for rank, (sid, binding) in enumerate(
        zip(order[100:], done["sealed"], strict=True), 101
    ):
        name = f"{rank:04d}_{sid}"
        p = RUN / "shots" / f"{name}.npz"
        s, q = [p.with_name(name + ext) for ext in ("_seal.json", "_revealed.json")]
        seal, reveal = read(s), read(q)
        assert binding == dict(
            shot=sid, curve_sha256=sha(p), seal_sha256=sha(s), revealed_sha256=sha(q)
        )
        assert seal["curve_sha256"] == binding["curve_sha256"]
        assert (seal["shot"], seal["rank"], reveal["shot"]) == (sid, rank, sid)
        assert seal["current_outcome_used_in_gate"] is False
        assert reveal["after_seal_sha256"] == binding["seal_sha256"]
        assert p.stat().st_mtime_ns <= s.stat().st_mtime_ns <= q.stat().st_mtime_ns
        parent, donor = POINT / "shots" / p.name, DONOR / "shots" / p.name
        assert seal["source_curve_hashes"] == [sha(parent), sha(donor)]
        r = truth[sid]
        assert io23.read_outcome("EAST", sid) == (r["disruptive"], r["down"])
        with (
            np.load(p, allow_pickle=False) as z,
            np.load(parent, allow_pickle=False) as a,
            np.load(donor, allow_pickle=False) as b,
        ):
            t = z["times"]
            assert np.array_equal(t, a["times"]) and np.array_equal(t, b["times"])
            assert np.array_equal(
                t, r["start"] + np.arange(r["samples"], dtype=float) / r["rate"]
            )
            base, real = a["static"], a["point_real"]
            assert np.array_equal(base, b["static"]) and np.array_equal(
                real, b["point_real"]
            )
            extra = (base < tau) & (real >= tau)
            indices = np.flatnonzero(extra)
            if sid in changed:
                case[sid] = dict(
                    real_added_points=len(indices), real_first_time=float(t[indices[0]])
                )
            local = dict(static=base, point_real=real)
            for k, (shadow, age) in rules.items():
                v = a[shadow] if shadow in a.files else b[shadow]
                support = np.flatnonzero((base < tau) & (v >= tau))
                # Independently derive authorization by two sorted-index lookups;
                # a confirming point must be in [q-age,q], inclusive.
                upper = np.searchsorted(support, indices, side="right")
                lower = np.searchsorted(support, indices - age, side="left")
                eligible = indices[upper > lower]
                expected = base.copy()
                expected[eligible] = real[eligible]
                assert np.array_equal(z[k], expected)
                local[k] = z[k]
                if sid in case:
                    case[sid][k] = dict(
                        shadow_added_points=len(support),
                        authorized_points=len(eligible),
                        shadow_first_time=float(t[support[0]])
                        if len(support)
                        else None,
                        first_authorized_time=float(t[eligible[0]])
                        if len(eligible)
                        else None,
                    )
            for k, v in local.items():
                row = _score(t, v, r, tau)
                assert row == saved[k][rank - 1] == reveal["rows"][k]
                recomputed[k].append(row)
    assert saved == recomputed
    for k, section in (("full360", slice(None)), ("suffix260", slice(100, None))):
        assert {a: _recount(rows[section]) for a, rows in recomputed.items()} == result[
            k
        ]
    for compare, base in (
        ("paired_to_static", "static"),
        ("paired_to_real", "point_real"),
    ):
        assert {
            k: pair(v, recomputed[base]) for k, v in recomputed.items() if k != base
        } == result[compare]
    write_new(
        OUT,
        dict(
            status="PASS_RECOUNT_CAUSAL_SHADOW_CONFIRMATION",
            recomputed_guard_curves=1040,
            case_analysis=case,
            summary=result,
            input_producer_sha256=sha(RUN / "completed.json"),
            verifier_sha256=sha(Path(__file__)),
            independent_validation=False,
        ),
    )
    print("PASS", case, flush=True)


if __name__ == "__main__":
    verify()
