"""Raw-HDF5 recount of the locked-polarity intervention; no neural code imported."""

import argparse
import json
from pathlib import Path

import numpy as np

from verify_confirmation_raw_replay_20261105 import (
    Hashes,
    RAW_ROOT,
    counts,
    need,
    pair,
    raw_truth,
    read,
    score,
)
from verify_moving_peak_replay_20261105 import check_curve, parity


ARMS = [
    "static",
    "intra",
    "inter",
    "both",
    "inter_safe",
    "both_safe",
    "inter_disruptive",
    "both_disruptive",
]


def verify(run, parent, audit, output):
    need(not output.exists(), "output exists")
    h = Hashes()
    cfg, done, p_cfg, old, p_audit = (
        read(p)
        for p in (
            run / "protocol.json",
            run / "completed.json",
            parent / "protocol.json",
            parent / "completed.json",
            audit,
        )
    )
    order, tau = cfg["order"], cfg["threshold"]
    need(
        len(order) == len(set(order)) == 360
        and order == p_cfg["order"]
        and tau == p_cfg["threshold"] == 0.7258946895599365
        and cfg["raw_sha256"] == p_cfg["raw_sha256"]
        and cfg["source_state_sha256"] == p_cfg["source_state_sha256"]
        and cfg["arms"] == ARMS
        and cfg["label_budget"] == 36
        and cfg["additional_label_reads"] == cfg["training_steps"] == 0
        and cfg["development_only"] is True
        and cfg["local_producer_sha256"]
        == p_audit["producer_sha256"]
        == h(parent / "completed.json")
        and p_audit["status"] == "PASS_LAYER0_FACTORIAL_HDF5_RECOUNT",
        "wrong intervention source or protocol",
    )
    need(
        done["status"] == "COMPLETE_LAYER0_MEMORY_POLARITY"
        and done["n_shots"] == len(done["sealed"]) == 360
        and done["n_curves"] == 2880
        and done["protocol_sha256"] == h(run / "protocol.json")
        and all(h(p) == d for p, d in cfg["input_sha256"].items()),
        "incomplete or unbound intervention",
    )
    rows = {k: [] for k in ARMS}
    totals = {"safe": 0, "disruptive": 0}
    old_state = previous = None
    last_ns = -1
    max_err = dict.fromkeys(ARMS[:4], 0.0)
    cases = []
    for rank, sid in enumerate(order, 1):
        stem = f"{rank:04d}_{sid}"
        array = run / "shots" / (stem + ".npz")
        seal = run / "shots" / (stem + "_seal.json")
        note = read(seal)
        parent_note = old["sealed"][rank - 1]
        ref = parent / "shots" / (stem + ".npz")
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        truth, times = raw_truth(raw, sid)
        need(
            done["sealed"][rank - 1] == dict(note, seal_sha256=h(seal))
            and (note["rank"], note["shot"]) == (rank, sid)
            and h(raw) == note["raw_sha256"] == cfg["raw_sha256"][str(sid)]
            and h(array) == note["curve_sha256"]
            and h(ref) == parent_note["curve_sha256"]
            and note["previous_seal_sha256"] == previous
            and note["seal_ns"] >= last_ns
            and note["state_sha256"] == parent_note["state_sha256"] == old_state
            and (note["n_safe"], note["n_disruptive"])
            == (totals["safe"], totals["disruptive"]),
            "wrong active origin state or curve seal",
        )
        previous, last_ns = h(seal), note["seal_ns"]
        with (
            np.load(array, allow_pickle=False) as f,
            np.load(ref, allow_pickle=False) as a,
        ):
            need(
                np.array_equal(f["times"], times)
                and np.array_equal(a["times"], times)
                and set(f.files) == {"times", *ARMS},
                "wrong raw clock or arms",
            )
            scored = {}
            for k in ARMS:
                v = f[k]
                check_curve(v, times, k)
                need(
                    np.array_equal(v[:135], a["static"][:135]),
                    "modified prefix before first write",
                )
                if k in max_err:
                    max_err[k] = max(max_err[k], parity(v, a[k], tau, f"{sid}/{k}"))
                scored[k] = score(times, v, truth, tau)
                rows[k].append(scored[k])
            before, after = scored["intra"], scored["both"]
            if before["status"] != after["status"]:
                i = (
                    before["first_index"]
                    if before["first_index"] is not None
                    else after["first_index"]
                )
                cases.append(
                    dict(
                        shot=sid,
                        rank=rank,
                        disruptive=truth["disruptive"],
                        status={k: scored[k]["status"] for k in ARMS},
                        lead={k: scored[k]["lead_ms"] for k in ARMS},
                        comparison_index=i,
                        scores_at_index={k: float(f[k][i]) for k in ARMS},
                    )
                )
        if rank % 20 == 0:
            block = rank // 20
            receipt = parent / "blocks" / f"{block:02d}_update.json"
            revealed = parent / "blocks" / f"{block:02d}_reveals.json"
            bound = old["blocks"][block - 1]
            need(
                h(receipt) == bound["update_sha256"]
                and h(revealed) == bound["reveals_sha256"],
                "origin receipt changed",
            )
            update = read(receipt)
            need(
                update["state_before_sha256"] == old_state,
                "parent release state changed",
            )
            old_state = update["state_after_sha256"]
            if update["decision"] == "accepted":
                [outcome] = read(revealed)["outcomes"]
                totals["disruptive" if outcome["disruptive"] else "safe"] += 1
    need(
        max_err == done["max_ref_error"] and sum(totals.values()) == old["n_accepted"],
        "terminal parity",
    )
    report_metrics = {k: counts(v) for k, v in rows.items()}
    need(
        {k: report_metrics[k] for k in ARMS[:4]}
        == p_audit["sections"]["full360"]["metrics"],
        "parent shot counts do not reproduce",
    )
    lost = [
        v for v in cases if v["status"]["intra"] == "tp" and v["status"]["both"] != "tp"
    ]
    fixed = [
        v for v in cases if v["status"]["intra"] == "fp" and v["status"]["both"] == "tn"
    ]
    h.check()
    result = dict(
        status="PASS_LAYER0_POLARITY_HDF5_RECOUNT",
        n_shots=360,
        n_curves=2880,
        original_origins=totals,
        full360=report_metrics,
        ref_curve_max=max_err,
        changed_cases=cases,
        lost_timely_when_full=dict(
            n=len(lost),
            also_lost_safe_only=sum(v["status"]["both_safe"] != "tp" for v in lost),
            recovered_disruptive_only=sum(
                v["status"]["both_disruptive"] == "tp" for v in lost
            ),
            at_original_alarm_safe_lowers=sum(
                v["scores_at_index"]["both_safe"] < v["scores_at_index"]["intra"]
                for v in lost
            ),
        ),
        removed_fp_when_full=dict(
            n=len(fixed),
            also_removed_safe_only=sum(v["status"]["both_safe"] == "tn" for v in fixed),
            also_removed_disruptive_only=sum(
                v["status"]["both_disruptive"] == "tn" for v in fixed
            ),
        ),
        strict_vs_intra={
            k: pair(rows[k], rows["intra"]) for k in ARMS if k.startswith("both")
        },
        capped_vs_intra={
            k: pair(rows[k], rows["intra"], True) for k in ARMS if k.startswith("both")
        },
        producer_sha256=h(run / "completed.json"),
        verifier_sha256=h(__file__),
        scope="Fixed previously accepted path, not a prospective origin-only policy or independent validation.",
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(result["status"], report_metrics, result["lost_timely_when_full"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "parent", "audit", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    verify(a.run, a.parent, a.audit, a.output)
