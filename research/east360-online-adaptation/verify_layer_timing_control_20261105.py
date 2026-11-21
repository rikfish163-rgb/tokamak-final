"""Read original HDF5 and independently score the timing-control replay."""

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


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "probe_outputs/layer_interaction_masks360_20261105_v1"
VERIFY = ROOT / "probe_outputs/layer_interaction_masks360_verification_20261105_v1.json"
FIRST_RUN = ROOT / "probe_outputs/layer_timing_control360_20261105_v1"
FIRST_RUN_SHA = "d194a9b2009c530c0340dd2e78558c9564433fe67c25c97bee399a344c70681e"
SITES = ("layer0", "layer1")
CONTROLS = ("stale", "initial", "donor")
ARMS = [
    "static",
    *(k for site in SITES for k in (site, *(f"{site}_{c}" for c in CONTROLS))),
]


def verify(run, output):
    need(not output.exists(), "output exists")
    h = Hashes()
    files = ("protocol.json", "per_shot.json", "summary.json", "completed.json")
    protocol, rows, summary, done = [read(run / k) for k in files]
    prior_cfg, prior_done = (
        read(PARENT / k) for k in ("protocol.json", "completed.json")
    )
    check = read(VERIFY)
    order, tau = protocol["order"], protocol["threshold"]
    clock = protocol["clock_protocol"]
    need(
        clock["name"] == "MONOTONIC_SEAL_OUTCOME_REVEAL_CHAIN_V1"
        and type(clock["initial_ns"]) is int
        and h(clock["recorder"]) == clock["sha256"],
        "missing bound monotonic recorder",
    )
    need(
        h(FIRST_RUN / "completed.json") == FIRST_RUN_SHA,
        "original timing replay changed",
    )
    v1 = read(FIRST_RUN / "completed.json")
    need(
        check["status"] == "PASS_LAYER_INTERACTION_HDF5_RECOUNT"
        and h(PARENT / "completed.json")
        == check["producer_sha256"]
        == protocol["parent_sha256"]
        and h(PARENT / "protocol.json")
        == prior_done["artifact_sha256"]["protocol.json"]
        and len(order) == len(set(order)) == 360
        and order == prior_cfg["order"]
        and tau == prior_cfg["threshold"] == 0.7258946895599365
        and protocol["raw_sha256"] == prior_cfg["raw_sha256"]
        and protocol["fit_sha256"] == prior_done["fit_sha256"]["moving_peak"]
        and protocol["source_state_sha256"] == prior_cfg["source_state_sha256"]
        and protocol["arms"] == ARMS
        and protocol["first"] == 135
        and protocol["stride"] == 32
        and protocol["label_budget"] == 20
        and protocol["training_steps"] == 0
        and protocol["development_only"] is True,
        "wrong parent, stream or comparison protocol",
    )
    need(
        done["status"] == "COMPLETE_LAYER_TIMING_CONTROL"
        and done["n_shots"] == len(done["sealed"]) == 360
        and done["n_curves"] == len(ARMS) * 360
        and done["artifact_sha256"] == {k: h(run / k) for k in files[:-1]}
        and all(h(p) == d for p, d in protocol["input_sha256"].items()),
        "incomplete or unbound run",
    )
    observed = {k: [] for k in ARMS}
    common = {k: [] for k in (*SITES, *(f"{k}_donor" for k in SITES))}
    prior_sha = None
    last_ns = clock["initial_ns"]
    wall_regressions, v1_wall_regressions = [], []
    last_wall = last_v1_wall = None
    previous_anchors = tail_sum = 0
    diffs = dict.fromkeys(("static", *SITES), 0.0)
    for rank, sid in enumerate(order, 1):
        stem = f"{rank:04d}_{sid}"
        curve, seal, reveal = (
            run / "shots" / (stem + end)
            for end in (".npz", "_seal.json", "_revealed.json")
        )
        binding = dict(
            shot=sid,
            curve_sha256=h(curve),
            seal_sha256=h(seal),
            revealed_sha256=h(reveal),
        )
        note, revealed = read(seal), read(reveal)
        stamped, result_stamp = note["clock"], revealed["clock"]
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        truth, time = raw_truth(raw, sid)
        anchors = len(range(135, len(time), 32))
        tail = max(0, anchors - previous_anchors) if rank > 20 else 0
        common_stop = (
            min(len(time), 135 + 32 * previous_anchors) if rank > 20 else len(time)
        )
        need(
            binding == done["sealed"][rank - 1]
            and (note["shot"], note["rank"], revealed["shot"]) == (sid, rank, sid)
            and note["curve_sha256"] == binding["curve_sha256"]
            and note["raw_sha256"] == protocol["raw_sha256"][str(sid)] == h(raw)
            and revealed["after_seal_sha256"] == binding["seal_sha256"]
            and note["current_outcome_used_in_prediction"] is False
            and note["active_fit_sha256"]
            == (protocol["fit_sha256"] if rank > 20 else None)
            and note["donor_id"] == (order[rank - 2] if rank > 20 else None)
            and note["donor_tail_uses"] == tail,
            f"invalid source/reveal/donor evidence: {sid}",
        )
        need(
            (
                stamped["ordinal"],
                result_stamp["outcome_ordinal"],
                result_stamp["ordinal"],
            )
            == (3 * rank - 2, 3 * rank - 1, 3 * rank)
            and stamped["previous_reveal_sha256"] == prior_sha
            and last_ns
            <= stamped["monotonic_ns"]
            <= result_stamp["outcome_read_start_ns"]
            <= result_stamp["outcome_read_end_ns"]
            <= result_stamp["monotonic_ns"],
            f"invalid monotonic process chronology: {sid}",
        )
        first_curve = FIRST_RUN / "shots" / (stem + ".npz")
        first_reveal = FIRST_RUN / "shots" / (stem + "_revealed.json")
        need(
            binding["curve_sha256"]
            == h(first_curve)
            == v1["sealed"][rank - 1]["curve_sha256"]
            and h(first_reveal) == v1["sealed"][rank - 1]["revealed_sha256"],
            f"clocked repeat changes the numerical replay: {sid}",
        )
        if last_wall is not None and last_wall > curve.stat().st_mtime_ns:
            wall_regressions.append(sid)
        if last_v1_wall is not None and last_v1_wall > first_curve.stat().st_mtime_ns:
            v1_wall_regressions.append(sid)
        last_wall, last_v1_wall = (
            reveal.stat().st_mtime_ns,
            first_reveal.stat().st_mtime_ns,
        )
        prior_sha, last_ns = binding["revealed_sha256"], result_stamp["monotonic_ns"]
        previous_anchors = anchors
        tail_sum += tail
        stats = note["updates"]
        need(set(stats) == (set(ARMS[1:]) if rank > 20 else set()), "update scope")
        if rank > 20:
            for site in SITES:
                for control in CONTROLS:
                    a, b = stats[site], stats[f"{site}_{control}"]
                    need(
                        0 <= a["nonzero"] == b["nonzero"] <= anchors
                        and 0 <= a["max_norm"] <= 2.000001
                        and abs(a["max_norm"] - b["max_norm"]) <= 1e-6,
                        "invalid aggregate norm receipt",
                    )
        parent_curve = PARENT / "shots" / (stem + ".npz")
        need(
            h(parent_curve) == prior_done["sealed"][rank - 1]["curve_sha256"],
            "changed parent curve",
        )
        with (
            np.load(curve, allow_pickle=False) as f,
            np.load(parent_curve, allow_pickle=False) as p,
        ):
            need(
                set(f.files) == {"times", *ARMS}
                and np.array_equal(f["times"], time)
                and np.array_equal(p["times"], time),
                "clock or arms changed",
            )
            for k in diffs:
                diffs[k] = max(diffs[k], parity(f[k], p[k], tau, f"{sid}/{k}"))
            for k in ARMS:
                v = f[k]
                check_curve(v, time, f"{sid}/{k}")
                stop = len(time) if rank <= 20 else 135
                need(
                    np.array_equal(v[:stop], f["static"][:stop]),
                    "pre-update prefix differs",
                )
                row = score(time, v, truth, tau)
                need(
                    row == revealed["rows"][k] == rows[k][rank - 1],
                    f"HDF re-score mismatch: {sid}/{k}",
                )
                observed[k].append(row)
                if k in common:
                    common[k].append(
                        score(time[:common_stop], v[:common_stop], truth, tau)
                    )
    final_stamp = done["clock_chain"]
    need(
        final_stamp["last_reveal_sha256"] == prior_sha
        and final_stamp["events"] == 3 * len(order)
        and final_stamp["monotonic_ns"] >= last_ns,
        "wrong terminal chain",
    )
    need(
        diffs == done["parity"] and tail_sum == done["donor_tail_uses"],
        "run summary parity",
    )
    for part, subset in (("full360", slice(None)), ("after20", slice(20, None))):
        scored = {k: observed[k][subset] for k in ARMS}
        need(
            summary[part]["metrics"] == {k: counts(v) for k, v in scored.items()},
            "wrong shot counts",
        )
        for capped in (False, True):
            for k in ARMS[1:]:
                kind = "pairs_capped" if capped else "pairs"
                need(
                    summary[part][kind][k] == pair(scored[k], scored["static"], capped),
                    f"wrong paired comparison to static: {k}",
                )
            for site in SITES:
                for control in CONTROLS:
                    expected = pair(scored[site], scored[f"{site}_{control}"], capped)
                    need(
                        summary[part]["real_vs_controls"][site][control][
                            str(capped).lower()
                        ]
                        == expected,
                        "wrong equal-amplitude transition",
                    )
    h.check()
    result = dict(
        status="PASS_LAYER_TIMING_HDF5_RECOUNT",
        n_shots=360,
        n_curves=360 * len(ARMS),
        producer_sha256=h(run / "completed.json"),
        verifier_sha256=h(__file__),
        parity=diffs,
        donor_tail_uses=tail_sum,
        monotonic_events=final_stamp["events"],
        wall_regressions=wall_regressions,
        original_wall_regressions=v1_wall_regressions,
        bitwise_curve_parity_vs_original=True,
        donor_common_exposure={
            site: dict(
                real=counts(common[site]),
                donor=counts(common[site + "_donor"]),
                strict_pair=pair(common[site], common[site + "_donor"]),
                capped_pair=pair(common[site], common[site + "_donor"], True),
            )
            for site in SITES
        },
        summary=summary,
        scope="Original-HDF recount, archive parity, and aggregate timing receipts; not a model retrain or future validation.",
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(result["status"], result["n_curves"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    verify(a.run, a.output)
