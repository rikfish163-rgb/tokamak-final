"""Independently recount sealed layer intervention curves against HDF5."""

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


ARCHIVE = (
    Path(__file__).resolve().parent / "probe_outputs/moving_peak_replay360_20261105_v1"
)
ARCHIVE_SHA = "ebc5ec31ba18305e40a906f6cad9f5b4cd05f9d1c59c1c0e7c61cb2267a4205b"
MASK = [
    "static",
    "moving_peak",
    "layer0",
    "layer1",
    "gain",
    "bias",
    "layer0_norm",
    "layer1_norm",
]
FIT = [
    "static",
    "interaction_control",
    "interaction_penalty",
    "interaction_control_stale",
    "interaction_penalty_stale",
]


def check_fit(run, order, fit_hashes, old, h):
    fit = read(run / "fit_receipt.json")
    fixed_path = ARCHIVE / "fit_receipt.json"
    fixed = read(fixed_path)
    need(
        fit["ids"] == fixed["ids"] == order[:20]
        and fit["initial_fit_sha256"] == fixed["initial_fit_sha256"]
        and fit["fixed_receipt_sha256"]
        == h(fixed_path)
        == old["artifact_sha256"]["fit_receipt.json"],
        "fitting history or initialization changed",
    )
    expected = [
        {"round": row["round"], "selection": row["selection"]}
        for row in fixed["arms"]["moving_peak"]["rounds"]
    ]
    need(
        fit["selections"] == expected
        and len(expected) == 3
        and fit["weighting"]["examples_per_round"] == 160,
        "different fixed safe/positive windows",
    )
    need(set(fit["arms"]) == set(FIT[1:3]), "different matched arms")
    for k, coeff in (("interaction_control", 0), ("interaction_penalty", 1)):
        entry = fit["arms"][k]
        need(
            entry["fit_sha256"] == fit_hashes[k]
            and entry["steps"] == 36
            and entry["coefficient"] == coeff
            and [rd["round"] for rd in entry["rounds"]] == [1, 2, 3]
            and [rd["steps"] for rd in entry["rounds"]] == [12] * 3,
            "wrong coefficients, optimizer budget, or fit digest",
        )
        for rd in entry["rounds"]:
            for v in (rd["initial"], rd["final"]):
                need(
                    np.isfinite(list(v.values())).all()
                    and min(v.values()) >= 0
                    and abs(
                        v["total"] - v["matched"] - coeff * v["interaction_penalty"]
                    )
                    <= 1e-6 * (1 + v["total"]),
                    "inconsistent fitted objective",
                )
    with (
        np.load(run / "fit_interaction_control.npz", allow_pickle=False) as a,
        np.load(ARCHIVE / "fit_moving_peak.npz", allow_pickle=False) as b,
    ):
        need(set(a.files) == set(b.files), "different frozen generator layout")
        maximum = max(float(np.max(np.abs(a[k] - b[k]))) for k in b.files)
        need(maximum <= 4e-6, "zero-coefficient fit no longer matches archive")
    return maximum


def verify(run, output):
    need(not output.exists(), "verification target already exists")
    h = Hashes()
    completion = run / "completed.json"
    cfg, done, saved, summary = [
        read(run / name)
        for name in ("protocol.json", "completed.json", "per_shot.json", "summary.json")
    ]
    need(h(ARCHIVE / "completed.json") == ARCHIVE_SHA, "wrong archive")
    old = read(ARCHIVE / "completed.json")
    need(
        h(ARCHIVE / "protocol.json") == old["artifact_sha256"]["protocol.json"],
        "wrong archive protocol",
    )
    parent = read(ARCHIVE / "protocol.json")
    mode = cfg["mode"]
    need(mode in ("mask", "fit"), "unexpected mode")
    arms = MASK if mode == "mask" else FIT
    ref_arm = "moving_peak" if mode == "mask" else "interaction_control"
    order, tau = cfg["order"], cfg["threshold"]
    need(
        len(order) == len(set(order)) == 360
        and order == parent["order"]
        and tau == parent["threshold"] == 0.7258946895599365
        and cfg["raw_sha256"] == parent["raw_sha256"]
        and cfg["source_state_sha256"] == parent["source_state_sha256"]
        and cfg["archive_sha256"] == ARCHIVE_SHA
        and cfg["arms"] == arms
        and cfg["label_budget"] == 20
        and cfg["first_anchor"] == 135
        and cfg["stride"] == 32
        and cfg["radius"] == 2.0
        and cfg["development_only"] is True,
        "changed study contract",
    )
    files = ["protocol.json", "per_shot.json", "summary.json"]
    if mode == "fit":
        files += ["fit_receipt.json", *(f"fit_{k}.npz" for k in arms[1:3])]
        fit_hashes = {k: h(run / f"fit_{k}.npz") for k in arms[1:3]}
    else:
        fit_hashes = {"moving_peak": h(ARCHIVE / "fit_moving_peak.npz")}
        need(
            fit_hashes["moving_peak"] == old["artifact_sha256"]["fit_moving_peak.npz"],
            "changed archived fit",
        )
    need(
        done["status"] == "COMPLETE_LAYER_INTERACTION_REPLAY"
        and done["mode"] == mode
        and done["n_shots"] == len(done["sealed"]) == 360
        and done["n_curves"] == 360 * len(arms)
        and done["fit_sha256"] == fit_hashes
        and done["artifact_sha256"] == {name: h(run / name) for name in files}
        and all(h(p) == digest for p, digest in cfg["input_sha256"].items()),
        "incomplete or unbound run",
    )
    fit_diff = check_fit(run, order, fit_hashes, old, h) if mode == "fit" else None
    out = {k: [] for k in arms}
    diffs = dict.fromkeys(("static", ref_arm), 0.0)
    prior_reveal = None
    for rank, sid in enumerate(order, 1):
        stem = f"{rank:04d}_{sid}"
        curve, seal, reveal = [
            run / "shots" / (stem + s) for s in (".npz", "_seal.json", "_revealed.json")
        ]
        binding = dict(
            shot=sid,
            curve_sha256=h(curve),
            seal_sha256=h(seal),
            revealed_sha256=h(reveal),
        )
        note, label = read(seal), read(reveal)
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        truth, time = raw_truth(raw, sid)
        need(
            binding == done["sealed"][rank - 1]
            and (note["rank"], note["shot"], label["shot"]) == (rank, sid, sid)
            and note["curve_sha256"] == binding["curve_sha256"]
            and label["after_seal_sha256"] == binding["seal_sha256"]
            and note["raw_sha256"] == cfg["raw_sha256"][str(sid)] == h(raw)
            and note["current_outcome_used_in_prediction"] is False
            and note["active_fit_sha256"] == ({} if rank <= 20 else fit_hashes)
            and curve.stat().st_mtime_ns
            <= seal.stat().st_mtime_ns
            <= reveal.stat().st_mtime_ns
            <= completion.stat().st_mtime_ns
            and (
                prior_reveal is None
                or prior_reveal.stat().st_mtime_ns <= curve.stat().st_mtime_ns
            ),
            f"seal, HDF hash, or reveal boundary: {sid}",
        )
        if mode == "fit" and rank == 20:
            next_curve = run / "shots" / f"0021_{order[20]}.npz"
            for name in files[3:]:
                need(
                    reveal.stat().st_mtime_ns
                    <= (run / name).stat().st_mtime_ns
                    <= next_curve.stat().st_mtime_ns,
                    "fit must follow reveal20",
                )
        prior_reveal = reveal
        expected_updates = set(arms[1:]) if rank > 20 else set()
        need(set(note["updates"]) == expected_updates, f"update arm scope: {sid}")
        n = len(range(135, len(time), 32))
        for stat in note["updates"].values():
            need(
                0 <= stat["nonzero"] <= n and 0 <= stat["max_norm"] <= 2.000001,
                f"update magnitude/count: {sid}",
            )
        previous = ARCHIVE / "shots" / (stem + ".npz")
        need(
            h(previous) == old["sealed"][rank - 1]["curve_sha256"], "archive curve hash"
        )
        with (
            np.load(curve, allow_pickle=False) as f,
            np.load(previous, allow_pickle=False) as p,
        ):
            need(
                set(f.files) == {"times", *arms}
                and np.array_equal(f["times"], time)
                and np.array_equal(p["times"], time),
                f"clock or probability schema: {sid}",
            )
            for name, ref in (("static", "static"), (ref_arm, "moving_peak")):
                diffs[name] = max(
                    diffs[name], parity(f[name], p[ref], tau, f"{sid}/{name}")
                )
            for k in arms:
                v = f[k]
                check_curve(v, time, f"{sid}/{k}")
                stop = len(time) if rank <= 20 else 135
                need(
                    np.array_equal(v[:stop], f["static"][:stop]),
                    "pre-update prefix differs",
                )
                row = score(time, v, truth, tau)
                need(
                    row == label["rows"][k] == saved[k][rank - 1],
                    f"outcome mismatch: {sid}/{k}",
                )
                out[k].append(row)
    need(diffs == done["parity_max_error"], "parity summary mismatch")
    report = {}
    for name, subset in (("full360", slice(None)), ("after20", slice(20, None))):
        rows = {k: out[k][subset] for k in arms}
        values = {k: counts(v) for k, v in rows.items()}
        need(values == summary[name]["metrics"], f"count mismatch: {name}")
        pairs = {k: pair(v, rows["static"]) for k, v in rows.items() if k != "static"}
        capped = {
            k: pair(v, rows["static"], capped=True)
            for k, v in rows.items()
            if k != "static"
        }
        need(
            pairs == summary[name]["pairs"] and capped == summary[name]["pairs_capped"],
            f"paired transition mismatch: {name}",
        )
        report[name] = dict(metrics=values, pairs=pairs, pairs_capped=capped)
    h.check()
    result = dict(
        status="PASS_LAYER_INTERACTION_HDF5_RECOUNT",
        mode=mode,
        n_shots=360,
        n_curves=360 * len(arms),
        producer_sha256=h(completion),
        verifier_sha256=h(__file__),
        parity=diffs,
        control_max_fit_error=fit_diff,
        summary=report,
        scope="Independent curve/HDF/receipt recount, not independent training or future validation.",
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(result["status"], mode, result["n_curves"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    verify(a.run, a.output)
