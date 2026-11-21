"""Leave-one-safe-shot-out threshold calibration for frozen cold-start readouts.

All fits and threshold decisions use only the ended prefix. The held-out safe
shot is excluded from both the density gain and classifier fit in its fold.
On the saved 360-shot replay only the thresholds change, never the curves or
the initial 20 predictions. This is calibration, not intra-shot learning.
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .commissioning_transfer_20261108 import (
    ARMS, SOURCE_STRIDE, STARTUP, combine, features, fit_readouts, make_rows, predict,
)
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warmstart_alignment_20261022.alignment import SIGNAL_KEYS, fit_statistics, make_map


CAPS = (.10, .15, .20)
NAMES = ("native", *ARMS)


def held_safe_cutoffs(peaks):
    """Conservative finite-sample ranks; no future FPR guarantee under drift."""
    x = np.sort(np.asarray(peaks, np.float32))
    if x.shape != (12,) or not np.isfinite(x).all() or x[0] < 0 or x[-1] > 1:
        raise ValueError("12 finite held-safe shot maxima in [0, 1] required")
    return {
        str(cap): float(np.nextafter(x[math.ceil((1 - cap) * (len(x) + 1)) - 1],
                                    np.float32(np.inf)))
        for cap in CAPS
    }


def prefix_mapping(source_statistics, records):
    rows = [{key: r[key] for key in SIGNAL_KEYS}
            for r in map(apply_input_contract, records)]
    stats = fit_statistics(rows, device="EAST", allowed_ids=[r["shot"] for r in rows])
    return make_map(source_statistics, stats, "density_gain")


def calibrate(output, reference):
    started = time.monotonic()
    configure()
    if not (reference / "result.json").is_file():
        raise ValueError("completed commissioning transfer required")
    protocol, fit = (read(reference / f) for f in ("protocol.json", "fit.json"))
    inputs = load_inputs()
    ids = inputs["ids"]
    if (protocol["order"] != ids or protocol["source_sha256"] != PARENT_SHA256
            or fit["target_fit_ids"] != ids[:STARTUP]):
        raise ValueError("fixed full stream and ended20 budget required")
    base, ckpt = load_parent(DEFAULT_PARENT)
    base.to("cuda")
    original = float(ckpt["threshold"])
    if original != protocol["threshold"]:
        raise ValueError("reference threshold mismatch")
    output.mkdir(parents=True)

    # Input labels here are for previously sealed shots only.
    raw, outcomes = {}, {}
    for i, sid in enumerate(ids[:STARTUP], 1):
        if not (reference / "shots" / f"{i:04d}_{sid}.npz").is_file():
            raise ValueError("unsealed startup prediction")
        if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
            raise ValueError("startup raw signals changed")
        raw[sid] = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        outcomes[sid] = io23.read_outcome("EAST", sid)
    safe = [sid for sid in raw if not outcomes[sid][0]]
    if len(safe) != 12:
        raise ValueError("locked startup 8 disruptive/12 safe required")

    source_shots, train_ids = load_source_train()
    if train_ids != fit["source_train_ids"]:
        raise ValueError("same source TRAIN data required")
    source_rows = []
    for group, shot in enumerate(source_shots):
        available = shot["available"].copy()
        available[:, io23.RAW23.index("MNM")] = False
        x, _ = features(base, dict(shot, available=available), "cuda")
        source_rows.append(make_rows(x, shot["times"], shot["is_disrupt"], shot["down"],
                                     shot["horizon"], 10., group, stride=SOURCE_STRIDE))
    source = combine(source_rows)
    if len(source.x) != fit["source_rows"]:
        raise ValueError("source training row count changed")
    fits = []
    stats = inputs["mapping"]["source_statistics"]
    for held in safe:
        train = [sid for sid in raw if sid != held]
        mapping = prefix_mapping(stats, [raw[sid] for sid in train])
        target = []
        for group, sid in enumerate(train):
            sig = prepare_actual(raw[sid], STARTUP + 1, mapping)
            x, _ = features(base, sig, "cuda")
            target.append(make_rows(x, raw[sid]["times"], *outcomes[sid], 125., 30., group))
        heads, _ = fit_readouts(base, source, combine(target), original, "cuda")
        sig = prepare_actual(raw[held], STARTUP + 1, mapping)
        held_max = {arm: float(score.max()) for arm, score in predict(base, sig, heads, "cuda").items()}
        fits.append(dict(held_safe=held, fit_ids=train, maxima=held_max,
                         density_gain=mapping["gain"][io23.RAW23.index("ne_nG")]))
        print(dict(stage="held_safe", n=len(fits), seconds=time.monotonic() - started), flush=True)

    thresholds = {
        arm: held_safe_cutoffs([f["maxima"][arm] for f in fits]) for arm in NAMES
    }
    path = output / "thresholds.json"
    write(path, dict(original=original, thresholds=thresholds, folds=fits,
                     scope="PAST20_ONLY_OUT_OF_FOLD_THRESHOLDS_NOT_A_GUARANTEE",
                     readouts_sha256=sha(reference / "readouts.npz")))
    threshold_sha = sha(path)

    # Scoring is deliberately downstream of the threshold selection above.
    result = read(reference / "result.json")
    previous = read(reference / "per_shot.json")
    rows = {str(cap): {arm: [] for arm in NAMES} for cap in CAPS}
    for i, sid in enumerate(ids, 1):
        name = f"{i:04d}_{sid}.npz"
        trace_path = reference / "shots" / name
        if sha(trace_path) != result["curve_sha256"][name]:
            raise ValueError("reference curve differs from seal")
        flag, down = outcomes[sid] if i <= STARTUP else io23.read_outcome("EAST", sid)
        with np.load(trace_path, allow_pickle=False) as trace:
            for cap, arms in rows.items():
                for arm in arms:
                    cutoff = original if i <= STARTUP else thresholds[arm][cap]
                    measured = score_curve(trace["times"], trace[arm], flag, down, cutoff)
                    arms[arm].append(dict(shot=sid, disruptive=flag, down_s=down, **measured))
    if sha(path) != threshold_sha or any(
        rows[cap][arm][:STARTUP] != previous["native"][:STARTUP]
        for cap in rows for arm in NAMES
    ):
        raise AssertionError("unchanged thresholds and sealed startup required")
    metrics = {cap: {arm: timing_decomposition(rs) for arm, rs in arms.items()}
               for cap, arms in rows.items()}
    paired = {
        cap: {arm: flips(previous["native"], rs) for arm, rs in arms.items()}
        for cap, arms in rows.items()
    }
    write(output / "per_shot.json", rows)
    write(output / "result.json", dict(
        status="COMPLETED_COLD_START_CALIBRATION_DEVELOPMENT",
        threshold_sha256=threshold_sha, metrics=metrics, changes_vs_native_fixed=paired,
        seconds=time.monotonic() - started,
    ))
    print({cap: {arm: (r["tp"], r["fp"]) for arm, r in a.items()}
           for cap, a in metrics.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    calibrate(args.output.resolve(), args.reference.resolve())
