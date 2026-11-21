"""Audit causal action coverage on source TRAIN and the already released EAST pool."""

import argparse
from pathlib import Path
import time

import numpy as np

import moving_peak_replay_20261105 as r
from deep_miss_history_features_20261106 import USED_CHANNELS
from layer_interaction_replay_20261105 import ARCHIVE
from run_layer0_action_credit_history_20261106 import load_history
from trajectory_replay_20261105 import SOURCE, baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes, score
from within_shot_coherence_20261106 import GROUPS


def rolling_observed(obs, width, minimum=1):
    counts = np.r_[0, np.cumsum(obs, dtype=np.int64)]
    full = np.zeros(len(obs), dtype=bool)
    if width <= len(obs):
        full[width - 1 :] = counts[width:] - counts[:-width] >= minimum
    return full


def available_next(mask):
    """An update computed after sample i may affect sample i+1, not sample i."""
    if mask.ndim != 2 or mask.shape[1] != 23 or mask.dtype != bool or not len(mask):
        raise ValueError("nonempty native-grid diagnostic availability required")
    obs = mask[:, USED_CHANNELS].any(1)
    native = rolling_observed(obs, 128)
    padded = np.maximum.accumulate(obs)
    # Current availability for a partial native window still refers to its last
    # 128 observations, so a long missing tail is not implicitly kept alive.
    padded[127:] = native[127:]
    mature = dict(native_padded=padded, native128=native)
    for b, number in ((16, 1), (16, 4), (256, 4)):
        old = rolling_observed(obs, b)
        ix = np.arange(127 + number * b, len(obs), dtype=np.int64)
        supported = np.zeros(len(obs), dtype=bool)
        if len(ix):
            earlier = ix[:, None] - 128 - b * np.arange(number)[None, :]
            supported[ix] = native[ix] & old[earlier].all(1)
        mature[f"extra{number}x{b}"] = supported
    for b in (16, 32):
        recent = np.column_stack(
            [rolling_observed(mask[:, idx].any(1), b, b // 2) for _, idx in GROUPS]
        )
        covered = np.zeros(len(obs), dtype=bool)
        if len(obs) > b:
            covered[b:] = (recent[b:] & recent[:-b]).sum(1) >= 2
        mature[f"pair2x{b}"] = covered
    return {k: np.r_[False, v[:-1]] for k, v in mature.items()}


def risk_roles(times, flag, down, horizon, min_lead):
    if flag:
        lead = np.round((down - times) * 1000, 9)
        pos = (lead >= -1e-6) & (lead <= horizon + 1e-6)
        east = (lead > 30) & (lead <= 125)
        return dict(
            safe=np.zeros(len(times), dtype=bool),
            timely_positive=pos & (lead > min_lead),
            outside=lead > horizon + 1e-6,
            east_like_positive=east & pos,
            east_like_negative=east & ~pos,
            early_500=lead >= 500,
        )
    none = np.zeros(len(times), dtype=bool)
    return dict(
        safe=~none,
        timely_positive=none,
        outside=none,
        east_like_positive=none,
        east_like_negative=none,
        early_500=none,
    )


def observe(data, flag, down, horizon, lead, p=None):
    support = available_next(data["available"])
    windows = risk_roles(data["times"], flag, down, horizon, lead)
    native = None
    if p is not None:
        native = score(
            data["times"],
            p,
            dict(shot=data["shot"], disruptive=flag, down=down),
            r.THRESHOLD,
        )
        first = native["first_index"]
        windows["safe_high_until_first"] = (
            windows["safe"]
            & (p >= r.THRESHOLD - 0.15)
            & (np.arange(len(p)) <= (first if first is not None else len(p) - 1))
        )
        windows["original_first_alarm"] = np.arange(len(p)) == (
            first if first is not None else -1
        )
    return dict(
        points=len(data["times"]),
        has_role={k: bool(x.any()) for k, x in windows.items()},
        exposures={
            k: {g: int((m & x).sum()) for g, x in windows.items()}
            for k, m in support.items()
        },
        native=native,
    )


def aggregate(rows):
    if not rows:
        raise ValueError("nonempty verified shot pool required")
    return {
        geometry: {k: sum(q["exposures"][geometry][k] > 0 for q in rows) for k in roles}
        for geometry, roles in rows[0]["exposures"].items()
    }


def run(source_preflight, out):
    if out.exists():
        raise FileExistsError(out)
    start, h = time.monotonic(), Hashes()
    h(source_preflight)
    frozen = r.read(source_preflight)
    if frozen["held_outcomes_read"] != 0 or not all(
        h(p) == sha for p, sha in frozen["input_sha256"].items()
    ):
        raise ValueError("bound source-only census required")
    order, cfg, done = reference()
    src = cfg["source_state_sha256"]
    fit = ARCHIVE / "fit_moving_peak.npz"
    released, deps, _ = load_history(order, cfg["raw_sha256"], src, fit)
    map_path = r.COLD / "input_map.json"
    bound = {
        str(p.resolve()): h(p)
        for p in (
            Path(__file__),
            source_preflight,
            fit,
            map_path,
            SOURCE / "completed.json",
            SOURCE / "protocol.json",
            Path(load_history.__code__.co_filename),
            Path(baseline_for.__code__.co_filename),
            Path(r.io23.__file__),
            Path(r.simple_data.__file__),
            Path(r.prepare_actual.__code__.co_filename),
            *deps,
        )
    }
    if (
        bound[str(map_path.resolve())] != cfg["input_sha256"][str(map_path)]
        or [q["shot"] for q in frozen["train"]] != r.simple_data.source_roles()["train"]
    ):
        raise ValueError("the original source/target split and cold map are required")
    adv, mapping = r.simple_data.advanced_times(), r.read(map_path)
    train, east = [], []
    for q in frozen["train"]:
        sid = q["shot"]
        digest = h(r.io23.path_for("J-TEXT", sid))
        if digest != q["sha256"]:
            raise ValueError("source bytes changed after geometry census")
        a = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        horizon, _, _ = r.io23.effective_horizon_ms(sid, adv)
        if flag != q["disrupted"] or len(a["times"]) != q["samples"]:
            raise ValueError("source timeline or authority changed")
        train.append(
            dict(
                shot=sid,
                sha256=digest,
                disruptive=flag,
                **observe(a, flag, down, horizon, 10),
            )
        )
    for q in released:
        sid, rank = q["shot"], q["rank"]
        path = r.io23.path_for("EAST", sid)
        if h(path) != cfg["raw_sha256"][str(sid)] or rank > q["available_after_rank"]:
            raise ValueError("EAST history outside original ended release")
        a = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
        t, p = baseline_for(rank, sid, done)
        if not np.array_equal(a["times"], t):
            raise ValueError("EAST history clock does not match prediction seal")
        flag, down = r.io23.read_outcome("EAST", sid)
        if (flag, down) != (q["disruptive"], q["down"]):
            raise ValueError("released label differs from original HDF5")
        adjusted = {**a, **r.prepare_actual(a, rank, mapping)}
        east.append(
            dict(**q, sha256=h(path), **observe(adjusted, flag, down, 125, 30, p))
        )
    timeline = {}
    for rank in range(20, 361, 20):
        past = [x for x in east if x["available_after_rank"] <= rank]
        timeline[rank] = dict(
            n=len(past),
            disrupted=sum(x["disruptive"] for x in past),
            coverage=aggregate(past),
        )
    h.check()
    r.write_new(
        out,
        dict(
            status="CENSUS_PREALARM_SUPPORT_NO_MODEL_FIT",
            source=train,
            original_east_releases=east,
            source_counts=aggregate(train),
            east_by_available_after_rank=timeline,
            new_target_labels_read=0,
            test_source_outcomes_read=0,
            inputs_sha256=bound,
            seconds=time.monotonic() - start,
        ),
    )
    print("source TRAIN shot coverage", aggregate(train), flush=True)
    print("EAST original releases", timeline[360], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_preflight", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.source_preflight, a.out)
