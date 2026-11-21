"""Independently count source risk labels at EAST-like lead times (TRAIN only)."""

import argparse
from pathlib import Path
import time

import h5py
import numpy as np

from analyze_deep_miss_history_20261106 import independent_support
import moving_peak_replay_20261105 as r
from verify_confirmation_raw_replay_20261105 import Hashes


def training_support(t, mask, flag, down, window):
    i = np.arange(len(t))
    if flag:
        lead = (down - t) * 1000
        valid = lead >= -1e-6
        y = valid & (lead <= window + 1e-6)
        ms = np.round(lead, 9)
        east = (ms > 30) & (ms <= 125 + 1e-6)
        source_timely = ms > 10
    else:
        valid = np.ones(len(t), dtype=bool)
        y = east = source_timely = np.zeros(len(t), dtype=bool)
    positives = i[y & (i >= 127)]
    regular = (i >= 127) & ((i - 127) % 64 == 0) & valid
    if len(positives):
        regular[positives[::4]] = True
        regular[positives[-1]] = True
    ix = i[regular]
    sr, ss, _ = independent_support(mask, ix)
    common = sr.all(1) & ss.all(1)
    groups = np.where(y[ix], 0, 2 if flag else 1)
    available = dict(zip(ix.tolist(), common.tolist(), strict=True))
    joint = np.fromiter((available.get(k, False) for k in i), dtype=bool)
    return dict(
        counts=[int((groups == k).sum()) for k in range(3)],
        common=[int(((groups == k) & common).sum()) for k in range(3)],
        all_east_like_y1=int((east & y).sum()),
        all_east_like_y0=int((east & ~y).sum()),
        sampled_east_like_y1=int((east & y & regular).sum()),
        sampled_east_like_y0=int((east & ~y & regular).sum()),
        common_east_like_y1=int((east & y & joint).sum()),
        common_east_like_y0=int((east & ~y & joint).sum()),
        common_source_timely_y1=int((source_timely & y & joint).sum()),
    )


def metadata(path):
    with h5py.File(path, "r") as f:
        is_disrupt = np.asarray(f["meta/IsDisrupt"][()]).reshape(-1)
        down = np.asarray(f["meta/DownTime"][()]).reshape(-1)
    if len(is_disrupt) != 1 or is_disrupt[0] not in (0, 1):
        raise ValueError("invalid source HDF5 outcome")
    if is_disrupt[0] and (len(down) != 1 or not np.isfinite(down[0])):
        raise ValueError("invalid source HDF5 disruption time")
    return bool(is_disrupt[0]), float(down[0]) if is_disrupt[0] else None


def run(out):
    if out.exists():
        raise FileExistsError(out)
    start, h = time.monotonic(), Hashes()
    roles = r.simple_data.source_roles()
    adv = r.simple_data.advanced_times()
    rows = []
    for sid in roles["train"]:
        path = r.io23.path_for("J-TEXT", sid)
        before = h(path)
        a = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = metadata(path)
        horizon = max(25.0, adv[sid]) if sid in adv else 25.0
        q = training_support(a["times"], a["available"], flag, down, horizon)
        if h(path) != before:
            raise ValueError("source raw changed during independent audit")
        rows.append(dict(shot=sid, sha256=before, disrupted=flag, horizon=horizon, **q))
    metrics = [k for k in rows[0] if k.startswith(("all_", "sampled_", "common_"))]
    total = {
        k: dict(samples=sum(r[k] for r in rows), shots=sum(r[k] > 0 for r in rows))
        for k in metrics
    }
    h.check()
    r.write_new(
        out,
        dict(
            status="AUDIT_CANONICAL_SOURCE_HORIZON_SUPPORT",
            source_train=len(rows),
            disrupted=sum(x["disrupted"] for x in rows),
            disrupted_with_horizon_over_30=sum(
                x["disrupted"] and x["horizon"] > 30 for x in rows
            ),
            groups=total,
            rows=rows,
            held_outcomes_read=0,
            code_sha256=h(Path(__file__)),
            seconds=time.monotonic() - start,
        ),
    )
    print(total, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
