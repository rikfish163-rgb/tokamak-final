"""Choose a complete native-clock history scale using J-TEXT TRAIN only."""

import argparse
from pathlib import Path
import time

import numpy as np

import moving_peak_replay_20261105 as r
from run_deep_miss_history_20261106 import source_training_anchors
from verify_confirmation_raw_replay_20261105 import Hashes

BLOCKS = (16, 24, 32, 40, 48, 64)
COUNTS = (
    "positive",
    "safe",
    "outside",
    "timely_positive",
    "east_like_positive",
    "east_like_negative",
)
COLUMNS = [i for i in range(23) if i not in (1, 2, 10)]
PLAN = r.ROOT / "COMMON_SUPPORT_HISTORY_RISK_PLAN_20261106.md"
STATUS = "TRAIN_ONLY_COMPLETE_HISTORY_SUPPORT"
NO_GEOMETRY = "TRAIN_ONLY_NO_GEOMETRY_PASSES"


def independent_support(mask, anchors, block):
    """Count visible windows via a valid convolution, independent of builder."""
    if (
        mask.ndim != 2
        or mask.shape[1] != 23
        or mask.dtype != bool
        or anchors.ndim != 1
        or anchors.dtype.kind not in "iu"
        or (anchors < 127).any()
        or (anchors >= len(mask)).any()
        or block not in BLOCKS
    ):
        raise ValueError("declared native-grid history geometry required")
    starts = anchors[:, None] - 127 - block * np.arange(1, 5)
    hit = np.zeros(starts.shape, dtype=bool)
    visible = starts >= 0
    if visible.any():
        full = np.convolve(
            mask[:, COLUMNS].any(1).astype(np.int16),
            np.ones(block, dtype=np.int16),
            mode="valid",
        )
        hit[visible] = full[starts[visible]] > 0
    return hit


def masks_at(times, flag, down, horizon, *, dense=False):
    y, valid = r.io23.labels_at(times, flag, down, horizon)
    ix = (
        127 + np.flatnonzero(valid[127:])
        if dense
        else source_training_anchors(y, valid)
    )
    pos = y[ix] == 1
    if flag:
        lead = np.round((down - times[ix]) * 1000, 9)
        east = (lead > 30) & (lead <= 125)
        return ix, dict(
            positive=pos,
            safe=np.zeros_like(pos),
            outside=~pos,
            timely_positive=pos & (lead > 10),
            east_like_positive=east & pos,
            east_like_negative=east & ~pos,
        )
    none = np.zeros_like(pos)
    return ix, dict(
        positive=none,
        safe=~none,
        outside=none,
        timely_positive=none,
        east_like_positive=none,
        east_like_negative=none,
    )


def candidate_rows(times, mask, flag, down, horizon, *, dense=False, clock_only=False):
    ix, masks = masks_at(times, flag, down, horizon, dense=dense)
    ans = {}
    for block in BLOCKS:
        supported = (
            ix >= 127 + 4 * block
            if clock_only
            else independent_support(mask, ix, block).all(1)
        )
        ans[block] = {k: int((m & supported).sum()) for k, m in masks.items()}
    return ans


def geometry_checks(candidates, disrupted, safe):
    """Evaluate each of the four locked floors without short-circuiting."""
    if set(candidates) != set(BLOCKS) or disrupted <= 0 or safe <= 0:
        raise ValueError("complete support ledger and both source classes required")
    checks = {}
    for block in BLOCKS:
        c = candidates[block]
        if set(c) != set(COUNTS):
            raise ValueError("support ledger is missing a source stratum")
        checks[block] = dict(
            timely_positive=100 * c["timely_positive"] >= 80 * disrupted,
            outside=100 * c["outside"] >= 60 * disrupted,
            safe=100 * c["safe"] >= 60 * safe,
            east_like_positive=c["east_like_positive"] >= 15,
        )
    return checks


def choose_block(candidates, disrupted, safe):
    """Largest source-supported geometry, or None with no fit authorized."""
    checks = geometry_checks(candidates, disrupted, safe)
    passed = [b for b in BLOCKS if all(checks[b].values())]
    return max(passed) if passed else None


def aggregate(rows, key="by_block"):
    by_window = {
        b: {k: sum(q[key][str(b)][k] for q in rows) for k in COUNTS} for b in BLOCKS
    }
    by_shot = {
        b: {k: sum(q[key][str(b)][k] > 0 for q in rows) for k in COUNTS} for b in BLOCKS
    }
    return by_window, by_shot


def run(out):
    if out.exists():
        raise FileExistsError(out)
    if not r.io23.JTEXT_INDEX.is_file():
        raise ValueError("canonical source index is required")
    start, h = time.monotonic(), Hashes()
    inputs = {
        str(p.resolve()): h(p)
        for p in (
            PLAN,
            Path(__file__),
            Path(source_training_anchors.__code__.co_filename),
            r.io23.JTEXT_INDEX,
            r.io23.JTEXT_ADVANCED_TIME_PATH,
            Path(r.io23.__file__),
            Path(r.simple_data.__file__),
        )
    }
    roles, adv = r.simple_data.source_roles(), r.simple_data.advanced_times()
    rows = []
    for sid in roles["train"]:
        path = r.io23.path_for("J-TEXT", sid)
        digest = h(path)
        shot = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        horizon, _, _ = r.io23.effective_horizon_ms(sid, adv)
        inputs_at_shot = shot["times"], shot["available"], flag, down, horizon
        grids = {
            "by_block": {},
            "by_block_dense": {"dense": True},
            "by_block_dense_clock_only": {"dense": True, "clock_only": True},
        }
        support_counts = {
            label: {
                str(b): v for b, v in candidate_rows(*inputs_at_shot, **kwargs).items()
            }
            for label, kwargs in grids.items()
        }
        rows.append(
            dict(
                shot=sid,
                sha256=digest,
                disrupted=flag,
                samples=len(shot["times"]),
                **support_counts,
            )
        )
    counts, unique = aggregate(rows)
    d = sum(q["disrupted"] for q in rows)
    s = len(rows) - d
    block = choose_block(unique, d, s)
    dense = {}
    for key in ("by_block_dense", "by_block_dense_clock_only"):
        win, distinct = aggregate(rows, key)
        dense[key] = dict(
            window_counts=win,
            shot_counts=distinct,
            candidate_under_same_floors=choose_block(distinct, d, s),
        )
    h.check()
    r.write_new(
        out,
        dict(
            status=STATUS if block is not None else NO_GEOMETRY,
            block=block,
            first_index=127 + 4 * block if block is not None else None,
            disrupted=d,
            safe=s,
            window_counts=counts,
            shot_counts=unique,
            sampled_checks=geometry_checks(unique, d, s),
            dense_diagnostics_not_used_for_selection=dense,
            train=rows,
            input_sha256=inputs,
            held_outcomes_read=0,
            seconds=time.monotonic() - start,
        ),
    )
    print("TRAIN eligible shots by scale:", unique, "selected", block, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
