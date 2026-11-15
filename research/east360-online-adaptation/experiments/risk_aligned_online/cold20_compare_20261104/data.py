"""Same ended-prefix supervision and fixed draw schedule for all controls."""

import math

import numpy as np

from .. import io23


def build_pools(shots, revealed, allowed_ids):
    ids = list(allowed_ids)
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(type(s) is not int for s in ids)
        or set(shots) != set(ids)
        or set(revealed) != set(ids)
    ):
        raise ValueError("only the same fully revealed prefix may enter training")
    records = []
    pools = {group: {} for group in ("positive", "safe", "earlier")}
    for sid in ids:
        shot = shots[sid]
        flag, down = revealed[sid]
        if (
            type(flag) is not bool
            or (flag and (down is None or not math.isfinite(down)))
            or (not flag and down is not None)
        ):
            raise ValueError(
                "HDF truth must be boolean and carry only real disruption time"
            )
        times, values, a = (
            np.asarray(shot["times"]),
            np.asarray(shot["values"]),
            np.asarray(shot["available"]),
        )
        if (
            shot["shot"] != sid
            or shot["device"] != "EAST"
            or times.ndim != 1
            or not len(times)
            or values.shape != (len(times), 23)
            or values.dtype != np.float32
            or a.shape != values.shape
            or a.dtype != bool
            or not np.isfinite(values[a]).all()
            or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()
        ):
            raise ValueError("matching finite EAST prefix signal and clock required")
        label, valid = io23.labels_at(times, flag, down, 125.0)
        for query in np.flatnonzero(valid):
            group = "safe" if not flag else ("positive" if label[query] else "earlier")
            index = len(records)
            records.append(
                dict(shot=sid, query=int(query), label=int(label[query]), group=group)
            )
            pools[group].setdefault(sid, []).append(index)
    if any(not per_shot for per_shot in pools.values()):
        raise ValueError("cold prefix lacks a required training stratum")
    return records, pools


def make_schedule(pools, *, steps=256, batch_size=128, seed=131):
    if (
        set(pools) != {"positive", "safe", "earlier"}
        or any(not v for v in pools.values())
        or type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or batch_size < 4
        or batch_size % 4
        or type(seed) is not int
    ):
        raise ValueError(
            "three nonempty strata and fixed positive balanced budget required"
        )
    rng = np.random.default_rng(seed)
    result = []
    for step in range(1, steps + 1):
        indices = []
        for group, count in (
            ("positive", batch_size // 2),
            ("safe", batch_size // 4),
            ("earlier", batch_size // 4),
        ):
            ids = sorted(pools[group])
            selected = rng.choice(ids, size=count, replace=True)
            indices.extend(int(rng.choice(pools[group][int(sid)])) for sid in selected)
        rng.shuffle(indices)
        result.append(dict(step=step, indices=indices))
    return result
