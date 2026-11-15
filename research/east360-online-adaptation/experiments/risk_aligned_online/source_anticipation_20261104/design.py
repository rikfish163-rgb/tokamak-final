"""Matched source-only teaching schedule and a prevalence-preserving control."""

import numpy as np

ARMS = ("point_contract", "anticipation125", "time_shuffled125")
STRATA = (
    "safe",
    "disruptive_30_to_125ms",
    "disruptive_earlier_than_125ms",
    "disruptive_0_to_30ms",
)
TASKS = ("native", "no_MNM", "no_MNM_CIII")


def validate(records):
    if not records:
        raise ValueError("nonempty source TRAIN records required")
    seen = set()
    for r in records:
        if (
            r.get("role") != "train"
            or type(r.get("hdf5_disruptive")) is not bool
            or r.get("hdf5_source_point_valid") is not True
            or r.get("anticipation_stratum") not in STRATA
            or any(
                r.get(k) not in (0, 1)
                for k in ("original_source_point_label", "future_disruption_125ms")
            )
        ):
            raise ValueError(
                "HDF-authoritative valid TRAIN targets and declared strata required"
            )
        safe = not r["hdf5_disruptive"]
        if safe != (r["anticipation_stratum"] == "safe") or (
            safe
            and (
                r["original_source_point_label"] != 0
                or r["future_disruption_125ms"] != 0
            )
        ):
            raise ValueError("safe HDF outcomes cannot have positive targets")
        key = r["shot"], r["query"]
        if key in seen:
            raise ValueError("distinct shot/query teaching records required")
        seen.add(key)


def make_targets(records, seed=131, *, schedule=None):
    """Keep per-coordinate labels stable and match actual sampled prevalence.

    A corpus-only shuffle (schedule=None) does not control stratified repeated
    exposure. Real fitting must supply its fixed schedule and require the
    returned exposure_matched flag. Unsampled coordinates remain untouched.
    """
    validate(records)
    point = np.array(
        [r["original_source_point_label"] for r in records], dtype=np.float32
    )
    future = np.array([r["future_disruption_125ms"] for r in records], dtype=np.float32)
    shuffled, rng = future.copy(), np.random.default_rng(seed)
    frequency = np.ones(len(records), dtype=np.int64)
    if schedule is not None:
        if not schedule:
            raise ValueError("nonempty fixed training schedule required")
        frequency[:] = 0
        for step in schedule:
            take = step.get("indices")
            if (
                not isinstance(take, list)
                or len(take) != 4
                or len(set(take)) != 4
                or any(type(i) is not int or not 0 <= i < len(records) for i in take)
            ):
                raise ValueError(
                    "four distinct valid coordinate indices per training step required"
                )
            np.add.at(frequency, take, 1)
    homogeneous, affected = [], []
    for sid in sorted({r["shot"] for r in records if r["hdf5_disruptive"]}):
        use = np.array([i for i, r in enumerate(records) if r["shot"] == sid])
        if len(np.unique(future[use])) == 1:
            homogeneous.append(sid)
        for count in sorted(set(frequency[use])):
            if count == 0:
                continue
            matched = use[frequency[use] == count]
            shuffled[matched] = rng.permutation(future[matched])
        if not np.array_equal(shuffled[use], future[use]):
            affected.append(sid)
        if shuffled[use].sum() != future[use].sum():
            raise AssertionError("time control changed per-shot class prevalence")
        if frequency[use] @ shuffled[use] != frequency[use] @ future[use]:
            raise AssertionError(
                "time control changed actual per-shot training prevalence"
            )
    return dict(zip(ARMS, (point, future, shuffled))), dict(
        seed=seed,
        permutation="within HDF-disruptive TRAIN shot x equal scheduled occurrence count; safe/unsampled unchanged",
        exposure_matched=schedule is not None,
        training_draws=int(frequency.sum()) if schedule is not None else None,
        unique_scheduled_records=int((frequency > 0).sum())
        if schedule is not None
        else None,
        future_positive_draws=int(frequency @ future) if schedule is not None else None,
        shuffled_positive_draws=int(frequency @ shuffled)
        if schedule is not None
        else None,
        changed_draws=int(frequency @ (shuffled != future))
        if schedule is not None
        else None,
        changed_records=int((shuffled != future).sum()),
        changed_shots=affected,
        homogeneous_disruptive_shots=homogeneous,
        positives=int(future.sum()),
        shuffled_positives=int(shuffled.sum()),
        record_order="same as supplied records",
    )


def make_schedule(records, steps=128, seed=131):
    validate(records)
    if type(steps) is not int or steps < 1:
        raise ValueError("positive integer schedule length required")
    pools = {name: {} for name in STRATA}
    for i, r in enumerate(records):
        pools[r["anticipation_stratum"]].setdefault(r["shot"], []).append(i)
    if any(not pool for pool in pools.values()):
        raise ValueError("all four source teaching strata required")
    rng, schedule = np.random.default_rng(seed), []
    for step in range(steps):
        take = []
        for name in STRATA:
            ids = sorted(pools[name])
            sid = ids[int(rng.integers(len(ids)))]
            options = pools[name][sid]
            take.append(options[int(rng.integers(len(options)))])
        schedule.append(
            dict(step=step + 1, task=TASKS[step % len(TASKS)], indices=take)
        )
    return schedule
