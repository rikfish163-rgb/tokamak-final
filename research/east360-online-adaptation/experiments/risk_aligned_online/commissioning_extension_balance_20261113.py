"""Audit and summarize retrospective EAST extension balance from completed scores."""

import argparse
from pathlib import Path

import numpy as np

from .commissioning_frozen_extension_20261113 import balanced_indices
from .history_gradient_metric_20261102.run import read, sha, write

SEED = 131
DRAWS = 1000
COMPARISONS = (
    ("current_5", "bias_only"),
    ("current_0", "bias_only"),
    ("current_5", "earlier_5"),
)
METRICS = ("delta_tpr_pp", "delta_fpr_pp")
PERCENTILES = (2.5, 50, 97.5)


def summarize(protocol, per_shot, manifest, result):
    order, arms, thresholds = (protocol[k] for k in ("order", "arms", "calibration"))
    ids = [row["shot"] for row in order]
    if len(set(ids)) != len(ids) or any(type(r["old835"]) is not bool for r in order):
        raise ValueError("protocol needs unique shot IDs and boolean old835 membership")
    if set(per_shot) != set(thresholds) or len(set(thresholds.values())) != 2:
        raise ValueError(
            "expected exactly two distinct fixed thresholds in the scored bands"
        )
    if result["raw_verified"] != len(ids) * len(arms) * len(thresholds):
        raise ValueError("incomplete score count")

    fixed_ids = [row["shot"] for row in manifest["order"]]
    if len(set(fixed_ids)) != len(fixed_ids):
        raise ValueError("duplicate shot in the fixed balance manifest")
    entries = {row["shot"]: row for row in order}
    if manifest["seed"] != protocol["balance_seed"] or any(
        entries.get(row["shot"]) != row for row in manifest["order"]
    ):
        raise ValueError("fixed balance manifest does not match the protocol")
    index = {sid: i for i, sid in enumerate(ids)}
    fixed = np.asarray([index[sid] for sid in fixed_ids], dtype=int)

    labels = None
    calls = {}
    for band, by_arm in per_shot.items():
        if set(by_arm) != set(arms):
            raise ValueError(f"scored arms differ from protocol at {band}")
        calls[band] = {}
        for arm, rows in by_arm.items():
            if [row["shot"] for row in rows] != ids:
                raise ValueError(f"shot IDs or order differ at {band}/{arm}")
            if any(
                type(row[key]) is not bool
                for row in rows
                for key in ("disruptive", "alarm", "timely")
            ):
                raise ValueError(
                    f"expected boolean outcomes and score flags at {band}/{arm}"
                )
            y = np.array([row["disruptive"] for row in rows], dtype=bool)
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise ValueError(f"outcome labels differ at {band}/{arm}")
            calls[band][arm] = np.array(
                [row["timely"] if row["disruptive"] else row["alarm"] for row in rows],
                dtype=np.int8,
            )

    old = np.array([row["old835"] for row in order], dtype=bool)
    if not np.array_equal(fixed, balanced_indices(labels, old)):
        raise ValueError("fixed manifest differs from the declared score-blind selection")
    strata, sizes = [], {}
    for name, membership in (("old835", True), ("other", False)):
        pos, neg = (
            np.flatnonzero((old == membership) & (labels == y)) for y in (True, False)
        )
        weight = min(len(pos), len(neg))
        fixed_pos, fixed_neg = (np.intersect1d(fixed, pool).size for pool in (pos, neg))
        if not weight or fixed_pos != weight or fixed_neg != weight:
            raise ValueError(f"fixed manifest is not maximal 1:1 within {name}")
        strata.append(((pos, neg), weight))
        sizes[name] = dict(
            n_disruptive=len(pos), n_safe=len(neg), balanced_per_class=weight
        )
    n = sum(weight for _, weight in strata)

    fixed_counts = {}
    fixed_pos, fixed_neg = fixed[labels[fixed]], fixed[~labels[fixed]]
    for band, by_arm in calls.items():
        fixed_counts[band] = {}
        metrics = result["groups"]["balanced_1to1"][band]["metrics"]
        for arm, call in by_arm.items():
            tp, fp = int(call[fixed_pos].sum()), int(call[fixed_neg].sum())
            counts = dict(tp=tp, fp=fp, n_disruptive=n, n_safe=n)
            if any(metrics[arm][k] != v for k, v in counts.items()):
                raise ValueError(
                    f"fixed manifest counts differ from result at {band}/{arm}"
                )
            fixed_counts[band][arm] = counts

    bands = {}
    for band, threshold in thresholds.items():
        if threshold in bands:
            first = bands[threshold][0]
            if any(not np.array_equal(calls[first][a], calls[band][a]) for a in arms):
                raise ValueError(
                    f"identical thresholds give different score flags at {first}/{band}"
                )
            bands[threshold].append(band)
        else:
            bands[threshold] = [band]
    differences = np.stack(
        [
            np.stack([calls[names[0]][a] - calls[names[0]][b] for a, b in COMPARISONS])
            for names in bands.values()
        ]
    )

    # The same stratum weights apply to the TPR and FPR denominators.
    standardized = (
        100
        / n
        * sum(
            weight
            * np.stack(
                [differences[..., pool].mean(axis=-1) for pool in pools], axis=-1
            )
            for pools, weight in strata
        )
    )
    rng = np.random.default_rng(SEED)
    draws = np.empty((DRAWS, *standardized.shape))
    absolute_calls = np.stack(
        [np.stack([calls[names[0]][a] for a in arms]) for names in bands.values()]
    )
    false_rates = np.empty((DRAWS, len(bands), len(arms)))
    for i in range(DRAWS):
        for j in range(2):  # disruptive, then safe
            selected = np.concatenate(
                [
                    pools[j]
                    if len(pools[j]) == weight
                    else rng.choice(pools[j], weight, replace=False)
                    for pools, weight in strata
                ]
            )
            # One selected ID set per class and draw for all arms and bands.
            draws[i, ..., j] = 100 / n * differences[..., selected].sum(axis=-1)
            if j == 1:
                false_rates[i] = absolute_calls[..., selected].sum(axis=-1) / n
    quantiles = np.percentile(draws, PERCENTILES, axis=0, method="linear")
    absolute_quantiles = np.percentile(100 * false_rates, PERCENTILES, axis=0)

    return dict(
        interpretation="Quantiles describe variability from balancing this retrospective cohort; "
        "they are not independent prospective confidence intervals.",
        standardization="Average within-stratum rate differences with weights proportional to "
        "the stratum minority counts, identical for disruptive and safe classes.",
        seed=SEED,
        draws=DRAWS,
        percentiles=PERCENTILES,
        strata=sizes,
        fixed_counts=fixed_counts,
        bands=[
            dict(
                threshold=threshold,
                aliases=names,
                absolute_fpr={
                    a: dict(quantiles_pct=absolute_quantiles[:, i, j].tolist(),
                            fraction_of_draws_at_or_below={
                                str(cap): float((false_rates[:, i, j] <= cap).mean())
                                for cap in (.10,.15,.20)
                            })
                    for j, a in enumerate(arms)
                },
                comparisons={
                    f"{a}_vs_{b}": dict(
                        quantiles_pp=dict(
                            zip(METRICS, quantiles[:, i, j, :].T.tolist(), strict=True)
                        ),
                        standardized_pp=dict(
                            zip(METRICS, standardized[i, j].tolist(), strict=True)
                        ),
                    )
                    for j, (a, b) in enumerate(COMPARISONS)
                },
            )
            for i, (threshold, names) in enumerate(bands.items())
        ],
    )


def false_alarm_support(run, protocol, seal, per_shot):
    """Separate inherited crossings that any same-clock write must leave unchanged."""
    out = {}
    rows = protocol["order"]
    for band in ("0.2", "0.15"):
        tau = protocol["calibration"][band]
        parent, candidate = (per_shot[band][k] for k in ("bias_only", "current_5"))
        marks = []
        for rank, (entry, a, b) in enumerate(zip(rows, parent, candidate, strict=True), 1):
            if a["disruptive"] or not (a["alarm"] or b["alarm"]):
                continue
            name = f"{rank:04d}_{entry['shot']}.npz"
            path = run / "shots" / name
            if sha(path) != seal["curve_sha256"][name]:
                raise ValueError(f"changed sealed false-alarm curve: {name}")
            with np.load(path, allow_pickle=False) as f:
                no_write = f["dose"][-1] == 0.
                ah, bh = (f[k] >= tau for k in ("bias_only", "current_5"))
            if bool(ah.any()) != a["alarm"] or bool(bh.any()) != b["alarm"]:
                raise ValueError(f"raw crossing disagrees with sealed score: {name}")
            np.testing.assert_array_equal(ah[no_write], bh[no_write])
            first = int(np.flatnonzero(bh)[0]) if b["alarm"] else None
            marks.append(dict(shot=entry["shot"], old835=entry["old835"],
                parent=a["alarm"], candidate=b["alarm"], any_immutable=bool((ah & no_write).any()),
                before_first_write=bool(first is not None and no_write[:first+1].all())))
        out[band] = marks
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="completed frozen extension directory")
    parser.add_argument("output", type=Path, help="new JSON report path")
    args = parser.parse_args()
    result = read(args.run / "result.json")
    if result["status"] != "COMPLETE_FROZEN_EXTENDED_EAST_DEVELOPMENT":
        raise ValueError("expected completed frozen extension scores")
    protocol = read(args.run / "protocol.json")
    seal = read(args.run / "seal.json")
    if set(seal["curve_sha256"]) != {
        f"{rank:04d}_{row['shot']}.npz" for rank, row in enumerate(protocol["order"], 1)
    }:
        raise ValueError("seal does not cover the protocol order")
    manifest = read(args.run / "balanced_cohort.json")
    per_shot = read(args.run / "per_shot.json")
    summary = summarize(protocol, per_shot, manifest, result)
    summary["false_alarm_support"] = false_alarm_support(args.run, protocol, seal, per_shot)
    write(args.output, summary)


if __name__ == "__main__":
    main()
