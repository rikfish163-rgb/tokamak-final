"""Post-hoc linearized capacity screen; NOT a fitted deployable online policy.

HDF-signed gradients include the outcome of each analyzed episode. Even LOSO
here is an offline capacity diagnostic, not chronological target-stream gain.
"""

import argparse
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import linprog

from ..history_gradient_metric_20261102.run import read, sha, write


def bounded_minimax(utilities, active):
    utilities, active = np.asarray(utilities), np.asarray(active)
    if (
        utilities.ndim != 2
        or utilities.shape[0] != 3
        or active.shape != utilities.shape[1:]
        or active.dtype != bool
        or not active.any()
        or not np.isfinite(utilities).all()
    ):
        raise ValueError(
            "finite3-group utilities and nonempty observed-channel mask required"
        )
    u = utilities[:, active]
    n = int(active.sum())
    fitted = linprog(
        np.r_[np.zeros(n), 1.0],
        A_ub=np.column_stack((u, -np.ones(3))),
        b_ub=np.zeros(3),
        A_eq=np.array([np.r_[np.ones(n), 0.0]]),
        b_eq=[n],
        bounds=[(0, 2)] * n + [(None, None)],
        method="highs",
    )
    if not fitted.success:
        raise RuntimeError("bounded utility LP failed: " + fitted.message)
    weights = np.zeros(len(active))
    weights[active] = fitted.x[:-1]
    values = utilities @ weights
    if (
        np.min(weights) < -1e-8
        or np.max(weights) > 2 + 1e-8
        or abs(weights.sum() - n) > 1e-6
    ):
        raise AssertionError("solver returned a channel budget violation")
    return dict(
        weights=weights.tolist(),
        active_channels=n,
        uniform_group_estimates=u.sum(1).tolist(),
        weighted_group_estimates=values.tolist(),
        worst_group_estimate=float(np.max(values)),
        scope="linearized HDF-signed capacity only, not an online policy",
    )


def group_utilities(rows):
    result = []
    for group in ("safe", "disruptive", "danger_window"):
        selected = [
            r
            for r in rows
            if (
                not r["disruptive"]
                if group == "safe"
                else r["disruptive"]
                and (group == "disruptive" or 30 < r["lead_ms"] <= 125)
            )
        ]
        ids = sorted({r["shot"] for r in selected})
        if not ids:
            raise ValueError("every training fold needs all declared groups")
        result.append(
            np.mean(
                [
                    np.mean(
                        [
                            [c["signed_first_order_delta_bce"] for c in r["channels"]]
                            for r in selected
                            if r["shot"] == sid
                        ],
                        axis=0,
                    )
                    for sid in ids
                ],
                axis=0,
            )
        )
    return np.asarray(result)


def main(trace, output):
    trace, output = Path(trace), Path(output)
    if output.exists():
        raise FileExistsError(output)
    completed = read(trace / "completed.json")
    if completed["status"] != "COMPLETED":
        raise ValueError("completed offline gradient trace required")
    for name, expected in completed["artifact_sha256"].items():
        if sha(trace / name) != expected:
            raise ValueError("trace artifact changed")
    rows = [r for r in read(trace / "rows.json") if r["stem"] == "original"]
    if len(rows) != 100 or len({r["shot"] for r in rows}) != 20:
        raise ValueError("fixed original-source100/20 development support required")

    def observed(records):
        return np.array(
            [any(r["channels"][c]["observed"] for r in records) for c in range(23)]
        )

    global_fit = bounded_minimax(group_utilities(rows), observed(rows))
    folds, validation = [], []
    for sid in sorted({r["shot"] for r in rows}):
        train = [r for r in rows if r["shot"] != sid]
        fit = bounded_minimax(group_utilities(train), observed(train))
        folds.append(
            dict(held_shot=sid, training_ids=sorted({r["shot"] for r in train}), **fit)
        )
        for r in [r for r in rows if r["shot"] == sid]:
            u = np.array([c["signed_first_order_delta_bce"] for c in r["channels"]])
            validation.append(
                dict(
                    shot=sid,
                    query=r["query"],
                    disruptive=r["disruptive"],
                    lead_ms=r["lead_ms"],
                    uniform=float(u.sum()),
                    weighted=float(u @ np.array(fit["weights"])),
                )
            )
    if {(r["shot"], r["query"]) for r in validation} != {
        (r["shot"], r["query"]) for r in rows
    }:
        raise AssertionError("held rows must preserve every unique shot/query identity")
    held_summary = {}
    for group in ("safe", "disruptive", "danger_window"):
        selected = [
            r
            for r in validation
            if (
                not r["disruptive"]
                if group == "safe"
                else r["disruptive"]
                and (group == "disruptive" or 30 < r["lead_ms"] <= 125)
            )
        ]
        ids = sorted({r["shot"] for r in selected})
        held_summary[group] = {
            key: float(
                np.mean(
                    [
                        np.mean([r[key] for r in selected if r["shot"] == sid])
                        for sid in ids
                    ]
                )
            )
            for key in ("uniform", "weighted")
        }
    output.mkdir(parents=True)
    with zipfile.ZipFile(
        output / "producer_code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(
            Path(__file__),
            str(
                Path(__file__)
                .resolve()
                .relative_to(Path(__file__).resolve().parents[3])
            ),
        )
    write(
        output / "capacity.json",
        dict(
            scope="POSTHOC_LINEARIZED_UPPER_CAPACITY_AND_OFFLINE_LOSO_NOT_NEURAL_GAIN",
            trace_completion_sha256=sha(trace / "completed.json"),
            code_sha256=sha(Path(__file__)),
            grouping="HDF safe / disruptive /30-125ms, shot-macro means",
            limits="0<=w<=2; observed-channel mean1; no weight returned for unobserved targets",
            hypothesis_chosen_after_trace=True,
            operational_gain_measured=False,
            chronological_validation=False,
            source_updates=0,
            raw_signal_reads=0,
            caution="weights use past-shot labels only if later integrated; current LOSO train includes later20 shots; no deployable causality claim; changed metric normalization and nonlinear effects need actual neural replay",
            all20_fit=global_fit,
            folds=folds,
            held_rows=validation,
            held_summary=held_summary,
        ),
    )
    write(
        output / "completed.json",
        dict(
            status="COMPLETED",
            neural_updates=0,
            artifact_sha256={
                name: sha(output / name)
                for name in ("capacity.json", "producer_code_snapshot.zip")
            },
        ),
    )
    print(
        dict(
            status="COMPLETED",
            global_worst_first_order=global_fit["worst_group_estimate"],
            offline_loso_first_order=held_summary,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.trace, args.output)
