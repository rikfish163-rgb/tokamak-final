"""Post-run history-readiness sensitivity, never a neural adaptation gain."""

import argparse
from pathlib import Path
import zipfile

import numpy as np

from .. import io23
from ..hdf5_score_recheck_20261103 import score_curve
from ..history_gradient_metric_20261102.run import read, sha, write
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .static_stream import timing_decomposition


def score_history(times, scores, disruptive, down, threshold, *, deadline, history):
    times, scores = np.asarray(times), np.asarray(scores)
    if (
        type(history) is not int
        or history < 1
        or deadline not in (10, 30)
        or times.ndim != 1
        or scores.shape != times.shape
        or not len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("finite matched curves and explicit positive history required")
    ready = len(times) >= history
    feasible = bool(
        ready and disruptive and round((down - times[history - 1]) * 1000, 9) > deadline
    )
    if not ready:
        # The policy cannot emit a full-window prediction, but the shot remains
        # in the full-cohort denominator. This is not a fabricated safe label.
        return dict(
            alarm=False,
            lead_ms=None,
            timely=False,
            history_ready=False,
            timely_history_feasible=False,
        )
    result = score_curve(
        times[history - 1 :], scores[history - 1 :], disruptive, down, threshold
    )
    result["timely"] = bool(
        result["lead_ms"] is not None and result["lead_ms"] > deadline
    )
    return dict(result, history_ready=True, timely_history_feasible=feasible)


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    roots = (
        (
            "EAST",
            30,
            io23.ROOT / "probe_outputs/qualified_source_static360_20261104_v1",
        ),
        (
            "J-TEXT",
            10,
            io23.ROOT / "probe_outputs/source_missing_diagnostic_20261104_v1",
        ),
    )
    all_rows, summaries, bindings = {}, {}, {}
    for device, deadline, root in roots:
        done, protocol, old = (
            read(root / name)
            for name in ("completed.json", "protocol.json", "per_shot.json")
        )
        for name, expected in done["artifact_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError("sealed run artifact changed")
        bindings[str(root / "completed.json")] = sha(root / "completed.json")
        by_arm = {arm: {str(n): [] for n in (1, 128)} for arm in old}
        ids = [r["shot"] for r in next(iter(old.values()))]
        if len(ids) != len(set(ids)) or any(
            [r["shot"] for r in rows] != ids for rows in old.values()
        ):
            raise ValueError("unique identical fixed-cohort ordering required")
        if set(ids) != set(io23.cohort_ids(device)):
            raise ValueError("locked device test membership required")
        for index, sid in enumerate(ids, 1):
            prefix = root / "shots" / f"{index:04d}_{sid}"
            curve = prefix.with_suffix(".npz")
            seal = read(prefix.with_name(prefix.name + "_seal.json"))
            if (
                seal != done["seals"][index - 1]
                or seal["shot"] != sid
                or sha(curve) != seal["curve_sha256"]
            ):
                raise ValueError("curve seal changed")
            disruptive, down = io23.read_outcome(device, sid)
            with np.load(curve, allow_pickle=False) as saved:
                times = saved["times"]
                for arm in old:
                    for n in (1, 128):
                        scored = score_history(
                            times,
                            saved[arm],
                            disruptive,
                            down,
                            protocol["threshold"],
                            deadline=deadline,
                            history=n,
                        )
                        if n == 1 and any(
                            scored[k] != old[arm][index - 1][k]
                            for k in ("alarm", "lead_ms", "timely")
                        ):
                            raise AssertionError("primary score does not reproduce")
                        by_arm[arm][str(n)].append(
                            dict(shot=sid, disruptive=disruptive, **scored)
                        )
        all_rows[device] = by_arm
        summaries[device] = {
            arm: {
                n: dict(
                    timing_decomposition(rows, deadline_ms=deadline),
                    history_unavailable_shots=sum(not r["history_ready"] for r in rows),
                    timely_history_feasible_disruptions=sum(
                        r["timely_history_feasible"] for r in rows
                    ),
                )
                for n, rows in policies.items()
            }
            for arm, policies in by_arm.items()
        }
    output.mkdir(parents=True)
    code = imported_code_hashes()
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
    write(output / "per_shot.json", all_rows)
    write(
        output / "summary.json",
        dict(
            scope="POSTHOC_HISTORY_POLICY_SENSITIVITY_NOT_PRIMARY_SCORE_OR_NEURAL_GAIN",
            groups=summaries,
            primary_scores_unchanged=True,
            source_label_windows_unchanged=True,
            thresholds_unchanged=True,
            original_shots_excluded=0,
            raw_signal_arrays_read=0,
            input_sha256=bindings,
            code_sha256=code,
        ),
    )
    write(
        output / "completed.json",
        dict(
            status="COMPLETED_POSTHOC_HISTORY_SENSITIVITY",
            artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    print(
        {
            device: {
                arm: {
                    n: {
                        k: v[k]
                        for k in (
                            "tp",
                            "fp",
                            "tardy",
                            "miss",
                            "history_unavailable_shots",
                            "timely_history_feasible_disruptions",
                        )
                    }
                    for n, v in policies.items()
                }
                for arm, policies in arms.items()
            }
            for device, arms in summaries.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
