"""Post-completion exposure, immutable-prefix and matched-feedback diagnostics."""

import json

import numpy as np

from ..history_gradient_metric_20261102.run import ROOT, read, sha, write


def main():
    root = ROOT / "probe_outputs/critical_shot_meta360_v1"
    done, protocol = read(root / "completed.json"), read(root / "protocol.json")
    assert done["status"] == "COMPLETED"
    verification = read(root / "verification_records.json")
    assert verification["status"] == "PASS"
    assert verification["hashes"]["completed_sha256"] == sha(root / "completed.json")
    rows = read(root / "per_shot.json")
    arms = protocol["arms"]
    fixed_prefix = []
    same_long_safe = []
    long_prefix_counts = {
        a: {str(ms): 0 for ms in (100, 500, 1000, 3000)} for a in arms
    }
    changes = {
        pair: []
        for pair in (
            "operational-cold",
            "operational-reverse",
            "operational-point",
            "operational-forecast",
        )
    }
    first_feedback = {flag: [] for flag in ("D", "S")}
    state_norms = {a: [] for a in arms if a != "cold"}
    first_alarm_positions = []
    for index, entry in enumerate(protocol["order"], 1):
        sid = entry["shot"]
        stem = f"{index:04d}_{sid}"
        seal = read(root / "shots" / (stem + "_seal.json"))
        curve_path = root / "shots" / (stem + ".npz")
        assert sha(curve_path) == seal["sha256"]
        with np.load(curve_path, allow_pickle=False) as data:
            times = data["times"]
            safe = not rows["cold"][index - 1]["disruptive"]
            if safe:
                crossing = np.flatnonzero(data["cold"] >= protocol["threshold"])
                first = int(crossing[0]) if len(crossing) else None
                if first is not None and (index <= 20 or first < 160):
                    fixed_prefix.append(
                        dict(
                            shot=sid,
                            index=index,
                            first_alarm_index=first,
                            reason="warmup_cost"
                            if index <= 20
                            else "before_first_adaptation_effect",
                        )
                    )
                    assert all(rows[a][index - 1]["alarm"] for a in arms)
                first_alarm_positions.append(
                    dict(
                        shot=sid,
                        record_duration_ms=float((times[-1] - times[0]) * 1000),
                        cold_first_alarm_index=first,
                        operational_first_alarm_index=next(
                            (
                                int(i)
                                for i in np.flatnonzero(
                                    data["operational"] >= protocol["threshold"]
                                )
                            ),
                            None,
                        ),
                    )
                )
                if (times[-1] - times[0]) * 1000 >= 3000:
                    same_long_safe.append(sid)
                    for arm in arms:
                        for ms in (100, 500, 1000, 3000):
                            eligible = (times - times[0]) * 1000 <= ms
                            long_prefix_counts[arm][str(ms)] += int(
                                np.any(data[arm][eligible] >= protocol["threshold"])
                            )
            if index > 20:
                for pair in changes:
                    _, other = pair.split("-")
                    delta = data["operational"].astype(np.float64) - data[other].astype(
                        np.float64
                    )
                    changes[pair].append(
                        dict(
                            shot=sid,
                            rmse=float(np.sqrt(np.mean(delta**2))),
                            max_absolute=float(np.max(np.abs(delta))),
                        )
                    )
        state_path = root / "shots" / (stem + "_states.npz")
        assert sha(state_path) == seal["states_sha256"]
        with np.load(state_path, allow_pickle=False) as data:
            for arm in state_norms:
                after = data[arm + "_after"]
                if len(after):
                    state_norms[arm].append(
                        float(np.linalg.norm(after[-1].astype(np.float64)))
                    )
            if index > 20 and len(data["operational_gradient"]):
                real, reverse = (
                    data["operational_gradient"][0].astype(np.float64),
                    data["reverse_gradient"][0].astype(np.float64),
                )
                assert np.array_equal(
                    data["operational_before"][0], data["reverse_before"][0]
                )
                nr, nb = np.linalg.norm(real), np.linalg.norm(reverse)
                flag = "S" if safe else "D"
                first_feedback[flag].append(
                    dict(
                        shot=sid,
                        cosine=float(np.clip(real @ reverse / nr / nb, -1, 1))
                        if nr and nb
                        else None,
                        real_gradient_norm=float(nr),
                        reverse_gradient_norm=float(nb),
                    )
                )
    n_safe = done["metrics"]["cold"]["full360"]["n_safe"]
    long_n = len(same_long_safe)
    first_summary = {}
    for flag, entries in first_feedback.items():
        values = [r["cosine"] for r in entries if r["cosine"] is not None]
        first_summary[flag] = dict(
            n=len(entries),
            defined=len(values),
            cosine_quantiles=np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
            if values
            else None,
            real_norm_quantiles=np.quantile(
                [r["real_gradient_norm"] for r in entries], [0, 0.5, 0.9, 1]
            ).tolist(),
        )
    result = dict(
        status="COMPLETED_POSTHOC_DIAGNOSTICS",
        immutable_false_positives=dict(
            n=len(fixed_prefix),
            fraction_of_all_safe=len(fixed_prefix) / n_safe,
            records=fixed_prefix,
            scope="Lower bound only for this fixed warmup/first160-unchanged contract; not a general model limit",
        ),
        same_long_safe_prefix_fpr=dict(
            n=long_n,
            shot_ids=same_long_safe,
            counts=long_prefix_counts,
            fractions={
                a: {ms: v / long_n for ms, v in values.items()}
                for a, values in long_prefix_counts.items()
            },
            origin="time since saved HDF record start, not physical discharge ignition",
            not_a_replacement_for_full_shot_fpr=True,
        ),
        probability_changes={
            pair: dict(
                n=len(v),
                rmse_quantiles=np.quantile(
                    [r["rmse"] for r in v], [0, 0.5, 0.9, 1]
                ).tolist(),
                maximum_absolute=max(r["max_absolute"] for r in v),
            )
            for pair, v in changes.items()
        },
        first_matched_feedback=first_summary,
        final_parameter_norm_quantiles={
            a: np.quantile(v, [0, 0.5, 0.9, 1]).tolist() for a, v in state_norms.items()
        },
        safe_first_alarm_positions=first_alarm_positions,
        current_outcomes_from_already_verified_records=True,
        raw_signal_reads=0,
        source_completed_sha256=sha(root / "completed.json"),
    )
    write(root / "posthoc_diagnostics.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "immutable_false_positives",
                    "same_long_safe_prefix_fpr",
                    "probability_changes",
                    "first_matched_feedback",
                    "final_parameter_norm_quantiles",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
