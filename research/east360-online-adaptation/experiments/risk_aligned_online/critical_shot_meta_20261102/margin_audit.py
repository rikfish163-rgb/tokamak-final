"""Post-hoc peak-score changes on previously missed/false-alarm shots."""

import json
from pathlib import Path

import numpy as np

from .. import io23
from ..history_gradient_metric_20261102.run import ROOT, read, sha, write


def main():
    root = ROOT / "probe_outputs/critical_shot_meta360_v1"
    done, p = read(root / "completed.json"), read(root / "protocol.json")
    assert done["status"] == "COMPLETED"
    assert read(root / "verification_records.json")["status"] == "PASS"
    per = read(root / "per_shot.json")
    rows = []
    for index, entry in enumerate(p["order"], 1):
        source = per["cold"][index - 1]
        flag, down = io23.read_outcome("EAST", entry["shot"])
        if (flag and source["timely"]) or (not flag and not source["alarm"]):
            continue
        stem = f"{index:04d}_{entry['shot']}"
        path = root / "shots" / (stem + ".npz")
        assert sha(path) == read(root / "shots" / (stem + "_seal.json"))["sha256"]
        with np.load(path, allow_pickle=False) as data:
            eligible = (
                np.round((down - data["times"]) * 1000, 9) > 30
                if flag
                else np.ones(len(data["times"]), bool)
            )
            baseline = float(data["cold"][eligible].max())
            arm_max = {a: float(data[a][eligible].max()) for a in p["arms"]}
            row = dict(
                shot=entry["shot"],
                index=index,
                disruptive=flag,
                cold_maximum=baseline,
                distance_to_threshold=abs(p["threshold"] - baseline),
                maxima=arm_max,
                supported_timely_endpoint_exists=bool(
                    np.any(eligible & (np.arange(len(eligible)) >= 160))
                ),
            )
            rows.append(row)
    summary = {}
    for group in ("missed_D_all", "missed_D_suffix", "cold_FP"):
        selected = [
            r
            for r in rows
            if (r["disruptive"] if group.startswith("missed") else not r["disruptive"])
            and (group != "missed_D_suffix" or r["index"] > 20)
        ]
        summary[group] = {}
        for arm in ("forecast", "point", "operational", "reverse"):
            # Positive means moving in the desired direction for this group.
            change = np.array(
                [
                    (r["maxima"][arm] - r["cold_maximum"])
                    * (1 if r["disruptive"] else -1)
                    for r in selected
                ]
            )
            gap = np.array([r["distance_to_threshold"] for r in selected])
            nonzero = gap > 0
            fraction = change[nonzero] / gap[nonzero]
            summary[group][arm] = dict(
                n=len(selected),
                toward_threshold=int((change > 0).sum()),
                away=int((change < 0).sum()),
                unchanged=int((change == 0).sum()),
                desired_max_change_quantiles=np.quantile(
                    change, [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                fraction_of_original_gap_closed_quantiles=np.quantile(
                    fraction, [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
            )
    result = dict(
        status="COMPLETED_POSTHOC_PEAK_MARGIN_DIAGNOSTIC",
        summary=summary,
        records=rows,
        no_new_policy_or_threshold=True,
        source_completed_sha256=sha(root / "completed.json"),
        code_sha256=sha(Path(__file__)),
        interpretation="Gap fractions describe sealed peak-score movement; they do not predict linear response to a larger parameter step.",
    )
    result["supersedes"] = (
        "posthoc_peak_margins.json: same numerical analysis; added explicit completion assertion and removed unused-variable lint issue"
    )
    write(root / "posthoc_peak_margins_v2.json", result)
    print(
        json.dumps(
            dict(
                summary=summary,
                boundary_cases=[r for r in rows if r["shot"] in (73790, 60239)],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
