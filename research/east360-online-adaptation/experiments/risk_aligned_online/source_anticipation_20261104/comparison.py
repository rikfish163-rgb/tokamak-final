"""Compare fixed source-VAL screens; never interpret sampled points as TPR/FPR."""

import argparse
import math
from pathlib import Path

import numpy as np

from ..history_gradient_metric_20261102.run import read, sha, write


def _index(rows):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in ("view", "head", "mode", "shot", "query"))
        if key in result:
            raise ValueError("duplicate neural record")
        result[key] = row
    return result


def paired_check(old, new):
    left, right = _index(old), _index(new)
    if left.keys() != right.keys():
        raise ValueError("different neural record grid")
    reference = 0
    for key, row in right.items():
        prior = left[key]
        if row["before_logit"] != prior["before_logit"]:
            raise ValueError("static logits changed")
        if row["head"] == "native_reference":
            if row["after_logit"] != prior["after_logit"]:
                raise ValueError("frozen native reference update changed")
            reference += 1
    return dict(
        records=len(right), static_exact=len(right), native_reference_exact=reference
    )


def feedback_dependence(rows, *, cutoff):
    indexed = _index(rows)
    result = {}
    candidates = [
        r for r in rows if r["head"] == "anticipation125" and r["mode"] == "real"
    ]
    for view in sorted({r["view"] for r in candidates}):
        result[view] = {}
        for group in sorted({r["stratum"] for r in candidates if r["view"] == view}):
            selected = [
                r for r in candidates if r["view"] == view and r["stratum"] == group
            ]
            holds = [
                indexed[view, r["head"], "hold", r["shot"], r["query"]]
                for r in selected
            ]
            result[view][group] = dict(
                records=len(selected),
                risk_up=sum(r["after_logit"] > r["before_logit"] for r in selected),
                cross_up=sum(
                    r["before_logit"] < cutoff <= r["after_logit"] for r in selected
                ),
                cross_down=sum(
                    r["after_logit"] < cutoff <= r["before_logit"] for r in selected
                ),
                mean_absolute_logit_move=float(
                    np.mean(
                        [abs(r["after_logit"] - r["before_logit"]) for r in selected]
                    )
                ),
                mean_absolute_real_hold_logit_difference=float(
                    np.mean(
                        [
                            abs(r["after_logit"] - h["after_logit"])
                            for r, h in zip(selected, holds)
                        ]
                    )
                ),
                real_hold_identical_states=sum(
                    r["state_sha256"] == h["state_sha256"]
                    for r, h in zip(selected, holds)
                ),
            )
    return result


def main(old, new, output):
    old, new, output = Path(old), Path(new), Path(output)
    if output.exists():
        raise FileExistsError(output)
    for root in (old, new):
        completion = read(root / "completed.json")
        if completion["status"] != "COMPLETED_SOURCE_AUXILIARY_VALIDATION":
            raise ValueError("completed source VAL screens required")
        for name, digest in completion["artifact_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("screen artifact changed")
    protocols = [read(root / "protocol.json") for root in (old, new)]
    for key in (
        "source_sha256",
        "design_completion_sha256",
        "raw_sha256",
        "threshold",
        "views",
        "heads",
        "modes",
    ):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError("unmatched source screen protocol: " + key)
    previous, current = [read(root / "scored.json") for root in (old, new)]
    threshold = protocols[0]["threshold"]
    cutoff = math.log(threshold / (1 - threshold))
    result = dict(
        scope="SOURCE_VAL_LOCAL_LOSS_AND_FEEDBACK_DIAGNOSTIC_NOT_OPERATIONAL_TPR_FPR",
        paired=paired_check(previous, current),
        old_feedback=feedback_dependence(previous, cutoff=cutoff),
        new_feedback=feedback_dependence(current, cutoff=cutoff),
        old_summary=read(old / "summary.json"),
        new_summary=read(new / "summary.json"),
        inputs={
            str(root / "completed.json"): sha(root / "completed.json")
            for root in (old, new)
        },
        code_sha256=sha(Path(__file__)),
        independent_future_validation=False,
    )
    output.mkdir(parents=True)
    write(output / "comparison.json", result)
    write(
        output / "completed.json",
        dict(
            status="COMPLETED",
            artifact_sha256={"comparison.json": sha(output / "comparison.json")},
        ),
    )
    print(
        dict(
            paired=result["paired"],
            candidate_gate=result["new_summary"]["candidate_gate"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.old, args.new, args.output)
