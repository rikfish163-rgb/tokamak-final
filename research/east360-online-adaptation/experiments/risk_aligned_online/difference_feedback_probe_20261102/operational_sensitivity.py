"""Post-hoc pointwise outcome surrogate; never replaces the declared screen.

The source/sample125ms labels stay unchanged. This is not whole-shot scoring:
five sampled points cannot establish the first-alarm TPR or FPR of a shot.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def operational_delta(before_logit, after_logit, disruptive):
    sign = -1 if disruptive else 1
    return float(
        np.logaddexp(0.0, sign * after_logit) - np.logaddexp(0.0, sign * before_logit)
    )


def audit(root):
    output = root / "operational_point_sensitivity.json"
    if output.exists():
        raise FileExistsError(output)
    done, p = read(root / "completed.json"), read(root / "protocol.json")
    assert done["status"] == "COMPLETED_ENDED20_LOCAL_PROBE"
    assert sha(root / "samples.json") == done["samples_sha256"]
    rows = read(root / "samples.json")
    assert all(r["shot"] in p["ids"] for r in rows)
    arms = {}
    for arm in p["arms"]:
        by_shot = defaultdict(list)
        for row in rows:
            if row["disruptive"]:
                assert row["actual_lead_ms"] > 30
            point = row["results"][arm]
            by_shot[(row["disruptive"], row["shot"])].append(
                operational_delta(
                    point["before_logit"], point["after_logit"], row["disruptive"]
                )
            )
        means = {}
        for name, flag in (("D", True), ("SAFE", False)):
            values = [
                float(np.mean(v)) for (kind, _), v in by_shot.items() if kind == flag
            ]
            means[name] = dict(
                n_shots=len(values),
                mean_delta_bce=float(np.mean(values)) if values else None,
            )
        means["balanced_outcome_delta_bce"] = (
            (means["D"]["mean_delta_bce"] + means["SAFE"]["mean_delta_bce"]) / 2
            if means["D"]["n_shots"] and means["SAFE"]["n_shots"]
            else None
        )
        arms[arm] = means
    result = dict(
        status="COMPLETED_POSTHOC_SENSITIVITY_NOT_REVISED_GATE",
        scope="POINTWISE_OUTCOME_SURROGATE_NOT_WHOLE_SHOT_ONLINE_PERFORMANCE",
        arms=arms,
        original_screen_decision=done["summary"]["decision"],
        original_labels_unchanged=True,
        training_objective_unchanged=True,
        prespecified_gate_unchanged=True,
        raw_data_read=0,
        completed_sha256=sha(root / "completed.json"),
        auditor_sha256=sha(Path(__file__)),
        independent_future_confirmation=False,
    )
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
