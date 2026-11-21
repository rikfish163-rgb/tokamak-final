"""Post-run clock coverage of the training-label 30--125ms interval.

The label horizon is not a measured physical precursor onset. Earlier
adaptation can still help a shot with no update in this interval; these counts
are descriptive, not a causal explanation of individual misses.
"""

import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def opportunity(times, down, history):
    groups = dict(clock_endpoints=[], observed_endpoints=[], accepted_endpoints=[])
    for record in history:
        end, effect = record["target_last"], record["first_affected"]
        assert effect == end + 1
        if effect >= len(times):
            continue
        mature_lead = round(float((down - times[end]) * 1000), 9)
        effect_lead = round(float((down - times[effect]) * 1000), 9)
        if mature_lead <= 125 + 1e-6 and effect_lead > 30:
            groups["clock_endpoints"].append(end)
            if record["observed"]:
                groups["observed_endpoints"].append(end)
            if record["accepted"]:
                groups["accepted_endpoints"].append(end)
    return groups


def audit(root):
    destination = root / "clock_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    completed = read(root / "completed.json")
    assert completed["status"] == "COMPLETED"
    p, saved = read(root / "protocol.json"), read(root / "per_shot.json")
    groups = {
        name: dict(n=0, with_clock=0, with_observed=0, with_accepted=0)
        for name in (
            "all_D",
            "warmup_D",
            "suffix_D",
            "suffix_cold_missed_D",
            "suffix_cold_timely_D",
        )
    }
    rows = []
    for i, record in enumerate(p["order"]):
        sid, prefix = record["shot"], f"{i + 1:04d}_{record['shot']}"
        flag, down = io23.read_outcome("EAST", sid)
        assert flag == saved["cold"][i]["disruptive"]
        if not flag:
            continue
        path = root / "shots" / f"{prefix}.npz"
        logs = root / "shots" / f"{prefix}_updates.json"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["sha256"] and sha(logs) == seal["updates_sha256"]
        history = read(logs)["updates"]["real"]
        with np.load(path, allow_pickle=False) as curves:
            evidence = opportunity(curves["times"], down, history)
        tags = ["all_D", "warmup_D" if i < 20 else "suffix_D"]
        if i >= 20:
            tags.append(
                "suffix_cold_timely_D"
                if saved["cold"][i]["timely"]
                else "suffix_cold_missed_D"
            )
        for tag in tags:
            count = groups[tag]
            count["n"] += 1
            count["with_clock"] += bool(evidence["clock_endpoints"])
            count["with_observed"] += bool(evidence["observed_endpoints"])
            count["with_accepted"] += bool(evidence["accepted_endpoints"])
        rows.append(
            dict(
                shot=sid, rank=i + 1, cold_timely=saved["cold"][i]["timely"], **evidence
            )
        )
    result = dict(
        status="COMPLETED_DESCRIPTIVE_POSTHOC_AUDIT",
        groups=groups,
        rows=rows,
        completed_sha256=sha(root / "completed.json"),
        auditor_sha256=sha(Path(__file__)),
        interval_ms="mature feedback within <=125ms; first effect strictly >30ms",
        physical_precursor_onset_measured=False,
        causal_miss_explanation=False,
        earlier_adaptation_can_still_help=True,
    )
    _durable_json(destination, result)
    print(groups, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
