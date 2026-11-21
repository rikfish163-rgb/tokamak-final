"""Verify the old event cache's outcomes against fresh TRAIN/VAL HDF truth."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import io23
from .event_conditional_feedback_20261102 import context_fit as c
from .history_gradient_metric_20261102.run import sha, write


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = sha(c.CACHE / "features.npz")
    if before != c.CACHE_SHA:
        raise ValueError("pinned event cache changed")
    with np.load(c.CACHE / "features.npz", allow_pickle=False) as archive:
        original = c.split_cache({k: archive[k] for k in c.FIELDS})
    receipts, all_unchanged, target_unchanged, calls = {}, {}, {}, []
    reader = io23.read_outcome
    allowed = set(io23.source_roles()["train"]) | set(io23.source_roles()["val"])

    def checked(device, sid):
        if device != "J-TEXT" or sid not in allowed:
            raise ValueError("TRAIN/VAL HDF outcomes only; never TEST or EAST")
        calls.append(sid)
        return reader(device, sid)

    with patch.object(io23, "read_outcome", side_effect=checked):
        for role, data in original.items():
            rebound, receipts[role] = c.hdf_authoritative_outcomes(data)
            all_unchanged[role] = all(np.array_equal(data[k], rebound[k]) for k in data)
            target_unchanged[role] = {}
            for shuffled in (False, True):
                old = c.training_targets(data, shuffled=shuffled)
                new = c.training_targets(rebound, shuffled=shuffled)
                target_unchanged[role][str(shuffled)] = all(
                    np.array_equal(a, b) for a, b in zip(old, new, strict=True)
                )
    if sha(c.CACHE / "features.npz") != before:
        raise AssertionError("historical cache changed")
    output.mkdir()
    result = dict(
        status="PASS",
        scope="FRESH_EVENT_CACHE_HDF_OUTCOME_RECONCILIATION_NOT_TRAINING",
        roles=receipts,
        all_cached_fields_unchanged=all_unchanged,
        true_and_shuffled_targets_unchanged=target_unchanged,
        hdf_outcome_reads=len(calls),
        distinct_hdf_shots=len(set(calls)),
        test_hdf_reads=0,
        east_hdf_reads=0,
        cache_contains_test_bytes=True,
        cache_file_unchanged=True,
        cache_sha256=before,
        input_and_code_sha256={
            str(p): sha(p)
            for p in (Path(__file__), Path(c.__file__), Path(io23.__file__))
        },
        model_training=False,
        online_gain_measured=False,
    )
    write(output / "recheck.json", result)
    write(
        output / "completed.json",
        dict(
            status="PASS",
            artifact_sha256={"recheck.json": sha(output / "recheck.json")},
        ),
    )
    print(
        {
            k: result[k]
            for k in (
                "status",
                "hdf_outcome_reads",
                "distinct_hdf_shots",
                "all_cached_fields_unchanged",
                "true_and_shuffled_targets_unchanged",
                "test_hdf_reads",
                "east_hdf_reads",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
