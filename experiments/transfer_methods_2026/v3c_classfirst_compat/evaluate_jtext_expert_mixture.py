#!/usr/bin/env python3
"""Independently recompute frozen J-TEXT expert-mixture predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import online_evidence as evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads((args.run / "predictions.json").read_text(encoding="utf-8"))
    ledger = evidence.read_jsonl(args.run / "query_ledger.jsonl")
    store = evidence.TruthStore(args.data_root)
    rebuilt = {}
    for view in ("static", "online"):
        rows = []
        seen = set()
        for sealed in predictions[view]:
            shot = int(sealed["shot"])
            if shot in seen:
                raise RuntimeError(f"duplicate prediction: {shot}")
            seen.add(shot)
            truth = store.truth("JTEXT", shot)
            rows.append({
                "shot": shot, **truth, "score": float(sealed["score"]),
                "alarmed": bool(sealed["alarmed"]),
                "alarm_time_s": sealed["alarm_time_s"],
            })
        rebuilt[view] = rows
    if [row["shot"] for row in rebuilt["static"]] != [row["shot"] for row in rebuilt["online"]]:
        raise RuntimeError("static and online rosters differ")
    if any(row.get("outcome_accessed_for_score") is not False for row in ledger):
        raise RuntimeError("query score accessed outcome")
    result = {
        "static": evidence.compute_metrics(rebuilt["static"], device="JTEXT"),
        "online": evidence.compute_metrics(rebuilt["online"], device="JTEXT"),
        "delta": evidence.compute_delta(
            rebuilt["online"], rebuilt["static"], device="JTEXT",
        ),
        "n_predictions": len(rebuilt["online"]),
        "query_count": sum(bool(row["selected"]) for row in ledger),
        "query_score_outcome_free": True,
        "truth_reloaded_from_hdf5": True,
    }
    (args.run / "independent_evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
