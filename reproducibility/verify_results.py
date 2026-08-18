from __future__ import annotations

from typing import Any


METRICS = ("tpr", "fpr", "tpr_at_30ms", "median_lead_ms")


def compare_multiseed_results(
    actual: dict[str, Any], expected: dict[str, Any], *, tolerance: float = 1e-8,
) -> dict[str, Any]:
    actual_by_seed = {int(row["seed"]): row for row in actual["seeds"]}
    expected_by_seed = {int(row["seed"]): row for row in expected["seeds"]}
    if set(actual_by_seed) != set(expected_by_seed):
        return {
            "passed": False,
            "checked_seeds": sorted(set(actual_by_seed) & set(expected_by_seed)),
            "error": "seed set mismatch",
        }
    differences = []
    for seed in sorted(expected_by_seed):
        actual_row = actual_by_seed[seed]
        expected_row = expected_by_seed[seed]
        for branch in ("static", "online"):
            for metric in METRICS:
                observed = float(actual_row[branch][metric])
                reference = float(expected_row[branch][metric])
                delta = abs(observed - reference)
                if delta > tolerance:
                    differences.append({
                        "seed": seed, "branch": branch, "metric": metric,
                        "actual": observed, "expected": reference, "delta": delta,
                    })
        if int(actual_row["query_count"]) != int(expected_row["query_count"]):
            differences.append({
                "seed": seed, "metric": "query_count",
                "actual": int(actual_row["query_count"]),
                "expected": int(expected_row["query_count"]),
            })
    return {
        "passed": not differences,
        "checked_seeds": sorted(expected_by_seed),
        "tolerance": tolerance,
        "differences": differences,
    }
