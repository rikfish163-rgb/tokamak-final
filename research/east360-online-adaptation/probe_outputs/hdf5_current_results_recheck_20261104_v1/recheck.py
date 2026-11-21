"""Re-score five recent sealed runs against a fresh all-disk HDF inventory.

Run from the worktree with rl_env Python. This is a post-run audit, not an
online label-access path. No raw data, historical output, or model is changed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "probe_outputs/hdf5_outcome_authority_20261104_v4"
RUNS = (
    "qualified_source_static360_20261104_v1",
    "direct_fast_memory360_20261104_v1",
    "direct_fast_memory_ridge360_20261104_v1",
    "shadow_commit360_20261104_v1",
    "n1_alignment360_20261104_v1",
)
EXPECTED_STATUS = {
    "qualified_source_static360_20261104_v1": "COMPLETED_STATIC_BASELINE_DIAGNOSTIC",
    "direct_fast_memory360_20261104_v1": "COMPLETED_DIRECT_MEMORY_INTRA_DIAGNOSTIC",
    "direct_fast_memory_ridge360_20261104_v1": "COMPLETED_DIRECT_MEMORY_INTRA_DIAGNOSTIC",
    "shadow_commit360_20261104_v1": "COMPLETED_SHADOW_COMMIT_DIAGNOSTIC",
    "n1_alignment360_20261104_v1": "COMPLETED_N1_ALIGNMENT_STATIC_CONTROL",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    from experiments.risk_aligned_online import io23
    from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import (
        score_curve,
    )

    output = Path(__file__).with_name("result.json")
    if output.exists():
        raise FileExistsError(output)
    done = read(INVENTORY / "completed.json")
    assert done["status"] == "PASS"
    for name, digest in done["artifact_sha256"].items():
        assert sha(INVENTORY / name) == digest
    east = {r["shot"]: r for r in read(INVENTORY / "hdf5_metadata.json")["EAST"]}
    cohort = read(io23.EAST_COHORT)["test"]
    fixed = set(cohort["disruption_shots"] + cohort["non_disruption_shots"])
    assert len(fixed) == 360
    results, input_hashes = {}, {}
    for name in RUNS:
        run = ROOT / "probe_outputs" / name
        p, per, summary, completion = (
            read(run / f)
            for f in ("protocol.json", "per_shot.json", "summary.json", "completed.json")
        )
        # These producers use explicit terminal diagnostic states, not PASS.
        # Require each run's observed schema instead of accepting arbitrary
        # completed-looking strings or treating missing status as success.
        assert completion["status"] == EXPECTED_STATUS[name], name
        for filename in ("protocol.json", "per_shot.json", "summary.json", "completed.json"):
            input_hashes[str(run / filename)] = sha(run / filename)
        order = [r["shot"] for r in p["order"]]
        assert len(order) == 360 and set(order) == fixed, name
        for rows in per.values():
            assert [r["shot"] for r in rows] == order, name
        rescored = {arm: [] for arm in per}
        for index, sid in enumerate(order, 1):
            truth = east[sid]
            stat = Path(truth["path"]).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (truth["size"], truth["mtime_ns"])
            assert io23.read_outcome("EAST", sid) == (truth["disruptive"], truth["down"])
            prefix = run / "shots" / f"{index:04d}_{sid}"
            curve = prefix.with_suffix(".npz")
            seal = read(prefix.with_name(prefix.name + "_seal.json"))
            assert seal["shot"] == sid
            assert seal["current_outcome_used_in_prediction"] is False
            assert sha(curve) == seal["curve_sha256"], (name, sid)
            with np.load(curve, allow_pickle=False) as values:
                expected_time = truth["start"] + np.arange(truth["samples"]) / truth["rate"]
                assert np.array_equal(values["times"], expected_time), (name, sid)
                for arm, old_rows in per.items():
                    old = old_rows[index - 1]
                    new = dict(
                        shot=sid,
                        disruptive=truth["disruptive"],
                        **score_curve(values["times"], values[arm], truth["disruptive"], truth["down"], p["threshold"]),
                    )
                    assert all(old[k] == v for k, v in new.items()), (name, sid, arm)
                    if "down_s" in old:
                        assert old["down_s"] == truth["down"], (name, sid, arm)
                    rescored[arm].append(new)
        metrics = {}
        for arm, rows in rescored.items():
            nd = sum(r["disruptive"] for r in rows)
            ns = len(rows) - nd
            tp = sum(r["timely"] for r in rows)
            fp = sum(r["alarm"] and not r["disruptive"] for r in rows)
            metric = dict(
                shots=len(rows), n_disruptive=nd, n_safe=ns, tp=tp, fp=fp,
                tpr=tp / nd, fpr=fp / ns, J=tp / nd - fp / ns,
                early_over3000=sum(r["timely"] and r["lead_ms"] > 3000 for r in rows),
            )
            assert all(summary[arm]["full360"][k] == v for k, v in metric.items()), (name, arm)
            metrics[arm] = metric
        results[name] = dict(
            shots=360, sealed_prediction_files=360, method_shot_curves=360 * len(per),
            original_order_preserved=True, outcomes_and_down_times_exact=True,
            per_shot_scores_exact=True, summaries_exact=True, metrics=metrics,
        )
        print(name, "PASS", 360 * len(per), flush=True)
    receipt = dict(
        status="PASS", scope="POSTRUN_HDF_OUTCOME_AND_SEALED_CURVE_RECHECK",
        inventory=str(INVENTORY), inventory_completion_sha256=sha(INVENTORY / "completed.json"),
        inventory_metadata_sha256=sha(INVENTORY / "hdf5_metadata.json"), runs=results,
        script_sha256=sha(Path(__file__)), scorer_sha256=sha(Path(score_curve.__code__.co_filename)),
        input_sha256=input_hashes, raw_files_modified=False, historical_results_modified=False,
        fixed_cohort_or_order_changed=False, model_training=False,
        independent_future_validation=False,
        limits="Saved curves are rescored, not model inference or historical training replay; seal flags do not certify real-time deployment.",
    )
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TOTAL", sum(r["method_shot_curves"] for r in results.values()), "PASS")


if __name__ == "__main__":
    main()
