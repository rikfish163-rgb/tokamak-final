"""Independent local BCE, fold-access, trace and file-binding recount."""

import json
from pathlib import Path

import numpy as np

from ..history_gradient_metric_20261102.run import ROOT, read, sha, write


def shot_mean(rows, key):
    ids = {r["shot"] for r in rows}
    return float(
        np.mean([np.mean([r[key] for r in rows if r["shot"] == sid]) for sid in ids])
    )


def main():
    root = ROOT / "probe_outputs/risk_bridge_forecaster_local20_v1"
    protocol, done = read(root / "protocol.json"), read(root / "completed.json")
    assert sha(root / "records.json") == done["records_sha256"]
    for filename, expected in protocol["code_sha256"].items():
        assert sha(ROOT / filename) == expected, filename
    for filename, expected in done["fold_files_sha256"].items():
        assert sha(root / filename) == expected, filename
    source = ROOT / "probe_outputs/difference_feedback_probe20_v1/samples.json"
    assert sha(source) == protocol["source_samples_sha256"]
    old = {(r["shot"], r["query"]): r for r in read(source)}
    rows = read(root / "records.json")
    assert len(rows) == 100 and {(r["shot"], r["query"]) for r in rows} == set(old)
    for fold in protocol["folds"]:
        assert len(fold["fit"]) == 15 and len(fold["held"]) == 5
        assert not set(fold["fit"]) & set(fold["held"])
        history = read(root / f"fold{fold['fold']}_history.json")
        assert len(history) == 128
        for step, item in enumerate(history, 1):
            assert item["step"] == step and [r["label"] for r in item["samples"]] == [
                0,
                1,
            ]
            assert all(
                r["shot"] in fold["fit"] and r["shot"] not in fold["held"]
                for r in item["samples"]
            )
        for head in read(root / f"fold{fold['fold']}_heads.json").values():
            table = np.asarray(head)
            assert table.shape == (25, 23) and np.isfinite(table).all()
    by_arm = {arm: [] for arm in rows[0]["results"]}
    operational = {arm: [] for arm in by_arm}
    max_original_gap = 0.0
    max_step = 0.0
    for row in rows:
        previous = old[(row["shot"], row["query"])]
        assert (
            row["label"] == previous["label"]
            and row["disruptive"] == previous["disruptive"]
        )
        assert row["shot"] in protocol["folds"][row["fold"]]["held"]
        reference = previous["results"]["difference_real"]["before_logit"]
        np.testing.assert_allclose(row["before_logit"], reference, atol=0, rtol=0)
        before = float(np.logaddexp(0, (1 - 2 * row["label"]) * row["before_logit"]))
        np.testing.assert_allclose(row["before_bce"], before, atol=1e-12, rtol=0)
        # Post-hoc operational-direction surrogate: do NOT change the stored
        # 125ms labels, original gates, or call this a first-alarm metric.
        if row["disruptive"]:
            assert previous["actual_lead_ms"] > 30
        shot_label = int(row["disruptive"])
        op_before = float(np.logaddexp(0, (1 - 2 * shot_label) * row["before_logit"]))
        for arm, value in row["results"].items():
            after = float(
                np.logaddexp(0, (1 - 2 * row["label"]) * value["after_logit"])
            )
            delta = after - before
            np.testing.assert_allclose(value["after_bce"], after, atol=1e-12, rtol=0)
            np.testing.assert_allclose(value["delta_bce"], delta, atol=1e-12, rtol=0)
            state = np.asarray(value["receipt"]["state256_after"])
            step = float(np.linalg.norm(state))
            max_step = max(max_step, step)
            assert step <= 0.100001 and value["query_action_rms"] <= 0.250001
            if arm.endswith("hold"):
                assert step == 0 and delta == 0 and not value["receipt"]["accepted"]
            if arm in ("original_real", "original_reverse"):
                original_key = "difference_" + arm.split("_")[-1]
                gap = abs(delta - previous["results"][original_key]["delta_bce"])
                max_original_gap = max(max_original_gap, gap)
                assert gap < 2e-6
            by_arm[arm].append(
                dict(
                    shot=row["shot"],
                    label=row["label"],
                    disruptive=row["disruptive"],
                    delta=delta,
                )
            )
            op_after = float(
                np.logaddexp(0, (1 - 2 * shot_label) * value["after_logit"])
            )
            operational[arm].append(
                dict(
                    shot=row["shot"],
                    disruptive=row["disruptive"],
                    delta=op_after - op_before,
                )
            )
    for arm, data in by_arm.items():
        means = {}
        for name in ("class0", "class1", "safe"):
            group = [
                r
                for r in data
                if (
                    not r["disruptive"]
                    if name == "safe"
                    else r["label"] == int(name[-1])
                )
            ]
            means[name] = shot_mean(group, "delta")
            np.testing.assert_allclose(
                done["summary"][arm][name]["mean_delta_bce"],
                means[name],
                atol=1e-12,
                rtol=0,
            )
        np.testing.assert_allclose(
            done["summary"][arm]["balanced_delta_bce"],
            (means["class0"] + means["class1"]) / 2,
            atol=1e-12,
            rtol=0,
        )
    meta = done["summary"]["meta_real"]
    gates = {k: meta[k]["mean_delta_bce"] < 0 for k in ("class0", "class1", "safe")}
    gates.update(
        {
            "better_than_" + arm: meta["balanced_delta_bce"]
            < done["summary"][arm]["balanced_delta_bce"]
            for arm in ("original_real", "forecast_real", "meta_reverse")
        }
    )
    gates["coverage"] = True
    gates["passed"] = all(gates.values())
    assert gates == done["gates"] and done["ready_for_full_stream"] == gates["passed"]
    posthoc = {}
    for arm, data in operational.items():
        d = shot_mean([r for r in data if r["disruptive"]], "delta")
        s = shot_mean([r for r in data if not r["disruptive"]], "delta")
        posthoc[arm] = dict(
            disruptive_shot_mean=d, safe_shot_mean=s, balanced=(d + s) / 2
        )
    result = dict(
        status="PASS",
        scope="independent saved-record BCE/fold/hash audit, not independent neural training replay",
        records=100,
        bce_pairs=900,
        fit_steps_checked=512,
        maximum_original_delta_bce_gap=max_original_gap,
        maximum_parameter_step=max_step,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
    )
    write(root / "verification_records.json", result)
    write(
        root / "posthoc_operational_direction.json",
        dict(
            scope="POSTHOC_POINT_SURROGATE_NOT_TPR_OR_GATE_CHANGE",
            summary=posthoc,
            source_completed_sha256=sha(root / "completed.json"),
            verifier_sha256=sha(Path(__file__)),
        ),
    )
    print(json.dumps(dict(verification=result, posthoc=posthoc)), flush=True)


if __name__ == "__main__":
    main()
