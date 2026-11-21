"""CPU record audit and context diagnostics; not a neural reimplementation."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..history_gradient_metric_20261102.run import read, sha, write


def audit(root):
    root = Path(root)
    done, p, records = [
        read(root / n) for n in ("completed.json", "protocol.json", "records.json")
    ]
    assert done["status"] == "COMPLETED_LOCAL_NEURAL_SCREEN"
    assert sha(root / "records.json") == done["records_sha256"]
    assert sha(root / "protocol.json") == done["protocol_sha256"]
    assert sha(root / "code_snapshot.zip") == done["code_snapshot_sha256"]
    for path, digest in p["bound_files_sha256"].items():
        assert sha(Path(path)) == digest
    base = Path(__file__).resolve().parents[3]
    assert all(sha(base / path) == digest for path, digest in p["code_sha256"].items())
    assert len(records) == len({(r["shot"], r["query"]) for r in records}) == 100
    assert {r["shot"] for r in records} == set(p["ids"])
    assert len(p["ids"]) == 20
    tasks = tuple(records[0]["results"])
    assert len(tasks) == 9
    bias = np.log(p["threshold"] / (1 - p["threshold"]))
    contexts = {}
    for arm in ("true", "shuffled"):
        q = np.asarray([r["context"][arm] for r in records], np.float64)
        assert q.shape == (100, 6) and np.isfinite(q).all() and (q >= 0).all()
        np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-6, rtol=0)
        entropy = -(q * np.log(np.maximum(q, 1e-300))).sum(1)
        contexts[arm] = dict(
            mean_probabilities=q.mean(0).tolist(),
            argmax_counts=np.bincount(q.argmax(1), minlength=6).tolist(),
            entropy_quantiles=np.quantile(entropy, [0.1, 0.5, 0.9]).tolist(),
            std_by_family=q.std(0).tolist(),
            outcome_groups={
                name: q[[r["disruptive"] == flag for r in records]].mean(0).tolist()
                for name, flag in (("D", True), ("S", False))
            },
        )
    for row in records:
        fold = p["folds"][row["fold"]]
        assert row["shot"] in fold["held"] and row["shot"] not in fold["fit"]
        assert set(row["results"]) == set(tasks)
        if row["disruptive"]:
            assert row["actual_lead_ms"] > 30
            assert row["point_label125"] == int(row["actual_lead_ms"] <= 125)
        else:
            assert row["point_label125"] == 0
        signed = -1 if row["disruptive"] else 1
        old = float(np.logaddexp(0.0, signed * (row["before_logit"] - bias)))
        np.testing.assert_allclose(old, row["before_bce"], atol=1e-12, rtol=1e-12)
        for value in row["results"].values():
            loss = float(np.logaddexp(0.0, signed * (value["after_logit"] - bias)))
            np.testing.assert_allclose(
                [loss, loss - old],
                [value["after_bce"], value["delta_bce"]],
                atol=1e-12,
                rtol=1e-12,
            )
            assert 0 <= value["query_rms"] <= 0.250001
            receipt = value["receipt"]
            g, w = [
                np.asarray(receipt[k], np.float64)
                for k in ("gradient256", "state256_after")
            ]
            assert (
                g.shape == w.shape == (256,)
                and np.isfinite(g).all()
                and np.isfinite(w).all()
            )
            np.testing.assert_allclose(
                [np.linalg.norm(g), np.linalg.norm(w)],
                [receipt["gradient_norm"], receipt["step_norm"]],
                atol=1e-14,
                rtol=1e-9,
            )
            assert receipt["accepted"] == (
                receipt["loss_proposal"] < receipt["loss_before"]
            )
            if not receipt["accepted"]:
                assert not w.any()
    diversity = {}
    for fold in p["folds"]:
        n = fold["fold"]
        history, tables = (
            read(root / f"fold{n}_history.json"),
            read(root / f"fold{n}_heads.json"),
        )
        assert len(history) == 128
        for i, row in enumerate(history, 1):
            assert row["step"] == i and len(row["samples"]) == 2
            assert all(
                e["shot"] in fold["fit"] and e["shot"] not in fold["held"]
                for e in row["samples"]
            )
        diversity[str(n)] = {}
        for arm in ("true", "shuffled"):
            b = np.asarray(tables[arm], np.float64)
            assert b.shape == (6, 25, 23) and np.isfinite(b).all()
            b -= b.mean(1, keepdims=True)
            total = (b * b).mean()
            diversity[str(n)][arm] = dict(
                centered_between_fraction=float(
                    ((b - b.mean(0, keepdims=True)) ** 2).mean() / total
                )
                if total
                else None,
                centered_table_rms=float(np.sqrt(total)),
            )
    summary = {}
    for arm in tasks:
        summary[arm] = {}
        for group in ("disruptive", "safe", "danger_window"):
            chosen = [
                r
                for r in records
                if (
                    not r["disruptive"]
                    if group == "safe"
                    else r["disruptive"]
                    and (group == "disruptive" or r["actual_lead_ms"] <= 125)
                )
            ]
            ids = sorted({r["shot"] for r in chosen})
            per_shot = [
                np.mean(
                    [r["results"][arm]["delta_bce"] for r in chosen if r["shot"] == sid]
                )
                for sid in ids
            ]
            summary[arm][group] = dict(
                n_points=len(chosen),
                n_shots=len(ids),
                mean_delta_bce=float(np.mean(per_shot)),
                helpful_points=sum(r["results"][arm]["delta_bce"] < 0 for r in chosen),
                harmful_points=sum(r["results"][arm]["delta_bce"] > 0 for r in chosen),
            )
        summary[arm]["balanced"] = (
            summary[arm]["disruptive"]["mean_delta_bce"]
            + summary[arm]["safe"]["mean_delta_bce"]
        ) / 2
    assert summary == done["summary"]
    true = summary["true_real"]
    gates = {
        g: true[g]["mean_delta_bce"] < -1e-8
        for g in ("disruptive", "safe", "danger_window")
    }
    gates.update(
        {
            "better_than_" + a: true["balanced"] < summary[a]["balanced"] - 1e-8
            for a in ("shared_real", "shuffled_real", "true_reverse", "true_hold")
        }
    )
    gates["passed"] = all(gates.values())
    assert gates == done["gates"] and done["ready_for_full_stream"] == gates["passed"]
    return dict(
        status="PASS",
        cases=100,
        methods_per_case=9,
        folds=4,
        summary=summary,
        gates=gates,
        context=contexts,
        table_diagnostics=diversity,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        scope="Recorded logits, state/gradient norms, fold separation and summaries; no neural rerun or physical-state certification.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output / "verification_records.json"
    if target.exists():
        raise FileExistsError(target)
    result = audit(args.output)
    write(target, result)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "cases", "methods_per_case", "gates")}
        )
    )
