"""Independent saved-array coefficient/KKT/condition/control audit."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from ..history_gradient_metric_20261102.run import ROOT, read, sha, write


def main():
    output = ROOT / "probe_outputs/risk_conditioned_feedback20_v1"
    parent, done = read(output / "protocol.json"), read(output / "completed.json")
    for name, expected in parent["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in done["artifact_sha256"].items():
        assert sha(output / name) == expected, name
    context = Path(parent["context_collection"])
    assert sha(context / "completed.json") == parent["context_completed_sha256"]
    assert sha(context / "conditions.json") == parent["conditions_sha256"]
    conditions = read(context / "conditions.json")
    source = Path(parent["source_collection"])
    assert sha(source / "completed.json") == parent["source_completed_sha256"]
    assert sha(source / "gradients.npz") == parent["source_gradients_sha256"]
    rows = read(source / "records.json")
    assert [(r["shot"], r["query"]) for r in rows] == [
        (r["shot"], r["query"]) for r in conditions
    ]
    assert all(r["maturity"] == r["query"] - 1 for r in conditions)
    with np.load(source / "gradients.npz", allow_pickle=False) as stored:
        g, r, obs, reverse = [
            stored[k].astype(np.float64)
            for k in ("real", "task", "observations", "reverse")
        ]
    u = np.array([r["condition"] for r in conditions])
    folds = read(output / "folds.json")
    assert len(folds) == 20
    fit_count = 0
    maximum = 0.0
    for fold in folds:
        chosen = [i for i, row in enumerate(rows) if row["shot"] != fold["held_shot"]]
        ids = sorted({rows[i]["shot"] for i in chosen})
        assert len(chosen) == 95 and len(ids) == 19
        active = np.flatnonzero(obs[chosen].sum(0) > 0)
        c = len(active)
        counts = Counter((rows[i]["label"], rows[i]["shot"]) for i in chosen)
        labels = {
            y: {rows[i]["shot"] for i in chosen if rows[i]["label"] == y}
            for y in (0, 1)
        }
        for arm, key in [("actual", "fit"), ("constant", "constant_fit")]:
            fitted = fold[key]
            assert (
                fitted["training_ids"] == ids
                and fitted["active_indices"] == active.tolist()
            )
            a, b = np.eye(2 * c) / (2 * c), np.ones(2 * c) / (2 * c)
            identity = 0.0
            for i in chosen:
                weight = (
                    0.5
                    / len(labels[rows[i]["label"]])
                    / counts[(rows[i]["label"], rows[i]["shot"])]
                )
                normalized = g[i, active] / np.linalg.norm(g[i].sum(0))
                task = r[i] / np.linalg.norm(r[i])
                condition = u[i] if arm == "actual" else 0.5
                design = np.concatenate(
                    ((1 - condition) * normalized, condition * normalized)
                )
                a += weight * (design @ design.T)
                b += weight * (design @ task)
                identity += weight * np.sum((normalized.sum(0) - task) ** 2)
            np.testing.assert_allclose(fitted["quadratic"], a, atol=1e-12, rtol=0)
            np.testing.assert_allclose(fitted["linear"], b, atol=1e-12, rtol=0)
            w = np.asarray(fitted["weights"])[:, active].flatten()
            assert (w >= 0).all() and (w <= 4).all()
            np.testing.assert_allclose(w.reshape(2, c).sum(1), c, atol=1e-8, rtol=0)
            for block in (np.arange(c), np.arange(c, 2 * c)):
                interior = block[(w[block] > 1e-7) & (w[block] < 4 - 1e-7)]
                assert len(interior) > 0
                mu = np.mean((b - a @ w)[interior])
                residual = a @ w - b + mu
                maximum = max(maximum, float(np.max(np.abs(residual[interior]))))
                assert np.max(np.abs(residual[interior])) < 1e-5
                assert (residual[block[w[block] <= 1e-7]] >= -1e-5).all()
                assert (residual[block[w[block] >= 4 - 1e-7]] <= 1e-5).all()
            # Both regularizer sums and ||risk||² sum to one.
            objective = float(w @ a @ w - 2 * b @ w + 2)
            np.testing.assert_allclose(
                fitted["objective_fit"], objective, atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(
                fitted["objective_identity"], identity, atol=1e-10, rtol=0
            )
            assert objective <= identity + 1e-9
            fit_count += 1
    fold_for = {f["held_shot"]: f for f in folds}
    directions = 0
    for mode, values, key in [
        ("real", g, "real"),
        ("reverse", reverse, "reverse_same_weights"),
        ("constant", g, "constant_condition"),
    ]:
        saved = {
            (r["shot"], r["query"]): r
            for r in read(output / ("held_" + mode + ".json"))
        }
        assert len(saved) == 100
        for i, row in enumerate(rows):
            f = fold_for[row["shot"]]
            fitted = f["constant_fit" if mode == "constant" else "fit"]
            condition = 0.5 if mode == "constant" else u[i]
            low, high = np.asarray(fitted["weights"])
            w = (1 - condition) * low + condition * high
            if mode != "constant":
                np.testing.assert_array_equal(
                    saved[(row["shot"], row["query"])]["weights"], w
                )
            for phase, vector in [
                ("before", values[i].sum(0)),
                ("after", w @ values[i]),
            ]:
                cosine = np.clip(
                    vector @ r[i] / np.linalg.norm(vector) / np.linalg.norm(r[i]), -1, 1
                )
                np.testing.assert_allclose(
                    saved[(row["shot"], row["query"])][phase + "_cosine"],
                    cosine,
                    atol=1e-12,
                    rtol=0,
                )
                directions += 1
        conditions_pass = []
        for label, safe, name in [
            (0, False, "class0"),
            (1, False, "class1"),
            (0, True, "safe"),
        ]:
            chosen = [
                v
                for v in saved.values()
                if v["label"] == label and (not safe or not v["disruptive"])
            ]
            by_shot = {
                sid: [v for v in chosen if v["shot"] == sid]
                for sid in {v["shot"] for v in chosen}
            }
            means = [
                float(
                    np.mean(
                        [
                            np.mean([v[p + "_cosine"] for v in group])
                            for group in by_shot.values()
                        ]
                    )
                )
                for p in ("before", "after")
            ]
            for phase, value in zip(("before", "after"), means, strict=True):
                np.testing.assert_allclose(
                    done[key]["groups"][name][phase + "_mean_cosine"],
                    value,
                    atol=1e-12,
                    rtol=0,
                )
            passed = means[1] > 0 and means[1] - means[0] >= (0 if safe else 0.01)
            assert done[key]["gates"][name] == passed
            conditions_pass.append(passed)
        assert done[key]["gates"]["passed"] == all(conditions_pass)
    deltas = [
        done["real"]["groups"][k]["after_mean_cosine"]
        - done["constant_condition"]["groups"][k]["after_mean_cosine"]
        for k in ("class1", "safe")
    ]
    passed = all(v >= 0 for v in deltas) and any(v > 1e-12 for v in deltas)
    assert done["conditioning_gate"] == passed
    assert done["ready_for_neural_probe"] == (
        passed and done["real"]["gates"]["passed"]
    )
    result = dict(
        status="PASS",
        fitted_tables=fit_count,
        held_direction_values=directions,
        maximum_interior_kkt=maximum,
        context_prefix_indices_checked=100,
        source="saved arrays; no independent neural replay",
        completed_sha256=sha(output / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
    )
    write(output / "verification_math.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
