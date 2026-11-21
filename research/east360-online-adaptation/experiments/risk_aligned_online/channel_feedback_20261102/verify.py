"""Recompute saved fit coefficients, KKT, held directions and gates independently.

Does not call either fitted method or its aggregation routine. No raw HDF5 read.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(kind):
    channel = kind == "channel_feedback_metric20_v1"
    path = ROOT / "probe_outputs" / kind
    protocol, done = read(path / "protocol.json"), read(path / "completed.json")
    for filename, expected in protocol["code_sha256"].items():
        assert sha(ROOT / filename) == expected, filename
    if channel:
        source = Path(protocol["source_collection"])
        assert sha(source / "completed.json") == protocol["source_completed_sha256"]
        for name in ("gradients", "records"):
            suffix = ".npz" if name == "gradients" else ".json"
            assert sha(source / (name + suffix)) == protocol[name + "_sha256"]
        rows = read(source / "records.json")
        with np.load(source / "gradients.npz", allow_pickle=False) as stored:
            g, r, obs, reverse = [
                stored[k].astype(np.float64)
                for k in ("real", "task", "observations", "reverse")
            ]
        files = done["artifact_sha256"]
        modes = [
            (g, "held_out_real.json", done["real"]),
            (reverse, "held_out_reverse.json", done["reverse_same_weights"]),
        ]
    else:
        source = Path(protocol["source_probe"])
        assert sha(source / "completed.json") == protocol["source_completed_sha256"]
        assert sha(source / "samples.json") == protocol["samples_sha256"]
        rows = read(source / "samples.json")
        g = np.asarray(
            [x["results"]["difference_real"]["receipt"]["gradient256"] for x in rows]
        )
        r = np.asarray([x["task_gradient256"] for x in rows])
        files = {
            "folds.json": done["folds_sha256"],
            "held_out.json": done["held_out_sha256"],
            "full_fit.json": done["full_fit_sha256"],
        }
        modes = [(g, "held_out.json", done["summary"])]
    for filename, expected in files.items():
        assert sha(path / filename) == expected, filename
    ids = np.asarray([row["shot"] for row in rows])
    folds = read(path / "folds.json")
    assert len(folds) == 20 and len(rows) == 100
    assert {f["held_shot"] for f in folds} == set(ids.tolist())
    kkt_max = 0.0
    coefficients_checked = 0
    for fold in folds:
        chosen = np.flatnonzero(ids != fold["held_shot"])
        fit = fold["fit"]
        assert fit["training_ids"] == sorted(set(ids[chosen].tolist()))
        assert len(fit["training_ids"]) == 19 and len(chosen) == 95
        counts = Counter((rows[i]["label"], rows[i]["shot"]) for i in chosen)
        by_label = {
            y: {rows[i]["shot"] for i in chosen if rows[i]["label"] == y}
            for y in (0, 1)
        }
        sample_weights = np.asarray(
            [
                0.5
                / len(by_label[rows[i]["label"]])
                / counts[(rows[i]["label"], rows[i]["shot"])]
                for i in chosen
            ]
        )
        assert np.isclose(sample_weights.sum(), 1)
        grad, risk = g[chosen], r[chosen]
        original = grad.sum(1) if channel else grad
        risk = risk / np.linalg.norm(risk, axis=1)[:, None]
        active = (
            np.flatnonzero(obs[chosen].sum(0) > 0) if channel else np.arange(g.shape[1])
        )
        if channel:
            assert fit["active_indices"] == active.tolist()
        s = grad[:, active] / np.linalg.norm(original, axis=1).reshape(
            (-1,) + (1,) * (grad.ndim - 1)
        )
        c = len(active)
        if channel:
            a = (
                sum(
                    weight * v @ v.T
                    for weight, v in zip(sample_weights, s, strict=True)
                )
                + np.eye(c) / c
            )
            b = (
                sum(
                    weight * v @ target
                    for weight, v, target in zip(sample_weights, s, risk, strict=True)
                )
                + 1 / c
            )
        else:
            a = np.diag(sample_weights @ (s * s) + 1 / c)
            b = sample_weights @ (s * risk) + 1 / c
        np.testing.assert_allclose(
            fit["quadratic"], a if channel else np.diag(a), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(fit["linear"], b, rtol=0, atol=1e-12)
        w = np.asarray(fit["weights"])[active]
        assert (
            np.isfinite(w).all()
            and w.min() >= 0
            and w.max() <= 4
            and abs(w.sum() - c) < 1e-8
        )
        interior = (w > 1e-7) & (w < 4 - 1e-7)
        assert interior.any()  # All actual fitted instances have free coordinates.
        mu = np.mean((b - a @ w)[interior])
        residual = a @ w - b + mu
        kkt_max = max(kkt_max, float(np.max(np.abs(residual[interior]))))
        assert np.max(np.abs(residual[interior])) < 1e-5
        assert (residual[w <= 1e-7] >= -1e-5).all()
        assert (residual[w >= 4 - 1e-7] <= 1e-5).all()
        prediction = np.einsum("c,ncd->nd", w, s) if channel else s * w
        objective = sample_weights @ np.sum((prediction - risk) ** 2, 1) + np.mean(
            (w - 1) ** 2
        )
        np.testing.assert_allclose(objective, fit["objective_fit"], rtol=0, atol=1e-10)
        assert objective <= fit["objective_identity"] + 1e-10
        coefficients_checked += 1
    fold_for = {f["held_shot"]: f for f in folds}
    n_directions = 0
    for values, filename, summary in modes:
        evaluated = {(x["shot"], x["query"]): x for x in read(path / filename)}
        assert len(evaluated) == len(rows)
        for i, row in enumerate(rows):
            w = np.asarray(fold_for[row["shot"]]["fit"]["weights"])
            old = values[i].sum(0) if channel else values[i]
            new = w @ values[i] if channel else w * values[i]
            item = evaluated[(row["shot"], row["query"])]
            for name, value in (("before_cosine", old), ("after_cosine", new)):
                cosine = np.clip(
                    value @ r[i] / np.linalg.norm(value) / np.linalg.norm(r[i]), -1, 1
                )
                np.testing.assert_allclose(item[name], cosine, rtol=0, atol=1e-12)
                n_directions += 1
        checks = []
        for label, safe, name in (
            (0, False, "class0"),
            (1, False, "class1"),
            (0, True, "safe"),
        ):
            chosen = [
                x
                for x in evaluated.values()
                if x["label"] == label and (not safe or not x["disruptive"])
            ]
            by_shot = {
                sid: [x for x in chosen if x["shot"] == sid]
                for sid in {x["shot"] for x in chosen}
            }
            means = [
                float(
                    np.mean(
                        [
                            np.mean([x[phase + "_cosine"] for x in group])
                            for group in by_shot.values()
                        ]
                    )
                )
                for phase in ("before", "after")
            ]
            for phase, mean in zip(("before", "after"), means, strict=True):
                np.testing.assert_allclose(
                    summary["groups"][name][phase + "_mean_cosine"],
                    mean,
                    rtol=0,
                    atol=1e-12,
                )
            passed = means[1] > 0 and means[1] - means[0] >= (0 if safe else 0.01)
            assert summary["gates"][name] == passed
            checks.append(passed)
        assert summary["gates"]["coverage"] and summary["gates"]["passed"] == all(
            checks
        )
    return dict(
        status="PASS",
        fits=coefficients_checked,
        held_direction_values=n_directions,
        maximum_interior_kkt=kkt_max,
        completed_sha256=sha(path / "completed.json"),
    )


def main():
    result = {
        name: verify(name)
        for name in ("history_gradient_metric20_v1", "channel_feedback_metric20_v1")
    }
    result["verifier_sha256"] = sha(Path(__file__))
    result["scope"] = "SAVED_ARRAY_MATH_AND_HASHES_NOT_INDEPENDENT_NEURAL_REPLAY"
    output = ROOT / "probe_outputs/channel_feedback_metric20_v1/verification_math.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
