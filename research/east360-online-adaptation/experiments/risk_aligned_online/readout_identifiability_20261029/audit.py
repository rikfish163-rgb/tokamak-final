"""Independent NumPy fitting/counting plus raw-data replay of the diagnostic.

Feature extraction reuses the tested production forward; the ridge solve and
score aggregation below are independent implementations. No classifier fitting.
"""

import argparse
import hashlib
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import collect, baselines, subset

VIEWS = ("all_tokens", "last_token", "raw8")


def numpy_fit(x, y, ids):
    x, y = np.asarray(x, float), np.asarray(y, float)
    counts = {sid: np.sum(ids == sid) for sid in set(ids)}
    w = np.array([1.0 / counts[sid] for sid in ids])
    w /= w.sum()
    mean = np.sum(w[:, None] * x, axis=0)
    constant = np.ptp(x, axis=0) == 0
    mean[constant] = x[0, constant]
    scale = np.sqrt(np.sum(w[:, None] * (x - mean) ** 2, axis=0))
    scale[constant] = 1
    target_mean = np.sum(w[:, None] * y, axis=0)
    a = (x - mean) / scale * np.sqrt(w[:, None])
    b = (y - target_mean) * np.sqrt(w[:, None])
    if len(x) < x.shape[1]:
        coef = a.T @ np.linalg.solve(a @ a.T + 0.01 * np.eye(len(x)), b)
    else:
        coef = np.linalg.solve(a.T @ a + 0.01 * np.eye(x.shape[1]), a.T @ b)
    return dict(mean=mean, scale=scale, target_mean=target_mean, coef=coef)


def fit_predict(train, evaluation, archive=None):
    result = {}
    support = train["mask"].any(0)
    max_coef_error = 0.0
    for view in VIEWS:
        out = np.full((len(evaluation["ids"]), 51, 23), np.nan)
        for c in np.flatnonzero(support):
            take = train["mask"][:, c]
            current = train["current"][take, c : c + 1]
            future = train["future"][take, :, c]
            target = np.concatenate([current, future, future - current], axis=1)
            f = numpy_fit(train[view][take], target, train["ids"][take])
            if archive is not None:
                for name, v in f.items():
                    old = archive[f"{view}_{c}_{name}"]
                    np.testing.assert_allclose(v, old, rtol=1e-7, atol=1e-8)
                    if name == "coef":
                        max_coef_error = max(
                            max_coef_error, float(np.max(np.abs(v - old)))
                        )
            out[:, :, c] = ((evaluation[view] - f["mean"]) / f["scale"]) @ f[
                "coef"
            ] + f["target_mean"]
        result[view + "_current"] = out[:, :1]
        result[view + "_absolute"] = out[:, 1:26]
        result[view + "_increment"] = out[:, 26:] + evaluation["current"][:, None]
    return result, support, max_coef_error


def recount(data):
    summaries = {}
    for key in data:
        if not key.startswith("prediction_"):
            continue
        name = key.removeprefix("prediction_")
        truth = (
            data["current"][:, None] if name.endswith("_current") else data["future"]
        )
        pred = data[key]
        available = np.broadcast_to(data["mask"][:, None], truth.shape)
        if pred.shape != truth.shape or not np.isfinite(pred[available]).all():
            raise ValueError("invalid supported prediction")
        err = np.zeros(truth.shape, float)
        err[available] = pred[available].astype(float) - truth[available].astype(float)
        losses = dict(
            mse=err * err,
            smooth_l1=np.where(np.abs(err) < 1, err * err / 2, np.abs(err) - 0.5),
        )
        rows = []
        for sid in np.unique(data["ids"]):
            index = np.flatnonzero(data["ids"] == sid)
            windows = []
            for j in index:
                n = int(available[j].sum())
                if n:
                    windows.append(
                        {k: float(v[j].sum() / n) for k, v in losses.items()}
                    )
            if windows:
                rows.append(
                    dict(
                        shot=int(sid),
                        windows=len(windows),
                        **{k: float(np.mean([r[k] for r in windows])) for k in losses},
                    )
                )
        if not rows:
            raise ValueError("no supported observations")
        counts = available.sum((0, 1))
        channel_sums = {k: v.sum((0, 1)) for k, v in losses.items()}
        summaries[name] = dict(
            shots=len(rows),
            windows=sum(r["windows"] for r in rows),
            per_shot=rows,
            channel_counts=counts.tolist(),
            channel_sums={k: v.tolist() for k, v in channel_sums.items()},
            channel_mean={
                k: [float(v[c] / counts[c]) if counts[c] else None for c in range(23)]
                for k, v in channel_sums.items()
            },
            **{k: float(np.mean([r[k] for r in rows])) for k in losses},
            smooth_l1_shot_sd=float(np.std([r["smooth_l1"] for r in rows], ddof=1))
            if len(rows) > 1
            else None,
        )
    if "hold" in summaries:
        hold = {r["shot"]: r["smooth_l1"] for r in summaries["hold"]["per_shot"]}
        for name, r in summaries.items():
            if name.endswith("_current"):
                continue  # Reconstruction is not the same target as future prediction.
            delta = np.array([v["smooth_l1"] - hold[v["shot"]] for v in r["per_shot"]])
            r.update(
                beats_hold_shots=int((delta < 0).sum()),
                delta_hold_mean=float(delta.mean()),
                delta_hold_shot_sd=float(delta.std(ddof=1)) if len(delta) > 1 else None,
            )
    return summaries


def compare_scores(recounted, recorded):
    assert set(recounted) == set(recorded)
    for name, r in recounted.items():
        q = recorded[name]
        assert (r["shots"], r["windows"]) == (q["shots"], q["windows"])
        for key in ("smooth_l1", "mse"):
            np.testing.assert_allclose(r[key], q[key], rtol=2e-7, atol=1e-9)
            np.testing.assert_allclose(
                r["channel_sums"][key], q["channel_sums"][key], rtol=2e-7, atol=1e-7
            )
        np.testing.assert_array_equal(r["channel_counts"], q["channel_counts"])


def compare_predictions(data, preds, mask, saved):
    for k in ("ids", "clocks", "current", "future"):
        np.testing.assert_array_equal(data[k], saved[k])
    np.testing.assert_array_equal(mask, saved["mask"])
    errors = []
    assert set(preds) == {
        k.removeprefix("prediction_") for k in saved if k.startswith("prediction_")
    }
    for name, p in preds.items():
        q = saved["prediction_" + name]
        np.testing.assert_allclose(p, q, rtol=1e-7, atol=1e-7, equal_nan=True)
        valid = np.isfinite(p)
        if valid.any():
            errors.append(float(np.max(np.abs(p[valid] - q[valid]))))
    return max(errors)


def audit(root):
    start = time.monotonic()
    p, final, results = [
        read(root, n) for n in ("protocol.json", "complete.json", "results.json")
    ]
    assert final["complete"] and not p["test_or_suffix_access"]
    for name, h in final["artifact_sha256"].items():
        assert sha(root / name) == h, name
    assert sha(DEFAULT_PARENT) == p["source_sha256"]
    for name, h in p["reference_hashes"].items():
        assert sha(Path(p["reference"]) / name) == h
    with zipfile.ZipFile(root / "code_snapshot.zip") as z:
        for name, h in p["code_sha256"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == h
            assert sha(io23.ROOT / name) == h
        assert (
            hashlib.sha256(z.read("READOUT_IDENTIFIABILITY_PLAN.md")).hexdigest()
            == p["plan_sha256"]
        )
    roles = simple_data.source_roles()
    assert p["train_ids"] == roles["train"] and p["val_ids"] == roles["val"]
    assert not set(p["train_ids"] + p["val_ids"]) & set(roles["test"])
    prev = Path(p["previous"])
    assert sha(prev / "complete.json") == p["previous_complete_sha256"]
    pp = read(prev, "protocol.json")
    assert p["east_ids"] == pp["order"][:20]
    source = Path(pp["source_root"])
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    heads = {}
    for name, path in dict(
        source_forecast=source / "forecast_head.pt", cold_forecast=prev / "cold_head.pt"
    ).items():
        assert sha(path) == p["heads"][name]
        heads[name] = torch.load(path, map_location="cuda", weights_only=True)
    train, tr = collect(
        net, p["train_ids"], "J-TEXT", dict(source_forecast=heads["source_forecast"])
    )
    val, vr = collect(
        net, p["val_ids"], "J-TEXT", dict(source_forecast=heads["source_forecast"])
    )
    assert dict(train=tr, val=vr) == read(root, "source_sampling.json")
    with np.load(root / "source_readouts.npz", allow_pickle=False) as weights:
        vpred, support, source_error = fit_predict(train, val, weights)
    vpred = dict(baselines(val), **vpred)
    with np.load(root / "source_val_predictions.npz", allow_pickle=False) as saved:
        source_prediction_error = compare_predictions(
            val, vpred, val["mask"] & support[None], saved
        )
        source_scores = recount(saved)
    compare_scores(source_scores, results["source_val"])
    del train, val, vpred
    print(
        dict(source_fit_and_validation_replayed=True, seconds=time.monotonic() - start),
        flush=True,
    )
    slow = torch.load(
        Path(p["reference"]) / "models/cold.pt", map_location="cuda", weights_only=True
    )["slow"]
    cold, cr = collect(
        net,
        p["east_ids"],
        "EAST",
        heads,
        read(Path(p["reference"]), "input_map.json"),
        slow,
    )
    assert cr == read(root, "cold_sampling.json")
    folds = read(root, "cold_fold_fits.json")
    pooled = {k: v.copy().astype(float) for k, v in baselines(cold).items()}
    mask = cold["mask"].copy()
    within_fold_scores = []
    for i in range(5):
        held_ids = p["east_ids"][i::5]
        assert folds[i]["held_ids"] == held_ids
        assert folds[i]["fit_ids"] == [s for s in p["east_ids"] if s not in held_ids]
        held = np.isin(cold["ids"], held_ids)
        # Score the exact same 16-shot fit on its training and held-out rows.
        # This distinguishes train/held-out error from the all20/16 size change.
        full_out, support, _ = fit_predict(subset(cold, ~held), cold)
        out = {k: v[held] for k, v in full_out.items()}
        training_data = subset(cold, ~held)
        training_data["mask"] &= support[None]
        training_data.update({"prediction_" + k: v[~held] for k, v in full_out.items()})
        within_fold_scores.append(
            dict(fold=i, fit_ids=folds[i]["fit_ids"], scores=recount(training_data))
        )
        assert folds[i]["unestimable_channels"] == np.flatnonzero(~support).tolist()
        mask[held] &= support[None]
        for name, v in out.items():
            if name not in pooled:
                pooled[name] = np.full((len(cold["ids"]), *v.shape[1:]), np.nan)
            pooled[name][held] = v
    with np.load(root / "cold_oof_predictions.npz", allow_pickle=False) as saved:
        cold_oof_prediction_error = compare_predictions(cold, pooled, mask, saved)
        cold_oof_scores = recount(saved)
    compare_scores(cold_oof_scores, results["cold_readout_oof"])
    with np.load(root / "cold_readouts.npz", allow_pickle=False) as weights:
        out, support, cold_error = fit_predict(cold, cold, weights)
    with np.load(root / "cold_train_predictions.npz", allow_pickle=False) as saved:
        cold_train_prediction_error = compare_predictions(
            cold, dict(baselines(cold), **out), cold["mask"] & support[None], saved
        )
        cold_train_scores = recount(saved)
        common_mask_data = {k: saved[k] for k in saved.files}
        common_mask_data["mask"] = mask
        cold_train_common_mask = recount(common_mask_data)
    compare_scores(cold_train_scores, results["cold_train"])
    record = dict(
        complete=True,
        complete_sha256=sha(root / "complete.json"),
        source_train_count=len(tr["receipts"]),
        source_val_count=len(vr["receipts"]),
        source_train_excluded=len(tr["exclusions"]),
        source_val_excluded=len(vr["exclusions"]),
        east_count=len(cr["receipts"]),
        test_and_suffix_reads=0,
        outcome_reads=0,
        independent_numpy_source_coef_max_error=source_error,
        independent_numpy_cold_coef_max_error=cold_error,
        source_prediction_max_error=source_prediction_error,
        cold_oof_prediction_max_error=cold_oof_prediction_error,
        cold_train_prediction_max_error=cold_train_prediction_error,
        seconds=time.monotonic() - start,
        source_val=source_scores,
        cold_readout_oof=cold_oof_scores,
        cold_train=cold_train_scores,
        cold_train_common_mask=cold_train_common_mask,
        within_fold_training=within_fold_scores,
        limitations="conditional auxiliary OOF only; shared production feature forward; source TRAIN score not independently replayed",
        scientific_goal_achieved=False,
        audit_code_sha256=sha(Path(__file__)),
    )
    _durable_json(root / "audit_with_fold_gap.json", record)
    print(
        {
            k: v
            for k, v in record.items()
            if k
            not in (
                "source_val",
                "cold_readout_oof",
                "cold_train",
                "cold_train_common_mask",
                "within_fold_training",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    audit(parser.parse_args().root)
