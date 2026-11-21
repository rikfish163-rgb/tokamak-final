"""Fit offline representation diagnostics on a sealed, producer-validated bank.

This is not an online risk head. Temporal causality of stored features is a
bank-producer contract; this schema does not contain their source timestamps.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


THRESHOLD = 0.7258946895599365
HARD_GAP = 0.25
L2 = 0.01  # Fixed before fitting; intercept is not penalized.
KEYS = ("shot", "rank", "query", "positive", "lead_ms", "native_logit", "state", "history")


def validate_bank(bank):
    missing = set(KEYS) - set(bank)
    if missing:
        raise ValueError(f"missing bank fields: {sorted(missing)}")
    bank = {key: np.asarray(bank[key]) for key in KEYS}
    n = bank["shot"].size
    for key in KEYS[:6]:
        if bank[key].shape != (n,):
            raise ValueError(f"{key} must have shape (N,)")
    for key in ("shot", "rank", "query"):
        if not np.issubdtype(bank[key].dtype, np.integer):
            raise ValueError(f"{key} must be an integer array")
    expected = {"positive": np.bool_, "lead_ms": np.float64, "native_logit": np.float32,
                "state": np.float32, "history": np.float32}
    for key, dtype in expected.items():
        if bank[key].dtype != np.dtype(dtype):
            raise ValueError(f"{key} must have dtype {np.dtype(dtype)}")
    if bank["state"].shape != (n, 66) or bank["history"].shape != (n, 128):
        raise ValueError("state/history must have shapes (N,66)/(N,128)")
    if not n or np.any((bank["rank"] < 1) | (bank["rank"] > 360)):
        raise ValueError("bank must be nonempty and ranks must be in 1..360")
    shots, shot_group = np.unique(bank["shot"], return_inverse=True)
    ranks, rank_group = np.unique(bank["rank"], return_inverse=True)
    pairs = np.unique(np.column_stack((shot_group, rank_group)), axis=0)
    if len(pairs) != len(shots) or len(pairs) != len(ranks):
        raise ValueError("shots and ranks must have a one-to-one mapping")
    for key in ("native_logit", "state", "history"):
        if not np.isfinite(bank[key]).all():
            raise ValueError(f"{key} must be finite")
    positive, lead = bank["positive"], bank["lead_ms"]
    if not (np.isnan(lead[~positive]).all() and
            np.all(np.isfinite(lead[positive]) & (lead[positive] > 30) & (lead[positive] <= 125))):
        raise ValueError("safe lead must be NaN and positive lead must satisfy 30 < lead <= 125 ms")
    return bank


def shot_weights(shots):
    """Equal total weight per shot and equal weight per row within each shot."""
    _, inverse, counts = np.unique(shots, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def class_shot_weights(shots, positive):
    """Half weight per class, then equal class-specific shots and their rows."""
    weights = np.zeros(len(shots), dtype=np.float64)
    for label in (False, True):
        mask = positive == label
        if not mask.any():
            raise ValueError("training requires both classes")
        weights[mask] = 0.5 * shot_weights(shots[mask])
    return weights


def shot_auc(shots, positive, scores):
    """Pairwise AUC with equal shots within each class and half credit for ties."""
    if not positive.any() or positive.all():
        return None
    weights = class_shot_weights(shots, positive)
    _, groups = np.unique(scores, return_inverse=True)
    pos = np.bincount(groups[positive], weights=weights[positive], minlength=groups.max() + 1)
    neg = np.bincount(groups[~positive], weights=weights[~positive], minlength=len(pos))
    return float(np.dot(pos, np.cumsum(neg) - 0.5 * neg) / (pos.sum() * neg.sum()))


def slice_masks(positive, native_logit):
    probability = expit(native_logit.astype(np.float64))
    hard_positive = positive & (probability <= THRESHOLD - HARD_GAP)
    hard_safe = ~positive & (probability >= THRESHOLD)
    return {"overall": np.ones(len(positive), dtype=bool), "positive": positive,
            "safe": ~positive, "hard_positive": hard_positive, "hard_safe": hard_safe,
            "hard": hard_positive | hard_safe}


def fit_probe(features, positive, shots):
    x = np.asarray(features, dtype=np.float64)
    y = positive.astype(np.float64)
    weights = class_shot_weights(shots, positive)
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2))
    scale[scale < 1e-12] = 1.0
    x = (x - mean) / scale

    def objective(parameters):
        coef, intercept = parameters[:-1], parameters[-1]
        logits = x @ coef + intercept
        loss = np.dot(weights, np.logaddexp(0, logits) - y * logits) + 0.5 * L2 * np.dot(coef, coef)
        residual = weights * (expit(logits) - y)
        return loss, np.r_[x.T @ residual + L2 * coef, residual.sum()]

    result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"logistic optimization failed: {result.message}")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficient": result.x[:-1].tolist(),
            "intercept": float(result.x[-1]), "objective": float(result.fun)}


def split_info(bank, mask):
    shot, rank = bank["shot"][mask], bank["rank"][mask]
    ranks, first = np.unique(rank, return_index=True)
    positive = bank["positive"][mask]
    positives = int(positive.sum())
    return {"n_rows": int(mask.sum()), "n_shots": len(ranks), "n_positive": positives,
            "n_safe": int(mask.sum()) - positives, "n_positive_shots": int(np.unique(shot[positive]).size),
            "n_safe_shots": int(np.unique(shot[~positive]).size), "row_indices": np.flatnonzero(mask).tolist(),
            "shots": [{"shot": int(shot[i]), "rank": int(r)} for r, i in zip(ranks, first)]}


def evaluate(shots, positive, native_logit, logits):
    bce = np.logaddexp(0, logits) - positive * logits
    metrics = {}
    for name, mask in slice_masks(positive, native_logit).items():
        metrics[name] = {"n_rows": int(mask.sum()), "n_shots": int(np.unique(shots[mask]).size),
                         "bce": float(shot_weights(shots[mask]) @ bce[mask]) if mask.any() else None}
    metrics["overall"]["auc"] = shot_auc(shots, positive, logits)
    per_shot = [{"shot": int(shot), "n_rows": int((shots == shot).sum()),
                 "bce": float(bce[shots == shot].mean())} for shot in np.unique(shots)]
    predictions = [{"logit": float(z), "probability": float(p), "bce": float(loss)}
                   for z, p, loss in zip(logits, expit(logits), bce)]
    return {"metrics": metrics, "per_shot": per_shot, "predictions": predictions}


def diagnose(bank, bank_sha256):
    """Return a JSON-serializable diagnostic; only the supplied bank is read."""
    bank = validate_bank(bank)
    shot, rank, positive = bank["shot"], bank["rank"], bank["positive"]
    splits = {"train_le_20": rank <= 20, "train_le_100": rank <= 100, "test_gt_100": rank > 100}
    if not splits["test_gt_100"].any():
        raise ValueError("test suffix rank > 100 is empty")
    for name in ("train_le_20", "train_le_100"):
        class_shot_weights(shot[splits[name]], positive[splits[name]])
    test = splits["test_gt_100"]
    native = bank["native_logit"][test]
    masks = slice_masks(positive[test], native)
    test_rows = [{"row_index": int(i), "shot": int(shot[i]), "rank": int(rank[i]),
                  "query": int(bank["query"][i]), "positive": bool(positive[i]),
                  "lead_ms": float(bank["lead_ms"][i]) if positive[i] else None,
                  "native_logit": float(bank["native_logit"][i]), "native_probability": float(p),
                  "hard_positive": bool(hp), "hard_safe": bool(hs)}
                 for i, p, hp, hs in zip(np.flatnonzero(test), expit(native.astype(np.float64)),
                                         masks["hard_positive"], masks["hard_safe"])]
    families = {"clock_logit": bank["state"][:, :2], "state": bank["state"],
                "state_history": np.concatenate((bank["state"], bank["history"]), axis=1)}
    models = {}
    for budget in ("train_le_20", "train_le_100"):
        train = splits[budget]
        models[budget] = {}
        for family, features in families.items():
            fit = fit_probe(features[train], positive[train], shot[train])
            x_test = (features[test].astype(np.float64) - fit["mean"]) / fit["scale"]
            logits = x_test @ fit["coefficient"] + fit["intercept"]
            models[budget][family] = {"fit": fit, **evaluate(shot[test], positive[test], native, logits)}
    return {"schema": "latent_history_probe_20261105/v1", "bank_sha256": bank_sha256,
            "description": "Offline selected-window representation diagnostic; not full-online statistics or an online risk head. HDF5 outcome provenance and feature causality are producer contracts, not verifiable from this schema.",
            "settings": {"threshold": THRESHOLD, "hard_positive_gap": HARD_GAP, "l2": L2,
                         "selection_contract": "bank producer selects geometrically capped timely positives and time-band safe maxima",
                         "training_weights": "equal classes, then shots within class, then rows within shot/class",
                         "bce_weights": "equal shots within slice, then rows within shot/slice",
                         "auc_weights": "equal shots within class, then rows within shot/class; ties half credit",
                         "scaling": "class/shot-weighted training mean and standard deviation only",
                         "objective": "weighted mean BCE + (l2/2)*squared coefficient norm; intercept unpenalized",
                         "prediction_alignment": "each model's predictions[i] corresponds to test_rows[i]"},
            "splits": {name: split_info(bank, mask) for name, mask in splits.items()},
            "test_rows": test_rows, "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path, help="sealed offline .npz bank")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report (will not overwrite)")
    args = parser.parse_args()
    raw = args.bank.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        bank = {key: archive[key] for key in KEYS if key in archive.files}
    report = diagnose(bank, hashlib.sha256(raw).hexdigest())
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(encoded)


if __name__ == "__main__":
    main()
