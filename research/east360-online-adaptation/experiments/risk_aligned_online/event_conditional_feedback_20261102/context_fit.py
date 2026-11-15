"""CPU-only conditional event-family context heads from an existing cache.

TRAIN fits both heads; VAL is diagnostic only. Hard outcomes are freshly read
from HDF5 for TRAIN/VAL, not trusted from the historical CSV-derived cache.
Archive hashing/decompression
includes TEST bytes, but no TEST features/targets are transformed, fitted,
predicted or scored. Role-identity checks also include TEST shot/event IDs.
This is not an event detector or evidence of EAST alarm improvement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..event_content_20261023.probe import independent_f1
from ..event_prefix_probe import balanced_event_weights, shuffle_event_targets
from .. import io23
from ..io23 import RAW23


CACHE = Path(
    "/media/hetaisheng/044A81D94A81C83E/Fuison/outputs/event_prefix_probe_20261021"
)
CACHE_SHA = "d534fe0b29da5d458d8743e986ea37e36bfd21d967262011b9f87dc2f8e008ae"
SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
LABELS_SHA = "24fa872346d5b269f2a1ff3d750a8d810c1bc66195c47c34bd665c9da00b197c"
PLAN = (
    Path(__file__).resolve().parents[3] / "EVENT_CONDITIONAL_FEEDBACK_PLAN_20261102.md"
)
CACHE_ROWS = 17187
EXPECTED_COUNTS = {
    "train": dict(shots=327, events=401, rows=10583),
    "val": dict(shots=83, events=112, rows=2598),
}
SEED, STEPS, BATCH_SIZE = 131, 1024, 512
SCALE_FLOOR = 1e-6
FIELDS = ("raw", "target", "shot", "endpoint", "event_id", "outcome", "role")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def counts(data):
    return dict(
        shots=len(np.unique(data["shot"])),
        events=len(np.unique(data["event_id"])),
        rows=len(data["target"]),
    )


def select_features(raw):
    """Use the same auxiliary-only selector as online context inference."""
    from .context import select_variation_features

    return select_variation_features(raw)


def split_cache(cache):
    """Select TRAIN/VAL before numerical validation; TEST identity checks only."""
    roles = np.asarray(cache["role"])
    n = len(roles)
    if roles.shape != (n,) or not np.isin(roles, ["train", "val", "test"]).all():
        raise ValueError("unexpected role array")
    if cache["raw"].shape != (n, 115) or any(
        cache[k].shape != (n,) for k in FIELDS[1:]
    ):
        raise ValueError("unaligned cache fields")
    for key in ("shot", "event_id"):
        ids = {
            r: set(cache[key][roles == r].tolist()) for r in ("train", "val", "test")
        }
        if any(
            ids[a] & ids[b]
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError(f"{key} role overlap")
    splits = {}
    for role in ("train", "val"):
        use = roles == role
        data = {k: np.asarray(cache[k][use]).copy() for k in FIELDS[:-1]}
        if not use.any() or not np.isfinite(data["raw"]).all():
            raise ValueError(f"nonempty finite {role} required")
        for key in FIELDS[1:-1]:
            if not np.issubdtype(data[key].dtype, np.integer):
                raise ValueError(f"integer {key} required")
        if not np.isin(data["target"], np.arange(6)).all():
            raise ValueError("six-family target required")
        if not np.isin(data["outcome"], [0, 1]).all():
            raise ValueError("binary source outcome required")
        for eid in np.unique(data["event_id"]):
            rows = data["event_id"] == eid
            if any(
                len(np.unique(data[k][rows])) != 1
                for k in ("target", "shot", "outcome")
            ):
                raise ValueError("inconsistent event metadata")
        splits[role] = data
    return splits


def hdf_authoritative_outcomes(data):
    """Replace only offline control strata, never family targets or features.

    Old cached outcomes remain provenance, not a fallback or agreement gate.
    A missing/invalid HDF raises; no cache value is used instead. Call after
    role splitting so TEST outcomes are not opened by this TRAIN/VAL loader.
    """
    result = dict(data)
    ids = np.unique(data["shot"])
    flags = {int(sid): int(io23.read_outcome("J-TEXT", int(sid))[0]) for sid in ids}
    canonical = np.asarray([flags[int(sid)] for sid in data["shot"]], dtype=np.int64)
    changed = canonical != data["outcome"]
    result["outcome"] = canonical
    return result, dict(
        authority="HDF5 meta/IsDisrupt", rows=len(canonical), shots=len(ids),
        changed_rows=int(changed.sum()),
        changed_shots=sorted(set(map(int, data["shot"][changed]))),
        hdf_outcomes={str(sid): flags[int(sid)] for sid in ids},
    )


def load_cache():
    """Exact MAIN path only; bind source/teacher via cache metadata, not torch.load."""
    completed = read_json(CACHE / "completed.json")
    protocol = read_json(CACHE / "protocol.json")
    if completed.get("status") != "COMPLETE_PROBE_ONLY" or not completed.get(
        "source_file_and_parameters_unchanged"
    ):
        raise ValueError("completed immutable cache required")
    if protocol.get("source_parent_sha256") != SOURCE_SHA:
        raise ValueError("source metadata hash mismatch")
    if protocol.get("labels_sha256") != LABELS_SHA:
        raise ValueError("labels metadata hash mismatch")
    paths = [
        CACHE / name for name in ("features.npz", "protocol.json", "completed.json")
    ]
    bound = {str(p): sha256(p) for p in paths}
    if (
        completed.get("feature_cache_sha256") != CACHE_SHA
        or bound[str(paths[0])] != CACHE_SHA
    ):
        raise ValueError("feature cache hash mismatch")
    with np.load(paths[0], allow_pickle=False) as archive:
        # NPZ members contain every role; only these fields are decompressed.
        cache = {k: archive[k] for k in FIELDS}
    if cache["raw"].shape != (CACHE_ROWS, 115):
        raise ValueError("unexpected full cache shape")
    splits = split_cache(cache)
    if any(counts(splits[r]) != EXPECTED_COUNTS[r] for r in ("train", "val")):
        raise ValueError("TRAIN/VAL counts mismatch")
    splits = {role: hdf_authoritative_outcomes(data)[0] for role, data in splits.items()}
    return splits, bound


def standardize_train(train):
    x = select_features(train["raw"]).astype(np.float64)
    weights = balanced_event_weights(train["event_id"])
    mean = np.average(x, weights=weights, axis=0)
    variance = np.average((x - mean) ** 2, weights=weights, axis=0)
    # Keep the audited probe's std floor, not an unrelated .001 std floor.
    scale = np.maximum(np.sqrt(variance), SCALE_FLOOR)
    return ((x - mean) / scale).astype(np.float32), mean, scale


def training_targets(train, *, shuffled):
    """Same shuffle formula; load_cache supplies HDF-authoritative strata."""
    y = train["target"].copy()
    if shuffled:
        boundaries = np.quantile(np.unique(train["shot"]), [0.25, 0.5, 0.75])
        strata = np.searchsorted(boundaries, train["shot"]) + 4 * train["outcome"]
        y = shuffle_event_targets(y, train["event_id"], strata, seed=SEED)
    weights = balanced_event_weights(train["event_id"])
    support = np.bincount(y, weights=weights, minlength=6)
    if (support <= 0).any():
        raise ValueError("all six families need TRAIN support")
    sampling = weights / support[y]
    sampling /= sampling.sum()
    return y, sampling


def evaluate_validation(val, prediction):
    matrix, f1 = independent_f1(val["target"], prediction, val["event_id"])
    support = matrix.sum(1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(6), where=support > 0)
    return dict(
        event_weighted_macro_f1=f1,
        per_class_recall={chr(65 + i): float(v) for i, v in enumerate(recall)},
        confusion_event_weighted=matrix.tolist(),
    )


def fit_head(train, val, *, shuffled):
    """Fixed final iterate, CPU tensors only; no legacy fit_probe call."""
    x, mean, scale = standardize_train(train)
    y, sampling = training_targets(train, shuffled=shuffled)
    xx, yy = torch.from_numpy(x), torch.from_numpy(y)
    # Seed only the CPU generator, with identical initial weights for both arms.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        head = torch.nn.Linear(115, 6, device="cpu")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.001)
    rng = np.random.default_rng(SEED)
    first_loss = None
    for _ in range(STEPS):
        idx = rng.choice(len(y), BATCH_SIZE, replace=True, p=sampling)
        loss = torch.nn.functional.cross_entropy(head(xx[idx]), yy[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite TRAIN loss")
        if first_loss is None:
            first_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    val_x = ((select_features(val["raw"]).astype(np.float64) - mean) / scale).astype(
        np.float32
    )
    with torch.no_grad():
        prediction = head(torch.from_numpy(val_x)).argmax(-1).numpy()
    artifact = dict(
        mean=mean.tolist(),
        scale=scale.tolist(),
        weight=head.weight.detach().numpy().tolist(),
        bias=head.bias.detach().numpy().tolist(),
    )
    metrics = evaluate_validation(val, prediction)
    metrics.update(
        optimizer_steps=STEPS,
        initialization_seed=SEED,
        sampler_seed=SEED,
        initial_train_loss=first_loss,
        final_train_loss=float(loss.detach()),
        shuffled_endpoint_fraction=float(np.mean(y != train["target"])),
    )
    return artifact, metrics


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    splits, bound = load_cache()
    from .context import select_variation_features

    for function in (
        main,
        balanced_event_weights,
        independent_f1,
        select_variation_features,
    ):
        p = Path(function.__code__.co_filename).resolve()
        bound[str(p)] = sha256(p)
    bound[str(PLAN)] = sha256(PLAN)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        scope="DEVELOPMENT_ONLY_TRAIN_EVENT_FAMILY_CONTEXT",
        cache_path=str(CACHE),
        cache_sha256=CACHE_SHA,
        source_sha256=SOURCE_SHA,
        labels_sha256=LABELS_SHA,
        source_binding="existing cache protocol metadata; checkpoint not loaded or rehashed",
        bound_files_sha256=bound,
        plan_sha256=bound[str(PLAN)],
        device="cpu",
        torch_version=torch.__version__,
        cache_contains_test=True,
        test_usage="archive hashing/decompression includes TEST bytes; role-identity checks only; no TEST transform/training/prediction/metrics",
        fit_role="train",
        hard_outcome_authority="fresh HDF5 IsDisrupt for TRAIN/VAL control strata; cached CSV outcome ignored",
        evaluation_role="val",
        training_counts=counts(splits["train"]),
        validation_counts=counts(splits["val"]),
        raw_order=list(RAW23),
        blocks=["last", "mean", "std", "halfchange", "mask"],
        retained_blocks=[2, 3, 4],
        zero_columns=sorted(
            set(range(46)) | {23 * b + RAW23.index("MNM") for b in range(5)}
        ),
        auxiliary_mask_only=True,
        base_model_unchanged=True,
        standardizer="TRAIN only; equal total weight per event",
        standard_deviation_floor=SCALE_FLOOR,
        sampling="balanced_event_weights / event-weighted class counts, normalized",
        shuffle="old shuffle_event_targets exactly; TRAIN outcome x TRAIN shot quartile",
        heads=["true", "shuffled"],
        head_shape=[115, 6],
        seed=SEED,
        steps=STEPS,
        batch_size=BATCH_SIZE,
        optimizer="AdamW",
        lr=0.003,
        weight_decay=0.001,
        identical_initialization=True,
        sampler_seed_per_head=SEED,
        selection="fixed final iterate; no epoch or hyperparameter selection",
        teacher="offline whole-record K6; upstream historical source-test exposure unresolved",
        unknown_intervals="excluded, not normal",
        event_detector=False,
        east_gain_measured=False,
        source_parameter_updates=0,
        east_parameter_updates=0,
        east_data_reads=0,
    )
    write_json(output / "protocol.json", protocol)
    heads, metrics = {}, {}
    for name in ("true", "shuffled"):
        heads[name], metrics[name] = fit_head(
            splits["train"], splits["val"], shuffled=name == "shuffled"
        )
    write_json(output / "context_heads.json", heads)
    write_json(
        output / "validation.json",
        dict(
            role="val",
            counts=counts(splits["val"]),
            heads=metrics,
            scope="conditional family classification only, not EAST performance",
            test_evaluated=False,
        ),
    )
    if any(sha256(Path(p)) != digest for p, digest in bound.items()):
        raise ValueError("bound inputs/code changed during fitting")
    completed = dict(
        status="COMPLETED",
        elapsed_seconds=time.monotonic() - started,
        feature_cache_sha256=CACHE_SHA,
        source_sha256=SOURCE_SHA,
        labels_sha256=LABELS_SHA,
        training_counts=counts(splits["train"]),
        validation_counts=counts(splits["val"]),
        head_optimizer_steps={name: STEPS for name in heads},
        bound_files_sha256=bound,
        source_model_loaded=False,
        source_parameter_updates=0,
        east_parameter_updates=0,
        east_data_reads=0,
        test_fit_rows=0,
        test_evaluated_rows=0,
        artifact_sha256={
            name: sha256(output / name)
            for name in ("protocol.json", "context_heads.json", "validation.json")
        },
    )
    write_json(output / "completed.json", completed)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
