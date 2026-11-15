"""CPU source-only context control with common, family-independent minibatches.

Uses the existing hashed cache and frozen feature selector, never raw signals or
the source checkpoint. Archive hashing/decompression includes TEST bytes; TEST
is not used for standardization, training, prediction, evaluation or selection.
Conditional family classification is not event detection or EAST alarm gain.
"""

import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ..event_conditional_feedback_20261102.context import probabilities
from ..event_conditional_feedback_20261102.context_fit import (
    CACHE,
    CACHE_SHA,
    LABELS_SHA,
    SCALE_FLOOR,
    SOURCE_SHA,
    counts,
    evaluate_validation,
    load_cache,
    sha256,
    standardize_train,
    training_targets,
    write_json,
)
from ..event_prefix_probe import balanced_event_weights


SEED, BATCH_SIZE, STEPS = 131, 512, 1024
ARMS = ("true", "shuffled")


def training_design(train):
    """Discard legacy class-dependent sampling; only its event labels are reused."""
    event_weights = balanced_event_weights(train["event_id"])
    sampling = event_weights / event_weights.sum()
    n_events = len(np.unique(train["event_id"]))
    targets, class_counts, class_weights = {}, {}, {}
    for name in ARMS:
        targets[name], _ = training_targets(train, shuffled=name == "shuffled")
        support = np.bincount(targets[name], weights=event_weights, minlength=6)
        if (support <= 0).any():
            raise ValueError("all six families need TRAIN event support")
        class_counts[name] = support
        class_weights[name] = n_events / (6 * support)
    return sampling, targets, class_counts, class_weights


def parameter_sha256(head):
    """Weight then bias, contiguous little-endian float32 bytes (fixed 6x115/6)."""
    digest = hashlib.sha256()
    for value in (head.weight, head.bias):
        digest.update(value.detach().numpy().astype("<f4").tobytes())
    return digest.hexdigest()


def fit_pair(train, val, steps=1024):
    """Return (heads, VAL metrics, pairing evidence); no filesystem writes."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    sampling, targets, class_counts, class_weights = training_design(train)
    x, mean, scale = standardize_train(train)
    xx = torch.from_numpy(x)
    yy = {name: torch.from_numpy(targets[name]) for name in ARMS}
    cw = {
        name: torch.as_tensor(class_weights[name], dtype=torch.float32, device="cpu")
        for name in ARMS
    }
    # One CPU initialization, then exact copies; do not consume any GPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        initial = torch.nn.Linear(115, 6, device="cpu")
    models = {name: copy.deepcopy(initial) for name in ARMS}
    initial_sha = parameter_sha256(initial)
    arms = {}
    for name in ARMS:
        digest = parameter_sha256(models[name])
        if digest != initial_sha:
            raise ValueError("initial parameter copy mismatch")
        arms[name] = dict(
            initial_parameters_sha256=digest,
            same_initial=True,
            event_class_counts=class_counts[name].tolist(),
            class_weights=class_weights[name].tolist(),
            optimizer_steps=steps,
        )
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
        for name, model in models.items()
    }
    rng = np.random.default_rng(SEED)
    schedule = hashlib.sha256()
    arm_schedules = {name: hashlib.sha256() for name in ARMS}
    for step in range(steps):
        # Exactly one draw; the SAME row indices feed both heads, regardless of y.
        idx = rng.choice(len(x), BATCH_SIZE, replace=True, p=sampling)
        idx_bytes = np.asarray(idx, dtype="<i8").tobytes()
        schedule.update(idx_bytes)
        batch_x = xx[idx]
        for name in ARMS:
            arm_schedules[name].update(idx_bytes)
            labels = yy[name][idx]
            losses = torch.nn.functional.cross_entropy(
                models[name](batch_x), labels, reduction="none"
            )
            # Do NOT use CE(weight=..., reduction='mean'), which divides by the
            # realized sum of class weights rather than this fixed batch size.
            loss = (losses * cw[name][labels]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite TRAIN loss")
            if step == 0:
                arms[name]["initial_train_loss"] = float(loss.detach())
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[name].step()
            arms[name]["final_train_loss"] = float(loss.detach())
    heads, metrics = {}, {}
    for name in ARMS:
        model = models[name]
        heads[name] = dict(
            mean=mean.tolist(),
            scale=scale.tolist(),
            weight=model.weight.detach().numpy().tolist(),
            bias=model.bias.detach().numpy().tolist(),
        )
        prediction = probabilities(val["raw"], heads[name]).argmax(1)
        metrics[name] = evaluate_validation(val, prediction)
        arms[name]["schedule_sha256"] = arm_schedules[name].hexdigest()
    evidence = dict(
        paired_minibatches=True,
        schedule_sha256=schedule.hexdigest(),
        schedule_encoding="concatenated per-step TRAIN-relative row indices, little-endian int64 C order",
        sampling_sha256=hashlib.sha256(sampling.astype("<f8").tobytes()).hexdigest(),
        initial_parameters_sha256=initial_sha,
        parameter_hash_encoding="weight then bias, little-endian float32 C order",
        initialization_seed=SEED,
        sampler_seed=SEED,
        batch_size=BATCH_SIZE,
        steps=steps,
        arms=arms,
    )
    return heads, metrics, evidence


def imported_code_hashes():
    """Bind actually imported local Python helpers, including transitive imports.

    No package glob, plan files, checkpoint loads or third-party tree hashing.
    Third-party versions are recorded separately in the protocol.
    """
    root = Path(__file__).resolve().parents[3]
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
                paths.add(path)
    return {str(path): sha256(path) for path in sorted(paths)}


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    splits, cache_bound = load_cache()
    code_bound = imported_code_hashes()
    bound = {**cache_bound, **code_bound}
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        scope="SOURCE_ONLY_COMMON_MINIBATCH_CONDITIONAL_CONTEXT_CONTROL",
        cache_path=str(CACHE),
        feature_cache_sha256=CACHE_SHA,
        source_sha256=SOURCE_SHA,
        labels_sha256=LABELS_SHA,
        source_binding="existing cache protocol metadata; checkpoint not loaded or rehashed",
        bound_files_sha256=bound,
        device="cpu",
        python_version=sys.version,
        numpy_version=np.__version__,
        torch_version=torch.__version__,
        training_counts=counts(splits["train"]),
        validation_counts=counts(splits["val"]),
        fit_role="train",
        hard_outcome_authority="fresh HDF5 IsDisrupt for TRAIN/VAL control strata; cached CSV outcome ignored",
        evaluation_role="val",
        cache_contains_test=True,
        test_usage="archive hashing/decompression includes TEST bytes; role-identity checks only; no TEST fitting/standardization/prediction/evaluation",
        selector="frozen event_conditional_feedback_20261102.context.select_variation_features",
        standardizer="TRAIN equal-event weighted mean/std; max(std,1e-6)",
        standard_deviation_floor=SCALE_FLOOR,
        shuffle="legacy whole-event family shuffle within TRAIN outcome x TRAIN shot-quartile strata",
        sampling="balanced_event_weights(event_id), normalized; no family dependence",
        loss="mean(per-row CE * n_events/(6*event_class_counts[y])); counts computed separately per head",
        heads=list(ARMS),
        head_shape=[115, 6],
        seed=SEED,
        steps=STEPS,
        batch_size=BATCH_SIZE,
        optimizer="AdamW",
        lr=0.003,
        weight_decay=0.001,
        paired_minibatches=True,
        initialization="one CPU nn.Linear, copied weight/bias for both heads",
        selection="fixed final iterate; no epoch/hyperparameter selection",
        teacher="offline whole-record K6; upstream historical source-test exposure unresolved",
        unknown_intervals="excluded, not normal",
        event_detector=False,
        east_gain_measured=False,
        source_parameter_updates=0,
        base_model_parameter_updates=0,
        east_parameter_updates=0,
        east_data_reads=0,
    )
    write_json(output / "protocol.json", protocol)
    heads, metrics, evidence = fit_pair(splits["train"], splits["val"], steps=STEPS)
    write_json(output / "context_heads.json", heads)
    write_json(
        output / "validation.json",
        dict(
            role="val",
            counts=counts(splits["val"]),
            heads=metrics,
            scope="conditional family classification only, not event detection or EAST performance",
            test_evaluated=False,
        ),
    )
    if (
        any(sha256(Path(p)) != digest for p, digest in bound.items())
        or imported_code_hashes() != code_bound
    ):
        raise ValueError("bound inputs/imported code changed during fitting")
    completed = dict(
        status="COMPLETED",
        elapsed_seconds=time.monotonic() - started,
        feature_cache_sha256=CACHE_SHA,
        source_sha256=SOURCE_SHA,
        labels_sha256=LABELS_SHA,
        training_counts=counts(splits["train"]),
        validation_counts=counts(splits["val"]),
        paired_minibatches=True,
        pairing=evidence,
        head_optimizer_steps={name: STEPS for name in ARMS},
        bound_files_sha256=bound,
        source_model_loaded=False,
        source_parameter_updates=0,
        base_model_parameter_updates=0,
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
