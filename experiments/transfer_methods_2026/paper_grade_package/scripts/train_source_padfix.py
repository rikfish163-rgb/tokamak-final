#!/usr/bin/env python3
"""Train a context-robust J-TEXT source checkpoint without changing inference.

The deployed encoder/head architecture is exactly the existing rebuilt17
Transformer.  Training adds paired causal context views, logit consistency, and
an outcome-conditional context-length adversary.  Every paired view ends at the
same causal endpoint and can only delete older history.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
import rebuilt_harness as RH  # noqa: E402
import train_jtext_source_transformer as BASE  # noqa: E402
import advanced_time_labels as ATL  # noqa: E402


CONTEXT_LENGTHS = (32, 64, 96, 128)
VALIDATION_GATES = {
    "timely_tpr_min": 0.90,
    "fpr_max": 0.15,
    "paired_context_probe_macro_auc_max": 0.65,
    "paired_context_mean_abs_risk_difference_max": 0.05,
    "paired_context_alarm_flip_rate_max": 0.05,
}


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx, tensor: torch.Tensor, weight: float,
    ) -> torch.Tensor:
        ctx.weight = float(weight)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return (-ctx.weight * gradient, None)


def reverse_gradient(tensor: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(tensor, float(weight))


def crop_older_context(
    windows: np.ndarray,
    valid_lengths: np.ndarray,
    target_lengths: np.ndarray,
) -> np.ndarray:
    """Delete only older samples while preserving each causal endpoint."""

    values = np.asarray(windows, dtype=np.float32)
    valid = np.asarray(valid_lengths, dtype=np.int64)
    target = np.asarray(target_lengths, dtype=np.int64)
    if values.ndim != 3 or valid.shape != (len(values),) or target.shape != valid.shape:
        raise ValueError("invalid context crop shapes")
    if np.any(target <= 0) or np.any(target > valid) or np.any(valid > values.shape[1]):
        raise ValueError("target context must be positive and no longer than observed history")
    result = values.copy()
    width = values.shape[1]
    for row, keep in enumerate(target.tolist()):
        result[row, : width - int(keep)] = 0.0
    return np.ascontiguousarray(result)


def context_bucket(lengths: np.ndarray) -> np.ndarray:
    lookup = {length: index for index, length in enumerate(CONTEXT_LENGTHS)}
    try:
        return np.asarray([lookup[int(length)] for length in lengths], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unregistered context length: {error.args[0]}") from error


def paired_representation_consistency(
    full: torch.Tensor, shorter: torch.Tensor,
) -> torch.Tensor:
    if full.shape != shorter.shape:
        raise ValueError("paired representations must have identical shapes")
    return F.smooth_l1_loss(full, shorter)


def validation_selection_key(metrics: dict[str, float]) -> tuple[float, ...]:
    """Rank epochs by preregistered gates before ordinary validation metrics.

    TEST220 remains unread.  If no epoch is admissible, the worst normalized
    validation-gate violation is minimized so a failed screen cannot be hidden
    by a favorable FPR or AUC.
    """

    ratios = (
        max(0.0, VALIDATION_GATES["timely_tpr_min"] - float(metrics["timely_tpr"]))
        / VALIDATION_GATES["timely_tpr_min"],
        max(0.0, float(metrics["fpr"]) - VALIDATION_GATES["fpr_max"])
        / VALIDATION_GATES["fpr_max"],
        max(
            0.0,
            float(metrics["paired_context_probe_macro_auc"])
            - VALIDATION_GATES["paired_context_probe_macro_auc_max"],
        ) / VALIDATION_GATES["paired_context_probe_macro_auc_max"],
        max(
            0.0,
            float(metrics["paired_context_mean_abs_risk_difference"])
            - VALIDATION_GATES["paired_context_mean_abs_risk_difference_max"],
        ) / VALIDATION_GATES["paired_context_mean_abs_risk_difference_max"],
        max(
            0.0,
            float(metrics["paired_context_alarm_flip_rate"])
            - VALIDATION_GATES["paired_context_alarm_flip_rate_max"],
        ) / VALIDATION_GATES["paired_context_alarm_flip_rate_max"],
    )
    admitted = float(max(ratios) <= 0.0)
    return (
        admitted,
        -float(max(ratios)),
        -float(sum(ratios)),
        -float(metrics["fpr"]),
        float(metrics["timely_tpr"]),
        float(metrics["auc"]),
    )


class ConditionalContextAdversary(nn.Module):
    """Separate pad probes prevent the adversary exploiting outcome priors."""

    def __init__(self, hidden: int = RH.D_MODEL) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(hidden, len(CONTEXT_LENGTHS)),
            nn.Linear(hidden, len(CONTEXT_LENGTHS)),
        ])

    def classification_loss(
        self,
        hidden: torch.Tensor,
        outcome: torch.Tensor,
        bucket: torch.Tensor,
    ) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        labels = outcome >= 0.5
        for group, head in enumerate(self.heads):
            selected = labels if group == 1 else ~labels
            if bool(selected.any()):
                terms.append(F.cross_entropy(head(hidden[selected]), bucket[selected]))
        if not terms:
            return hidden.sum() * 0.0
        return torch.stack(terms).mean()

    def confusion_loss(
        self,
        hidden: torch.Tensor,
        outcome: torch.Tensor,
    ) -> torch.Tensor:
        """Make each trained context head uniform without label permutation."""

        terms: list[torch.Tensor] = []
        labels = outcome >= 0.5
        for group, head in enumerate(self.heads):
            selected = labels if group == 1 else ~labels
            if bool(selected.any()):
                log_probability = F.log_softmax(head(hidden[selected]), dim=1)
                terms.append(-log_probability.mean())
        if not terms:
            return hidden.sum() * 0.0
        return torch.stack(terms).mean()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _paired_views(
    windows: np.ndarray,
    valid_lengths: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(np.asarray(valid_lengths) >= max(CONTEXT_LENGTHS))
    if not len(eligible):
        return (
            np.empty((0,) + windows.shape[1:], np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
            eligible,
        )
    choices = np.asarray(CONTEXT_LENGTHS[:-1], np.int64)
    target = choices[rng.integers(0, len(choices), size=len(eligible))]
    cropped = crop_older_context(windows[eligible], valid_lengths[eligible], target)
    return cropped, target, labels[eligible], eligible


def _all_context_views(
    windows: np.ndarray,
    valid_lengths: np.ndarray,
    labels: np.ndarray,
    *,
    maximum_examples: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return all registered lengths for identical eligible endpoints."""

    eligible = np.flatnonzero(np.asarray(valid_lengths) >= max(CONTEXT_LENGTHS))
    eligible = eligible[: int(maximum_examples)]
    if not len(eligible):
        return (
            np.empty((0,) + windows.shape[1:], np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
            np.empty(0, np.int64),
        )
    base = windows[eligible]
    observed = valid_lengths[eligible]
    view_rows: list[np.ndarray] = []
    view_lengths: list[np.ndarray] = []
    view_labels: list[np.ndarray] = []
    view_buckets: list[np.ndarray] = []
    for bucket, context in enumerate(CONTEXT_LENGTHS):
        target = np.full(len(eligible), context, np.int64)
        view_rows.append(crop_older_context(base, observed, target))
        view_lengths.append(target)
        view_labels.append(labels[eligible])
        view_buckets.append(np.full(len(eligible), bucket, np.int64))
    return (
        np.concatenate(view_rows),
        np.concatenate(view_lengths),
        np.concatenate(view_labels),
        np.concatenate(view_buckets),
    )


def _validation_context_audit(
    encoder: nn.Module,
    head: nn.Module,
    train: list[dict],
    validation: list[dict],
    *,
    threshold: float,
    device: str,
) -> dict[str, float]:
    """Probe context identity using same-endpoint views on disjoint shots."""

    def rows(shots: list[dict], limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        windows: list[np.ndarray] = []
        lengths: list[int] = []
        buckets: list[int] = []
        for shot in shots[:limit]:
            if len(shot["v"]) < 128:
                continue
            endpoint = len(shot["v"]) - 1
            full, valid = BASE.causal_window(shot["v"], endpoint=endpoint)
            if valid < 128:
                continue
            for bucket, context in enumerate(CONTEXT_LENGTHS):
                windows.append(
                    crop_older_context(
                        full[None], np.asarray([valid]), np.asarray([context]),
                    )[0]
                )
                lengths.append(context)
                buckets.append(bucket)
        hidden: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        encoder.eval(); head.eval()
        for begin in range(0, len(windows), 512):
            x = torch.from_numpy(np.stack(windows[begin:begin + 512])).to(device)
            n = torch.as_tensor(lengths[begin:begin + 512], device=device)
            with torch.no_grad():
                z = encoder(x, valid_lengths=n)
                hidden.append(z.cpu().numpy())
                probabilities.append(torch.sigmoid(head(z)).cpu().numpy())
        return np.vstack(hidden), np.asarray(buckets), np.concatenate(probabilities)

    train_hidden, train_bucket, _ = rows(train[::3], 480)
    val_hidden, val_bucket, val_probability = rows(validation, len(validation))
    probe = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=2000, solver="lbfgs",
        random_state=20260820,
    ).fit(train_hidden, train_bucket)
    probability = probe.predict_proba(val_hidden)
    prediction = probability.argmax(axis=1)
    grouped = val_probability.reshape(-1, len(CONTEXT_LENGTHS))
    full = grouped[:, -1]
    shorter = grouped[:, :-1]
    differences = np.abs(shorter - full[:, None])
    flips = (shorter >= threshold) != (full[:, None] >= threshold)
    return {
        "paired_context_probe_macro_auc": float(
            roc_auc_score(val_bucket, probability, multi_class="ovr", average="macro")
        ),
        "paired_context_probe_balanced_accuracy": float(
            balanced_accuracy_score(val_bucket, prediction)
        ),
        "paired_context_mean_abs_risk_difference": float(differences.mean()),
        "paired_context_p95_abs_risk_difference": float(np.percentile(differences, 95)),
        "paired_context_alarm_flip_rate": float(flips.mean()),
        "paired_context_validation_shots": int(len(grouped)),
    }


def train(args: argparse.Namespace) -> tuple[Path, dict]:
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

    roles = BASE.load_canonical_jtext_roles(ROOT)
    source_train = BASE.load_jtext_shots(roles["train"])
    validation = BASE.load_jtext_shots(roles["val"])
    advanced_path = Path(args.advanced_time_path).expanduser().resolve()
    advanced = (
        ATL.load_advanced_time_ms(advanced_path)
        if args.label_policy == "advanced_floor" else None
    )
    source_train, label_audit = ATL.attach_positive_horizons(
        source_train, device="J-TEXT", policy=args.label_policy,
        advanced_time_ms=advanced,
    )
    training_label = BASE.source_training_label_contract(
        policy=args.label_policy, audit=label_audit,
        advanced_time_path=advanced_path if advanced is not None else None,
    )
    center, scale = GS.fit_scale(source_train)
    GS.apply_scale(source_train, center, scale)
    GS.apply_scale(validation, center, scale)
    sampler = BASE.BalancedSourceSampler(source_train, seed=args.seed)
    rng = np.random.default_rng(args.seed + 771)

    encoder = GS.Enc(len(RH.REBUILD)).to(device)
    head = GS.Head().to(device)
    adversary = ConditionalContextAdversary().to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    adversary_optimizer = torch.optim.AdamW(
        adversary.parameters(), lr=1e-3, weight_decay=1e-4,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "source_checkpoint.pt"
    history: list[dict] = []
    best_rank = None
    best_epoch = 0
    stale = 0
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train(); head.train(); adversary.train()
        epoch_terms = {name: [] for name in (
            "loss", "task", "view", "consistency", "representation_consistency",
            "adversary", "adversary_confusion",
        )}
        for _ in range(int(args.steps_per_epoch)):
            windows, lengths, labels = sampler.sample_batch(args.batch_size)
            cropped, cropped_lengths, cropped_labels, eligible = _paired_views(
                windows, lengths, labels, rng,
            )
            context_windows, context_lengths, context_labels, context_buckets = (
                _all_context_views(windows, lengths, labels)
            )
            x = torch.from_numpy(windows).to(device)
            n = torch.from_numpy(lengths).to(device)
            y = torch.from_numpy(labels).to(device)
            hidden = encoder(x, valid_lengths=n)
            logits = head(hidden)
            task_loss = F.binary_cross_entropy_with_logits(logits, y)
            view_loss = logits.sum() * 0.0
            consistency = logits.sum() * 0.0
            representation_consistency = logits.sum() * 0.0
            adversary_loss = logits.sum() * 0.0
            adversary_confusion = logits.sum() * 0.0
            if len(eligible):
                x_view = torch.from_numpy(cropped).to(device)
                n_view = torch.from_numpy(cropped_lengths).to(device)
                y_view = torch.from_numpy(cropped_labels).to(device)
                hidden_view = encoder(x_view, valid_lengths=n_view)
                logits_view = head(hidden_view)
                view_loss = F.binary_cross_entropy_with_logits(logits_view, y_view)
                consistency = F.smooth_l1_loss(logits_view, logits[torch.as_tensor(eligible, device=device)])
                representation_consistency = paired_representation_consistency(
                    hidden_view, hidden[torch.as_tensor(eligible, device=device)],
                )
            if len(context_windows):
                context_hidden = encoder(
                    torch.from_numpy(context_windows).to(device),
                    valid_lengths=torch.from_numpy(context_lengths).to(device),
                )
                context_outcome = torch.from_numpy(context_labels).to(device)
                context_target = torch.from_numpy(context_buckets).to(device)
                adversary_optimizer.zero_grad(set_to_none=True)
                adversary_loss = adversary.classification_loss(
                    context_hidden.detach(), context_outcome, context_target,
                )
                adversary_loss.backward()
                adversary_optimizer.step()
                for parameter in adversary.parameters():
                    parameter.requires_grad_(False)
                adversary_confusion = adversary.confusion_loss(
                    context_hidden, context_outcome,
                )
            loss = (
                task_loss
                + args.context_view_weight * view_loss
                + args.consistency_weight * consistency
                + args.representation_consistency_weight * representation_consistency
                + args.pad_adversary_weight * adversary_confusion
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                1.0,
            )
            optimizer.step()
            for parameter in adversary.parameters():
                parameter.requires_grad_(True)
            for name, value in (
                ("loss", loss), ("task", task_loss), ("view", view_loss),
                ("consistency", consistency), ("adversary", adversary_loss),
                ("representation_consistency", representation_consistency),
                ("adversary_confusion", adversary_confusion),
            ):
                epoch_terms[name].append(float(value.detach().cpu()))

        val_rows = BASE.score_shots(encoder, head, validation, device=device)
        point = BASE.select_validation_threshold(val_rows, target_tpr=args.target_tpr)
        metrics, _ = BASE.metrics_at_threshold(val_rows, threshold=point["threshold"])
        context_audit = _validation_context_audit(
            encoder, head, source_train, validation,
            threshold=float(point["threshold"]), device=device,
        )
        metrics.update(context_audit)
        metrics.update({name + "_loss": float(np.mean(values)) for name, values in epoch_terms.items()})
        metrics["epoch"] = epoch
        metrics["passes_all_validation_gates"] = bool(
            validation_selection_key(metrics)[0]
        )
        history.append(metrics)
        rank = validation_selection_key(metrics)
        print(
            f"epoch={epoch:02d} loss={metrics['loss_loss']:.5f} "
            f"val_auc={metrics['auc']:.4f} val_tpr10={metrics['timely_tpr']:.4f} "
            f"val_fpr={metrics['fpr']:.4f} probe_auc="
            f"{metrics['paired_context_probe_macro_auc']:.4f} "
            f"all_gates={metrics['passes_all_validation_gates']} "
            f"threshold={metrics['threshold']:.6f}",
            flush=True,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank; best_epoch = epoch; stale = 0
            payload = BASE._checkpoint_payload(
                encoder=encoder, head=head, center=center, scale=scale,
                roles=roles, seed=args.seed, epoch=epoch, validation=metrics,
                training_label=training_label,
            )
            payload["training_regularization"] = {
                "context_lengths": list(CONTEXT_LENGTHS),
                "context_view_weight": args.context_view_weight,
                "consistency_weight": args.consistency_weight,
                "representation_consistency_weight": args.representation_consistency_weight,
                "pad_adversary_weight": args.pad_adversary_weight,
                "pad_adversary_condition": "separate outcome-specific heads",
                "pad_adversary_strategy": "alternating detached probe plus uniform-confusion encoder loss",
                "causal_view_contract": "same endpoint; delete older history only",
                "epoch_selection": "VAL160 preregistered source and context gates first",
            }
            payload["validation_context_audit"] = context_audit
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary); temporary.replace(checkpoint_path)
            torch.save(adversary.state_dict(), output / "training_only_context_adversary.pt")
        else:
            stale += 1
        if epoch >= int(args.minimum_epochs) and stale >= int(args.patience):
            break

    encoder, head, checkpoint = GS.load_source_checkpoint(
        checkpoint_path, channels=len(RH.REBUILD), expected_seed=args.seed, device=device,
    )
    audit = dict(checkpoint["validation_context_audit"])
    summary = {
        "status": "SOURCE_PADFIX_TRAINED_TEST_UNREAD",
        "schema_version": GS.SOURCE_CHECKPOINT_SCHEMA,
        "seed": int(args.seed),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "split_counts": dict(BASE.EXPECTED_COUNTS),
        "split_sha256": BASE.split_sha256(roles),
        "label_authority": "HDF5 meta only",
        "training_label": training_label,
        "training_label_audit": label_audit,
        "feature_names": list(RH.REBUILD),
        "model_config": GS.source_model_config(len(RH.REBUILD)),
        "best_epoch": best_epoch,
        "history": history,
        "validation_context_audit": audit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": BASE.sha256_file(checkpoint_path),
        "encoder_state_sha256": _state_sha256(checkpoint["encoder"]),
        "test_shards_read": 0,
        "elapsed_seconds": time.time() - started,
        "training_regularization": checkpoint["training_regularization"],
        "validation_gates": dict(VALIDATION_GATES),
        "passes_all_validation_gates": bool(
            validation_selection_key({
                **checkpoint["validation_metrics"],
                **audit,
            })[0]
        ),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return checkpoint_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--minimum-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--steps-per-epoch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-tpr", type=float, default=0.90)
    parser.add_argument(
        "--label-policy", choices=ATL.POLICIES, default="fixed125",
    )
    parser.add_argument(
        "--advanced-time-path", type=Path, default=BASE.JTEXT_ADVANCED_TIME,
    )
    parser.add_argument("--context-view-weight", type=float, default=0.50)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--representation-consistency-weight", type=float, default=0.0)
    parser.add_argument("--pad-adversary-weight", type=float, default=0.005)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, summary = train(args)
    print(json.dumps({
        "status": summary["status"],
        "seed": summary["seed"],
        "best_epoch": summary["best_epoch"],
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "validation_context_audit": summary["validation_context_audit"],
        "training_regularization": summary["training_regularization"],
    }, indent=2, sort_keys=True), flush=True)
    if args.evaluate_test:
        result = BASE.evaluate_test(
            checkpoint, output=Path(args.output).expanduser().resolve(),
            seed=args.seed, device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
