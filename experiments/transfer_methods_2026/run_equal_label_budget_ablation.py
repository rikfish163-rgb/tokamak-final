#!/usr/bin/env python3
"""Retrospective equal-label-budget front-load controls for balanced360.

This script answers a narrow diagnostic question: if the labels acquired by
the active online policy had instead been supplied to the target-migration
stage before evaluation, would a frozen target model improve?  Training shots
are never scored.  The primary result is therefore on the label-disjoint
complement, with the original static and causal-online predictions restricted
to exactly the same shots.

The selected-query arm is deliberately noncausal/oracle: query identities were
created while developing the policy on balanced360.  The label-matched random
arm is also retrospective.  Neither arm is independent test evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import dual_threshold_alarm as dual_alarm  # noqa: E402
import dynamic_hazard_adapter as dynamic  # noqa: E402
import evaluate_dynamic_positive_gate as positive_gate  # noqa: E402
import grid_search as grid  # noqa: E402
import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
import run_dynamic_hazard_adapter as dha  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_online_active_a3 as a3  # noqa: E402
import run_two_timescale_online as two_runner  # noqa: E402
import run_unified_online_active as unified  # noqa: E402
import shot_alarm as alarm  # noqa: E402


SCHEMA_VERSION = "equal-label-budget-frontload-v2"
ARMS = ("support_only", "selected", "random_uniform_blocked")
BASE_EPOCHS = 12
BASE_BATCH_SIZE = 128
PRIVATE_EPOCHS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(values: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _same_down_time(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def load_active_query_ids(
    ledger: Path,
    *,
    roster: Sequence[int],
    outcome_reader: Callable[[int], Mapping[str, Any]],
) -> list[int]:
    """Validate the archived reveal ledger against authoritative HDF5 labels."""

    roster_list = [int(value) for value in roster]
    roster_set = set(roster_list)
    rows = load_jsonl(Path(ledger))
    selected = [int(row["shot"]) for row in rows]
    if len(selected) != len(set(selected)):
        raise RuntimeError("active-query reveal ledger contains duplicate shots")
    if any(shot not in roster_set for shot in selected):
        raise RuntimeError("active-query reveal ledger contains a shot outside balanced360")
    if selected != sorted(selected, key=roster_list.index):
        raise RuntimeError("active-query reveal ledger is not in stream order")
    for row, shot in zip(rows, selected):
        outcome = dict(outcome_reader(shot))
        ledger_label = bool(row["revealed_is_disrupt"])
        ledger_down = row.get("revealed_down_time_s")
        if (
            ledger_label != bool(outcome["is_disrupt"])
            or not _same_down_time(ledger_down, outcome.get("down_time_s"))
        ):
            raise RuntimeError(
                f"shot {shot}: reveal ledger differs from HDF5-authoritative outcome"
            )
    return selected


def load_label_free_query_ids(ledger: Path, *, roster: Sequence[int]) -> list[int]:
    """Recover selected identities without opening any outcome field."""

    roster_list = [int(value) for value in roster]
    rows = load_jsonl(Path(ledger))
    if [int(row["shot"]) for row in rows] != roster_list:
        raise RuntimeError("query-decision ledger does not match balanced360 order")
    selected = [int(row["shot"]) for row in rows if bool(row["selected"])]
    if len(selected) != len(set(selected)):
        raise RuntimeError("query-decision ledger selected a shot more than once")
    return selected


def block_matched_random_ids(
    *,
    roster: Sequence[int],
    excluded: Sequence[int],
    reference_ids: Sequence[int],
    block_size: int,
    seed: int,
    replicate: int,
) -> list[int]:
    """Draw a label-agnostic random control with the same per-block budget."""

    roster_list = [int(value) for value in roster]
    excluded_set = {int(value) for value in excluded}
    reference_set = {int(value) for value in reference_ids}
    if replicate < 0 or block_size <= 0 or len(roster_list) % int(block_size):
        raise ValueError("invalid block-random control contract")
    if not reference_set.issubset(set(roster_list)):
        raise ValueError("reference query set must lie inside the stream roster")
    generator = np.random.default_rng(int(seed) * 1009 + int(replicate) * 9176 + 41)
    chosen: list[int] = []
    for start in range(0, len(roster_list), int(block_size)):
        block = roster_list[start:start + int(block_size)]
        count = sum(shot in reference_set for shot in block)
        pool = [shot for shot in block if shot not in excluded_set]
        if len(pool) < count:
            raise RuntimeError("insufficient disjoint shots for block-random control")
        if count:
            chosen.extend(
                int(value) for value in generator.choice(pool, size=count, replace=False)
            )
    chosen_set = {int(value) for value in chosen}
    result = [shot for shot in roster_list if shot in chosen_set]
    if len(result) != len(reference_set) or chosen_set & excluded_set:
        raise RuntimeError("random-control budget construction failed")
    return result


def frontload_partition(
    *, roster: Sequence[int], extra_train_ids: Sequence[int], support_ids: Sequence[int],
) -> dict[str, list[int]]:
    roster_list = [int(value) for value in roster]
    extra = [int(value) for value in extra_train_ids]
    support = [int(value) for value in support_ids]
    if len(roster_list) != len(set(roster_list)) or roster_list != sorted(roster_list):
        raise ValueError("balanced360 roster must be unique and chronological")
    if len(extra) != len(set(extra)) or any(value not in set(roster_list) for value in extra):
        raise ValueError("front-load shots must be a unique balanced360 subset")
    if set(support) & set(roster_list):
        raise RuntimeError("fixed EAST support and balanced360 must be disjoint")
    evaluation = [shot for shot in roster_list if shot not in set(extra)]
    if set(extra) & set(evaluation) or set(support) & set(evaluation):
        raise RuntimeError("front-load training/evaluation overlap")
    return {
        "support_ids": support,
        "extra_train_ids": extra,
        "evaluation_ids": evaluation,
    }


def common_evaluation_ids(
    *, roster: Sequence[int], selected_train_ids: Sequence[int],
    random_train_ids: Sequence[int],
) -> list[int]:
    excluded = {int(value) for value in selected_train_ids} | {
        int(value) for value in random_train_ids
    }
    return [int(value) for value in roster if int(value) not in excluded]


def class_counts(ids: Sequence[int], outcomes: Mapping[int, Mapping[str, Any]]) -> dict[str, int]:
    disruptive = sum(bool(outcomes[int(shot)]["is_disrupt"]) for shot in ids)
    return {"shots": len(ids), "disruptive": int(disruptive), "safe": len(ids) - int(disruptive)}


def read_online_update_budget(path: Path) -> dict[str, int]:
    receipts = []
    handle = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz" else
        path.open("r", encoding="utf-8")
    )
    with handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "unified_revealed_finalize":
                receipts.append(row)
    error_groups = {"FP", "FN", "TARDY"}
    return {
        "revealed_shots": len(receipts),
        "model_update_shots": sum(row.get("group") in error_groups for row in receipts),
        "shared_optimizer_steps": sum(int(row.get("shared", {}).get("steps", 0)) for row in receipts),
        "private_optimizer_steps": sum(int(row.get("private", {}).get("steps", 0)) for row in receipts),
        "shared_accepted_updates": sum(bool(row.get("shared", {}).get("updated")) for row in receipts),
        "private_accepted_updates": sum(bool(row.get("private", {}).get("updated")) for row in receipts),
    }


def load_rebuilt_training_shots(ids: Sequence[int]) -> list[dict[str, Any]]:
    shots, missing = grid._load_rebuilt_shots("EAST", [int(value) for value in ids])
    if missing or len(shots) != len(ids):
        raise RuntimeError(f"missing front-load EAST HDF5 shots: {missing}")
    return shots


def historical_target_sampler(
    shots: Sequence[Mapping[str, Any]], *, pos_ms: float, gray_ms: float,
    seed: int,
) -> Callable[[float], tuple[np.ndarray, np.float32]]:
    """Exact shot-first sampler used by the sealed migrated parent.

    This is intentionally not ``grid.build_sampler``.  That general helper was
    later changed to choose the class before the shot, which reweights the
    registered 15D/95N support cohort and does not reproduce the sealed parent.
    The historical recipe first samples a shot uniformly, then takes a positive
    window with probability ``pp`` only when that shot has one.
    """

    positive: list[list[int]] = []
    negative: list[list[int]] = []
    for shot in shots:
        values = np.asarray(shot["v"])
        sample_count = len(values)
        times = float(shot["start"]) + np.arange(sample_count) / float(shot["rate"])
        pos_endpoints: list[int] = []
        neg_endpoints: list[int] = []
        if bool(shot["is_disrupt"]) and shot.get("down") is not None:
            for endpoint in range(grid.W - 1, sample_count):
                ttd_ms = (float(shot["down"]) - times[endpoint]) * 1000.0
                if 0.0 <= ttd_ms <= float(pos_ms):
                    pos_endpoints.append(endpoint)
                elif float(pos_ms) + float(gray_ms) < ttd_ms <= 60_000.0:
                    neg_endpoints.append(endpoint)
        else:
            neg_endpoints = list(range(grid.W - 1, sample_count))
        positive.append(pos_endpoints)
        negative.append(neg_endpoints)

    generator = np.random.default_rng(int(seed))

    def sample(positive_probability: float = 0.5) -> tuple[np.ndarray, np.float32]:
        while True:
            shot_index = int(generator.integers(len(shots)))
            pos_endpoints = positive[shot_index]
            neg_endpoints = negative[shot_index]
            if not pos_endpoints and not neg_endpoints:
                continue
            if pos_endpoints and generator.random() < float(positive_probability):
                endpoint = pos_endpoints[int(generator.integers(len(pos_endpoints)))]
                label = np.float32(1.0)
            else:
                if not neg_endpoints:
                    continue
                endpoint = neg_endpoints[int(generator.integers(len(neg_endpoints)))]
                label = np.float32(0.0)
            window = np.asarray(shots[shot_index]["v"])[
                endpoint - grid.W + 1:endpoint + 1
            ]
            return np.ascontiguousarray(window), label

    return sample


def train_frontloaded_target(
    *, seed: int, source_checkpoint: Path, support_raw: list[dict[str, Any]],
    extra_raw: list[dict[str, Any]], scaler_center: np.ndarray,
    scaler_scale: np.ndarray,
) -> dict[str, Any]:
    support = grid.transform(support_raw, "base")
    extra = grid.transform(extra_raw, "base")
    for group in (support, extra):
        grid.apply_scale(group, scaler_center, scaler_scale)
    expanded = [*support, *extra]
    sampler = historical_target_sampler(
        expanded,
        pos_ms=grid.POS["ea125"]["ea"],
        gray_ms=grid.GRAY["gray0"]["ea"],
        seed=seed,
    )
    enc, head = grid.train_adapt(
        None, sampler, len(grid.RH.REBUILD), BASE_EPOCHS, BASE_BATCH_SIZE, seed,
        source_checkpoint=source_checkpoint,
    )
    enc.eval()
    head.eval()

    fixed_support_scores = np.asarray([
        float(alarm.full_risk(enc, head, shot).max()) for shot in support
    ])
    fixed_support_labels = np.asarray([
        bool(shot["is_disrupt"]) for shot in support
    ])
    selected_threshold = alarm.select_tpr_first_threshold(
        fixed_support_scores, fixed_support_labels, target_tpr=0.80,
    )
    threshold = float(selected_threshold["threshold"])

    scales = a1.fit_support_physics_scales(support)
    expanded_cache = dha.build_support_cache(enc, head, expanded, scales)
    fixed_cache = dha.build_support_cache(enc, head, support, scales)
    torch.manual_seed(int(seed) + 5000)
    shared = dynamic.DynamicHazardAdapter(
        feature_dim=dha.FEATURE_DIM,
        physics_dim=dha.PHYSICS_DIM,
        hidden=16,
        max_logit_residual=1.0,
        min_physics_families=2,
    ).to(alarm.DEV)
    shared_history = dha.train_adapter(
        shared, expanded_cache, seed=seed, threshold=threshold,
    )
    minimum_positive, positive_count = positive_gate.support_positive_threshold(
        shared, fixed_cache, 0.10,
    )

    support_ids = [int(shot["shot"]) for shot in support]
    median8, scale8 = unified.fit_causal_raw8_scaler(support_ids)
    private, private_batches, private_contract = unified.fit_private_model(
        enc, head, shared, expanded, scales, median8, scale8,
        threshold=threshold,
        minimum_positive=minimum_positive,
        seed=seed,
        epochs=PRIVATE_EPOCHS,
    )
    private_window_count = sum(len(batch["label"]) for batch in private_batches)
    compute = {
        "base_optimizer_steps": BASE_EPOCHS * 32,
        "shared_optimizer_steps": 4 * 40 + 8 * 48,
        "private_optimizer_steps": PRIVATE_EPOCHS * max(1, private_window_count // 64),
        "base_batch_size": BASE_BATCH_SIZE,
        "shared_batch_size": 128,
        "private_batch_size": 64,
        "private_supervision_windows": private_window_count,
        "note": "standard migration retraining compute; label-budget matched, not FLOP-matched to online inter-updates",
    }
    return {
        "encoder": enc,
        "head": head,
        "shared": shared.eval(),
        "private": private.eval(),
        "scales": scales,
        "median8": median8,
        "scale8": scale8,
        "support_selected_threshold": threshold,
        "support_threshold_metrics": selected_threshold,
        "minimum_positive_residual": minimum_positive,
        "positive_window_count": positive_count,
        "shared_history": shared_history,
        "private_contract": private_contract,
        "compute": compute,
        "expanded_cache_windows": int(sum(len(row["label"]) for row in expanded_cache)),
    }


def score_frontloaded_shot(
    *, shot_id: int, model: Mapping[str, Any], center: np.ndarray,
    scale: np.ndarray, thresholds: Mapping[str, float],
) -> dict[str, Any]:
    enc = model["encoder"]
    head = model["head"]
    shot = two_runner.load_east_signal_only(shot_id, center, scale)
    base_risk = alarm.full_risk(enc, head, shot)
    endpoints = np.arange(
        alarm.W - 1 + dha.LAG, len(shot["v"]), dtype=np.int64,
    )
    if not len(endpoints):
        raise RuntimeError(f"shot {shot_id}: trace is shorter than the model context")
    frozen, direct_logit = dha.extract_frozen_batch(enc, head, shot["v"], endpoints)
    risk_indices = endpoints - (alarm.W - 1)
    direct_probability = torch.sigmoid(torch.from_numpy(direct_logit)).numpy()
    if np.max(np.abs(direct_probability - base_risk[risk_indices])) > 1e-5:
        raise RuntimeError(f"shot {shot_id}: direct/base trace mismatch")
    base_dynamic = np.log(
        np.clip(base_risk[risk_indices], 1e-6, 1 - 1e-6)
        / np.clip(1 - base_risk[risk_indices], 1e-6, 1 - 1e-6)
    ).astype(np.float32)
    physics, physics_available = a3.physics_family_trace(shot_id, model["scales"])
    family_scores = {
        family: physics[:, index] for index, family in enumerate(a3.FAMILIES)
    }
    family_available = {
        family: physics_available[:, index]
        for index, family in enumerate(a3.FAMILIES)
    }
    physics_support, _known, _count = audit.combine_family_scores(
        family_scores, family_available,
    )
    raw, raw_available = unified.causal_raw8_signal_only(shot_id)
    raw, raw_available = unified.scale_raw8(
        raw, raw_available, model["median8"], model["scale8"],
    )
    combined_logit = np.log(
        np.clip(base_risk, 1e-6, 1 - 1e-6)
        / np.clip(1 - base_risk, 1e-6, 1 - 1e-6)
    ).astype(np.float32)
    initial = np.arange(min(dha.LAG, len(combined_logit)), dtype=np.int64)
    initial_private = unified.make_private_batch(
        raw, raw_available, initial, combined_logit,
    )
    with torch.no_grad():
        combined_logit[initial] = model["private"](
            initial_private["raw"].to(alarm.DEV),
            initial_private["available"].to(alarm.DEV),
            initial_private["base_logit"].to(alarm.DEV),
        )[0].cpu().numpy()
    combined_logit[risk_indices] = unified.static_combined_logits(
        model["shared"], model["private"],
        frozen=frozen,
        base_logit=base_dynamic,
        physics=physics[endpoints],
        physics_available=physics_available[endpoints],
        raw=raw,
        raw_available=raw_available,
        minimum_positive=model["minimum_positive_residual"],
        risk_offset=dha.LAG,
    )
    result: dict[str, Any] = {
        "shot": int(shot_id),
        "combined_max_probability": float(torch.sigmoid(torch.from_numpy(combined_logit)).max()),
    }
    for name, threshold in thresholds.items():
        threshold_logit = math.log(float(threshold) / (1.0 - float(threshold)))
        decision = dual_alarm.dual_threshold_alarm(
            combined_logit - threshold_logit,
            physics_support[alarm.W - 1:],
            **unified.INITIAL_ALARM_PARAMETERS,
        )
        alarm_time = (
            float(shot["start"])
            + (alarm.W - 1 + int(decision["alarm_index"])) / float(shot["rate"])
            if decision["alarmed"] else None
        )
        result[name] = {
            **decision,
            "alarm_time_s": alarm_time,
            "threshold": float(threshold),
        }
    return result


def rows_for_metric(records: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    return [
        {"shot": int(row["shot"]), **dict(row[key])}
        for row in records
    ]


def restrict_rows(rows: Sequence[Mapping[str, Any]], ids: Sequence[int]) -> list[dict[str, Any]]:
    by_shot = {int(row["shot"]): dict(row) for row in rows}
    wanted = [int(value) for value in ids]
    if len(by_shot) != len(rows) or any(shot not in by_shot for shot in wanted):
        raise RuntimeError("baseline prediction ledger does not cover the evaluation roster")
    return [by_shot[shot] for shot in wanted]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(131, 133, 137), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--random-replicate", type=int, default=0)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--online-artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if alarm.DEV != "cuda":
        raise RuntimeError("registered equal-budget ablation requires rl_env CUDA")
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)

    # The frozen encoder is evaluated through both current-only and
    # current-plus-history batches.  TF32 can make those algebraically equal
    # paths differ by more than the registered 1e-5 seal tolerance, so this
    # ablation uses true FP32 matmul rather than weakening the guard.
    torch.set_float32_matmul_precision("highest")
    roster = sorted(int(value) for value in common.DataSpec().east["test_locked"])
    if len(roster) != 360:
        raise RuntimeError("equal-budget ablation requires balanced360")
    online_dir = Path(
        args.online_artifacts
        or PACKAGE / f"evidence/primary_online_artifacts/seed{args.seed}"
    ).expanduser().resolve()
    source_checkpoint = Path(
        args.source_checkpoint or grid.default_source_checkpoint(args.seed)
    ).expanduser().resolve()
    source_hash_before = sha256_file(source_checkpoint)
    baseline_summary = json.loads((online_dir / "summary.json").read_text())
    if int(baseline_summary["seed"]) != args.seed:
        raise RuntimeError("online artifact seed mismatch")
    frozen_parent_threshold = float(baseline_summary["threshold"])

    outcome_cache: dict[int, dict[str, Any]] = {}

    def outcome_reader(shot: int) -> dict[str, Any]:
        shot = int(shot)
        if shot not in outcome_cache:
            outcome_cache[shot] = a1.read_outcome_after_seal(shot)
        return dict(outcome_cache[shot])

    reveal_ledger = online_dir / "reveals.jsonl"
    query_ledger = online_dir / "query_decisions.jsonl"
    if args.arm == "support_only":
        active_ids = load_label_free_query_ids(query_ledger, roster=roster)
        active_outcomes: dict[int, dict[str, Any]] = {}
        active_counts = None
        extra_ids: list[int] = []
    else:
        active_ids = load_active_query_ids(
            reveal_ledger, roster=roster, outcome_reader=outcome_reader,
        )
        active_outcomes = {shot: outcome_reader(shot) for shot in active_ids}
        active_counts = class_counts(active_ids, active_outcomes)
    if args.arm == "selected":
        extra_ids = active_ids
    elif args.arm == "random_uniform_blocked":
        extra_ids = block_matched_random_ids(
            roster=roster,
            # Natural overlap with the active set is allowed.  Requiring
            # disjointness is infeasible when an active 20-shot block contains
            # more than ten queries.  The common evaluation mask removes the
            # union of both training sets.
            excluded=[],
            reference_ids=active_ids,
            block_size=20,
            seed=args.seed,
            replicate=args.random_replicate,
        )
    extra_label_outcomes = {shot: outcome_reader(shot) for shot in extra_ids}

    support_raw, no_test = grid.load_east(include_test=False)
    if no_test:
        raise RuntimeError("support-only loader unexpectedly returned test shots")
    support_ids = [int(shot["shot"]) for shot in support_raw]
    partition = frontload_partition(
        roster=roster, extra_train_ids=extra_ids, support_ids=support_ids,
    )
    if args.arm == "random_uniform_blocked":
        # Active-query labels were opened to validate the archived selected arm.
        # Exclude those shots as well, so no pre-read label enters this arm's
        # evaluation roster.  This is also the common mask required for a
        # direct selected-vs-random comparison.
        partition["evaluation_ids"] = common_evaluation_ids(
            roster=roster,
            selected_train_ids=active_ids,
            random_train_ids=extra_ids,
        )
    extra_raw = load_rebuilt_training_shots(extra_ids)
    extra_outcomes = {int(shot["shot"]): {
        "is_disrupt": bool(shot["is_disrupt"]), "down_time_s": shot["down"],
    } for shot in extra_raw}
    for shot in extra_ids:
        if (
            not _same_down_time(
                extra_outcomes[shot]["down_time_s"], extra_label_outcomes[shot]["down_time_s"],
            )
            or bool(extra_outcomes[shot]["is_disrupt"])
            != bool(extra_label_outcomes[shot]["is_disrupt"])
        ):
            raise RuntimeError(f"shot {shot}: front-load training label mismatch")

    _source_enc, _source_head, source_contract = grid.load_source_checkpoint(
        source_checkpoint,
        channels=len(grid.RH.REBUILD),
        expected_seed=args.seed,
        device="cpu",
    )
    center = source_contract["scaler_center"].cpu().numpy()
    scale = source_contract["scaler_scale"].cpu().numpy()
    del _source_enc, _source_head
    print(
        f"[FRONTLOAD] seed={args.seed} arm={args.arm} "
        f"support={len(support_ids)} extra={len(extra_ids)} eval={len(partition['evaluation_ids'])}",
        flush=True,
    )
    model = train_frontloaded_target(
        seed=args.seed,
        source_checkpoint=source_checkpoint,
        support_raw=support_raw,
        extra_raw=extra_raw,
        scaler_center=center,
        scaler_scale=scale,
    )
    trained_model_hashes = {
        "encoder_head": a1.model_sha256(model["encoder"], model["head"]),
        "shared_adapter": auxiliary.module_sha256(model["shared"]),
        "private_adapter": auxiliary.module_sha256(model["private"]),
    }
    if args.arm == "support_only":
        archived_hashes = baseline_summary["hashes"]
        expected_fidelity = {
            "encoder_head": archived_hashes["base_before_after"],
            "shared_adapter": archived_hashes["static_shared"],
            "private_adapter": archived_hashes["static_private"],
            "threshold": frozen_parent_threshold,
        }
        actual_fidelity = {
            **trained_model_hashes,
            "threshold": float(model["support_selected_threshold"]),
        }
        fidelity = {
            "schema_version": "sealed-parent-fidelity-v1",
            "expected": expected_fidelity,
            "actual": actual_fidelity,
            "passed": actual_fidelity == expected_fidelity,
            "sampler": "historical-shot-first-then-within-shot-window-v1",
        }
        (args.out / "parent_fidelity.json").write_text(
            json.dumps(fidelity, indent=2, sort_keys=True) + "\n"
        )
        if not fidelity["passed"]:
            raise RuntimeError(
                "support-only parent fidelity failed; see parent_fidelity.json"
            )
    thresholds = {
        "frozen_parent_threshold": frozen_parent_threshold,
        "support110_recalibrated_threshold": float(model["support_selected_threshold"]),
    }

    records: list[dict[str, Any]] = []
    writer = a1.JsonlSealWriter(args.out / "predictions.jsonl")
    oracle = audit.LazyShotOutcomeOracle(partition["evaluation_ids"], outcome_reader)
    try:
        for index, shot in enumerate(partition["evaluation_ids"]):
            record = {
                "stream_index": index,
                **score_frontloaded_shot(
                    shot_id=shot,
                    model=model,
                    center=center,
                    scale=scale,
                    thresholds=thresholds,
                ),
            }
            writer.append(record)
            records.append(record)
            oracle.seal_current(shot)
            oracle.discard_unqueried(shot)
            if (index + 1) % 40 == 0:
                print(f"[FRONTLOAD] sealed {index + 1}/{len(partition['evaluation_ids'])}", flush=True)
    finally:
        writer.close()
    oracle.open_final_evaluation()
    evaluation_outcomes = oracle.all_outcomes()

    static_rows = load_jsonl(online_dir / "static_predictions.jsonl")
    online_rows = load_jsonl(online_dir / "predictions.jsonl")
    evaluation_ids = partition["evaluation_ids"]
    own_metrics = {
        "frontload_frozen_parent_threshold": unified.alarm_metrics(
            rows_for_metric(records, "frozen_parent_threshold"), evaluation_outcomes,
        ),
        "frontload_support110_recalibrated_threshold": unified.alarm_metrics(
            rows_for_metric(records, "support110_recalibrated_threshold"), evaluation_outcomes,
        ),
        "original_static_same_complement": unified.alarm_metrics(
            restrict_rows(static_rows, evaluation_ids), evaluation_outcomes,
        ),
        "original_online_same_complement": unified.alarm_metrics(
            restrict_rows(online_rows, evaluation_ids), evaluation_outcomes,
        ),
    }
    labels = np.asarray([
        bool(evaluation_outcomes[shot]["is_disrupt"]) for shot in evaluation_ids
    ])
    scores = np.asarray([float(row["combined_max_probability"]) for row in records])
    own_metrics["frontload_auc"] = float(grid.auc(labels.astype(int), scores))

    update_path = online_dir / "updates.jsonl.gz"
    if not update_path.exists():
        update_path = online_dir / "updates.jsonl"
    online_update_budget = read_online_update_budget(update_path)
    source_hash_after = sha256_file(source_checkpoint)
    if source_hash_after != source_hash_before:
        raise RuntimeError("front-load training modified the immutable source checkpoint")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "seed": args.seed,
            "arm": args.arm,
            "encoder": model["encoder"].state_dict(),
            "head": model["head"].state_dict(),
            "shared": model["shared"].state_dict(),
            "private": model["private"].state_dict(),
            "thresholds": thresholds,
            "source_checkpoint_sha256": source_hash_before,
            "extra_train_ids_sha256": sha256_ids(extra_ids),
        },
        args.out / "frontloaded_target_state.pt",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION",
        "arm": args.arm,
        "seed": args.seed,
        "random_replicate": (
            args.random_replicate if args.arm == "random_uniform_blocked" else None
        ),
        "label_budget": {
            "active_query_count": len(active_ids),
            "active_query_class_counts": active_counts,
            "frontload_class_counts": class_counts(extra_ids, extra_label_outcomes),
            "online_effective_update_budget": online_update_budget,
            "definition_warning": (
                "query labels and parameter-changing error updates are distinct budgets; "
                "standard front-load migration uses every acquired label"
            ),
        },
        "partition": {
            **partition,
            "support_ids_sha256": sha256_ids(support_ids),
            "active_query_ids_sha256": sha256_ids(active_ids),
            "extra_train_ids_sha256": sha256_ids(extra_ids),
            "evaluation_ids_sha256": sha256_ids(evaluation_ids),
            "train_evaluation_overlap": 0,
            "evaluation_class_counts_opened_after_seal": class_counts(
                evaluation_ids, evaluation_outcomes,
            ),
        },
        "thresholds": {
            "frozen_parent": frozen_parent_threshold,
            "support110_recalibrated": model["support_selected_threshold"],
            "support110_recalibrated_selection": model["support_threshold_metrics"],
            "source": "fixed original EAST110 only; balanced360 complement never selects a threshold",
        },
        "metrics_on_label_disjoint_complement": own_metrics,
        "training": {
            "shared_positive_window_count_fixed110": model["positive_window_count"],
            "expanded_cache_windows": model["expanded_cache_windows"],
            "private_contract": model["private_contract"],
            "compute": model["compute"],
            "trainable_scope": (
                "top target Transformer block and target head; shared17 hazard adapter; "
                "EAST-private raw8 dynamics residual"
            ),
            "source_checkpoint_immutable": True,
        },
        "hashes": {
            "source_checkpoint_before_after": source_hash_before,
            "active_reveal_ledger": sha256_file(reveal_ledger),
            "label_free_query_decision_ledger": sha256_file(query_ledger),
            "online_summary": sha256_file(online_dir / "summary.json"),
            "online_updates": sha256_file(update_path),
            "predictions": sha256_file(args.out / "predictions.jsonl"),
            "frontloaded_target_state": sha256_file(args.out / "frontloaded_target_state.pt"),
            **trained_model_hashes,
        },
        "leakage_contract": (
            "extra labeled balanced360 shots are used only for front-load training and are "
            "removed from evaluation; the random arm also removes archived active-query shots "
            "whose labels were opened to validate its reference budget; complement features are "
            "predicted without reading /meta; "
            "all complement outcomes are opened only after every prediction is sealed"
        ),
        "claim_boundary": (
            "retrospective oracle diagnostic: balanced360 selected the query identities and "
            "informed architecture search; this ablation tests label-budget sufficiency on a "
            "label-disjoint complement but is not independent generalization evidence"
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "seed": args.seed,
        "arm": args.arm,
        "extra": len(extra_ids),
        "evaluation": len(evaluation_ids),
        "metrics": own_metrics,
    }, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
