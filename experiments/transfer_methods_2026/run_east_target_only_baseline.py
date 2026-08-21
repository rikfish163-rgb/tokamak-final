#!/usr/bin/env python3
"""Frozen EAST-only target baseline for the v3c paper contract.

This runner deliberately has two layers of API:

* CPU-only contract helpers are importable by unit tests without allocating a
  CUDA context; and
* ``run`` lazily imports the existing v3c model/sidecar modules and executes a
  support-only target fit followed by a sealed, static balanced360 replay.

The two registered arms differ only in the 17-channel scaler:
``target_only_local17`` fits it on EAST support signals, while
``target_only_matched17`` consumes the immutable source-checkpoint scaler but
does not read J-TEXT samples.  Neither arm implements native23; that is an
explicit future experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
CONFIG_PATH = HERE / "paper_grade_package/configs/east_target_only_v1.json"
SUPPORT_MANIFEST = ROOT / "config/splits/east_paper131_split_all_in.json"
BALANCED_MANIFEST = ROOT / "config/splits/east_hdf5_authoritative_balanced360_v1.json"

ARM_NAMES = ("target_only_local17", "target_only_matched17")
NATIVE23_STATUS = "future_not_implemented"
FEATURE_NAMES = (
    "Bt", "ip_A", "Bt_over_Ip", "Ip_diff", "v_loop", "Z_proxy",
    "sxr_kurt", "sxr_skew", "sxr_var", "xuv_kurt", "xuv_skew", "xuv_var",
    "Mir_avg_fre", "n=1 amplitude", "mode_number_n", "dx_a", "dy_a",
)
WINDOW = 128
POSITIVE_TTD_MS = (0.0, 125.0)
NEGATIVE_TTD_MS = (125.0, 60000.0)
TARGET_EPOCHS = 12
STEPS_PER_EPOCH = 32
TARGET_STEPS = TARGET_EPOCHS * STEPS_PER_EPOCH
BATCH_SIZE = 128
TARGET_LR = 1e-4
WEIGHT_DECAY = 1e-4
SUPPORT_TARGET_TPR = 0.80
MINIMUM_LEAD_MS = 30.0
GRAY_MS = 0.0
PRIVATE_EPOCHS = 8
# Publication runs use the strict FP32 matmul path below.  Keep the original
# sealed-risk identity tolerance; TF32's batch-dependent approximation is not
# accepted as a reason to weaken this gate.
BASE_RISK_NUMERICAL_ATOL = 1e-5

ALARM_PARAMETERS = {
    "high_threshold": 0.0,
    "physics_weight": 8.0,
    "trend_weight": 12.0,
    "early_threshold": 7.9,
    "fast_width": 5,
    "slow_width": 100,
}

_OUTCOME_KEYS = frozenset({
    "label", "truth", "outcome", "is_disrupt", "down", "down_time_s",
    "DownTime", "revealed_is_disrupt", "revealed_down_time_s", "lead_ms",
})


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_ids(values: Sequence[int]) -> str:
    ordered = sorted(int(value) for value in values)
    if len(ordered) != len(set(ordered)):
        raise ValueError("shot IDs must be unique before hashing")
    return sha256_bytes(canonical_json(ordered).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def portable_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return "<external-checkpoint>"


def load_contract(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "east-target-only-v1":
        raise ValueError("target-only configuration schema mismatch")
    if tuple(config.get("arms", ())) != ARM_NAMES:
        raise ValueError("target-only arm contract mismatch")
    if config.get("native23", {}).get("status") != NATIVE23_STATUS:
        raise ValueError("native23 must remain explicitly unimplemented in v1")
    return config


def _manifest_roster(path: Path, role: str) -> tuple[list[int], list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    role_payload = payload[role]
    return (
        [int(value) for value in role_payload["disruption_shots"]],
        [int(value) for value in role_payload["non_disruption_shots"]],
    )


def load_rosters() -> tuple[list[int], list[int]]:
    support_disrupt, support_safe = _manifest_roster(SUPPORT_MANIFEST, "train")
    balanced_disrupt, balanced_safe = _manifest_roster(BALANCED_MANIFEST, "test")
    return sorted(support_disrupt + support_safe), sorted(balanced_disrupt + balanced_safe)


def validate_roster_contract(
    support_ids: Sequence[int],
    balanced_ids: Sequence[int],
    support_labels: Sequence[bool],
    balanced_labels: Sequence[bool],
) -> dict[str, Any]:
    """Validate roster cardinalities and HDF5-authoritative label counts.

    The caller supplies labels read from HDF5.  Manifest class headings are
    intentionally not accepted as a label source here.
    """

    support = [int(value) for value in support_ids]
    balanced = [int(value) for value in balanced_ids]
    if len(support) != len(set(support)) or len(balanced) != len(set(balanced)):
        raise ValueError("roster contains duplicate shot IDs")
    if set(support) & set(balanced):
        raise ValueError("support and balanced360 rosters overlap")
    if len(support) != 110:
        raise ValueError(f"support roster must contain 110 shots, got {len(support)}")
    if len(balanced) != 360:
        raise ValueError(f"balanced360 roster must contain 360 shots, got {len(balanced)}")
    if len(support_labels) != len(support) or len(balanced_labels) != len(balanced):
        raise ValueError("HDF5 label vectors do not align with their rosters")
    support_counts = {
        "disruptive": int(sum(bool(value) for value in support_labels)),
        "safe": int(sum(not bool(value) for value in support_labels)),
    }
    balanced_counts = {
        "disruptive": int(sum(bool(value) for value in balanced_labels)),
        "safe": int(sum(not bool(value) for value in balanced_labels)),
    }
    if support_counts != {"disruptive": 15, "safe": 95}:
        raise ValueError(f"support HDF5 counts changed: {support_counts}")
    if balanced_counts != {"disruptive": 180, "safe": 180}:
        raise ValueError(f"balanced360 HDF5 counts changed: {balanced_counts}")
    return {
        "support_counts": support_counts,
        "balanced360_counts": balanced_counts,
        "support_ids_sha256": sha256_ids(support),
        "balanced360_ids_sha256": sha256_ids(balanced),
        "role_overlap": 0,
        "label_authority": "EAST HDF5 meta/IsDisrupt and meta/DownTime",
    }


def historical_window_indices(
    shots: Sequence[Mapping[str, Any]], *, pos_ms: float = 125.0,
    gray_ms: float = 0.0,
) -> list[dict[str, list[int]]]:
    """Build the sealed shot-first endpoint pools for ea125/gray0."""

    if float(pos_ms) != 125.0 or float(gray_ms) != 0.0:
        raise ValueError("v1 fixes ea125 with gray0; other windows need a new contract")
    result: list[dict[str, list[int]]] = []
    for shot in shots:
        values = np.asarray(shot["v"])
        if values.ndim != 2 or len(values) < WINDOW:
            raise ValueError("every target-training shot needs at least 128 samples")
        times = float(shot["start"]) + np.arange(len(values), dtype=np.float64) / float(shot["rate"])
        positive: list[int] = []
        negative: list[int] = []
        if bool(shot.get("is_disrupt")) and shot.get("down") is not None:
            down = float(shot["down"])
            if not math.isfinite(down):
                raise ValueError("disruptive support shot has non-finite DownTime")
            for endpoint in range(WINDOW - 1, len(values)):
                ttd_ms = (down - times[endpoint]) * 1000.0
                if -1e-9 <= ttd_ms <= 125.0 + 1e-9:
                    positive.append(endpoint)
                elif ttd_ms > 125.0 + 1e-9 and ttd_ms <= 60000.0 + 1e-9:
                    negative.append(endpoint)
        else:
            negative = list(range(WINDOW - 1, len(values)))
        result.append({"positive": positive, "negative": negative})
    return result


def historical_target_sampler(
    shots: Sequence[Mapping[str, Any]], *, pos_ms: float = 125.0,
    gray_ms: float = 0.0, seed: int = 0,
) -> Callable[[float], tuple[np.ndarray, np.float32]]:
    """Return the exact registered sampler: shot first, window second."""

    pools = historical_window_indices(shots, pos_ms=pos_ms, gray_ms=gray_ms)
    generator = np.random.default_rng(int(seed))

    def sample(positive_probability: float = 0.5) -> tuple[np.ndarray, np.float32]:
        probability = float(positive_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("positive probability must lie in [0, 1]")
        while True:
            shot_index = int(generator.integers(len(shots)))
            positive = pools[shot_index]["positive"]
            negative = pools[shot_index]["negative"]
            if not positive and not negative:
                continue
            if positive and generator.random() < probability:
                endpoint = positive[int(generator.integers(len(positive)))]
                label = np.float32(1.0)
            else:
                if not negative:
                    continue
                endpoint = negative[int(generator.integers(len(negative)))]
                label = np.float32(0.0)
            window = np.asarray(shots[shot_index]["v"])[endpoint - WINDOW + 1:endpoint + 1]
            sample.last_sample = {
                "shot": int(shots[shot_index]["shot"]),
                "endpoint": int(endpoint),
                "label": float(label),
            }
            return np.ascontiguousarray(window), label

    sample.last_sample = None  # type: ignore[attr-defined]
    return sample


def select_support_threshold(
    support_scores: Sequence[float], support_labels: Sequence[bool], *,
    target_tpr: float = SUPPORT_TARGET_TPR,
) -> dict[str, float]:
    """Select a threshold from support scores only.

    There is intentionally no evaluation-score parameter.  A caller trying to
    pass balanced360 scores therefore fails at the Python API boundary.
    """

    scores = np.asarray(support_scores, dtype=np.float64)
    labels = np.asarray(support_labels, dtype=bool)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError("support scores and labels must be aligned non-empty vectors")
    if not np.isfinite(scores).all() or not labels.any() or labels.all():
        raise ValueError("support threshold needs finite scores and both classes")
    if not 0.0 < float(target_tpr) <= 1.0:
        raise ValueError("target TPR must lie in (0, 1]")
    best: tuple[tuple[float, float, float], float, float, float] | None = None
    candidates = sorted(set(float(value) for value in scores), reverse=True)
    candidates.append(float("-inf"))
    for threshold in candidates:
        tpr = float(np.mean(scores[labels] >= threshold))
        fpr = float(np.mean(scores[~labels] >= threshold))
        if tpr < float(target_tpr):
            continue
        order = (fpr, -threshold, -tpr)
        if best is None or order < best[0]:
            best = (order, threshold, tpr, fpr)
    if best is None:
        raise RuntimeError("support cannot reach the registered target TPR")
    return {"threshold": best[1], "tpr": best[2], "fpr": best[3]}


def assert_prediction_seal(records: Sequence[Mapping[str, Any]], roster: Sequence[int]) -> None:
    """Require a complete, ordered, outcome-free prediction seal."""

    expected = [int(value) for value in roster]
    actual = [int(row["shot"]) for row in records]
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("prediction seal is not complete and in roster order")
    for row in records:
        if not bool(row.get("sealed")):
            raise ValueError("prediction seal contains an unsealed row")
        found = _OUTCOME_KEYS.intersection(row)
        if found:
            raise ValueError(f"prediction seal contains outcome field: {sorted(found)[0]}")


def training_ledger(
    *, arm: str, seed: int, step_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown target-only arm: {arm}")
    return {
        "schema_version": "east-target-only-training-ledger-v1",
        "arm": arm,
        "seed": int(seed),
        "optimizer_steps": int(len(step_records)),
        "batch_size": BATCH_SIZE,
        "sampler": "historical-shot-first-then-within-shot-window-v1",
        "positive_ttd_ms": list(POSITIVE_TTD_MS),
        "negative_ttd_ms": list(NEGATIVE_TTD_MS),
        "balanced360_labels_used": False,
        "threshold_source": "support110_hdf5_authoritative_only",
        "step_records": [dict(row) for row in step_records],
    }


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ledger: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _runtime(data_root: Path, device: str) -> SimpleNamespace:
    """Import existing v3c modules only after the CPU contract layer is used."""

    os.environ["FUSION_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for the registered runner but is unavailable")
    import dual_threshold_alarm as dual_alarm
    import dynamic_hazard_adapter as dynamic
    import evaluate_dynamic_positive_gate as positive_gate
    import grid_search as grid
    import online_active_audit as audit
    import run_dynamic_hazard_adapter as dha
    import run_online_active_a1 as a1
    import run_online_active_a3 as a3
    import run_unified_online_active as unified
    import run_two_timescale_online as two_runner
    import shot_alarm as alarm
    import paper_grade_package.scripts.build_evidence as evidence

    # Existing v3c helpers use this module-level device selector.
    alarm.DEV = device
    grid.DEV = device
    return SimpleNamespace(
        torch=torch, dual_alarm=dual_alarm, dynamic=dynamic,
        positive_gate=positive_gate, grid=grid, audit=audit, dha=dha,
        a1=a1, a3=a3, unified=unified, two_runner=two_runner,
        alarm=alarm, evidence=evidence,
    )


def _set_deterministic(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _model_hash(*modules: Any) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{module_index}:{name}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _load_support(rt: SimpleNamespace, support_ids: Sequence[int]) -> list[dict[str, Any]]:
    shots, missing = rt.grid._load_rebuilt_shots("EAST", [int(value) for value in support_ids])
    if missing or len(shots) != len(support_ids):
        raise RuntimeError(f"missing support HDF5 shots: {missing}")
    return rt.grid.transform(shots, "base")


def _load_matched_scaler(rt: SimpleNamespace, seed: int, path: Path | None) -> tuple[np.ndarray, np.ndarray, Path]:
    checkpoint_path = Path(path or rt.grid.default_source_checkpoint(seed)).expanduser().resolve()
    _enc, _head, checkpoint = rt.grid.load_source_checkpoint(
        checkpoint_path, channels=len(FEATURE_NAMES), expected_seed=seed, device="cpu",
    )
    center = checkpoint["scaler_center"].cpu().numpy().astype(np.float32)
    scale = checkpoint["scaler_scale"].cpu().numpy().astype(np.float32)
    return center, scale, checkpoint_path


def _train_target_only(
    rt: SimpleNamespace, support: Sequence[Mapping[str, Any]], *, seed: int,
    device: str, arm: str,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = rt.torch
    _set_deterministic(torch, seed)
    encoder = rt.grid.Enc(len(FEATURE_NAMES)).to(device)
    head = rt.grid.Head().to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=TARGET_LR, weight_decay=WEIGHT_DECAY,
    )
    sampler = historical_target_sampler(support, seed=seed)
    step_records: list[dict[str, Any]] = []
    for epoch in range(TARGET_EPOCHS):
        encoder.train()
        head.train()
        for step_in_epoch in range(STEPS_PER_EPOCH):
            xs: list[np.ndarray] = []
            ys: list[np.float32] = []
            pairs: list[tuple[int, int, float]] = []
            for _ in range(BATCH_SIZE):
                x, y = sampler(0.5)
                xs.append(x)
                ys.append(y)
                meta = getattr(sampler, "last_sample", None)
                if meta is not None:
                    pairs.append((int(meta["shot"]), int(meta["endpoint"]), float(meta["label"])))
            xb = torch.from_numpy(np.stack(xs)).to(device)
            yb = torch.from_numpy(np.asarray(ys, dtype=np.float32)).to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(encoder(xb)), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_index = epoch * STEPS_PER_EPOCH + step_in_epoch + 1
            step_records.append({
                "step": step_index,
                "epoch": epoch + 1,
                "step_in_epoch": step_in_epoch + 1,
                "batch_size": BATCH_SIZE,
                "loss": float(loss.detach().cpu()),
                "positive_windows": int(np.sum(np.asarray(ys) > 0.5)),
                "sample_pairs_sha256": sha256_bytes(canonical_json(pairs).encode()),
            })
    ledger = training_ledger(arm=arm, seed=seed, step_records=step_records)
    ledger["trainable_parameters"] = int(sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters()))
    ledger["model_architecture"] = "rebuilt17-patch-transformer-128x8-d64-l2-h4-v1"
    return encoder, head, ledger


def _full_risk(rt: SimpleNamespace, encoder: Any, head: Any, shot: Mapping[str, Any], device: str) -> np.ndarray:
    torch = rt.torch
    values = np.asarray(shot["v"], dtype=np.float32)
    if len(values) < WINDOW:
        raise RuntimeError(f"shot {shot['shot']} is shorter than the 128-point context")
    endpoints = np.arange(WINDOW - 1, len(values), dtype=np.int64)
    result = np.empty(len(endpoints), dtype=np.float32)
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for start in range(0, len(endpoints), 512):
            batch = endpoints[start:start + 512]
            windows = np.stack([values[index - WINDOW + 1:index + 1] for index in batch])
            result[start:start + len(batch)] = torch.sigmoid(
                head(encoder(torch.from_numpy(windows).to(device)))
            ).detach().cpu().numpy()
    return result


def _fit_target_sidecars(
    rt: SimpleNamespace, encoder: Any, head: Any, support: list[dict[str, Any]],
    *, seed: int, threshold: float, device: str,
) -> dict[str, Any]:
    scales = rt.a1.fit_support_physics_scales(support)
    cache = rt.dha.build_support_cache(encoder, head, support, scales)
    torch = rt.torch
    torch.manual_seed(int(seed) + 5000)
    shared = rt.dynamic.DynamicHazardAdapter(
        feature_dim=rt.dha.FEATURE_DIM,
        physics_dim=rt.dha.PHYSICS_DIM,
        hidden=16,
        max_logit_residual=1.0,
        min_physics_families=2,
    ).to(device)
    shared_history = rt.dha.train_adapter(shared, cache, seed=seed, threshold=threshold)
    minimum_positive, positive_count = rt.positive_gate.support_positive_threshold(
        shared, cache, 0.10,
    )
    support_ids = [int(shot["shot"]) for shot in support]
    median8, scale8 = rt.unified.fit_causal_raw8_scaler(support_ids)
    private, private_batches, private_contract = rt.unified.fit_private_model(
        encoder, head, shared, support, scales, median8, scale8,
        threshold=threshold,
        minimum_positive=minimum_positive,
        seed=seed,
        epochs=PRIVATE_EPOCHS,
        base_risk_atol=BASE_RISK_NUMERICAL_ATOL,
    )
    compute = {
        "base_optimizer_steps": TARGET_STEPS,
        "shared_optimizer_steps": 4 * 40 + 8 * 48,
        "private_optimizer_steps": PRIVATE_EPOCHS * max(
            1, sum(len(batch["label"]) for batch in private_batches) // 64
        ),
        "base_batch_size": BATCH_SIZE,
        "shared_batch_size": 128,
        "private_batch_size": 64,
        "private_supervision_windows": int(sum(len(batch["label"]) for batch in private_batches)),
        "compute_claim": "target-window/optimizer-step matched; not FLOP-matched",
    }
    return {
        "shared": shared.eval(),
        "private": private.eval(),
        "scales": scales,
        "median8": median8,
        "scale8": scale8,
        "minimum_positive_residual": float(minimum_positive),
        "positive_window_count": int(positive_count),
        "shared_history": shared_history,
        "private_contract": private_contract,
        "compute": compute,
    }


def _score_static_shot(
    rt: SimpleNamespace, *, shot_id: int, encoder: Any, head: Any,
    sidecars: Mapping[str, Any], center: np.ndarray, scale: np.ndarray,
    threshold: float, device: str,
) -> tuple[dict[str, Any], np.ndarray]:
    shot = rt.two_runner.load_east_signal_only(int(shot_id), center, scale)
    base_risk = _full_risk(rt, encoder, head, shot, device)
    endpoints = np.arange(rt.alarm.W - 1 + rt.dha.LAG, len(shot["v"]), dtype=np.int64)
    if not len(endpoints):
        raise RuntimeError(f"shot {shot_id}: no dynamic endpoint after the 128-point context")
    frozen, direct_logit = rt.dha.extract_frozen_batch(encoder, head, shot["v"], endpoints)
    risk_indices = endpoints - (rt.alarm.W - 1)
    direct_probability = rt.torch.sigmoid(rt.torch.from_numpy(direct_logit)).numpy()
    maximum_base_delta = float(np.max(np.abs(
        direct_probability - base_risk[risk_indices]
    )))
    if maximum_base_delta > BASE_RISK_NUMERICAL_ATOL:
        raise RuntimeError(
            f"shot {shot_id}: base risk trace mismatch "
            f"(max_abs_delta={maximum_base_delta:.9g}, "
            f"atol={BASE_RISK_NUMERICAL_ATOL:.9g})"
        )
    base_logit = np.log(
        np.clip(base_risk, 1e-6, 1 - 1e-6)
        / np.clip(1 - base_risk, 1e-6, 1 - 1e-6)
    ).astype(np.float32)
    physics, physics_available = rt.a3.physics_family_trace(
        int(shot_id), sidecars["scales"],
    )
    family_scores = {
        family: physics[:, index] for index, family in enumerate(rt.a3.FAMILIES)
    }
    family_available = {
        family: physics_available[:, index] for index, family in enumerate(rt.a3.FAMILIES)
    }
    physics_support, physics_known, _family_count = rt.audit.combine_family_scores(
        family_scores, family_available,
    )
    raw, raw_available = rt.unified.causal_raw8_signal_only(int(shot_id))
    raw, raw_available = rt.unified.scale_raw8(
        raw, raw_available, sidecars["median8"], sidecars["scale8"],
    )
    combined_logit = base_logit.copy()
    initial = np.arange(min(rt.dha.LAG, len(combined_logit)), dtype=np.int64)
    initial_private = rt.unified.make_private_batch(raw, raw_available, initial, combined_logit)
    with rt.torch.no_grad():
        combined_logit[initial] = sidecars["private"](
            initial_private["raw"].to(device),
            initial_private["available"].to(device),
            initial_private["base_logit"].to(device),
        )[0].cpu().numpy()
    combined_logit[risk_indices] = rt.unified.static_combined_logits(
        sidecars["shared"], sidecars["private"],
        frozen=frozen,
        base_logit=np.asarray(base_logit[risk_indices], dtype=np.float32),
        physics=physics[endpoints],
        physics_available=physics_available[endpoints],
        raw=raw,
        raw_available=raw_available,
        minimum_positive=sidecars["minimum_positive_residual"],
        risk_offset=rt.dha.LAG,
    )
    threshold_logit = math.log(float(threshold) / (1.0 - float(threshold)))
    decision = rt.dual_alarm.dual_threshold_alarm(
        combined_logit - threshold_logit,
        physics_support[rt.alarm.W - 1:],
        **ALARM_PARAMETERS,
    )
    alarm_time = (
        float(shot["start"])
        + (rt.alarm.W - 1 + int(decision["alarm_index"])) / float(shot["rate"])
        if decision["alarmed"] else None
    )
    row = {
        "shot": int(shot_id),
        "alarmed": bool(decision["alarmed"]),
        "alarm_index": decision["alarm_index"],
        "route": decision["route"],
        "early_score": decision["early_score"],
        "alarm_time_s": alarm_time,
        "threshold": float(threshold),
        "base_max_probability": float(np.max(base_risk)),
        "combined_max_probability": float(np.max(rt.torch.sigmoid(rt.torch.from_numpy(combined_logit)).numpy())),
        "physics_known_fraction": float(np.mean(physics_known[rt.alarm.W - 1:])),
        "sealed": True,
    }
    return row, combined_logit


def _lineage_paths() -> list[Path]:
    return [
        Path(__file__), CONFIG_PATH, SUPPORT_MANIFEST, BALANCED_MANIFEST,
        HERE / "rebuilt_harness.py", HERE / "grid_search.py",
        HERE / "shot_alarm.py", HERE / "dual_threshold_alarm.py",
        HERE / "run_dynamic_hazard_adapter.py", HERE / "run_online_active_a1.py",
        HERE / "run_online_active_a3.py", HERE / "run_unified_online_active.py",
        HERE / "run_two_timescale_online.py", HERE / "dynamic_hazard_adapter.py",
        HERE / "east_private_continual.py", HERE / "east_private23_sidecar.py",
        HERE / "paper_grade_package/scripts/build_evidence.py",
    ]


def run(
    *, arm: str, seed: int, data_root: Path, out: Path,
    device: str = "cuda", source_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError(f"arm must be one of {ARM_NAMES}")
    if arm == "target_only_local17" and source_checkpoint is not None:
        raise ValueError("local17 must not receive a source checkpoint")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    config = load_contract()
    out.mkdir(parents=True)
    (out / "risk_traces").mkdir()
    started = time.time()
    rt = _runtime(data_root, device)
    torch = rt.torch
    torch.set_float32_matmul_precision("highest")

    support_ids, balanced_ids = load_rosters()
    support = _load_support(rt, support_ids)
    support_labels = [bool(shot["is_disrupt"]) for shot in support]

    # Before balanced360 is touched for prediction, validate all support labels
    # and establish the role contract from HDF5-authoritative support data.
    # balanced labels are intentionally not read here.
    roster_contract = {
        "support_size": len(support_ids),
        "balanced360_size": len(balanced_ids),
        "support_counts": {
            "disruptive": int(sum(support_labels)),
            "safe": int(sum(not value for value in support_labels)),
        },
        "support_ids_sha256": sha256_ids(support_ids),
        "balanced360_ids_sha256": sha256_ids(balanced_ids),
        "role_overlap": int(bool(set(support_ids) & set(balanced_ids))),
        "label_authority": "EAST HDF5 meta/IsDisrupt and meta/DownTime",
    }
    if roster_contract["support_size"] != 110 or roster_contract["balanced360_size"] != 360:
        raise ValueError("roster cardinality contract failed")
    if roster_contract["support_counts"] != {"disruptive": 15, "safe": 95}:
        raise ValueError(f"support HDF5 counts changed: {roster_contract['support_counts']}")
    if roster_contract["role_overlap"]:
        raise ValueError("support and balanced360 rosters overlap")

    if arm == "target_only_local17":
        center, scale = rt.grid.fit_scale(support)
        scaler_source = "EAST support110 signal-only median/IQR"
        source_path = None
    else:
        center, scale, source_path = _load_matched_scaler(rt, seed, source_checkpoint)
        scaler_source = "immutable source checkpoint scaler; target run did not read J-TEXT samples"
    for group in (support,):
        rt.grid.apply_scale(group, center, scale)

    encoder, head, ledger = _train_target_only(
        rt, support, seed=seed, device=device, arm=arm,
    )
    support_scores = np.asarray([
        float(np.max(_full_risk(rt, encoder, head, shot, device))) for shot in support
    ])
    threshold_metrics = select_support_threshold(
        support_scores, support_labels, target_tpr=SUPPORT_TARGET_TPR,
    )
    threshold = float(threshold_metrics["threshold"])
    support_rows = [
        {"shot": int(shot["shot"]), "score": float(score), "label": bool(label)}
        for shot, score, label in zip(support, support_scores, support_labels)
    ]
    sidecars = _fit_target_sidecars(
        rt, encoder, head, support, seed=seed, threshold=threshold, device=device,
    )

    # Evaluate a freshly materialized deployment copy.  This makes the model
    # used for all 360 sealed predictions exactly the state serialized below,
    # and avoids carrying any transient CUDA execution state from training.
    encoder_state = {
        name: value.detach().cpu() for name, value in encoder.state_dict().items()
    }
    head_state = {
        name: value.detach().cpu() for name, value in head.state_dict().items()
    }
    training_model_hash = _model_hash(encoder, head)
    sealed_encoder = rt.grid.Enc(len(FEATURE_NAMES)).to(device)
    sealed_head = rt.grid.Head().to(device)
    sealed_encoder.load_state_dict(encoder_state)
    sealed_head.load_state_dict(head_state)
    sealed_encoder.eval()
    sealed_head.eval()
    sealed_model_hash = _model_hash(sealed_encoder, sealed_head)
    if sealed_model_hash != training_model_hash:
        raise RuntimeError("fresh deployment copy changed the target-only model state")
    encoder, head = sealed_encoder, sealed_head
    config = dict(config)
    config["runtime_numerics"] = {
        "base_risk_probability_atol": BASE_RISK_NUMERICAL_ATOL,
        "matmul_precision": "highest",
        "reason": "strict FP32 CUDA batch-composition equivalence check",
        "fresh_deployment_copy": True,
        "training_model_sha256": training_model_hash,
        "sealed_model_sha256": sealed_model_hash,
    }

    _write_json(out / "config_snapshot.json", config)
    model_state_path = out / "target_only_state.pt"
    if model_state_path.exists():
        raise FileExistsError(model_state_path)
    torch.save({
        "schema_version": "east-target-only-state-v1",
        "arm": arm,
        "seed": int(seed),
        "feature_names": list(FEATURE_NAMES),
        "threshold": threshold,
        "scaler_center": torch.as_tensor(center, dtype=torch.float32).cpu(),
        "scaler_scale": torch.as_tensor(scale, dtype=torch.float32).cpu(),
        "raw8_median": torch.as_tensor(sidecars["median8"], dtype=torch.float32).cpu(),
        "raw8_scale": torch.as_tensor(sidecars["scale8"], dtype=torch.float32).cpu(),
        "minimum_positive_residual": sidecars["minimum_positive_residual"],
        "encoder": encoder_state,
        "head": head_state,
        "shared": {name: value.detach().cpu() for name, value in sidecars["shared"].state_dict().items()},
        "private": {name: value.detach().cpu() for name, value in sidecars["private"].state_dict().items()},
    }, model_state_path)

    _write_jsonl(out / "training_ledger.jsonl", ledger["step_records"])
    _write_json(out / "training_ledger_summary.json", ledger)
    _write_json(out / "support_threshold.json", {
        "source": "support110_hdf5_authoritative_only",
        "target_tpr": SUPPORT_TARGET_TPR,
        "metrics": threshold_metrics,
        "scores_sha256": sha256_array(support_scores),
        "labels_sha256": sha256_bytes(canonical_json(support_labels).encode()),
        "balanced360_labels_used": False,
    })
    _write_jsonl(out / "support_scores.jsonl", support_rows)

    records: list[dict[str, Any]] = []
    prediction_path = out / "balanced360_predictions.jsonl"
    if prediction_path.exists():
        raise FileExistsError(prediction_path)
    with prediction_path.open("x", encoding="utf-8") as writer:
        for stream_index, shot_id in enumerate(balanced_ids):
            row, combined_logit = _score_static_shot(
                rt, shot_id=int(shot_id), encoder=encoder, head=head,
                sidecars=sidecars, center=center, scale=scale,
                threshold=threshold, device=device,
            )
            row["stream_index"] = int(stream_index)
            records.append(row)
            writer.write(canonical_json(row) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
            np.savez_compressed(
                out / "risk_traces" / f"{int(shot_id)}.npz",
                combined_logit=np.asarray(combined_logit, dtype=np.float32),
            )
    assert_prediction_seal(records, balanced_ids)
    prediction_hash = sha256_file(prediction_path)

    # The first access to balanced360 outcome metadata is deliberately below
    # the complete prediction seal.  This is the only evaluation label read.
    truth = rt.evidence.load_east_truth(balanced_ids)
    roster_contract = validate_roster_contract(
        support_ids, balanced_ids, support_labels,
        [bool(truth[int(shot_id)]["is_disrupt"]) for shot_id in balanced_ids],
    )
    metrics = rt.evidence.alarm_metrics(
        records, truth, minimum_lead_ms=MINIMUM_LEAD_MS,
    )
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "roster_snapshot.json", {
        "support_ids": support_ids,
        "support_hdf5_labels": support_labels,
        "balanced360_ids": balanced_ids,
        "balanced360_hdf5_labels": [
            bool(truth[int(shot_id)]["is_disrupt"]) for shot_id in balanced_ids
        ],
        "label_authority": "EAST HDF5 meta/IsDisrupt and meta/DownTime",
        "outcome_snapshot_stage": "after_complete_prediction_seal",
    })

    dependency_hashes = {}
    for path in _lineage_paths():
        if not path.is_file():
            raise FileNotFoundError(f"lineage dependency is missing: {path}")
        dependency_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    artifact_paths = [
        out / "config_snapshot.json", out / "training_ledger.jsonl",
        out / "training_ledger_summary.json", out / "support_threshold.json",
        out / "support_scores.jsonl", out / "balanced360_predictions.jsonl",
        out / "metrics.json", out / "roster_snapshot.json", model_state_path,
    ]
    artifact_hashes = {
        str(path.relative_to(out)): sha256_file(path)
        for path in artifact_paths if path.is_file()
    }
    lineage = {
        "schema_version": "east-target-only-lineage-v1",
        "contract_schema": config["schema_version"],
        "arm": arm,
        "seed": int(seed),
        "claim_boundary": config["claim_boundary"],
        "data_root": "<external-data-root>",
        "data_root_serialized": False,
        "native23": {"status": NATIVE23_STATUS},
        "roster_contract": roster_contract,
        "features": {
            "names": list(FEATURE_NAMES),
            "window": WINDOW,
            "architecture": "rebuilt17-patch-transformer-128x8-d64-l2-h4-v1",
        },
        "scaler": {
            "source": scaler_source,
            "center_sha256": sha256_array(center),
            "scale_sha256": sha256_array(scale),
            "source_checkpoint": portable_path(source_path),
            "source_checkpoint_sha256": sha256_file(source_path) if source_path else None,
        },
        "training": {
            "optimizer_steps": TARGET_STEPS,
            "batch_size": BATCH_SIZE,
            "sampler": "historical-shot-first-then-within-shot-window-v1",
            "positive_ttd_ms": list(POSITIVE_TTD_MS),
            "negative_ttd_ms": list(NEGATIVE_TTD_MS),
            "balanced360_labels_used": False,
            "ledger_sha256": sha256_file(out / "training_ledger.jsonl"),
        },
        "threshold": {
            "metrics": threshold_metrics,
            "threshold": threshold,
            "source": "support110_hdf5_authoritative_only",
            "balanced360_labels_used": False,
        },
        "alarm": {
            "parameters": ALARM_PARAMETERS,
            "implementation": "dual_threshold_alarm.dual_threshold_alarm",
            "online_updates": False,
        },
        "models": {
            "encoder_head_sha256": _model_hash(encoder, head),
            "shared_adapter_sha256": _model_hash(sidecars["shared"]),
            "private_adapter_sha256": _model_hash(sidecars["private"]),
            "state_artifact_sha256": sha256_file(model_state_path),
        },
        "sidecar_compute": sidecars["compute"],
        "seal": {
            "prediction_rows": len(records),
            "prediction_order_sha256": sha256_ids([int(row["shot"]) for row in records]),
            "prediction_ledger_sha256": prediction_hash,
            "outcome_read_after_complete_seal": True,
        },
        "evaluation": {
            "minimum_lead_ms": MINIMUM_LEAD_MS,
            "primary_metric": "operational_tpr",
            "metrics_path": "metrics.json",
        },
        "code_hashes": dependency_hashes,
        "artifact_hashes": artifact_hashes,
        "elapsed_s": float(time.time() - started),
    }
    _write_json(out / "lineage.json", lineage)
    summary = {
        "schema_version": "east-target-only-result-v1",
        "arm": arm,
        "seed": int(seed),
        "claim_boundary": config["claim_boundary"],
        "metrics": metrics,
        "threshold": threshold_metrics,
        "lineage_sha256": sha256_file(out / "lineage.json"),
        "prediction_ledger_sha256": prediction_hash,
        "balanced360_independent_test": False,
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--seed", type=int, choices=(131, 133, 137), required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(
        arm=args.arm, seed=args.seed, data_root=args.data_root,
        out=args.out, device=args.device,
        source_checkpoint=args.source_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
