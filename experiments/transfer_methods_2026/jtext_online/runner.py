#!/usr/bin/env python3
"""Run the hash-bound J-TEXT Adv25 TEST220 active diagnostic.

The runner loads three frozen ``max(25 ms, AdvancedTime)`` source models,
builds causal TEST220 traces, and executes a chronological delayed-feedback
Hedge replay.  It never uses the J-TEXT online state as an EAST transfer
parent.  Raw HDF5 remains external and the output directory must be empty.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
from jtext_online import controller, io  # noqa: E402
from jtext_online.labels import ADV25_POLICY, sha256_file  # noqa: E402


SEEDS = controller.SEEDS
SOURCE_CHECKPOINT_SHA256 = {
    131: "85eaa32b912adf83cd5bc794cba55f1271d51728c3f278b114b0f11bbf2bf91e",
    133: "9c2b908c2958b17f1b18275fb7aef0923aa129afd90f2e4b4497f81a4fabe783",
    137: "c1594751c72209eab1990cba309cdb49a33163e2c0a44f16abe397c3d961cc01",
}
DEFAULT_SOURCE_ROOT = (
    ROOT / "experiments/transfer_methods_2026/paper_grade_package/evidence"
    / "jtext_adv25_source_lineage"
)
DEFAULT_ADVANCED_TIME = ROOT / "shot_list/AdvancedTime_J-TEXT.json"


def _checkpoint_path(source_root: Path, seed: int) -> Path:
    candidates = (
        source_root / f"seed{seed}_source_checkpoint.pt",
        source_root / f"seed{seed}/source/source_checkpoint.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no Adv25 source checkpoint for seed {seed}; checked {list(candidates)}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_checkpoint(
    path: Path, seed: int, *, allow_unregistered: bool,
) -> dict[str, Any]:
    actual = sha256_file(path)
    expected = SOURCE_CHECKPOINT_SHA256[int(seed)]
    if not allow_unregistered and actual != expected:
        raise RuntimeError(
            f"Adv25 source SHA mismatch for seed {seed}: {actual} != {expected}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("training_label", {}).get("policy") != ADV25_POLICY:
        raise RuntimeError(f"seed {seed}: source checkpoint is not {ADV25_POLICY}")
    if int(payload.get("test_shards_read", -1)) != 0:
        raise RuntimeError(f"seed {seed}: source checkpoint read TEST220")
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"seed {seed}: source checkpoint seed mismatch")
    return payload


def _trace_for_signal(
    encoder: torch.nn.Module, head: torch.nn.Module, signal: io.SignalShot,
    *, device: torch.device, batch_size: int,
) -> np.ndarray:
    encoder.eval()
    head.eval()
    output = np.empty(len(signal.values), dtype=np.float32)
    for begin in range(0, len(signal.values), int(batch_size)):
        endpoints = range(begin, min(begin + int(batch_size), len(signal.values)))
        rows = [signal.causal_window(endpoint) for endpoint in endpoints]
        windows = torch.from_numpy(np.stack([row[0] for row in rows])).to(device)
        lengths = torch.as_tensor([row[1] for row in rows], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = head(encoder(windows, valid_lengths=lengths))
        output[begin : begin + len(rows)] = logits.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError(f"non-finite J-TEXT source trace for shot {signal.shot_id}")
    return output


def _load_models_and_traces(
    *, source_root: Path, signals: Mapping[int, io.SignalShot], device: torch.device,
    batch_size: int, allow_unregistered: bool,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, str], dict[int, dict[str, Any]]]:
    traces: dict[int, dict[int, np.ndarray]] = {}
    thresholds: dict[int, float] = {}
    checkpoint_hashes: dict[int, str] = {}
    receipts: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        checkpoint = _checkpoint_path(source_root, seed)
        payload = _validate_checkpoint(
            checkpoint, seed, allow_unregistered=allow_unregistered,
        )
        encoder, head, _ = GS.load_source_checkpoint(
            checkpoint, channels=io.FEATURE_COUNT, expected_seed=seed, device=device,
        )
        center = payload.get("scaler_center")
        scale = payload.get("scaler_scale")
        if not isinstance(center, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise RuntimeError(f"seed {seed}: source scaler is missing")
        center_np = center.detach().cpu().numpy()
        scale_np = scale.detach().cpu().numpy()
        source_signals = {
            shot: signal.scaled(center_np, scale_np)
            for shot, signal in signals.items()
        }
        traces[seed] = {
            shot: _trace_for_signal(
                encoder, head, source_signals[shot],
                device=device, batch_size=batch_size,
            )
            for shot in signals
        }
        threshold = float(payload.get("validation_threshold", float("nan")))
        if not 0.0 < threshold < 1.0:
            raise RuntimeError(f"seed {seed}: invalid VAL160 threshold")
        thresholds[seed] = threshold
        checkpoint_hashes[seed] = sha256_file(checkpoint)
        receipts[seed] = {
            "path": str(checkpoint), "sha256": checkpoint_hashes[seed],
            "registered_sha256": SOURCE_CHECKPOINT_SHA256[seed],
            "training_label_policy": payload["training_label"]["policy"],
            "validation_threshold": threshold,
            "test_shards_read": int(payload["test_shards_read"]),
            "feature_names": list(payload.get("feature_names", [])),
        }
    return traces, thresholds, checkpoint_hashes, receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    device = torch.device(args.device)
    roster_info = io.load_ordered_test220(ROOT)
    roster = roster_info.shots
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    advanced_time = Path(args.advanced_time).expanduser().resolve()
    if not advanced_time.is_file():
        raise FileNotFoundError(f"AdvancedTime JSON does not exist: {advanced_time}")
    signals = {
        shot: io.load_signal_only(data_root, shot)
        for shot in roster
    }
    traces, thresholds, checkpoint_hashes, checkpoint_receipts = _load_models_and_traces(
        source_root=source_root, signals=signals, device=device,
        batch_size=args.batch_size, allow_unregistered=args.allow_unregistered_checkpoint,
    )
    oracle = io.OrderedOutcomeOracle(
        data_root=data_root, roster=roster, advanced_time_path=advanced_time,
    )
    result = controller.run_ordered_stream(
        roster=roster, expert_traces=traces, thresholds=thresholds,
        signals=signals, oracle=oracle, budget_percent=args.budget,
        block_size=args.block_size, inter_eta=args.inter_eta,
        intra_eta=args.intra_eta, query_policy=args.query_policy,
        query_seed=args.query_seed,
    )
    manifest = {
        "schema_version": "jtext-adv25-active-mounted-v1",
        "claim_status": result["claim_status"],
        "method": result["method"],
        "representation_updated": result["representation_updated"],
        "jtext_online_state_used_for_east": False,
        "east_transfer_parent": "static_adv25_source_checkpoint_only",
        "ordered_test220": {
            "schema_version": roster_info.schema_version,
            "claim_status": roster_info.claim_status,
            "order_rule": roster_info.order_rule,
            "count": len(roster), "roster_sha256": roster_info.roster_sha256,
            "roster_file_sha256": roster_info.roster_file_sha256,
        },
        "label_contract": {
            "policy": ADV25_POLICY, "floor_ms": 25.0,
            "advanced_time_sha256": sha256_file(advanced_time),
            "outcome_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
        },
        "budget_percent": result["budget_percent"],
        "query_count": result["query_count"],
        "source_checkpoint_sha256": checkpoint_hashes,
        "source_checkpoint_receipts": checkpoint_receipts,
        "threshold_source": "each frozen source checkpoint validation_threshold selected on JTEXT VAL160",
        "thresholds": thresholds,
        "causal_order": "signal prefix -> predict -> seal -> query -> reveal -> next-block update",
        "test_outcomes_used_before_query": False,
        "test_outcomes_used_for_trace_or_query_score": False,
        "hdf5_outcomes_used_only_after_selected_query": True,
        "outcome_reads_during_stream": result["outcome_reads_during_stream"],
        "runtime": {
            "python": sys.executable, "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "claim_boundary": (
            "TEST220 participated in method development; this is a diagnostic replay. "
            "The online component changes frozen-expert decision weights only and "
            "is never an EAST transfer parent."
        ),
    }
    summary = dict(result)
    predictions = summary.pop("predictions")
    ledger = summary.pop("query_ledger")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "predictions.json").write_text(
        json.dumps(predictions, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "query_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in ledger),
        encoding="utf-8",
    )
    (args.output / "mainline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "manifest": manifest, "output": str(args.output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--advanced-time", type=Path, default=DEFAULT_ADVANCED_TIME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=(10, 20, 30), default=20)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--inter-eta", type=float, default=0.50)
    parser.add_argument("--intra-eta", type=float, default=0.0)
    parser.add_argument("--query-policy", choices=("active", "random"), default="active")
    parser.add_argument("--query-seed", type=int, default=131)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--allow-unregistered-checkpoint", action="store_true",
        help="allow a schema-compatible but non-release checkpoint for diagnostics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    summary = result["summary"]
    print(json.dumps({
        "output": result["output"], "static": summary["static"],
        "online": summary["online"], "query_count": summary["query_count"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
