#!/usr/bin/env python3
"""Run the retained two-timescale J-TEXT online-active line.

This entry point is intentionally separate from the EAST runner.  It starts
from a static Adv25 source checkpoint, runs the historical causal residual
adapter (including the post-reveal error-memory path), and writes the result
only to the mounted artifact filesystem.  The online J-TEXT state is never a
permitted EAST parent.

The implementation uses a temporary directory in ``/dev/shm`` while the
stream is running.  The complete directory is copied to the final mounted
root only after the stream has finished; this avoids high-frequency JSONL
writes to the NTFS project volume without silently changing the experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
import rebuilt_harness as RH  # noqa: E402
import train_jtext_source_transformer as SOURCE  # noqa: E402
from mainline_paths import assert_final_artifact_root  # noqa: E402
from v3c_compat import jtext_online_io as V3C_IO  # noqa: E402
from v3c_compat import run_jtext_online_active as V3C_RUNNER  # noqa: E402


SEEDS = (131, 133, 137)
ADVANCED_POLICY = "advanced_floor25"
TEST_COUNT = 220
CLAIM_STATUS = "TEST_DRIVEN_EXPLORATION"
RESEARCH_CLAIM = "JTEXT_TEST220_ORDERED_PREQUENTIAL_DIAGNOSTIC"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"non-object JSONL row in {path}")
    return rows


def _ordered_test220(root: Path) -> tuple[int, ...]:
    index_path = root / "artifacts/canonical/jtext_source_bundle_p131_v1/index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    raw = payload.get("split_shot_ids", {}).get("test")
    if not isinstance(raw, list):
        raise RuntimeError("canonical J-TEXT index has no test roster")
    roster = tuple(int(value) for value in raw)
    if len(roster) != TEST_COUNT or len(set(roster)) != TEST_COUNT:
        raise RuntimeError("canonical TEST220 roster count/uniqueness changed")
    if roster != tuple(sorted(roster)):
        raise RuntimeError("J-TEXT online roster must be ascending by shot ID")
    return roster


def _validate_source(checkpoint: Mapping[str, Any], *, seed: int, root: Path) -> None:
    if int(checkpoint.get("seed", -1)) != int(seed):
        raise RuntimeError("source checkpoint seed does not match requested seed")
    if checkpoint.get("training_label", {}).get("policy") != ADVANCED_POLICY:
        raise RuntimeError("J-TEXT online parent is not the Adv25 source checkpoint")
    if int(checkpoint.get("test_shards_read", -1)) != 0:
        raise RuntimeError("source checkpoint was selected using TEST220")
    label = checkpoint.get("training_label")
    metadata = label.get("advanced_time_metadata") if isinstance(label, dict) else None
    if not isinstance(metadata, dict):
        raise RuntimeError("Adv25 source checkpoint lacks label metadata")
    if metadata.get("policy") != "max(floor_ms, AdvancedTime_ms)":
        raise RuntimeError("source checkpoint has an unexpected label policy")
    if metadata.get("ambiguous_context_until_ms") is not None:
        raise RuntimeError("ambiguous-context source is not the selected mainline")
    advanced_path = root / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"
    expected_sha = str(label.get("advanced_time_source_sha256", ""))
    if expected_sha != sha256_file(advanced_path):
        raise RuntimeError("source AdvancedTime SHA differs from the workspace")


def _scaled_loader(
    data_root: Path, center: np.ndarray, scale: np.ndarray,
):
    def load(shot_id: int) -> V3C_IO.JTextSignalShot:
        raw = V3C_IO.load_jtext_signal_only(data_root, int(shot_id))
        values = np.clip((raw.values - center) / scale, -10.0, 10.0)
        values = np.where(raw.availability, values, 0.0).astype(np.float32)
        return V3C_IO.JTextSignalShot(
            shot_id=raw.shot_id, values=values,
            availability=raw.availability, times_s=raw.times_s,
            sample_rate_hz=raw.sample_rate_hz, start_time_s=raw.start_time_s,
        )
    return load


def _outcome(data_root: Path, shot_id: int) -> tuple[bool, float | None]:
    path = V3C_IO.jtext_shot_path(data_root, int(shot_id))
    with h5py.File(path, "r") as handle:
        disruptive = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    if disruptive and (not len(down) or not np.isfinite(float(down[0]))):
        raise RuntimeError(f"disruptive J-TEXT shot has invalid DownTime: {shot_id}")
    return disruptive, float(down[0]) if disruptive else None


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s) or not len(y) or not np.isfinite(s).all():
        raise ValueError("AUC inputs are not finite and aligned")
    positive = s[y]
    negative = s[~y]
    if not len(positive) or not len(negative):
        return None
    difference = positive[:, None] - negative[None, :]
    return float(np.mean(difference > 0.0) + 0.5 * np.mean(difference == 0.0))


def _metrics(
    rows_by_shot: Mapping[int, Sequence[Mapping[str, Any]]],
    roster: Sequence[int], *, data_root: Path, threshold: float,
) -> dict[str, Any]:
    labels: list[bool] = []
    scores: list[float] = []
    alarmed: list[bool] = []
    leads: list[float] = []
    timely_leads: list[float] = []
    early_shots: list[int] = []
    shot_rows: list[dict[str, Any]] = []
    for shot in roster:
        endpoint_rows = [
            row for row in rows_by_shot[int(shot)]
            if row.get("event") == "endpoint_predict"
        ]
        endpoint_rows.sort(key=lambda row: int(row["endpoint"]))
        if not endpoint_rows:
            raise RuntimeError(f"missing endpoint predictions for J-TEXT shot {shot}")
        risks = np.asarray([float(row["risk"]) for row in endpoint_rows])
        endpoints = np.asarray([int(row["endpoint"]) for row in endpoint_rows])
        if not np.isfinite(risks).all() or not np.array_equal(
            endpoints, np.arange(len(endpoints), dtype=np.int64)
        ):
            raise RuntimeError(f"endpoint trace is incomplete for J-TEXT shot {shot}")
        disruptive, down = _outcome(data_root, int(shot))
        alarm_positions = np.flatnonzero(risks >= float(threshold))
        alarm_position = int(alarm_positions[0]) if len(alarm_positions) else None
        signal = V3C_IO.load_jtext_signal_only(data_root, int(shot))
        alarm_time = None
        lead_ms = None
        if alarm_position is not None:
            alarm_time = float(signal.start_time_s) + (
                float(endpoints[alarm_position]) / float(signal.sample_rate_hz)
            )
            if disruptive:
                lead_ms = (float(down) - alarm_time) * 1000.0
        timely = bool(disruptive and lead_ms is not None and lead_ms > 10.0)
        if disruptive and lead_ms is not None:
            leads.append(float(lead_ms))
            if lead_ms > 3000.0:
                early_shots.append(int(shot))
        if timely:
            timely_leads.append(float(lead_ms))
        labels.append(bool(disruptive))
        scores.append(float(np.max(risks)))
        alarmed.append(alarm_position is not None)
        shot_rows.append({
            "shot": int(shot),
            "is_disrupt": bool(disruptive),
            "score": float(np.max(risks)),
            "alarmed": bool(alarm_position is not None),
            "alarm_endpoint": alarm_position,
            "alarm_time_s": alarm_time,
            "lead_ms": lead_ms,
            "timely": timely,
        })
    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(alarmed, dtype=bool)
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))
    operational_tp = len(timely_leads)
    disruptions = int(np.sum(y))
    safe = int(np.sum(~y))
    return {
        "n_shots": len(roster),
        "n_disruptions": disruptions,
        "n_non_disruptions": safe,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "tardy": tp - operational_tp,
        "operational_tp": operational_tp,
        "operational_fn": disruptions - operational_tp,
        "raw_tpr": float(tp / max(disruptions, 1)),
        "operational_tpr": float(operational_tp / max(disruptions, 1)),
        "tpr": float(tp / max(disruptions, 1)),
        "fpr": float(fp / max(safe, 1)),
        "auc": _auc(y, scores),
        "median_lead_ms": (
            float(np.median(timely_leads)) if timely_leads else None
        ),
        "raw_alarm_lead_ms": {
            "n": len(leads),
            "median_ms": float(np.median(leads)) if leads else None,
        },
        "operational_tp_lead_ms": {
            "n": len(timely_leads),
            "median_ms": (
                float(np.median(timely_leads)) if timely_leads else None
            ),
        },
        "early_over_3000_ms_count": len(early_shots),
        "early_over_3000_ms_shots": early_shots,
        "minimum_lead_rule": "lead_ms > 10",
        "threshold": float(threshold),
        "shot_rows": shot_rows,
    }


def _copy_tree_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite final artifact: {destination}")
        destination.rmdir()
    shutil.copytree(source, destination)
    source_files = sorted(
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    )
    destination_files = sorted(
        path.relative_to(destination) for path in destination.rglob("*") if path.is_file()
    )
    if source_files != destination_files:
        raise RuntimeError("scratch and final artifact file inventories differ")
    for relative in source_files:
        if sha256_file(source / relative) != sha256_file(destination / relative):
            raise RuntimeError(f"final artifact copy hash mismatch: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if Path(sys.executable).resolve().parent.parent.name != "rl_env":
        raise RuntimeError("J-TEXT V3C runner is locked to the rl_env Python")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if int(args.seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    data_root = args.data_root.expanduser().resolve()
    checkpoint_path = args.source_checkpoint.expanduser().resolve()
    final_output = assert_final_artifact_root(args.output)
    if final_output.exists() and any(final_output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {final_output}")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"source checkpoint is missing or symlinked: {checkpoint_path}")

    device = torch.device(args.device)
    encoder, head, checkpoint = GS.load_source_checkpoint(
        checkpoint_path, channels=len(RH.REBUILD), expected_seed=args.seed,
        device=device,
    )
    _validate_source(checkpoint, seed=args.seed, root=ROOT)
    roster = _ordered_test220(ROOT)
    center = checkpoint["scaler_center"].detach().cpu().numpy()
    scale = checkpoint["scaler_scale"].detach().cpu().numpy()
    signal_loader = _scaled_loader(data_root, center, scale)

    def oracle_factory(remaining: tuple[int, ...]):
        return V3C_IO.JTextLazyOutcomeOracle(
            data_root=data_root, ordered_roster=remaining,
            advanced_time_path=ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json",
        )

    controller = V3C_RUNNER.build_cli_controller(
        seed=args.seed, encoder=encoder, head=head, device=device, advanced=True,
    )
    parent_hash = controller.persistent_hash()
    scratch_parent = Path(args.scratch_root).expanduser().resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(
        prefix=f"jtext_v3c_seed{int(args.seed)}_",
        dir=str(scratch_parent),
    ))
    try:
        result = V3C_RUNNER.run_jtext_online_stream(
            ordered_shots=roster,
            signal_loader=signal_loader,
            source_trace_builder=V3C_RUNNER._selected_source_trace_builder(
                encoder, head, device=device,
            ),
            oracle_factory=oracle_factory,
            controller=controller,
            arm=args.arm,
            budget_percent=args.budget,
            seed=args.seed,
            output=scratch,
            alarm_threshold=float(checkpoint["validation_threshold"]),
            update_only_on_operational_error=bool(
                args.update_only_on_operational_error
            ),
        )
        summary = dict(result["summary"])
        if not bool(summary.get("completed")):
            raise RuntimeError("J-TEXT online stream did not complete")
        endpoint_rows = _read_jsonl(scratch / "endpoint_receipts.jsonl")
        by_shot: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in endpoint_rows:
            if row.get("event") == "endpoint_predict":
                by_shot[int(row["shot"])].append(row)
        # The source trace and the adjusted trace are in the same endpoint
        # receipt.  Rebuild the source view without reading outcomes.
        static_by_shot: dict[int, list[dict[str, Any]]] = defaultdict(list)
        online_by_shot: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for shot, rows in by_shot.items():
            for row in rows:
                static = dict(row)
                static["risk"] = float(1.0 / (1.0 + np.exp(-float(row["source_logit"]))))
                online = dict(row)
                static_by_shot[shot].append(static)
                online_by_shot[shot].append(online)
        threshold = float(checkpoint["validation_threshold"])
        metrics = {
            "static": _metrics(
                static_by_shot, roster, data_root=data_root, threshold=threshold,
            ),
            "online": _metrics(
                online_by_shot, roster, data_root=data_root, threshold=threshold,
            ),
        }
        metrics["delta_online_minus_static"] = {
            key: (
                None if metrics["online"].get(key) is None
                or metrics["static"].get(key) is None
                else metrics["online"][key] - metrics["static"][key]
            )
            for key in ("raw_tpr", "operational_tpr", "fpr", "auc")
        }
        metrics["delta_online_minus_static"]["j_operational"] = (
            metrics["online"]["operational_tpr"] - metrics["online"]["fpr"]
            - metrics["static"]["operational_tpr"] + metrics["static"]["fpr"]
        )
        # ``shot_rows`` are useful for analysis but duplicate the full endpoint
        # ledger only at shot level.  Keep them in metrics.json and a compact
        # top-level summary; do not use them for any update decision.
        summary["metrics"] = metrics
        summary["claim_status"] = CLAIM_STATUS
        summary["research_claim"] = RESEARCH_CLAIM
        summary["parent_contract"] = {
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "jtext_online_state_used_for_east": False,
            "source_parent_adapter_sha256": parent_hash,
            "source_model_sha256_before_after": module_hash(nn.Sequential(encoder, head)),
            "label_policy": ADVANCED_POLICY,
        }
        _write_json(scratch / "metrics.json", metrics)
        _write_json(scratch / "summary.json", summary)
        manifest = {
            "schema_version": "jtext-v3c-mounted-run-v1",
            "claim_status": CLAIM_STATUS,
            "research_claim": RESEARCH_CLAIM,
            "seed": int(args.seed),
            "arm": args.arm,
            "budget_percent": int(args.budget),
            "data_root": str(data_root),
            "ordered_test220": list(roster),
            "ordered_test220_sha256": sha256_json(list(roster)),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "advanced_time_path": str(ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"),
            "advanced_time_sha256": sha256_file(
                ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"
            ),
            "threshold": threshold,
            "threshold_source": "J-TEXT VAL160 embedded in static Adv25 checkpoint",
            "causal_order": "signal prefix -> predict -> endpoint seal -> query -> reveal -> update",
            "test_outcomes_read_before_query": False,
            "final_evaluation_opened_after_stream": True,
            "jtext_online_state_used_for_east": False,
            "online_parameter_count": int(controller.online_parameter_count),
            "update_only_on_operational_error": bool(
                args.update_only_on_operational_error
            ),
            "parent_adapter_sha256_before": parent_hash,
            "parent_adapter_sha256_after": controller.persistent_hash(),
            "source_model_sha256_before_after": module_hash(nn.Sequential(encoder, head)),
            "runtime": {
                "python": sys.executable,
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "git_head": _git_head(),
            },
            "compatibility_code": {
                "runner": sha256_file(HERE / "v3c_compat/run_jtext_online_active.py"),
                "adapter": sha256_file(HERE / "v3c_compat/jtext_online_adapter.py"),
                "io": sha256_file(HERE / "v3c_compat/jtext_online_io.py"),
            },
            "metrics_file_sha256": sha256_file(scratch / "metrics.json"),
            "query_count": int(summary["query_count"]),
            "accepted_updates": int(summary["accepted_updates"]),
            "claim_boundary": (
                "TEST220 was used for method development; this is a causal diagnostic, "
                "not independent future-stream confirmation."
            ),
        }
        _write_json(scratch / "run_manifest.json", manifest)
        _copy_tree_verified(scratch, final_output)
        final_summary = json.loads((final_output / "summary.json").read_text())
        if final_summary.get("metrics", {}).get("online", {}).get("n_shots") != TEST_COUNT:
            raise RuntimeError("final artifact summary failed TEST220 verification")
        return {
            "output": str(final_output),
            "summary": final_summary,
            "metrics": metrics,
        }
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=False)


def _git_head() -> str | None:
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--arm", choices=V3C_RUNNER.ARMS, default="full_active_advanced")
    parser.add_argument("--budget", type=int, choices=(10, 20, 30), default=20)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument(
        "--update-only-on-operational-error",
        action="store_true",
        help=(
            "after reveal, protect correct TP/TN shots from persistent neural "
            "gradient updates; FP/FN/TARDY remain eligible"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps({
        "output": result["output"],
        "static": result["metrics"]["static"],
        "online": result["metrics"]["online"],
        "delta": result["metrics"]["delta_online_minus_static"],
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
