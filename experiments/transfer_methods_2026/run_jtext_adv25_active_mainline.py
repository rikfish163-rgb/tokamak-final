#!/usr/bin/env python3
"""Run the retained J-TEXT Adv25 active-learning comparison.

This entry point is deliberately separate from the EAST transfer runner.  It
uses only the three *static* Adv25 J-TEXT source checkpoints, builds causal
TEST220 traces without opening outcome metadata, and then reuses the archived
delayed-feedback active-Hedge controller.  The controller changes only its
shot-level mixture weights; it does not change the Transformer representation.
That distinction is recorded in the result so this diagnostic cannot be
mistaken for evidence of neural representation learning.

The traces and result are constructed in /dev/shm and copied to the mounted
artifact root only after all checks pass.  HDF5 outcomes are accessed by the
controller only after each selected shot's prediction has been sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:  # Package import in tests; direct import when invoked as a script.
    from .mainline_paths import assert_final_artifact_root, assert_project_workspace_path
except ImportError:  # pragma: no cover - exercised by the CLI entry point
    from mainline_paths import assert_final_artifact_root, assert_project_workspace_path


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
from v3c_compat import jtext_online_io as JIO  # noqa: E402
from v3c_compat import run_jtext_online_active as V3C  # noqa: E402
from v3c_classfirst_compat import run_jtext_online_expert_mixture as MIX  # noqa: E402


SEEDS = (131, 133, 137)
ADV25_POLICY = "advanced_floor25"
TEST_COUNT = 220
SOURCE_PROFILES = {
    # The original J-TEXT active diagnostic used these VAL-selected source
    # checkpoints.  Keep it registered so old receipts remain reproducible.
    "jtext_exact": {
        131: "85eaa32b912adf83cd5bc794cba55f1271d51728c3f278b114b0f11bbf2bf91e",
        133: "9c2b908c2958b17f1b18275fb7aef0923aa129afd90f2e4b4497f81a4fabe783",
        137: "c1594751c72209eab1990cba309cdb49a33163e2c0a44f16abe397c3d961cc01",
    },
    # The source profile used by the retained EAST shot-first V3C replay.
    # These are also selected from source validation history, never from
    # TEST220 or balanced360.
    "east_success": {
        131: "1ab6b8e3b30992400ea6df8ffe235cb9d0023730f5a0c77d4de0f087ae85912d",
        133: "c8c5945f58a938d9c7d8197650b01b9dacf2be7cf0ae317825f422d1283142da",
        137: "5183dc09239409325dd4f10fde8a6af4b6da4e28accc9ff88b11ff07a4d826cf",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}:{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def _ordered_roster() -> tuple[int, ...]:
    # The current repository stores the authoritative 1353/160/220 split in
    # the canonical source-bundle index.  Do not fall back to ``splits/test``:
    # that legacy text file is a different 173-shot experiment.  The ordered
    # stream is derived from the predeclared canonical TEST role and carries no
    # outcome fields.
    roles = JIO.load_canonical_jtext_roles(ROOT)
    roster = tuple(sorted(int(shot) for shot in roles["test"]))
    if len(roster) != TEST_COUNT or len(set(roster)) != TEST_COUNT:
        raise RuntimeError("canonical J-TEXT TEST220 roster is not valid")
    if tuple(roster) != tuple(roles["test"]):
        raise RuntimeError("canonical J-TEXT TEST220 role is not shot ordered")
    if any(set(roles[left]) & set(roles[right]) for left, right in (
        ("train", "val"), ("train", "test"), ("val", "test"),
    )):
        raise RuntimeError("canonical J-TEXT roles overlap")
    return roster


def _scaled_loader(data_root: Path, center: np.ndarray, scale: np.ndarray):
    def load(shot_id: int) -> JIO.JTextSignalShot:
        raw = JIO.load_jtext_signal_only(data_root, int(shot_id))
        values = np.clip((raw.values - center) / scale, -10.0, 10.0)
        values = np.where(raw.availability, values, 0.0).astype(np.float32)
        return JIO.JTextSignalShot(
            shot_id=raw.shot_id, values=values,
            availability=raw.availability, times_s=raw.times_s,
            sample_rate_hz=raw.sample_rate_hz, start_time_s=raw.start_time_s,
        )
    return load


def _validate_checkpoint(
    path: Path, seed: int, *, expected_source_sha: str,
) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_source_sha:
        raise RuntimeError(
            f"Adv25 source SHA mismatch for seed {seed}: {actual} != "
            f"{expected_source_sha}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("training_label", {}).get("policy") != ADV25_POLICY:
        raise RuntimeError(f"seed {seed}: source checkpoint is not Adv25")
    if int(payload.get("test_shards_read", -1)) != 0:
        raise RuntimeError(f"seed {seed}: source checkpoint read TEST220")
    return payload


def _write_trace_cache(
    *, seed: int, checkpoint: Path, data_root: Path, roster: Sequence[int],
    cache_dir: Path, trace_path: Path, expected_source_sha: str,
) -> dict[str, Any]:
    payload = _validate_checkpoint(
        checkpoint, seed, expected_source_sha=expected_source_sha,
    )
    device = torch.device("cuda")
    encoder, head, _ = GS.load_source_checkpoint(
        checkpoint, channels=len(GS.RH.REBUILD), expected_seed=seed, device=device,
    )
    center = payload["scaler_center"].detach().cpu().numpy()
    scale = payload["scaler_scale"].detach().cpu().numpy()
    loader = _scaled_loader(data_root, center, scale)
    builder = V3C._selected_source_trace_builder(encoder, head, device=device)
    cache_dir.mkdir(parents=True, exist_ok=False)
    offsets = [0]
    all_logits: list[np.ndarray] = []
    cache_path = cache_dir / "endpoint_receipts.jsonl"
    with cache_path.open("x", encoding="utf-8") as handle:
        for index, shot in enumerate(roster):
            signal = loader(int(shot))
            logits = np.asarray(builder(signal), dtype=np.float32).reshape(-1)
            if logits.shape != (len(signal.values),) or not np.isfinite(logits).all():
                raise RuntimeError(f"seed {seed}, shot {shot}: invalid causal trace")
            all_logits.append(logits)
            offsets.append(offsets[-1] + len(logits))
            for endpoint, value in enumerate(logits):
                handle.write(json.dumps({
                    "event": "endpoint_predict", "shot": int(shot),
                    "endpoint": int(endpoint), "source_logit": float(value),
                }, sort_keys=True, allow_nan=False) + "\n")
    flat = np.concatenate(all_logits).astype(np.float32, copy=False)
    np.savez_compressed(
        trace_path, shot_ids=np.asarray(roster, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64), source_logits=flat,
    )
    return {
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "endpoint_receipts": str(cache_path),
        "endpoint_receipts_sha256": sha256_file(cache_path),
        "trace_npz": str(trace_path),
        "trace_npz_sha256": sha256_file(trace_path),
        "trace_array_sha256": sha256_array(flat),
        "shot_count": len(roster),
        "endpoint_count": int(len(flat)),
        "outcome_fields_present": False,
    }


def _copy_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.rmdir()
    shutil.copytree(source, destination)
    source_files = sorted(
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    )
    destination_files = sorted(
        path.relative_to(destination) for path in destination.rglob("*") if path.is_file()
    )
    if source_files != destination_files:
        raise RuntimeError("J-TEXT artifact inventory changed during publication")
    for relative in source_files:
        if sha256_file(source / relative) != sha256_file(destination / relative):
            raise RuntimeError(f"J-TEXT artifact hash mismatch: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    if executable.parent.parent.name != "rl_env":
        raise RuntimeError(f"J-TEXT active mainline requires rl_env, got {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("J-TEXT active mainline requires CUDA")
    data_root = assert_project_workspace_path(
        args.data_root, label="J-TEXT data root"
    )
    source_root = assert_project_workspace_path(
        args.source_root, label="J-TEXT source root"
    )
    expected_source = SOURCE_PROFILES[args.source_profile]
    final = assert_final_artifact_root(args.output)
    if final.exists() and any(final.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {final}")
    roster = _ordered_roster()
    checkpoints = {
        seed: source_root / f"seed{seed}" / "source" / "source_checkpoint.pt"
        for seed in SEEDS
    }
    for seed, path in checkpoints.items():
        if path.is_symlink():
            raise RuntimeError(f"J-TEXT seed{seed} source checkpoint must not be a symlink: {path}")
        path = assert_project_workspace_path(
            path, label=f"J-TEXT seed{seed} source checkpoint"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing Adv25 source checkpoint for seed {seed}: {path}")

    scratch_parent = args.scratch_root.expanduser().resolve()
    if not scratch_parent.is_dir() or scratch_parent == Path("/"):
        raise ValueError(f"scratch root must be an existing directory: {scratch_parent}")
    scratch = Path(tempfile.mkdtemp(prefix="jtext_adv25_active_", dir=str(scratch_parent)))
    try:
        cache_paths: dict[int, Path] = {}
        trace_receipts: dict[str, Any] = {}
        for seed in SEEDS:
            cache = scratch / f"cache_seed{seed}"
            trace = scratch / f"source_trace_seed{seed}.npz"
            print(f"[JTEXT-ACTIVE] building causal Adv25 trace seed={seed}", flush=True)
            trace_receipts[str(seed)] = _write_trace_cache(
                seed=seed, checkpoint=checkpoints[seed], data_root=data_root,
                roster=roster, cache_dir=cache, trace_path=trace,
                expected_source_sha=expected_source[seed],
            )
            cache_paths[seed] = cache

        result_dir = scratch / "mixture_result"
        result = MIX.run(
            cache_paths=cache_paths, checkpoints=checkpoints, data_root=data_root,
            output=result_dir, budget_percent=int(args.budget), block_size=20,
            # The registered positive J-TEXT result is inter-shot active Hedge;
            # intra-shot fast-weight ablation is kept at zero here so the
            # online gain is attributable to delayed queried feedback.
            intra_eta=0.0, inter_eta=0.50,
            query_policy="active", query_seed=131,
        )
        summary = dict(result)
        if int(summary.get("query_count", -1)) != {
            10: 22, 20: 44, 30: 66,
        }[int(args.budget)]:
            raise RuntimeError("J-TEXT query budget contract failed")
        if int(summary.get("shot_count", -1)) != TEST_COUNT:
            raise RuntimeError("J-TEXT active result did not process TEST220")
        manifest = {
            "schema_version": "jtext-adv25-active-mounted-v1",
            "claim_status": "TEST_DRIVEN_EXPLORATION",
            "method": "delayed-feedback-active-hedge-inter-only",
            "representation_updated": False,
            "jtext_online_state_used_for_east": False,
            "east_transfer_parent": "static_adv25_source_checkpoint_only",
            "budget_percent": int(args.budget),
            "query_count": int(summary["query_count"]),
            "ordered_test220": list(roster),
            "ordered_test220_sha256": sha256_json(list(roster)),
            "ordered_roster_source": (
                "artifacts/canonical/jtext_source_bundle_p131_v1/index.json"
                "::split_shot_ids.test"
            ),
            "source_root": str(source_root),
            "source_profile": args.source_profile,
            "source_checkpoint_sha256": {
                str(seed): sha256_file(checkpoints[seed]) for seed in SEEDS
            },
            "trace_receipts": trace_receipts,
            "causal_order": "signal prefix -> predict -> seal -> query -> reveal -> inter update",
            "causal_order_detail": {
                "shot": [
                    "observe the current causal TEST220 signal prefix",
                    "predict and seal the complete shot trace",
                    "select queries without reading the current outcome",
                    "reveal HDF5 outcome only for selected shots",
                    "update the mixture before the next shot/block",
                ],
                "outcome_access_before_query": "none",
            },
            # Keep the negative access assertion explicit in the published
            # receipt.  The longer legacy field below remains for old audits.
            "test_outcomes_used_before_query": False,
            "test_outcomes_used_for_trace_or_query_score": False,
            "hdf5_outcomes_used_only_after_selected_query": True,
            "threshold_source": "per-seed static Adv25 checkpoint validation_threshold from JTEXT VAL160",
            "runtime": {
                "python": str(executable), "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "claim_boundary": (
                "TEST220 was used during method development; this is a causal "
                "diagnostic, and the active component changes decision weights "
                "rather than the Transformer representation."
            ),
        }
        (result_dir / "mainline_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Keep compact reproducible traces and the mixture's ledgers, while
        # never publishing the temporary source-code tree.
        publish = scratch / "publish"
        publish.mkdir()
        for name in ("summary.json", "predictions.json", "query_ledger.jsonl", "mainline_manifest.json"):
            shutil.copy2(result_dir / name, publish / name)
        for seed in SEEDS:
            shutil.copy2(scratch / f"source_trace_seed{seed}.npz", publish / f"source_trace_seed{seed}.npz")
            shutil.copy2(cache_paths[seed] / "endpoint_receipts.jsonl", publish / f"endpoint_receipts_seed{seed}.jsonl")
        _copy_verified(publish, final)
        published = json.loads((final / "summary.json").read_text(encoding="utf-8"))
        return {"output": str(final), "summary": published, "manifest": manifest}
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-profile", choices=tuple(SOURCE_PROFILES), default="jtext_exact",
        help=(
            "registered VAL-selected static source checkpoint profile; use "
            "jtext_exact with the canonical J-TEXT source-root unless an "
            "explicit EAST-success source bundle is supplied"
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=(10, 20, 30), default=20)
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    summary = result["summary"]
    print(json.dumps({
        "output": result["output"], "static": summary["static"],
        "online": summary["online"],
        "delta": summary["delta"],
        "query_count": summary["query_count"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
