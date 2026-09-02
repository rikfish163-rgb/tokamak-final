#!/usr/bin/env python3
"""Reproduce the retained Adv25 -> EAST V3C online diagnostic lineage.

The historical V3C parent is deliberately isolated from the newer development
modules because a support-sampler change altered the parent hash and removed
the previously observed gain.  This entry point runs the exact compatibility
snapshot in a temporary ``/dev/shm`` code tree, then copies the complete,
hash-verified result to the mounted project disk.

The runner is intentionally strict:

* only ``rl_env`` Python with CUDA is accepted;
* source, support, and chronological balanced360 rosters are explicit;
* the compatibility code is hashed before execution;
* no output is written to ``/home`` or ``/dev`` as a final artifact;
* a non-empty destination is never overwritten;
* the historical expected parent/metric receipt is checked before publication.

This is a diagnostic replay because balanced360 was used for method
development.  It is not an independent future-test runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch

from mainline_paths import assert_final_artifact_root


REPO_ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
COMPAT_DIR = REPO_ROOT / "experiments" / "transfer_methods_2026" / "v3c_classfirst_compat"
DATA_ROOT_DEFAULT = Path(os.environ.get("FUSION_DATA_ROOT", REPO_ROOT / "data"))
SEEDS = (131, 133, 137)
RUN_MODES = ("locked", "development")
EXPECTED_SOURCE_SHA = {
    131: "85eaa32b912adf83cd5bc794cba55f1271d51728c3f278b114b0f11bbf2bf91e",
    133: "9c2b908c2958b17f1b18275fb7aef0923aa129afd90f2e4b4497f81a4fabe783",
    137: "c1594751c72209eab1990cba309cdb49a33163e2c0a44f16abe397c3d961cc01",
}
EXPECTED_PARENT_SHA = {
    131: "d4d52def0fd46446b7dcadd0c4fe458edde42ade759259bfb166d91b99ef34a2",
    133: "fdb37d8dc9770c2c15dad93604dfcc5a13bdc9681a1526cdc293d5137613dcb1",
    137: "6c16f3d1581538765961bd0c9baeb3bd78fc054c467ea00f6e7ec16df0a933e1",
}


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


def _file_inventory(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )


def _hash_inventory(root: Path) -> dict[str, str]:
    # The compatibility runtime deliberately copies Python implementation
    # files only.  README/provenance text is kept in the repository but is not
    # imported by the historical runtime, so it must not make the copy check
    # fail spuriously.
    files = sorted(
        str(path.relative_to(root))
        for path in root.glob("*.py")
        if path.is_file()
    )
    return {relative: sha256_file(root / relative) for relative in files}


def _require_rl_env_cuda() -> None:
    executable = Path(sys.executable).resolve()
    if executable.parent.parent.name != "rl_env":
        raise RuntimeError(
            "EAST V3C compatibility replay is locked to the rl_env Python; "
            f"got {executable}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("EAST V3C compatibility replay requires CUDA")


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    _require_rl_env_cuda()
    if int(args.seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    source = args.source_checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    final = assert_final_artifact_root(args.output)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"source checkpoint must be a regular file: {source}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"EAST data root is missing: {data_root}")
    if not COMPAT_DIR.is_dir():
        raise FileNotFoundError(f"compatibility snapshot is missing: {COMPAT_DIR}")
    expected = EXPECTED_SOURCE_SHA[int(args.seed)]
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(
            f"Adv25 source checkpoint SHA mismatch for seed {args.seed}: "
            f"{actual} != {expected}"
        )
    if final.exists() and any(final.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty final artifact: {final}")
    return source, data_root, final


def _prepare_runtime_tree(scratch_root: Path) -> tuple[Path, dict[str, str]]:
    """Copy code/config into a root where historical ``__file__`` paths work."""

    runtime = Path(tempfile.mkdtemp(prefix="east_v3c_compat_", dir=str(scratch_root)))
    runtime_modules = runtime / "experiments" / "transfer_methods_2026"
    runtime_modules.mkdir(parents=True)
    for path in sorted(COMPAT_DIR.glob("*.py")):
        shutil.copy2(path, runtime_modules / path.name)
    # The historical modules resolve config and shot-list paths relative to
    # their project root.  These are small, versioned contracts, not data.
    for name in ("config", "shot_list", "splits"):
        source = REPO_ROOT / name
        if not source.is_dir():
            raise FileNotFoundError(f"required project contract is missing: {source}")
        shutil.copytree(source, runtime / name)
    copied_hashes = _hash_inventory(runtime_modules)
    expected_hashes = _hash_inventory(COMPAT_DIR)
    if copied_hashes != expected_hashes:
        raise RuntimeError("compatibility code copy failed its hash inventory check")
    return runtime, copied_hashes


def _run_subprocess(command: Sequence[str], *, env: Mapping[str, str]) -> None:
    print("[EAST-V3C] " + " ".join(command), flush=True)
    subprocess.run(list(command), env=dict(env), check=True)


def _unified_command(
    *, python: str, runtime: Path, source: Path, a1: Path, adapter: Path,
    output: Path, seed: int, mode: str,
) -> list[str]:
    script = runtime / "experiments/transfer_methods_2026/run_unified_online_active.py"
    if mode not in RUN_MODES:
        raise ValueError(f"unknown EAST V3C run mode {mode!r}; expected {RUN_MODES}")
    # The two command families are intentionally explicit.  The locked
    # family contains only the pre-registered V8/V3C values and no
    # development-only selector, episode, or token-budget arguments.  The
    # development family is retained for diagnosis, but it may not be used as
    # the mainline evidence because balanced360 is allowed to influence its
    # choice.  Changing either family creates a new lineage.
    if mode == "locked":
        values: list[str] = [
            python, str(script), "--seed", str(seed), "--out", str(output),
            "--max-shots", "360", "--private-support-epochs", "8",
            "--adaptation-events", "64", "--intra-steps", "1", "--inter-steps", "2",
            "--intra-lr", "0.0001", "--private-intra-lr", "0.0001",
            "--intra-physics-weight", "0.10", "--inter-lr", "0.0005",
            "--private-inter-lr", "0.0002", "--alarm-lr", "0.02",
            "--query-block-size", "20", "--query-quota", "4",
            "--query-history-size", "80", "--query-quantile", "0.80",
            "--query-uncertainty-weight", "0.65", "--query-value-lr", "0.05",
            "--novel-early-min-physics-support", "0.0",
            "--locked-best-v8", "--source-checkpoint", str(source),
            "--a1-dir", str(a1), "--adapter-dir", str(adapter),
        ]
    else:
        values = [
            python, str(script), "--seed", str(seed), "--out", str(output),
            "--max-shots", "360", "--private-support-epochs", "8",
            "--adaptation-events", "64", "--intra-steps", "1", "--inter-steps", "2",
            "--intra-lr", "0.0001", "--private-intra-lr", "0.0001",
            "--intra-physics-weight", "0.10", "--inter-lr", "0.0005",
            "--private-inter-lr", "0.0002", "--alarm-lr", "0.02",
            "--query-block-size", "20", "--query-quota", "6",
            "--query-history-size", "160", "--query-quantile", "0.70",
            "--query-uncertainty-weight", "0.55", "--query-value-lr", "0.05",
            "--novel-early-min-physics-support", "0.75",
            "--novel-early-min-trend", "0.4", "--causal-error-memory",
            "--error-memory-radius", "0.35", "--error-memory-min-support", "2",
            "--error-memory-weight", "0.0", "--query-initial-cutoff", "0.35",
            "--query-token-capacity", "12", "--strict-window-active",
            "--window-candidate-stride", "8", "--window-update-rate", "0.25",
            "--window-token-capacity", "4", "--window-history-size", "64",
            "--window-value-quantile", "0.80", "--window-initial-cutoff", "0.40",
            "--window-minimum-history", "8", "--window-minimum-gap", "8",
            "--typed-endpoint-memory", "--endpoint-memory-radius", "0.35",
            "--endpoint-memory-min-shots", "2", "--endpoint-memory-positive", "0.35",
            "--endpoint-memory-negative", "0.25", "--endpoint-memory-physics-gate", "0.50",
            "--endpoint-memory-protection-ratio", "1.0", "--endpoint-memory-stride", "16",
            "--episode-value-v3", "--episode-fp-threshold", "0.85",
            "--episode-minimum-confidence", "0.75", "--episode-max-extra-points", "8",
            "--episode-min-history", "12", "--episode-min-per-class", "3",
            "--episode-high-escape-margin", "99.0", "--episode-physics-escape", "0.85",
            "--episode-protected-sequence", "--episode-sequence-l2", "0.30",
            "--episode-protection-margin", "0.06", "--episode-negative-quantile", "0.90",
            "--episode-safe-quantile", "0.25", "--episode-isolate-learning-feedback",
            "--replay-max-distance", "0.35", "--episode-maximum-points", "64",
            "--episode-shoulder-margin", "0.50", "--episode-query-mode", "shadow",
            "--episode-gradient-mode", "legacy", "--source-checkpoint", str(source),
            "--a1-dir", str(a1), "--adapter-dir", str(adapter),
            "--development-search",
        ]
    return values


def _validate_receipt(
    summary: Mapping[str, Any], *, seed: int, source_sha: str, mode: str,
) -> None:
    if summary.get("status") != "ONLINE_DIAGNOSTIC":
        raise RuntimeError(f"unexpected EAST V3C status: {summary.get('status')!r}")
    if summary.get("claim_boundary") is None:
        raise RuntimeError("EAST V3C summary lacks its diagnostic claim boundary")
    if int(summary.get("stream_shots", -1)) != 360:
        raise RuntimeError("EAST V3C replay did not process all 360 shots")
    order = summary.get("order_contract", {})
    if not bool(order.get("chronological_main_result")) or bool(order.get("is_reordered")):
        raise RuntimeError("EAST V3C result is not the chronological stream")
    if bool(order.get("query_selection_label_oracle_used")):
        raise RuntimeError("EAST V3C query selection used a label oracle")
    execution = summary.get("execution_protocol", {})
    if not bool(execution.get("phase1_static_complete_before_phase2")):
        raise RuntimeError("EAST V3C phase ordering is not sealed")
    hashes = summary.get("hashes", {})
    if hashes.get("source_checkpoint") != source_sha:
        raise RuntimeError("EAST V3C summary source SHA does not match the requested source")
    if hashes.get("base_before_after") != EXPECTED_PARENT_SHA[int(seed)]:
        raise RuntimeError(
            f"EAST V3C support parent hash changed for seed {seed}: "
            f"{hashes.get('base_before_after')} != {EXPECTED_PARENT_SHA[int(seed)]}"
        )
    if mode == "locked":
        if not bool(execution.get("hyperparameters_locked_before_phase1")):
            raise RuntimeError("locked EAST replay did not lock parameters before phase 1")
        if bool(execution.get("selection_on_360_in_this_run")):
            raise RuntimeError("locked EAST replay selected parameters on balanced360")
        # The compatibility runtime uses the more precise phrase
        # ``not an untouched future-stream confirmation``.  Accept the two
        # explicitly registered contract phrasings, rather than relying on a
        # brittle substring that made a completed replay fail publication.
        claim_boundary = str(summary.get("claim_boundary", "")).lower()
        accepted_boundaries = (
            "not independent test evidence",
            "not an untouched future-stream confirmation",
            "rather than untouched future-stream confirmation",
        )
        if not any(phrase in claim_boundary for phrase in accepted_boundaries):
            raise RuntimeError("locked EAST replay lacks its development-only claim boundary")
    elif mode == "development":
        if not bool(execution.get("selection_on_360_in_this_run")):
            raise RuntimeError("development EAST replay is missing its selection marker")
    else:
        raise ValueError(f"unknown EAST V3C run mode {mode!r}")
    results = summary.get("results", {})
    static = results.get("static_unified", {})
    online = results.get("online_unified", {})
    required = ("tpr_at_30ms", "fpr", "TP", "FP", "FN", "TN")
    for name, row in (("static", static), ("online", online)):
        if not isinstance(row, Mapping) or any(key not in row for key in required):
            raise RuntimeError(f"EAST V3C {name} result lacks required metrics")
        if not all(
            isinstance(row[key], (int, float)) and float(row[key]) == float(row[key])
            for key in required
        ):
            raise RuntimeError(f"EAST V3C {name} result contains non-finite metrics")
    print(
        "[EAST-V3C] verified diagnostic result: "
        f"static TPR@30={float(static['tpr_at_30ms']):.6f}, "
        f"FPR={float(static['fpr']):.6f}; online TPR@30="
        f"{float(online['tpr_at_30ms']):.6f}, FPR={float(online['fpr']):.6f}",
        flush=True,
    )


def _copy_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite final artifact: {destination}")
        destination.rmdir()
    shutil.copytree(source, destination)
    if _file_inventory(source) != _file_inventory(destination):
        raise RuntimeError("final artifact inventory differs from scratch result")
    for relative in _file_inventory(source):
        if sha256_file(source / relative) != sha256_file(destination / relative):
            raise RuntimeError(f"final artifact hash mismatch: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source, data_root, final = _validate_inputs(args)
    scratch_parent = args.scratch_root.expanduser().resolve()
    if scratch_parent == Path("/") or not scratch_parent.is_dir():
        raise ValueError(f"scratch root must be an existing directory: {scratch_parent}")
    runtime, code_hashes = _prepare_runtime_tree(scratch_parent)
    try:
        a1 = runtime / "support_a1"
        adapter = runtime / "support_dynamic"
        final_run = runtime / "final_episode_v3"
        env = os.environ.copy()
        env.update({
            "FUSION_PROJECT_ROOT": str(runtime),
            "FUSION_DATA_ROOT": str(data_root),
            "PYTHONPATH": str(runtime / "experiments/transfer_methods_2026"),
        })
        python = sys.executable
        a1_script = runtime / "experiments/transfer_methods_2026/run_online_active_a1.py"
        dynamic_script = runtime / "experiments/transfer_methods_2026/run_dynamic_hazard_adapter.py"
        _run_subprocess([
            python, str(a1_script), "--seed", str(args.seed), "--out", str(a1),
            "--source-checkpoint", str(source), "--derive-support-threshold",
        ], env=env)
        _run_subprocess([
            python, str(dynamic_script), "--seed", str(args.seed), "--a1", str(a1),
            "--out", str(adapter), "--source-checkpoint", str(source),
        ], env=env)
        unified = _unified_command(
            python=python, runtime=runtime, source=source, a1=a1,
            adapter=adapter, output=final_run, seed=int(args.seed), mode=args.mode,
        )
        _run_subprocess(unified, env=env)
        summary_path = final_run / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("EAST V3C runner did not produce summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_receipt(
            summary, seed=int(args.seed), source_sha=sha256_file(source), mode=args.mode,
        )
        manifest = {
            "schema_version": "east-v3c-classfirst-mounted-v1",
            "status": "ONLINE_DIAGNOSTIC",
            "claim_status": (
                "FROZEN_REPLAY_DEVELOPMENT" if args.mode == "locked"
                else "TEST_DRIVEN_EXPLORATION"
            ),
            "run_mode": args.mode,
            "seed": int(args.seed),
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "data_root": str(data_root),
            "support_scope": "EAST support110 only",
            "test_scope": "EAST balanced360 chronological stream",
            "support_sampler": "class-first -> shot -> endpoint (historical V3C compatibility)",
            "causal_order": "predict -> seal -> query -> reveal -> update",
            "balanced360_outcomes_used_before_query": False,
            "jtext_online_checkpoint_used_for_east": False,
            "runtime": {
                "python": python,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "compatibility_code_sha256": code_hashes,
            "unified_command": unified,
            "selection_on_360_in_this_run": args.mode == "development",
            "summary_sha256_before_publish": sha256_file(summary_path),
            "claim_boundary": (
                "balanced360 participated in architecture/parameter development; "
                "this is a causal diagnostic replay, not untouched future confirmation"
            ),
        }
        (runtime / "mainline_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _copy_verified(runtime, final)
        published = json.loads((final / "final_episode_v3/summary.json").read_text())
        return {
            "output": str(final),
            "summary": published,
            "manifest": json.loads((final / "mainline_manifest.json").read_text()),
        }
    finally:
        if not args.keep_scratch:
            shutil.rmtree(runtime, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=RUN_MODES, default="locked",
        help=(
            "locked uses the pre-registered V3C configuration; development is "
            "an explicit balanced360 parameter-search diagnostic"
        ),
    )
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    summary = result["summary"]
    print(json.dumps({
        "output": result["output"],
        "static": summary["results"]["static_unified"],
        "online": summary["results"]["online_unified"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
