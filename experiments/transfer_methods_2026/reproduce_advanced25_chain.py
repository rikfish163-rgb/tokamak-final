#!/usr/bin/env python3
"""Reproduce the single requested J-TEXT -> EAST Adv25 chain.

The default EAST online lineage is the *locked* historical V3C/v8 replay.
The newer ``episode-v3`` development chain is available only as an explicit
diagnostic option.  Keeping this choice explicit is important: otherwise a
label-only source change is silently compared with a different online
algorithm and the result cannot answer the requested experiment.

Only the J-TEXT source training label policy differs from the historical
fixed-125-ms source in the default chain:

    H_JTEXT,i = max(25 ms, AdvancedTime_i)

The J-TEXT online branch is an independent diagnostic stage.  Its updated
state is never passed to EAST; both EAST stages receive the newly trained
*static* source checkpoint.  Existing output directories are rejected so a
partial run cannot silently mix lineages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments" / "transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
CONFIG = PACKAGE / "configs" / "primary_chain_v3c.json"
ADVANCED_POLICY = "advanced_floor25"
ADVANCED_TIME_PATH = ROOT / "shot_list" / "J-TEXT" / "AdvancedTime_J-TEXT.json"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from mainline_paths import assert_final_artifact_root  # noqa: E402

LOCKED_ONLINE_LINEAGE = "locked_v8"
DEVELOPMENT_ONLINE_LINEAGE = "development_v3c"
ONLINE_LINEAGES = (LOCKED_ONLINE_LINEAGE, DEVELOPMENT_ONLINE_LINEAGE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def options(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in payload.items():
        if isinstance(value, bool):
            if value:
                result.append(option_name(name))
        else:
            result.extend((option_name(name), str(value)))
    return result


def _source_checkpoint_for(
    run_root: Path, *, source_checkpoint: Path | None, retrain_source: bool
) -> Path:
    if source_checkpoint is not None and retrain_source:
        raise ValueError("choose --retrain-source or --source-checkpoint, not both")
    if retrain_source:
        return run_root / "source" / "source_checkpoint.pt"
    if source_checkpoint is None:
        raise ValueError(
            "Adv25 chain needs --retrain-source or an explicit Adv25 --source-checkpoint"
        )
    return source_checkpoint.expanduser().resolve()


def command_plan(
    *,
    seed: int,
    output_root: Path,
    retrain_source: bool,
    source_checkpoint: Path | None = None,
    jtext_budget_fraction: float = 0.10,
    jtext_query_cutoff: float = 0.50,
    online_lineage: str = LOCKED_ONLINE_LINEAGE,
) -> dict[str, list[str]]:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    registered = tuple(int(value) for value in contract["seeds"])
    if int(seed) not in registered:
        raise ValueError(f"unregistered seed: {seed}; expected {registered}")
    if not 0.0 < float(jtext_budget_fraction) <= 1.0:
        raise ValueError("J-TEXT query budget fraction must lie in (0, 1]")
    if not 0.0 <= float(jtext_query_cutoff) <= 1.0:
        raise ValueError("J-TEXT query cutoff must lie in [0, 1]")
    if online_lineage not in ONLINE_LINEAGES:
        raise ValueError(
            f"unknown EAST online lineage {online_lineage!r}; "
            f"expected one of {ONLINE_LINEAGES}"
        )

    run_root = output_root.expanduser().resolve() / f"seed{int(seed)}"
    source_dir = run_root / "source"
    jtext_online_dir = run_root / "jtext_online"
    a1_dir = run_root / "a1"
    adapter_dir = run_root / "dynamic_hazard_adapter"
    east_online_dir = run_root / "final_episode_v3"
    source_path = _source_checkpoint_for(
        run_root, source_checkpoint=source_checkpoint,
        retrain_source=retrain_source,
    )

    source = [
        sys.executable,
        str(HERE / "train_jtext_source_transformer.py"),
        "--seed", str(int(seed)),
        "--output", str(source_dir),
        "--device", "cuda",
        *options(contract["source"]),
        "--jtext-label-policy", ADVANCED_POLICY,
    ]
    jtext_online = [
        sys.executable,
        str(HERE / "run_jtext_rebuilt17_online.py"),
        "--seed", str(int(seed)),
        "--source-checkpoint", str(source_path),
        "--output", str(jtext_online_dir),
        "--device", "cuda",
        "--budget-fraction", str(float(jtext_budget_fraction)),
        "--query-cutoff", str(float(jtext_query_cutoff)),
    ]
    a1 = [
        sys.executable,
        str(HERE / "run_online_active_a1.py"),
        "--seed", str(int(seed)),
        "--out", str(a1_dir),
        "--source-checkpoint", str(source_path),
        *options(contract["a1"]),
    ]
    adapter = [
        sys.executable,
        str(HERE / "run_dynamic_hazard_adapter.py"),
        "--seed", str(int(seed)),
        "--a1", str(a1_dir),
        "--out", str(adapter_dir),
        "--source-checkpoint", str(source_path),
    ]
    east_online = [
        sys.executable,
        str(HERE / "run_unified_online_active.py"),
        "--seed", str(int(seed)),
        "--out", str(east_online_dir),
        "--source-checkpoint", str(source_path),
        "--a1-dir", str(a1_dir),
        "--adapter-dir", str(adapter_dir),
    ]
    if online_lineage == LOCKED_ONLINE_LINEAGE:
        # The runner validates every locked-v8 parameter before creating an
        # output directory.  Do not append the development config here: its
        # quota, history, episode gate and gradient mode are a different
        # experiment, not a harmless implementation detail.
        east_online.append("--locked-best-v8")
    else:
        east_online.extend(options(contract["online"]))
    return {
        "source": source,
        "jtext_online": jtext_online,
        "a1": a1,
        "adapter": adapter,
        "east_online": east_online,
    }


def _require_dynamic_checkpoint(path: Path, *, seed: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Adv25 source checkpoint missing or symlinked: {path}")
    # Importing the source module keeps the provenance check identical to the
    # standalone J-TEXT online runner.
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("seed", -1)) != int(seed):
        raise ValueError("source checkpoint seed differs from requested chain seed")
    training_label = checkpoint.get("training_label")
    if not isinstance(training_label, dict):
        raise ValueError("Adv25 source checkpoint has no training-label metadata")
    if training_label.get("policy") != ADVANCED_POLICY:
        raise ValueError("chain source checkpoint is not trained with Adv25 labels")
    if int(checkpoint.get("test_shards_read", -1)) != 0:
        raise ValueError("chain source checkpoint was selected after reading TEST220")
    metadata = training_label.get("advanced_time_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Adv25 source checkpoint lacks AdvancedTime contract metadata")
    # v1 and v3 encode the same selected label rule.  v3 added an explicit
    # ambiguity-band field; accepting the older immutable v1 artifact is safe
    # only after the common rule and the no-ambiguity conditions below have
    # been checked.  This keeps the historical Adv25 V3C replay reproducible
    # without treating a legacy fixed-125 checkpoint as equivalent.
    schema_version = metadata.get("schema_version")
    if schema_version not in {
        "jtext-advanced-floor25-label-contract-v1",
        "jtext-advanced-floor25-label-contract-v3",
    }:
        raise ValueError("Adv25 source checkpoint has an unknown label-contract schema")
    if metadata.get("policy") != "max(floor_ms, AdvancedTime_ms)":
        raise ValueError("Adv25 source checkpoint has an unexpected AdvancedTime policy")
    if metadata.get("ambiguous_context_until_ms") is not None:
        raise ValueError("Adv25 main chain cannot use an ambiguity-band checkpoint")
    if training_label.get("ambiguous_context_until_ms") is not None:
        raise ValueError("Adv25 main chain cannot use an ambiguity-band checkpoint")
    if int(training_label.get("ambiguous_window_count", 0)) != 0:
        raise ValueError("Adv25 source checkpoint reports ambiguous training windows")
    recorded_sha = str(
        training_label.get("advanced_time_source_sha256", metadata.get("source_sha256", ""))
    )
    if not recorded_sha:
        raise ValueError("Adv25 source checkpoint lacks AdvancedTime source SHA")
    actual_sha = sha256_file(ADVANCED_TIME_PATH)
    if recorded_sha != actual_sha:
        raise ValueError("Adv25 source checkpoint AdvancedTime source SHA differs from workspace")
    return checkpoint


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing chain artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"chain artifact is not a JSON object: {path}")
    return value


def _verify_lineage(run_root: Path, *, source_path: Path, seed: int) -> dict[str, Any]:
    source = _require_dynamic_checkpoint(source_path, seed=seed)
    source_sha = sha256_file(source_path)
    jtext_summary = _read_json(run_root / "jtext_online" / "summary.json")
    if jtext_summary.get("source_checkpoint_sha256") != source_sha:
        raise ValueError("J-TEXT online parent SHA differs from static source")
    if jtext_summary.get("causal_contract", {}).get(
        "jtext_online_checkpoint_used_for_east"
    ) is not False:
        raise ValueError("J-TEXT online branch is marked as an EAST parent")

    expected = {
        "a1": run_root / "a1" / "summary.json",
        "dynamic_hazard_adapter": run_root / "dynamic_hazard_adapter" / "summary.json",
        "east_online": run_root / "final_episode_v3" / "summary.json",
    }
    lineage: dict[str, Any] = {
        "source_checkpoint_sha256": source_sha,
        "source_training_label": source.get("training_label"),
        "artifacts": {},
    }
    for name, path in expected.items():
        payload = _read_json(path)
        hashes = payload.get("hashes", {})
        recorded = hashes.get("source_checkpoint")
        if recorded is not None and recorded != source_sha:
            raise ValueError(f"{name} does not use the Adv25 static source SHA")
        lineage["artifacts"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "recorded_source_checkpoint_sha256": recorded,
        }
    return lineage


def _write_manifest(
    run_root: Path,
    *,
    seed: int,
    plan: dict[str, list[str]],
    source_path: Path,
    stage: str,
    online_lineage: str,
    lineage: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "jtext-advanced-floor25-four-stage-chain-v1",
        "stage": stage,
        "seed": int(seed),
        "label_policy": ADVANCED_POLICY,
        "label_formula": "H_JTEXT_i=max(25 ms, AdvancedTime_i)",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": (
            sha256_file(source_path) if source_path.is_file() else None
        ),
        "commands": {name: shlex.join(command) for name, command in plan.items()},
        "environment": {
            "python": sys.executable,
            "required_environment": "rl_env",
            "device": "cuda",
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "lineage_contract": {
            "jtext_online_parent_for_east": False,
            "east_uses": "static Adv25 source checkpoint",
            "east_v3c_code_changed": False,
            "east_online_lineage": online_lineage,
            "test_stream_status": "diagnostic_test_driven_exploration",
        },
    }
    if lineage is not None:
        payload["lineage_verification"] = lineage
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = run_root / ".chain_manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(run_root / "chain_manifest.json")


def _run_stage(name: str, command: list[str], *, cwd: Path, run_root: Path) -> None:
    log_path = run_root / f"{name}.log"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite existing stage log: {log_path}")
    run_root.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n\n")
        log.flush()
        environment = os.environ.copy()
        environment["FUSION_PROJECT_ROOT"] = str(ROOT)
        subprocess.run(
            command,
            check=True,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(131, 133, 137), required=True)
    parser.add_argument(
        "--stage",
        choices=("plan", "source", "jtext-online", "target", "east-online", "all"),
        default="plan",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retrain-source", action="store_true")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--jtext-budget-fraction", type=float, default=0.10)
    parser.add_argument("--jtext-query-cutoff", type=float, default=0.50)
    parser.add_argument(
        "--online-lineage",
        choices=ONLINE_LINEAGES,
        default=LOCKED_ONLINE_LINEAGE,
        help=(
            "EAST online implementation lineage; locked_v8 is the mainline, "
            "development_v3c is diagnostic only"
        ),
    )
    args = parser.parse_args()

    if Path(sys.executable).resolve().parent.parent.name != "rl_env":
        raise RuntimeError(
            "this chain is locked to the rl_env CUDA runtime; invoke its python explicitly"
        )

    # ``plan`` is a read-only command and may be rendered against a temporary
    # path by tests.  Every stage that can create a manifest, log, checkpoint,
    # or copied result must validate the final root before constructing any
    # stage directory.  This closes the old path hole where the chain
    # orchestrator bypassed the guards used by the individual runners.
    output_root = args.output_root.expanduser().resolve()
    if args.stage != "plan":
        output_root = assert_final_artifact_root(output_root)
    run_root = output_root / f"seed{int(args.seed)}"
    plan = command_plan(
        seed=args.seed,
        output_root=output_root,
        retrain_source=bool(args.retrain_source),
        source_checkpoint=args.source_checkpoint,
        jtext_budget_fraction=args.jtext_budget_fraction,
        jtext_query_cutoff=args.jtext_query_cutoff,
        online_lineage=args.online_lineage,
    )
    source_path = Path(plan["jtext_online"][plan["jtext_online"].index("--source-checkpoint") + 1])
    print(json.dumps({name: shlex.join(command) for name, command in plan.items()}, indent=2))
    if args.stage == "plan":
        return 0

    if args.stage in {"source", "all"}:
        if not args.retrain_source:
            raise ValueError("--stage source/all requires --retrain-source for Adv25")
        if (run_root / "source").exists():
            raise FileExistsError(f"source output already exists: {run_root / 'source'}")
        _write_manifest(
            run_root, seed=args.seed, plan=plan, source_path=source_path,
            stage="source_running", online_lineage=args.online_lineage,
        )
        _run_stage("source", plan["source"], cwd=ROOT, run_root=run_root)
        _require_dynamic_checkpoint(source_path, seed=args.seed)

    if args.stage in {"jtext-online", "all"}:
        if (run_root / "jtext_online").exists():
            raise FileExistsError(f"J-TEXT online output already exists: {run_root / 'jtext_online'}")
        _write_manifest(
            run_root, seed=args.seed, plan=plan, source_path=source_path,
            stage="jtext_online_running", online_lineage=args.online_lineage,
        )
        _run_stage("jtext_online", plan["jtext_online"], cwd=ROOT, run_root=run_root)

    if args.stage in {"target", "all"}:
        if (run_root / "a1").exists() or (run_root / "dynamic_hazard_adapter").exists():
            raise FileExistsError(f"EAST target output already exists under {run_root}")
        _require_dynamic_checkpoint(source_path, seed=args.seed)
        _write_manifest(
            run_root, seed=args.seed, plan=plan, source_path=source_path,
            stage="target_running", online_lineage=args.online_lineage,
        )
        _run_stage("a1", plan["a1"], cwd=ROOT, run_root=run_root)
        _run_stage("dynamic_hazard_adapter", plan["adapter"], cwd=ROOT, run_root=run_root)

    if args.stage in {"east-online", "all"}:
        if (run_root / "final_episode_v3").exists():
            raise FileExistsError(f"EAST online output already exists: {run_root / 'final_episode_v3'}")
        _require_dynamic_checkpoint(source_path, seed=args.seed)
        _write_manifest(
            run_root, seed=args.seed, plan=plan, source_path=source_path,
            stage="east_online_running", online_lineage=args.online_lineage,
        )
        _run_stage("east_online", plan["east_online"], cwd=ROOT, run_root=run_root)

    if args.stage == "all":
        lineage = _verify_lineage(run_root, source_path=source_path, seed=args.seed)
        _write_manifest(
            run_root, seed=args.seed, plan=plan, source_path=source_path,
            stage="complete_diagnostic", online_lineage=args.online_lineage,
            lineage=lineage,
        )
        print(json.dumps(lineage, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
