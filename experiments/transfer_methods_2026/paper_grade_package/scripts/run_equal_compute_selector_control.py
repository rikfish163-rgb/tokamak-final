#!/usr/bin/env python3
"""Run the development-only equal-compute selector-roster control.

Both arms replay fixed, explicit, label-free query manifests through the same
inter-only assimilation operator.  Every queried shot receives 16 current and
16 immutable support rows plus two shared and two private optimizer steps.
The archived active identities are a post-development roster treatment, not a
re-execution of the adaptive selector policy and not prospective evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
PRIMARY = PACKAGE / "evidence/primary_online_artifacts"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


timescale = load_module("equal_compute_timescale", HERE / "online_timescale_ablation.py")
reproduce = load_module(
    "equal_compute_reproduce", Path(__file__).with_name("reproduce_primary_chain.py")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def active_manifest(seed: int, output: Path) -> Path:
    source = PRIMARY / f"seed{seed}"
    static_path = source / "static_predictions.jsonl"
    decisions_path = source / "query_decisions.jsonl"
    roster = [int(row["shot"]) for row in read_jsonl(static_path)]
    manifest = timescale.fixed_query_manifest_from_decisions(
        read_jsonl(decisions_path), roster=roster,
    )
    manifest.update({
        "schema_version": "equal-compute-query-manifest-v1",
        "kind": "archived_active_query_roster_development_treatment",
        "label_oracle_used": False,
        "claim_status": "BALANCED360_POST_DEVELOPMENT_ROSTER_TREATMENT",
        "source": {
            "query_decisions_sha256": sha256_file(decisions_path),
            "static_predictions_sha256": sha256_file(static_path),
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise FileExistsError(f"refusing to overwrite different manifest: {output}")
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def strip_flags(command: list[str], flags: set[str]) -> list[str]:
    return [value for value in command if value not in flags]


def set_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend((name, str(value)))


def equal_compute_command(
    *, seed: int, output_root: Path, manifest: Path, out: Path,
) -> list[str]:
    command = list(reproduce.command_plan(
        seed=int(seed), output_root=output_root, retrain_source=False,
    )["online"])
    command = strip_flags(command, {
        "--episode-value-v3", "--episode-protected-sequence",
        "--episode-isolate-learning-feedback", "--causal-error-memory",
        "--typed-endpoint-memory", "--strict-window-active",
    })
    set_option(command, "--timescale-mode", "inter_only")
    set_option(command, "--inter-steps", "2")
    set_option(command, "--out", str(out))
    command.extend((
        "--selector-compute-control",
        "--fixed-compute-current-windows", "16",
        "--fixed-compute-replay-windows", "16",
        "--fixed-query-manifest", str(manifest),
    ))
    return command


def normalized_command(command: list[str]) -> list[str]:
    result = list(command)
    for option in ("--out", "--fixed-query-manifest"):
        index = result.index(option) + 1
        result[index] = f"<{option[2:]}>"
    return result


def run(command: list[str], environment: dict[str, str]) -> None:
    print("[EQUAL-COMPUTE] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[131, 133, 137])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--stage", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    if args.replicates <= 0:
        raise ValueError("replicates must be positive")
    output_root = args.output_root.expanduser().resolve()
    random_builder = load_module(
        "equal_compute_random_builder",
        Path(__file__).with_name("run_causal_random_query_baseline.py"),
    )
    environment = dict(os.environ)
    environment["FUSION_PROJECT_ROOT"] = str(ROOT)
    plan: dict[str, Any] = {
        "schema_version": "equal-compute-selector-control-plan-v1",
        "claim_status": "BALANCED360_METHOD_DEVELOPMENT_DIAGNOSTIC",
        "estimand": "archived active roster versus random rosters under fixed assimilation",
        "seeds": [int(seed) for seed in args.seeds],
        "replicates": int(args.replicates),
        "commands": [],
    }
    for seed in args.seeds:
        manifest_root = output_root / "manifests"
        active = active_manifest(
            int(seed), manifest_root / f"seed{seed}_archived_active.json",
        )
        expected_randoms = [
            manifest_root / f"seed{seed}_random_query_r{replicate}.json"
            for replicate in range(int(args.replicates))
        ]
        existing = [path.exists() for path in expected_randoms]
        if any(existing) and not all(existing):
            raise FileExistsError("partial equal-compute random manifest set")
        if all(existing):
            randoms = expected_randoms
            for manifest_path in randoms:
                timescale.require_explicit_non_oracle_query_manifest(
                    manifest_path
                )
        else:
            randoms = random_builder.build_manifests(
                seed=int(seed), output=manifest_root,
                replicates=int(args.replicates),
            )
            for manifest_path in randoms:
                payload = json.loads(manifest_path.read_text())
                payload["schema_version"] = "equal-compute-query-manifest-v1"
                payload["claim_status"] = (
                    "BALANCED360_POST_DEVELOPMENT_RANDOM_ROSTER_TREATMENT"
                )
                manifest_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
        commands = [equal_compute_command(
            seed=int(seed), output_root=output_root, manifest=active,
            out=output_root / f"seed{seed}/active_fixed_compute",
        )]
        commands.extend(
            equal_compute_command(
                seed=int(seed), output_root=output_root, manifest=manifest,
                out=output_root / f"seed{seed}/random_fixed_compute_r{replicate}",
            )
            for replicate, manifest in enumerate(randoms)
        )
        normalized = [normalized_command(command) for command in commands]
        if any(item != normalized[0] for item in normalized[1:]):
            raise RuntimeError("active/random commands differ beyond output and query manifest")
        plan["commands"].append({"seed": int(seed), "runs": commands})
        if args.stage == "run":
            for command in commands:
                out = Path(command[command.index("--out") + 1])
                if not (out / "summary.json").is_file():
                    run(command, environment)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    print(json.dumps({
        "output_root": str(output_root), "stage": args.stage,
        "planned_runs": len(args.seeds) * (1 + int(args.replicates)),
    }, indent=2))


if __name__ == "__main__":
    main()
