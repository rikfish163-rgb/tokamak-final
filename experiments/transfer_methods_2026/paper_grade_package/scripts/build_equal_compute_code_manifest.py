#!/usr/bin/env python3
"""Hash the release code that reconstructs the fixed-shape roster screen.

This is deliberately a post-run release snapshot, not an execution-time seal.
The distinction is serialized so a content hash cannot be mistaken for
preregistration or immutable run provenance.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = (
    ROOT / "experiments/transfer_methods_2026/paper_grade_package/evidence"
    / "equal_compute_selector_control_v1/code_environment_manifest.json"
)
CODE_SUFFIXES = {".py", ".pyi", ".sh"}
ENVIRONMENT_FILES = {"pyproject.toml", "uv.lock", ".python-version"}
CONFIG_PREFIXES = ("config/", "configs/")
CI_FILES = {".github/workflows/ci.yml"}


def reconstruction_source_files() -> tuple[str, ...]:
    """Return the Git-indexed local code/config closure used by reconstruction.

    Selecting every tracked Python/shell source file avoids an incomplete
    hand-maintained import list, including dynamic imports.  Configuration,
    dependency locks and the CI contract are included explicitly.  Data,
    checkpoints, ledgers, figures and manuscripts are bound by their dedicated
    release manifests instead.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT,
        check=True, capture_output=True,
    )
    tracked = result.stdout.decode().split("\0")
    selected = {
        relative for relative in tracked if relative and (
            Path(relative).suffix in CODE_SUFFIXES
            or relative in ENVIRONMENT_FILES
            or relative in CI_FILES
            or relative.startswith(CONFIG_PREFIXES)
        )
    }
    if not selected:
        raise RuntimeError("Git index contains no reconstruction sources")
    return tuple(sorted(selected))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    files = []
    source_files = reconstruction_source_files()
    for relative in source_files:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    payload = {
        "schema_version": "equal-compute-release-code-manifest-v2",
        "provenance_status": "POST_RUN_RELEASE_SNAPSHOT_NOT_EXECUTION_TIME_SEAL",
        "provenance_boundary": (
            "These hashes bind the complete Git-indexed local Python/shell, "
            "configuration, dependency-lock and CI snapshot used to reconstruct "
            "the released analysis. "
            "They were produced after the runs and do not prove which uncommitted "
            "source bytes executed historically. Run commands, output hashes, "
            "parents and per-query receipts remain separate evidence."
        ),
        "selection_contract": {
            "source": "git ls-files --cached",
            "code_suffixes": sorted(CODE_SUFFIXES),
            "environment_files": sorted(ENVIRONMENT_FILES),
            "config_prefixes": list(CONFIG_PREFIXES),
            "ci_files": sorted(CI_FILES),
            "excluded_artifacts_bound_elsewhere": [
                "raw data", "checkpoints", "prediction ledgers", "figures",
                "manuscript outputs",
            ],
        },
        "source_files": files,
        "release_environment_snapshot": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ("torch", "numpy", "h5py", "matplotlib", "pytest")
            },
            "executable_contract": (
                "GPU/reference: CUDA-capable rl_env; "
                "public CPU audit: uv sync --frozen --only-group ci "
                "--no-install-project, then pinned torch 2.11.0 from the "
                "official PyTorch CPU index"
            ),
            "python_executable_name": Path(sys.executable).name,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "source_files": len(files)}))


if __name__ == "__main__":
    main()
