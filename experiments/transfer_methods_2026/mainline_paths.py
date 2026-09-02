"""Filesystem contract for the retained J-TEXT -> EAST mainline.

Raw data and generated runs are large and the local ext4 volume is nearly
full.  The mainline therefore fails closed when an artifact root resolves to
``/home`` (the old accidental location) or to the local root filesystem.
Scratch computation may still use ``/dev/shm``; only the final artifact root
is governed here.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "fuison_ext4_runs"
LEGACY_LOCAL_ROOT = Path("/home/hetaisheng/fuison_ext4_runs")
PROJECT_MOUNT_ROOT = PROJECT_ROOT.parent


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"no existing parent for artifact root: {path}")
        current = current.parent
    return current


def resolve_artifact_root(candidate: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a final artifact directory and enforce the mounted-disk rule."""

    raw = (
        candidate
        if candidate is not None
        else os.environ.get("FUSION_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)
    )
    path = Path(raw).expanduser().resolve()
    legacy = LEGACY_LOCAL_ROOT.resolve()
    home = Path("/home").resolve()
    if path == legacy or legacy in path.parents:
        raise RuntimeError(
            f"refusing legacy local artifact root {path}; use the mounted-disk "
            f"root {DEFAULT_ARTIFACT_ROOT}"
        )
    if path == home or home in path.parents:
        raise RuntimeError(f"final artifacts must not be written under /home: {path}")
    if Path("/dev") == path or Path("/dev") in path.parents:
        raise RuntimeError(
            f"/dev/shm and other /dev paths are scratch space, not final artifacts: {path}"
        )

    # A different non-root mount is not automatically the project disk.  The
    # current experiment contract is deliberately narrower: artifacts must be
    # below the mounted workspace that contains this checkout.  This also
    # catches the case where the mount disappeared and the path became an
    # ordinary directory on the local root filesystem.
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise RuntimeError(
            f"final artifacts must be under the mounted project workspace "
            f"{PROJECT_ROOT}: {path}"
        )

    parent = _existing_parent(path)
    root_device = os.stat("/").st_dev
    parent_device = parent.stat().st_dev
    project_device = os.stat(PROJECT_ROOT).st_dev
    if project_device == root_device:
        raise RuntimeError(
            f"project workspace is not on a mounted non-root filesystem: {PROJECT_ROOT}"
        )
    if os.stat(PROJECT_MOUNT_ROOT).st_dev != project_device:
        raise RuntimeError(
            f"project mount identity changed: {PROJECT_MOUNT_ROOT} does not share "
            f"device {project_device} with {PROJECT_ROOT}"
        )
    if parent_device == root_device:
        raise RuntimeError(
            f"final artifact root is on the local root filesystem ({path}); "
            "use the mounted project disk or set FUSION_ARTIFACT_ROOT explicitly"
        )
    if parent_device != project_device:
        raise RuntimeError(
            f"final artifact root is on a different filesystem than the project "
            f"mount ({path})"
        )
    return path


def assert_project_workspace_path(
    path: str | os.PathLike[str], *, label: str = "path"
) -> Path:
    """Validate a data or artifact path belongs to the mounted project tree.

    Unlike :func:`assert_final_artifact_root`, this helper does not require the
    target to be an output directory.  It is used for raw-data roots so a run
    cannot silently read a same-named local copy after the mount changes.
    """

    resolved = Path(path).expanduser().resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise RuntimeError(
            f"{label} must be under the mounted project workspace {PROJECT_ROOT}: "
            f"{resolved}"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    root_device = os.stat("/").st_dev
    project_device = os.stat(PROJECT_ROOT).st_dev
    if project_device == root_device or os.stat(resolved).st_dev != project_device:
        raise RuntimeError(
            f"{label} is not on the mounted project filesystem: {resolved}"
        )
    return resolved


def assert_final_artifact_root(path: str | os.PathLike[str]) -> Path:
    """Validate a caller-provided final root before creating any files."""

    return resolve_artifact_root(path)


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "LEGACY_LOCAL_ROOT",
    "PROJECT_MOUNT_ROOT",
    "PROJECT_ROOT",
    "assert_final_artifact_root",
    "assert_project_workspace_path",
    "resolve_artifact_root",
]
