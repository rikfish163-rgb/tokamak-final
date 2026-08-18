from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py

from reproducibility.paths import project_root


class DataContractError(RuntimeError):
    pass


RAW23 = (
    "Bt", "CIII", "MNM", "Mir_VV", "Mir_avg_amp", "Mir_avg_fre",
    "Z_proxy", "dx_a", "dy_a", "ip", "mode_number_n", "n=1 amplitude",
    "ne_nG", "qa_proxy", "sxr_kurt", "sxr_mean", "sxr_skew", "sxr_var",
    "v_loop", "xuv_kurt", "xuv_ratio", "xuv_skew", "xuv_var",
)


def _shot_path(data: Path, device: str, shot: int) -> Path:
    return data / device / "unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"


def _read_ids(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().splitlines() if line.strip()]


def _east_ids(root: Path) -> list[int]:
    support = json.loads(
        (root / "config/splits/east_paper131_split_all_in.json").read_text()
    )["train"]
    test = json.loads(
        (root / "config/splits/east_hdf5_authoritative_balanced360_v1.json").read_text()
    )["test"]
    values = []
    for block in (support, test):
        values.extend(block["disruption_shots"])
        values.extend(block["non_disruption_shots"])
    return sorted({int(value) for value in values})


def _jtext_roles(root: Path) -> dict[str, list[int]]:
    expected = {"train": 1353, "val": 160, "test": 220}
    roles = {
        name: _read_ids(root / "splits" / f"{name}.txt") for name in expected
    }
    for name, count in expected.items():
        values = roles[name]
        if len(values) != count or len(values) != len(set(values)):
            raise DataContractError(
                f"J-TEXT {name} split must contain {count} unique shots"
            )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(roles[left]) & set(roles[right])
        if overlap:
            raise DataContractError(
                f"J-TEXT split overlap {left}/{right}: {sorted(overlap)[:10]}"
            )
    return roles


def _jtext_ids(root: Path) -> list[int]:
    roles = _jtext_roles(root)
    return sorted({shot for values in roles.values() for shot in values})


def _validate_hdf5(path: Path, *, full: bool) -> None:
    try:
        with h5py.File(path, "r") as handle:
            required = ("data/ip", "meta/IsDisrupt", "meta/DownTime")
            missing = [name for name in required if name not in handle]
            if missing:
                raise DataContractError(f"{path}: missing HDF5 entries {missing}")
            if full:
                absent = [name for name in RAW23 if f"data/{name}" not in handle]
                if absent:
                    raise DataContractError(f"{path}: missing raw23 channels {absent}")
    except OSError as exc:
        raise DataContractError(f"cannot read HDF5 {path}: {exc}") from exc


def audit_data(data: Path, *, full: bool = False) -> dict[str, Any]:
    data = Path(data).expanduser().resolve()
    root = project_root()
    required_dirs = [
        data / "J-TEXT" / "unified_hdf5",
        data / "EAST" / "unified_hdf5",
    ]
    for directory in required_dirs:
        if not directory.is_dir():
            raise DataContractError(f"missing required data directory: {directory}")

    ids = {"J-TEXT": _jtext_ids(root), "EAST": _east_ids(root)}
    missing: dict[str, list[int]] = {"J-TEXT": [], "EAST": []}
    checked = 0
    digest = hashlib.sha256()
    for device, shot_ids in ids.items():
        candidates = shot_ids if full else shot_ids[:1] + shot_ids[-1:]
        for shot in shot_ids:
            path = _shot_path(data, device, shot)
            if not path.is_file():
                missing[device].append(shot)
        for shot in candidates:
            path = _shot_path(data, device, shot)
            if path.is_file():
                _validate_hdf5(path, full=full)
                stat = path.stat()
                digest.update(f"{device}:{shot}:{stat.st_size}".encode())
                checked += 1
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        counts = {key: len(value) for key, value in missing.items()}
        examples = {key: value[:10] for key, value in missing.items()}
        raise DataContractError(
            f"missing HDF5 shots counts={counts} examples={examples}"
        )
    return {
        "passed": True,
        "data_root": str(data),
        "full": bool(full),
        "expected_shots": {key: len(value) for key, value in ids.items()},
        "hdf5_files_opened": checked,
        "inventory_sha256": digest.hexdigest(),
        "label_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
    }
