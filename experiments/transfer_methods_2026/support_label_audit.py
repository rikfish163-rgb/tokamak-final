"""Audit the legacy EAST support roster against authoritative HDF5 labels.

The historical ``east_paper131_split_all_in.json`` file declares a 10/100
disruption/safe role split.  The compatibility V3C runner loads the actual
``meta/IsDisrupt`` value from each HDF5 file, however.  This module makes that
distinction explicit without changing the historical replay: the role roster
remains the sampling contract, while HDF5 remains the label authority.

Only the 110 support shots are inspected here.  The chronological balanced360
test outcomes are deliberately not opened by this audit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_support_labels(
    declared_roles: Mapping[str, Sequence[int]],
    authoritative_labels: Mapping[int, int],
) -> dict[str, Any]:
    """Summarize declared support roles versus HDF5-derived binary labels.

    ``declared_roles`` contains the role-roster view (``disruption`` and
    ``safe``).  ``authoritative_labels`` maps every roster ID to the scalar
    HDF5 ``meta/IsDisrupt`` value.  The function is intentionally independent
    of HDF5 so the role/label accounting can be unit-tested with tiny data.
    """

    disruption = [int(value) for value in declared_roles.get("disruption", ())]
    safe = [int(value) for value in declared_roles.get("safe", ())]
    normalized_authoritative: dict[int, int] = {}
    for raw_shot, raw_value in authoritative_labels.items():
        shot = int(raw_shot)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"authoritative label is not numeric: {raw_shot!r}") from exc
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise ValueError(f"authoritative label is not binary: {raw_shot!r}")
        if shot in normalized_authoritative:
            raise ValueError(f"duplicate authoritative shot ID: {shot}")
        normalized_authoritative[shot] = int(value)
    if len(disruption) != len(set(disruption)):
        raise ValueError("duplicate disruption shot IDs in support roster")
    if len(safe) != len(set(safe)):
        raise ValueError("duplicate safe shot IDs in support roster")
    overlap = set(disruption) & set(safe)
    if overlap:
        raise ValueError(f"support role overlap: {sorted(overlap)}")

    declared_by_shot = {shot: 1 for shot in disruption}
    declared_by_shot.update({shot: 0 for shot in safe})
    missing = sorted(set(declared_by_shot) - set(normalized_authoritative))
    extra = sorted(set(normalized_authoritative) - set(declared_by_shot))
    if missing:
        raise ValueError(f"support roster lacks authoritative labels: {missing}")
    if extra:
        raise ValueError(f"authoritative labels contain non-roster shots: {extra}")
    mismatches = [
        {
            "shot": shot,
            "declared_role": "disruption" if declared else "safe",
            "declared_is_disrupt": declared,
            "hdf5_is_disrupt": normalized_authoritative[shot],
        }
        for shot, declared in sorted(declared_by_shot.items())
        if normalized_authoritative[shot] != declared
    ]
    authoritative_counts = {
        "disruption": sum(value == 1 for value in normalized_authoritative.values()),
        "safe": sum(value == 0 for value in normalized_authoritative.values()),
    }
    return {
        "declared_counts": {
            "disruption": len(disruption),
            "safe": len(safe),
            "total": len(declared_by_shot),
        },
        "authoritative_counts": {
            **authoritative_counts,
            "total": len(normalized_authoritative),
        },
        "mismatches": mismatches,
        "composition_matches_declared": not mismatches,
        "roster_sha256": _sha256_json({
            "disruption": disruption,
            "safe": safe,
        }),
    }


def audit_east_support110(
    data_root: str | Path,
    roster_path: str | Path,
) -> dict[str, Any]:
    """Read support metadata from HDF5 and return a reproducible audit.

    The function reads no signal arrays and never opens balanced360.  A
    missing/invalid support file is a hard error because silently substituting
    a different label source would invalidate the transfer parent.
    """

    import h5py
    import numpy as np

    data_root = Path(data_root).expanduser().resolve()
    roster_path = Path(roster_path).expanduser().resolve()
    payload = json.loads(roster_path.read_text(encoding="utf-8"))
    train = payload["train"]
    declared_roles = {
        "disruption": [int(value) for value in train["disruption_shots"]],
        "safe": [int(value) for value in train["non_disruption_shots"]],
    }
    authoritative: dict[int, int] = {}
    invalid_down_time: list[int] = []
    for shot, declared in (
        [(shot, 1) for shot in declared_roles["disruption"]]
        + [(shot, 0) for shot in declared_roles["safe"]]
    ):
        path = data_root / "EAST" / "unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"missing EAST support HDF5: {path}")
        with h5py.File(path, "r") as handle:
            if "meta/IsDisrupt" not in handle:
                raise KeyError(f"EAST support shot {shot} lacks meta/IsDisrupt")
            raw = np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)
            if raw.size != 1 or not np.isfinite(float(raw[0])) or float(raw[0]) not in (0.0, 1.0):
                raise ValueError(f"EAST support shot {shot} has invalid IsDisrupt")
            actual = int(raw[0])
            authoritative[shot] = actual
            if actual == 1:
                if "meta/DownTime" not in handle:
                    invalid_down_time.append(shot)
                else:
                    down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
                    if down.size != 1 or not np.isfinite(float(down[0])):
                        invalid_down_time.append(shot)

    result = summarize_support_labels(declared_roles, authoritative)
    result.update({
        "schema_version": "east-support-label-audit-v1",
        "label_authority": "EAST HDF5 meta/IsDisrupt",
        "down_time_authority": "EAST HDF5 meta/DownTime",
        "roster_path": str(roster_path),
        "roster_file_sha256": hashlib.sha256(roster_path.read_bytes()).hexdigest(),
        "data_root": str(data_root),
        "invalid_disruptive_down_time_shots": invalid_down_time,
        "support_contract_mode": "legacy_role_roster_hdf5_authoritative",
    })
    if invalid_down_time:
        raise ValueError(
            "disruptive EAST support shots have invalid DownTime: "
            f"{invalid_down_time}"
        )
    return result


__all__ = ["audit_east_support110", "summarize_support_labels"]
