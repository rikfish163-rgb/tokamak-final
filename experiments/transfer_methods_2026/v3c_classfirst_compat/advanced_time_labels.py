"""Per-shot positive-horizon contracts for the AdvancedTime ablation.

HDF5 ``meta/IsDisrupt`` remains the sole outcome authority.  This module only
attaches an upper time-to-disruption boundary to already-disruptive shots:

* J-TEXT: ``max(25 ms, AdvancedTime_ms)``, missing -> 25 ms;
* EAST: ``max(125 ms, AdvancedTime_ms)``, missing -> 125 ms.

The JSON values are interpreted directly as milliseconds.  No seconds-to-ms
conversion is permitted in this code path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


POLICIES = ("fixed125", "advanced_floor")
FLOOR_MS = {"J-TEXT": 25.0, "EAST": 125.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_advanced_time_ms(path: Path) -> dict[int, float]:
    """Load a shot->duration mapping, preserving its native millisecond unit."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"AdvancedTime JSON does not exist: {resolved}")
    try:
        pairs = json.loads(resolved.read_text(encoding="utf-8"), object_pairs_hook=list)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid AdvancedTime JSON: {resolved}") from exc
    if not isinstance(pairs, list) or any(
        not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs
    ):
        raise ValueError("AdvancedTime JSON must be one shot-to-duration object")
    result: dict[int, float] = {}
    for raw_shot, raw_value in pairs:
        try:
            shot = int(raw_shot)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid AdvancedTime shot ID: {raw_shot!r}") from exc
        if str(shot) != str(raw_shot).strip() or shot <= 0:
            raise ValueError(f"invalid AdvancedTime shot ID: {raw_shot!r}")
        if shot in result:
            raise ValueError(f"duplicate AdvancedTime shot ID after normalization: {shot}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(
                f"shot {shot}: AdvancedTime must be finite positive milliseconds"
            )
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"shot {shot}: AdvancedTime must be finite positive milliseconds"
            )
        result[shot] = value
    if not result:
        raise ValueError("AdvancedTime JSON is empty")
    return result


def resolve_positive_horizon_ms(
    *, shot_id: int, device: str, policy: str,
    advanced_time_ms: Mapping[int, float] | None = None,
) -> float:
    """Resolve only the positive upper boundary; never resolve an outcome."""

    if device not in FLOOR_MS:
        raise ValueError(f"unsupported device: {device}")
    if policy not in POLICIES:
        raise ValueError(f"unsupported label policy: {policy}")
    if policy == "fixed125":
        return 125.0
    floor = FLOOR_MS[device]
    value = None if advanced_time_ms is None else advanced_time_ms.get(int(shot_id))
    if value is None:
        return floor
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"shot {int(shot_id)}: AdvancedTime must be finite positive milliseconds"
        )
    return max(floor, value)


def attach_positive_horizons(
    shots: Sequence[dict], *, device: str, policy: str,
    advanced_time_ms: Mapping[int, float] | None = None,
) -> tuple[list[dict], dict]:
    """Copy shots and attach horizons to HDF5-disruptive shots only."""

    if policy == "advanced_floor" and advanced_time_ms is None:
        raise ValueError("advanced_floor requires an AdvancedTime mapping")
    rows: list[dict] = []
    horizons: list[float] = []
    covered = 0
    above_floor = 0
    safe_ignored = 0
    floor = FLOOR_MS.get(device)
    if floor is None:
        raise ValueError(f"unsupported device: {device}")
    for original in shots:
        shot = dict(original)
        shot.pop("positive_horizon_ms", None)
        shot.pop("advanced_time_ms", None)
        shot.pop("positive_horizon_origin", None)
        if bool(shot.get("is_disrupt")):
            shot_id = int(shot["shot"])
            raw = None if advanced_time_ms is None else advanced_time_ms.get(shot_id)
            horizon = resolve_positive_horizon_ms(
                shot_id=shot_id, device=device, policy=policy,
                advanced_time_ms=advanced_time_ms,
            )
            shot["positive_horizon_ms"] = float(horizon)
            shot["advanced_time_ms"] = float(raw) if raw is not None else None
            shot["positive_horizon_origin"] = (
                "fixed125" if policy == "fixed125"
                else "advanced_time" if raw is not None else "floor_fallback"
            )
            horizons.append(float(horizon))
            covered += int(raw is not None)
            above_floor += int(raw is not None and float(raw) > floor)
        else:
            safe_ignored += int(
                advanced_time_ms is not None and int(shot["shot"]) in advanced_time_ms
            )
        rows.append(shot)
    disruptive = len(horizons)
    audit = {
        "device": device,
        "policy": policy,
        "unit": "ms",
        "floor_ms": float(floor),
        "disruptive_shots": disruptive,
        "advanced_time_covered": int(covered),
        "advanced_time_missing": int(disruptive - covered),
        "advanced_time_above_floor": int(above_floor),
        "safe_shots_ignored": int(safe_ignored),
        "horizon_min_ms": min(horizons) if horizons else None,
        "horizon_median_ms": (
            float(sorted(horizons)[len(horizons) // 2]) if horizons else None
        ),
        "horizon_max_ms": max(horizons) if horizons else None,
    }
    return rows, audit
