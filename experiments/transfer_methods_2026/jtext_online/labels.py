"""Explicit J-TEXT AdvancedTime label contract.

HDF5 ``meta/IsDisrupt`` and ``meta/DownTime`` remain the outcome authority.
The AdvancedTime JSON only changes the upper edge of the positive temporal
window after a disruptive outcome has been revealed.  It is never used for a
safe shot and never used by the prediction-time query score.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


ADV25_POLICY = "advanced_floor25"
FLOOR_MS = 25.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_advanced_time_ms(path: Path) -> dict[int, float]:
    """Load the shot-to-duration mapping in its native millisecond unit."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"AdvancedTime JSON does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid AdvancedTime JSON: {resolved}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("AdvancedTime JSON must be a non-empty object")

    result: dict[int, float] = {}
    for raw_shot, raw_value in payload.items():
        try:
            shot = int(raw_shot)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid AdvancedTime shot ID: {raw_shot!r}") from exc
        if str(shot) != str(raw_shot).strip() or shot <= 0:
            raise ValueError(f"invalid AdvancedTime shot ID: {raw_shot!r}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"shot {shot}: AdvancedTime must be finite positive ms")
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"shot {shot}: AdvancedTime must be finite positive ms")
        if shot in result:
            raise ValueError(f"duplicate AdvancedTime shot ID: {shot}")
        result[shot] = value
    return result


def resolve_positive_horizon_ms(
    *, shot_id: int, is_disrupt: bool, advanced_time_ms: Mapping[int, float] | None,
) -> float | None:
    """Resolve ``max(25 ms, AdvancedTime)`` for disruptive shots only."""

    if not bool(is_disrupt):
        return None
    raw = None if advanced_time_ms is None else advanced_time_ms.get(int(shot_id))
    if raw is None:
        return FLOOR_MS
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"shot {int(shot_id)}: AdvancedTime must be finite positive ms")
    return max(FLOOR_MS, value)


def build_positive_target(
    times_s: np.ndarray, *, is_disrupt: bool, down_time_s: float | None,
    positive_horizon_ms: float | None,
) -> np.ndarray:
    """Build a temporal target after a selected shot has been revealed."""

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("target times must be finite and one-dimensional")
    target = np.zeros(len(times), dtype=np.float32)
    if not bool(is_disrupt):
        return target
    if down_time_s is None or positive_horizon_ms is None:
        raise ValueError("disruptive target requires DownTime and positive horizon")
    down = float(down_time_s)
    horizon = float(positive_horizon_ms)
    if not math.isfinite(down) or not math.isfinite(horizon) or horizon < FLOOR_MS:
        raise ValueError("invalid disruptive target metadata")
    ttd_ms = (down - times) * 1000.0
    target[(ttd_ms >= -1e-9) & (ttd_ms <= horizon + 1e-9)] = 1.0
    return target
