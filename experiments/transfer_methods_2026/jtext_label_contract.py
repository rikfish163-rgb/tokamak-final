"""Explicit J-TEXT per-shot label-horizon contracts.

The historical V3C source run used a fixed 125 ms horizon.  The main updated
source contract changes only that horizon:

    H_i = max(25 ms, AdvancedTime_i)

For the main contract, ``ttd > H_i`` is negative.  A wider unlabeled context
guard is available only as an explicitly requested diagnostic option; it must
never be silently applied to the main checkpoint.  ``AdvancedTime`` is a
weak-label horizon only.  HDF5 ``IsDisrupt`` and ``DownTime`` remain the
authority for the hard outcome and terminal time.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "jtext-advanced-floor25-label-contract-v3"
DEFAULT_FLOOR_MS = 25.0
# The guard is intentionally opt-in.  The requested Adv25 main experiment is
# a literal single-factor label change and therefore uses no ambiguity band.
DEFAULT_AMBIGUOUS_CONTEXT_UNTIL_MS: float | None = None
OPTIONAL_AMBIGUOUS_CONTEXT_UNTIL_MS = 125.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_advanced_time_ms(path: Path) -> dict[int, float]:
    """Load and validate the shot-list AdvancedTime values in milliseconds."""

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"J-TEXT AdvancedTime file does not exist: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("J-TEXT AdvancedTime must be a JSON object shot_id -> milliseconds")
    result: dict[int, float] = {}
    for raw_shot, raw_value in payload.items():
        shot = int(raw_shot)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid J-TEXT AdvancedTime for shot {shot}: {raw_value!r}"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid J-TEXT AdvancedTime for shot {shot}: {raw_value!r}")
        if shot in result:
            raise ValueError(f"duplicate J-TEXT AdvancedTime shot {shot}")
        result[shot] = value
    return result


def effective_horizon_ms(
    shot_id: int,
    advanced_times_ms: Mapping[int, float],
    *,
    floor_ms: float = DEFAULT_FLOOR_MS,
) -> tuple[float, float | None, bool]:
    """Return ``(effective, raw, used_fallback)`` for one shot.

    Missing AdvancedTime is an explicit, auditable fallback to the 25 ms floor;
    it is not silently converted to zero or inferred from the outcome.
    """

    floor = float(floor_ms)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("J-TEXT AdvancedTime floor must be finite and positive")
    raw = advanced_times_ms.get(int(shot_id))
    if raw is None:
        return floor, None, True
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid AdvancedTime for shot {shot_id}: {raw!r}")
    return max(floor, value), value, False


def negative_start_ms(
    effective_horizon: float,
    *,
    ambiguous_until_ms: float | None = DEFAULT_AMBIGUOUS_CONTEXT_UNTIL_MS,
) -> float:
    """Return the exclusive lower boundary of the source negative class.

    Positive windows remain exactly ``0 <= ttd <= H``.  When a context guard is
    supplied, ``H < ttd <= guard`` is an unlabeled transition band and is not
    sampled as either class.  Passing ``None`` reproduces the historical
    ``negative_start == H`` behavior when no optional guard is supplied.
    """

    horizon = float(effective_horizon)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("effective horizon must be finite and positive")
    if ambiguous_until_ms is None:
        return horizon
    guard = float(ambiguous_until_ms)
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("ambiguous context boundary must be finite and non-negative")
    return max(horizon, guard)


def contract_metadata(
    *,
    path: Path,
    used_shot_ids: list[int] | tuple[int, ...],
    advanced_times_ms: Mapping[int, float],
    disruptive_shot_ids: list[int] | tuple[int, ...] | None = None,
    floor_ms: float = DEFAULT_FLOOR_MS,
    ambiguous_until_ms: float | None = DEFAULT_AMBIGUOUS_CONTEXT_UNTIL_MS,
) -> dict[str, object]:
    """Create provenance for the roles that were actually loaded.

    ``used_shot_ids`` is the complete role population.  Only disruptive shots
    consume an AdvancedTime value, so callers should provide
    ``disruptive_shot_ids`` when HDF5 outcomes are available.  This keeps safe
    shots from being reported as missing weak-label horizons.
    """

    used_ids = sorted({int(value) for value in used_shot_ids})
    horizon_ids = used_ids if disruptive_shot_ids is None else sorted(
        {int(value) for value in disruptive_shot_ids}
    )
    if not set(horizon_ids).issubset(set(used_ids)):
        raise ValueError("disruptive_shot_ids must be a subset of used_shot_ids")
    effective: list[float] = []
    fallback: list[int] = []
    raw_present = 0
    for shot_id in horizon_ids:
        horizon, raw, used_fallback = effective_horizon_ms(
            shot_id, advanced_times_ms, floor_ms=floor_ms
        )
        effective.append(horizon)
        if raw is not None:
            raw_present += 1
        if used_fallback:
            fallback.append(shot_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": "max(floor_ms, AdvancedTime_ms)",
        "floor_ms": float(floor_ms),
        "source": str(Path(path).expanduser().resolve()),
        "source_sha256": sha256_file(Path(path).expanduser().resolve()),
        "used_shot_count": len(used_ids),
        "used_disruptive_shot_count": len(horizon_ids),
        "raw_advanced_time_present_count": raw_present,
        "fallback_shot_count": len(fallback),
        "fallback_shot_ids": fallback,
        "effective_horizon_min_ms": min(effective) if effective else None,
        "effective_horizon_max_ms": max(effective) if effective else None,
        "ambiguous_context_until_ms": (
            float(ambiguous_until_ms) if ambiguous_until_ms is not None else None
        ),
        "negative_start_rule": (
            "max(effective_horizon_ms, ambiguous_context_until_ms)"
            if ambiguous_until_ms is not None
            else "effective_horizon_ms"
        ),
    }


__all__ = [
    "DEFAULT_FLOOR_MS",
    "DEFAULT_AMBIGUOUS_CONTEXT_UNTIL_MS",
    "OPTIONAL_AMBIGUOUS_CONTEXT_UNTIL_MS",
    "SCHEMA_VERSION",
    "contract_metadata",
    "effective_horizon_ms",
    "load_advanced_time_ms",
    "negative_start_ms",
    "sha256_file",
]
