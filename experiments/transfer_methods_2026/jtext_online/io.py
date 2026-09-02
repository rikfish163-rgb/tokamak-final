"""Causal signal I/O and delayed HDF5 outcome access for J-TEXT TEST220."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from . import labels


RAW_SIGNAL_NAMES = (
    "Bt", "ip", "v_loop", "sxr_kurt", "sxr_skew", "sxr_var",
    "xuv_kurt", "xuv_skew", "xuv_var", "Mir_avg_fre",
    "n=1 amplitude", "mode_number_n", "dx_a", "dy_a",
)
FEATURE_NAMES = (
    "Bt", "ip_A", "Bt_over_Ip", "Ip_diff", "v_loop", "Z_proxy",
    "sxr_kurt", "sxr_skew", "sxr_var", "xuv_kurt", "xuv_skew",
    "xuv_var", "Mir_avg_fre", "n=1 amplitude", "mode_number_n",
    "dx_a", "dy_a",
)
FEATURE_COUNT = len(FEATURE_NAMES)
WINDOW_POINTS = 128
ORDERED_ROSTER_SCHEMA = "jtext-test220-ordered-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def jtext_shot_path(data_root: Path, shot_id: int) -> Path:
    shot = int(shot_id)
    return (
        Path(data_root).expanduser().resolve()
        / "J-TEXT/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    )


def _scalar_attr(dataset: h5py.Dataset, name: str) -> float:
    if name not in dataset.attrs:
        raise RuntimeError(f"J-TEXT signal metadata is missing {name}")
    raw = np.asarray(dataset.attrs[name]).reshape(-1)
    if raw.size != 1:
        raise RuntimeError(f"J-TEXT {name} must be scalar")
    value = float(raw[0])
    if not math.isfinite(value):
        raise RuntimeError(f"J-TEXT {name} must be finite")
    if name == "SampleRate" and value <= 0.0:
        raise RuntimeError("J-TEXT SampleRate must be positive")
    return value


def prefix_causal_rebuild_features(
    raw_signals: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the 17 channels without future-dependent availability checks.

    The only special handling is for the two MHD channels.  A zero-valued MHD
    sample is treated as unavailable until a non-zero observation has appeared
    in the prefix; an all-zero shot therefore never becomes spuriously
    available because a later sample was inspected.
    """

    missing = [name for name in RAW_SIGNAL_NAMES if name not in raw_signals]
    if missing:
        raise KeyError(f"J-TEXT raw signals are missing: {missing}")
    arrays = {
        name: np.asarray(raw_signals[name], dtype=np.float32)
        for name in RAW_SIGNAL_NAMES
    }
    lengths = {len(value) for value in arrays.values()}
    if any(value.ndim != 1 or not len(value) for value in arrays.values()):
        raise ValueError("J-TEXT raw signals must be non-empty one-dimensional arrays")
    if len(lengths) != 1:
        raise ValueError("J-TEXT raw signals have unequal lengths")

    length = len(next(iter(arrays.values())))
    out = np.zeros((length, FEATURE_COUNT), dtype=np.float32)
    available = np.zeros((length, FEATURE_COUNT), dtype=bool)

    def put(name: str, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        column = FEATURE_NAMES.index(name)
        finite = np.isfinite(values)
        mask = finite if valid is None else finite & np.asarray(valid, dtype=bool)
        out[:, column] = np.where(mask, values, 0.0).astype(np.float32)
        available[:, column] = mask

    ip = arrays["ip"] * 1000.0  # J-TEXT stores Ip in kA; convert exactly once.
    ip_finite = np.isfinite(ip)
    denominator = np.where(np.abs(ip) > 1e-9, ip, np.nan)
    put("Bt", arrays["Bt"])
    put("ip_A", ip, ip_finite)
    put(
        "Bt_over_Ip", arrays["Bt"] / denominator,
        ip_finite & np.isfinite(arrays["Bt"]) & (np.abs(ip) > 1e-9),
    )
    ip_diff = np.zeros_like(ip)
    ip_diff[1:] = np.diff(ip)
    put("Ip_diff", ip_diff, ip_finite)
    put("v_loop", arrays["v_loop"])
    put("Z_proxy", arrays["v_loop"] / denominator,
        np.isfinite(arrays["v_loop"]) & (np.abs(ip) > 1e-9))
    for name in (
        "sxr_kurt", "sxr_skew", "sxr_var", "xuv_kurt", "xuv_skew",
        "xuv_var", "dx_a", "dy_a", "n=1 amplitude",
    ):
        put(name, arrays[name])

    for name in ("Mir_avg_fre", "mode_number_n"):
        raw = arrays[name]
        evidence = np.isfinite(raw) & ~np.isclose(raw, 0.0)
        seen = np.logical_or.accumulate(evidence)
        put(name, raw, np.isfinite(raw) & seen)
    return np.ascontiguousarray(out), np.ascontiguousarray(available)


def causal_prefix_window(
    values: np.ndarray, endpoint: int, *, width: int = WINDOW_POINTS,
) -> tuple[np.ndarray, int]:
    """Return a fixed window containing only samples through ``endpoint``."""

    array = np.asarray(values, dtype=np.float32)
    endpoint = int(endpoint)
    width = int(width)
    if array.ndim != 2 or array.shape[1] != FEATURE_COUNT:
        raise ValueError("causal J-TEXT values must have shape [time, 17]")
    if not 0 <= endpoint < len(array):
        raise IndexError("causal endpoint is outside the shot")
    if width <= 0:
        raise ValueError("causal window width must be positive")
    start = max(0, endpoint - width + 1)
    history = array[start : endpoint + 1]
    window = np.zeros((width, FEATURE_COUNT), dtype=np.float32)
    window[-len(history) :] = history
    return np.ascontiguousarray(window), int(len(history))


@dataclass(frozen=True)
class SignalShot:
    """Immutable signal-only view; no HDF5 outcome fields are stored."""

    shot_id: int
    values: np.ndarray
    availability: np.ndarray
    times_s: np.ndarray
    sample_rate_hz: float
    start_time_s: float

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        availability = np.ascontiguousarray(self.availability, dtype=bool)
        times = np.ascontiguousarray(self.times_s, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
            raise ValueError("signal shot has an invalid rebuilt17 shape")
        if availability.shape != values.shape or times.shape != (len(values),):
            raise ValueError("signal shot arrays are not aligned")
        if not math.isfinite(float(self.sample_rate_hz)) or self.sample_rate_hz <= 0:
            raise ValueError("signal shot has an invalid sample rate")
        if not math.isfinite(float(self.start_time_s)):
            raise ValueError("signal shot has an invalid start time")
        values.setflags(write=False)
        availability.setflags(write=False)
        times.setflags(write=False)
        object.__setattr__(self, "shot_id", int(self.shot_id))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))
        object.__setattr__(self, "start_time_s", float(self.start_time_s))

    def scaled(self, center: np.ndarray, scale: np.ndarray) -> "SignalShot":
        center = np.asarray(center, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        if center.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,):
            raise ValueError("J-TEXT scaler shape must be [17]")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("J-TEXT scaler must be finite and positive")
        values = np.clip((self.values - center) / scale, -10.0, 10.0)
        values = np.where(self.availability, values, 0.0).astype(np.float32)
        return SignalShot(
            shot_id=self.shot_id, values=values, availability=self.availability,
            times_s=self.times_s, sample_rate_hz=self.sample_rate_hz,
            start_time_s=self.start_time_s,
        )

    def causal_window(self, endpoint: int) -> tuple[np.ndarray, int]:
        return causal_prefix_window(self.values, endpoint)


def load_signal_only(data_root: Path, shot_id: int) -> SignalShot:
    """Read signal datasets and time attributes, never HDF5 ``meta``."""

    path = jtext_shot_path(data_root, shot_id)
    if not path.is_file():
        raise FileNotFoundError(f"J-TEXT HDF5 shot does not exist: {path}")
    raw: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for name in RAW_SIGNAL_NAMES:
            key = f"data/{name}"
            if key not in handle:
                raise KeyError(f"J-TEXT shot {int(shot_id)} is missing {key}")
            value = np.asarray(handle[key][:], dtype=np.float32)
            if value.ndim != 1 or not len(value):
                raise ValueError(f"J-TEXT signal {name} is not a non-empty 1-D trace")
            raw[name] = value
        rate = _scalar_attr(handle["data/ip"], "SampleRate")
        start = _scalar_attr(handle["data/ip"], "StartTime")
    values, availability = prefix_causal_rebuild_features(raw)
    times = start + np.arange(len(values), dtype=np.float64) / rate
    return SignalShot(
        shot_id=int(shot_id), values=values, availability=availability,
        times_s=times, sample_rate_hz=rate, start_time_s=start,
    )


@dataclass(frozen=True)
class RevealedOutcome:
    shot_id: int
    is_disrupt: bool
    down_time_s: float | None
    advanced_time_ms: float | None
    positive_horizon_ms: float | None


def _read_hdf5_outcome(data_root: Path, shot_id: int) -> tuple[bool, float | None]:
    path = jtext_shot_path(data_root, shot_id)
    with h5py.File(path, "r") as handle:
        if "meta/IsDisrupt" not in handle or "meta/DownTime" not in handle:
            raise KeyError(f"J-TEXT shot {int(shot_id)} lacks authoritative meta fields")
        disruptive = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        raw_down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    if not disruptive:
        return False, None
    if raw_down.size != 1 or not math.isfinite(float(raw_down[0])):
        raise ValueError(f"disruptive J-TEXT shot {int(shot_id)} has invalid DownTime")
    return True, float(raw_down[0])


class OrderedOutcomeOracle:
    """Block transaction enforcing predict -> seal -> query -> reveal.

    The oracle cannot reveal a shot until every prediction in the current block
    has been sealed and the shot is explicitly selected.  Unselected shots are
    advanced without reading their outcome.  This is the causal boundary used
    by the diagnostic replay; post-stream evaluation has a separate explicit
    ``read_all_after_stream`` call.
    """

    def __init__(
        self, *, data_root: Path, roster: Sequence[int],
        advanced_time_path: Path,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.roster = tuple(int(shot) for shot in roster)
        if not self.roster or len(self.roster) != len(set(self.roster)):
            raise ValueError("J-TEXT online roster must be non-empty and unique")
        self.advanced_time_path = Path(advanced_time_path).expanduser().resolve()
        self._advanced: dict[int, float] | None = None
        self._cursor = 0
        self._block: tuple[int, ...] | None = None
        self._sealed: dict[int, str] = {}
        self._ready = False
        self._revealed: dict[int, RevealedOutcome] = {}
        self._outcome_reads = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def outcome_reads(self) -> int:
        return self._outcome_reads

    @property
    def revealed_shots(self) -> tuple[int, ...]:
        return tuple(sorted(self._revealed))

    def begin_block(self, shots: Sequence[int]) -> None:
        block = tuple(int(shot) for shot in shots)
        expected = self.roster[self._cursor : self._cursor + len(block)]
        if self._block is not None or self._ready:
            raise RuntimeError("previous J-TEXT prediction block is still open")
        if not block or block != expected:
            raise RuntimeError("J-TEXT block does not follow the ordered roster")
        self._block = block
        self._sealed = {}

    def seal_prediction(self, shot_id: int, prediction_sha256: str) -> None:
        if self._block is None or self._ready:
            raise RuntimeError("begin_block must precede prediction sealing")
        shot = int(shot_id)
        if shot not in self._block or shot in self._sealed:
            raise RuntimeError("invalid or duplicate J-TEXT prediction seal")
        if not isinstance(prediction_sha256, str) or _SHA256.fullmatch(prediction_sha256) is None:
            raise ValueError("prediction seal requires a SHA256 digest")
        self._sealed[shot] = prediction_sha256

    def finish_prediction_block(self) -> None:
        if self._block is None or set(self._sealed) != set(self._block):
            raise RuntimeError("every current-shot prediction must be sealed first")
        self._ready = True

    def reveal_for_query(self, shot_id: int, *, selected: bool) -> RevealedOutcome:
        if not self._ready or self._block is None:
            raise RuntimeError("outcome reveal requires a sealed prediction block")
        shot = int(shot_id)
        if not selected:
            raise RuntimeError("outcome reveal requires an explicit selected query")
        if shot not in self._block or shot in self._revealed:
            raise RuntimeError("invalid or duplicate J-TEXT outcome reveal")
        is_disrupt, down_time_s = _read_hdf5_outcome(self.data_root, shot)
        self._outcome_reads += 1
        advanced_value: float | None = None
        horizon: float | None = None
        if is_disrupt:
            if self._advanced is None:
                self._advanced = labels.load_advanced_time_ms(self.advanced_time_path)
            advanced_value = self._advanced.get(shot)
            horizon = labels.resolve_positive_horizon_ms(
                shot_id=shot, is_disrupt=True, advanced_time_ms=self._advanced,
            )
        outcome = RevealedOutcome(
            shot_id=shot, is_disrupt=is_disrupt, down_time_s=down_time_s,
            advanced_time_ms=advanced_value, positive_horizon_ms=horizon,
        )
        self._revealed[shot] = outcome
        return outcome

    def complete_block(self, selected_shots: Sequence[int]) -> None:
        if not self._ready or self._block is None:
            raise RuntimeError("prediction block is not ready to complete")
        selected = {int(shot) for shot in selected_shots}
        if not selected.issubset(set(self._block)):
            raise RuntimeError("selected J-TEXT shot is outside the current block")
        if not selected.issubset(self._revealed):
            raise RuntimeError("all selected J-TEXT outcomes must be revealed")
        self._cursor += len(self._block)
        self._block = None
        self._sealed = {}
        self._ready = False

    def read_all_after_stream(self) -> dict[int, RevealedOutcome]:
        """Open remaining labels only after the complete prediction stream."""

        if self._cursor != len(self.roster) or self._block is not None or self._ready:
            raise RuntimeError("full outcome evaluation requires a completed stream")
        if self._advanced is None:
            self._advanced = labels.load_advanced_time_ms(self.advanced_time_path)
        result = dict(self._revealed)
        for shot in self.roster:
            if shot in result:
                continue
            is_disrupt, down_time_s = _read_hdf5_outcome(self.data_root, shot)
            self._outcome_reads += 1
            advanced_value = self._advanced.get(shot) if is_disrupt else None
            result[shot] = RevealedOutcome(
                shot_id=shot, is_disrupt=is_disrupt, down_time_s=down_time_s,
                advanced_time_ms=advanced_value,
                positive_horizon_ms=labels.resolve_positive_horizon_ms(
                    shot_id=shot, is_disrupt=is_disrupt,
                    advanced_time_ms=self._advanced,
                ),
            )
        return result


@dataclass(frozen=True)
class OrderedTest220:
    schema_version: str
    claim_status: str
    order_rule: str
    shots: tuple[int, ...]
    roster_sha256: str
    roster_file_sha256: str


def load_ordered_test220(root: Path) -> OrderedTest220:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "config/splits/jtext_test220_ordered_v1.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ordered TEST220 manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ORDERED_ROSTER_SCHEMA:
        raise ValueError("ordered TEST220 schema mismatch")
    roster_path = root / str(payload.get("roster_path", ""))
    if not roster_path.is_file():
        raise FileNotFoundError("ordered TEST220 roster does not exist")
    if sha256_file(roster_path) != payload.get("roster_file_sha256"):
        raise ValueError("ordered TEST220 roster file SHA256 mismatch")
    shots: list[int] = []
    for line in roster_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if not line.strip().isdigit():
            raise ValueError("ordered TEST220 roster must contain only shot IDs")
        shots.append(int(line.strip()))
    if len(shots) != 220 or len(set(shots)) != 220 or shots != sorted(shots):
        raise ValueError("ordered TEST220 roster must contain 220 sorted unique shots")
    if sha256_json(shots) != payload.get("roster_sha256"):
        raise ValueError("ordered TEST220 roster JSON SHA256 mismatch")
    role_sets = []
    for role in ("train", "val", "test"):
        path = root / f"splits/{role}.txt"
        role_sets.append({int(line) for line in path.read_text().split() if line})
    if set(shots) != role_sets[2] or any(role_sets[i] & role_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("ordered TEST220 conflicts with the canonical split roles")
    return OrderedTest220(
        schema_version=ORDERED_ROSTER_SCHEMA,
        claim_status=str(payload["claim_status"]),
        order_rule=str(payload["order_rule"]), shots=tuple(shots),
        roster_sha256=str(payload["roster_sha256"]),
        roster_file_sha256=str(payload["roster_file_sha256"]),
    )
