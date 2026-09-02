"""Causal, outcome-isolated J-TEXT I/O for the selected mainline.

This module is deliberately split into two boundaries:

* :func:`load_jtext_signal_only` can read only the fourteen raw signals and
  strict time-base attributes needed by rebuilt17;
* :class:`JTextLazyOutcomeOracle` can read HDF5 outcomes and AdvancedTime only
  after a prediction has been sealed and the shot has been selected for query.

The same prefix-causal feature builder and zero-left-padded window function are
used by source training, static TEST220 inference, and the online diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import h5py
import numpy as np

from . import advanced_time_labels as advanced_labels
import rebuilt_harness as rebuilt


SOURCE_RAW_SIGNAL_NAMES = (
    "Bt", "ip", "v_loop", "sxr_kurt", "sxr_skew", "sxr_var",
    "xuv_kurt", "xuv_skew", "xuv_var", "Mir_avg_fre",
    "n=1 amplitude", "mode_number_n", "dx_a", "dy_a",
)
REBUILD_FEATURE_NAMES = tuple(rebuilt.REBUILD)
SOURCE_INPUT_SCHEMA = "jtext-rebuilt17-prefix-causal-v2"
PREPROCESSING_MANIFEST_SCHEMA = "jtext-prefix-causal-preprocessing-manifest-v1"
ORDERED_TEST220_SCHEMA = "jtext-test220-ordered-v1"
TEST220_CLAIM_STATUS = "TEST_DRIVEN_EXPLORATION"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def jtext_shot_path(data_root: Path, shot_id: int) -> Path:
    shot = int(shot_id)
    return (
        Path(data_root).expanduser().resolve()
        / "J-TEXT/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    )


def _strict_timebase(ip_signal: h5py.Dataset) -> tuple[float, float]:
    for name in ("SampleRate", "StartTime"):
        if name not in ip_signal.attrs:
            raise RuntimeError(f"J-TEXT signal metadata is missing {name}")
        raw = np.asarray(ip_signal.attrs[name]).reshape(-1)
        if raw.size != 1:
            raise RuntimeError(f"J-TEXT {name} must be scalar")
        value = float(raw[0])
        if not math.isfinite(value):
            raise RuntimeError(f"J-TEXT {name} must be finite")
        if name == "SampleRate" and value <= 0.0:
            raise RuntimeError("J-TEXT SampleRate must be positive")
    return (
        float(np.asarray(ip_signal.attrs["SampleRate"]).reshape(-1)[0]),
        float(np.asarray(ip_signal.attrs["StartTime"]).reshape(-1)[0]),
    )


def prefix_causal_rebuild_features(
    raw_signals: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build rebuilt17 using availability known at each causal prefix only."""

    missing = [name for name in SOURCE_RAW_SIGNAL_NAMES if name not in raw_signals]
    if missing:
        raise KeyError(f"J-TEXT rebuilt17 raw signals are missing: {missing}")
    arrays: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for name in SOURCE_RAW_SIGNAL_NAMES:
        value = np.asarray(raw_signals[name], dtype=np.float32)
        if value.ndim != 1 or not len(value):
            raise ValueError(f"J-TEXT signal {name} must be a non-empty 1-D trace")
        arrays[name] = value
        lengths.add(len(value))
    if len(lengths) != 1:
        raise ValueError("J-TEXT rebuilt17 raw signals have unequal lengths")
    return rebuilt.rebuild_features_prefix_causal(arrays, "J-TEXT")


def causal_prefix_window(
    values: np.ndarray, endpoint: int, window_points: int = 128,
) -> tuple[np.ndarray, int]:
    """Return only history through ``endpoint`` with constant-zero left padding."""

    array = np.asarray(values, dtype=np.float32)
    endpoint = int(endpoint)
    width = int(window_points)
    if array.ndim != 2:
        raise ValueError("J-TEXT causal input must have shape [time, channel]")
    if not 0 <= endpoint < len(array):
        raise IndexError("J-TEXT causal endpoint is outside the shot")
    if width <= 0:
        raise ValueError("J-TEXT causal window must be positive")
    start = max(0, endpoint - width + 1)
    history = array[start : endpoint + 1]
    window = np.zeros((width, array.shape[1]), dtype=np.float32)
    window[-len(history) :] = history
    return np.ascontiguousarray(window), int(len(history))


@dataclass(frozen=True)
class JTextSignalShot:
    """Label-free, immutable signal view for one J-TEXT shot."""

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
        if values.ndim != 2 or values.shape[1] != len(REBUILD_FEATURE_NAMES):
            raise ValueError("J-TEXT signal shot has an invalid rebuilt17 shape")
        if availability.shape != values.shape or times.shape != (len(values),):
            raise ValueError("J-TEXT signal shot arrays are not aligned")
        if not math.isfinite(float(self.sample_rate_hz)) or self.sample_rate_hz <= 0:
            raise ValueError("J-TEXT signal shot has an invalid sample rate")
        if not math.isfinite(float(self.start_time_s)):
            raise ValueError("J-TEXT signal shot has an invalid start time")
        values.setflags(write=False)
        availability.setflags(write=False)
        times.setflags(write=False)
        object.__setattr__(self, "shot_id", int(self.shot_id))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))
        object.__setattr__(self, "start_time_s", float(self.start_time_s))

    def causal_window(
        self, endpoint: int, *, window_points: int = 128,
    ) -> tuple[np.ndarray, int]:
        return causal_prefix_window(self.values, endpoint, window_points)


def load_jtext_signal_only(data_root: Path, shot_id: int) -> JTextSignalShot:
    """Load rebuilt17 inputs without touching HDF5 ``meta`` or label files."""

    path = jtext_shot_path(data_root, shot_id)
    if not path.is_file():
        raise FileNotFoundError(f"J-TEXT HDF5 shot does not exist: {path}")
    raw: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for name in SOURCE_RAW_SIGNAL_NAMES:
            key = f"data/{name}"
            if key not in handle:
                raise KeyError(f"J-TEXT shot {int(shot_id)} is missing {key}")
            value = np.asarray(handle[key][:], dtype=np.float32)
            if value.ndim != 1 or not len(value):
                raise ValueError(
                    f"J-TEXT shot {int(shot_id)} signal {name} is not a non-empty 1-D trace"
                )
            raw[name] = value
        rate, start = _strict_timebase(handle["data/ip"])
    values, availability = prefix_causal_rebuild_features(raw)
    times = start + np.arange(len(values), dtype=np.float64) / rate
    return JTextSignalShot(
        shot_id=int(shot_id), values=values, availability=availability,
        times_s=times, sample_rate_hz=rate, start_time_s=start,
    )


@dataclass(frozen=True)
class JTextRevealedOutcome:
    shot_id: int
    is_disrupt: bool
    down_time_s: float | None
    advanced_time_ms: float | None
    positive_horizon_ms: float | None


def _read_hdf5_outcome(data_root: Path, shot_id: int) -> tuple[bool, float | None]:
    path = jtext_shot_path(data_root, shot_id)
    with h5py.File(path, "r") as handle:
        if "meta/IsDisrupt" not in handle:
            raise KeyError(f"J-TEXT shot {int(shot_id)} has no meta/IsDisrupt")
        raw_outcome = np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)
        if raw_outcome.size != 1:
            raise ValueError("J-TEXT IsDisrupt must be scalar")
        is_disrupt = bool(raw_outcome[0])
        if not is_disrupt:
            return False, None
        if "meta/DownTime" not in handle:
            raise KeyError(f"disruptive J-TEXT shot {int(shot_id)} has no DownTime")
        raw_down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
        if raw_down.size != 1 or not math.isfinite(float(raw_down[0])):
            raise ValueError(
                f"disruptive J-TEXT shot {int(shot_id)} has no finite DownTime"
            )
        return True, float(raw_down[0])


class JTextLazyOutcomeOracle:
    """Ordered post-seal oracle; unselected shots can only be discarded."""

    def __init__(
        self, *, data_root: Path, ordered_roster: Sequence[int],
        advanced_time_path: Path | None,
    ) -> None:
        roster = tuple(int(shot) for shot in ordered_roster)
        if not roster or len(roster) != len(set(roster)):
            raise ValueError("J-TEXT online roster must be non-empty and unique")
        self._data_root = Path(data_root).expanduser().resolve()
        self._roster = roster
        self._advanced_time_path = (
            Path(advanced_time_path).expanduser().resolve()
            if advanced_time_path is not None else None
        )
        self._cursor = 0
        self._sealed_shot: int | None = None
        self._revealed: set[int] = set()
        self._discarded: set[int] = set()

    @property
    def cursor(self) -> int:
        return self._cursor

    def seal_current(self, shot_id: int, *, prediction_sha256: str) -> None:
        shot = int(shot_id)
        if self._cursor >= len(self._roster):
            raise RuntimeError("J-TEXT online roster is already complete")
        if shot in self._revealed or shot in self._discarded:
            raise RuntimeError("duplicate J-TEXT shot finalization")
        if self._sealed_shot is not None:
            raise RuntimeError("the previous J-TEXT shot seal is still open")
        if shot != self._roster[self._cursor]:
            raise RuntimeError("J-TEXT shot seal violates the ordered roster")
        if not isinstance(prediction_sha256, str) or _SHA256.fullmatch(
            prediction_sha256
        ) is None:
            raise ValueError("J-TEXT prediction seal requires a SHA256 digest")
        self._sealed_shot = shot

    def reveal_for_query(
        self, shot_id: int, *, selected: bool,
    ) -> JTextRevealedOutcome:
        shot = int(shot_id)
        if shot in self._revealed:
            raise RuntimeError("duplicate J-TEXT outcome reveal")
        if self._sealed_shot != shot:
            raise RuntimeError("J-TEXT outcome reveal requires the current shot seal")
        if not selected:
            raise RuntimeError("J-TEXT outcome reveal requires a selected query")
        is_disrupt, down_time_s = _read_hdf5_outcome(self._data_root, shot)
        advanced_time_ms: float | None = None
        positive_horizon_ms: float | None = None
        if is_disrupt:
            advanced: dict[int, float] | None = None
            if self._advanced_time_path is not None:
                advanced = advanced_labels.load_advanced_time_ms(
                    self._advanced_time_path
                )
                advanced_time_ms = advanced.get(shot)
            positive_horizon_ms = advanced_labels.resolve_positive_horizon_ms(
                shot_id=shot, device="J-TEXT", policy="advanced_floor",
                advanced_time_ms=advanced,
            )
        result = JTextRevealedOutcome(
            shot_id=shot,
            is_disrupt=is_disrupt,
            down_time_s=down_time_s,
            advanced_time_ms=(
                float(advanced_time_ms) if advanced_time_ms is not None else None
            ),
            positive_horizon_ms=positive_horizon_ms,
        )
        self._revealed.add(shot)
        self._sealed_shot = None
        self._cursor += 1
        return result

    def discard_unqueried(self, shot_id: int) -> None:
        """Advance one sealed shot without reading any outcome-bearing source."""

        shot = int(shot_id)
        if self._sealed_shot != shot:
            raise RuntimeError("J-TEXT unqueried discard requires the current shot seal")
        self._discarded.add(shot)
        self._sealed_shot = None
        self._cursor += 1


def build_postseal_jtext_target(
    revealed: JTextRevealedOutcome, times_s: np.ndarray,
) -> np.ndarray:
    """Construct a temporal target only from a selected, revealed outcome."""

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("J-TEXT post-seal target requires finite one-dimensional times")
    target = np.zeros(len(times), dtype=np.float32)
    if not revealed.is_disrupt:
        return target
    if revealed.down_time_s is None or revealed.positive_horizon_ms is None:
        raise ValueError("disruptive J-TEXT reveal lacks DownTime or positive horizon")
    ttd_ms = (float(revealed.down_time_s) - times) * 1000.0
    target[(ttd_ms >= -1e-9) & (ttd_ms <= revealed.positive_horizon_ms + 1e-9)] = 1.0
    return target


@dataclass(frozen=True)
class OrderedJTextTest220:
    schema_version: str
    claim_status: str
    order_rule: str
    shots: tuple[int, ...]
    roster_sha256: str
    roster_file_sha256: str


def load_canonical_jtext_roles(root: Path) -> dict[str, list[int]]:
    root = Path(root).expanduser().resolve()
    index = root / "artifacts/canonical/jtext_source_bundle_p131_v1/index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        raw = payload.get("split_shot_ids")
        if not isinstance(raw, Mapping):
            raise ValueError("canonical J-TEXT bundle has no split_shot_ids")
        roles = {
            role: [int(shot) for shot in raw[role]]
            for role in ("train", "val", "test")
        }
    else:
        roles = {
            role: [
                int(line.strip())
                for line in (root / f"splits/{role}.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            for role in ("train", "val", "test")
        }
    expected = {"train": 1353, "val": 160, "test": 220}
    if {role: len(shots) for role, shots in roles.items()} != expected:
        raise ValueError("canonical J-TEXT role counts changed")
    for role, shots in roles.items():
        if len(shots) != len(set(shots)):
            raise ValueError(f"duplicate shot IDs in J-TEXT {role}")
    if any(
        set(roles[left]) & set(roles[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("canonical J-TEXT roles overlap")
    return roles


def load_ordered_test220(root: Path) -> OrderedJTextTest220:
    root = Path(root).expanduser().resolve()
    config_path = root / "config/splits/jtext_test220_ordered_v1.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ORDERED_TEST220_SCHEMA:
        raise ValueError("ordered J-TEXT TEST220 schema mismatch")
    if payload.get("claim_status") != TEST220_CLAIM_STATUS:
        raise ValueError("ordered J-TEXT TEST220 claim status mismatch")
    if payload.get("order_rule") != "ascending_shot_id":
        raise ValueError("ordered J-TEXT TEST220 order rule mismatch")
    roster_path = root / str(payload.get("roster_path", ""))
    if not roster_path.is_file():
        raise FileNotFoundError("ordered J-TEXT TEST220 roster does not exist")
    if _sha256_file(roster_path) != payload.get("roster_file_sha256"):
        raise ValueError("ordered J-TEXT TEST220 roster file SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in roster_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(
        not isinstance(row, dict) or set(row) != {"shot", "stream_index"}
        for row in rows
    ):
        raise ValueError("ordered J-TEXT TEST220 roster is not label-free")
    if [row["stream_index"] for row in rows] != list(range(220)):
        raise ValueError("ordered J-TEXT TEST220 indices are not contiguous")
    shots = tuple(int(row["shot"]) for row in rows)
    if len(shots) != 220 or len(set(shots)) != 220:
        raise ValueError("ordered J-TEXT TEST220 must contain 220 unique shots")
    if list(shots) != sorted(shots):
        raise ValueError("ordered J-TEXT TEST220 is not ascending by shot ID")
    if _sha256_json(list(shots)) != payload.get("roster_sha256"):
        raise ValueError("ordered J-TEXT TEST220 roster SHA256 mismatch")
    roles = load_canonical_jtext_roles(root)
    if list(shots) != roles["test"]:
        raise ValueError("ordered J-TEXT TEST220 differs from the canonical test role")
    return OrderedJTextTest220(
        schema_version=ORDERED_TEST220_SCHEMA,
        claim_status=TEST220_CLAIM_STATUS,
        order_rule="ascending_shot_id",
        shots=shots,
        roster_sha256=str(payload["roster_sha256"]),
        roster_file_sha256=str(payload["roster_file_sha256"]),
    )


def build_preprocessing_manifest(root: Path) -> dict[str, object]:
    """Hash the live code closure that defines selected source preprocessing."""

    root = Path(root).expanduser().resolve()
    relative_files = (
        "experiments/transfer_methods_2026/jtext_online_io.py",
        "experiments/transfer_methods_2026/rebuilt_harness.py",
        "experiments/transfer_methods_2026/grid_search.py",
        "experiments/transfer_methods_2026/train_jtext_source_transformer.py",
    )
    files: list[dict[str, str]] = []
    for relative in relative_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"source preprocessing closure is missing {relative}")
        files.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema_version": PREPROCESSING_MANIFEST_SCHEMA,
        "files": files,
        "parameters": {
            "feature_builder": "rebuilt17",
            "mhd_availability": "finite-and-prefix-causal-seen-nonzero-v1",
            "history_alignment": "right",
            "left_padding": "constant_zero",
            "window_points": 128,
            "valid_lengths_required": True,
        },
    }


def build_source_input_contract(root: Path) -> dict[str, object]:
    manifest = build_preprocessing_manifest(root)
    return {
        "schema": SOURCE_INPUT_SCHEMA,
        "raw_signal_names": list(SOURCE_RAW_SIGNAL_NAMES),
        "feature_names": list(REBUILD_FEATURE_NAMES),
        "window_points": 128,
        "history_alignment": "right",
        "left_padding": "constant_zero",
        "valid_lengths_required": True,
        "valid_length_range": [1, 128],
        "patch_mask_rule": "patch_end <= window_points - valid_length",
        "availability": "finite-and-prefix-causal-seen-nonzero-v1",
        "preprocessing_manifest": manifest,
        "preprocessing_sha256": _sha256_json(manifest),
    }
