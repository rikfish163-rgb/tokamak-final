"""Outcome-blind, current-only offline ordering of a supplied EAST roster.

The scheduler uses each shot's initial 100 ms, including prefixes of shots
that are future relative to the replay. These prefixes are not model inputs.
The squared-current quantity is a relative proxy, not measured damage.
"""

import math

import h5py
import numpy as np

from .. import io23


PREFIX_SECONDS = 0.1
CURRENT_REFERENCE_A = 100_000.0


def _finite_scalar(value, name):
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _prefix_length(rate):
    rate = _finite_scalar(rate, "SampleRate")
    if not math.isclose(rate, 1000.0, rel_tol=0.0, abs_tol=io23.NOMINAL_EAST_RATE_ATOL_HZ):
        raise ValueError("EAST prefix requires the nominal 1 kHz clock")
    return rate, math.ceil(PREFIX_SECONDS * rate - 1e-10)


def read_prefix(shot):
    """Read only the raw Ip prefix and its own clock, never other channels or meta."""
    with h5py.File(io23.path_for("EAST", shot), "r") as handle:
        # Do not allow links or virtual/external storage to redirect the read to
        # another channel or file. This checks only the two requested path links.
        if not all(isinstance(handle.get(path, getlink=True), h5py.HardLink)
                   for path in ("data", "data/ip")):
            raise ValueError("data/ip must be a local hard-linked dataset")
        ip = handle["data/ip"]
        if not isinstance(ip, h5py.Dataset) or ip.is_virtual or ip.external:
            raise ValueError("data/ip must be a local dataset")
        rate, n = _prefix_length(ip.attrs.get("SampleRate"))
        start = _finite_scalar(ip.attrs.get("StartTime"), "StartTime")
        return {"planning_ip": ip[:n], "rate": rate, "start": start}


def _record(shot, raw):
    rate, n = _prefix_length(raw["rate"])
    start = _finite_scalar(raw["start"], "StartTime")
    # Slice before conversion: an injected lazy reader must not touch the
    # remainder of a shot merely to compute this initial-prefix statistic.
    prefix = np.asarray(raw["planning_ip"][:n])
    if prefix.ndim != 1 or len(prefix) != n or prefix.dtype.kind not in "iuf":
        raise ValueError("insufficient or invalid initial Ip prefix")
    try:
        prefix = np.asarray(prefix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("initial Ip prefix must be numeric") from error
    if not np.isfinite(prefix).all():
        raise ValueError("initial Ip prefix must be finite; no imputation")
    with np.errstate(over="ignore", invalid="ignore"):
        current = float(np.median(np.abs(prefix)))
    if not math.isfinite(current) or current <= 0.0:
        raise ValueError("initial absolute Ip median must be positive and finite")
    try:
        energy = (current / CURRENT_REFERENCE_A) ** 2
    except OverflowError as error:
        raise ValueError("nonfinite current-energy proxy") from error
    if not math.isfinite(energy):
        raise ValueError("nonfinite current-energy proxy")
    return {
        "shot": shot,
        "initial_ip_A": current,
        "current_energy_proxy": energy,
        "proxy": energy,
        "prefix_samples": n,
        "actual_rate_hz": rate,
        "signal_start_time": start,
    }


def order_stream(ids, reader=read_prefix):
    """Order every supplied shot, or fail without exclusions or fallbacks.

    ``reader(shot)`` returns ``planning_ip``, ``rate``, and ``start``. Rate
    and start retain the original sample grid: ``start + index / rate``.
    """
    try:
        ids = list(ids)
    except TypeError as error:
        raise ValueError("distinct positive integer shot IDs required") from error
    if (not ids or any(type(shot) is not int or shot <= 0 for shot in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("distinct positive integer shot IDs required")
    rows = [_record(shot, reader(shot)) for shot in ids]
    return sorted(rows, key=lambda row: (row["proxy"], row["shot"]))
