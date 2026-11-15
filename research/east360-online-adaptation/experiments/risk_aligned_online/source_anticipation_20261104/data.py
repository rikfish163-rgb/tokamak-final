"""Annotate locked J-TEXT TRAIN/VAL episodes without changing archived labels.

The 125 ms label is an auxiliary hypothesis, not the original J-TEXT label.
The default reader opens only outcome metadata and /data/ip clock/shape.
"""

from collections import Counter
import hashlib
import json
import math
from numbers import Integral, Real

import h5py
import numpy as np

from .. import io23
from ..source_event_tasks_20261102 import data as source_archive


_FIELDS = frozenset({
    "hdf5_disruptive", "hdf5_down_time", "hdf5_query_time", "hdf5_lead_ms", "auxiliary_lead_ms",
    "source_horizon_ms", "hdf5_source_point_valid", "original_source_point_label",
    "future_disruption_125ms", "timely_eligible", "anticipation_stratum",
    "archived_mismatches", "any_archived_mismatch",
})
_MISMATCH_KEYS = ("disruptive", "label", "group", "lead_ms")
_LEGACY_TIME_TOLERANCE_MS = 1e-6


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return float(value)


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def annotate(record, *, disruptive, down, query_time, horizon_ms):
    """Copy one archived episode and append HDF5-derived outcome fields.

    Original ``label``, ``group`` and ``disruptive`` are provenance, never
    overwritten. The source label uses io23's 1e-6 ms boundary tolerance.
    Auxiliary boundaries use HDF5 lead rounded to 9 decimal places in ms to
    remove binary clock-subtraction noise; the unrounded lead is retained.
    The auxiliary interval is closed 0..125 ms; timely means strictly >30 ms.
    Post-terminal disruptive rows are ineligible.
    """
    if not isinstance(record, dict) or _FIELDS.intersection(record):
        raise ValueError("unannotated archived record without derived-field collisions required")
    if not {"label", "group", "disruptive", "lead_ms"}.issubset(record):
        raise ValueError("archived outcome provenance is incomplete")
    old_flag = _boolean(record["disruptive"], "archived disruptive")
    if isinstance(record["label"], (bool, np.bool_)) or record["label"] not in (0, 1):
        raise ValueError("archived point label must be numeric 0 or 1")
    if record["group"] not in ("safe", "early", "danger"):
        raise ValueError("unknown archived source group")
    flag = _boolean(disruptive, "HDF5 disruptive")
    query = _finite_number(query_time, "query_time")
    horizon = _finite_number(horizon_ms, "horizon_ms")
    if horizon < 25:
        raise ValueError("J-TEXT effective horizon must be at least 25 ms")

    down_time = _finite_number(down, "HDF5 disruption time") if flag else None
    lead = (down_time - query) * 1000 if flag else None
    if flag and not math.isfinite(lead):
        raise ValueError("nonfinite HDF5 lead")
    auxiliary_lead = round(lead, 9) if flag else None
    valid = not flag or lead >= -_LEGACY_TIME_TOLERANCE_MS
    point = int(flag and valid and lead <= horizon + _LEGACY_TIME_TOLERANCE_MS)
    future = int(flag and 0 <= auxiliary_lead <= 125)
    timely = not flag or auxiliary_lead > 30
    if not flag:
        stratum, expected_group = "safe", "safe"
    elif auxiliary_lead < 0:
        stratum = "post_disruption"
        expected_group = "danger" if point else "post_disruption"
    elif auxiliary_lead <= 30:
        stratum, expected_group = "disruptive_0_to_30ms", "danger" if point else "early"
    elif auxiliary_lead <= 125:
        stratum, expected_group = "disruptive_30_to_125ms", "danger" if point else "early"
    else:
        stratum, expected_group = "disruptive_earlier_than_125ms", "danger" if point else "early"

    old_lead = record["lead_ms"]
    lead_matches = old_lead is None if not flag else (
        isinstance(old_lead, Real) and not isinstance(old_lead, (bool, np.bool_))
        and math.isfinite(old_lead) and abs(float(old_lead) - lead) <= _LEGACY_TIME_TOLERANCE_MS
    )
    mismatches = dict(
        disruptive=old_flag != flag,
        label=record["label"] != point,
        group=record["group"] != expected_group,
        lead_ms=not lead_matches,
    )
    return dict(
        record,
        hdf5_disruptive=flag,
        hdf5_down_time=down_time,
        hdf5_query_time=query,
        hdf5_lead_ms=lead,
        auxiliary_lead_ms=auxiliary_lead,
        source_horizon_ms=horizon,
        hdf5_source_point_valid=valid,
        original_source_point_label=point,
        future_disruption_125ms=future,
        timely_eligible=timely,
        anticipation_stratum=stratum,
        archived_mismatches=mismatches,
        any_archived_mismatch=any(mismatches.values()),
    )


def _numeric_scalar(value, name):
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or array.dtype.kind not in "biuf" or not np.isfinite(array[0]):
        raise ValueError(f"HDF5 {name} must be one finite numeric scalar")
    return float(array[0])


def _read_metadata(shot, *, allowed_ids):
    """Internal J-TEXT reader; caller first validates locked role membership."""
    if shot not in allowed_ids:
        raise ValueError("shot is not in the locked source role")
    with h5py.File(io23.path_for("J-TEXT", shot), "r") as handle:
        flag = _numeric_scalar(handle["meta/IsDisrupt"][()], "IsDisrupt")
        raw_down = handle["meta/DownTime"][()]
        ip = handle["data/ip"]
        if ip.ndim != 1:
            raise ValueError("HDF5 ip must be one-dimensional")
        samples = ip.shape[0]
        start = _numeric_scalar(ip.attrs["StartTime"], "ip StartTime")
        rate = _numeric_scalar(ip.attrs["SampleRate"], "ip SampleRate")
    if flag not in (0, 1):
        raise ValueError("HDF5 IsDisrupt must be binary")
    down = _numeric_scalar(raw_down, "DownTime") if flag else None
    return dict(disruptive=bool(flag), down=down, start_time=start, sample_rate_hz=rate, samples=samples)


def _metadata(reader, shot):
    value = reader(shot)
    if not isinstance(value, dict) or not {"disruptive", "down", "start_time", "sample_rate_hz", "samples"} <= value.keys():
        raise ValueError("complete HDF5 outcome and ip clock metadata required")
    flag = _boolean(value["disruptive"], "HDF5 disruptive")
    down = _finite_number(value["down"], "HDF5 disruption time") if flag else None
    start = _finite_number(value["start_time"], "ip StartTime")
    rate = _finite_number(value["sample_rate_hz"], "ip SampleRate")
    samples = value["samples"]
    if (not math.isclose(rate, 1000.0, rel_tol=0, abs_tol=io23.RATE_ATOL_HZ)
            or isinstance(samples, (bool, np.bool_)) or not isinstance(samples, Integral) or samples < 1):
        raise ValueError("strict 1 kHz source clock and positive ip length required")
    return dict(disruptive=flag, down=down, start_time=start, sample_rate_hz=rate, samples=int(samples))


def _validate_records(rows, ids, role):
    allowed = set(ids)
    if len(allowed) != len(ids) or not allowed:
        raise ValueError("distinct nonempty locked source IDs required")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("role") != role or row.get("device") != "J-TEXT":
            raise ValueError("record outside requested J-TEXT role")
        if any(isinstance(row.get(k), (bool, np.bool_)) or not isinstance(row.get(k), Integral)
               for k in ("shot", "origin", "maturity", "query")):
            raise ValueError("integer shot and clock indices required")
        sid, origin, maturity, query = (row[k] for k in ("shot", "origin", "maturity", "query"))
        if (sid not in allowed or origin < 127 or (origin - 127) % 25
                or maturity != origin + 25 or not origin + 26 <= query <= origin + 50
                or row.get("feedback_observed") is not True):
            raise ValueError("locked role or 25-point maturity/query boundary violated")
        if (sid, query) in seen:
            raise ValueError("duplicate shot/query episode")
        seen.add((sid, query))


def build_records(role, *, metadata_reader=None):
    """Return (annotated rows, receipt), reading only this locked role's shots.

    ``metadata_reader(shot)`` can be injected for tests; its mapping has
    disruptive/down/start_time/sample_rate_hz/samples. No signal arrays, event
    CSVs, model files or TEST/EAST HDF5 data are consulted here. The current
    canonical AdvancedTime mapping is always read rather than overridden.
    """
    if role not in ("train", "val"):
        raise ValueError("only locked J-TEXT train or val roles are permitted")
    rows, ids, provenance = source_archive.load_records(role)
    _validate_records(rows, ids, role)  # check every identity before any raw read
    advanced = io23.advanced_times()
    allowed = set(ids)
    reader = (lambda sid: _read_metadata(sid, allowed_ids=allowed)) if metadata_reader is None else metadata_reader
    metadata, result = {}, []
    for row in rows:
        sid, query = int(row["shot"]), int(row["query"])
        if sid not in metadata:
            metadata[sid] = _metadata(reader, sid)
        info = metadata[sid]
        if query >= info["samples"]:
            raise ValueError(f"query outside HDF5 ip clock: {sid}")
        horizon = io23.effective_horizon_ms(sid, advanced)[0]
        result.append(annotate(row, disruptive=info["disruptive"], down=info["down"],
                               query_time=info["start_time"] + query / info["sample_rate_hz"],
                               horizon_ms=horizon))
    mismatch_counts = Counter(key for row in result for key in _MISMATCH_KEYS if row["archived_mismatches"][key])
    fields = [dict(shot=sid, **metadata[sid]) for sid in sorted(metadata)]
    metadata_sha = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode("utf-8")).hexdigest()
    receipt = dict(role=role, records=len(result), shots=len(metadata), metadata_shots_read=len(metadata),
                   metadata_source="hdf5" if metadata_reader is None else "injected",
                   records_with_mismatch=sum(row["any_archived_mismatch"] for row in result),
                   mismatch_counts={key: mismatch_counts[key] for key in _MISMATCH_KEYS},
                   source_metadata_fields_sha256=metadata_sha,
                   source_sha256=provenance.get("source_sha256"),
                   archive_artifact_sha256=provenance.get("artifact_sha256", {}))
    return result, receipt
