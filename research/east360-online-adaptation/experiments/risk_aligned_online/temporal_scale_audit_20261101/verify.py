#!/usr/bin/env python3
"""CPU-only, bounded independent verification of temporal_scale_audit_v1.

Only canonical TRAIN and the first 20 reference-order IDs may be opened as
HDF5.  Derived metrics and the shot-equal aggregation are implemented here;
no audit core, window-feature or summarize implementation is imported.
The original flat23 I/O, EAST mask and affine-map specifications are reused
as *contracts*, but their functions are not called.  The trusted, hash-locked
source checkpoint is opened on CPU solely to check its scaler.
"""

from __future__ import annotations

import collections
import hashlib
import json
import mmap
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import warnings
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "probe_outputs/temporal_scale_audit_v1"
REF = ROOT / "probe_outputs/stratified_cold20_context360_v1"
SOURCE = ROOT / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
INDEX = ROOT / "artifacts/canonical/jtext_source_bundle_p131_v1/index.json"
ADVANCED = ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"
RESULT_PATH = OUT / "verification.json"
SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
REFERENCE_SHA = "ef6e2fccd56f7a8b7bc0f93c44315cf77028ee6e122d74758e5f0678baa937ef"
RTOL, ATOL = 1e-10, 0.0  # Predeclared; never changed in response to results.
MODES = ("fixed128", "record_windows", "timely_positive")
LAGS = (1, 2, 4, 8, 16, 32, 64)
FIELDS = ("raw_variance", "asinh_std", "patch_mean_retention", "observed",
          "complete_patches", *(f"D{i}" for i in LAGS))
CHANNELS = ("Bt", "CIII", "MNM", "Mir_VV", "Mir_avg_amp", "Mir_avg_fre",
            "Z_proxy", "dx_a", "dy_a", "ip", "mode_number_n", "n=1 amplitude",
            "ne_nG", "qa_proxy", "sxr_kurt", "sxr_mean", "sxr_skew", "sxr_var",
            "v_loop", "xuv_kurt", "xuv_ratio", "xuv_skew", "xuv_var")

started = time.monotonic()
result = {
    "verdict": "INCOMPLETE",
    "numerical_verdict": "INCOMPLETE",
    "tolerance": {"dtype": "float64", "rtol": RTOL, "atol": ATOL,
                  "equal_nan": True, "counts_indices_hashes": "exact"},
    "scope": "J-TEXT canonical TRAIN1353 and first 20 ended EAST shots only",
    "checks": {},
    "limitations": [
        "Only sampled window features are recomputed from primary signals; all stored summary fields are recomputed from NPZ rows.",
        "The reference completion is byte-hashed, not semantically parsed; its digest cannot prove historical access discipline.",
        "The completed manifest does not itself anchor the protocol or archive digest; archive members are checked against protocol hashes and current files.",
        "The source signal contract and the EAST sentinel/affine-map definitions are reused as specifications, although no original I/O, feature, or summary functions are imported.",
        "A verification result cannot establish physical time-scale validity, causal benefit, or the historical absence of all unauthorized reads.",
    ],
    "scope_incident": (
        "Before this verifier ran, the operator's diagnostic printed aggregate target-suffix "
        "performance/outcome counts from reference completion metadata. No individual "
        "target-suffix HDF5 signals or outcomes were opened. This aggregate exposure "
        "breached the requested suffix-outcome boundary."
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save():
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    # Serialize before opening the result so a reporting error cannot truncate
    # the most recent persisted checkpoint. NumPy scalar conversion is lossless.
    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False,
                            default=json_scalar) + "\n"
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())


def check(name, ok, **details):
    result["checks"][name] = {"pass": bool(ok), **details}
    if not ok:
        print("FAIL", name, flush=True)


def bounded(items, limit=15):
    return list(items)[:limit]


def canonical_train():
    """Extract only the TRAIN ID array, not other manifest records/outcomes."""
    with INDEX.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        start = mm.rfind(b'"split_shot_ids"')
        if start < 0:
            raise ValueError("canonical split_shot_ids missing")
        key = mm.find(b'"train"', start)
        left = mm.find(b"[", key)
        right = mm.find(b"]", left)
        if min(key, left, right) < 0:
            raise ValueError("canonical TRAIN array missing")
        return [int(x) for x in json.loads(mm[left:right + 1])]


def first_array_objects(path, count):
    """Incrementally decode exactly the prefix, without parsing suffix records."""
    decoder = json.JSONDecoder()
    objects = []
    with Path(path).open(encoding="utf-8") as f:
        if f.read(1) != "[":
            raise ValueError("order must begin with an array")
        for _ in range(count):
            ch = f.read(1)
            while ch and (ch.isspace() or ch == ","):
                ch = f.read(1)
            if ch != "{":
                raise ValueError("order prefix has too few objects")
            buf = ch
            while True:
                try:
                    item, end = decoder.raw_decode(buf)
                    if end == len(buf):
                        objects.append(item)
                        break
                except json.JSONDecodeError:
                    pass
                ch = f.read(1)
                if not ch:
                    raise ValueError("truncated order prefix")
                buf += ch
    return objects


def selected_numeric_values(path, ids):
    """Access AdvancedTime values only for the explicitly selected TRAIN IDs."""
    allowed = {str(int(s)) for s in ids}
    found = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            key, separator, value = line.partition(":")
            key = key.strip().strip('"')
            if separator and key in allowed:
                found[int(key)] = float(value.strip().rstrip(","))
    return found


def selected_reference_hashes(path, ids):
    """Extract only requested first-20 raw-input hashes from the reference."""
    allowed = {str(int(s)) for s in ids}
    found = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            key, separator, value = line.partition(":")
            key = key.strip().strip('"')
            if separator and key in allowed:
                found[int(key)] = json.loads(value.strip().rstrip(","))
                if len(found) == len(allowed):
                    break
    return found


def source_path(device, sid, allowed):
    sid = int(sid)
    if device not in allowed or sid not in allowed[device]:
        raise PermissionError(f"refusing unauthorized HDF5 access: {device} {sid}")
    return ROOT / "data" / device / "unified_hdf5" / str(sid // 100 * 100) / f"{sid}.hdf5"


def hdf5_label(device, sid, allowed):
    with h5py.File(source_path(device, sid, allowed), "r") as f:
        value = np.asarray(f["meta/IsDisrupt"][()]).reshape(-1)
        if len(value) != 1 or value[0] not in (0, 1):
            raise ValueError(f"invalid outcome field for {device} {sid}")
        return bool(value[0])


def raw_shot(device, sid, allowed):
    """Independent flat23 reader implementing the existing input contract."""
    with h5py.File(source_path(device, sid, allowed), "r") as f:
        cols = [np.asarray(f["data"][name][:], dtype=np.float32) for name in CHANNELS]
        ds = f["data/ip"]
        rate = float(np.asarray(ds.attrs["SampleRate"]).reshape(-1)[0])
        start = float(np.asarray(ds.attrs["StartTime"]).reshape(-1)[0])
        disruptive = bool(np.asarray(f["meta/IsDisrupt"][()]).reshape(-1)[0])
        down = (float(np.asarray(f["meta/DownTime"][()]).reshape(-1)[0])
                if disruptive else None)
        if len(set(map(len, cols))) != 1:
            raise ValueError(f"unaligned channels for {device} {sid}")
        for name in CHANNELS:
            channel = f["data"][name]
            for attr, expected in (("SampleRate", rate), ("StartTime", start)):
                actual = float(np.asarray(channel.attrs[attr]).reshape(-1)[0])
                if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL):
                    raise ValueError(f"unaligned {attr}: {device} {sid} {name}")
    data = np.stack(cols, axis=1)
    mask = np.isfinite(data)
    data = np.where(mask, data, np.float32(0)).astype(np.float32)
    if device == "J-TEXT":
        data[:, 9] *= np.float32(1000.0)
    else:
        mask[:, 2] = False
        mask[:, 1] &= ~np.isclose(data[:, 1], -10.0, rtol=0.0, atol=0.01)
    return {"values": data, "mask": mask, "rate": rate,
            "times": start + np.arange(len(data), dtype=np.float64) / rate,
            "disruptive": disruptive, "down": down}


def map_coordinates(raw, mask, mapping, device):
    mapped = raw.copy()
    if device == "EAST":
        for c, (gain, offset) in enumerate(zip(mapping["gain"], mapping["offset"])):
            if gain != 1.0 or offset != 0.0:
                selected = mask[:, c]
                mapped[selected, c] = (raw[selected, c].astype(np.float64) * gain + offset).astype(np.float32)
    return mapped


def independent_features(raw, mapped, mask, center, scale):
    """Build D(lag) from explicit index pairsets; compute complete patch ANOVA."""
    features = np.full((23, len(FIELDS)), np.nan, dtype=np.float64)
    pair_counts = np.zeros((23, len(LAGS)), dtype=np.uint16)
    statuses = collections.Counter()
    for c in range(23):
        valid = mask[:, c] & np.isfinite(raw[:, c])
        loc = np.flatnonzero(valid)
        raw_observed = raw[loc, c].astype(np.float64)
        features[c, 3] = len(loc)
        variance = np.nan
        if len(loc):
            mean = raw_observed.mean(dtype=np.float64)
            variance = np.mean((raw_observed - mean) ** 2, dtype=np.float64)
            features[c, 0] = variance
            standardized = np.arcsinh((mapped[loc, c].astype(np.float64) - center[c]) / scale[c])
            features[c, 1] = np.std(standardized, dtype=np.float64)
        good_patches = [p for p in range(16) if valid[8 * p:8 * p + 8].all()]
        features[c, 4] = len(good_patches)
        if len(good_patches) < 8:
            statuses["patch_insufficient_complete"] += 1
        else:
            blocks = np.stack([np.arcsinh((mapped[8*p:8*p+8, c].astype(np.float64)
                                           - center[c]) / scale[c]) for p in good_patches])
            block_means = blocks.mean(axis=1)
            between = np.mean((block_means - block_means.mean()) ** 2)
            within = np.mean((blocks - block_means[:, None]) ** 2)
            total = between + within
            if total > 0:
                features[c, 2] = between / total
                statuses["patch_defined"] += 1
            else:
                statuses["patch_constant_undefined"] += 1
        for j, lag in enumerate(LAGS):
            right = np.intersect1d(loc, loc + lag, assume_unique=True)
            pair_counts[c, j] = len(right)
            if len(loc) < 64:
                statuses["D_insufficient_observations"] += 1
            elif variance == 0:
                statuses["D_constant_undefined"] += 1
            elif len(right) < 32:
                statuses["D_insufficient_pairs"] += 1
            else:
                delta = raw[right, c].astype(np.float64) - raw[right - lag, c].astype(np.float64)
                features[c, 5 + j] = np.mean(delta ** 2, dtype=np.float64) / (2 * variance)
                statuses["D_defined"] += 1
    return features, pair_counts, statuses


def quantile_linear(values, q):
    values = np.sort(np.asarray(values, dtype=np.float64))
    point = (len(values) - 1) * q
    lo, hi = int(np.floor(point)), int(np.ceil(point))
    return float(values[lo] + (values[hi] - values[lo]) * (point - lo))


def recompute_summary(arrays):
    """First average windows within a shot, then give every shot equal weight."""
    out = {}
    for mode, name in enumerate(MODES):
        out[name] = {}
        for group, label in (("ALL", None), ("SAFE", False), ("DISRUPT", True)):
            use = arrays["mode"] == mode
            if label is not None:
                use &= arrays["disruptive"] == label
            ids = np.unique(arrays["shot"][use])
            means = []
            for sid in ids:
                x = arrays["features"][use & (arrays["shot"] == sid)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    means.append(np.nanmean(x, axis=0, dtype=np.float64))
            means = np.asarray(means, dtype=np.float64).reshape((-1, 23, len(FIELDS)))
            channels = {}
            for c, channel in enumerate(CHANNELS):
                metrics = {}
                for k, field in enumerate(FIELDS):
                    v = means[:, c, k]
                    v = v[np.isfinite(v)]
                    metrics[field] = {"n_shots": int(len(v)),
                                      "q10": quantile_linear(v, .1) if len(v) else None,
                                      "median": quantile_linear(v, .5) if len(v) else None,
                                      "q90": quantile_linear(v, .9) if len(v) else None}
                channels[channel] = metrics
            out[name][group] = {"n_shots": len(ids), "n_windows": int(use.sum()),
                                "channels": channels}
    return out


def compare_tree(expected, observed, path="", mismatches=None):
    if mismatches is None:
        mismatches = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            mismatches.append({"path": path, "error": "dictionary keys differ"})
        else:
            for k in expected:
                compare_tree(expected[k], observed[k], f"{path}/{k}", mismatches)
    elif expected is None:
        if observed is not None:
            mismatches.append({"path": path, "expected": None, "actual": observed})
    elif isinstance(expected, (int, bool)):
        if type(expected) != type(observed) or expected != observed:
            mismatches.append({"path": path, "expected": expected, "actual": observed})
    else:
        if observed is None or not np.isclose(float(expected), float(observed),
                                              rtol=RTOL, atol=ATOL, equal_nan=True):
            mismatches.append({"path": path, "expected": expected, "actual": observed})
    return mismatches


def expected_endpoints(shot, horizon, device):
    n = len(shot["times"])
    if shot["disruptive"]:
        lead_unrounded = (shot["down"] - shot["times"]) * 1000
        valid = lead_unrounded >= -1e-6
        labels = (lead_unrounded <= horizon + 1e-6) & valid
        lead = np.round(lead_unrounded, 9)
    else:
        valid = np.ones(n, dtype=bool)
        labels = np.zeros(n, dtype=bool)
        lead = np.full(n, np.nan)
    eligible = np.flatnonzero(valid & (np.arange(n) >= 127))
    def choose(indices, cap):
        if not len(indices):
            return []
        return indices[np.linspace(0, len(indices)-1, min(cap, len(indices)), dtype=int)].tolist()
    return {
        "fixed128": [127] if n > 127 and valid[127] else [],
        "record_windows": choose(eligible, 16),
        "timely_positive": choose(np.flatnonzero(valid & labels & (lead > (10 if device == "J-TEXT" else 30))
                                                  & (np.arange(n) >= 127)), 8),
    }


def run():
    # The one explicitly requested recovery preserves the exact damaged result
    # fragment and the observed reporting failure, with identical audit criteria.
    # Otherwise a prior result is never overwritten.
    if sys.argv[1:] == ["--resume-after-report-error"]:
        fragment_hash = sha256(RESULT_PATH)
        if fragment_hash != "700525a148479d6bf87542183df2f00b54800d08ad55e705f4179009ec49fc96":
            raise FileExistsError("existing result is not the known failed-report fragment")
        result["prior_attempts"] = [{
            "verdict": "ERROR",
            "exit_code": 2,
            "error": "TypeError: Object of type bool is not JSON serializable (NumPy bool_)",
            "failure_result_persistence": "The original error handler hit the same serialization error and truncated its JSON; the surviving bytes are preserved below.",
            "partial_result_sha256": fragment_hash,
            "partial_result_utf8": RESULT_PATH.read_text(encoding="utf-8"),
            "verifier_sha256": "9031598a7239e393389e8cbec258d703b53c065e7e2578bee5567ffbcc44c14e",
            "criteria_changed": False,
        }]
    elif not sys.argv[1:]:
        with RESULT_PATH.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    else:
        raise ValueError("unsupported arguments")
    save()
    protocol = read_json(OUT / "protocol.json")
    completed = read_json(OUT / "completed.json")
    inventory = read_json(OUT / "inventory.json")
    summary = read_json(OUT / "summary.json")
    train = canonical_train()
    target = [int(x["shot"]) for x in first_array_objects(REF / "order.json", 20)]
    allowed = {"J-TEXT": set(train), "EAST": set(target)}
    expected_flags = {
        "scope": "SOURCE_TRAIN_AND_ENDED_TARGET20_DESCRIPTIVE_TEMPORAL_AUDIT",
        "target_suffix_signals_read": 0, "target_suffix_outcomes_read": 0,
        "source_holdout_signals_read": 0, "no_training": True, "no_gpu": True,
        "label_authority": "HDF5", "time_axis_changed": False,
        "physical_time_scale_certified": False,
    }
    flag_errors = {k: {"expected": v, "actual": protocol.get(k)}
                   for k, v in expected_flags.items() if protocol.get(k) != v}
    check("protocol_contract", not flag_errors and protocol["modes"] == list(MODES)
          and protocol["fields"] == list(FIELDS) and protocol["lags"] == list(LAGS),
          flag_errors=flag_errors)
    membership_ok = (len(train) == len(set(train)) == 1353 and len(target) == len(set(target)) == 20
                     and protocol["source_ids"] == train and protocol["target_ids"] == target
                     and len(inventory) == 1373
                     and [(r["device"], r["shot"]) for r in inventory] ==
                     [("J-TEXT", i) for i in train] + [("EAST", i) for i in target])
    check("exact_authorized_membership_and_order", membership_ok,
          source_count=len(train), target_count=len(target), target_ids=target,
          source_first_middle_last=[train[0], train[len(train)//2], train[-1]])
    if not membership_ok:
        raise RuntimeError("membership mismatch: no HDF5 read is authorized")
    by_key = {(r["device"], r["shot"]): r for r in inventory}
    mapping = read_json(REF / "input_map.json")
    map_ok = (mapping["calibration_ids"] == target
              and mapping["source_statistics"]["shot_ids"] == train
              and mapping["target_statistics"]["shot_ids"] == target
              and len(mapping["gain"]) == len(mapping["offset"]) == 23
              and np.isfinite(mapping["gain"]).all() and np.isfinite(mapping["offset"]).all()
              and (np.asarray(mapping["gain"]) > 0).all())
    check("mapping_membership", map_ok)

    hashes = {name: sha256(OUT / name) for name in
              ("protocol.json", "code_snapshot.zip", "completed.json", "inventory.json",
               "summary.json", "source_windows.npz", "target_windows.npz")}
    hashes.update({"reference_completed.json": sha256(REF / "completed.json"),
                   "input_map.json": sha256(REF / "input_map.json"),
                   "source_checkpoint": sha256(SOURCE), "verifier.py": sha256(__file__)})
    result["sha256"] = hashes
    hash_errors = []
    for filename, key in (("inventory.json", "inventory_sha256"),
                          ("summary.json", "summary_sha256"),
                          ("source_windows.npz", "source_windows_sha256"),
                          ("target_windows.npz", "target_windows_sha256")):
        if hashes[filename] != completed.get(key):
            hash_errors.append(filename)
    if hashes["reference_completed.json"] != REFERENCE_SHA or protocol["reference_completed_sha256"] != REFERENCE_SHA:
        hash_errors.append("reference_completed")
    if hashes["source_checkpoint"] != SOURCE_SHA or protocol["source_sha256"] != SOURCE_SHA:
        hash_errors.append("locked_source_checkpoint")
    if hashes["input_map.json"] != protocol["input_map_sha256"]:
        hash_errors.append("input_map")
    if protocol["reference"] != str(REF) or protocol["source_checkpoint"] != str(SOURCE):
        hash_errors.append("declared_paths")
    check("completion_protocol_input_hashes", not hash_errors, mismatches=hash_errors,
          completion_status=completed.get("status"))

    code_errors = []
    code = protocol["code_sha256"]
    with zipfile.ZipFile(OUT / "code_snapshot.zip") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(code):
            code_errors.append("archive member list differs from protocol")
        for path, digest in code.items():
            try:
                zipped = hashlib.sha256(archive.read(path)).hexdigest()
                live = sha256(ROOT / path)
                if zipped != digest or live != digest:
                    code_errors.append(path)
            except (KeyError, OSError):
                code_errors.append(path)
    check("code_archive_protocol_live_hashes", not code_errors,
          member_count=len(code), mismatches=bounded(code_errors), n_mismatches=len(code_errors))
    if hashes["source_checkpoint"] != SOURCE_SHA:
        raise RuntimeError("untrusted checkpoint: refusing deserialization")
    import torch  # Imported only after the source hash is checked; CPU map only.
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    center = np.asarray(checkpoint["source_center"], dtype=np.float64)
    scale = np.asarray(checkpoint["source_scale"], dtype=np.float64)
    scaler_ok = (center.shape == scale.shape == (23,) and np.isfinite(center).all()
                 and np.isfinite(scale).all() and (scale > 0).all()
                 and np.array_equal(center, np.asarray(protocol["center"], dtype=np.float64))
                 and np.array_equal(scale, np.asarray(protocol["scale"], dtype=np.float64))
                 and np.array_equal(center, checkpoint["model"]["center"].numpy().astype(np.float64))
                 and np.array_equal(scale, checkpoint["model"]["scale"].numpy().astype(np.float64)))
    check("sourcefixed_checkpoint_scaler", scaler_ok, shape=list(center.shape))
    del checkpoint

    seal_errors, seals = [], []
    for i, sid in enumerate(target):
        prefix = REF / "shots" / f"{i+1:04d}_{sid}"
        seal = read_json(Path(str(prefix) + "_seal.json"))
        actual = sha256(prefix.with_suffix(".npz"))
        ok = (seal.get("shot") == sid and seal.get("index") == i+1
              and seal.get("outcome_available") is False and seal.get("sha256") == actual)
        seals.append({"index": i+1, "shot": sid, "sha256": actual, "pass": ok})
        if not ok:
            seal_errors.append({"index": i+1, "shot": sid})
    check("actual_first20_prediction_seals", not seal_errors, count=len(seals),
          seals=seals, mismatches=seal_errors)
    save()
    print("Provenance and seals checked; reading authorized HDF5 metadata and hashes", flush=True)

    target_input_hashes = selected_reference_hashes(REF / "input_hashes.json", target)
    check("reference_first20_input_hash_inventory", len(target_input_hashes) == 20 and
          all(target_input_hashes.get(sid) == by_key[("EAST", sid)]["sha256"] for sid in target),
          target_hash_count=len(target_input_hashes))
    class_counts = {d: {"False": 0, "True": 0} for d in allowed}
    raw_errors = []
    for device, ids in (("J-TEXT", train), ("EAST", target)):
        for sid in ids:
            path = source_path(device, sid, allowed)
            observed_hash = sha256(path)
            label = hdf5_label(device, sid, allowed)
            row = by_key[(device, sid)]
            class_counts[device][str(label)] += 1
            if observed_hash != row["sha256"] or label is not row["disruptive"]:
                raw_errors.append({"device": device, "shot": sid,
                                   "sha_matches": observed_hash == row["sha256"],
                                   "hdf5_disruptive": label, "inventory_disruptive": row["disruptive"]})
    completion_counts_ok = (completed["source_class_counts"] == class_counts["J-TEXT"]
                            and completed["target_class_counts"] == class_counts["EAST"]
                            and completed["source_shots"] == len(train)
                            and completed["target_shots"] == len(target)
                            and completed["status"] == "COMPLETED_DESCRIPTIVE")
    check("all_hdf5_hashes_and_outcome_classes", not raw_errors and completion_counts_ok,
          checked_shots=sum(map(len, allowed.values())), class_counts=class_counts,
          inventory_mismatches=bounded(raw_errors), n_mismatches=len(raw_errors),
          completion_counts_match=completion_counts_ok)

    cadence = {}
    for device in allowed:
        samples = np.asarray([r["dt_ms"] for r in inventory if r["device"] == device], dtype=np.float64)
        rate_tol = 1e-9 if device == "J-TEXT" else .5
        lower, upper = 1000 / (1000 + rate_tol), 1000 / (1000 - rate_tol)
        ok = np.isfinite(samples) & ((samples >= lower) | np.isclose(samples, lower, rtol=RTOL, atol=ATOL))
        ok &= ((samples <= upper) | np.isclose(samples, upper, rtol=RTOL, atol=ATOL))
        cadence[device] = {"count": len(samples), "min_ms": float(samples.min()),
                           "max_ms": float(samples.max()), "lower_ms": lower, "upper_ms": upper,
                           "outside": int((~ok).sum()), "nominal_rate_tolerance_hz": rate_tol}
    check("inventory_1khz_cadence_bounds", all(v["outside"] == 0 for v in cadence.values()),
          devices=cadence)
    save()

    arrays = {}
    summary_comparisons = {}
    for device, filename in (("J-TEXT", "source_windows.npz"), ("EAST", "target_windows.npz")):
        with np.load(OUT / filename, allow_pickle=False) as data:
            arrays[device] = {k: data[k] for k in data.files}
        a = arrays[device]
        n = len(a["shot"])
        shape_ok = (set(a) == {"shot", "mode", "end", "disruptive", "features", "pair_counts"}
                    and a["features"].shape == (n, 23, len(FIELDS))
                    and a["pair_counts"].shape == (n, 23, len(LAGS))
                    and all(a[k].shape == (n,) for k in ("shot", "mode", "end", "disruptive"))
                    and all(np.issubdtype(a[k].dtype, np.integer) for k in ("shot", "mode", "end"))
                    and a["features"].dtype == np.float64 and a["pair_counts"].dtype == np.uint16
                    and a["disruptive"].dtype == np.bool_)
        check(f"{device}_npz_schema", shape_ok, rows=n)
        if not shape_ok:
            raise RuntimeError(f"invalid NPZ shape for {device}")
        row_errors = []
        seen = set()
        counts = collections.Counter()
        for sid, mode, end, disrupt in zip(a["shot"], a["mode"], a["end"], a["disruptive"]):
            sid, mode, end, disrupt = int(sid), int(mode), int(end), bool(disrupt)
            key = (sid, mode, end)
            row = by_key.get((device, sid))
            bad = (row is None or mode not in range(3) or key in seen)
            if not bad:
                bad = (end < 127 or end >= row["n_points"] or disrupt != row["disruptive"]
                       or (mode == 0 and end != 127) or (mode == 2 and not disrupt))
            if bad:
                row_errors.append({"shot": sid, "mode": mode, "end": end})
            seen.add(key)
            counts[(sid, mode)] += 1
        for row in inventory:
            if row["device"] != device:
                continue
            for mode, name in enumerate(MODES):
                if counts[(row["shot"], mode)] != row["eligible"][name]:
                    row_errors.append({"shot": row["shot"], "mode": name,
                                       "stored": counts[(row["shot"], mode)],
                                       "inventory": row["eligible"][name]})
        mode_counts = {name: int((a["mode"] == i).sum()) for i, name in enumerate(MODES)}
        check(f"{device}_npz_rows_endpoints_membership", not row_errors,
              rows=n, modes=mode_counts, n_errors=len(row_errors), errors=bounded(row_errors))

        features, pairs = a["features"], a["pair_counts"]
        observed, patches, variance = features[:, :, 3], features[:, :, 4], features[:, :, 0]
        count_ok = (np.isfinite(observed).all() and np.isfinite(patches).all()
                    and (observed == np.floor(observed)).all() and (patches == np.floor(patches)).all()
                    and ((0 <= observed) & (observed <= 128)).all()
                    and ((0 <= patches) & (patches <= 16)).all()
                    and (pairs <= observed[:, :, None]).all()
                    and all((pairs[:, :, i] <= 128-lag).all() for i, lag in enumerate(LAGS)))
        needed = (observed[:, :, None] >= 64) & (variance[:, :, None] > 0) & (pairs >= 32)
        d = features[:, :, 5:]
        patch = features[:, :, 2]
        status_ok = (not np.isinf(features).any()
                     and np.array_equal(np.isfinite(features[:, :, 0]), observed > 0)
                     and np.array_equal(np.isfinite(features[:, :, 1]), observed > 0)
                     and np.array_equal(np.isfinite(d), needed)
                     and not np.isfinite(patch[patches < 8]).any()
                     and not np.isfinite(patch[(variance == 0) & (observed > 0)]).any()
                     and ((patch[np.isfinite(patch)] >= 0) & (patch[np.isfinite(patch)] <= 1)).all())
        check(f"{device}_global_defined_constant_status", count_ok and status_ok,
              count_bounds_pass=count_ok, status_rules_pass=status_ok,
              d_defined=int(np.isfinite(d).sum()), d_undefined=int(np.isnan(d).sum()),
              zero_variance_channels=int(((variance == 0) & (observed > 0)).sum()),
              patch_defined=int(np.isfinite(patch).sum()), patch_undefined=int(np.isnan(patch).sum()))

        independently_computed = recompute_summary(a)
        errors = compare_tree(independently_computed, summary[device], device)
        summary_comparisons[device] = {"mismatches": len(errors), "examples": bounded(errors)}
        check(f"{device}_all_summary_fields_equalshot", not errors,
              mode_group_channel_field_combinations=3*3*23*len(FIELDS),
              compared_metric_values=3*3*23*len(FIELDS)*4,
              mismatches=len(errors), examples=bounded(errors))
        save()
        print("Checked NPZ rows, statuses and complete summary:", device, n, flush=True)

    # Synthetic boundary cases verify explicit missing, constant, pair and patch rules.
    c0, s0 = np.zeros(23), np.ones(23)
    sample = np.broadcast_to(np.arange(128, dtype=np.float32)[:, None], (128, 23)).copy()
    masks = np.ones_like(sample, dtype=bool)
    sample[:, 0] = 5.0             # Constant with 16 complete patches.
    masks[:, 1] = False            # All missing.
    masks[63:, 2] = False          # 63 observed: D undefined.
    masks[1::2, 3] = False         # 64 observed, no lag-1 pairs.
    masks[56:, 4] = False          # Seven complete patches.
    masks[64:, 5] = False          # Eight complete patches.
    f, p, _ = independent_features(sample, sample, masks, c0, s0)
    synthetic_ok = (np.isnan(f[0, 5:]).all() and np.isnan(f[0, 2]) and f[0, 0] == 0
                    and f[1, 3] == 0 and np.isnan(f[1, :3]).all()
                    and f[2, 3] == 63 and np.isnan(f[2, 5:]).all()
                    and f[3, 3] == 64 and p[3, 0] == 0 and np.isnan(f[3, 5])
                    and p[3, 1] >= 32 and np.isfinite(f[3, 6])
                    and f[4, 4] == 7 and np.isnan(f[4, 2])
                    and f[5, 4] == 8 and np.isfinite(f[5, 2])
                    and np.isclose(quantile_linear([1, 2, 3], .1), 1.2, rtol=RTOL, atol=ATOL))
    check("synthetic_undefined_constant_and_threshold_boundaries", synthetic_ok)

    src_rows = [by_key[("J-TEXT", sid)] for sid in train]
    src_indices = [0, len(train)//2, len(train)-1]
    for row in src_rows:
        if row["disruptive"] and row["eligible"]["timely_positive"] and row["shot"] not in [train[i] for i in src_indices]:
            src_indices.append(train.index(row["shot"]))
            if len(src_indices) == 5:
                break
    selected = [("J-TEXT", train[i], i) for i in src_indices] + [("EAST", target[i], i) for i in (0, 9, 19)]
    selection_ok = (len(selected) >= 8 and len({(d, s) for d, s, _ in selected}) == len(selected)
                    and {by_key[(d, s)]["disruptive"] for d, s, _ in selected if d == "J-TEXT"} == {False, True})
    check("spot_selection", selection_ok, shots=[{"device": d, "shot": s, "order_index": i,
          "disruptive": by_key[(d, s)]["disruptive"]} for d, s, i in selected])
    advance = selected_numeric_values(ADVANCED, [s for d, s, _ in selected if d == "J-TEXT"])
    spot_errors, spot_records = [], []
    status_total = collections.Counter()
    sampled_mode_counts = collections.Counter()
    for device, sid, index in selected:
        shot = raw_shot(device, sid, allowed)
        row = by_key[(device, sid)]
        dt = float(np.median(np.diff(shot["times"])) * 1000)
        if (len(shot["values"]) != row["n_points"] or shot["disruptive"] is not row["disruptive"]
                or not np.isclose(dt, row["dt_ms"], rtol=RTOL, atol=ATOL)):
            spot_errors.append({"device": device, "shot": sid, "error": "HDF5 identity/count/cadence"})
        horizon = max(25., advance.get(sid, 25.)) if device == "J-TEXT" else 125.
        views = expected_endpoints(shot, horizon, device)
        mapped = map_coordinates(shot["values"], shot["mask"], mapping, device)
        a = arrays[device]
        sampled = {}
        for mode, name in enumerate(MODES):
            positions = np.flatnonzero((a["shot"] == sid) & (a["mode"] == mode))
            ends = a["end"][positions].tolist()
            if ends != views[name]:
                spot_errors.append({"device": device, "shot": sid, "mode": name,
                                    "error": "endpoint list", "actual": ends, "expected": views[name]})
            chosen = sorted(set([0, len(positions)//2, len(positions)-1])) if len(positions) else []
            sampled[name] = []
            for k in chosen:
                pos = int(positions[k])
                end = int(a["end"][pos])
                span = slice(end-127, end+1)
                computed, counts, statuses = independent_features(
                    shot["values"][span], mapped[span], shot["mask"][span], center, scale)
                matches = np.isclose(computed, a["features"][pos], rtol=RTOL, atol=ATOL, equal_nan=True)
                pair_matches = counts == a["pair_counts"][pos]
                if not matches.all() or not pair_matches.all():
                    spot_errors.append({"device": device, "shot": sid, "mode": name, "end": end,
                                        "feature_mismatches": int((~matches).sum()),
                                        "pair_mismatches": int((~pair_matches).sum()),
                                        "feature_examples": [{"channel": CHANNELS[int(c)], "field": FIELDS[int(j)],
                                            "computed": None if np.isnan(computed[c,j]) else float(computed[c,j]),
                                            "stored": None if np.isnan(a["features"][pos,c,j]) else float(a["features"][pos,c,j])}
                                            for c,j in np.argwhere(~matches)[:5]]})
                status_total.update(statuses)
                sampled_mode_counts[(device, name)] += 1
                sampled[name].append({"row": pos, "end": end, "features_match": bool(matches.all()),
                                      "pair_counts_match": bool(pair_matches.all())})
        spot_records.append({"device": device, "shot": sid, "order_index": index,
                             "disruptive": shot["disruptive"], "n_points": len(shot["times"]),
                             "sample_rate_hz": shot["rate"], "dt_ms": dt,
                             "expected_endpoints": views, "sampled_rows": sampled})
    mode_coverage = {f"{d}/{m}": sampled_mode_counts[(d, m)] for d in allowed for m in MODES}
    check("primary_hdf5_independent_window_spotchecks", not spot_errors and all(mode_coverage.values()),
          shots=spot_records, sampled_rows_by_mode=mode_coverage,
          total_sampled_rows=sum(mode_coverage.values()), derived_status_counts=dict(status_total),
          n_errors=len(spot_errors), errors=bounded(spot_errors))
    result["numerical_verdict"] = ("PASS" if all(x["pass"] for x in result["checks"].values()) else "FAIL")
    result["verdict"] = ("PASS_WITH_SCOPE_INCIDENT" if result["numerical_verdict"] == "PASS" else "FAIL")
    save()
    print("Verification verdict:", result["verdict"], flush=True)


def timeout_handler(signum, frame):
    raise TimeoutError(f"received signal {signum} before verification completed")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, timeout_handler)
    try:
        run()
    except FileExistsError:
        print("Refusing to overwrite an existing verification result", file=sys.stderr)
        sys.exit(2)
    except BaseException as exc:
        result["verdict"] = "INCOMPLETE" if isinstance(exc, TimeoutError) else "ERROR"
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        try:
            save()
        except BaseException as save_exc:
            print(f"Could not persist verification failure: {save_exc}", file=sys.stderr)
        print(f"Verification stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if result["numerical_verdict"] == "PASS" else 1)
