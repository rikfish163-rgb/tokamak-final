"""CPU-only verification of the saved, ended-20 gate-loss audit.

This checks raw-file and archived-code provenance, HDF5 metadata, selection,
stored vectors, and independently aggregates stored row values. It deliberately
does not import the audit's summarizer or run any source-model forward pass.
"""

import hashlib
import json
import math
from pathlib import Path
import zipfile

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "probe_outputs/gate_loss_audit20_v1"
REFERENCE = ROOT / "probe_outputs/stratified_cold20_context360_v1"
SOURCE = (
    ROOT / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
)
COLD = REFERENCE / "models/cold.pt"
SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
COLD_SHA = "a06377a218a0f06def1eb3652cc091b5c514653681bffcf5dbfda7906f9db344"
SPLIT_TOLERANCE = 2e-5
NAMES = ("original", "DC", "AC", "centered")
CHANNELS = (
    "Bt",
    "CIII",
    "MNM",
    "Mir_VV",
    "Mir_avg_amp",
    "Mir_avg_fre",
    "Z_proxy",
    "dx_a",
    "dy_a",
    "ip",
    "mode_number_n",
    "n=1 amplitude",
    "ne_nG",
    "qa_proxy",
    "sxr_kurt",
    "sxr_mean",
    "sxr_skew",
    "sxr_var",
    "v_loop",
    "xuv_kurt",
    "xuv_ratio",
    "xuv_skew",
    "xuv_var",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    def reject_nonfinite(value):
        raise ValueError(f"nonfinite JSON constant {value} in {path}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_nonfinite)


def select_origins(points):
    """Select up to eight equally spaced candidates; endpoint must exist."""
    candidates = list(range(159, points - 32, 128))
    size = min(8, len(candidates))
    if size == 0:
        return []
    if size == 1:
        return [candidates[0] - 25]
    return [
        candidates[i * (len(candidates) - 1) // (size - 1)] - 25 for i in range(size)
    ]


def distribution(values):
    """Median and linear 10th/90th percentiles from finite, non-null rows."""
    ordered = sorted(float(v) for v in values if v is not None)
    require(all(math.isfinite(v) for v in ordered), "nonfinite distribution value")
    if not ordered:
        return {"n": 0, "median": None, "q10": None, "q90": None}

    def percentile(probability):
        offset = (len(ordered) - 1) * probability
        lo = int(offset)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (offset - lo)

    return {
        "n": len(ordered),
        "median": percentile(0.5),
        "q10": percentile(0.1),
        "q90": percentile(0.9),
    }


def split_errors(total, dc, ac):
    """Return stored-vector float64 error and the original float32 residual."""
    total, dc, ac = (np.asarray(x, dtype=np.float32) for x in (total, dc, ac))
    require(total.shape == dc.shape == ac.shape, "incompatible split vectors")
    require(all(np.isfinite(x).all() for x in (total, dc, ac)), "nonfinite split")
    g, d, a = (x.astype(np.float64) for x in (total, dc, ac))
    denominator = max(np.linalg.norm(g), np.linalg.norm(d) + np.linalg.norm(a), 1e-12)
    exact_stored = float(np.linalg.norm(g - d - a) / denominator)
    float32_residual = (total - dc - ac).astype(np.float64)
    return exact_stored, float(np.linalg.norm(float32_residual) / denominator)


def first_records(path, count):
    """Read exactly the first array objects, never decoding suffix records."""
    result = []
    with path.open("rb", buffering=0) as handle:
        require(handle.read(1) == b"[", "invalid reference order")
        while len(result) < count:
            ch = handle.read(1)
            while ch and ch in b" \t\r\n,":
                ch = handle.read(1)
            require(ch == b"{", "reference order has fewer than 20 records")
            record = bytearray(ch)
            depth, in_string, escaped = 1, False, False
            while depth:
                ch = handle.read(1)
                require(bool(ch), "truncated reference order")
                record.extend(ch)
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == b"\\":
                        escaped = True
                    elif ch == b'"':
                        in_string = False
                elif ch == b'"':
                    in_string = True
                elif ch == b"{":
                    depth += 1
                elif ch == b"}":
                    depth -= 1
            result.append(json.loads(record))
    return result


def first_hashes(path, count):
    """Read just the first 20 entries of the reference hash manifest."""
    result = {}
    with path.open("rb", buffering=0) as handle:
        require(handle.readline().strip() == b"{", "invalid hash manifest")
        for _ in range(count):
            line = handle.readline().strip().rstrip(b",")
            entry = json.loads(b"{" + line + b"}")
            require(
                len(entry) == 1 and not (set(entry) & result.keys()), "bad hash entry"
            )
            result.update(entry)
    return result


def summary_from_rows(rows):
    """Independent aggregation; no audit.run imports or summarize calls."""
    result = {}
    for state in (0.0, 0.2):
        subset = [row for row in rows if row["state_per_gate"] == state]
        stats = {}
        for key in (
            "original_reverse_cosine",
            "centered_reverse_cosine",
            "DC_to_total_cosine",
            "AC_to_total_cosine",
        ):
            stats[key] = distribution(row[key] for row in subset)
        stats["gradient_norms"] = {
            name: distribution(row["gradient_norms"][name] for row in subset)
            for name in NAMES
        }
        stats["relative_AC_norm"] = distribution(
            row["gradient_norms"]["AC"] / row["gradient_norms"]["original"]
            for row in subset
            if row["gradient_norms"]["original"] > 0
        )
        stats["risk_probes"] = {}
        for name in NAMES:
            eligible = [
                row["proposals"][name]
                for row in subset
                if row["risk_label"] is not None
            ]
            deltas = [r["future_risk_loss_delta"] for r in eligible]
            require(
                all(v is not None and math.isfinite(v) for v in deltas),
                "invalid risk delta",
            )
            stats["risk_probes"][name] = {
                "n": len(eligible),
                "moving": sum(r["step_norm"] > 0 for r in eligible),
                "improved": sum(v < 0 for v in deltas),
                "worsened": sum(v > 0 for v in deltas),
                "unchanged": sum(v == 0 for v in deltas),
                "delta": distribution(deltas),
                "cosine": distribution(r["risk_gradient_cosine"] for r in eligible),
            }
        result[str(state)] = stats
    return result


def compare_tree(actual, recorded, path="summary"):
    if isinstance(actual, dict):
        require(
            isinstance(recorded, dict) and actual.keys() == recorded.keys(),
            f"{path}: keys differ",
        )
        return sum(
            compare_tree(v, recorded[k], f"{path}.{k}") for k, v in actual.items()
        )
    if actual is None:
        require(recorded is None, f"{path}: null mismatch")
    elif isinstance(actual, int):
        require(type(recorded) is int and actual == recorded, f"{path}: count mismatch")
    else:
        require(
            isinstance(recorded, (int, float))
            and math.isfinite(recorded)
            and math.isclose(actual, recorded, rel_tol=1e-12, abs_tol=1e-14),
            f"{path}: {actual!r} != {recorded!r}",
        )
    return 1


def verify_archive(protocol):
    manifest = protocol["code_sha256"]
    require(isinstance(manifest, dict) and len(manifest) > 0, "empty code manifest")
    with zipfile.ZipFile(AUDIT / "code_snapshot.zip") as archive:
        names = archive.namelist()
        require(
            len(names) == len(set(names)) and set(names) == set(manifest),
            "archive inventory mismatch",
        )
        for name, expected in manifest.items():
            path = (ROOT / name).resolve()
            require(
                path.is_relative_to(ROOT)
                and (
                    name.endswith(".py") or name == "GATE_LOSS_AUDIT_PLAN_20261101.md"
                ),
                f"unsafe or unexpected archived path: {name}",
            )
            require(
                hashlib.sha256(archive.read(name)).hexdigest() == expected,
                f"archived hash mismatch: {name}",
            )
            require(
                digest(path) == expected,
                f"current source differs from archived code: {name}",
            )
    require(
        manifest["GATE_LOSS_AUDIT_PLAN_20261101.md"] == protocol["plan_sha256"],
        "plan hash mismatch",
    )
    return len(manifest)


def raw_metadata_and_masks(path, origins):
    """Read only this authorized shot's HDF5 metadata and 26-point masks."""
    masks = {}
    with h5py.File(path, "r") as h5:
        ip = h5["data/ip"]
        points = len(ip)
        rate = float(np.asarray(ip.attrs["SampleRate"]).reshape(-1)[0])
        start = float(np.asarray(ip.attrs["StartTime"]).reshape(-1)[0])
        require(
            math.isfinite(rate) and abs(rate - 1000) <= 0.5 and math.isfinite(start),
            "invalid HDF5 clock",
        )
        disruptive = bool(np.asarray(h5["meta/IsDisrupt"][()]).reshape(-1)[0])
        end_data = np.asarray(h5["meta/DownTime"][()]).reshape(-1)
        down = float(end_data[0]) if disruptive and len(end_data) else None
        require(
            not disruptive or (down is not None and math.isfinite(down)),
            "invalid HDF5 outcome",
        )
        for origin in origins:
            columns = [
                np.asarray(h5["data"][name][origin : origin + 26], dtype=np.float32)
                for name in CHANNELS
            ]
            require(all(len(c) == 26 for c in columns), "incomplete target window")
            observed = np.stack([np.isfinite(c) for c in columns], axis=1)
            observed[:, 1] &= ~np.isclose(columns[1], -10.0, atol=1e-2, rtol=0.0)
            observed[:, 2] = False
            target_mask = observed[1:] & observed[0]
            target_mask[:, 10] = False
            target_mask &= target_mask[::-1].copy()
            masks[origin] = target_mask.sum(axis=0).astype(int).tolist()
    return points, rate, start, disruptive, down, masks


def verify():
    protocol = load_json(AUDIT / "protocol.json")
    result = load_json(AUDIT / "completed.json")
    inventory = load_json(AUDIT / "inventory.json")
    rows = load_json(AUDIT / "rows.json")
    mapping_path = REFERENCE / "input_map.json"
    mapping = load_json(mapping_path)
    ids = protocol["target_ids"]
    require(
        protocol["scope"] == "ENDED20_ONLY_OFFLINE_GRADIENT_DIAGNOSTIC_NOT_CONTROLLER",
        "wrong scope",
    )
    require(
        len(ids) == len(set(ids)) == 20 and all(type(x) is int for x in ids),
        "wrong shot membership",
    )
    require(
        protocol["states"] == [0, 0.2] and protocol["max_origins_per_shot"] == 8,
        "unexpected fixed design",
    )
    require(
        protocol["no_training"] is True
        and protocol["target_suffix_signals_read"] == 0
        and protocol["target_suffix_outcomes_read"] == 0,
        "unexpected declared scope",
    )
    require(
        Path(protocol["reference"]) == REFERENCE
        and Path(protocol["source_path"]) == SOURCE,
        "unexpected reference paths",
    )
    require(
        protocol["source_sha256"] == SOURCE_SHA == digest(SOURCE),
        "source hash mismatch",
    )
    require(
        protocol["cold_state_sha256"] == COLD_SHA == digest(COLD), "cold hash mismatch"
    )
    require(
        protocol["input_map_sha256"] == digest(mapping_path),
        "input mapping hash mismatch",
    )
    require(
        mapping["calibration_ids"] == ids, "mapping calibration membership mismatch"
    )
    prefix = first_records(REFERENCE / "order.json", 20)
    require(
        [record["shot"] for record in prefix] == ids,
        "reference first-20 membership mismatch",
    )
    require(
        all(
            record.get("commissioning_phase") == "cold_coverage20" for record in prefix
        ),
        "not the cold prefix",
    )
    reference_hashes = first_hashes(REFERENCE / "input_hashes.json", 20)
    require(
        list(reference_hashes) == list(map(str, ids))
        and reference_hashes == protocol["input_hashes"],
        "reference first-20 input hashes mismatch",
    )
    archived = verify_archive(protocol)
    require(
        result["status"] == "COMPLETED_AUDIT"
        and result["input_shots"] == 20
        and result["rows"] == len(rows)
        and result["shots_with_origins"] == 20
        and result["no_training"] is True
        and result["no_suffix_access"] is True
        and result["source_state_unchanged"] is True,
        "invalid completion metadata",
    )
    require(
        [r["shot"] for r in inventory] == ids, "inventory membership/order mismatch"
    )
    require(len(rows) == 320, "expected 320 state rows")
    expected_keys = []
    metadata = {}
    for shot, item in zip(ids, inventory, strict=True):
        raw = ROOT / "data/EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        require(
            digest(raw) == reference_hashes[str(shot)],
            f"raw input hash mismatch: {shot}",
        )
        points, rate, start, flag, down, masks = raw_metadata_and_masks(
            raw, item["origins"]
        )
        origins = select_origins(points)
        require(
            item["points"] == points and item["origins"] == origins,
            f"origin inventory mismatch: {shot}",
        )
        require(item["disruptive"] is flag, f"HDF5 outcome mismatch: {shot}")
        metadata[shot] = (points, rate, start, flag, down, masks)
        expected_keys.extend(
            (shot, origin, state) for origin in origins for state in (0.0, 0.2)
        )
    require(
        [(r["shot"], r["origin"], r["state_per_gate"]) for r in rows] == expected_keys,
        "row membership, origin, state, or order mismatch",
    )
    stored_errors, mathematical_errors = [], []
    eligible_counts = {"0.0": 0, "0.2": 0}
    for row in rows:
        shot, origin = row["shot"], row["origin"]
        points, rate, start, flag, down, masks = metadata[shot]
        require(
            row["disruptive"] is flag
            and row["maturity"] == origin + 25
            and row["future_endpoint"] == origin + 57 < points,
            f"clock mismatch: {shot}/{origin}",
        )
        lead = (
            round(
                (down - (np.float64(start) + np.float64(origin + 57) / rate)) * 1000, 9
            )
            if flag
            else None
        )
        label = float(lead <= 125) if flag and lead > 30 else 0.0 if not flag else None
        require(
            row["future_lead_ms"] == lead and row["risk_label"] == label,
            f"HDF5 proxy-label mismatch: {shot}/{origin}",
        )
        eligible_counts[str(row["state_per_gate"])] += label is not None
        count_by_channel = masks[origin]
        require(
            row["observed_per_channel"] == count_by_channel
            and row["observed"] == sum(count_by_channel),
            f"observed-mask mismatch: {shot}/{origin}",
        )
        saturation, same_sign = (
            row["saturated_per_channel"],
            row["all_observed_same_sign_saturated"],
        )
        require(
            len(saturation) == len(same_sign) == 23
            and all(
                type(s) is int
                and 0 <= s <= count
                and type(full) is bool
                and (not full or (count > 0 and s == count))
                for s, full, count in zip(
                    saturation, same_sign, count_by_channel, strict=True
                )
            ),
            "invalid saturation metadata",
        )
        gradients = {}
        require(
            set(row["proposals"]) == set(NAMES)
            and set(row["gradient_norms"]) == set(NAMES),
            "proposal inventory mismatch",
        )
        for name in NAMES:
            proposal = row["proposals"][name]
            vec = np.asarray(proposal["gradient32"], dtype=np.float32)
            require(
                vec.shape == (32,) and np.isfinite(vec).all(), "invalid stored gradient"
            )
            gradients[name] = vec.astype(np.float64)
            require(
                math.isclose(
                    float(np.linalg.norm(gradients[name])),
                    row["gradient_norms"][name],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                "stored gradient norm mismatch",
            )
            require(
                math.isfinite(proposal["step_norm"]) and proposal["step_norm"] >= 0,
                "invalid step norm",
            )
            if label is None:
                require(
                    all(
                        proposal[k] is None
                        for k in (
                            "risk_gradient_cosine",
                            "future_risk_loss_delta",
                            "first_order_risk_delta",
                        )
                    ),
                    "risk proxy recorded outside eligible cohort",
                )
            else:
                require(
                    math.isfinite(proposal["future_risk_loss_delta"])
                    and math.isfinite(proposal["first_order_risk_delta"]),
                    "nonfinite eligible risk proxy",
                )
        require(
            (row["future_risk_loss_before"] is None) == (label is None),
            "risk-before eligibility mismatch",
        )
        if label is not None:
            require(
                math.isfinite(row["future_risk_loss_before"]),
                "nonfinite risk-before value",
            )
        for key in ("original_reverse_cosine", "centered_reverse_cosine"):
            value = row[key]
            require(
                value is None or (math.isfinite(value) and abs(value) <= 1 + 1e-12),
                f"invalid stored {key}",
            )
        for name in ("DC", "AC"):
            g, total = gradients[name], gradients["original"]
            denominator = np.linalg.norm(g) * np.linalg.norm(total)
            value = float(np.dot(g, total) / denominator) if denominator else None
            compare_tree(value, row[f"{name}_to_total_cosine"], f"row.{name}_cosine")
        mathematical, stored = split_errors(
            *(row["proposals"][name]["gradient32"] for name in ("original", "DC", "AC"))
        )
        require(
            mathematical <= SPLIT_TOLERANCE and stored <= SPLIT_TOLERANCE,
            "gradient split tolerance exceeded",
        )
        require(
            math.isclose(
                stored, row["split_relative_error"], rel_tol=1e-12, abs_tol=1e-16
            ),
            "reported split error mismatch",
        )
        require(
            math.isfinite(row["direct_gradient_relative_error"])
            and 0 <= row["direct_gradient_relative_error"] <= SPLIT_TOLERANCE,
            "reported direct gradient error exceeds tolerance",
        )
        mathematical_errors.append(mathematical)
        stored_errors.append(stored)
    require(
        len(expected_keys) == 320 and sum(bool(x["origins"]) for x in inventory) == 20,
        "incomplete origin inventory",
    )
    require(
        math.isclose(
            max(stored_errors),
            result["max_split_relative_error"],
            rel_tol=1e-12,
            abs_tol=1e-16,
        ),
        "maximum reported split mismatch",
    )
    summary = summary_from_rows(rows)
    summary_leaves = compare_tree(summary, result["summary"])
    return {
        "status": "PASS_WITH_LIMITS",
        "scope": "ENDED20_ONLY_OFFLINE_GRADIENT_DIAGNOSTIC",
        "outcome_authority": "HDF5 meta/IsDisrupt and meta/DownTime",
        "counts": {
            "raw_hdf5_hashes_verified_against_two_manifests": len(ids),
            "hdf5_outcomes_and_origin_inventories_verified": len(inventory),
            "hdf5_disruptive_shots": sum(item[3] for item in metadata.values()),
            "unique_origins": len(expected_keys) // 2,
            "state_rows": len(rows),
            "gradient_vectors_and_norms_verified": len(rows) * len(NAMES),
            "dc_ac_cosines_verified_from_vectors": len(rows) * 2,
            "split_equations_verified": len(rows),
            "target_masks_and_proxy_labels_verified": len(rows),
            "summary_leaves_verified": summary_leaves,
            "summary_distributions_verified": 34,
            "summary_median_q10_q90_values_verified": 102,
            "risk_probe_groups_verified": 2 * len(NAMES),
            "risk_probe_count_fields_verified": 40,
            "archived_and_current_code_files_verified": archived,
            "risk_eligible_rows_by_state": eligible_counts,
        },
        "relative_split_tolerance": SPLIT_TOLERANCE,
        "maximum_stored_vector_float64_split_error": max(mathematical_errors),
        "maximum_reproduced_float32_split_error": max(stored_errors),
        "source_sha256": SOURCE_SHA,
        "cold_sha256": COLD_SHA,
        "verifier_sha256": digest(Path(__file__)),
        "quantile_method": "linear order-statistic interpolation",
        "audit_input_sha256": {
            name: digest(AUDIT / name)
            for name in (
                "protocol.json",
                "inventory.json",
                "rows.json",
                "completed.json",
                "code_snapshot.zip",
            )
        },
        "recomputed_summary": summary,
        "limits": [
            "All gradients were stored from the source-model forward/VJP; source forwards and VJPs were not independently recomputed.",
            "The two gate states were checked as stored row metadata, not re-established by replaying the source model.",
            "Reverse-gradient and risk-gradient vectors were not saved; their row cosines and risk-loss deltas were aggregated but not independently regenerated.",
            "Claims that no training, no suffix access, and source state unchanged are audit-record declarations, not independently monitored execution traces.",
            "The first 20 shots were already used in preparation; stored vectors and archive consistency do not prove causal physics or independent future performance.",
        ],
        "suffix_signal_outcome_or_metric_files_opened": 0,
        "gpu_used": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
