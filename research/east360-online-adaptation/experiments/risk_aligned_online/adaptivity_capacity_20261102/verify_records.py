"""CPU-only audit of saved post-hoc records, not neural-model replay."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_COMPLETED_SHA256 = "2b28ac02039b9106627c5aca865d23bcdd340ca2eda5d55d328aea46850f0d42"
SOURCE_SHA256 = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
FAMILIES = ("gates", "ffn_lora")
PLAN = "ADAPTIVITY_CAPACITY_PLAN_20261102.md"
MODULE = "experiments/risk_aligned_online/adaptivity_capacity_20261102"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bytes(path):
    return Path(path).read_bytes()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON value {value}")

    return json.loads(read_bytes(path), object_pairs_hook=pairs, parse_constant=invalid)


def finite_number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"invalid {name}")
    require(math.isfinite(value), f"nonfinite {name}")
    return float(value)


def read_outcome(path):
    """Read only the authoritative metadata fields from the current shot."""
    with h5py.File(path, "r") as raw:
        flags = np.asarray(raw["meta/IsDisrupt"][()]).reshape(-1)
        require(len(flags) == 1 and flags[0] in (0, 1), "invalid HDF5 IsDisrupt")
        disruptive = bool(flags[0])
        values = np.asarray(raw["meta/DownTime"][()]).reshape(-1)
    if not disruptive:
        return False, None
    require(len(values) == 1 and np.isfinite(values[0]), "invalid HDF5 DownTime")
    return True, float(values[0])


def evaluate_curve(times, probability, logit, disruptive, down, threshold):
    """Independently recompute the whole-curve decision and mutable margin."""
    times, probability, logit = map(np.asarray, (times, probability, logit))
    require(times.ndim == 1 and len(times) > 160, "invalid curve length")
    require(probability.shape == times.shape and logit.shape == times.shape, "invalid curve shapes")
    require(all(np.issubdtype(x.dtype, np.floating) for x in (times, probability, logit)), "invalid curve dtype")
    require(all(np.isfinite(x).all() for x in (times, probability, logit)), "nonfinite curve")
    require(np.all(np.diff(times) > 0), "timestamps must increase")
    require(np.all((probability >= 0) & (probability <= 1)), "invalid probabilities")
    threshold = finite_number(threshold, "threshold")
    require(0 < threshold < 1, "invalid threshold")
    require(type(disruptive) is bool, "invalid disruption flag")
    if disruptive:
        down = finite_number(down, "DownTime")
        remaining = (down - times) * 1000
        valid_alarm = remaining >= -1e-6
        eligible = (np.arange(len(times)) >= 160) & (np.round(remaining, 9) > 30)
    else:
        valid_alarm = np.ones(len(times), dtype=bool)
        eligible = np.arange(len(times)) >= 160
    require(eligible.any(), "no mutable eligible endpoint")
    alarms = np.flatnonzero(valid_alarm & (probability >= threshold))
    alarm = bool(len(alarms))
    lead = round(float((down - times[alarms[0]]) * 1000), 9) if disruptive and alarm else None
    timely = bool(lead is not None and lead > 30)
    boundary = math.log(threshold / (1 - threshold))
    peak = float(logit[eligible].max())
    margin = peak - boundary if disruptive else boundary - peak
    operational_hit = timely if disruptive else not alarm
    return {
        "alarm": alarm,
        "lead_ms": lead,
        "timely": timely,
        "logit_margin": margin,
        "operational_hit": operational_hit,
        "witness": bool(operational_hit and margin >= 0.05),
        "early_over3000": bool(timely and lead > 3000),
    }


def check_reference_case(times, cold, disruptive, down, threshold):
    """Check the selected case is a cold miss or a mutable cold false alarm."""
    require(times.ndim == cold.ndim == 1 and times.shape == cold.shape and len(times) > 160, "invalid reference cold curve")
    require(np.isfinite(times).all() and np.isfinite(cold).all() and np.all(np.diff(times) > 0), "invalid reference cold data")
    require(np.all((cold >= 0) & (cold <= 1)), "invalid reference cold probability")
    if disruptive:
        remaining = (down - times) * 1000
        alarms = np.flatnonzero((remaining >= -1e-6) & (cold >= threshold))
        require(not len(alarms) or round(float(remaining[alarms[0]]), 9) <= 30, "reference cold disruption is not a miss")
        require(np.any((np.arange(len(times)) >= 160) & (np.round(remaining, 9) > 30)), "reference cold disruption has no mutable timely endpoint")
    else:
        alarms = np.flatnonzero(cold >= threshold)
        require(len(alarms) > 0 and alarms[0] >= 160, "reference cold false alarm is not mutable")


def check_snapshot(output, protocol):
    declared = protocol["code_sha256"]
    require(isinstance(declared, dict) and declared, "missing code sha256 mapping")
    required = {PLAN, *(f"{MODULE}/{name}.py" for name in ("geometry", "model", "run"))}
    require(required <= set(declared), "incomplete source snapshot")
    require(declared[PLAN] == protocol["plan_sha256"], "plan sha256 mismatch")
    for name in declared:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in name, "invalid source path")
    content = read_bytes(output / "code_snapshot.zip")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        info = archive.infolist()
        names = [entry.filename for entry in info]
        require(len(names) == len(set(names)) and set(names) == set(declared), "snapshot entries mismatch")
        require(sum(entry.file_size for entry in info) <= 512 * 1024 * 1024, "oversized source snapshot")
        for entry in info:
            require(not entry.is_dir() and entry.file_size <= 64 * 1024 * 1024, "invalid source entry")
            require(sha256_bytes(archive.read(entry)) == declared[entry.filename], f"source sha256 mismatch: {entry.filename}")
    return sha256_bytes(content)


def load_arrays(data, keys):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        require(set(keys) <= set(archive.files), "missing curve arrays")
        return {key: archive[key] for key in keys}


def check_state(data, family):
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and set(payload) == {"state"}, "invalid state payload")
    state = payload["state"]
    shape = (16, 2) if family == "gates" else (4, 64)
    require(isinstance(state, torch.Tensor) and tuple(state.shape) == shape and state.is_floating_point(), "invalid state shape or type")
    require(bool(torch.isfinite(state).all()), "nonfinite state")
    if family == "gates":
        require(bool((state >= -1e-7).all()) and bool((state.sum(-1) <= 1 + 1e-7).all()), "infeasible gate state")
    return state


def check_score(saved, actual):
    require(isinstance(saved, dict), "missing receipt score")
    for key in ("alarm", "timely", "operational_hit", "witness", "early_over3000"):
        require(type(saved.get(key)) is bool and saved[key] == actual[key], f"receipt score mismatch: {key}")
    for key in ("lead_ms", "logit_margin"):
        if actual[key] is None:
            require(saved.get(key) is None, f"receipt score mismatch: {key}")
        else:
            value = finite_number(saved.get(key), key)
            require(math.isclose(value, actual[key], rel_tol=1e-12, abs_tol=1e-10), f"receipt score mismatch: {key}")


def check_protocol(protocol, allow_preflight):
    require(protocol.get("scope") == "POSTHOC_LABEL_ORACLE_FINITE_WITNESSES_NOT_ONLINE", "invalid scope")
    require(protocol.get("families") == list(FAMILIES), "invalid families")
    require(type(protocol.get("preflight")) is bool, "invalid preflight flag")
    order = protocol["order"]
    require(isinstance(order, list), "invalid order")
    if protocol["preflight"]:
        require(allow_preflight and len(order) == 1, "preflight requires opt-in and exactly one case")
    else:
        require(len(order) == 74, "complete run must contain exactly 74 cases")
    for row in order:
        require(isinstance(row, list) and len(row) == 2, "invalid order entry")
        rank, shot = row
        require(type(rank) is int and rank > 20 and type(shot) is int and shot > 0, "invalid rank or shot")
    require(len({row[0] for row in order}) == len(order) and len({row[1] for row in order}) == len(order), "duplicate order entry")
    require([r[0] for r in order] == sorted(r[0] for r in order), "order not increasing")
    expected = {
        "radius": 0.25,
        "max_steps": 96,
        "learning_rate": 0.05,
        "certification_interval": 24,
        "objective_margin": 0.10,
        "witness_margin": 0.05,
        "first_mutable_index": 160,
    }
    for key, value in expected.items():
        require(type(protocol.get(key)) is not bool and protocol.get(key) == value, f"invalid protocol {key}")
    require(protocol.get("uses_current_case_label_and_future_record") is True, "label-oracle scope missing")
    require(protocol.get("can_be_called_online_gain") is False, "invalid online-gain claim")
    require(protocol.get("reference_completed_sha256") == REFERENCE_COMPLETED_SHA256, "invalid reference sha256")
    require(protocol.get("source_sha256") == SOURCE_SHA256, "invalid fixed source sha256")
    threshold = finite_number(protocol.get("threshold"), "threshold")
    require(0 < threshold < 1, "invalid threshold")
    return order, threshold


def verify(output, *, allow_preflight=False):
    """Verify records without writing output or importing model/runner code."""
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    order, threshold = check_protocol(protocol, allow_preflight)
    completed = read_json(output / "completed.json")
    require(completed.get("status") == "COMPLETED_POSTHOC_CAPACITY", "run is not completed")
    require(completed.get("source_state_unchanged") is True and completed.get("no_online_gain_claim") is True, "invalid completion scope")
    require(completed.get("method_promoted") is False and completed.get("scientific_goal_achieved") is False, "invalid completion claim")
    outcomes = completed.get("outcomes")
    require(isinstance(outcomes, list) and len(outcomes) == 2 * len(order), "invalid completed outcomes")
    reference = Path(protocol["reference"])
    for path, expected in (
        (reference / "completed.json", protocol["reference_completed_sha256"]),
        (reference / "protocol.json", protocol["reference_protocol_sha256"]),
        (reference / "timing_bounds.json", protocol["clock_bounds_sha256"]),
        (output / "source_basis.pt", protocol["basis_sha256"]),
    ):
        require(sha256_file(path) == expected, f"sha256 mismatch: {path.name}")
    ref_protocol = read_json(reference / "protocol.json")
    require(ref_protocol["source_sha256"] == protocol["source_sha256"], "source sha256 mismatch")
    require(ref_protocol["cold_state_sha256"] == protocol["cold_sha256"], "cold sha256 mismatch")
    require(ref_protocol["threshold"] == threshold, "threshold mismatch with reference")
    for path, expected in (
        (Path(ref_protocol["source_checkpoint"]), protocol["source_sha256"]),
        (Path(ref_protocol["cold_state"]), protocol["cold_sha256"]),
        (Path(ref_protocol["reference"]) / "input_map.json", ref_protocol["input_map_sha256"]),
    ):
        require(sha256_file(path) == expected, f"sha256 mismatch: {path.name}")
    snapshot_hash = check_snapshot(output, protocol)
    input_hash_path = Path(ref_protocol["reference"]) / "input_hashes.json"
    input_hashes = read_json(input_hash_path)
    require(isinstance(input_hashes, dict), "invalid reference input hashes")
    reference_order = ref_protocol["order"]
    summary = {family: {label: {"n": 0, "witnesses": 0, "early_over3000": 0}
                        for label in ("D", "SAFE")} for family in FAMILIES}
    fingerprints = []
    for rank, shot in order:
        require(rank <= len(reference_order) and reference_order[rank - 1]["shot"] == shot, "reference order mismatch")
        prefix = f"{rank:04d}_{shot}"
        raw_path = ROOT / "data/EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        raw_hash = sha256_file(raw_path)
        require(input_hashes.get(str(shot)) == raw_hash, f"reference input sha256 mismatch: {shot}")
        disruptive, down = read_outcome(raw_path)
        reference_curves = load_arrays(read_bytes(reference / "shots" / f"{prefix}.npz"), ("times", "cold"))
        check_reference_case(reference_curves["times"], reference_curves["cold"], disruptive, down, threshold)
        for family in FAMILIES:
            case = output / "cases" / prefix
            receipt_path = case / f"{family}_receipt.json"
            receipt_bytes = read_bytes(receipt_path)
            receipt = read_json(receipt_path)
            require(receipt.get("rank") == rank and receipt.get("shot") == shot and receipt.get("family") == family, "receipt identity mismatch")
            require(type(receipt.get("disruptive")) is bool and receipt["disruptive"] == disruptive, "HDF5 outcome mismatch")
            require(receipt.get("label_oracle") is True and receipt.get("not_online") is True, "invalid receipt scope")
            require(receipt.get("raw_sha256") == raw_hash, "receipt raw sha256 mismatch")
            state_bytes = read_bytes(case / f"{family}_state.pt")
            curve_bytes = read_bytes(case / f"{family}_curves.npz")
            require(sha256_bytes(state_bytes) == receipt.get("state_sha256"), "state sha256 mismatch")
            require(sha256_bytes(curve_bytes) == receipt.get("curves_sha256"), "curves sha256 mismatch")
            state = check_state(state_bytes, family)
            curves = load_arrays(curve_bytes, ("times", "cold", "probability", "logit"))
            require(np.array_equal(curves["times"], reference_curves["times"]) and np.array_equal(curves["cold"], reference_curves["cold"]), "reference times or cold mismatch")
            require(np.array_equal(curves["probability"][:160], curves["cold"][:160]), "immutable prefix mismatch")
            actual = evaluate_curve(curves["times"], curves["probability"], curves["logit"], disruptive, down, threshold)
            logits = curves["logit"].astype(np.float64)
            predicted = np.empty_like(logits)
            nonnegative = logits >= 0
            predicted[nonnegative] = 1 / (1 + np.exp(-logits[nonnegative]))
            exp_logits = np.exp(logits[~nonnegative])
            predicted[~nonnegative] = exp_logits / (1 + exp_logits)
            require(np.allclose(curves["probability"], predicted, rtol=1e-6, atol=1e-7), "probability and logit mismatch")
            check_score(receipt.get("score"), actual)
            best_step, steps_run = receipt.get("best_step"), receipt.get("steps_run")
            require(type(best_step) is int and type(steps_run) is int and 0 <= best_step <= steps_run <= 96 and best_step % 24 == 0 and steps_run % 24 == 0, "invalid optimization steps")
            if best_step == 0:
                require(not bool(torch.count_nonzero(state)) and np.array_equal(curves["probability"], curves["cold"]), "invalid zero-step record")
            validation = receipt.get("validation")
            require(isinstance(validation, dict), "missing validation record")
            rms = finite_number(validation.get("max_hidden_rms"), "validation RMS")
            difference = finite_number(validation.get("cached_direct_logit_difference"), "validation difference")
            require(0 <= rms <= 0.250001 and 0 <= difference <= 2e-5, "recorded validation exceeds bounds")
            index = len(fingerprints)
            expected_outcome = {key: receipt[key] for key in ("rank", "shot", "disruptive", "family", "score", "best_step", "steps_run")}
            require(outcomes[index] == expected_outcome, "completed outcome differs from receipt")
            group = summary[family]["D" if disruptive else "SAFE"]
            group["n"] += 1
            group["witnesses"] += int(actual["witness"])
            group["early_over3000"] += int(actual["witness"] and actual["early_over3000"])
            fingerprints.append({
                "rank": rank, "shot": shot, "family": family,
                "receipt_sha256": sha256_bytes(receipt_bytes),
                "state_sha256": sha256_bytes(state_bytes),
                "curves_sha256": sha256_bytes(curve_bytes),
                "raw_sha256": raw_hash,
            })
    if not protocol["preflight"]:
        require(all(summary[family]["D"]["n"] == 54 and summary[family]["SAFE"]["n"] == 20 for family in FAMILIES), "full cohort must be D54/SAFE20")
    require(summary == completed.get("summary"), "completed summary mismatch")
    return {
        "status": "PASS",
        "mode": "PREFLIGHT_ONLY" if protocol["preflight"] else "COMPLETE",
        "scope": "POSTHOC_LABEL_ORACLE_FINITE_WITNESSES_NOT_ONLINE",
        "cases": len(order),
        "records": len(fingerprints),
        "summary": summary,
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "completed_sha256": sha256_file(output / "completed.json"),
        "code_snapshot_sha256": snapshot_hash,
        "reference_input_hashes_path": str(input_hash_path),
        "reference_input_hashes_sha256": sha256_file(input_hash_path),
        "recorded_rms_limitation": "Recorded RMS and cached/direct differences are checked against bounds; this is not an independent neural recomputation.",
        "receipt_hash_limitation": "Receipt SHA-256 values are fingerprints, not externally anchored receipt signatures.",
        "interpretation": "Post-hoc labelled finite search, not online TPR or a global capacity bound.",
        "record_fingerprints": fingerprints,
    }


def write_report(output, report):
    rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with (Path(output) / "verification_records.json").open("x", encoding="utf-8") as stream:
        stream.write(rendered)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-preflight", action="store_true")
    args = parser.parse_args(argv)
    report = verify(args.output, allow_preflight=args.allow_preflight)
    write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "mode", "cases", "records", "summary")}))


if __name__ == "__main__":
    main()
