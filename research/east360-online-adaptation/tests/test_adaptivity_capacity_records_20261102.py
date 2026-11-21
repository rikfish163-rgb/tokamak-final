import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.adaptivity_capacity_20261102 import verify_records


def digest(data):
    return hashlib.sha256(data).hexdigest()


def npz_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def tensor_bytes(state):
    buffer = io.BytesIO()
    torch.save({"state": state}, buffer)
    return buffer.getvalue()


def raw_bytes(disruptive):
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as raw:
        raw.create_dataset("meta/IsDisrupt", data=[int(disruptive)])
        raw.create_dataset("meta/DownTime", data=[0.2])
    return buffer.getvalue()


class MemoryFiles:
    def __init__(self):
        self.data = {}

    def put(self, path, value):
        self.data[Path(path)] = value

    def read(self, path):
        return self.data[Path(path)]

    def put_json(self, path, value):
        self.put(path, json.dumps(value, allow_nan=False).encode())

    def json(self, path):
        return json.loads(self.read(path))

    def arrays(self, path):
        with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}


def synthetic_records(monkeypatch, *, preflight=False):
    fs = MemoryFiles()
    root = Path("/virtual/worktree")
    output = root / "output"
    reference = root / "reference"
    source, cold, mapping = root / "source.pt", root / "cold.pt", root / "input_map.json"
    for path, data in ((source, b"source"), (cold, b"cold"), (mapping, b"mapping")):
        fs.put(path, data)
    reference_complete = b"sealed reference completion"
    fs.put(reference / "completed.json", reference_complete)
    fs.put(reference / "timing_bounds.json", b"sealed reference clock bounds")
    order = [[rank, 10000 + rank] for rank in range(21, 95)]
    selected = order[:1] if preflight else order
    reference_protocol = {
        "order": [{"shot": 10000 + rank} for rank in range(1, 95)],
        "threshold": 0.5,
        "source_checkpoint": str(source),
        "source_sha256": digest(fs.read(source)),
        "cold_state": str(cold),
        "cold_state_sha256": digest(fs.read(cold)),
        "reference": str(root),
        "input_map_sha256": digest(fs.read(mapping)),
    }
    fs.put_json(reference / "protocol.json", reference_protocol)
    source_files = {
        "ADAPTIVITY_CAPACITY_PLAN_20261102.md": b"plan",
        "experiments/risk_aligned_online/adaptivity_capacity_20261102/geometry.py": b"geometry",
        "experiments/risk_aligned_online/adaptivity_capacity_20261102/model.py": b"model",
        "experiments/risk_aligned_online/adaptivity_capacity_20261102/run.py": b"runner",
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in source_files.items():
            zipped.writestr(name, content)
    fs.put(output / "code_snapshot.zip", archive.getvalue())
    fs.put(output / "source_basis.pt", b"basis")
    protocol = {
        "scope": "POSTHOC_LABEL_ORACLE_FINITE_WITNESSES_NOT_ONLINE",
        "preflight": preflight,
        "families": ["gates", "ffn_lora"],
        "order": selected,
        "reference": str(reference),
        "reference_completed_sha256": digest(reference_complete),
        "reference_protocol_sha256": digest(fs.read(reference / "protocol.json")),
        "clock_bounds_sha256": digest(fs.read(reference / "timing_bounds.json")),
        "source_sha256": digest(fs.read(source)),
        "cold_sha256": digest(fs.read(cold)),
        "basis_sha256": digest(fs.read(output / "source_basis.pt")),
        "threshold": 0.5,
        "radius": 0.25,
        "max_steps": 96,
        "learning_rate": 0.05,
        "certification_interval": 24,
        "objective_margin": 0.1,
        "witness_margin": 0.05,
        "first_mutable_index": 160,
        "uses_current_case_label_and_future_record": True,
        "can_be_called_online_gain": False,
        "plan_sha256": digest(source_files["ADAPTIVITY_CAPACITY_PLAN_20261102.md"]),
        "code_sha256": {name: digest(value) for name, value in source_files.items()},
    }
    fs.put_json(output / "protocol.json", protocol)
    times = np.arange(200, dtype=np.float64) / 1000
    low, high = np.float32(0.1), np.float32(0.9)
    low_logit = np.float32(math.log(1 / 9))
    high_logit = np.float32(math.log(9))
    states = {
        "gates": tensor_bytes(torch.full((16, 2), 0.1)),
        "ffn_lora": tensor_bytes(torch.full((4, 64), 0.1)),
    }
    outcomes, input_hashes = [], {}
    raw_by_flag = {True: raw_bytes(True), False: raw_bytes(False)}
    for index, (rank, shot) in enumerate(selected):
        disruptive = index < 54
        prefix = f"{rank:04d}_{shot}"
        raw_path = root / "data/EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        fs.put(raw_path, raw_by_flag[disruptive])
        input_hashes[str(shot)] = digest(fs.read(raw_path))
        original = np.full(len(times), low, dtype=np.float32)
        if not disruptive:
            original[160] = high
        fs.put(reference / "shots" / f"{prefix}.npz", npz_bytes(times=times, cold=original))
        probability = np.full(len(times), low, dtype=np.float32)
        logit = np.full(len(times), low_logit, dtype=np.float32)
        if disruptive:
            probability[165], logit[165] = high, high_logit
        score = {
            "alarm": disruptive,
            "lead_ms": 35.0 if disruptive else None,
            "timely": disruptive,
            "logit_margin": float(high_logit) if disruptive else -float(low_logit),
            "operational_hit": True,
            "witness": True,
            "early_over3000": False,
        }
        for family in ("gates", "ffn_lora"):
            case = output / "cases" / prefix
            state_path, curve_path = case / f"{family}_state.pt", case / f"{family}_curves.npz"
            fs.put(state_path, states[family])
            fs.put(curve_path, npz_bytes(times=times, cold=original, probability=probability, logit=logit))
            receipt = {
                "rank": rank,
                "shot": shot,
                "disruptive": disruptive,
                "family": family,
                "score": dict(score),
                "best_step": 24,
                "steps_run": 24,
                "validation": {"max_hidden_rms": 0.2, "cached_direct_logit_difference": 0.0},
                "trace": [],
                "stages": [],
                "raw_sha256": digest(fs.read(raw_path)),
                "state_sha256": digest(fs.read(state_path)),
                "curves_sha256": digest(fs.read(curve_path)),
                "label_oracle": True,
                "not_online": True,
            }
            fs.put_json(case / f"{family}_receipt.json", receipt)
            outcomes.append({key: receipt[key] for key in (
                "rank", "shot", "disruptive", "family", "score", "best_step", "steps_run"
            )})
    fs.put_json(root / "input_hashes.json", input_hashes)
    counts = {"D": min(len(selected), 54), "SAFE": max(len(selected) - 54, 0)}
    summary = {
        family: {label: {"n": count, "witnesses": count, "early_over3000": 0}
                 for label, count in counts.items()}
        for family in ("gates", "ffn_lora")
    }
    fs.put_json(output / "completed.json", {
        "status": "COMPLETED_POSTHOC_CAPACITY",
        "summary": summary,
        "outcomes": outcomes,
        "source_state_unchanged": True,
        "no_online_gain_claim": True,
        "method_promoted": False,
        "scientific_goal_achieved": False,
        "seconds": 1.0,
    })
    real_outcome = verify_records.read_outcome
    monkeypatch.setattr(verify_records, "ROOT", root)
    monkeypatch.setattr(verify_records, "REFERENCE_COMPLETED_SHA256", digest(reference_complete))
    monkeypatch.setattr(verify_records, "SOURCE_SHA256", digest(fs.read(source)))
    monkeypatch.setattr(verify_records, "read_bytes", fs.read)
    monkeypatch.setattr(verify_records, "sha256_file", lambda path: digest(fs.read(path)))
    monkeypatch.setattr(verify_records, "read_outcome", lambda path: real_outcome(io.BytesIO(fs.read(path))))
    return fs, output, reference


def curve(*, index=None, logit_value=1.0):
    times = np.arange(200, dtype=np.float64) / 1000
    logits = np.full(200, -2.0)
    if index is not None:
        logits[index] = logit_value
    probability = 1 / (1 + np.exp(-logits))
    return times, probability, logits


@pytest.mark.parametrize("index,timely", [(169, True), (170, False)])
def test_lead_must_be_strictly_more_than_30_ms(index, timely):
    times, probability, logit = curve(index=index)
    score = verify_records.evaluate_curve(times, probability, logit, True, 0.2, 0.5)
    assert score["timely"] is timely
    assert score["witness"] is timely


def test_immutable_prefix_false_alarm_prevents_safe_witness():
    times, probability, logit = curve(index=30)
    score = verify_records.evaluate_curve(times, probability, logit, False, None, 0.5)
    assert score["alarm"] and not score["operational_hit"]
    assert score["logit_margin"] >= 0.05
    assert not score["witness"]


@pytest.mark.parametrize("margin,witness", [(0.05, True), (0.049999, False)])
def test_witness_requires_mutable_logit_margin(margin, witness):
    times, probability, logit = curve(index=165, logit_value=margin)
    score = verify_records.evaluate_curve(times, probability, logit, True, 0.2, 0.5)
    assert score["timely"]
    assert score["witness"] is witness


@pytest.mark.parametrize("down,early", [(3.16, False), (3.160001, True)])
def test_early_alarm_is_strictly_over_3000_ms(down, early):
    times, probability, logit = curve(index=160)
    score = verify_records.evaluate_curve(times, probability, logit, True, down, 0.5)
    assert score["early_over3000"] is early


def test_complete_recalculates_74_by_two_and_fingerprints_receipts(monkeypatch):
    fs, output, _ = synthetic_records(monkeypatch)
    report = verify_records.verify(output)
    assert report["status"] == "PASS"
    assert report["mode"] == "COMPLETE"
    assert report["cases"] == 74 and report["records"] == 148
    assert report["summary"]["gates"]["D"] == {"n": 54, "witnesses": 54, "early_over3000": 0}
    assert report["summary"]["ffn_lora"]["SAFE"]["n"] == 20
    first = report["record_fingerprints"][0]
    receipt = output / "cases/0021_10021/gates_receipt.json"
    assert first["receipt_sha256"] == digest(fs.read(receipt))
    assert "not an independent neural" in report["recorded_rms_limitation"].lower()


def test_preflight_requires_explicit_opt_in_and_exactly_one_case(monkeypatch):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    with pytest.raises(ValueError, match="preflight"):
        verify_records.verify(output)
    report = verify_records.verify(output, allow_preflight=True)
    assert report["mode"] == "PREFLIGHT_ONLY" and report["cases"] == 1 and report["records"] == 2
    protocol = fs.json(output / "protocol.json")
    protocol["order"].append([22, 10022])
    fs.put_json(output / "protocol.json", protocol)
    with pytest.raises(ValueError, match="preflight"):
        verify_records.verify(output, allow_preflight=True)


@pytest.mark.parametrize("artifact", ["gates_state.pt", "gates_curves.npz"])
def test_tampered_artifact_hash_fails(monkeypatch, artifact):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    path = output / "cases/0021_10021" / artifact
    fs.put(path, fs.read(path) + b"tamper")
    with pytest.raises(ValueError, match="sha256"):
        verify_records.verify(output, allow_preflight=True)


def test_tampered_raw_input_hash_fails(monkeypatch):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    raw = Path("/virtual/worktree/data/EAST/unified_hdf5/10000/10021.hdf5")
    fs.put(raw, fs.read(raw) + b"tamper")
    with pytest.raises(ValueError, match="sha256"):
        verify_records.verify(output, allow_preflight=True)


def test_prior_input_hashes_come_from_centered_reference_parent(monkeypatch):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    hashes = fs.json(Path("/virtual/worktree/input_hashes.json"))
    hashes["10021"] = "0" * 64
    fs.put_json(Path("/virtual/worktree/input_hashes.json"), hashes)
    with pytest.raises(ValueError, match="sha256"):
        verify_records.verify(output, allow_preflight=True)


def test_tampered_snapshot_fails_even_if_zip_remains_valid(monkeypatch):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    path = output / "code_snapshot.zip"
    original, modified = io.BytesIO(fs.read(path)), io.BytesIO()
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(modified, "w") as dst:
        for name in src.namelist():
            content = src.read(name)
            dst.writestr(name, content + b"altered" if name.endswith("geometry.py") else content)
    fs.put(path, modified.getvalue())
    with pytest.raises(ValueError, match="sha256"):
        verify_records.verify(output, allow_preflight=True)


def test_altered_prefix_is_rejected_even_with_updated_curve_hash(monkeypatch):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    case = output / "cases/0021_10021"
    curve_path, receipt_path = case / "gates_curves.npz", case / "gates_receipt.json"
    arrays = fs.arrays(curve_path)
    arrays["probability"][0] = 0.9
    arrays["logit"][0] = np.float32(math.log(9))
    fs.put(curve_path, npz_bytes(**arrays))
    receipt = fs.json(receipt_path)
    receipt["curves_sha256"] = digest(fs.read(curve_path))
    fs.put_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="prefix"):
        verify_records.verify(output, allow_preflight=True)


def test_reference_times_and_cold_are_checked_independently(monkeypatch):
    fs, output, reference = synthetic_records(monkeypatch, preflight=True)
    path = reference / "shots/0021_10021.npz"
    arrays = fs.arrays(path)
    arrays["times"][0] += 1e-6
    fs.put(path, npz_bytes(**arrays))
    with pytest.raises(ValueError, match="reference"):
        verify_records.verify(output, allow_preflight=True)


def test_reference_cold_disruption_must_actually_be_a_miss(monkeypatch):
    fs, output, reference = synthetic_records(monkeypatch, preflight=True)
    ref_path = reference / "shots/0021_10021.npz"
    reference_arrays = fs.arrays(ref_path)
    reference_arrays["cold"][165] = 0.9
    fs.put(ref_path, npz_bytes(**reference_arrays))
    case = output / "cases/0021_10021"
    for family in ("gates", "ffn_lora"):
        path, receipt_path = case / f"{family}_curves.npz", case / f"{family}_receipt.json"
        arrays = fs.arrays(path)
        arrays["cold"][165] = 0.9
        fs.put(path, npz_bytes(**arrays))
        receipt = fs.json(receipt_path)
        receipt["curves_sha256"] = digest(fs.read(path))
        fs.put_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="reference cold"):
        verify_records.verify(output, allow_preflight=True)


def test_reference_safe_fp_in_fixed_prefix_is_not_changeable(monkeypatch):
    fs, output, reference = synthetic_records(monkeypatch)
    ref_path = reference / "shots/0075_10075.npz"
    reference_arrays = fs.arrays(ref_path)
    reference_arrays["cold"][20] = 0.9
    fs.put(ref_path, npz_bytes(**reference_arrays))
    case = output / "cases/0075_10075"
    for family in ("gates", "ffn_lora"):
        path, receipt_path = case / f"{family}_curves.npz", case / f"{family}_receipt.json"
        arrays = fs.arrays(path)
        arrays["cold"][20] = arrays["probability"][20] = 0.9
        arrays["logit"][20] = np.float32(math.log(9))
        fs.put(path, npz_bytes(**arrays))
        receipt = fs.json(receipt_path)
        receipt["curves_sha256"] = digest(fs.read(path))
        fs.put_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="reference cold"):
        verify_records.verify(output)


@pytest.mark.parametrize("field,value", [("max_hidden_rms", 0.250002), ("max_hidden_rms", -1), ("cached_direct_logit_difference", 0.000021)])
def test_bad_recorded_validation_fails(monkeypatch, field, value):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    path = output / "cases/0021_10021/gates_receipt.json"
    receipt = fs.json(path)
    receipt["validation"][field] = value
    fs.put_json(path, receipt)
    with pytest.raises(ValueError, match="validation"):
        verify_records.verify(output, allow_preflight=True)


def test_wrong_completed_summary_or_receipt_decision_is_rejected(monkeypatch):
    fs, output, _ = synthetic_records(monkeypatch, preflight=True)
    completed = fs.json(output / "completed.json")
    completed["summary"]["gates"]["D"]["witnesses"] = 0
    fs.put_json(output / "completed.json", completed)
    with pytest.raises(ValueError, match="summary"):
        verify_records.verify(output, allow_preflight=True)


def test_output_creation_is_exclusive_and_refuses_overwrite(monkeypatch):
    writes = []

    def open_once(path, mode, *, encoding):
        assert path.name == "verification_records.json" and mode == "x" and encoding == "utf-8"
        if writes:
            raise FileExistsError(path)
        writes.append(True)
        return io.StringIO()

    monkeypatch.setattr(Path, "open", open_once)
    verify_records.write_report(Path("/virtual/output"), {"status": "PASS"})
    with pytest.raises(FileExistsError):
        verify_records.write_report(Path("/virtual/output"), {"status": "PASS"})


def test_real_hdf5_metadata_reader_uses_authoritative_fields():
    assert verify_records.read_outcome(io.BytesIO(raw_bytes(True))) == (True, 0.2)
    assert verify_records.read_outcome(io.BytesIO(raw_bytes(False))) == (False, None)
