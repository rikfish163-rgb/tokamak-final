"""Independent CPU verifier tests using only synthetic, in-memory records."""

from copy import deepcopy
import hashlib
import io
import json
import math
import sys
import zipfile

import pytest

from experiments.risk_aligned_online.difference_feedback_probe_20261102 import (
    verify_records as verifier,
)
from experiments.risk_aligned_online.difference_feedback_probe_20261102.verify_records import (
    recompute_summary,
    stable_bce,
    validate_membership,
    validate_sample,
)


ARMS = (
    "centered_real",
    "difference_real",
    "difference_reverse",
    "difference_hold",
)


def receipt(gradient, *, observed=1, arm="difference_real"):
    vector = [gradient] * 256
    loss = 0.0 if arm == "difference_hold" else 1.0
    return {
        "gradient256": vector,
        "state256_after": [0.0] * 256,
        "gradient_norm": math.hypot(*vector),
        "step_norm": 0.0,
        "observed": observed,
        "accepted": False,
        "gradient_computed": observed > 0,
        "reason": "not_improved" if observed else "no_target",
        "loss_before": loss if observed else None,
        "loss_proposal": loss if observed else None,
        "loss_retained": loss if observed else None,
        "max_action_rms": 0.0 if observed or arm == "centered_real" else None,
    }


def sample():
    gradients = {
        "centered_real": 0.003,
        "difference_real": 0.003 - 0.001,
        "difference_reverse": 0.001,
        "difference_hold": 0.0,
    }
    row = {
        "shot": 11,
        "query": 400,
        "disruptive": True,
        "label": 1,
        "query_time": 0.4,
        "requested_lead_ms": 100,
        "actual_lead_ms": 100.0,
        "origin": 374,
        "target_first": 375,
        "target_last": 399,
        "first_affected": 400,
        "observed": 1,
        "state256_before": [0.0] * 256,
        "task_gradient256": [0.0] * 256,
        "base_hold_gradient256": [0.001] * 256,
        "difference_identity_max_absolute": 0.0,
        "difference_identity_relative": 0.0,
        "results": {},
    }
    for arm in ARMS:
        row["results"][arm] = {
            "before_logit": 0.0,
            "after_logit": 0.0,
            "before_probability": 0.5,
            "after_probability": 0.5,
            "before_bce": math.log(2),
            "after_bce": math.log(2),
            "delta_bce": 0.0,
            "risk_gradient_cosine": None,
            "query_action_rms": 0.0,
            "receipt": receipt(gradients[arm], arm=arm),
        }
    meta = {
        "length": 1000,
        "start": 0.0,
        "rate": 1000.0,
        "disruptive": True,
        "down": 0.5,
    }
    return row, meta


def summary_row(shot, query, disruptive, label, delta):
    return {
        "shot": shot,
        "query": query,
        "disruptive": disruptive,
        "label": label,
        "results": {
            arm: {"delta_bce": delta if arm != "difference_hold" else 0.0}
            for arm in ARMS
        },
    }


def test_logaddexp_bce_is_stable_and_label_correct_at_extreme_logits():
    assert stable_bce(1000.0, 0) == pytest.approx(1000.0)
    assert stable_bce(-1000.0, 1) == pytest.approx(1000.0)
    assert stable_bce(-1000.0, 0) == 0.0
    assert stable_bce(1000.0, 1) == 0.0
    assert stable_bce(0.0, 0) == pytest.approx(math.log(2))


def test_independent_summary_uses_per_shot_means_and_safe_guard():
    rows = [
        summary_row(1, 1, True, 1, -0.2),
        summary_row(1, 2, True, 1, -0.4),
        summary_row(1, 3, True, 1, -0.6),
        summary_row(2, 1, True, 1, 0.2),
        summary_row(1, 4, True, 0, 0.6),
        summary_row(10, 1, False, 0, 0.1),
        summary_row(10, 2, False, 0, 0.1),
        summary_row(10, 3, False, 0, 0.1),
        summary_row(11, 1, False, 0, -0.2),
    ]
    summary = recompute_summary(rows)
    real = summary["arms"]["difference_real"]
    assert real["class1"] == {
        "n_points": 4,
        "n_shots": 2,
        "mean_delta_bce": pytest.approx(-0.1),
    }
    assert real["class0"]["mean_delta_bce"] == pytest.approx(1 / 6)
    assert real["safe"]["mean_delta_bce"] == pytest.approx(-0.05)
    assert real["balanced_delta_bce"] == pytest.approx((-0.1 + 1 / 6) / 2)
    assert summary["decision"] == "coverage_insufficient"


def test_valid_sample_and_metadata_are_accepted():
    row, meta = sample()
    verified = validate_sample(row, meta)
    assert verified["results"]["difference_real"]["delta_bce"] == 0.0


def test_wrong_label_time_or_query_out_of_range_is_rejected():
    row, meta = sample()
    row["label"] = 0
    with pytest.raises(ValueError, match="label"):
        validate_sample(row, meta)
    row, meta = sample()
    row["query_time"] += 0.01
    with pytest.raises(ValueError, match="time"):
        validate_sample(row, meta)
    row, meta = sample()
    row["query"] = 1000
    with pytest.raises(ValueError, match="query"):
        validate_sample(row, meta)


def test_delta_bce_tampering_is_detected_from_logits():
    row, meta = sample()
    row["results"]["difference_reverse"]["delta_bce"] = -0.01
    with pytest.raises(ValueError, match="delta_bce"):
        validate_sample(row, meta)


def test_gradient_identity_tampering_is_detected_even_when_norm_is_updated():
    row, meta = sample()
    rec = row["results"]["difference_real"]["receipt"]
    rec["gradient256"][0] += 1e-3
    rec["gradient_norm"] = math.hypot(*rec["gradient256"])
    with pytest.raises(ValueError, match="gradient identity"):
        validate_sample(row, meta)


def test_gradient_identity_accepts_float32_near_zero_cancellation():
    row, meta = sample()
    row["base_hold_gradient256"] = [0.003] * 256
    rec = row["results"]["difference_real"]["receipt"]
    rec["gradient256"] = [1e-7] * 256
    rec["gradient_norm"] = math.hypot(*rec["gradient256"])
    # The relative discrepancy is large; the prescribed absolute tolerance applies.
    row["difference_identity_max_absolute"] = 1e-7
    row["difference_identity_relative"] = math.hypot(*rec["gradient256"]) / 1e-30
    validate_sample(row, meta)


def test_nonzero_hold_and_nonzero_rejected_state_are_detected():
    row, meta = sample()
    rec = row["results"]["difference_hold"]["receipt"]
    rec["gradient256"][0] = 1e-4
    rec["gradient_norm"] = 1e-4
    with pytest.raises(ValueError, match="hold"):
        validate_sample(row, meta)
    row, meta = sample()
    rec = row["results"]["difference_reverse"]["receipt"]
    rec["state256_after"][0] = 0.1
    rec["step_norm"] = 0.1
    with pytest.raises(ValueError, match="rejected"):
        validate_sample(row, meta)


def test_hold_objective_and_recorded_gradient_identity_diagnostics_are_checked():
    row, meta = sample()
    rec = row["results"]["difference_hold"]["receipt"]
    rec.update(loss_before=1.0, loss_proposal=1.0, loss_retained=1.0)
    with pytest.raises(ValueError, match="hold"):
        validate_sample(row, meta)
    row, meta = sample()
    row["difference_identity_max_absolute"] = 0.1
    with pytest.raises(ValueError, match="identity"):
        validate_sample(row, meta)


def test_strict_descent_and_gradient_norm_are_checked():
    row, meta = sample()
    row["results"]["difference_reverse"]["receipt"]["accepted"] = True
    with pytest.raises(ValueError, match="descent"):
        validate_sample(row, meta)
    row, meta = sample()
    row["results"]["difference_reverse"]["receipt"]["gradient_norm"] = 999.0
    with pytest.raises(ValueError, match="gradient_norm"):
        validate_sample(row, meta)


def test_no_observation_rms_none_is_specific_to_difference_receipts():
    row, meta = sample()
    row["observed"] = 0
    row["base_hold_gradient256"] = [0.0] * 256
    for arm in ARMS:
        row["results"][arm]["receipt"] = receipt(0.0, observed=0, arm=arm)
    validate_sample(row, meta)
    row["results"]["difference_real"]["receipt"]["max_action_rms"] = 0.0
    with pytest.raises(ValueError, match="max_action_rms"):
        validate_sample(row, meta)


def test_selection_membership_rejects_out_of_scope_and_duplicates():
    row, _ = sample()
    selection = [
        {
            "shot": 11,
            "disruptive": True,
            "selected": [
                {"query": 400, "requested_lead_ms": 100, "actual_lead_ms": 100.0}
            ],
            "skipped": [],
        }
    ]
    validate_membership([row], selection, [11])
    wrong = deepcopy(row)
    wrong["shot"] = 12
    with pytest.raises(ValueError, match="shot"):
        validate_membership([wrong], selection, [11])
    with pytest.raises(ValueError, match="duplicate"):
        validate_membership([row, deepcopy(row)], selection, [11])


def test_sampling_is_rebuilt_from_time_metadata_not_raw_signals():
    _, meta = sample()
    selected, skipped = verifier.expected_sampling(meta)
    assert [item["query"] for item in selected] == [250, 350, 400, 440, 465]
    assert not skipped
    safe = {**meta, "disruptive": False, "down": None}
    selected, skipped = verifier.expected_sampling(safe)
    assert [item["query"] for item in selected] == [160, 369, 579, 789, 999]
    assert not skipped
    assert verifier.expected_sampling({**safe, "length": 160}) == (
        [],
        [{"reason": "insufficient_prefix"}],
    )


def test_missing_hdf5_dependency_reports_use_of_rl_env_without_opening_file(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ValueError, match="requires h5py; use rl_env"):
        verifier.read_metadata("no-file-opened.hdf5")


def test_accepted_state_requires_strict_descent_and_normalized_gradient_step():
    row, meta = sample()
    rec = row["results"]["difference_reverse"]["receipt"]
    rec.update(
        accepted=True,
        reason="improved",
        loss_proposal=0.9,
        loss_retained=0.9,
        state256_after=[-0.00625] * 256,
        step_norm=0.1,
    )
    validate_sample(row, meta)
    rec["state256_after"] = [0.00625] * 256
    with pytest.raises(ValueError, match="direction"):
        validate_sample(row, meta)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_gradient_is_rejected(bad):
    row, meta = sample()
    row["results"]["difference_real"]["receipt"]["gradient256"][5] = bad
    with pytest.raises(ValueError, match="gradient256"):
        validate_sample(row, meta)


def test_gate_recomputation_checks_each_hold_point_and_safe_mean():
    rows = [summary_row(100 + shot, 0, True, 1, -4e-5) for shot in range(6)]
    rows += [summary_row(200 + shot, 0, False, 0, 0.0) for shot in range(8)]
    for row in rows:
        row["results"]["centered_real"]["delta_bce"] = 0.0
        row["results"]["difference_reverse"]["delta_bce"] = 0.0
    assert recompute_summary(rows)["gates"]["passed"] is True
    rows[0]["results"]["difference_hold"]["delta_bce"] = 2e-10
    rows[1]["results"]["difference_hold"]["delta_bce"] = -2e-10
    result = recompute_summary(rows)
    assert result["arms"]["difference_hold"]["class1"]["mean_delta_bce"] == 0
    assert result["gates"]["hold_zero"] is False
    assert result["decision"] == "implementation_check_failed"


@pytest.fixture
def synthetic_probe(monkeypatch):
    """Exercise orchestration entirely in memory: no fixture files or HDF5."""
    ids = list(range(1001, 1021))
    meta = {
        "length": 161,
        "start": 0.0,
        "rate": 1000.0,
        "disruptive": False,
        "down": None,
    }
    selected = {"query": 160, "requested_lead_ms": None, "actual_lead_ms": None}
    rows, sampling = [], []
    for sid in ids:
        row, _ = sample()
        row.update(
            shot=sid,
            query=160,
            disruptive=False,
            label=0,
            query_time=0.16,
            requested_lead_ms=None,
            actual_lead_ms=None,
            origin=134,
            target_first=135,
            target_last=159,
            first_affected=160,
        )
        rows.append(row)
        sampling.append(
            {
                "shot": sid,
                "disruptive": False,
                "selected": [dict(selected)],
                "skipped": [],
            }
        )
    hashes = {str(sid): f"raw-{sid}" for sid in ids}
    protocol = {
        "scope": "CONDITIONAL_ENDED20_LOCAL_PROBE_NOT_ONLINE",
        "arms": list(ARMS),
        "ids": ids,
        "raw_signal_ids_allowed": ids,
        "source_checkpoint": str(verifier.SOURCE),
        "source_sha256": verifier.SOURCE_SHA256,
        "cold_state": str(verifier.COLD),
        "cold_state_sha256": "cold-hash",
        "calibration_reference": str(verifier.CALIBRATION),
        "input_map_sha256": "map-hash",
        "basis_sha256": "basis-hash",
        "input_hashes": hashes,
        "reference": str(verifier.ROOT / "probe_outputs/lora_stream_v1"),
        "reference_completed_sha256": verifier.REFERENCE_SHA256,
        "outcome_authority": "HDF5",
        "east_horizon_ms": 125,
        "strict_lead_ms": 30,
        "label_used_for_update": False,
        "query_selection_uses_ended_outcomes": True,
        "reset_each_local_sample": True,
        "step_size": 0.1,
        "anchor": 0.01,
        "radius": 0.25,
    }
    done = {
        "status": "COMPLETED_ENDED20_LOCAL_PROBE",
        "raw_signal_ids_read": ids,
        "raw_outcome_ids_read": ids,
        "n_shots": 20,
        "n_samples": len(rows),
        "suffix_raw_signals_read": 0,
        "source_tensors_unchanged": True,
        "online_performance_measured": False,
        "independent_future_confirmation": False,
        "samples_sha256": "sample-hash",
        "sampling_sha256": "sampling-hash",
        "summary": recompute_summary(rows),
    }
    payloads = {
        "completed.json": done,
        "protocol.json": protocol,
        "input_map.json": {"calibration_ids": ids},
        "samples.json": rows,
        "sampling.json": sampling,
    }
    files = {
        "source.pt": verifier.SOURCE_SHA256,
        "cold.pt": "cold-hash",
        "input_map.json": "map-hash",
        "source_basis.pt": "basis-hash",
        "samples.json": "sample-hash",
        "sampling.json": "sampling-hash",
        "completed.json": "completed-hash",
        "protocol.json": "protocol-hash",
    }
    reads = []

    def fake_hash(path):
        if path.suffix == ".hdf5":
            assert int(path.stem) in ids
            return hashes[path.stem]
        return files[path.name]

    def fake_metadata(path):
        reads.append(int(path.stem))
        return meta

    monkeypatch.setattr(verifier, "read_json", lambda path: payloads[path.name])
    monkeypatch.setattr(verifier, "sha256_file", fake_hash)
    monkeypatch.setattr(verifier, "check_snapshot", lambda _protocol: "snapshot-hash")
    monkeypatch.setattr(verifier, "read_metadata", fake_metadata)
    return payloads, reads


def test_full_in_memory_probe_accepts_twenty_not_one_hundred_samples(synthetic_probe):
    _, reads = synthetic_probe
    report = verifier.verify_probe()
    assert report["status"] == "PASS"
    assert report["n_shots"] == report["n_samples"] == 20
    assert report["summary_recomputed"]["decision"] == "coverage_insufficient"
    assert len(reads) == 20


def test_out_of_scope_sample_fails_before_any_metadata_reads(synthetic_probe):
    payloads, reads = synthetic_probe
    payloads["samples.json"][0]["shot"] = 999999
    with pytest.raises(ValueError, match="shot"):
        verifier.verify_probe()
    assert reads == []


def test_incomplete_status_stops_before_any_metadata_reads(synthetic_probe):
    payloads, reads = synthetic_probe
    payloads["completed.json"]["status"] = "RUNNING"
    with pytest.raises(ValueError, match="not completed"):
        verifier.verify_probe()
    assert reads == []


def test_hash_mismatch_stops_before_any_metadata_reads(synthetic_probe, monkeypatch):
    _, reads = synthetic_probe
    original = verifier.sha256_file
    monkeypatch.setattr(
        verifier,
        "sha256_file",
        lambda path: "tampered" if path.name == "samples.json" else original(path),
    )
    with pytest.raises(ValueError, match="samples hash"):
        verifier.verify_probe()
    assert reads == []


@pytest.mark.parametrize("tamper", [None, "zip", "current"])
def test_source_snapshot_checks_both_zip_and_current_hashes_in_memory(
    monkeypatch, tamper
):
    names = [
        verifier.PLAN,
        f"{verifier.MODULE}/__init__.py",
        f"{verifier.MODULE}/run.py",
        f"{verifier.MODULE}/core.py",
        f"{verifier.MODULE}/summary.py",
    ]
    contents = {name: f"synthetic {name}".encode() for name in names}
    digests = {
        name: hashlib.sha256(value).hexdigest() for name, value in contents.items()
    }
    protocol = {"code_sha256": digests, "plan_sha256": digests[verifier.PLAN]}
    archived = dict(contents)
    if tamper == "zip":
        archived[names[1]] = b"modified archived code"
    buffer = io.BytesIO()
    actual_zipfile = zipfile.ZipFile
    with actual_zipfile(buffer, "w") as archive:
        for name, value in archived.items():
            archive.writestr(name, value)
    zip_bytes = buffer.getvalue()

    def fake_sha(path):
        if path.name == "code_snapshot.zip":
            return "archive-hash"
        name = path.relative_to(verifier.ROOT.resolve()).as_posix()
        value = (
            b"modified current code"
            if tamper == "current" and name == names[1]
            else contents[name]
        )
        return hashlib.sha256(value).hexdigest()

    monkeypatch.setattr(verifier, "sha256_file", fake_sha)
    monkeypatch.setattr(
        verifier.zipfile,
        "ZipFile",
        lambda _path: actual_zipfile(io.BytesIO(zip_bytes), "r"),
    )
    if tamper:
        expected = "snapshot" if tamper == "zip" else "current"
        with pytest.raises(ValueError, match=f"{expected}.*hash"):
            verifier.check_snapshot(protocol)
    else:
        assert verifier.check_snapshot(protocol) == "archive-hash"


def test_cli_defaults_to_stdout_without_opening_report(monkeypatch, capsys):
    monkeypatch.setattr(verifier, "verify_probe", lambda: {"status": "PASS"})

    def no_file(*_args, **_kwargs):
        raise AssertionError("default CLI must not create a report")

    monkeypatch.setattr(verifier.Path, "open", no_file)
    assert verifier.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "PASS"}


def test_cli_write_report_uses_fixed_path_and_exclusive_create(monkeypatch, capsys):
    result = {"status": "PASS", "summary_recomputed": {"decision": "screen_failed"}}
    monkeypatch.setattr(verifier, "verify_probe", lambda: result)
    written = []

    class MemoryReport(io.StringIO):
        def __exit__(self, *args):
            written.append(self.getvalue())
            return super().__exit__(*args)

    def fake_open(path, mode, *, encoding):
        assert path == verifier.OUTPUT / "verification_records.json"
        assert mode == "x" and encoding == "utf-8"
        return MemoryReport()

    monkeypatch.setattr(verifier.Path, "open", fake_open)
    assert verifier.main(["--write-report"]) == 0
    assert len(written) == 1
    assert json.loads(written[0]) == result
    assert json.loads(capsys.readouterr().out) == result


def test_cli_refuses_to_overwrite_existing_report(monkeypatch, capsys):
    monkeypatch.setattr(verifier, "verify_probe", lambda: {"status": "PASS"})

    def existing(path, mode, *, encoding):
        assert path == verifier.OUTPUT / "verification_records.json"
        assert mode == "x" and encoding == "utf-8"
        raise FileExistsError("report already exists")

    monkeypatch.setattr(verifier.Path, "open", existing)
    assert verifier.main(["--write-report"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FAIL" and "already exists" in output["error"]


def test_cli_does_not_write_a_report_when_verification_fails(monkeypatch, capsys):
    def fail():
        raise ValueError("verification failed")

    def no_file(*_args, **_kwargs):
        raise AssertionError("failed verification must not open a report")

    monkeypatch.setattr(verifier, "verify_probe", fail)
    monkeypatch.setattr(verifier.Path, "open", no_file)
    assert verifier.main(["--write-report"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
