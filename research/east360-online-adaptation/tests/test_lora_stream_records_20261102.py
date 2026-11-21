"""Synthetic, in-memory tests; no live experiment files or GPUs are read."""

import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from experiments.risk_aligned_online.lora_stream_20261102 import verify_records as audit


ARMS = ("cold", "real", "reverse", "hold")
LEARNERS = ARMS[1:]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(**arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def record(*, observed=False):
    state = [0.0] * 256
    after, gradient = state.copy(), state.copy()
    if observed:
        after[0], gradient[0] = float(np.float32(-0.1)), 1.0
    return dict(
        origin=134,
        target_first=135,
        target_last=159,
        risk_sealed_through=159,
        first_affected=160,
        forecast_sha256="a" * 64,
        mask_sha256="b" * 64,
        target_sha256="c" * 64,
        state256_before=state,
        state256_after=after,
        gradient256=gradient,
        gradient_norm=float(observed),
        step_norm=abs(after[0]),
        observed=500 if observed else 0,
        accepted=observed,
        loss_before=1.0 if observed else None,
        loss_proposal=0.9 if observed else None,
        loss_retained=0.9 if observed else None,
        reason="improved" if observed else "no_target",
        gradient_computed=observed,
        max_action_rms=0.01 if observed else 0.0,
    )


def info(*, warmup=False, observed=False):
    return dict(
        reset_each_shot=True,
        temporary_parameters=256,
        updates={a: [] if warmup else [record(observed=observed)] for a in LEARNERS},
        action_statistics={
            a: dict(endpoints=0 if warmup else 161, capped_endpoints=0, max_rms=0.0)
            for a in ARMS
        },
    )


def toy_metrics(rows):
    disruptive = sum(row["disruptive"] for row in rows)
    safe = len(rows) - disruptive
    tp = sum(row["timely"] for row in rows)
    fp = sum(row["alarm"] and not row["disruptive"] for row in rows)
    return dict(
        n=len(rows),
        n_disrupt=disruptive,
        n_safe=safe,
        tp=tp,
        fp=fp,
        tpr=tp / disruptive if disruptive else None,
        fpr=fp / safe if safe else None,
        J=tp / disruptive - fp / safe if disruptive and safe else None,
        early_over3000=sum(row["timely"] and row["lead_ms"] > 3000 for row in rows),
    )


class MemoryFiles:
    def __init__(self):
        self.files = {}

    def put(self, path, data):
        self.files[Path(path)] = data

    def json_put(self, path, data):
        self.put(path, json.dumps(data, allow_nan=False).encode())

    def read(self, path):
        return self.files[Path(path)]

    def json(self, path):
        return json.loads(self.read(path))


@pytest.fixture
def ledger(monkeypatch):
    """A complete 360-case ledger with four known flips, entirely in RAM."""
    fs = MemoryFiles()
    root = Path("/virtual/fuison")
    output, reference, calibration = (
        root / "output",
        root / "reference",
        root / "calibration",
    )
    source, cold = root / "source.pt", root / "cold.pt"
    fs.put(source, b"source")
    fs.put(cold, b"cold")
    fs.put(output / "source_basis.pt", b"basis")
    fs.put(reference / "completed.json", b"sealed reference")
    order = [dict(shot=10000 + i) for i in range(360)]
    fs.json_put(
        calibration / "input_map.json",
        {"calibration_ids": [r["shot"] for r in order[:20]]},
    )
    source_hash = digest(fs.read(source))
    reference_hash = digest(fs.read(reference / "completed.json"))
    names = [
        "LORA_STREAM_PLAN_20261102.md",
        "experiments/risk_aligned_online/lora_stream_20261102/core.py",
        "experiments/risk_aligned_online/lora_stream_20261102/run.py",
    ]
    code = {name: ("source:" + name).encode() for name in names}
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as output_zip:
        for name, data in code.items():
            output_zip.writestr(name, data)
            fs.put(root / name, data)
    fs.put(output / "code_snapshot.zip", zipped.getvalue())
    mapping_hash = digest(fs.read(calibration / "input_map.json"))
    ref_protocol = dict(
        order=order,
        source_checkpoint=str(source),
        source_sha256=source_hash,
        cold_state=str(cold),
        cold_state_sha256=digest(fs.read(cold)),
        input_map_sha256=mapping_hash,
        threshold=0.5,
        reference=str(calibration),
    )
    fs.json_put(reference / "protocol.json", ref_protocol)
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_SHOT_RESET_FFN_GRADIENT",
        order=order,
        arms=list(ARMS),
        reference=str(reference),
        reference_completed_sha256=reference_hash,
        calibration_reference=str(calibration),
        source_checkpoint=str(source),
        source_sha256=source_hash,
        cold_state=str(cold),
        cold_state_sha256=digest(fs.read(cold)),
        input_map_sha256=mapping_hash,
        basis_sha256=digest(fs.read(output / "source_basis.pt")),
        threshold=0.5,
        warmup20_cost_included=True,
        outcome_authority="HDF5",
        temporary_parameters=256,
        persistent_neural_updates=0,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        forecast_horizon=25,
        step_size=0.1,
        parameter_anchor=0.01,
        endpoint_hidden_rms_cap=0.25,
        cap_uses_only_current_window=True,
        forecast_objective="observed_temporal_centered_SmoothL1",
        no_label_input_to_predict=True,
        plan_sha256=digest(code[names[0]]),
        code_sha256={name: digest(data) for name, data in code.items()},
    )
    fs.json_put(output / "protocol.json", protocol)
    times = np.arange(161, dtype=np.float64) / 1000
    empty_curve = np.full(161, 0.1, dtype=np.float32)
    hit_curve = empty_curve.copy()
    hit_curve[160] = 0.9
    stored_curves = {}
    for cold_hit, real_hit in ((False, False), (True, False), (False, True)):
        cold_curve = hit_curve if cold_hit else empty_curve
        real_curve = hit_curve if real_hit else empty_curve
        stored_curves[cold_hit, real_hit] = (
            archive(
                times=times,
                cold=cold_curve,
                real=real_curve,
                reverse=cold_curve,
                hold=cold_curve,
            ),
            archive(times=times, cold=cold_curve, previous_real=cold_curve),
        )
    warm_log = json.dumps(info(warmup=True)).encode()
    live_log = json.dumps(info()).encode()
    rows = {arm: [] for arm in ARMS}
    raw_hashes, outcomes = {}, {}
    for index, item in enumerate(order):
        shot = item["shot"]
        prefix = f"{index + 1:04d}_{shot}"
        disruptive = index % 2 == 0
        cold_hit, real_hit = index in (21, 22), index in (20, 23)
        curve_bytes, reference_bytes = stored_curves[cold_hit, real_hit]
        fs.put(output / "shots" / f"{prefix}.npz", curve_bytes)
        fs.put(reference / "shots" / f"{prefix}.npz", reference_bytes)
        log_bytes = warm_log if index < 20 else live_log
        fs.put(output / "shots" / f"{prefix}_updates.json", log_bytes)
        fs.json_put(
            output / "shots" / f"{prefix}_seal.json",
            dict(
                shot=shot,
                index=index + 1,
                outcome_available=False,
                warmup_curves_reused=index < 20,
                sha256=digest(curve_bytes),
                updates_sha256=digest(log_bytes),
            ),
        )
        raw = root / "data/EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        raw_bytes = f"synthetic raw {shot}".encode()
        fs.put(raw, raw_bytes)
        raw_hashes[str(shot)] = digest(raw_bytes)
        outcomes[raw] = (disruptive, 0.2 if disruptive else None)
        for arm in ARMS:
            alarm = real_hit if arm == "real" else cold_hit
            rows[arm].append(
                dict(
                    shot=shot,
                    disruptive=disruptive,
                    alarm=alarm,
                    lead_ms=40.0 if disruptive and alarm else None,
                    timely=bool(disruptive and alarm),
                )
            )
    fs.json_put(calibration / "input_hashes.json", raw_hashes)
    fs.json_put(output / "per_shot.json", rows)
    metrics = {
        arm: {
            scope: toy_metrics(part)
            for scope, part in (
                ("full360", rows[arm]),
                ("warmup20", rows[arm][:20]),
                ("suffix340", rows[arm][20:]),
            )
        }
        for arm in ARMS
    }
    no_flips = dict(rescued_D=[], lost_D=[], removed_FP=[], added_FP=[])
    flips = {arm: copy.deepcopy(no_flips) for arm in LEARNERS}
    flips["real"] = dict(
        rescued_D=[10020], lost_D=[10022], removed_FP=[10021], added_FP=[10023]
    )
    totals = {
        arm: dict(events=340, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for arm in LEARNERS
    }
    actions = {
        arm: dict(endpoints=340 * 161, capped_endpoints=0, max_rms=0.0) for arm in ARMS
    }
    fs.json_put(
        output / "completed.json",
        dict(
            status="COMPLETED",
            metrics=metrics,
            flips=flips,
            update_statistics=totals,
            action_statistics=actions,
            fresh_curves=1360,
            reused_warmup_curves=80,
            source_state_unchanged=True,
            operational_gate=False,
            control_separation_gate=False,
            method_promoted=False,
            scientific_goal_achieved=False,
            independent_future_confirmation=False,
        ),
    )
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "SOURCE_SHA256", source_hash)
    monkeypatch.setattr(audit, "REFERENCE_COMPLETED_SHA256", reference_hash)
    monkeypatch.setattr(audit, "read_bytes", fs.read)
    monkeypatch.setattr(audit, "read_json", fs.json)
    monkeypatch.setattr(audit, "sha256_file", lambda path: digest(fs.read(path)))
    monkeypatch.setattr(audit, "read_outcome", lambda path: outcomes[Path(path)])
    return fs, output


def test_strict_30_and_3000_boundaries():
    times, probability = np.array([0.0]), np.array([0.9])
    at_30 = audit.evaluate_curve(times, probability, True, 0.03, 0.5)
    after_30 = audit.evaluate_curve(times, probability, True, 0.030000001, 0.5)
    at_3000 = audit.evaluate_curve(times, probability, True, 3.0, 0.5)
    after_3000 = audit.evaluate_curve(times, probability, True, 3.000001, 0.5)
    assert at_30["lead_ms"] == 30 and not at_30["timely"]
    assert after_30["timely"]
    rows = [dict(disruptive=True, **at_3000), dict(disruptive=True, **after_3000)]
    assert audit.aggregate(rows)["early_over3000"] == 1


def test_first_alarm_is_used_and_postterminal_crossings_are_ignored():
    times = np.array([0.0, 0.02, 0.10, 0.13])
    scores = np.array([0.1, 0.9, 0.9, 0.9])
    result = audit.evaluate_curve(times, scores, True, 0.12, 0.5)
    assert result == dict(alarm=True, lead_ms=100.0, timely=True)
    scores[:3] = 0.1
    assert audit.evaluate_curve(times, scores, True, 0.12, 0.5) == dict(
        alarm=False, lead_ms=None, timely=False
    )


def test_safe_prefix_false_alarm_cannot_be_counted_as_removed():
    times = np.arange(161) / 1000
    cold = np.full(161, 0.1)
    cold[5] = 0.9
    candidate = cold.copy()
    candidate[160] = 0.1
    first = dict(
        shot=1, disruptive=False, **audit.evaluate_curve(times, cold, False, None, 0.5)
    )
    second = dict(
        shot=1,
        disruptive=False,
        **audit.evaluate_curve(times, candidate, False, None, 0.5),
    )
    assert first["alarm"] and second["alarm"]
    assert audit.compare_flips([first], [second])["removed_FP"] == []


def test_complete_synthetic_360_ledger(ledger):
    _, output = ledger
    report = audit.verify(output)
    assert report["status"] == "PASS"
    assert report["cases"] == 360
    assert report["flips"]["real"] == dict(
        rescued_D=[10020], lost_D=[10022], removed_FP=[10021], added_FP=[10023]
    )
    assert report["metrics"]["real"]["full360"]["n_safe"] == 180


def test_incomplete_refuses_before_reading_any_other_file(monkeypatch):
    def only_completion(path):
        assert Path(path).name == "completed.json"
        return {"status": "RUNNING"}

    monkeypatch.setattr(audit, "read_json", only_completion)
    monkeypatch.setattr(audit, "read_bytes", lambda _: pytest.fail("premature read"))
    monkeypatch.setattr(
        audit, "read_outcome", lambda _: pytest.fail("premature outcome")
    )
    with pytest.raises(ValueError, match="complet"):
        audit.verify(Path("/virtual/output"))


@pytest.mark.parametrize(
    "target",
    ["source.pt", "cold", "basis", "input_map", "raw", "curve", "log", "archive"],
)
def test_hash_tampering_is_rejected(ledger, target):
    fs, output = ledger
    if target == "source.pt":
        path = Path("/virtual/fuison/source.pt")
    elif target == "cold":
        path = Path("/virtual/fuison/cold.pt")
    elif target == "basis":
        path = output / "source_basis.pt"
    elif target == "input_map":
        path = Path("/virtual/fuison/calibration/input_map.json")
    elif target == "raw":
        path = Path("/virtual/fuison/data/EAST/unified_hdf5/10000/10000.hdf5")
    elif target == "archive":
        path = output / "code_snapshot.zip"
        original = zipfile.ZipFile(io.BytesIO(fs.read(path)))
        changed = io.BytesIO()
        with original, zipfile.ZipFile(changed, "w") as altered:
            for name in original.namelist():
                altered.writestr(name, original.read(name) + b"tamper")
        fs.put(path, changed.getvalue())
    else:
        suffix = ".npz" if target == "curve" else "_updates.json"
        path = output / "shots" / ("0001_10000" + suffix)
    if target != "archive":
        fs.put(path, fs.read(path) + b"tamper")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        audit.verify(output)


def test_resealed_clock_tampering_is_rejected(ledger):
    fs, output = ledger
    prefix = "0021_10020"
    log_path = output / "shots" / f"{prefix}_updates.json"
    log = fs.json(log_path)
    log["updates"]["real"][0]["first_affected"] = 159
    fs.json_put(log_path, log)
    seal_path = output / "shots" / f"{prefix}_seal.json"
    seal = fs.json(seal_path)
    seal["updates_sha256"] = digest(fs.read(log_path))
    fs.json_put(seal_path, seal)
    with pytest.raises(ValueError, match="clock"):
        audit.verify(output)


@pytest.mark.parametrize(
    "prefix,index,message",
    [
        ("0021_10020", 10, "prefix160"),
        ("0001_10000", 160, "warmup reuse"),
    ],
)
def test_resealed_prefix_or_warmup_curve_tampering_is_rejected(
    ledger, prefix, index, message
):
    fs, output = ledger
    curve_path = output / "shots" / f"{prefix}.npz"
    with np.load(io.BytesIO(fs.read(curve_path)), allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["real"][index] = 0.9
    fs.put(curve_path, archive(**arrays))
    seal_path = output / "shots" / f"{prefix}_seal.json"
    seal = fs.json(seal_path)
    seal["sha256"] = digest(fs.read(curve_path))
    fs.json_put(seal_path, seal)
    with pytest.raises(ValueError, match=message):
        audit.verify(output)


@pytest.mark.parametrize(
    "field", ["full360", "warmup20", "suffix340", "flip", "updates", "actions"]
)
def test_wrong_completed_statistics_are_rejected(ledger, field):
    fs, output = ledger
    path = output / "completed.json"
    done = fs.json(path)
    if field == "flip":
        done["flips"]["real"]["removed_FP"] = []
    elif field == "updates":
        done["update_statistics"]["real"]["accepted"] += 1
    elif field == "actions":
        done["action_statistics"]["real"]["capped_endpoints"] += 1
    else:
        done["metrics"]["real"][field]["n_safe"] += 1
    fs.json_put(path, done)
    with pytest.raises(ValueError, match="metric|flip|statistics"):
        audit.verify(output)


def test_update_checks_norms_acceptance_and_common_support():
    valid = info(observed=True)
    audit.check_log(valid, 161, warmup=False)
    invalid = copy.deepcopy(valid)
    invalid["updates"]["real"][0]["gradient_norm"] = 2
    with pytest.raises(ValueError, match="gradient"):
        audit.check_log(invalid, 161, warmup=False)
    invalid = copy.deepcopy(valid)
    invalid["updates"]["reverse"][0]["loss_proposal"] = 1.0
    with pytest.raises(ValueError, match="accept"):
        audit.check_log(invalid, 161, warmup=False)
    invalid = copy.deepcopy(valid)
    invalid["updates"]["hold"][0]["mask_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="mask"):
        audit.check_log(invalid, 161, warmup=False)
    invalid = copy.deepcopy(valid)
    invalid["updates"]["real"][0]["step_norm"] = 0.0
    with pytest.raises(ValueError, match="step norm"):
        audit.check_log(invalid, 161, warmup=False)


def test_unobserved_target_must_not_change_state():
    invalid = info()
    invalid["updates"]["real"][0]["state256_after"][0] = 1
    with pytest.raises(ValueError):
        audit.check_log(invalid, 161, warmup=False)


def test_state_must_start_at_zero_and_continue_within_shot():
    valid = info(observed=True)
    for arm in LEARNERS:
        previous = valid["updates"][arm][0]["state256_after"]
        second = record(observed=True)
        second.update(
            origin=262,
            target_first=263,
            target_last=287,
            risk_sealed_through=287,
            first_affected=288,
            state256_before=previous.copy(),
            state256_after=previous.copy(),
            gradient256=[0.0] * 256,
            gradient_norm=0.0,
            step_norm=0.0,
            accepted=False,
            loss_before=1.0,
            loss_proposal=1.0,
            loss_retained=1.0,
            reason="not_improved",
        )
        valid["updates"][arm].append(second)
    for stats in valid["action_statistics"].values():
        stats["endpoints"] = 289
    audit.check_log(valid, 289, warmup=False)
    invalid = copy.deepcopy(valid)
    invalid["updates"]["hold"][1]["state256_before"][0] += 0.1
    with pytest.raises(ValueError, match="continuity"):
        audit.check_log(invalid, 289, warmup=False)
    invalid = copy.deepcopy(valid)
    invalid["updates"]["real"][0]["state256_before"][0] = 1.0
    with pytest.raises(ValueError, match="continuity"):
        audit.check_log(invalid, 289, warmup=False)


def test_write_report_refuses_existing_file(monkeypatch):
    def exists(_, mode, **kwargs):
        assert mode == "x"
        raise FileExistsError("already exists")

    monkeypatch.setattr(Path, "open", exists)
    with pytest.raises(FileExistsError):
        audit.write_report(Path("/virtual/output"), {"status": "PASS"})
