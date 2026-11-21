import hashlib
import json

import h5py
import numpy as np
import pytest

from experiments.risk_aligned_online.east_decoder_calibration_20261102 import (
    audit360 as audit,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def record(end, before=0.0, after=0.0, *, observed=10, accepted=False):
    gradient = np.zeros(256, dtype=np.float32)
    gradient[0] = 0.25 if observed else 0.0
    state = np.zeros(256, dtype=np.float32)
    state[0] = after
    return dict(
        origin=end - 25,
        maturity=end,
        sealed_through=end,
        first_affected=end + 1,
        forecast_sha256="a" * 64,
        mask_sha256="b" * 64,
        target_sha256="c" * 64,
        observed=observed,
        accepted=accepted,
        reason="improved" if accepted else "not_improved" if observed else "no_target",
        loss_before=1.0 if observed else None,
        loss_proposal=(0.9 if accepted else 1.0) if observed else None,
        gradient_norm=0.25 if observed else 0.0,
        step_norm=float(abs(np.float32(after) - np.float32(before))),
        gradient256=gradient.tolist(),
        state256_after=state.tolist(),
    )


def updates():
    return {
        arm: [record(159, 0, 0.125, accepted=True), record(287, 0.125, 0.125)]
        for arm in audit.ARMS[1:]
    }


def test_classification_first_preterminal_crossing_strict_boundaries():
    t = np.array([0.0, 0.999, 1.0, 3.969, 3.970, 3.971, 4.001])
    for idx, lead, timely in ((0, 4000, True), (1, 3001, True), (2, 3000, True),
                              (3, 31, True), (4, 30, False), (5, 29, False)):
        curve = np.zeros(len(t))
        curve[idx:] = 0.5
        row = audit.classify_curve(t, curve, True, 4.0, 0.5)
        assert row == {"alarm": True, "lead_ms": float(lead), "timely": timely}
    assert not audit.classify_curve(t, t == 4.001, True, 4.0, 0.5)["alarm"]
    assert audit.classify_curve(t, t == 4.001, False, None, 0.5)["alarm"]


def test_recount_and_flips_include_late_disruptions():
    before = [
        dict(shot=1, disruptive=True, alarm=True, timely=True, lead_ms=3001.0),
        dict(shot=2, disruptive=True, alarm=True, timely=False, lead_ms=30.0),
        dict(shot=3, disruptive=False, alarm=True, timely=False, lead_ms=None),
        dict(shot=4, disruptive=False, alarm=False, timely=False, lead_ms=None),
    ]
    after = [
        dict(shot=1, disruptive=True, alarm=True, timely=False, lead_ms=30.0),
        dict(shot=2, disruptive=True, alarm=True, timely=True, lead_ms=3000.0),
        dict(shot=3, disruptive=False, alarm=False, timely=False, lead_ms=None),
        dict(shot=4, disruptive=False, alarm=True, timely=False, lead_ms=None),
    ]
    result = audit.recount(before)
    assert (result["tp"], result["fp"], result["fpr"], result["J"]) == (1, 1, .5, 0)
    assert result["early_over3000"] == 1
    assert result["median_timely_lead_ms"] == 3001
    assert audit.recount(after)["early_over3000"] == 0
    assert audit.paired_flips(before, after) == dict(
        rescued_D=[2], lost_D=[1], removed_FP=[3], added_FP=[4]
    )


def test_receipts_validate_clock_acceptance_norms_and_cross_event_state():
    result = audit.check_updates(updates(), 289, False)
    assert all(v["events"] == 2 and v["accepted"] == 1 for v in result.values())
    assert all(v["max_state_norm"] == .125 and v["max_gradient_norm"] == .25
               and v["max_step_norm"] == .125 for v in result.values())
    no_target = {a: [record(159, observed=0)] for a in audit.ARMS[1:]}
    assert all(v["accepted"] == 0 for v in audit.check_updates(no_target, 160, False).values())
    assert all(v["events"] == 0 for v in audit.check_updates(
        {a: [] for a in audit.ARMS[1:]}, 1000, True
    ).values())
    with pytest.raises(ValueError, match="clock"):
        audit.check_updates(updates(), 289, True)


@pytest.mark.parametrize("mutation", [
    "missing_event", "extra_event", "clock", "origin", "sealed", "affected",
    "strict_accept", "false_reason", "bad_hash", "mask_mismatch", "target_mismatch",
    "gradient_norm", "step_norm", "state_length", "gradient_nonfinite",
    "rejected_state", "previous_state", "observed_type", "no_target_loss",
])
def test_bad_receipts_are_rejected(mutation):
    data = updates()
    a, b = audit.ARMS[1:3]
    first, second = data[a]
    if mutation == "missing_event":
        data[a].pop()
    elif mutation == "extra_event":
        data[a].append(record(415))
    elif mutation in ("clock", "origin", "sealed", "affected"):
        key = {"clock": "maturity", "origin": "origin", "sealed": "sealed_through",
               "affected": "first_affected"}[mutation]
        first[key] += 1
    elif mutation == "strict_accept":
        first["loss_proposal"] = first["loss_before"]
    elif mutation == "false_reason":
        first["reason"] = "not_improved"
    elif mutation == "bad_hash":
        first["forecast_sha256"] = "f" * 63
    elif mutation == "mask_mismatch":
        data[b][0]["mask_sha256"] = "d" * 64
    elif mutation == "target_mismatch":
        data[b][0]["target_sha256"] = "d" * 64
    elif mutation == "gradient_norm":
        first["gradient_norm"] += 0.01
    elif mutation == "step_norm":
        first["step_norm"] += 0.01
    elif mutation == "state_length":
        first["state256_after"].pop()
    elif mutation == "gradient_nonfinite":
        first["gradient256"][0] = float("nan")
    elif mutation == "rejected_state":
        second["state256_after"][0] += 0.125
        second["step_norm"] = 0.125
    elif mutation == "previous_state":
        first["state256_after"][0] = 0.25
        first["step_norm"] = 0.25
    elif mutation == "observed_type":
        first["observed"] = True
    else:
        data = {a: [record(159, observed=0)] for a in audit.ARMS[1:]}
        data[a][0]["loss_before"] = 0.0
    with pytest.raises((ValueError, AssertionError)):
        audit.check_updates(data, 160 if mutation == "no_target_loss" else 289, False)


def test_curve_prefix_warmup_and_reference_are_checked():
    t = np.arange(200, dtype=np.float64) / 1000
    cold = np.zeros(200, dtype=np.float32)
    curves = {a: cold.copy() for a in audit.ARMS}
    curves["worst_real"][160] = 0.75
    audit.validate_curves(t, curves, t, cold, False)
    with pytest.raises(ValueError, match="warmup"):
        audit.validate_curves(t, curves, t, cold, True)
    curves["worst_real"][1] = 0.75
    with pytest.raises(ValueError, match="160"):
        audit.validate_curves(t, curves, t, cold, False)
    curves["worst_real"][1] = 0
    curves["static"][180] = 0.1
    with pytest.raises(ValueError, match="reference"):
        audit.validate_curves(t, curves, t, cold, False)


def test_incomplete_run_cannot_read_protocol_or_create_report(tmp_path, monkeypatch):
    write(tmp_path / "completed.json", {"status": "RUNNING"})
    monkeypatch.setattr(audit, "REFERENCE", tmp_path / "forbidden")
    with pytest.raises(ValueError, match="COMPLETED"):
        audit.main(tmp_path)
    assert not (tmp_path / "audit_cpu.json").exists()


def score(tp, fp, count, d, s):
    tpr = tp / d if d else None
    fpr = fp / s if s else None
    return dict(n=count, n_disrupt=d, n_safe=s, tp=tp, fp=fp, tpr=tpr, fpr=fpr,
                J=tpr - fpr if d and s else None, early_over3000=0,
                median_timely_lead_ms=88.0 if tp else None)


@pytest.fixture(scope="module")
def synthetic_360(tmp_path_factory):
    base = tmp_path_factory.mktemp("audit360")
    root, reference, cv = (base / name for name in ("run", "reference", "cv"))
    for path in (root / "shots", reference / "shots", cv):
        path.mkdir(parents=True)
    ids = list(range(1, 361))
    order = [{"shot": sid} for sid in ids]
    gate = {"passed": False, "disruptive_improved": False}
    write(cv / "completed.json", {"status": "COMPLETED", "gate": gate})
    write(reference / "protocol.json", dict(order=order, threshold=0.5,
          input_map_sha256="1" * 64, cold_state_sha256="2" * 64))
    write(reference / "completed.json", {"status": "COMPLETED",
                                        "protocol_sha256": sha(reference / "protocol.json")})
    write(reference / "verification_neural.json", {"status": "PASS",
                                                  "completed_sha256": sha(reference / "completed.json")})
    protocol = dict(
        scope="DIAGNOSTIC_COMPLETE360_AFTER_LOCAL_SURROGATE_GATE_FAILED",
        arms=list(audit.ARMS), order=order, threshold=0.5, head_fit_ids=ids[:20],
        local_cv_gate=gate, local_cv_completed_sha256=sha(cv / "completed.json"),
        reference_completed_sha256=sha(reference / "completed.json"),
        input_map_sha256="1" * 64, cold_sha256="2" * 64,
        warmup20_cost_included=True, suffix_supervised_updates=0, suffix_head_updates=0,
        method_promoted=False, independent_confirmation=False,
    )
    write(root / "protocol.json", protocol)
    t = np.arange(289, dtype=np.float64) / 1000
    expected_flips = {k: [] for k in ("rescued_D", "lost_D", "removed_FP", "added_FP")}
    ref_entries = []
    saved = {a: [] for a in audit.ARMS}
    empty = {a: [] for a in audit.ARMS[1:]}
    for index, sid in enumerate(ids, 1):
        hdf = base / "data/EAST/unified_hdf5" / str(sid // 100 * 100) / f"{sid}.hdf5"
        hdf.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf, "w") as handle:
            handle.create_dataset("meta/IsDisrupt", data=int(sid % 2 == 1))
            handle.create_dataset("meta/DownTime", data=.288)
        stem = f"{index:04d}_{sid}"
        static = np.zeros(len(t), dtype=np.float32)
        if sid % 4 in (0, 1):
            static[200:] = 0.75
        curves = {a: static.copy() for a in audit.ARMS}
        if index > 20 and sid % 8 in (0, 1, 2, 3):
            action = {0: "removed_FP", 1: "lost_D", 2: "added_FP", 3: "rescued_D"}[sid % 8]
            expected_flips[action].append(sid)
            curves["worst_real"][200:] = 0.0 if sid % 8 in (0, 1) else 0.75
        for arm in audit.ARMS:
            alarm = bool(curves[arm][200] >= 0.5)
            disruptive = sid % 2 == 1
            saved[arm].append(dict(shot=sid, disruptive=disruptive, alarm=alarm,
                                   lead_ms=88.0 if alarm and disruptive else None,
                                   timely=alarm and disruptive))
        ref_curve = reference / "shots" / f"{stem}.npz"
        np.savez_compressed(ref_curve, times=t, cold=static)
        ref_seal = reference / "shots" / f"{stem}_seal.json"
        write(ref_seal, dict(index=index, shot=sid, sha256=sha(ref_curve),
                             current_outcome_used_in_prediction=False))
        ref_entries.append(dict(index=index, shot=sid, curve_sha256=sha(ref_curve),
                                seal_sha256=sha(ref_seal)))
        curve, rec, seal = (root / "shots" / f"{stem}{suffix}" for suffix in
                            (".npz", "_updates.json", "_seal.json"))
        np.savez_compressed(curve, times=t, **curves)
        write(rec, empty if index <= 20 else updates())
        write(seal, dict(index=index, shot=sid, curve_sha256=sha(curve),
                         updates_sha256=sha(rec), current_outcome_used_in_prediction=False,
                         reused_warmup=index <= 20))
    write(reference / "shot_seals.json", [
        {key: entry[key] for key in ("index", "shot", "seal_sha256")} for entry in ref_entries
    ])
    write(reference / "completed.json", dict(status="COMPLETED",
          protocol_sha256=sha(reference / "protocol.json"),
          shot_seals_sha256=sha(reference / "shot_seals.json")))
    write(reference / "verification_neural.json", dict(status="PASS",
          completed_sha256=sha(reference / "completed.json")))
    protocol["reference_completed_sha256"] = sha(reference / "completed.json")
    write(root / "protocol.json", protocol)
    write(reference / "verification_records.json", dict(
        status="PASS", hashes=dict(completed_sha256=sha(reference / "completed.json"),
                                    protocol_sha256=sha(reference / "protocol.json"),
                                    shot_seals_sha256=sha(reference / "shot_seals.json"),
                                    shot_files=ref_entries)))
    scores = {a: dict(full360=score(90, 90, 360, 180, 180),
                      warmup20=score(5, 5, 20, 10, 10),
                      suffix340=score(85, 85, 340, 170, 170)) for a in audit.ARMS}
    scores["worst_real"]["full360"] = score(90, 89, 360, 180, 180)
    scores["worst_real"]["suffix340"] = score(85, 84, 340, 170, 170)
    flips = {a: {k: [] for k in expected_flips} for a in audit.ARMS[1:]}
    flips["worst_real"] = expected_flips
    write(root / "per_shot.json", saved)
    done = dict(status="COMPLETED", metrics=scores, flips=flips,
                gate=dict(operational=False, controls={a: True for a in
                    ("prior_real", "forecast_real", "worst_reverse", "worst_hold")},
                          passed=False),
                local_gate_remains_failed=True, independent_confirmation=False,
                protocol_sha256=sha(root / "protocol.json"),
                per_shot_sha256=sha(root / "per_shot.json"))
    write(root / "completed.json", done)
    return root, reference, cv, expected_flips


def setup_synthetic(monkeypatch, synthetic_360):
    root, reference, cv, _ = synthetic_360
    monkeypatch.setattr(audit, "REFERENCE", reference)
    monkeypatch.setattr(audit, "CV", cv)
    monkeypatch.setattr(audit, "REFERENCE_SHA", sha(reference / "completed.json"))
    monkeypatch.setattr(audit.io23, "DATA_ROOT", root.parent / "data")
    def no_signal_read(*args, **kwargs):
        raise AssertionError("signal access forbidden")
    monkeypatch.setattr(audit.io23, "read_signals", no_signal_read)
    return root


def test_synthetic_seal_tampering_and_wrong_cohort_fail_closed(monkeypatch, synthetic_360):
    root = setup_synthetic(monkeypatch, synthetic_360)
    receipt = root / "shots/0021_21_updates.json"
    original = receipt.read_bytes()
    try:
        receipt.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="shot seal"):
            audit.main(root)
    finally:
        receipt.write_bytes(original)
    assert not (root / "audit_cpu.json").exists()
    original_outcome = audit.io23.read_outcome
    monkeypatch.setattr(audit.io23, "read_outcome", lambda device, sid:
                        (True, .288) if sid == 2 else original_outcome(device, sid))
    with pytest.raises(ValueError, match="180D/180S"):
        audit.main(root)
    assert not (root / "audit_cpu.json").exists()


def test_synthetic_reference_neural_status_must_bind_same_frozen_run(monkeypatch, synthetic_360):
    root = setup_synthetic(monkeypatch, synthetic_360)
    path = synthetic_360[1] / "verification_neural.json"
    original = path.read_bytes()
    try:
        write(path, {"status": "PASS", "completed_sha256": "0" * 64})
        with pytest.raises(ValueError, match="reference binding"):
            audit.main(root)
    finally:
        path.write_bytes(original)
    assert not (root / "audit_cpu.json").exists()


def test_reference_manifest_is_hash_bound_to_frozen_completion(monkeypatch, synthetic_360):
    root = setup_synthetic(monkeypatch, synthetic_360)
    path = synthetic_360[1] / "shot_seals.json"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="reference binding"):
            audit.main(root)
    finally:
        path.write_bytes(original)
    assert not (root / "audit_cpu.json").exists()


def test_cv_gate_cannot_be_relabelled_as_passed(monkeypatch, synthetic_360, tmp_path):
    original = setup_synthetic(monkeypatch, synthetic_360)
    root, cv = tmp_path / "run", tmp_path / "cv"
    root.mkdir()
    cv.mkdir()
    protocol = json.loads((original / "protocol.json").read_text())
    protocol["local_cv_gate"]["passed"] = True
    write(cv / "completed.json", {"status": "COMPLETED", "gate": protocol["local_cv_gate"]})
    protocol["local_cv_completed_sha256"] = sha(cv / "completed.json")
    write(root / "protocol.json", protocol)
    write(root / "per_shot.json", {})
    write(root / "completed.json", dict(status="COMPLETED", protocol_sha256=sha(root / "protocol.json"),
                                        per_shot_sha256=sha(root / "per_shot.json"),
                                        local_gate_remains_failed=True))
    monkeypatch.setattr(audit, "CV", cv)
    with pytest.raises(ValueError, match="CV gate must remain failed"):
        audit.main(root)
    assert not (root / "audit_cpu.json").exists()


def test_full_synthetic_cpu_audit_and_exclusive_output(monkeypatch, synthetic_360):
    root = setup_synthetic(monkeypatch, synthetic_360)
    report = audit.main(root)
    assert report["status"] == "PASS"
    assert report["audit_type"] == "RECORDED_DATA_CONSISTENCY_ONLY"
    assert report["metrics"]["static"]["full360"] == score(90, 90, 360, 180, 180)
    assert report["metrics"]["worst_real"]["suffix340"] == score(85, 84, 340, 170, 170)
    assert report["flips"]["worst_real"]["full360"] == synthetic_360[3]
    assert all(not ids for ids in report["flips"]["worst_real"]["warmup20"].values())
    assert report["local_cv_gate"]["passed"] is False
    assert report["scope"] == "DIAGNOSTIC_COMPLETE360_AFTER_LOCAL_SURROGATE_GATE_FAILED"
    assert report["neural_replay_performed"] is False
    assert report["hidden_rms_verified"] is False
    assert report["evidence"]["shot_seals_verified"] == 360
    assert report["evidence"]["reference_cold_curves_equal"] == 360
    assert json.loads((root / "audit_cpu.json").read_text()) == report
    with pytest.raises(FileExistsError):
        audit.main(root)


def test_tampered_seals_and_cohort_are_detected(monkeypatch, synthetic_360, tmp_path):
    root = setup_synthetic(monkeypatch, synthetic_360)
    seal = json.loads((root / "shots/0001_1_seal.json").read_text())
    audit.check_seal(seal, 1, 1, root / "shots/0001_1.npz",
                     root / "shots/0001_1_updates.json", True)
    seal["curve_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seal"):
        audit.check_seal(seal, 1, 1, root / "shots/0001_1.npz",
                         root / "shots/0001_1_updates.json", True)
    assert tmp_path != root
