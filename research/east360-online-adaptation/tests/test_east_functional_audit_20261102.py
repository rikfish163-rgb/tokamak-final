"""CPU-only synthetic contracts; no experiment files or signal datasets are read."""

from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from experiments.risk_aligned_online.east_decoder_calibration_20261102 import (
    functional_audit as audit,
)


def test_successful_audit_report_can_be_saved_without_overwriting(tmp_path):
    output = tmp_path / "audit_cpu.json"
    report = dict(status="PASS", scope="RECORDED_DATA_CONSISTENCY_ONLY")
    audit.save_report(report, output)
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError):
        audit.save_report(report, output)
    with pytest.raises(ValueError):
        audit.save_report(dict(status="FAILED"), tmp_path / "invalid.json")


def trial(fraction, objective):
    return dict(
        fraction=fraction,
        diagnostic_loss=objective,
        protection_loss=0.0,
        objective=objective,
    )


def receipt(end, *, new, mode):
    empty = mode == "empty"
    accepted = mode == "accepted"
    gradient = [0.0] * 256
    gradient[0] = 0.0 if empty else 0.25
    state = [0.0] * 256
    state[0] = 0.125
    result = dict(
        origin=end - 25,
        maturity=end,
        sealed_through=end,
        first_affected=end + 1,
        forecast_sha256="a" * 64,
        mask_sha256="b" * 64,
        target_sha256="c" * 64,
        observed=0 if empty else 2,
        accepted=accepted,
        reason="no_target" if empty else ("improved" if accepted else "not_improved"),
        loss_before=None if empty else 1.0,
        loss_proposal=None if empty else (0.9 if accepted else 1.3),
        gradient_norm=0.0 if empty else 0.25,
        step_norm=0.125 if accepted else 0.0,
        gradient256=gradient,
        state256_after=state,
        chosen_fraction=1.0 if accepted else None,
    )
    if new:
        result["tested_steps"] = (
            []
            if empty
            else (
                [trial(1.0, 0.9)]
                if accepted
                else [
                    trial(f, v) for f, v in zip(audit.FRACTIONS, (1.2, 1.1, 1.0, 1.3))
                ]
            )
        )
    return result


def updates():
    return {
        arm: [
            receipt(159, new=arm != "old_real", mode="accepted"),
            receipt(287, new=arm != "old_real", mode="rejected"),
            receipt(415, new=arm != "old_real", mode="empty"),
        ]
        for arm in audit.ARMS[1:]
    }


def test_valid_receipts_reset_and_count_backtracking():
    records = updates()
    totals = audit.check_updates(records, 416, warmup=False)
    assert totals["old_real"]["events"] == 3
    assert totals["old_real"]["accepted"] == 1
    assert totals["old_real"]["tested_steps"] == 0
    assert totals["hidden_real"]["tested_steps"] == 5
    assert totals["hidden_real"]["all_rejected"] == 1
    assert totals["hidden_real"]["no_target"] == 1
    assert totals["hidden_real"]["max_state_norm"] == 0.125
    # The first step is checked against zero independently for every shot.
    records["hidden_real"][0]["step_norm"] = 0.0
    with pytest.raises(ValueError, match="step norm"):
        audit.check_updates(records, 416, warmup=False)


@pytest.mark.parametrize(
    ("arm", "index", "key", "value", "message"),
    [
        ("old_real", 0, "origin", 135, "clock"),
        ("hidden_real", 0, "first_affected", 159, "clock"),
        ("hidden_real", 0, "mask_sha256", "d" * 64, "mask"),
        ("parameter_real", 0, "target_sha256", "d" * 64, "target"),
        ("hidden_real", 0, "gradient_norm", 1.0, "gradient norm"),
        ("hidden_real", 0, "accepted", False, "acceptance"),
        ("hidden_real", 0, "chosen_fraction", 0.25, "chosen"),
        ("old_real", 0, "chosen_fraction", 0.25, "chosen"),
        ("hidden_real", 1, "state256_after", [0.0] * 256, "step norm"),
        ("hidden_real", 2, "observed", 1, "nonfinite receipt"),
    ],
)
def test_receipt_tampering_is_rejected(arm, index, key, value, message):
    records = updates()
    records[arm][index][key] = value
    with pytest.raises(ValueError, match=message):
        audit.check_updates(records, 416, warmup=False)


def test_rejected_state_cannot_move_even_if_norm_is_rewritten():
    records = updates()
    r = records["hidden_real"][1]
    r["state256_after"][0] = 0.25
    r["step_norm"] = 0.125
    with pytest.raises(ValueError, match="rejected state"):
        audit.check_updates(records, 416, warmup=False)


def test_old_receipt_has_no_tested_steps_and_only_full_step():
    records = updates()
    records["old_real"][0]["tested_steps"] = [trial(1.0, 0.9)]
    with pytest.raises(ValueError, match="old.*tested_steps"):
        audit.check_updates(records, 416, warmup=False)


def test_old_chosen_fraction_must_be_numeric_not_a_boolean():
    records = updates()
    records["old_real"][0]["chosen_fraction"] = True
    with pytest.raises(ValueError):
        audit.check_updates(records, 416, warmup=False)


@pytest.mark.parametrize(
    "tamper", ["wrong_fraction", "earlier_improved", "empty", "short_reject"]
)
def test_backtracking_requires_first_strict_improvement_or_four_rejections(tamper):
    records = updates()
    if tamper == "wrong_fraction":
        records["hidden_real"][1]["tested_steps"][1]["fraction"] = 0.5
    elif tamper == "earlier_improved":
        records["hidden_real"][1]["tested_steps"][0] = trial(1.0, 0.9)
    elif tamper == "empty":
        records["hidden_real"][0]["tested_steps"] = []
    else:
        records["hidden_real"][1]["tested_steps"].pop()
    with pytest.raises(ValueError):
        audit.check_updates(records, 416, warmup=False)


def test_fraction_prefix_accepts_second_candidate_and_checks_proposal():
    records = updates()
    r = records["hidden_real"][0]
    r["tested_steps"] = [trial(1.0, 1.0), trial(0.25, 0.9)]
    r["chosen_fraction"] = 0.25
    audit.check_updates(records, 416, warmup=False)
    r["loss_proposal"] = 0.8
    with pytest.raises(ValueError, match="proposal"):
        audit.check_updates(records, 416, warmup=False)


def test_acceptance_is_strict_even_for_one_float64_ulp():
    records = updates()
    r = records["hidden_real"][0]
    lower = float(np.nextafter(1.0, 0.0))
    r["loss_proposal"] = lower
    r["tested_steps"] = [trial(1.0, lower)]
    audit.check_updates(records, 416, warmup=False)
    r["loss_proposal"] = 1.0
    r["tested_steps"] = [trial(1.0, 1.0)]
    with pytest.raises(ValueError, match="acceptance"):
        audit.check_updates(records, 416, warmup=False)


def test_loss_addition_tolerance_is_separate_from_acceptance():
    records = updates()
    r = records["parameter_real"][0]
    diagnostic, protection = float(np.float32(0.8)), float(np.float32(0.1))
    objective = float(np.float32(np.float32(diagnostic) + np.float32(protection)))
    r["tested_steps"] = [
        dict(
            fraction=1.0,
            diagnostic_loss=diagnostic,
            protection_loss=protection,
            objective=objective,
        )
    ]
    r["loss_proposal"] = objective
    audit.check_updates(records, 416, warmup=False)
    r["tested_steps"][0]["protection_loss"] = 0.2
    with pytest.raises(ValueError, match="objective"):
        audit.check_updates(records, 416, warmup=False)


def test_loss_tolerance_does_not_hide_tiny_absolute_discrepancy():
    records = updates()
    r = records["hidden_real"][0]
    r["tested_steps"] = [
        dict(fraction=1.0, diagnostic_loss=0.0, protection_loss=0.0, objective=1e-13)
    ]
    r["loss_proposal"] = 1e-13
    with pytest.raises(ValueError, match="objective"):
        audit.check_updates(records, 416, warmup=False)


def test_warmup_must_have_no_events():
    empty = {arm: [] for arm in audit.ARMS[1:]}
    audit.check_updates(empty, 416, warmup=True)
    empty["hidden_hold"].append(updates()["hidden_hold"][0])
    with pytest.raises(ValueError, match="clock"):
        audit.check_updates(empty, 416, warmup=True)


def test_curves_reference_prefix_and_warmup_are_exact():
    times = np.arange(200, dtype=np.float64) / 1000
    static = np.full(200, 0.25)
    old = static.copy()
    old[160:] = 0.5
    curves = {arm: static.copy() for arm in audit.ARMS}
    curves["old_real"] = old.copy()
    audit.validate_curves(times, curves, times, static, old, False)
    curves["hidden_real"][159] = 0.25000001
    with pytest.raises(ValueError, match="first 160"):
        audit.validate_curves(times, curves, times, static, old, False)
    curves["hidden_real"][159] = 0.25
    curves["old_real"][160] = 0.6
    with pytest.raises(ValueError, match="old_real"):
        audit.validate_curves(times, curves, times, static, old, False)
    curves["old_real"] = old.copy()
    with pytest.raises(ValueError, match="warmup"):
        audit.validate_curves(times, curves, times, static, old, True)


def test_seal_requires_identity_outcome_exclusion_and_hashes():
    seal = dict(
        index=21,
        shot=123,
        curve_sha256="a" * 64,
        updates_sha256="b" * 64,
        current_outcome_used_in_prediction=False,
        reused_warmup=False,
    )
    audit.check_seal(seal, 21, 123, "a" * 64, "b" * 64, False)
    for key, value in (
        ("shot", 124),
        ("curve_sha256", "c" * 64),
        ("current_outcome_used_in_prediction", True),
        ("reused_warmup", True),
    ):
        bad = {**seal, key: value}
        with pytest.raises(ValueError, match="seal"):
            audit.check_seal(bad, 21, 123, "a" * 64, "b" * 64, False)


def archive(files):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, contents in files:
            z.writestr(name, contents)
    buffer.seek(0)
    return buffer


def test_snapshot_checks_file_members_and_hashes():
    root = Path("/synthetic")
    contents = b"frozen code"
    bound = {str(root / "src.py"): hashlib.sha256(contents).hexdigest()}
    audit.verify_snapshot(archive([("src.py", contents)]), bound, root)
    with pytest.raises(ValueError, match="snapshot"):
        audit.verify_snapshot(archive([("src.py", b"tampered")]), bound, root)
    with pytest.raises(ValueError, match="snapshot"):
        audit.verify_snapshot(
            archive([("src.py", contents), ("extra", b"")]), bound, root
        )
    with pytest.raises(ValueError, match="snapshot"):
        audit.verify_snapshot(
            archive([("src.py", contents)]), {"/elsewhere/src.py": "a" * 64}, root
        )


def test_score_strict_lead_and_all_flip_categories():
    times = np.array([0.0, 0.1])
    # Exactly 30 ms is not timely, but over 3000 ms is counted separately as TP.
    assert (
        audit.classify_curve(times, np.array([0.0, 1.0]), True, 0.13, 0.5)["timely"]
        is False
    )
    early = audit.classify_curve(times, np.array([1.0, 0.0]), True, 3.001, 0.5)
    assert early["timely"] is True
    before = [
        dict(shot=1, disruptive=True, alarm=False, lead_ms=None, timely=False),
        dict(shot=2, disruptive=True, alarm=True, lead_ms=40.0, timely=True),
        dict(shot=3, disruptive=False, alarm=True, lead_ms=None, timely=False),
        dict(shot=4, disruptive=False, alarm=False, lead_ms=None, timely=False),
    ]
    after = deepcopy(before)
    after[0].update(early)
    after[1].update(alarm=False, lead_ms=None, timely=False)
    after[2]["alarm"] = False
    after[3]["alarm"] = True
    assert audit.paired_flips(before, after) == dict(
        rescued_D=[1], lost_D=[2], removed_FP=[3], added_FP=[4]
    )
    score = audit.recount(after)
    assert (score["tp"], score["fp"], score["early_over3000"]) == (1, 1, 1)
    assert score["median_timely_lead_ms"] == 3001.0


def test_incomplete_main_never_reads_outcomes(monkeypatch):
    monkeypatch.setattr(audit, "read", lambda path: {"status": "RUNNING"})
    monkeypatch.setattr(
        audit.io23, "read_outcome", lambda *_: pytest.fail("outcome read")
    )
    with pytest.raises(ValueError, match="COMPLETED"):
        audit.main(Path("/nonexistent"))


def test_import_does_not_load_torch_or_the_runner():
    code = (
        "import sys; "
        "from experiments.risk_aligned_online.east_decoder_calibration_20261102 "
        "import functional_audit; "
        "assert 'torch' not in sys.modules; "
        "assert not any(k.endswith('.functional_stream') for k in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_completed_header_binds_prior_gate_files_and_code(monkeypatch):
    root, project = Path("/synthetic/output"), Path("/synthetic")
    old, cv, ancestor = project / "old", project / "cv", project / "ancestor"
    monkeypatch.setattr(audit, "ROOT", project)
    monkeypatch.setattr(audit, "OLD", old)
    monkeypatch.setattr(audit, "CV", cv)
    monkeypatch.setattr(audit, "ANCESTOR", ancestor)
    ids = list(range(1, 361))
    order = [{"shot": sid} for sid in ids]
    gate = {"passed": False}
    old_protocol = dict(
        order=order,
        head_fit_ids=ids[:20],
        threshold=0.5,
        source_sha256="1" * 64,
        cold_sha256="2" * 64,
        input_map_sha256="3" * 64,
        prior_sha256="4" * 64,
        local_cv_completed_sha256="5" * 64,
        local_cv_gate=gate,
    )
    bound = {
        str(project / "EAST_FUNCTIONAL_PROTECTION360_PLAN_20261102.md"): "7" * 64,
        str(Path(audit.__file__).with_name("functional_stream.py")): "8" * 64,
        str(Path(audit.__file__).with_name("functional_update.py")): "9" * 64,
    }
    protocol = {
        **old_protocol,
        "old_completed_sha256": audit.OLD_SHA,
        "arms": list(audit.ARMS),
        "fractions": list(audit.FRACTIONS),
        "scope": "DIAGNOSTIC_FUNCTIONAL_PROTECTION_SAME_FROZEN_HEAD",
        "head_preselected": "worst_meta",
        "candidate_preselected": "hidden_real",
        "head_training_steps": 0,
        "suffix_supervised_updates": 0,
        "warmup20_cost_included": True,
        "method_promoted": False,
        "independent_confirmation": False,
        "heads_sha256": "a" * 64,
        "basis_sha256": "b" * 64,
        "bound_code_sha256": bound,
    }
    done = dict(
        status="COMPLETED",
        protocol_sha256="c" * 64,
        per_shot_sha256="d" * 64,
        old_local_gate_remains_failed=True,
        independent_confirmation=False,
        optimizer_steps=0,
        source_unchanged=True,
        cold_unchanged=True,
        heads_unchanged=True,
    )
    old_done = dict(
        status="COMPLETED",
        protocol_sha256="e" * 64,
        per_shot_sha256="f" * 64,
        local_gate_remains_failed=True,
        metrics={"static": {"full360": {"tp": 125, "fp": 35}}},
    )
    cpu = dict(
        status="PASS",
        evidence={"completed_sha256": audit.OLD_SHA, "protocol_sha256": "e" * 64},
        local_cv_gate=gate,
    )
    documents = {
        root / "completed.json": done,
        root / "protocol.json": protocol,
        old / "completed.json": old_done,
        old / "protocol.json": old_protocol,
        old / "audit_cpu.json": cpu,
        old / "verification_neural.json": dict(
            status="PASS", completed_sha256=audit.OLD_SHA
        ),
        old / "calibration.json": dict(heads_sha256="a" * 64),
        cv / "completed.json": dict(
            status="COMPLETED",
            gate=gate,
            artifact_sha256={"diagnostic_basis.pt": "b" * 64},
        ),
        ancestor / "protocol.json": dict(calibration_reference=str(project / "map")),
        project / "map/input_map.json": dict(calibration_ids=ids[:20]),
    }
    hashes = {
        root / "completed.json": "0" * 64,
        root / "protocol.json": "c" * 64,
        root / "per_shot.json": "d" * 64,
        old / "completed.json": audit.OLD_SHA,
        old / "protocol.json": "e" * 64,
        old / "per_shot.json": "f" * 64,
        cv / "completed.json": "5" * 64,
        old / "heads.pt": "a" * 64,
        cv / "diagnostic_basis.pt": "b" * 64,
        project / "map/input_map.json": "3" * 64,
        project
        / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt": "1"
        * 64,
        project / "probe_outputs/stratified_cold20_context360_v1/models/cold.pt": "2"
        * 64,
        project / "probe_outputs/direct_decoder_source_v1/decoders.pt": "4" * 64,
        **{Path(path): digest for path, digest in bound.items()},
    }
    monkeypatch.setattr(audit, "read", lambda path: documents[Path(path)])
    monkeypatch.setattr(audit, "sha", lambda path: hashes[Path(path)])
    snapshots = []
    monkeypatch.setattr(audit, "verify_snapshot", lambda *args: snapshots.append(args))
    assert audit.check_header(root)[4] == ids
    assert snapshots == [(root / "code_snapshot.zip", bound, project)]
    done["old_local_gate_remains_failed"] = False
    with pytest.raises(ValueError, match="CV gate"):
        audit.check_header(root)
    done["old_local_gate_remains_failed"] = True
    protocol["candidate_preselected"] = "parameter_real"
    with pytest.raises(ValueError, match="scope"):
        audit.check_header(root)
    protocol["candidate_preselected"] = "hidden_real"
    hashes[old / "heads.pt"] = "0" * 64
    with pytest.raises(ValueError, match="heads or basis"):
        audit.check_header(root)


def test_gate_uses_fixed_absolute_thresholds_and_all_controls():
    def metric(tp, fp):
        return {"full360": {"tp": tp, "fp": fp}}

    scores = {arm: metric(125, 35) for arm in audit.ARMS}
    scores["hidden_real"] = metric(126, 35)
    gate = audit.diagnostic_gate(scores)
    assert gate["passed"] is True
    scores["hidden_reverse"] = metric(127, 35)
    assert audit.diagnostic_gate(scores)["passed"] is False
    scores["hidden_reverse"] = metric(125, 35)
    scores["hidden_real"] = metric(126, 36)  # net 90, not strictly above 90
    assert audit.diagnostic_gate(scores)["operational"] is False


def test_synthetic_report_compares_every_row_metric_and_flip():
    rows = {arm: [] for arm in audit.ARMS}
    for i in range(360):
        disruptive = i < 180
        timely = disruptive and i < 125
        alarm = timely if disruptive else i < 215
        row = dict(
            shot=i + 1,
            disruptive=disruptive,
            alarm=alarm,
            lead_ms=50.0 if timely else None,
            timely=timely,
        )
        for arm in audit.ARMS:
            rows[arm].append(dict(row))
    metrics, flips, gate = audit.summarize(rows)
    done = dict(
        metrics=metrics, flips={a: v["full360"] for a, v in flips.items()}, gate=gate
    )
    audit.compare_report(rows, deepcopy(rows), done)
    bad = deepcopy(done)
    bad["metrics"]["hidden_hold"]["suffix340"]["tp"] += 1
    with pytest.raises(ValueError, match="report"):
        audit.compare_report(rows, rows, bad)
    bad_rows = deepcopy(rows)
    bad_rows["hidden_real"][100]["alarm"] = False
    with pytest.raises(ValueError, match="report"):
        audit.compare_report(rows, bad_rows, done)
    assert json.dumps(dict(metrics=metrics, flips=flips, gate=gate), allow_nan=False)


def test_synthetic_complete360_pipeline_uses_only_sealed_metadata(monkeypatch):
    """Exercise all 360 shots in memory, including fresh per-shot state."""
    ids = list(range(1, 361))
    root = Path("/synthetic/output")
    times = np.arange(161, dtype=np.float64) / 1000
    expected_rows = {arm: [] for arm in audit.ARMS}
    for sid in ids:
        disruptive = sid <= 180
        alarm = sid <= 125 or 181 <= sid <= 215
        row = dict(
            shot=sid,
            disruptive=disruptive,
            alarm=alarm,
            lead_ms=1000.0 if disruptive and alarm else None,
            timely=bool(disruptive and alarm),
        )
        for arm in audit.ARMS:
            expected_rows[arm].append(dict(row))
    metrics, flips, gate = audit.summarize(expected_rows)
    done = dict(
        metrics=metrics,
        flips={a: v["full360"] for a, v in flips.items()},
        gate=gate,
        protocol_sha256="e" * 64,
        per_shot_sha256="f" * 64,
    )
    old_done = dict(
        metrics={"static": metrics["static"], "worst_real": metrics["old_real"]}
    )
    protocol = dict(threshold=0.5, scope="synthetic")
    monkeypatch.setattr(
        audit,
        "check_header",
        lambda _: (
            done,
            protocol,
            old_done,
            {"gate": {"passed": False}},
            ids,
            "d" * 64,
        ),
    )

    def glob(directory, pattern):
        suffix = pattern[1:]
        return [directory / f"{i:04d}_{sid}{suffix}" for i, sid in enumerate(ids, 1)]

    monkeypatch.setattr(audit.Path, "glob", glob)

    def fake_read(path):
        path = Path(path)
        if path.name == "per_shot.json":
            if path.parent == root:
                return deepcopy(expected_rows)
            return {
                "static": expected_rows["static"],
                "worst_real": expected_rows["old_real"],
            }
        index, sid = map(int, path.name.split("_")[:2])
        if path.name.endswith("_seal.json"):
            return dict(
                index=index,
                shot=sid,
                curve_sha256="a" * 64,
                updates_sha256="b" * 64,
                current_outcome_used_in_prediction=False,
                reused_warmup=index <= 20,
            )
        assert path.name.endswith("_updates.json")
        result = {arm: [] for arm in audit.ARMS[1:]}
        if index > 20:
            for arm in result:
                r = receipt(159, new=arm != "old_real", mode="empty")
                r["state256_after"] = [0.0] * 256
                result[arm].append(r)
        return result

    def fake_sha(path):
        path = Path(path)
        if path.name.endswith(".npz"):
            return "a" * 64
        if path.name.endswith("_updates.json"):
            return "b" * 64
        if path.name.endswith("_seal.json"):
            return "c" * 64
        if path.name == "protocol.json":
            return "e" * 64
        if path.name == "per_shot.json":
            return "f" * 64
        if path.name == "code_snapshot.zip":
            return "1" * 64
        assert path.name == "completed.json"
        return "d" * 64 if path.parent == root else audit.OLD_SHA

    class FakeNpz:
        def __init__(self, path):
            sid = int(Path(path).stem.split("_")[1])
            curve = np.zeros(len(times), dtype=np.float32)
            if sid <= 125 or 181 <= sid <= 215:
                curve[0] = 1
            self.items = {
                "times": times,
                **{arm: curve for arm in audit.ARMS},
                "worst_real": curve,
            }
            self.files = list(self.items)
            if Path(path).is_relative_to(root):
                self.items.pop("worst_real")
                self.files.remove("worst_real")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __getitem__(self, key):
            return self.items[key]

    checked = []
    real_check_seal = audit.check_seal

    def check_seal(*args):
        real_check_seal(*args)
        checked.append((args[1], args[2]))

    def outcome(device, sid):
        assert device == "EAST"
        assert checked[-2:] == [(sid, sid), (sid, sid)]
        return sid <= 180, 1.0 if sid <= 180 else None

    monkeypatch.setattr(audit, "read", fake_read)
    monkeypatch.setattr(audit, "sha", fake_sha)
    monkeypatch.setattr(audit.np, "load", lambda path, **_: FakeNpz(path))
    monkeypatch.setattr(audit, "check_seal", check_seal)
    monkeypatch.setattr(audit.io23, "read_outcome", outcome)
    monkeypatch.setattr(
        audit.io23, "read_signals", lambda *_: pytest.fail("signal read")
    )
    report = audit.main(root)
    assert report["status"] == "PASS"
    assert report["metrics"]["static"]["full360"]["n_disrupt"] == 180
    assert report["metrics"]["static"]["full360"]["n_safe"] == 180
    assert report["update_statistics"]["hidden_real"]["no_target"] == 340
    assert report["neural_replay_performed"] is False
    assert report["evidence"]["code_snapshot_sha256"] == "1" * 64
    assert report["evidence"]["per_shot_sha256"] == "f" * 64
