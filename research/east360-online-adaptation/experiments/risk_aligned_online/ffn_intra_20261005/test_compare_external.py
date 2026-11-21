"""CPU-only synthetic schema tests; none of these fixtures are experiment evidence."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from experiments.risk_aligned_online import representation_analysis as analysis
from experiments.risk_aligned_online.ffn_intra_20261005 import compare_external as compare


def _json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    _json(path, value)


def _pair(tmp_path, monkeypatch, count=6):
    """Tiny versions of the *different* redesign and FFN on-disk schemas."""
    source = tmp_path / "synthetic-parent.bin"
    prior = tmp_path / "synthetic-prior.bin"
    source.write_bytes(b"synthetic qualified parent, not a checkpoint")
    prior.write_bytes(b"synthetic forecast prior, not a checkpoint")
    shots = list(range(1001, 1001 + count))
    order = [{"shot": shot, "order_method": "shot_number",
              "provenance": {"sample_count": 4}} for shot in shots]
    order_path = tmp_path / "original-order.json"
    # The real runners reserialize these originals; the copy's bytes differ.
    order_path.write_text(json.dumps(order, separators=(",", ":")))
    roster = {"shot_ids": shots, "order_sha256": _digest(order_path),
              "selection": "synthetic label-blind fixture",
              "selection_protocol": {"rate_contract": "nominal", "rate_atol_hz": .5,
                                     "actual_rate_and_times_preserved": True,
                                     "label_blind": True, "outcome_class_filter": False,
                                     "prediction_filter": False}}
    roster_path = tmp_path / "original-roster.json"
    roster_path.write_text(json.dumps(roster, separators=(",", ":")))
    launcher = b"# Synthetic reviewed FFN launcher; never executed.\n"
    monkeypatch.setattr(compare, "_REVIEWED_FFN_LAUNCHER_SHA256",
                        hashlib.sha256(launcher).hexdigest())
    runner = b"def run_arm():\n    return 'synthetic schema fixture'\n"
    code = {name: f"# synthetic {name}\n".encode() for name in compare._INPUT_HELPERS}
    paths = []
    for arm in ("static", "fixed"):
        root = tmp_path / f"synthetic-{arm}-run"
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        paths.append(arm_dir)
        _json(root / "order.json", order)
        _json(root / "roster.json", roster)
        with zipfile.ZipFile(root / "code_snapshot.zip", "w") as archive:
            for name, content in code.items():
                archive.writestr(name, content)
            archive.writestr("redesign_20261004/run.py", runner)
            if arm == "fixed":
                archive.writestr("ffn_intra_20261005/run.py", launcher)
        protocol = dict(
            lineage=("INDEPENDENT_EAST_INTERNAL_REDESIGN_NOT_V3C" if arm == "static"
                     else "FFN_INTRA_SOURCE_LEARNED_SIGNAL_LOSS_NOT_V3C"),
            source_sha256=_digest(source), prior_sha256=_digest(prior),
            source_gate={"qualified": True, "checkpoint_sha256": _digest(source)},
            model_config={"window": 128, "input_view": "source"}, threshold=.5,
            order_sha256=_digest(order_path), roster_sha256=_digest(roster_path),
            code_snapshot_sha256=_digest(root / "code_snapshot.zip"), alarm_tuning=False,
            args=dict(checkpoint=str(source), prior=str(prior), order=str(order_path),
                      roster=str(roster_path), arms=[arm], limit=0))
        if arm == "static":
            protocol.update(
                input_contract=compare._INPUT_CONTRACT, rate_contract="nominal",
                rate_atol_hz=.5, roster_count=count, source_offline_only=True,
                target_labels_before_first_shot=0, support110_used_for_initialization=False,
                threshold_origin="fixed J-TEXT VAL160", east_label_ms=125.,
                lead_rule="strict >30ms",
                prior_protocol={"parent_sha256": _digest(source), "source_only": True,
                                "static_classifier_bitwise_unchanged": True},
                helper_code_sha256={k: hashlib.sha256(v).hexdigest() for k, v in code.items()},
                code_sha256={"run.py": hashlib.sha256(runner).hexdigest()})
            protocol["args"]["rate_contract"] = "nominal"
        else:
            # Actual FFN schema omits input_contract and rate_contract and calls
            # run_arm(arm='intra'), even though the directory is named fixed.
            protocol.update(
                source_only_initialization=True, inter_updates=False,
                target_label_access="natural reveal after whole-shot durable seal",
                source_rule_protocol=dict(parent_sha256=_digest(source), prior_sha256=_digest(prior),
                                          source_only=True, classifier_and_decoder_unchanged=True,
                                          threshold=.5))
        _json(root / "protocol.json", protocol)
        rows = []
        for index, shot in enumerate(shots, 1):
            case = (index - 1) % 6
            disruptive = case not in (2, 4)
            times = np.array([.9, 1., 3.97, 4.01])
            # TP->TP (3100->3000ms), TARDY->TP, FP->TN, FN->TP,
            # TN->FP, TP->FN. The final point must be masked for disruptions.
            static_scores = ([.9, .1, .1, .9], [.1, .1, .5, .9], [.1, .5, .1, .1],
                             [.1, .1, .1, .9], [.1] * 4, [.9, .1, .1, .9])
            fixed_scores = ([.1, .9, .1, .9], [.1, .9, .1, .9], [.1] * 4,
                            [.9, .1, .1, .9], [.5, .1, .1, .1], [.1, .1, .1, .9])
            scores = np.array((static_scores if arm == "static" else fixed_scores)[case])
            stem = f"{index:04d}_{shot}"
            npz = arm_dir / f"{stem}_sealed.npz"
            np.savez(npz, shot=shot, times=times, scores=scores)
            _json(arm_dir / f"{stem}_seal.json",
                  dict(shot=shot, arm="static" if arm == "static" else "intra",
                       threshold=.5, npz=npz.name, sha256=_digest(npz)))
            _json(arm_dir / f"{stem}_audit.json", {"shot": shot})
            valid = times <= 4. if disruptive else np.ones(4, bool)
            rows.append(dict(shot=shot, is_disrupt=disruptive, down=4. if disruptive else None,
                             times=times[valid], scores=scores[valid]))
        metrics = analysis.metrics_for_rows(rows, .5)
        metrics.pop("class_counts")
        for prediction in metrics["predictions"]:
            prediction.pop("class")
        summary = dict(arm="static" if arm == "static" else "intra", metrics=metrics,
                       observed_count=count, roster_count=count, accepted_inter_updates=0,
                       accepted_intra_updates=0 if arm == "static" else count,
                       elapsed_seconds=1., evidence="DEVELOPMENT_LOCKED_STREAM")
        _json(arm_dir / "summary.json", summary)
        _json(root / ("summary.json" if arm == "static" else "comparison.json"), {arm: summary})

    def outcome(trace, device, advanced):
        assert device == "EAST" and advanced == {}
        disruptive = (trace["shot"] - 1001) % 6 not in (2, 4)
        return dict(trace, is_disrupt=disruptive, down=4. if disruptive else None,
                    valid=np.ones(len(trace["times"]), bool))

    return *paths, outcome


def _run(pair, **kwargs):
    static, fixed, loader = pair
    return compare.compare_runs(static, fixed, expected_count=6, outcome_loader=loader, **kwargs)


def test_tiny_actual_schemas_metrics_flips_and_strict_early_alarm(tmp_path, monkeypatch):
    pair = _pair(tmp_path, monkeypatch)
    output = tmp_path / "synthetic-report.json"
    report = _run(pair, output=output)
    overall = report["segments"]["overall"]
    assert overall["shot_count"] == 6
    assert overall["static"]["class_counts"] == dict(TP=2, FN=1, TARDY=1, TN=1, FP=1)
    assert overall["fixed"]["class_counts"] == dict(TP=3, FN=1, TARDY=0, TN=1, FP=1)
    assert overall["static"]["tpr"] == .5
    assert overall["fixed"]["tpr"] == .75
    assert overall["fixed"]["fpr"] == .5
    assert overall["fixed"]["J"] == .25
    assert overall["paired_flips"]["helpful_count"] == 3
    assert overall["paired_flips"]["harmful_count"] == 2
    early = overall["early_above_3s"]
    assert early["static"]["shots"] == [1001, 1006]
    assert early["fixed"]["shots"] == [1004]
    assert early["gained_shots"] == [1004]
    assert early["lost_shots"] == [1001, 1006]
    assert report["segments"]["31to120"]["static"]["J"] is None
    assert report["segments"]["121toend"]["shot_count"] == 0
    assert report["verification"]["full2953_verified"] is False
    assert report["verification"]["input_contract"]["fixed_declaration_present"] is False
    assert report["verification"]["input_contract"]["basis"] == "verified_archived_code"
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError):
        _run(pair, output=output)


def test_exact_segment_boundaries(tmp_path, monkeypatch):
    static, fixed, loader = _pair(tmp_path, monkeypatch, count=122)
    report = compare.compare_runs(static, fixed, expected_count=122, outcome_loader=loader)
    for name, first, last, count in (("overall", 1, 122, 122), ("first30", 1, 30, 30),
                                     ("31to120", 31, 120, 90), ("121toend", 121, 122, 2)):
        segment = report["segments"][name]
        assert (segment["first_index"], segment["last_index"], segment["shot_count"]) == (first, last, count)
        assert segment["paired_flips"]["shot_count"] == count
    assert report["segments"]["121toend"]["paired_flips"]["flips"][0]["shots"] == [1122]


@pytest.mark.parametrize("filename", ["summary.json", "../comparison.json"])
def test_incomplete_rejected_before_trace_or_outcome_access(tmp_path, monkeypatch, filename):
    pair = _pair(tmp_path, monkeypatch)
    (pair[1] / filename).unlink()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("loaded incomplete run"))
    with pytest.raises(ValueError, match="incomplete|completion|summary"):
        _run(pair)


def test_default_and_cli_cannot_accept_a_tiny_prefix(tmp_path, monkeypatch, capsys):
    static, fixed, loader = _pair(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="2953"):
        compare.compare_runs(static, fixed, outcome_loader=loader)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(SystemExit) as exc:
        compare.main(["--static", str(static), "--fixed", str(fixed), "--output", str(output)])
    assert exc.value.code == 2
    assert "2953" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(("file", "change", "message"), [
    ("protocol.json", lambda p: p.pop("threshold"), "threshold"),
    ("protocol.json", lambda p: p.pop("source_sha256"), "source_sha256"),
    ("protocol.json", lambda p: p.update(source_sha256="f" * 64), "source|parent"),
    ("protocol.json", lambda p: p.update(prior_sha256="f" * 64), "prior"),
    ("protocol.json", lambda p: p.update(threshold=float(np.nextafter(.5, 1.))), "threshold"),
    ("protocol.json", lambda p: p["source_gate"].update(qualified=False), "qualified"),
    ("protocol.json", lambda p: p.update(input_contract="different"), "input_contract"),
    ("protocol.json", lambda p: p.update(rate_contract="strict"), "rate_contract"),
    ("protocol.json", lambda p: p["args"].update(limit=2), "limit|incomplete"),
    ("protocol.json", lambda p: p.update(source_only_initialization=False), "source_only"),
    ("protocol.json", lambda p: p.update(inter_updates=True), "inter_updates"),
    ("protocol.json", lambda p: p.update(alarm_tuning=True), "alarm_tuning"),
    ("protocol.json", lambda p: p.update(target_label_access="before seal"), "target_label_access"),
    ("order.json", lambda o: o.reverse(), "order"),
    ("roster.json", lambda r: r["shot_ids"].pop(), "roster"),
])
def test_metadata_mismatches_are_rejected(tmp_path, monkeypatch, file, change, message):
    pair = _pair(tmp_path, monkeypatch)
    _edit(pair[1].parent / file, change)
    with pytest.raises(ValueError, match=message):
        _run(pair)


@pytest.mark.parametrize("field", ["observed_count", "roster_count", "metrics"])
def test_missing_summary_fields_are_not_inferred(tmp_path, monkeypatch, field):
    pair = _pair(tmp_path, monkeypatch)
    _edit(pair[1] / "summary.json", lambda s: s.pop(field))
    with pytest.raises(ValueError, match=field):
        _run(pair)


def test_sha_validation_of_both_arms_precedes_any_npz_or_outcome(tmp_path, monkeypatch):
    pair = _pair(tmp_path, monkeypatch)
    last = pair[1] / "0006_1006_sealed.npz"
    last.write_bytes(last.read_bytes() + b"tampered")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("opened before all SHAs checked"))
    with pytest.raises(ValueError, match="SHA"):
        _run(pair)


def test_hash_helper_called_for_every_seal_before_join_and_again_on_use(tmp_path, monkeypatch):
    static, fixed, loader = _pair(tmp_path, monkeypatch)
    events = []
    real_hash = analysis._sha256

    def tracked_hash(path):
        if path.suffix == ".npz":
            events.append(("hash", path))
        return real_hash(path)

    def tracked_loader(*args):
        events.append(("outcome", args[0]["shot"]))
        return loader(*args)

    monkeypatch.setattr(analysis, "_sha256", tracked_hash)
    compare.compare_runs(static, fixed, expected_count=6, outcome_loader=tracked_loader)
    assert {p for kind, p in events[:12] if kind == "hash"} == {
        *static.glob("*_sealed.npz"), *fixed.glob("*_sealed.npz")}
    assert all(kind == "hash" for kind, _ in events[:12])
    assert sum(kind == "hash" for kind, _ in events) == 24
    assert sum(kind == "outcome" for kind, _ in events) == 12


@pytest.mark.parametrize("target", ["synthetic-parent.bin", "synthetic-prior.bin",
                                    "original-order.json", "original-roster.json",
                                    "synthetic-fixed-run/code_snapshot.zip"])
def test_recorded_artifact_hashes_are_checked(tmp_path, monkeypatch, target):
    pair = _pair(tmp_path, monkeypatch)
    path = tmp_path / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        _run(pair)


def test_unreviewed_ffn_launcher_cannot_supply_missing_contract(tmp_path, monkeypatch):
    pair = _pair(tmp_path, monkeypatch)
    monkeypatch.setattr(compare, "_REVIEWED_FFN_LAUNCHER_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="launcher|input.contract"):
        _run(pair)


@pytest.mark.parametrize("field", ["shot", "arm", "threshold", "npz", "sha256"])
def test_receipt_fields_are_mandatory(tmp_path, monkeypatch, field):
    pair = _pair(tmp_path, monkeypatch)
    _edit(pair[1] / "0001_1001_seal.json", lambda r: r.pop(field))
    with pytest.raises(ValueError, match=field):
        _run(pair)


@pytest.mark.parametrize("field", ["is_disrupt", "down", "valid", "shot"])
def test_outcome_fields_are_mandatory(tmp_path, monkeypatch, field):
    static, fixed, loader = _pair(tmp_path, monkeypatch)

    def missing(*args):
        result = loader(*args)
        result.pop(field)
        return result

    with pytest.raises(ValueError, match=field):
        compare.compare_runs(static, fixed, expected_count=6, outcome_loader=missing)


def test_corrupt_summary_metrics_and_missing_prediction_field(tmp_path, monkeypatch):
    pair = _pair(tmp_path, monkeypatch)
    summary = json.loads((pair[1] / "summary.json").read_text())
    for change, message in ((lambda s: s["metrics"].update(tp=0), "tp"),
                            (lambda s: s["metrics"].update(tp=3.5), "tp"),
                            (lambda s: s["metrics"]["predictions"][0].pop("alarm"), "alarm")):
        changed = deepcopy(summary)
        change(changed)
        _json(pair[1] / "summary.json", changed)
        _json(pair[1].parent / "comparison.json", {"fixed": changed})
        with pytest.raises(ValueError, match=message):
            _run(pair)


@pytest.mark.parametrize("suffix", ["_seal.json", "_sealed.npz", "_audit.json"])
def test_completed_summary_does_not_excuse_missing_sealed_artifacts(tmp_path, monkeypatch, suffix):
    pair = _pair(tmp_path, monkeypatch)
    (pair[1] / f"0006_1006{suffix}").unlink()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("opened incomplete seals"))
    with pytest.raises(ValueError, match="incomplete"):
        _run(pair)


@pytest.mark.parametrize(("change", "message"), [
    (lambda r: r.update(threshold=float(np.nextafter(.5, 1.))), "threshold"),
    (lambda r: r.update(arm="fixed"), "arm"),
    (lambda r: r.update(shot=999), "shot"),
    (lambda r: r.update(npz="../outside.npz"), "npz"),
])
def test_receipts_must_match_exact_declared_identity(tmp_path, monkeypatch, change, message):
    pair = _pair(tmp_path, monkeypatch)
    _edit(pair[1] / "0001_1001_seal.json", change)
    with pytest.raises(ValueError, match=message):
        _run(pair)


@pytest.mark.parametrize(("change", "message"), [
    (lambda trace: trace.pop("shot"), "missing shot"),
    (lambda trace: trace.update(shot=999), "shot mismatch"),
    (lambda trace: trace.update(times=np.array([.8, 1., 3.97, 4.01])), "time grid"),
    (lambda trace: trace.update(times=np.array([1., .9, 3.97, 4.01])), "increasing"),
])
def test_resealed_npz_identity_and_time_grid_are_not_inferred(tmp_path, monkeypatch, change, message):
    pair = _pair(tmp_path, monkeypatch)
    path = pair[1] / "0001_1001_sealed.npz"
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    change(trace)
    np.savez(path, **trace)
    _edit(pair[1] / "0001_1001_seal.json", lambda r: r.update(sha256=_digest(path)))
    with pytest.raises(ValueError, match=message):
        _run(pair)


@pytest.mark.parametrize(("member", "content", "message"), [
    ("causal_input.py", b"# different input contract\n", "input_contract helper"),
    ("redesign_20261004/run.py", b"def run_arm():\n    return 'changed'\n", "run_arm"),
    ("ffn_intra_20261005/run.py", b"# changed launcher\n", "launcher"),
])
def test_snapshot_contract_changes_rejected_even_with_new_archive_hash(
        tmp_path, monkeypatch, member, content, message):
    pair = _pair(tmp_path, monkeypatch)
    root = pair[1].parent
    path = root / "code_snapshot.zip"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[member] = content
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    _edit(root / "protocol.json", lambda p: p.update(code_snapshot_sha256=_digest(path)))
    with pytest.raises(ValueError, match=message):
        _run(pair)


def test_output_cannot_be_written_inside_either_run(tmp_path, monkeypatch):
    pair = _pair(tmp_path, monkeypatch)
    for path in (pair[0].parent / "sidecar.json", pair[1] / "sidecar.json"):
        with pytest.raises(ValueError, match="outside"):
            _run(pair, output=path)
        assert not path.exists()


def test_cli_writes_synthetic_report_without_importing_runner(tmp_path, monkeypatch):
    pair = _pair(tmp_path, monkeypatch)
    monkeypatch.setattr(compare, "compare_runs", partial(compare.compare_runs,
                        expected_count=6, outcome_loader=pair[2]))
    output = tmp_path / "synthetic-cli-report.json"
    assert compare.main(["--static", str(pair[0]), "--fixed", str(pair[1]),
                         "--output", str(output)]) == 0
    assert json.loads(output.read_text())["analysis_only"] is True


def test_import_is_cpu_only_and_does_not_import_model_or_runner():
    code = """
import sys
from experiments.risk_aligned_online.ffn_intra_20261005 import compare_external
assert 'torch' not in sys.modules
for name in sys.modules:
    if name.startswith('experiments.risk_aligned_online.'):
        assert name.rsplit('.', 1)[-1] not in {'run', 'model', 'simple_online', 'simple_train'}
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
