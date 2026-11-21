"""Synthetic-only CPU audit tests; no trial import or live output access."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.risk_aligned_online.source_event_tasks_20261102 import audit


def episode(shot, group, origin=127, role="val"):
    return dict(shot=shot, group=group, origin=origin, maturity=origin + 25,
                query=origin + 26, label=float(group == "danger"),
                disruptive=group != "safe", lead_ms=None if group == "safe" else 20.,
                feedback_observed=True, role=role, device="J-TEXT")


def macro(rows):
    """Independent test oracle with deliberately unequal per-shot sample counts."""
    result = {}
    for group in ("safe", "danger", "early"):
        local = [r for r in rows if r["group"] == group]
        by_shot = {}
        for r in local:
            by_shot.setdefault(r["shot"], []).append(r["delta_bce"])
        result[group] = dict(n_shots=len(by_shot), n_points=len(local),
            mean_delta_bce=float(np.mean([np.mean(v) for v in by_shot.values()])) if local else None,
            helpful_points=sum(r["delta_bce"] < 0 for r in local),
            harmful_points=sum(r["delta_bce"] > 0 for r in local))
    result["balanced"] = sum(result[g]["mean_delta_bce"] for g in ("safe", "danger")) / 2
    return result


@pytest.fixture
def synthetic(monkeypatch):
    val = [episode(101 if i % 2 else 102, "safe", 127 + i * 25) for i in range(600)]
    val += [episode(103, "danger", 127 + i * 25) for i in range(300)]
    val += [episode(103, "early", 10027 + i * 25) for i in range(46)]
    train = [episode(11, "safe", role="train"), episode(12, "danger", role="train"),
             episode(12, "early", 152, "train")]
    refs = dict(train=(train, [11, 12], {"role": "train"}),
                val=(val, [101, 102, 103], {"role": "val"}))
    def with_family(r):
        return dict(r, origin_family=-1, query_family=5)
    archived_val = [with_family(r) for r in val]
    tasks = ("native", "no_MNM", "no_MNM_CIII")
    arms = tuple(a + "_" + m for a in ("shared", "true", "shuffled")
                 for m in ("real", "reverse", "hold"))
    schedule = [dict(step=s + 1, task=tasks[s % 3], indices=[0, 1]) for s in range(512)]
    history = [dict(r, objectives=dict.fromkeys(("shared", "true", "shuffled"), .1),
                    gradient_norms=dict.fromkeys(("shared", "true", "shuffled"), .2),
                    accepted=dict.fromkeys(("shared", "true", "shuffled"), 2)) for r in schedule]
    rows, summary = [], {}
    for task in tasks:
        summary[task] = {}
        for arm in arms:
            local = []
            for i, r in enumerate(archived_val):
                old = .2 if i % 2 else .1
                improvement = .04 if arm == "true_real" else .01
                new = old + (improvement if r["label"] else -improvement)
                before = float(np.logaddexp(0., old) - r["label"] * old)
                after = float(np.logaddexp(0., new) - r["label"] * new)
                local.append(dict(r, task=task, arm=arm, before_logit=old, after_logit=new,
                    before_bce=before, after_bce=after, delta_bce=after - before,
                    context=[1 / 6] * 6, query_rms=.02, accepted=True,
                    loss_before=1., loss_proposal=.9, gradient_norm=.2, step_norm=.1))
            summary[task][arm] = macro(local)
            rows.extend(local)
    protocol = dict(scope="SOURCE_ONLY_OFFLINE_EVENT_MASKED_TASK_TEACHING",
        source_sha256=audit.SOURCE_SHA, qualified_source_parameters_updated=0,
        source_slow_zero=True, east_reads=0, test_signals_read=0,
        train_role="train", val_role="val", steps=512, batch_size=2, seed=131,
        lr=.001, optimizer="Adam", gradient_clip=1., selection="fixed last iterate",
        source_labels="max(25ms,AdvancedTime), strict lead>10ms",
        tasks=dict(native=[], no_MNM=["MNM"], no_MNM_CIII=["MNM", "CIII"]),
        arms=["shared", "true", "shuffled"], local_episode_not_continuous_stream=True,
        train_provenance={"role": "train", "raw_inputs_rehashed": True},
        val_provenance={"role": "val", "raw_inputs_rehashed": True})
    payload = {"protocol.json": protocol, "val_records.json": archived_val,
        "train_records_selected.json": [with_family(r) for r in train[:2]],
        "schedule.json": schedule, "training.json": history,
        "validation_records.json": rows, "validation_summary.json": summary,
        "preflight.json": dict(status="PASS"), "census.json": {},
        "tables.pt": "opaque bytes: NEVER deserialize", "code_snapshot.zip": "opaque"}
    gate = {k: True for k in ("safe_improved", "danger_improved", "better_than_shared_real",
        "better_than_shuffled_real", "better_than_true_reverse", "better_than_true_hold",
        "native_protected", "passed")}
    payload["completed.json"] = dict(status="COMPLETED", gate=gate, source_unchanged=True,
        source_parameter_updates=0, east_reads=0, test_signals_read=0,
        source_train_episodes_selected=2, validation_episodes=946,
        validation_records=25542, full_east_stream_tested=False)

    def seal():
        manifest = {name: digest(value) for name, value in payload.items() if name != "completed.json"}
        payload["completed.json"].update(artifact_sha256=manifest,
                                        schedule_sha256=manifest["schedule.json"])

    seal()
    monkeypatch.setattr(audit, "read", lambda path: payload[Path(path).name])
    monkeypatch.setattr(audit, "sha", lambda path: digest(payload[Path(path).name]))
    monkeypatch.setattr(audit, "load_records", lambda role: refs[role])
    return payload, seal, refs


def digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=True).encode()).hexdigest()


def test_complete_synthetic_archive_and_main_write_contract(synthetic, monkeypatch):
    payload, _, _ = synthetic
    saved = []
    monkeypatch.setattr(audit, "write", lambda path, value: saved.append((Path(path), value)))
    result = audit.main(Path("/synthetic-only/not-a-real-output"))
    assert result["status"] == "PASS" and result["validation_records"] == 946 * 3 * 9
    assert result["gate"] == payload["completed.json"]["gate"]
    assert result["summary"] == payload["validation_summary.json"]
    assert result["neural_replayed"] is False and result["raw_inputs_read"] == 0
    assert saved == [(Path("/synthetic-only/not-a-real-output/audit_cpu.json"), result)]


@pytest.mark.parametrize("name", ["protocol.json", "schedule.json", "validation_records.json", "tables.pt"])
def test_hash_tampering_and_missing_manifest(synthetic, name):
    payload, _, _ = synthetic
    payload["completed.json"]["artifact_sha256"][name] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        audit.verify(Path("/synthetic"))


@pytest.mark.parametrize("change", ["duplicate", "missing", "task", "arm", "label", "order"])
def test_exact_cartesian_coverage_and_archived_record_correspondence(synthetic, change):
    p, seal, _ = synthetic
    rows = p["validation_records.json"]
    if change == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif change == "missing":
        rows.pop()
    elif change in ("task", "arm"):
        rows[0][change] = "unknown"
    elif change == "label":
        rows[0]["label"] = 1.
    else:
        p["val_records.json"][:2] = reversed(p["val_records.json"][:2])
    seal()
    with pytest.raises(ValueError):
        audit.verify(Path("/synthetic"))


@pytest.mark.parametrize("field,value", [("context", [.2] * 6), ("context", [float("nan")] * 6),
    ("query_rms", .250002), ("query_rms", -.01), ("before_bce", 5.),
    ("delta_bce", .1), ("loss_proposal", 1.), ("gradient_norm", -1.)])
def test_numeric_receipt_q_and_loss_tampering(synthetic, field, value):
    p, seal, _ = synthetic
    p["validation_records.json"][0][field] = value
    seal()
    with pytest.raises(ValueError):
        audit.verify(Path("/synthetic"))


def test_q_controls_and_frozen_before_must_match(synthetic):
    p, seal, _ = synthetic
    p["validation_records.json"][946]["context"] = [1., 0., 0., 0., 0., 0.]
    seal()
    with pytest.raises(ValueError, match="context"):
        audit.verify(Path("/synthetic"))


@pytest.mark.parametrize("change", ["early", "val", "task", "indices", "history"])
def test_training_schedule_rejects_role_early_task_and_unpaired_history(synthetic, change):
    p, seal, refs = synthetic
    if change == "early":
        p["train_records_selected.json"][1] = dict(refs["train"][0][2], origin_family=-1, query_family=5)
    elif change == "val":
        p["train_records_selected.json"][0] = p["val_records.json"][0]
    elif change == "task":
        p["schedule.json"][1]["task"] = "native"
    elif change == "indices":
        p["schedule.json"][0]["indices"] = [0, 0]
    else:
        p["training.json"][0]["indices"] = [1, 0]
    seal()
    with pytest.raises(ValueError):
        audit.verify(Path("/synthetic"))


@pytest.mark.parametrize("change", ["summary", "gate", "status", "protocol", "ids"])
def test_summary_gate_completion_protocol_and_current_roles(synthetic, change):
    p, seal, refs = synthetic
    if change == "summary":
        p["validation_summary.json"]["native"]["true_real"]["safe"]["mean_delta_bce"] += .01
    elif change == "gate":
        p["completed.json"]["gate"]["passed"] = False
    elif change == "status":
        p["completed.json"]["status"] = "RUNNING"
    elif change == "protocol":
        p["protocol.json"]["source_sha256"] = "other"
    else:
        refs["val"][1].remove(101)
    seal()
    with pytest.raises(ValueError):
        audit.verify(Path("/synthetic"))


def test_stable_double_softplus_and_rms_boundary():
    row = dict(before_logit=1000., after_logit=999., label=0., before_bce=1000.,
        after_bce=999., delta_bce=-1., context=[1., 0., 0., 0., 0., 0.],
        query_rms=.250001, accepted=True, loss_before=2., loss_proposal=1.,
        gradient_norm=.1, step_norm=.1)
    assert audit.check_numeric(row) <= 2e-6
    row.update(accepted=False, step_norm=.01)
    with pytest.raises(ValueError):
        audit.check_numeric(row)


def test_no_target_and_rejected_step_are_distinct_from_accepted():
    row = dict(before_logit=0., after_logit=0., label=1., before_bce=np.log(2.),
        after_bce=np.log(2.), delta_bce=0., context=[1., 0., 0., 0., 0., 0.],
        query_rms=0., accepted=False, loss_before=None, loss_proposal=None,
        gradient_norm=0., step_norm=0.)
    assert audit.check_numeric(row) == 0.
    row.update(loss_before=1., loss_proposal=1.1, gradient_norm=.5)
    assert audit.check_numeric(row) == 0.
    row["loss_proposal"] = .9
    with pytest.raises(ValueError):
        audit.check_numeric(row)


def test_macro_is_shot_equal_not_window_equal():
    rows = [dict(shot=1, group="safe", delta_bce=-1.)] * 2
    rows += [dict(shot=2, group="safe", delta_bce=.5),
             dict(shot=3, group="danger", delta_bce=.2)]
    result = audit.summarize_rows(rows)
    assert result["safe"]["mean_delta_bce"] == -.25
    assert result["early"]["mean_delta_bce"] is None
    assert result["balanced"] == (-.25 + .2) / 2


def test_fixed_gate_strict_margin_and_native_boundary():
    arms = {a: dict(balanced=0., safe=dict(mean_delta_bce=0.),
                   danger=dict(mean_delta_bce=0.)) for a in audit.ARMS}
    summary = {task: deepcopy(arms) for task in audit.TASKS}
    chosen = summary["no_MNM_CIII"]["true_real"]
    chosen.update(balanced=-1e-8, safe=dict(mean_delta_bce=-1e-8),
                  danger=dict(mean_delta_bce=-1e-8))
    gate = audit.compute_gate(summary)
    assert not gate["safe_improved"] and not gate["better_than_shared_real"]
    chosen.update(balanced=-2e-8, safe=dict(mean_delta_bce=-2e-8),
                  danger=dict(mean_delta_bce=-2e-8))
    summary["native"]["true_real"]["safe"]["mean_delta_bce"] = 1e-6
    assert audit.compute_gate(summary)["passed"]
    summary["native"]["true_real"]["safe"]["mean_delta_bce"] = 1.000001e-6
    assert not audit.compute_gate(summary)["native_protected"]


def test_import_has_no_trial_model_or_gpu_dependency():
    import ast

    tree = ast.parse(Path(audit.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("trial" in name or "model" in name or "torch" in name for name in imports)
