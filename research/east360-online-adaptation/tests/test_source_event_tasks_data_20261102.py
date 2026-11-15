"""CPU-only archive binding and source-task sampling contracts."""

from collections import Counter
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.event_prefix_probe import Event
from experiments.risk_aligned_online.source_event_tasks_20261102 import data


def row(shot=11, group="safe", **changes):
    return dict(shot=shot, origin=127, maturity=152, query=153, group=group,
                label=float(group == "danger"), lead_ms=None if group == "safe" else 20.,
                disruptive=group != "safe", feedback_observed=True, **changes)


@pytest.fixture
def archive(monkeypatch):
    roles = dict(train=[11, 12], val=[21, 22], test=[31])
    payload = {
        "protocol.json": dict(scope="JTEXT_TRAIN_CONTEXT_VECTOR_TEACHING_NOT_JTEXT_ONLINE",
            source_sha256=data.SOURCE_SHA, train_ids=roles["train"], val_ids=roles["val"],
            test_signals_read=0, east_signals_read=0, label="max(25ms, AdvancedTime)",
            query_lead_strict_ms=10., first_origin=127, maturity_offset=25,
            query_offset=[26, 50], max_per_group_per_shot=8),
        "audit_cpu_v1.json": dict(verified=True, raw_inputs=4, component_not_stream=True,
                                  validation_windows=2),
    }
    for role, ids in roles.items():
        if role == "test":
            continue
        payload[f"{role}_records.json"] = [row(ids[0]), row(ids[1], "danger")]
        payload[f"{role}_input_hashes.json"] = {str(sid): "a" * 64 for sid in ids}
    def seal(value):
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()
    payload["complete.json"] = dict(complete=True, source_unchanged=True,
        test_signals_read=0, east_signals_read=0,
        artifact_sha256={name: seal(value) for name, value in payload.items()
                        if name != "audit_cpu_v1.json"})
    monkeypatch.setattr(data, "EVIDENCE_SHA", {
        name: seal(payload[name]) for name in ("complete.json", "audit_cpu_v1.json")})
    monkeypatch.setattr(data, "_read", lambda name: (deepcopy(payload[name]), seal(payload[name])))
    monkeypatch.setattr(io23, "source_roles", lambda: deepcopy(roles))
    monkeypatch.setattr(io23, "advanced_times", lambda: {12: 40., 22: 40.})
    for name in ("read_signals", "read_outcome", "path_for"):
        monkeypatch.setattr(io23, name, lambda *a, **k: pytest.fail("raw/HDF access forbidden"))
    return payload, roles


@pytest.mark.parametrize("role", ["train", "val"])
def test_load_archive_binds_evidence_and_never_opens_raw(archive, role):
    payload, roles = archive
    rows, ids, provenance = data.load_records(role)
    assert ids == roles[role]
    assert all(r["role"] == role and r["device"] == "J-TEXT" for r in rows)
    assert "role" not in payload[f"{role}_records.json"][0]
    assert provenance["raw_inputs_rehashed"] is False
    assert provenance["source_sha256"] == data.SOURCE_SHA
    assert set(provenance["artifact_sha256"]) == {
        "complete.json", "audit_cpu_v1.json", "protocol.json",
        f"{role}_records.json", f"{role}_input_hashes.json"}


@pytest.mark.parametrize("role", ["test", "TEST", "EAST", "training", None])
def test_only_train_val_roles(role):
    with pytest.raises(ValueError):
        data.load_records(role)


@pytest.mark.parametrize("name", ["complete.json", "audit_cpu_v1.json", "protocol.json",
                                  "train_records.json", "train_input_hashes.json"])
def test_archive_tampering_rejected(archive, name):
    payload, _ = archive
    if isinstance(payload[name], dict):
        payload[name]["tampered"] = True
    else:
        payload[name][0]["shot"] = 31
    with pytest.raises(ValueError, match="hash"):
        data.load_records("train")


@pytest.mark.parametrize("changes", [dict(shot=31), dict(origin=128), dict(maturity=153),
    dict(query=152), dict(query=178), dict(label=1.), dict(disruptive=True),
    dict(feedback_observed=False), dict(role="test"), dict(device="EAST"),
    dict(lead_ms=12.)])
def test_record_semantics_rejected_even_if_manifest_resealed(archive, changes):
    payload, _ = archive
    payload["train_records.json"][0].update(changes)
    _reseal(payload)
    with pytest.raises(ValueError):
        data.load_records("train")


def _reseal(payload):
    """Keep the pinned completion mocked separately to exercise row validation."""
    records = payload["train_records.json"]
    payload["complete.json"]["artifact_sha256"]["train_records.json"] = (
        hashlib.sha256(json.dumps(records).encode()).hexdigest())
    data.EVIDENCE_SHA["complete.json"] = hashlib.sha256(
        json.dumps(payload["complete.json"]).encode()).hexdigest()


@pytest.mark.parametrize("group,lead,label,ok", [
    ("early", 41., 0., True), ("early", 30., 0., False),
    ("danger", 30., 1., True), ("danger", 41., 1., False),
    ("danger", 10., 1., False), ("early", float("nan"), 0., False)])
def test_advanced_time_labels_and_early_not_whole_disruption(archive, group, lead, label, ok):
    payload, _ = archive
    payload["train_records.json"][1].update(group=group, lead_ms=lead, label=label)
    _reseal(payload)
    if ok:
        assert data.load_records("train")[0][1]["group"] == group
    else:
        with pytest.raises(ValueError):
            data.load_records("train")


def test_current_roles_not_only_archived_roles(archive):
    _, roles = archive
    roles["train"] = [11, 31]
    with pytest.raises(ValueError):
        data.load_records("train")


@pytest.mark.parametrize("tamper", [False, True])
def test_load_events_hash_checks_parser_and_excludes_test_and_east(monkeypatch, tamper):
    from experiments.risk_aligned_online import event_prefix_probe as parser

    class Labels:
        def read_bytes(self):
            return b"tampered" if tamper else b"labels"

    path = Labels()
    monkeypatch.setattr(parser, "DEFAULT_LABELS", path)
    monkeypatch.setattr(parser, "LABEL_SHA256", hashlib.sha256(b"labels").hexdigest())
    monkeypatch.setattr(io23, "source_roles", lambda: dict(train=[11], val=[21], test=[31]))
    events = {sid: [Event(sid, 1, 5, 1., 2.)] for sid in (11, 21, 31, 74087)}
    metadata = {sid: dict(outcome=0) for sid in events}

    def read_events(actual):
        assert actual is path
        return events, metadata

    monkeypatch.setattr(parser, "read_events", read_events)
    if tamper:
        with pytest.raises(ValueError, match="hash"):
            data.load_events()
    else:
        grouped, meta = data.load_events()
        assert set(grouped) == set(meta) == {11, 21}


def test_attach_absolute_events_unknown_gap_end_and_nonordinal_families():
    times = .15 + np.arange(200) / 1000
    records = [row(), row(12)]
    grouped = {11: [Event(11, 1, 5, .277, .278), Event(11, 2, 0, .303, .304)]}
    frozen = deepcopy(records)
    attached = data.attach_event(records, {11: times, 12: times}, grouped)
    assert [(r["origin_family"], r["query_family"]) for r in attached] == [(5, 0), (-1, -1)]
    assert records == frozen
    grouped[11] = [Event(11, 1, 3, .276, .277)]
    assert data.attach_event(records, {11: times, 12: times}, grouped)[0]["origin_family"] == -1


@pytest.mark.parametrize("bad", ["missing", "short", "unordered", "mixed"])
def test_attach_rejects_invalid_clock_or_events(bad):
    times = {11: np.arange(200) / 1000}
    grouped = {}
    if bad == "missing":
        times = {}
    elif bad == "short":
        times[11] = times[11][:153]
    elif bad == "unordered":
        times[11][2] = times[11][1]
    else:
        grouped = {11: [Event(12, 1, 0, 0., 1.)]}
    with pytest.raises(ValueError):
        data.attach_event([row()], times, grouped)


@pytest.mark.parametrize("task,masked", [("native", []), ("no_MNM", [2]),
                                         ("no_MNM_CIII", [2, 1])])
def test_mask_task_only_copies_values_and_masks_availability(task, masked):
    values = np.arange(69, dtype=np.float32).reshape(3, 23)
    values[0, 4] = np.nan
    available = np.ones((3, 23), bool)
    available[0, 4] = False
    signal = dict(values=values, available=available)
    result = data.mask_task(signal, task)
    np.testing.assert_array_equal(result["values"], values)
    expected = available.copy()
    expected[:, masked] = False
    np.testing.assert_array_equal(result["available"], expected)
    assert not np.shares_memory(result["values"], values)
    assert not np.shares_memory(result["available"], available)
    assert available[:, 2].all()


@pytest.mark.parametrize("extra", ["label", "y", "times", "disruptive", "down"])
def test_mask_rejects_outcome_or_extra_fields(extra):
    with pytest.raises(ValueError):
        data.mask_task(dict(values=np.zeros((2, 23)), available=np.ones((2, 23), bool),
                            **{extra: None}), "native")


def test_mask_rejects_unknown_task_bad_shape_and_nonboolean_mask():
    for task, values, available in [("other", np.zeros((2, 23)), np.ones((2, 23), bool)),
        ("native", np.zeros((2, 22)), np.ones((2, 22), bool)),
        ("native", np.zeros((2, 23)), np.ones((2, 23)))]:
        with pytest.raises(ValueError):
            data.mask_task(dict(values=values, available=available), task)


def test_sampler_equal_stratum_then_shot_then_record_not_flat_windows():
    records = [row(11, origin_family=-1)] * 30 + [row(12, origin_family=-1)]
    records += [row(13, origin_family=5)] * 70 + [row(14, "danger", origin_family=1)]
    records += [row(15, "early", origin_family=0)]
    original = deepcopy(records)
    take = data.sample_indices(records, np.random.default_rng(131), per_class=6000)
    assert [records[i]["group"] for i in take] == ["safe"] * 6000 + ["danger"] * 6000
    counts = Counter(records[i]["shot"] for i in take[:6000])
    assert 2800 < counts[13] < 3200
    assert 1300 < counts[11] < 1700 and 1300 < counts[12] < 1700
    assert len(set(i for i in take if i < 30)) == 30
    assert records == original
    assert take == data.sample_indices(records, np.random.default_rng(131), per_class=6000)


@pytest.mark.parametrize("records,count", [([row()], 1), ([row(12, "danger")], 1),
    ([row(), row(12, "danger")], 0), ([row(), row(12, "danger")], True),
    ([row(origin_family="A"), row(12, "danger")], 1),
    ([row(role="val"), row(12, "danger")], 1),
    ([row(), row(11, "danger")], 1)])
def test_sampler_rejects_missing_group_invalid_family_or_role(records, count):
    with pytest.raises(ValueError):
        data.sample_indices(records, np.random.default_rng(131), per_class=count)


def test_census_counts_distinct_shots_and_keeps_early_separate():
    rows = [row(11, role="train", origin_family=-1, query_family=2)] * 3
    rows += [row(12, "danger", role="train", origin_family=5, query_family=2),
             row(12, "early", role="train", origin_family=5, query_family=-1)]
    summary = data.summarize_records(rows)
    assert summary["records"] == 5 and summary["shots"] == 2
    train = summary["roles"]["train"]
    assert train["groups"]["safe"] == dict(records=3, shots=1)
    assert train["groups"]["early"] == dict(records=1, shots=1)
    assert train["disruptive"]["true"] == dict(records=2, shots=1)
    assert train["origin_family"]["5"]["danger"] == dict(records=1, shots=1)
    assert train["query_family"]["-1"]["early"] == dict(records=1, shots=1)
