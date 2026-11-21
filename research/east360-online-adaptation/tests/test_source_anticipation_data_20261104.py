"""Source anticipation labels: synthetic metadata only, no raw shot reads."""

from copy import deepcopy
import math

import numpy as np
import pytest

from experiments.risk_aligned_online.source_anticipation_20261104 import data


def archived(shot=11, *, group="safe", label=0.0, disruptive=False, lead_ms=None):
    return dict(
        shot=shot,
        origin=127,
        maturity=152,
        query=153,
        group=group,
        label=label,
        disruptive=disruptive,
        lead_ms=lead_ms,
        feedback_observed=True,
        role="train",
        device="J-TEXT",
        note="preserve me",
    )


@pytest.mark.parametrize(
    "lead,source_label,future,timely,stratum,valid",
    [
        (-1.0, 0, 0, False, "post_disruption", False),
        (0.0, 1, 1, False, "disruptive_0_to_30ms", True),
        (25.0, 1, 1, False, "disruptive_0_to_30ms", True),
        (30.0, 0, 1, False, "disruptive_0_to_30ms", True),
        (30.001, 0, 1, True, "disruptive_30_to_125ms", True),
        (60.0, 0, 1, True, "disruptive_30_to_125ms", True),
        (125.0, 0, 1, True, "disruptive_30_to_125ms", True),
        (125.001, 0, 0, True, "disruptive_earlier_than_125ms", True),
    ],
)
def test_disruptive_boundaries_are_explicit(lead, source_label, future, timely, stratum, valid):
    row = archived()
    before = deepcopy(row)
    out = data.annotate(row, disruptive=True, down=lead / 1000, query_time=0.0, horizon_ms=25.0)
    assert row == before and all(out[k] == v for k, v in before.items())
    assert out["hdf5_disruptive"] is True
    assert out["original_source_point_label"] == source_label
    assert out["future_disruption_125ms"] == future
    assert out["timely_eligible"] is timely
    assert out["anticipation_stratum"] == stratum
    assert out["hdf5_source_point_valid"] is valid


def test_archived_early_negative_becomes_auxiliary_positive_without_relabelling():
    row = archived(group="early", disruptive=True, lead_ms=60.0)
    out = data.annotate(row, disruptive=True, down=1.06, query_time=1.0, horizon_ms=25.0)
    assert out["label"] == 0.0 and out["group"] == "early"
    assert out["original_source_point_label"] == 0 and out["future_disruption_125ms"] == 1
    assert out["anticipation_stratum"] == "disruptive_30_to_125ms"
    assert not out["any_archived_mismatch"]


def test_early_disruptive_negative_is_never_called_safe():
    row = archived(group="early", disruptive=True, lead_ms=200.0)
    out = data.annotate(row, disruptive=True, down=1.2, query_time=1.0, horizon_ms=25.0)
    assert out["future_disruption_125ms"] == 0 and out["timely_eligible"]
    assert out["anticipation_stratum"] == "disruptive_earlier_than_125ms"
    assert out["hdf5_disruptive"] is True and not out["any_archived_mismatch"]


def test_hdf_outcome_wins_but_archived_fields_and_disagreements_remain_visible():
    row = archived(group="danger", label=1.0, disruptive=True, lead_ms=20.0)
    out = data.annotate(row, disruptive=False, down=99.0, query_time=1.0, horizon_ms=25.0)
    assert out["label"] == 1.0 and out["group"] == "danger" and out["disruptive"] is True
    assert out["hdf5_disruptive"] is False and out["hdf5_down_time"] is None
    assert out["hdf5_lead_ms"] is None and out["original_source_point_label"] == 0
    assert out["future_disruption_125ms"] == 0 and out["timely_eligible"] is True
    assert out["anticipation_stratum"] == "safe"
    assert out["any_archived_mismatch"] and all(out["archived_mismatches"].values())


def test_archive_label_and_lead_mismatches_are_independent():
    row = archived(group="danger", label=1.0, disruptive=True, lead_ms=20.0)
    out = data.annotate(row, disruptive=True, down=1.06, query_time=1.0, horizon_ms=25.0)
    assert out["archived_mismatches"] == dict(disruptive=False, label=True, group=True, lead_ms=True)
    assert out["label"] == 1.0 and out["group"] == "danger"


def test_source_horizon_and_canonical_tolerance_are_distinct_from_auxiliary():
    row = archived(group="danger", label=1.0, disruptive=True, lead_ms=150.0)
    out = data.annotate(row, disruptive=True, down=0.15, query_time=0.0, horizon_ms=200.0)
    assert out["original_source_point_label"] == 1 and out["future_disruption_125ms"] == 0
    row = archived(group="danger", label=1.0, disruptive=True, lead_ms=25.0000005)
    out = data.annotate(row, disruptive=True, down=0.0250000005, query_time=0.0, horizon_ms=25.0)
    assert out["original_source_point_label"] == 1  # io23's 1e-6 ms tolerance


@pytest.mark.parametrize(
    "query,down,future,timely,stratum",
    [
        (1.0, 1.03, 1, False, "disruptive_0_to_30ms"),
        (1.0, 1.030000001, 1, True, "disruptive_30_to_125ms"),
        (0.15, 0.275, 1, True, "disruptive_30_to_125ms"),
        (0.15, 0.275000001, 0, True, "disruptive_earlier_than_125ms"),
    ],
)
def test_decimal_clock_boundaries_are_not_misclassified_by_float_roundoff(
    query, down, future, timely, stratum
):
    out = data.annotate(archived(), disruptive=True, down=down, query_time=query, horizon_ms=25.0)
    assert out["future_disruption_125ms"] == future
    assert out["timely_eligible"] is timely
    assert out["anticipation_stratum"] == stratum


@pytest.mark.parametrize("failure", ["outcome_type", "missing_down", "nan_down", "nan_clock", "horizon", "label", "group", "collision"])
def test_annotate_rejects_malformed_inputs_without_fallback(failure):
    row = archived()
    kwargs = dict(disruptive=True, down=1.2, query_time=1.0, horizon_ms=25.0)
    if failure == "outcome_type":
        kwargs["disruptive"] = 1
    elif failure == "missing_down":
        kwargs["down"] = None
    elif failure == "nan_down":
        kwargs["down"] = float("nan")
    elif failure == "nan_clock":
        kwargs["query_time"] = float("nan")
    elif failure == "horizon":
        kwargs["horizon_ms"] = 24
    elif failure == "label":
        row["label"] = 2
    elif failure == "group":
        row["group"] = "unknown"
    else:
        row["future_disruption_125ms"] = 1
    with pytest.raises((TypeError, ValueError)):
        data.annotate(row, **kwargs)


@pytest.fixture
def synthetic_archive(monkeypatch):
    rows = [archived(11), archived(12, group="early", disruptive=True, lead_ms=60.0)]
    second = archived(11)
    second.update(origin=152, maturity=177, query=178)
    rows.append(second)
    calls = []

    def loader(role):
        assert role == "train"
        return deepcopy(rows), [11, 12], {"role": "train", "source_sha256": "synthetic"}

    def reader(shot):
        calls.append(shot)
        return dict(disruptive=shot == 12, down=1.213 if shot == 12 else None,
                    start_time=1.0, sample_rate_hz=1000.0, samples=500)

    monkeypatch.setattr(data.source_archive, "load_records", loader)
    monkeypatch.setattr(data.io23, "advanced_times", lambda: {12: 40.0})
    return rows, calls, reader


def test_build_records_uses_locked_role_and_one_metadata_read_per_shot(synthetic_archive):
    rows, calls, reader = synthetic_archive
    out, receipt = data.build_records("train", metadata_reader=reader)
    assert len(out) == 3 and calls == [11, 12]
    assert receipt["role"] == "train" and receipt["records"] == 3 and receipt["shots"] == 2
    assert receipt["metadata_shots_read"] == 2 and receipt["records_with_mismatch"] == 0
    assert receipt["metadata_source"] == "injected"
    assert out[1]["source_horizon_ms"] == 40.0
    assert out[1]["hdf5_query_time"] == pytest.approx(1.153)
    assert out[1]["original_source_point_label"] == 0 and out[1]["future_disruption_125ms"] == 1
    assert all(all(o[k] == v for k, v in r.items()) for r, o in zip(rows, out))
    assert all("origin_family" not in r and "query_family" not in r for r in out)


def test_builder_counts_hdf_archive_disagreements_instead_of_rewriting(synthetic_archive):
    rows, _, reader = synthetic_archive
    rows[1].update(group="danger", label=1.0, lead_ms=20.0)
    out, receipt = data.build_records("train", metadata_reader=reader)
    assert out[1]["label"] == 1.0 and out[1]["group"] == "danger"
    assert out[1]["original_source_point_label"] == 0 and out[1]["future_disruption_125ms"] == 1
    assert receipt["records_with_mismatch"] == 1
    assert receipt["mismatch_counts"] == dict(disruptive=0, label=1, group=1, lead_ms=1)


def test_builder_metadata_fingerprint_is_stable_and_detects_changes(synthetic_archive):
    _, _, reader = synthetic_archive
    _, first = data.build_records("train", metadata_reader=reader)
    _, second = data.build_records("train", metadata_reader=reader)

    def changed(shot):
        value = reader(shot)
        if shot == 12:
            value["down"] += 0.001
        return value

    _, third = data.build_records("train", metadata_reader=changed)
    assert len(first["source_metadata_fields_sha256"]) == 64
    assert first["source_metadata_fields_sha256"] == second["source_metadata_fields_sha256"]
    assert first["source_metadata_fields_sha256"] != third["source_metadata_fields_sha256"]


def test_builder_uses_current_advanced_times_without_accepting_override(synthetic_archive, monkeypatch):
    _, _, reader = synthetic_archive
    monkeypatch.setattr(data.io23, "advanced_times", lambda: {12: 100.0})
    out, _ = data.build_records("train", metadata_reader=reader)
    assert out[1]["source_horizon_ms"] == 100.0
    assert out[1]["original_source_point_label"] == 1 and out[1]["label"] == 0.0
    assert out[1]["archived_mismatches"]["label"]
    with pytest.raises(TypeError):
        data.build_records("train", metadata_reader=reader, advanced_times={12: 25.0})


@pytest.mark.parametrize("role", ["test", "EAST", "TRAIN", ""])
def test_builder_rejects_unapproved_roles_before_readers(role, synthetic_archive):
    _, calls, reader = synthetic_archive
    with pytest.raises(ValueError):
        data.build_records(role, metadata_reader=reader)
    assert calls == []


@pytest.mark.parametrize("failure", ["unauthorized_shot", "wrong_role", "east_device", "duplicate_query", "origin", "maturity", "query", "query_outside_clock"])
def test_builder_validates_identity_and_maturity_without_silent_repair(synthetic_archive, failure):
    rows, calls, reader = synthetic_archive
    if failure == "unauthorized_shot":
        rows[1]["shot"] = 999
    elif failure == "wrong_role":
        rows[1]["role"] = "val"
    elif failure == "east_device":
        rows[1]["device"] = "EAST"
    elif failure == "duplicate_query":
        rows.append(deepcopy(rows[0]))
    elif failure == "origin":
        rows[1]["origin"] = 128
    elif failure == "maturity":
        rows[1]["maturity"] = 151
    elif failure == "query":
        rows[1]["query"] = 152
    else:
        def reader(shot):
            calls.append(shot)
            return dict(disruptive=False, down=None, start_time=1.0, sample_rate_hz=1000.0, samples=100)
    with pytest.raises(ValueError):
        data.build_records("train", metadata_reader=reader)
    if failure != "query_outside_clock":
        assert calls == []


@pytest.mark.parametrize("bad", ["missing", "nan_start", "bad_rate", "bad_samples", "bad_flag", "bad_down"])
def test_builder_rejects_bad_metadata(synthetic_archive, bad):
    _, _, _reader = synthetic_archive

    def reader(_shot):
        m = dict(disruptive=True, down=2.0, start_time=1.0, sample_rate_hz=1000.0, samples=500)
        if bad == "missing":
            del m["start_time"]
        elif bad == "nan_start":
            m["start_time"] = math.nan
        elif bad == "bad_rate":
            m["sample_rate_hz"] = 999.0
        elif bad == "bad_samples":
            m["samples"] = 0
        elif bad == "bad_flag":
            m["disruptive"] = 3
        else:
            m["down"] = math.nan
        return m

    with pytest.raises((ValueError, TypeError, KeyError)):
        data.build_records("train", metadata_reader=reader)


def test_default_hdf_reader_reads_only_outcomes_and_ip_metadata(monkeypatch):
    accessed = []

    class Scalar:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, index):
            assert index == ()
            return self.value

    class IP:
        shape = (500,)
        ndim = 1
        attrs = {"StartTime": 1.0, "SampleRate": 1000.0}

        def __getitem__(self, _index):
            raise AssertionError("signal array must not be read")

    class File:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getitem__(self, key):
            accessed.append(key)
            return {"meta/IsDisrupt": Scalar(np.array([1])),
                    "meta/DownTime": Scalar(np.array([1.5])), "data/ip": IP()}[key]

    def open_fake(path, mode):
        assert mode == "r" and "J-TEXT" in str(path) and "1051510" in str(path)
        return File()

    monkeypatch.setattr(data.h5py, "File", open_fake)
    out = data._read_metadata(1051510, allowed_ids={1051510})
    assert out == dict(disruptive=True, down=1.5, start_time=1.0, sample_rate_hz=1000.0, samples=500)
    assert accessed == ["meta/IsDisrupt", "meta/DownTime", "data/ip"]


def test_default_reader_rejects_unlocked_shot_before_open(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("out-of-role HDF5 must not be opened")

    monkeypatch.setattr(data.h5py, "File", forbidden)
    with pytest.raises(ValueError, match="locked"):
        data._read_metadata(999, allowed_ids={1051510})
