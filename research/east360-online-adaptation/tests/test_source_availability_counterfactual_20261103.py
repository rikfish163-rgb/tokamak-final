from contextlib import nullcontext
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online import (
    source_availability_counterfactual_20261103 as counterfactual,
)


def signal_fixture():
    values = np.arange(6 * 23, dtype=np.float32).reshape(6, 23)
    available = np.ones_like(values, dtype=bool)
    values[0, 1] = -10.0  # The pointwise sentinel must never be filled.
    values[1, 1] = -9.995  # Inside the same contract tolerance.
    values[2, 1] = 0.0  # Genuine available zero.
    values[3, 1] = 47.0
    available[3, 1] = False
    values[4, 1] = np.nan
    values[0, 2] = 0.0  # Genuine available MNM zero.
    values[1, 2] = 0.0  # Unavailable zero is not an observation.
    available[1, 2] = False
    return {"values": values, "available": available}


@pytest.mark.parametrize("channels", [(1,), (2,), (1, 2)])
def test_only_valid_selected_values_change_and_contract_flags_are_preserved(channels):
    signal = signal_fixture()
    original = {name: value.copy() for name, value in signal.items()}
    center = np.arange(23, dtype=np.float32) + 100
    contract = apply_input_contract(signal)

    result = counterfactual.neutralize_observed(signal, center, channels)

    np.testing.assert_array_equal(result["available"], contract["available"])
    expected = contract["values"].copy()
    for channel in channels:
        valid = contract["available"][:, channel] & np.isfinite(expected[:, channel])
        expected[valid, channel] = center[channel]
    np.testing.assert_array_equal(result["values"], expected)
    np.testing.assert_array_equal(signal["values"], original["values"])
    np.testing.assert_array_equal(signal["available"], original["available"])
    assert not np.shares_memory(result["values"], signal["values"])
    assert not np.shares_memory(result["available"], signal["available"])


def test_sentinel_is_masked_before_centering_and_zero_is_a_real_observation():
    signal = signal_fixture()
    center = np.arange(23, dtype=np.float32) + 100
    view = counterfactual.neutralize_observed(signal, center, (1, 2))

    assert not view["available"][0, 1] and view["values"][0, 1] == -10.0
    assert (
        not view["available"][1, 1] and view["values"][1, 1] == signal["values"][1, 1]
    )
    assert view["available"][2, 1] and view["values"][2, 1] == center[1]
    assert view["available"][0, 2] and view["values"][0, 2] == center[2]
    assert not view["available"][1, 2] and view["values"][1, 2] == 0.0
    assert view["available"][4, 1] and np.isnan(view["values"][4, 1])


def test_transform_is_prefix_local_and_window_padding_remains_unavailable():
    signal = signal_fixture()
    center = np.arange(23, dtype=np.float32) + 100
    full = counterfactual.neutralize_observed(signal, center, (1, 2))
    prefix = {name: value[:3].copy() for name, value in signal.items()}
    prefix_view = counterfactual.neutralize_observed(prefix, center, (1, 2))
    for name in ("values", "available"):
        np.testing.assert_array_equal(full[name][:3], prefix_view[name])

    values, available = simple_data.window(prefix_view, end=2, length=5)
    assert not available[:2].any()
    assert not values[:2].any()
    assert available[-1, 1] and values[-1, 1] == center[1]


@pytest.mark.parametrize("channels", [(), (0,), (3,), (1, 1), (True,), [1], ("1",)])
def test_rejects_non_declared_channel_tuples(channels):
    with pytest.raises((TypeError, ValueError)):
        counterfactual.neutralize_observed(signal_fixture(), np.zeros(23), channels)


@pytest.mark.parametrize(
    "center", [np.zeros(22), np.full(23, np.nan), np.full(23, np.inf)]
)
def test_rejects_invalid_source_centers(center):
    with pytest.raises(ValueError):
        counterfactual.neutralize_observed(signal_fixture(), center, (1,))


def test_rejects_outcome_or_other_metadata_and_nonboolean_flags():
    signal = signal_fixture()
    with pytest.raises(ValueError):
        counterfactual.neutralize_observed(
            {**signal, "is_disrupt": True}, np.zeros(23), (1,)
        )
    with pytest.raises(ValueError):
        counterfactual.neutralize_observed(
            {**signal, "available": signal["available"].astype(np.int8)},
            np.zeros(23),
            (1,),
        )


class CPUModel(torch.nn.Module):
    def __init__(self, center):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center))
        self.calls = []

    def forward(self, values, available):
        self.calls.append((values.clone(), available.clone()))
        return {"logit": values[:, -1, 3]}


def test_all_three_arms_use_cpu_256_batch_causal_windows():
    values = np.zeros((257, 23), np.float32)
    values[:, 3] = np.arange(257, dtype=np.float32) / 100
    available = np.ones_like(values, dtype=bool)
    center = np.arange(23, dtype=np.float32)
    model = CPUModel(center)

    curves = counterfactual.predict_curves(
        model, {"values": values, "available": available}, center
    )

    assert tuple(curves) == tuple(counterfactual.ARM_CHANNELS)
    assert [len(batch) for batch, _ in model.calls] == [256, 1] * 3
    for first_batch, first_flags in model.calls[::2]:
        assert first_batch.shape == (256, 128, 23)
        assert not first_flags[0, :-1].any()
        assert first_flags[0, -1].all()
        assert first_batch[0, -1, 3] == 0.0
    for curve in curves.values():
        assert curve.shape == (257,)
        np.testing.assert_allclose(
            curve, torch.sigmoid(torch.from_numpy(values[:, 3])).numpy()
        )


def test_paired_comparison_is_by_same_shot_and_strict_timing():
    times = np.array([0.989, 0.990])
    new = {name: np.array([0.8, 0.2]) for name in counterfactual.ARM_CHANNELS}
    ref = {name: np.array([0.2, 0.8]) for name in counterfactual.REFERENCE_ARMS}
    record = counterfactual.compare_curves(11, times, new, ref, True, 1.0, 0.7)

    assert record["shot"] == 11
    assert all(row["shot"] == 11 and row["timely"] for row in record["new"].values())
    assert all(
        row["shot"] == 11 and not row["timely"] for row in record["reference"].values()
    )
    for comparisons in record["comparisons"].values():
        assert set(comparisons) == set(counterfactual.REFERENCE_ARMS)
        assert all(row["timely_changed"] for row in comparisons.values())


def test_paired_comparison_rejects_wrong_shape_or_nonfinite_curves():
    times = np.array([0.0, 1.0])
    new = {name: np.zeros(2) for name in counterfactual.ARM_CHANNELS}
    ref = {name: np.zeros(2) for name in counterfactual.REFERENCE_ARMS}
    new[next(iter(new))] = np.zeros(1)
    with pytest.raises(ValueError):
        counterfactual.compare_curves(11, times, new, ref, False, None, 0.7)
    new[next(iter(new))] = np.array([0.0, np.nan])
    with pytest.raises(ValueError):
        counterfactual.compare_curves(11, times, new, ref, False, None, 0.7)


def test_completed_and_all_reference_files_are_pinned():
    manifest = counterfactual.reference_manifest(counterfactual.REFERENCE)
    assert manifest["completed.json"] == counterfactual.REFERENCE_COMPLETED_SHA256
    assert (
        counterfactual.manifest_digest(manifest)
        == counterfactual.REFERENCE_MANIFEST_SHA256
    )
    assert len([name for name in manifest if name.endswith(".npz")]) == 220
    assert len([name for name in manifest if name.endswith("_seal.json")]) == 220


def test_reference_raw_hash_must_match_the_same_shot():
    expected = {"11": "a" * 64, "12": "b" * 64}
    counterfactual.check_raw_hash(11, "a" * 64, expected)
    with pytest.raises(ValueError):
        counterfactual.check_raw_hash(11, "b" * 64, expected)
    with pytest.raises(ValueError):
        counterfactual.check_raw_hash(13, "a" * 64, expected)


def test_locked_parent_and_completed_reference_agree_on_cpu():
    model, parent = counterfactual.load_parent(counterfactual.DEFAULT_PARENT)
    ids = counterfactual.io23.source_roles()["test"]
    manifest, protocol, raw_hashes = counterfactual.load_reference(
        ids, float(parent["threshold"])
    )
    assert len(ids) == len(raw_hashes) == 220
    assert manifest["completed.json"] == counterfactual.REFERENCE_COMPLETED_SHA256
    assert protocol["source_sha256"] == counterfactual.PARENT_SHA256
    np.testing.assert_array_equal(model.center.numpy(), parent["source_center"])
    np.testing.assert_array_equal(model.scale.numpy(), parent["source_scale"])
    for name in (
        "io23.py",
        "simple_data.py",
        "simple23.py",
        "causal_input.py",
        "source_missing_diagnostics_20261103.py",
        "redesign_20261004/prepare_prior.py",
        "context_vector_20261030/source_run.py",
        "east_context_operator_20261031/run.py",
        "history_gradient_metric_20261102/run.py",
        "source_event_tasks_20261102/context_pair.py",
    ):
        path = (Path(counterfactual.__file__).parent / name).resolve()
        assert protocol["code_sha256"][str(path)] == counterfactual.sha(path)


@pytest.mark.parametrize("index", [1, 220])
def test_real_reference_curve_loader_verifies_shot_seal_and_time_grid(index):
    manifest = counterfactual.reference_manifest(counterfactual.REFERENCE)
    protocol = counterfactual.read(counterfactual.REFERENCE / "protocol.json")
    sid = protocol["ids"][index - 1]
    path = counterfactual.REFERENCE / "shots" / f"{index:04d}_{sid}.npz"
    with np.load(path, allow_pickle=False) as archive:
        times = archive["times"]
    curves = counterfactual.load_reference_curves(index, sid, times, manifest)
    assert set(curves) == set(counterfactual.REFERENCE_ARMS)
    assert all(
        curve.shape == times.shape and np.isfinite(curve).all()
        for curve in curves.values()
    )
    with pytest.raises(ValueError, match="time grid mismatch"):
        counterfactual.load_reference_curves(index, sid, times + 1, manifest)


def test_reference_digest_failure_is_rejected_before_reading_metadata(monkeypatch):
    monkeypatch.setattr(counterfactual, "reference_manifest", lambda path: {})
    with pytest.raises(ValueError, match="artifact manifest changed"):
        counterfactual.load_reference([], 0.7)


def test_existing_or_reference_output_is_rejected_before_cuda_configuration(
    monkeypatch,
):
    def forbidden():
        raise AssertionError("must reject output before CUDA configuration")

    monkeypatch.setattr(counterfactual, "configure", forbidden)
    with pytest.raises(FileExistsError):
        counterfactual.main(Path(counterfactual.__file__))
    with pytest.raises(ValueError, match="inside the sealed reference"):
        counterfactual.main(counterfactual.REFERENCE / "not-a-new-run-output")


def test_source_snapshot_detects_bitwise_changes_and_mode_changes():
    model = CPUModel(np.zeros(23, np.float32)).eval()
    frozen = counterfactual.snapshot_source(model)
    original_hash = counterfactual.source_snapshot_hash(frozen)
    counterfactual.assert_source_unchanged(model, frozen)
    model.center[0] = -0.0  # Numerically equal, but not the same source state.
    with pytest.raises(AssertionError):
        counterfactual.assert_source_unchanged(model, frozen)
    assert (
        counterfactual.source_snapshot_hash(counterfactual.snapshot_source(model))
        != original_hash
    )
    model.center[0] = 0.0
    model.train()
    with pytest.raises(AssertionError):
        counterfactual.assert_source_unchanged(model, frozen)


def test_new_curve_archive_uses_exclusive_creation(monkeypatch):
    class InMemoryArchive(io.BytesIO):
        def __exit__(self, *args):
            return False

    target = Path("/nonexistent/curve.npz")
    stream = InMemoryArchive()

    def fake_open(path, mode):
        assert path == target and mode == "xb"
        return stream

    monkeypatch.setattr(Path, "open", fake_open)
    times = np.array([0.0, 1.0])
    curves = {arm: np.array([0.2, 0.8]) for arm in counterfactual.ARM_CHANNELS}
    counterfactual.save_new_curves(target, times, curves)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        assert set(archive.files) == {"times", *counterfactual.ARM_CHANNELS}
        np.testing.assert_array_equal(archive["times"], times)


def test_output_seal_verification_detects_tampering(monkeypatch):
    root = Path("/nonexistent/counterfactual_output")
    seal = dict(
        caption=counterfactual.CAPTION,
        shot=11,
        curve_sha256="curvehash",
        raw_hdf5_sha256="rawhash",
        current_outcome_used_in_prediction=False,
    )
    monkeypatch.setattr(counterfactual, "read", lambda path: seal)
    monkeypatch.setattr(counterfactual, "sha", lambda path: "curvehash")
    counterfactual.verify_output_seals(root, [11], {"11": "rawhash"})
    monkeypatch.setattr(counterfactual, "sha", lambda path: "tampered")
    with pytest.raises(ValueError, match="output curve seal"):
        counterfactual.verify_output_seals(root, [11], {"11": "rawhash"})


def test_cpu_mocked_runner_seals_each_shot_before_outcome_and_reference(monkeypatch):
    ids = [11, 12]
    times = np.array([0.989, 0.990])
    outcomes = {11: (True, 1.0), 12: (False, None)}
    threshold = 0.7
    refs = {arm: np.array([0.2, 0.8]) for arm in counterfactual.REFERENCE_ARMS}
    new = {arm: np.array([0.8, 0.2]) for arm in counterfactual.ARM_CHANNELS}
    files, events, reveals = {}, [], []
    source_names = (
        "io23.py",
        "simple_data.py",
        "simple23.py",
        "causal_input.py",
        "source_missing_diagnostics_20261103.py",
        "redesign_20261004/prepare_prior.py",
        "context_vector_20261030/source_run.py",
        "east_context_operator_20261031/run.py",
        "history_gradient_metric_20261102/run.py",
        "source_event_tasks_20261102/context_pair.py",
    )
    code = {
        str((Path(counterfactual.__file__).parent / name).resolve()): "codehash"
        for name in source_names
    }
    code[str(Path(counterfactual.__file__).resolve())] = "codehash"
    source = CPUModel(np.zeros(23, np.float32)).eval()
    source.register_buffer("scale", torch.ones(23))
    source.cuda = lambda: source  # No CUDA call or device is used in this CPU test.
    parent = dict(
        threshold=threshold, source_center=np.zeros(23), source_scale=np.ones(23)
    )
    expected_hashes = {str(sid): f"raw{sid}" for sid in ids}
    bound_reference = {"test": "referencehash"}
    output = Path("/nonexistent/counterfactual_cpu_mock_output")

    monkeypatch.setattr(counterfactual, "configure", lambda: None)
    monkeypatch.setattr(counterfactual, "load_parent", lambda path: (source, parent))
    monkeypatch.setattr(counterfactual.io23, "source_roles", lambda: {"test": ids})
    monkeypatch.setattr(
        counterfactual,
        "load_reference",
        lambda shot_ids, t: (bound_reference, {"code_sha256": code}, expected_hashes),
    )
    monkeypatch.setattr(counterfactual, "imported_code_hashes", lambda: dict(code))
    monkeypatch.setattr(
        counterfactual, "reference_manifest", lambda path: bound_reference
    )
    monkeypatch.setattr(counterfactual, "source_gate", lambda shot_ids: nullcontext())
    monkeypatch.setattr(
        counterfactual.io23, "path_for", lambda device, sid: Path(f"/raw/{sid}")
    )
    monkeypatch.setattr(Path, "mkdir", lambda self: None)
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter(files))
    monkeypatch.setattr(Path, "is_file", lambda self: self in files)

    def fake_sha(path):
        path = Path(path)
        if path == counterfactual.DEFAULT_PARENT:
            return counterfactual.PARENT_SHA256
        if path.parent == Path("/raw"):
            return f"raw{path.name}"
        return "codehash" if str(path) in code else "artifacthash"

    def fake_write(path, payload):
        files[Path(path)] = payload
        events.append(("write", Path(path).name))

    def fake_save(path, actual_times, curves):
        files[Path(path)] = {"times": actual_times, **curves}
        events.append(("save", Path(path).name))

    def fake_signal(device, sid):
        with pytest.raises(RuntimeError, match="no truth"):
            counterfactual.io23.read_outcome(device, sid)
        events.append(("signal", sid))
        return dict(
            shot=sid,
            device=device,
            times=times,
            values=np.zeros((2, 23), np.float32),
            available=np.ones((2, 23), bool),
        )

    def fake_predict(model, signal, center):
        with pytest.raises(RuntimeError, match="no truth"):
            counterfactual.io23.read_outcome("J-TEXT", 11)
        events.append(("predict", None))
        return new

    def fake_outcome(device, sid):
        assert any(
            name.endswith(f"_{sid}_seal.json")
            for kind, name in events
            if kind == "write"
        )
        reveals.append(sid)
        events.append(("outcome", sid))
        return outcomes[sid]

    def fake_reference(index, sid, actual_times, manifest):
        assert reveals[-1] == sid
        np.testing.assert_array_equal(actual_times, times)
        events.append(("reference", sid))
        return refs

    def fake_read(path):
        if Path(path) in files:
            return files[Path(path)]
        assert Path(path).name == "per_shot.json" and reveals == ids
        return {
            arm: [
                counterfactual.score_shot(
                    sid, times, refs[arm], *outcomes[sid], threshold
                )
                for sid in ids
            ]
            for arm in counterfactual.REFERENCE_ARMS
        }

    monkeypatch.setattr(counterfactual, "sha", fake_sha)
    monkeypatch.setattr(counterfactual, "write", fake_write)
    monkeypatch.setattr(counterfactual, "read", fake_read)
    monkeypatch.setattr(counterfactual, "save_new_curves", fake_save)
    monkeypatch.setattr(counterfactual.simple_data, "read_shot", fake_signal)
    monkeypatch.setattr(counterfactual, "predict_curves", fake_predict)
    monkeypatch.setattr(counterfactual.io23, "read_outcome", fake_outcome)
    monkeypatch.setattr(counterfactual, "load_reference_curves", fake_reference)

    counterfactual.main(output)

    assert reveals == ids
    assert [
        x for x in events if x[0] in ("signal", "predict", "outcome", "reference")
    ] == [
        ("signal", 11),
        ("predict", None),
        ("outcome", 11),
        ("reference", 11),
        ("signal", 12),
        ("predict", None),
        ("outcome", 12),
        ("reference", 12),
    ]
    assert set(files[output / "shots/0001_11.npz"]) == {
        "times",
        *counterfactual.ARM_CHANNELS,
    }
    assert files[output / "completed.json"]["source_unchanged"] is True
    assert files[output / "summary.json"]["caption"] == counterfactual.CAPTION
