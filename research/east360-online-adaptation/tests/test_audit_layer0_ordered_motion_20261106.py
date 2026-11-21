"""Synthetic-only tests for the independent ordered-motion recount."""

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

import audit_layer0_ordered_motion_20261106 as audit


def test_ordered_motion_reversal_flips_sign_with_identical_support():
    rng = np.random.default_rng(901)
    z = rng.normal(size=(2, 25, 23))
    available = np.ones_like(z, dtype=bool)
    available[0, 5, 3] = False
    origin_valid = np.ones((2, 23), dtype=bool)
    origin_valid[1, 4] = False

    actual, mask = audit.ordered_block(z, available, origin_valid)
    reversed_actual, reversed_mask = audit.ordered_block(
        z[:, ::-1], available[:, ::-1], origin_valid
    )

    np.testing.assert_array_equal(mask, reversed_mask)
    np.testing.assert_allclose(actual, -reversed_actual, atol=1e-14)
    assert actual[0, 0, 7] == pytest.approx(
        0.5 * ((z[0, 5, 7] - z[0, 0, 7]) + (z[0, 24, 7] - z[0, 19, 7]))
    )
    assert not mask[0, 0, 3]  # an endpoint is missing
    assert not mask[1, :, 4].any()  # origin itself is missing
    assert not mask[:, :, [2, 10]].any()
    assert np.count_nonzero(actual[~mask]) == 0


@pytest.mark.parametrize("endpoint", [0, 5, 19, 24])
def test_every_endpoint_is_required(endpoint):
    z = np.zeros((1, 25, 23))
    observed = np.ones_like(z, dtype=bool)
    observed[0, endpoint, 5] = False
    _, valid = audit.ordered_block(z, observed, np.ones((1, 23), dtype=bool))
    assert not valid[0, 0, 5]


def test_sampling_boundaries_are_exact():
    assert audit.source_origins(202).tolist() == []
    assert audit.source_origins(203).tolist() == [177]
    assert audit.source_origins(206).tolist() == [177, 178, 179, 180]
    assert len(audit.source_origins(1000)) == 16
    assert audit.cold_origins(231).tolist() == []
    assert audit.cold_origins(232).tolist() == [206]
    assert audit.cold_origins(264).tolist() == [206, 238]


def test_common_past_support_origin_parity_and_observation_guards():
    n = 230
    time = np.arange(n, dtype=float)
    z = np.broadcast_to(time[:, None] * 0.01, (n, 23)).copy()
    values = np.sinh(z)
    available = np.ones((n, 23), dtype=bool)
    origins = np.array([177, 178])
    available[177 - 24, 3] = False  # past j=0 endpoint
    available[177 + 1, 4] = False  # future j=0 endpoint
    values[177 - 24, 3] = np.nan
    values[177 + 1, 4] = np.nan
    target, common, persistence, past, origin = audit.reconstruct(
        values, available, origins, "val", (np.zeros(23), np.ones(23))
    )
    assert origin[:, 1].tolist() == [True, False]
    assert not origin[:, 2].any()
    assert not past[1, :, 1].any()
    assert not common[1, :, 1].any()  # origin validity also masks future
    assert not past[:, :, [2, 10]].any()
    assert not common[:, :, [2, 10]].any()
    assert not past[0, 0, 3] and not common[0, 0, 3]
    assert past[0, 0, 4] and not common[0, 0, 4]
    assert persistence[0, 0, 4] == pytest.approx(0.05)
    assert target[0, 0, 5] == pytest.approx(0.05)
    assert np.count_nonzero(persistence[~past]) == 0


def test_window_smooth_l1_uses_one_common_mask_and_guards():
    predictions = np.zeros((2, 6, 10, 23))
    target = np.zeros((2, 10, 23))
    mask = np.zeros((2, 10, 23), dtype=bool)
    mask[0, 0, :2] = True
    predictions[0, 0, 0, :2] = [1, 2]
    predictions[0, 1, 0, :2] = [0, -1]
    predictions[:, :, 9, 22] = 100000  # ignored for every arm
    errors, supported, counts = audit.window_losses(predictions, target, mask)
    assert supported.tolist() == [True, False]
    assert counts.tolist() == [2, 0]
    assert errors[0, 0] == pytest.approx(1.0)  # (0.5 + 1.5) / 2
    assert errors[0, 1] == pytest.approx(0.25)
    assert np.all(errors[1] == 0)
    with pytest.raises(ValueError):
        audit.window_losses(predictions, target, mask.astype(int))
    predictions[1, 5, 9, 22] = np.nan
    with pytest.raises(ValueError):
        audit.window_losses(predictions, target, mask)


def _shot(shot, means, supported=1):
    return {"shot": shot, "n_supported": supported, "mean": np.array(means, float)}


def test_group_is_shot_equal_and_gate_is_strict_on_means_and_majorities():
    rows = [
        _shot(1, [5, 5, 5, 0, 2, 5], 30),
        _shot(2, [5, 5, 5, 0, 2, 5], 1),
        _shot(3, [0, 0, 0, 1, 0, 0], 1),
    ]
    result = audit.summarize(rows, n_shots=3, n_forecasts=32, valid_cells=100)
    assert result["mean"]["joint"] == pytest.approx(1 / 3)
    assert result["joint_lower_loss_shots"]["zero"] == 2
    assert result["pass_gate"]
    rows[1] = _shot(2, [0, 0, 0, 1, 2, 0])
    assert not audit.summarize(rows, 3, 32, 100)["pass_gate"]
    # Reverse is subject to the group mean gate even though it is not a
    # majority comparator; ties must fail.
    tied_reverse = [_shot(1, [5, 5, 5, 1, 1, 5])]
    assert not audit.summarize(tied_reverse, 1, 1, 1)["pass_gate"]


def _synthetic_group(role, ids=(17, 18)):
    """Create sealed forecasts from simple equations, independently of audit code."""
    n = 300 if role == "cold" else 206
    inventory, arrays = [], {k: [] for k in audit.PREDICTION_KEYS}
    signals = {}
    for sid in ids:
        t = np.arange(n, dtype=float)
        z = np.broadcast_to((0.00005 * (t - 100) ** 2)[:, None], (n, 23)).copy()
        values = np.sinh(z)
        available = np.ones_like(values, dtype=bool)
        if role == "cold":
            available[:, 2] = False
            origins = [o for o in range(135 - 25, n - 25, 32) if o >= 177]
            device = "EAST"
        else:
            origins = [177, 178, 179, 180]
            device = "J-TEXT"
        inventory.append(
            dict(shot=sid, raw_sha256="synthetic", n_points=n, n_forecasts=len(origins))
        )
        signals[sid] = dict(shot=sid, device=device, values=values, available=available)
        for parity, origin in enumerate(origins):
            origin_valid = available[origin].copy()
            if role == "val":
                origin_valid[2] = False
                if parity % 2:
                    origin_valid[1] = False
            past_valid = np.zeros((10, 23), dtype=bool)
            persistence = np.zeros((10, 23))
            target = np.zeros((10, 23))
            for j in range(10):
                points = (j, j + 5, 19 - j, 24 - j)
                for c in range(23):
                    valid = origin_valid[c] and c not in (2, 10)
                    valid = valid and all(available[origin - 24 + k, c] for k in points)
                    past_valid[j, c] = valid
                    if valid:
                        p = [math.asinh(values[origin - 24 + k, c]) for k in points]
                        persistence[j, c] = ((p[1] - p[0]) + (p[3] - p[2])) / 2
                    f = [math.asinh(values[origin + 1 + k, c]) for k in points]
                    target[j, c] = ((f[1] - f[0]) + (f[3] - f[2])) / 2
            local, joint = target + 0.02, target.copy()
            predictions = np.stack(
                (
                    np.zeros_like(target),
                    persistence,
                    local,
                    joint,
                    2 * local - joint,
                    target + 0.05,
                )
            )
            for key, item in dict(
                shot=sid,
                origin=origin,
                predictions=predictions,
                origin_valid=origin_valid,
                past_valid=past_valid,
            ).items():
                arrays[key].append(item)
    return {k: np.asarray(v) for k, v in arrays.items()}, inventory, signals


@pytest.mark.parametrize("role", ["val", "cold"])
def test_synthetic_group_recounts_sealed_arms_and_equal_shots(role):
    packed, inventory, signals = _synthetic_group(role)
    calls = []

    def reader(device, sid, **kwargs):
        calls.append((device, sid, kwargs))
        return signals[sid]

    def prepare(raw, index, mapping):
        assert index == 21 and mapping == {"synthetic": True}
        calls.append(("prepare", raw["shot"]))
        return raw

    result = audit.score_group(
        packed,
        inventory,
        role,
        {"synthetic": True},
        (np.zeros(23), np.ones(23)),
        lambda path: "synthetic",
        lambda device, sid: str(sid),
        reader,
        prepare,
    )
    assert result["n_shots"] == result["supported_shots"] == 2
    assert result["supported_forecasts"] == len(packed["shot"])
    assert result["pass_gate"]
    assert result["mean"]["joint"] == pytest.approx(0, abs=1e-25)
    assert result["mean"]["reverse"] > result["mean"]["local"]
    assert result["joint_lower_loss_shots"]["stale"] == 2
    assert sum(c[0] == "prepare" for c in calls) == (2 if role == "cold" else 0)
    assert all(
        c[2] == {"rate_contract": "nominal"}
        for c in calls
        if role == "cold" and c[0] == "EAST"
    )


@pytest.mark.parametrize(
    "kind",
    [
        "zero",
        "persistence",
        "persistence_invalid",
        "midpoint",
        "past",
        "origin",
        "duplicate",
        "schedule",
    ],
)
def test_seal_and_schedule_tampering_is_rejected(kind):
    packed, inventory, signals = _synthetic_group("val")
    if kind == "zero":
        packed["predictions"][0, 0, 0, 2] = 1
    elif kind == "persistence":
        packed["predictions"][0, 1, 0, 3] += 1
    elif kind == "persistence_invalid":
        packed["predictions"][0, 1, 0, 2] = (
            1e-12  # even sub-tolerance values are not zero
        )
    elif kind == "midpoint":
        packed["predictions"][0, 4, 0, 3] += 1
    elif kind == "past":
        packed["past_valid"][0, 0, 3] = False
    elif kind == "origin":
        packed["origin_valid"][1, 1] = True
    elif kind == "duplicate":
        packed["origin"][1] = packed["origin"][0]
    else:
        packed["origin"][0] -= 1
    with pytest.raises(ValueError):
        audit.score_group(
            packed,
            inventory,
            "val",
            {},
            (np.zeros(23), np.ones(23)),
            lambda p: "synthetic",
            lambda d, s: str(s),
            lambda d, s, **kw: signals[s],
            lambda *a: None,
        )


def test_stale_is_sealed_and_scored_but_not_reconstructed():
    packed, inventory, signals = _synthetic_group("val")
    packed["predictions"][:, 5] = 99
    result = audit.score_group(
        packed,
        inventory,
        "val",
        {},
        (np.zeros(23), np.ones(23)),
        lambda p: "synthetic",
        lambda d, s: str(s),
        lambda d, s, **kw: signals[s],
        lambda *a: None,
    )
    assert result["mean"]["stale"] > 90


def test_float32_sealed_predictions_match_independent_float64_recount():
    packed, inventory, signals = _synthetic_group("val")
    packed["predictions"] = packed["predictions"].astype(np.float32)
    for signal in signals.values():
        signal["values"] = signal["values"].astype(np.float32)
    result = audit.score_group(
        packed,
        inventory,
        "val",
        {},
        (np.zeros(23), np.ones(23)),
        lambda p: "synthetic",
        lambda d, s: str(s),
        lambda d, s, **kw: signals[s],
        lambda *a: None,
    )
    assert result["pass_gate"]


def test_normalization_validation_rejects_nonfinite_and_nonpositive_scale():
    audit.check_norm((np.zeros(23), np.ones(23)))
    for norm in (
        (np.zeros(22), np.ones(23)),
        (np.zeros(23), np.zeros(23)),
        (np.full(23, np.nan), np.ones(23)),
    ):
        with pytest.raises(ValueError):
            audit.check_norm(norm)


def _metadata():
    digest = "a" * 64
    names = (
        "protocol.json",
        "inventory.json",
        "head.npz",
        "normalization.npz",
        "val_predictions.npz",
        "cold_predictions.npz",
    )
    roles = {"train": [1, 2], "val": [3], "test": [4]}
    mapping = {"calibration_ids": list(range(20, 40))}
    cfg = dict(
        training=roles["train"],
        validation=roles["val"],
        cold_signal_only=mapping["calibration_ids"],
        source_state_sha256=digest,
        inputs={"/synthetic/map.json": digest, "/synthetic/source.pt": digest},
        mapping="/synthetic/map.json",
        arms=list(audit.ARMS),
        forecast_horizon=25,
        window=128,
        shape_lag=5,
        pair_offset=25,
        start_origin=177,
        outcomes_read=0,
        source_test_signals_read=0,
        source_risk_updates=0,
        first=135,
        stride=32,
    )

    def row(s):
        return dict(shot=s, raw_sha256=digest, n_points=300, n_forecasts=16)

    inventory = {
        "train": [row(x) for x in roles["train"]],
        "val": [row(x) for x in roles["val"]],
        "cold": [row(x) for x in mapping["calibration_ids"]],
    }
    done = dict(
        status="COMPLETE_ORDERED_MOTION_SOURCE_AND_HELD_FORECASTS",
        artifact_sha256={n: digest for n in names},
    )
    return done, cfg, inventory, roles, mapping


@pytest.mark.parametrize(
    "corruption",
    [
        "status",
        "artifact",
        "artifact_changed",
        "input_changed",
        "traversal",
        "mapping_binding",
        "arm",
        "source_state",
        "flag",
        "schedule",
        "train_schedule",
        "roles",
        "cold",
    ],
)
def test_protocol_and_inventory_guards(corruption):
    done, cfg, inventory, roles, mapping = _metadata()
    audit.check_metadata(
        Path("/synthetic/run"), done, cfg, inventory, roles, mapping, lambda p: "a" * 64
    )
    if corruption == "status":
        done["status"] = "INCOMPLETE"
    elif corruption == "artifact":
        del done["artifact_sha256"]["val_predictions.npz"]
    elif corruption == "artifact_changed":
        done["artifact_sha256"]["val_predictions.npz"] = "b" * 64
    elif corruption == "input_changed":
        cfg["inputs"]["/synthetic/source.pt"] = "b" * 64
    elif corruption == "traversal":
        done["artifact_sha256"]["../extra"] = "a" * 64
    elif corruption == "mapping_binding":
        del cfg["inputs"][cfg["mapping"]]
    elif corruption == "arm":
        cfg["arms"][0] = "other"
    elif corruption == "source_state":
        cfg["source_state_sha256"] = "wrong"
    elif corruption == "flag":
        cfg["outcomes_read"] = 1
    elif corruption == "schedule":
        cfg["start_origin"] = 176
    elif corruption == "train_schedule":
        inventory["train"][0]["n_forecasts"] -= 1
    elif corruption == "roles":
        inventory["val"][0]["shot"] = 4
    else:
        mapping["calibration_ids"] = mapping["calibration_ids"][:-1]
    with pytest.raises(ValueError):
        audit.check_metadata(
            Path("/synthetic/run"),
            done,
            cfg,
            inventory,
            roles,
            mapping,
            lambda p: "a" * 64,
        )


def test_command_line_forwards_two_positional_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(audit, "verify", lambda run, out: calls.append((run, out)))
    audit.main(["/synthetic/fit", "/synthetic/result.json"])
    assert calls == [(Path("/synthetic/fit"), Path("/synthetic/result.json"))]


def test_hash_guard_detects_mutation_without_filesystem_writes(monkeypatch):
    path = Path("/synthetic/sealed")
    signature = [1]
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(
        Path, "open", lambda self, mode: io.BytesIO(b"sealed synthetic bytes")
    )
    monkeypatch.setattr(audit, "signature", lambda p: signature[0])
    hashes = audit.Hashes()
    expected = hashlib.sha256(b"sealed synthetic bytes").hexdigest()
    assert hashes(path) == hashes(path) == expected
    hashes.check()
    signature[0] += 1
    with pytest.raises(ValueError):
        hashes(path)
    with pytest.raises(ValueError):
        hashes.check()


def test_locked_source_reference_requires_both_seal_and_matching_state(monkeypatch):
    ref = Path("/synthetic/source-reference")
    mapping_path = Path("/synthetic/map.json")
    digest, state = "a" * 64, "b" * 64
    payloads = {
        str(ref / "completed.json"): {"artifact_sha256": {"protocol.json": digest}},
        str(ref / "protocol.json"): {
            "source_state_sha256": state,
            "input_sha256": {str(mapping_path): digest},
        },
    }
    monkeypatch.setattr(audit, "SOURCE_REFERENCE", ref)
    monkeypatch.setattr(audit, "SOURCE_REFERENCE_COMPLETED_SHA256", digest)
    monkeypatch.setattr(audit, "read_json", lambda path: payloads[str(path)])
    audit.check_source_reference(state, mapping_path, lambda path: digest)
    with pytest.raises(ValueError):
        audit.check_source_reference("c" * 64, mapping_path, lambda path: digest)
    with pytest.raises(ValueError):
        audit.check_source_reference(state, mapping_path, lambda path: "c" * 64)
    with pytest.raises(ValueError):
        audit.check_source_reference(
            state, Path("/synthetic/another-map.json"), lambda path: digest
        )
    payloads[str(ref / "protocol.json")]["input_sha256"][str(mapping_path)] = "c" * 64
    with pytest.raises(ValueError):
        audit.check_source_reference(state, mapping_path, lambda path: digest)
    payloads[str(ref / "protocol.json")]["input_sha256"][str(mapping_path)] = digest
    payloads[str(ref / "completed.json")]["artifact_sha256"]["protocol.json"] = "c" * 64
    with pytest.raises(ValueError):
        audit.check_source_reference(state, mapping_path, lambda path: digest)


def test_output_must_not_overwrite_existing_path(monkeypatch):
    monkeypatch.setattr(audit, "signal_dependencies", lambda: (None, None, None))
    monkeypatch.setattr(Path, "exists", lambda self: True)
    with pytest.raises(FileExistsError):
        audit.verify(Path("/synthetic/run"), Path("/synthetic/existing.json"))


def test_json_result_io_is_exclusive_and_valid_without_disk(monkeypatch):
    class InMemoryOutput(io.StringIO):
        def close(self):
            pass

    output, calls = InMemoryOutput(), []

    def open_synthetic(path, mode="r", *, encoding=None):
        calls.append((str(path), mode, encoding))
        return output if mode == "x" else io.StringIO(output.getvalue())

    monkeypatch.setattr(Path, "open", open_synthetic)
    path = Path("/synthetic/result.json")
    audit.write_json(path, {"status": "PASS_ORDERED_MOTION_FIDELITY_RECOUNT", "enabled": False})
    assert json.loads(output.getvalue())["enabled"] is False
    assert audit.read_json(path)["status"] == "PASS_ORDERED_MOTION_FIDELITY_RECOUNT"
    assert calls == [(str(path), "x", "utf-8"), (str(path), "r", "utf-8")]


@pytest.mark.parametrize("enabled", [True, False])
def test_in_memory_verify_runs_complete_synthetic_protocol(monkeypatch, enabled):
    from types import SimpleNamespace

    digest = "a" * 64
    run, output = Path("/synthetic/run"), Path("/synthetic/result.json")
    done, cfg, _, _, mapping = _metadata()
    cfg["training"], cfg["validation"] = [1, 2], [17, 18]
    cfg["inputs"]["/synthetic/source-index.json"] = digest
    roles = dict(train=[1, 2], val=[17, 18], test=[3])
    val, val_inventory, val_signals = _synthetic_group("val", roles["val"])
    cold, cold_inventory, cold_signals = _synthetic_group(
        "cold", mapping["calibration_ids"]
    )
    if not enabled:
        val["predictions"][:, 3] = val["predictions"][:, 2]
        val["predictions"][:, 4] = val["predictions"][:, 2]
    for row in val_inventory + cold_inventory:
        row["raw_sha256"] = digest
    train_inventory = [
        dict(shot=s, raw_sha256=digest, n_points=300, n_forecasts=16)
        for s in roles["train"]
    ]
    inventory = dict(train=train_inventory, val=val_inventory, cold=cold_inventory)
    json_inputs = {
        str(run / "completed.json"): done,
        str(run / "protocol.json"): cfg,
        str(run / "inventory.json"): inventory,
        cfg["mapping"]: mapping,
    }
    ref = Path("/synthetic/source-reference")
    json_inputs[str(ref / "completed.json")] = {
        "artifact_sha256": {"protocol.json": digest}
    }
    json_inputs[str(ref / "protocol.json")] = {
        "source_state_sha256": digest,
        "input_sha256": {cfg["mapping"]: digest},
    }
    calls, result = [], {}

    class Hashes:
        def __call__(self, path):
            calls.append(("hash", str(path)))
            return digest

        def check(self):
            calls.append(("hashes_checked",))

    class Archive:
        def __init__(self, data):
            self.data, self.files = data, list(data)

        def __getitem__(self, key):
            return self.data[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def load(path, *, allow_pickle):
        assert allow_pickle is False
        arrays = {
            "normalization.npz": dict(center=np.zeros(23), scale=np.ones(23)),
            "val_predictions.npz": val,
            "cold_predictions.npz": cold,
        }
        return Archive(arrays[Path(path).name])

    def reader(device, sid, **kwargs):
        calls.append(("read", device, sid))
        return (val_signals if device == "J-TEXT" else cold_signals)[sid]

    def prepare(raw, index, this_mapping):
        assert index == 21 and this_mapping is mapping
        return raw

    io23 = SimpleNamespace(
        JTEXT_INDEX=Path("/synthetic/source-index.json"),
        path_for=lambda d, s: f"/synthetic/{d}/{s}",
        read_outcome=lambda *args: pytest.fail("outcome read"),
    )
    simple_data = SimpleNamespace(source_roles=lambda: roles, read_shot=reader)
    monkeypatch.setattr(audit, "SOURCE_REFERENCE", ref)
    monkeypatch.setattr(audit, "SOURCE_REFERENCE_COMPLETED_SHA256", digest)
    monkeypatch.setattr(
        audit, "signal_dependencies", lambda: (io23, simple_data, prepare)
    )
    monkeypatch.setattr(audit, "Hashes", Hashes)
    monkeypatch.setattr(audit, "read_json", lambda p: json_inputs[str(p)])
    monkeypatch.setattr(audit.np, "load", load)
    monkeypatch.setattr(
        audit, "write_json", lambda p, payload: result.update(path=p, payload=payload)
    )
    audit.verify(run, output)
    assert result["path"] == output
    payload = result["payload"]
    assert payload["status"] == "PASS_ORDERED_MOTION_FIDELITY_RECOUNT"
    assert payload["enabled"] is enabled
    assert payload["fit_completed_sha256"] == payload["fit_head_sha256"] == digest
    assert set(payload["groups"]) == {"val", "cold"}
    assert ("hashes_checked",) in calls
    assert [entry[1:] for entry in calls if entry[0] == "read"] == (
        [("J-TEXT", s) for s in roles["val"]]
        + [("EAST", s) for s in mapping["calibration_ids"]]
    )
