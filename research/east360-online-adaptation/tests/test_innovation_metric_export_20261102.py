import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.innovation_metric_20261102 import run
from experiments.risk_aligned_online.innovation_metric_20261102 import stream


def test_fixed_arm_head_mapping_and_controls():
    tables = {name: torch.full((25, 23), n + 1.0) for n, name in enumerate(run.HEADS)}
    old = torch.full((25, 23), 9.0)
    heads = run.build_heads(tables, old)
    assert tuple(heads) == run.ARMS
    assert heads["cold"] == (None, "real", "contrast")
    assert heads["contrast_unit"] == (old, "real", "contrast")
    assert heads["innovation_unit"] == (tables["unit_operational"], "real", "unit")
    assert heads["innovation_metric"] == (
        tables["metric_operational"],
        "real",
        "metric",
    )
    assert heads["metric_forecast"] == (tables["metric_forecast"], "real", "metric")
    assert not torch.count_nonzero(heads["metric_zero"][0])
    assert heads["metric_zero"][1:] == ("real", "metric")
    assert heads["metric_reverse"] == (
        tables["metric_operational"],
        "reverse",
        "metric",
    )
    assert heads["metric_hold"] == (tables["metric_operational"], "hold", "metric")


def test_invalid_head_tables_are_rejected():
    tables = {name: torch.zeros(25, 23) for name in run.HEADS}
    with pytest.raises(ValueError):
        run.build_heads(
            {**tables, "unexpected": torch.zeros(25, 23)}, tables[run.HEADS[0]]
        )
    tables[run.HEADS[0]][0, 0] = float("nan")
    with pytest.raises(ValueError):
        run.build_heads(tables, torch.zeros(25, 23))


def test_candidate_gate_is_strict_and_checks_all_four_controls():
    base = {"fpr": 0.2, "tp": 100, "fp": 30, "J": 0.5}
    scores = {a: {"full360": dict(base)} for a in run.ARMS}
    scores["innovation_metric"]["full360"] = {"fpr": 0.2, "tp": 100, "fp": 30, "J": 0.6}
    gates = run.score_gates(scores)
    assert gates["operational_gate"]
    assert gates["control_separation_gate"]
    assert gates["stage_success_pending_verification"]
    for arm in run.CONTROLS:
        scores[arm]["full360"]["J"] = 0.6
        assert not run.score_gates(scores)["stage_success_pending_verification"]
        scores[arm]["full360"]["J"] = 0.5
    for field, value in (("fpr", 0.200001), ("tp", 99), ("fp", 31), ("J", 0.5)):
        original = scores["innovation_metric"]["full360"][field]
        scores["innovation_metric"]["full360"][field] = value
        assert not run.score_gates(scores)["stage_success_pending_verification"]
        scores["innovation_metric"]["full360"][field] = original


def _curves():
    times = np.arange(200, dtype=np.float64) / 1000
    cold = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    operational = cold.copy()
    operational[160:] = 0.25
    curves = {a: cold.copy() for a in run.ARMS}
    curves["contrast_unit"] = operational.copy()
    return times, cold, operational, curves


def test_exact_reference_and_first_160_checks_are_bitwise():
    times, cold, operational, curves = _curves()
    run.validate_curves(times, curves, times, cold, operational, warmup=False)
    changed = {a: v.copy() for a, v in curves.items()}
    changed["innovation_metric"][0] = -0.0
    with pytest.raises(ValueError, match="first 160"):
        run.validate_curves(times, changed, times, cold, operational, warmup=False)
    changed = {a: v.copy() for a, v in curves.items()}
    changed["innovation_metric"][159] = np.nextafter(
        changed["innovation_metric"][159], np.float32(1)
    )
    with pytest.raises(ValueError, match="first 160"):
        run.validate_curves(times, changed, times, cold, operational, warmup=False)
    changed = {a: v.copy() for a, v in curves.items()}
    changed["contrast_unit"][170] = 0.3
    with pytest.raises(ValueError, match="contrast"):
        run.validate_curves(times, changed, times, cold, operational, warmup=False)
    changed = {a: v.copy() for a, v in curves.items()}
    changed["cold"] = changed["cold"].astype(np.float64)
    with pytest.raises(ValueError, match="cold"):
        run.validate_curves(times, changed, times, cold, operational, warmup=False)


def test_warmup_requires_every_curve_to_reuse_cold():
    times, cold, _, curves = _curves()
    curves = {a: cold.copy() for a in curves}
    run.validate_curves(times, curves, times, cold, cold, warmup=True)
    curves["metric_hold"][180] = 0.9
    with pytest.raises(ValueError, match="warmup"):
        run.validate_curves(times, curves, times, cold, cold, warmup=True)


def _training_receipts():
    ids = list(range(1, 21))
    done = dict(
        status="COMPLETED",
        steps=128,
        head_optimizer_steps=128,
        source_tensors_unchanged=True,
        ids=ids,
        outcome_reads=20,
        suffix_shots_read=0,
        plan_sha256="plan",
    )
    protocol = dict(
        scope="ENDED20_INNOVATION_METRIC_EQUAL_BUDGET",
        ids=ids,
        source_sha256="source",
        cold_state_sha256="cold",
        input_map_sha256="map",
        plan_sha256="plan",
        heads=dict(
            unit_operational="unit",
            metric_operational="metric",
            metric_forecast="metric",
        ),
        forecast_parameters_per_head=575,
        steps=128,
        optimizer="Adam",
        learning_rate=0.001,
        seed=131,
        gradient_clip=1.0,
        temporal_gradient_truncation_updates=1,
        first_update=159,
        update_interval=128,
        parameter_unit_step=0.1,
        metric_learning_rate=1.0,
        relative_metric_damping=0.01,
        absolute_metric_damping=1e-8,
        metric_increment_hidden_rms_cap=0.05,
        total_hidden_rms_cap=0.25,
        warmup_cost_retained=True,
    )
    return ids, done, protocol


@pytest.mark.parametrize(
    "which,key,value",
    [
        ("done", "steps", 127),
        ("done", "head_optimizer_steps", 127),
        ("done", "suffix_shots_read", 1),
        ("done", "plan_sha256", "other"),
        ("protocol", "ids", list(range(2, 22))),
        ("protocol", "cold_state_sha256", "other"),
        ("protocol", "input_map_sha256", "other"),
        ("protocol", "metric_increment_hidden_rms_cap", 0.1),
        ("protocol", "scope", "INNOVATION_METRIC_PREFLIGHT"),
    ],
)
def test_training_receipt_enforces_fixed_budget_and_provenance(which, key, value):
    ids, done, protocol = _training_receipts()
    run.validate_training(done, protocol, ids, "source", "cold", "map", "plan")
    (done if which == "done" else protocol)[key] = value
    with pytest.raises(ValueError):
        run.validate_training(done, protocol, ids, "source", "cold", "map", "plan")


def test_curves_states_metadata_and_seal_exist_before_outcome(monkeypatch):
    saved = {}

    class MemoryFile(io.BytesIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def __exit__(self, *args):
            saved[self.path] = self.getvalue()
            return super().__exit__(*args)

    def open_binary(path, mode):
        assert mode == "xb" and str(path) not in saved
        return MemoryFile(str(path))

    def write_json(path, value):
        assert str(path) not in saved
        saved[str(path)] = json.dumps(value).encode()

    def digest(path):
        return hashlib.sha256(saved[str(path)]).hexdigest()

    root, index, sid = Path("/no-files-created"), 21, 42
    stem = root / "shots/0021_42"
    times, cold, operational, curves = _curves()
    arrays, info = run.export_traces(
        {a: [] for a in run.ARMS}, {a: {} for a in run.ARMS}
    )
    observed = []

    def outcome(device, shot):
        assert (device, shot) == ("EAST", sid)
        seal = json.loads(saved[str(stem) + "_seal.json"])
        for suffix, key in (
            (".npz", "sha256"),
            ("_states.npz", "states_sha256"),
            ("_updates.json", "updates_sha256"),
        ):
            assert digest(Path(str(stem) + suffix)) == seal[key]
        assert seal["current_outcome_used_in_prediction"] is False
        observed.append(True)
        return False, None

    monkeypatch.setattr(Path, "open", open_binary)
    monkeypatch.setattr(run, "write", write_json)
    monkeypatch.setattr(run, "sha", digest)
    monkeypatch.setattr(run.io23, "read_outcome", outcome)
    rows, seal_record = run.seal_and_score(
        root, index, sid, times, curves, arrays, info, 0.5
    )
    assert observed == [True]
    assert set(rows) == set(run.ARMS)
    assert seal_record["seal_sha256"] == digest(Path(str(stem) + "_seal.json"))
    with np.load(io.BytesIO(saved[str(stem) + ".npz"]), allow_pickle=False) as archive:
        assert set(archive.files) == {"times", *run.ARMS}
        np.testing.assert_array_equal(archive["cold"], cold)
        np.testing.assert_array_equal(archive["contrast_unit"], operational)


@pytest.mark.parametrize("index", [1, 21])
def test_warmup_reuses_curves_and_suffix_calls_uncached_stream(monkeypatch, index):
    sid, reference = 42, Path("/reference-in-memory")
    times, cold, operational, _ = _curves()
    if index <= 20:
        operational = cold.copy()
    data = io.BytesIO()
    np.savez_compressed(data, times=times, cold=cold, operational=operational)
    seal = dict(
        shot=sid, index=index, current_outcome_used_in_prediction=False, sha256="curve"
    )
    record = dict(
        shot=sid, index=index, seal_sha256_at_verification="seal", curve_sha256="curve"
    )
    original_np_load = np.load
    seen = []

    def read_reference(path, *, allow_pickle):
        assert path == reference / "shots" / f"{index:04d}_{sid}.npz"
        assert allow_pickle is False
        return original_np_load(io.BytesIO(data.getvalue()), allow_pickle=False)

    def fake_sha(path):
        if path.name.endswith("_seal.json"):
            return "seal"
        if path.suffix == ".npz":
            return "curve"
        return "input"

    def fake_simulate(model, signal, basis, heads):
        assert set(signal) == {"values", "available"}
        seen.append("stream")
        curves = {a: cold.copy() for a in run.ARMS}
        curves["contrast_unit"] = operational.copy()
        return dict(
            probability=curves,
            tapes={a: [] for a in run.ARMS},
            actions={
                a: dict(endpoints=len(times), capped_endpoints=0, max_rms=0.0)
                for a in run.ARMS
            },
        )

    def fake_seal(
        root, shot_index, shot, actual_times, curves, arrays, info, threshold
    ):
        assert (shot_index, shot, threshold) == (index, sid, run.THRESHOLD)
        assert run.bitwise_equal(actual_times, times)
        assert set(arrays) == {
            a + suffix
            for a in run.ARMS
            for suffix in ("_before", "_after", "_gradient")
        }
        assert (
            all(run.bitwise_equal(curves[a], cold) for a in run.ARMS)
            if index <= 20
            else True
        )
        seen.append("seal")
        return {a: {} for a in run.ARMS}, {"index": index}

    monkeypatch.setattr(run, "REFERENCE", reference)
    monkeypatch.setattr(run, "read", lambda path: seal)
    monkeypatch.setattr(run, "sha", fake_sha)
    monkeypatch.setattr(run.np, "load", read_reference)
    monkeypatch.setattr(run.io23, "path_for", lambda *_: Path("/input"))
    monkeypatch.setattr(
        run.simple_data,
        "read_shot",
        lambda *args, **kwargs: {
            "times": times,
            "values": np.zeros((200, 23)),
            "available": np.ones((200, 23), bool),
        },
    )
    monkeypatch.setattr(run, "apply_input_contract", lambda x: x)
    monkeypatch.setattr(run, "transform", lambda signal, mapping: signal)
    monkeypatch.setattr(run, "simulate", fake_simulate)
    monkeypatch.setattr(run, "seal_and_score", fake_seal)
    run.run_shot(
        Path("/no-files-created"), index, sid, record, "input", None, None, None, {}
    )
    assert seen == (["seal"] if index <= 20 else ["stream", "seal"])


def test_existing_output_fails_before_any_gpu_setup(monkeypatch):
    monkeypatch.setattr(
        run, "configure", lambda: pytest.fail("must fail before GPU setup")
    )
    with pytest.raises(FileExistsError):
        run.main(Path("."))


def test_new_stream_uses_common_mask_and_never_rewrites_sealed_prefix(monkeypatch):
    model = SimpleNamespace(base=SimpleNamespace(center=torch.zeros(1)))
    calls = []

    def encode(_model, signal, ends, _basis):
        return dict(
            index=torch.tensor(ends), value=torch.tensor(signal["values"][ends, 0])
        )

    def forward(_model, pack, state):
        return dict(
            logit=pack["value"] * 0.001 + state[0, 0],
            rms=torch.zeros(len(pack["index"])),
            capped=torch.zeros(len(pack["index"]), dtype=torch.bool),
        )

    def auxiliary(_model, point, state, _head):
        return torch.zeros(1, 25, 23) + point["index"][0] + state[0, 0]

    def update(_model, point, state, truth, hold, mask, _head, geometry):
        origin = int(point["index"][0])
        calls.append(
            (
                origin,
                int(truth[0, 0, 0]),
                int(hold[0, 0, 0]),
                mask.cpu().numpy().copy(),
                geometry,
            )
        )
        next_state = state.clone()
        next_state[0, 0] += 1
        return next_state, {"accepted": True}

    for name, function in (
        ("encode", encode),
        ("forward", forward),
        ("auxiliary", auxiliary),
        ("update", update),
    ):
        monkeypatch.setattr(stream, name, function)
    values = np.repeat(np.arange(448, dtype=np.float32)[:, None], 23, axis=1)
    available = np.ones_like(values, dtype=bool)
    available[145, 0] = False
    signal = dict(values=values, available=available)
    tables = {name: torch.zeros(25, 23) for name in run.HEADS}
    heads = run.build_heads(tables, torch.zeros(25, 23))
    full = stream.simulate(model, signal, None, heads)
    assert [event["maturity"] for event in full["tapes"]["innovation_metric"]] == [
        159,
        287,
        415,
    ]
    first = calls[:7]
    assert [c[1] for c in first] == [135, 135, 135, 135, 135, 159, 134]
    assert [c[4] for c in first] == [
        "contrast",
        "unit",
        "metric",
        "metric",
        "metric",
        "metric",
        "metric",
    ]
    assert all(c[2] == 134 and np.array_equal(c[3], first[0][3]) for c in first)
    assert not first[0][3][0, :, 10].any()
    for arm in run.ARMS:
        assert run.bitwise_equal(
            full["probability"][arm][:160], full["probability"]["cold"][:160]
        )
        assert full["actions"][arm]["endpoints"] == 448
    prefix = {key: value[:416].copy() for key, value in signal.items()}
    truncated = stream.simulate(model, prefix, None, heads)
    for arm in run.ARMS:
        assert run.bitwise_equal(
            full["probability"][arm][:416], truncated["probability"][arm]
        )
