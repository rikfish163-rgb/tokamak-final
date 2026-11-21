import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.full_path_meta_20261102 import run


def test_only_six_fresh_heads_are_run_with_the_unchanged_metric_rule():
    full, forecast = torch.ones(25, 23), torch.full((25, 23), 2.0)
    heads = run.build_heads(full, forecast)
    assert tuple(heads) == run.FRESH
    assert set(heads).isdisjoint(run.REUSED)
    for prefix, table in (("full", full), ("forecast", forecast)):
        for mode in ("real", "reverse", "hold"):
            head, target, geometry = heads[prefix + "_" + mode]
            assert head is table and target == mode and geometry == "metric"
    with pytest.raises(ValueError):
        run.build_heads(torch.zeros(24, 23), forecast)


def metric(tp, fp):
    return dict(
        tp=tp,
        fp=fp,
        n_disrupt=180,
        n_safe=180,
        tpr=tp / 180,
        fpr=fp / 180,
        J=(tp - fp) / 180,
    )


def test_fp_plus_one_is_allowed_only_when_net_gain_and_control_separation_hold():
    scores = {a: {"full360": metric(125, 35)} for a in run.ARMS}
    scores["full_real"]["full360"] = metric(127, 36)
    gates = run.score_gates(scores, metric(125, 35))
    assert gates["operational_gate"] and gates["stage_success_pending_verification"]
    for arm in ("full_reverse", "full_hold", "forecast_real", "truncated_metric"):
        scores[arm]["full360"] = metric(127, 36)
        assert not run.score_gates(scores, metric(125, 35))[
            "stage_success_pending_verification"
        ]
        scores[arm]["full360"] = metric(125, 35)
    assert not run.score_gates(scores, metric(127, 36))["control_separation_gate"]
    for tp, fp in ((126, 36), (128, 37), (124, 30)):
        scores["full_real"]["full360"] = metric(tp, fp)
        assert not run.score_gates(scores, metric(125, 35))["operational_gate"]


def test_forecast_has_separate_matched_controls():
    scores = {a: {"full360": metric(125, 35)} for a in run.ARMS}
    scores["forecast_real"]["full360"] = metric(125, 33)
    gates = run.score_gates(scores, metric(125, 35))
    assert all(gates["forecast_control_comparisons"].values())
    scores["forecast_hold"]["full360"] = metric(125, 33)
    assert not all(
        run.score_gates(scores, metric(125, 35))[
            "forecast_control_comparisons"
        ].values()
    )


def test_mathematical_ties_ignore_last_bit_differences_in_float_J():
    scores = {a: {"full360": metric(125, 35)} for a in run.ARMS}
    scores["cold"]["full360"]["J"] = 0.49999999999999994
    scores["full_real"]["full360"] = metric(126, 36)
    scores["full_real"]["full360"]["J"] = 0.5000000000000001
    assert not run.score_gates(scores, metric(125, 35))["operational_gate"]
    scores["cold"]["full360"] = metric(124, 35)
    gates = run.score_gates(scores, metric(124, 35))
    assert (
        gates["operational_gate"]
        and not gates["control_comparisons"]["truncated_metric"]
    )
    scores["forecast_real"]["full360"] = metric(126, 36)
    scores["forecast_real"]["full360"]["J"] = 0.5000000000000001
    assert not any(
        run.score_gates(scores, metric(124, 35))[
            "forecast_control_comparisons"
        ].values()
    )


def reference_arrays(warmup=False):
    cold = np.linspace(0, 1, 200, dtype=np.float32)
    truncated, forecast = cold.copy(), cold.copy()
    if not warmup:
        truncated[160:], forecast[160:] = 0.4, 0.3
    return dict(
        times=np.arange(200, dtype=np.float64) / 1000,
        cold=cold,
        innovation_metric=truncated,
        metric_forecast=forecast,
    )


@pytest.mark.parametrize("index", [1, 21])
def test_warmup_borrowing_and_six_arm_uncached_suffix_metadata(monkeypatch, index):
    reference = reference_arrays(index <= 20)
    binary = io.BytesIO()
    np.savez_compressed(binary, **reference)
    original_load = np.load
    sid = 42
    record = dict(shot=sid, index=index, curve_sha256="curve", seal_sha256="seal")
    seal = dict(
        shot=sid, index=index, sha256="curve", current_outcome_used_in_prediction=False
    )
    calls, captured = [], {}

    def load(path, *, allow_pickle):
        assert not allow_pickle
        return original_load(io.BytesIO(binary.getvalue()), allow_pickle=False)

    def simulate(model, signal, basis, heads):
        assert set(heads) == set(run.FRESH) and set(signal) == {"values", "available"}
        calls.append("simulate")
        curves = {a: reference["cold"].copy() for a in run.FRESH}
        curves["forecast_real"] = reference["metric_forecast"].copy()
        return dict(
            probability=curves,
            tapes={a: [] for a in run.FRESH},
            actions={
                a: dict(endpoints=200, capped_endpoints=0, max_rms=0.0)
                for a in run.FRESH
            },
        )

    def read_shot(*args, **kwargs):
        assert index > 20
        return dict(
            times=reference["times"],
            values=np.zeros((200, 23)),
            available=np.ones((200, 23), bool),
        )

    def save(root, index, sid, times, curves, arrays, info):
        captured.update(curves=curves, arrays=arrays, info=info)
        calls.append("seal")
        return {a: {} for a in run.ARMS}, {"shot": sid}

    monkeypatch.setattr(run.np, "load", load)
    monkeypatch.setattr(run, "read", lambda path: seal)
    monkeypatch.setattr(
        run,
        "sha",
        lambda path: (
            "seal"
            if path.name.endswith("_seal.json")
            else ("curve" if path.suffix == ".npz" else "input")
        ),
    )
    monkeypatch.setattr(run.io23, "path_for", lambda *args: Path("/raw-not-opened"))
    monkeypatch.setattr(run.simple_data, "read_shot", read_shot)
    monkeypatch.setattr(run, "apply_input_contract", lambda x: x)
    monkeypatch.setattr(run, "transform", lambda signal, mapping: signal)
    monkeypatch.setattr(run, "simulate", simulate)
    monkeypatch.setattr(run, "seal_and_score", save)
    heads = run.build_heads(torch.zeros(25, 23), torch.ones(25, 23))
    run.run_shot(
        Path("/no-files-created"), index, sid, record, "input", None, None, heads, {}
    )
    assert calls == (["seal"] if index <= 20 else ["simulate", "seal"])
    assert set(captured["curves"]) == set(run.ARMS)
    for arm, reference_arm in run.REUSED.items():
        assert captured["arrays"][arm + "_before"].shape == (0, 4, 64)
        assert captured["info"]["updates"][arm] == []
        execution = captured["info"]["execution"][arm]
        assert execution["reference_arm"] == ("cold" if index <= 20 else reference_arm)
        assert execution["kind"].startswith("reused") and execution["reference"]
    assert (
        captured["info"]["statistics_scope"]
        == "recomputed_this_run_only_not_historical_activity"
    )
    for arm in run.FRESH:
        assert captured["info"]["execution"][arm]["kind"] == (
            "reused_warmup" if index <= 20 else "fresh_uncached_gpu"
        )


def test_borrowed_and_forecast_and_first_160_curves_are_bitwise_checked():
    ref = reference_arrays()
    curves = {a: ref["cold"].copy() for a in run.ARMS}
    curves["truncated_metric"] = ref["innovation_metric"].copy()
    curves["forecast_real"] = ref["metric_forecast"].copy()
    run.validate_curves(ref["times"], curves, ref, warmup=False)
    for arm, position, value in (
        ("truncated_metric", 170, 0.5),
        ("forecast_real", 170, 0.5),
        ("full_real", 0, -0.0),
    ):
        changed = {a: v.copy() for a, v in curves.items()}
        changed[arm][position] = value
        with pytest.raises(ValueError):
            run.validate_curves(ref["times"], changed, ref, warmup=False)


def test_all_curve_and_trace_evidence_is_sealed_before_outcome(monkeypatch):
    saved = {}

    class File(io.BytesIO):
        def __init__(self, path):
            super().__init__()
            self.path = str(path)

        def __exit__(self, *args):
            saved[self.path] = self.getvalue()
            return super().__exit__(*args)

    def write(path, value):
        assert str(path) not in saved
        saved[str(path)] = json.dumps(value).encode()

    def digest(path):
        return hashlib.sha256(saved[str(path)]).hexdigest()

    monkeypatch.setattr(
        Path,
        "open",
        lambda path, mode: File(path) if mode == "xb" else pytest.fail(mode),
    )
    monkeypatch.setattr(run, "write", write)
    monkeypatch.setattr(run, "sha", digest)
    seen = []

    def outcome(device, sid):
        assert (device, sid) == ("EAST", 42)
        stem = "/no-files-created/shots/0021_42"
        seal = json.loads(saved[stem + "_seal.json"])
        for suffix, key in (
            (".npz", "sha256"),
            ("_states.npz", "states_sha256"),
            ("_updates.json", "updates_sha256"),
        ):
            assert digest(Path(stem + suffix)) == seal[key]
        seen.append(True)
        return False, None

    monkeypatch.setattr(run.io23, "read_outcome", outcome)
    ref = reference_arrays()
    curves = {a: ref["cold"].copy() for a in run.ARMS}
    rows, receipt = run.seal_and_score(
        Path("/no-files-created"), 21, 42, ref["times"], curves, {}, {}
    )
    assert seen == [True] and set(rows) == set(run.ARMS)
    assert receipt["seal_sha256"] == digest(
        Path("/no-files-created/shots/0021_42_seal.json")
    )


def test_snapshot_uses_entry_plan_and_imported_modules_without_package_glob(
    monkeypatch,
):
    archived, saved = [], {}
    entry = Path(run.__file__)
    imported = entry.parent / "core.py"
    monkeypatch.setattr(
        run,
        "sys",
        SimpleNamespace(
            modules={
                "experiments.risk_aligned_online.full_path_meta_20261102.core": SimpleNamespace(
                    __file__=str(imported)
                )
            }
        ),
    )
    monkeypatch.setattr(Path, "glob", lambda *args: pytest.fail("no package glob"))
    monkeypatch.setattr(run, "sha", lambda path: "hash")
    monkeypatch.setattr(
        run, "write", lambda path, value: saved.update({path.name: value})
    )

    class Archive:
        def __init__(self, *args):
            assert args[1] == "x"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, path, name):
            archived.append(path)

    monkeypatch.setattr(run.zipfile, "ZipFile", Archive)
    run.snapshot(Path("/no-files-created"), {})
    assert set(archived) == {entry, run.PLAN, run.EVALUATION, imported}
    assert saved["protocol.json"]["code_sha256"]


def test_existing_output_fails_before_gpu_setup(monkeypatch):
    monkeypatch.setattr(run, "configure", lambda: pytest.fail("no GPU setup"))
    with pytest.raises(FileExistsError):
        run.main(Path("."))


@pytest.mark.parametrize(
    "corruption", [None, "steps", "source", "plan", "head_shape", "history", "cohort"]
)
def test_full_training_contract_rejects_budget_parent_or_artifact_drift(
    monkeypatch, corruption
):
    ids = list(range(1, 21))
    reference = dict(
        order=[{"shot": s} for s in ids],
        source_sha256="source",
        cold_state_sha256="cold",
        input_map_sha256="map",
    )
    done = dict(
        status="COMPLETED",
        steps=128,
        head_optimizer_steps=128,
        ids=ids,
        source_tensors_unchanged=True,
        outcome_reads=20,
        suffix_shots_read=0,
        heads_sha256="heads",
        history_sha256="history",
        training_cohort_sha256="cohort",
        number_heads=1,
        plan_sha256="plan",
    )
    protocol = dict(
        steps=128,
        ids=ids,
        source_sha256="source",
        cold_state_sha256="cold",
        input_map_sha256="map",
        scope="ENDED20_FULL_PATH_RISK_META",
        heads={"full_operational": "metric"},
        temporal_gradient_truncation_updates=None,
        diagnostic_incoming_state_detached=True,
        plan_sha256="plan",
    )
    tables = {"full_operational": np.zeros((25, 23)).tolist()}
    history = [dict(step=i, sample_shots=[1, 9]) for i in range(1, 129)]
    files = {
        "completed.json": done,
        "protocol.json": protocol,
        "heads.json": tables,
        "history.json": history,
    }
    if corruption == "steps":
        done["head_optimizer_steps"] = 127
    elif corruption == "source":
        protocol["source_sha256"] = "different"
    elif corruption == "plan":
        done["plan_sha256"] = "different"
    elif corruption == "head_shape":
        tables["full_operational"] = [[0.0]]
    elif corruption == "history":
        history[-1]["sample_shots"] = [1, 21]
    monkeypatch.setattr(run, "read", lambda path: files[path.name])

    def digest(path):
        if path == run.PLAN:
            return "plan"
        value = {
            "heads.json": "heads",
            "history.json": "history",
            "training_cohort.json": "cohort",
        }[path.name]
        return (
            "corrupt"
            if corruption == "cohort" and path.name == "training_cohort.json"
            else value
        )

    monkeypatch.setattr(run, "sha", digest)
    if corruption is None:
        assert (
            run.load_training(Path("/in-memory-training"), reference, full=True)
            is tables
        )
    else:
        with pytest.raises(ValueError):
            run.load_training(Path("/in-memory-training"), reference, full=True)


def test_fake_complete360_counts_fresh_and_borrowed_curves_without_real_io(monkeypatch):
    ids = list(range(1, 361))
    calibration, cold = Path("/in-memory-calibration"), Path("/in-memory-cold")
    reference = dict(
        order=[{"shot": s} for s in ids],
        threshold=run.THRESHOLD,
        calibration_reference=str(calibration),
        source_sha256="source",
        cold_state=str(cold),
        cold_state_sha256="cold",
        input_map_sha256="map",
        input_hashes_sha256="inputs",
    )
    done = dict(
        status="COMPLETED",
        protocol_sha256="protocol",
        metrics={"metric_zero": {"full360": metric(125, 35)}},
    )
    records = dict(
        status="PASS",
        hashes=dict(
            completed_sha256=run.REFERENCE_SHA,
            protocol_sha256="protocol",
            shot_files=[{"shot": s} for s in ids],
        ),
    )
    files = {
        run.REFERENCE / "protocol.json": reference,
        run.REFERENCE / "completed.json": done,
        run.REFERENCE / "verification_records.json": records,
        run.REFERENCE / "verification_neural.json": dict(
            status="PASS", completed_sha256=run.REFERENCE_SHA
        ),
        calibration / "input_map.json": {"calibration_ids": ids[:20]},
        calibration / "input_hashes.json": {str(s): "raw" for s in ids},
    }
    hashes = {
        run.REFERENCE / "completed.json": run.REFERENCE_SHA,
        run.REFERENCE / "protocol.json": "protocol",
        run.DEFAULT_PARENT: "source",
        cold: "cold",
        calibration / "input_map.json": "map",
        calibration / "input_hashes.json": "inputs",
    }
    saved, seen = {}, []
    monkeypatch.setattr(run, "read", lambda path: files[path])
    monkeypatch.setattr(run, "sha", lambda path: hashes.get(path, "unchanged"))
    monkeypatch.setattr(
        run, "write", lambda path, value: saved.update({path.name: value})
    )
    monkeypatch.setattr(run, "configure", lambda: None)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "_durable_json", lambda *args: None)
    model = SimpleNamespace(
        base=SimpleNamespace(
            center=SimpleNamespace(device=SimpleNamespace(type="cuda"))
        ),
        parameters=lambda: [],
        state_dict=lambda: {"frozen": torch.zeros(1)},
    )
    monkeypatch.setattr(
        run, "load_model", lambda: (model, {"threshold": run.THRESHOLD}, cold)
    )
    monkeypatch.setattr(run, "source_basis", lambda model: None)
    monkeypatch.setattr(
        run,
        "load_training",
        lambda root, reference, full: {
            "full_operational" if full else "metric_forecast": np.zeros((25, 23))
        },
    )
    original_tensor = torch.tensor
    monkeypatch.setattr(
        run.torch,
        "tensor",
        lambda data, *, dtype, device: original_tensor(data, dtype=dtype),
    )

    def snapshot(root, protocol):
        protocol["code_sha256"] = {}
        saved["protocol.json"] = protocol

    monkeypatch.setattr(run, "snapshot", snapshot)

    def shot(root, index, sid, record, expected_hash, model, basis, heads, mapping):
        assert record["shot"] == sid == ids[index - 1] and set(heads) == set(run.FRESH)
        seen.append(sid)
        flag = index <= 8 or 21 <= index <= 192
        row = dict(
            shot=sid,
            disruptive=flag,
            timely=flag,
            alarm=flag,
            lead_ms=100.0 if flag else None,
        )
        info = dict(
            updates={a: [] for a in run.ARMS},
            actions={
                a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in run.ARMS
            },
        )
        return (
            {a: dict(row) for a in run.ARMS},
            info,
            {"shot": sid, "seal_sha256": "sealed"},
        )

    monkeypatch.setattr(run, "run_shot", shot)
    run.main(Path("/no-files-created-complete360"))
    assert seen == ids
    result = saved["completed.json"]
    assert (
        result["fresh_curves"],
        result["reused_suffix_curves"],
        result["reused_warmup_curves"],
    ) == (2040, 680, 160)
    assert result["total_curves"] == 2880
    assert result["curve_execution_counts"] == dict(
        fresh_suffix=2040, copied_preparation=120, referenced_parent=720
    )
    assert all(len(rows) == 360 for rows in saved["per_shot.json"].values())
    assert result["forecast_followup_posthoc_selected_from_prior_eight_arms"] is True
    assert result["forecast_preregistered_winner"] is False
    assert saved["protocol.json"]["real_time_latency_verified"] is False
    assert result["method_promoted"] is result["scientific_goal_achieved"] is False
    for arm in run.REUSED:
        assert result["recomputed_update_statistics"][arm]["events"] == 0
