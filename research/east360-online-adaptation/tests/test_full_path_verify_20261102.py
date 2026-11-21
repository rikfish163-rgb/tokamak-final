"""CPU-only contracts; synthetic arrays and archives never leave memory."""

import copy
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

from experiments.risk_aligned_online.full_path_meta_20261102 import verify as v


def test_first_alarm_uses_strict_30_ms_and_does_not_cap_3000_ms():
    times = np.array([0.0, 3.969, 3.970, 4.001])
    assert v.classify_curve(times, [0, 0, 1, 1], True, 4.0, 0.5) == {
        "alarm": True,
        "lead_ms": 30.0,
        "timely": False,
    }
    row = v.classify_curve(times, [1, 0, 0, 0], True, 4.0, 0.5)
    assert row["timely"] and row["lead_ms"] == 4000.0
    assert not v.classify_curve(times, [0, 0, 0, 1], True, 4.0, 0.5)["alarm"]


def curve_fixture():
    times = np.arange(300, dtype=np.float64) / 1000
    cold = np.zeros(300, dtype=np.float32)
    reference = dict(
        times=times,
        cold=cold,
        innovation_metric=cold.copy(),
        metric_forecast=cold.copy(),
    )
    reference["innovation_metric"][160:] = 0.7
    reference["metric_forecast"][160:] = 0.3
    curves = {a: cold.copy() for a in v.ARMS}
    curves["truncated_metric"] = reference["innovation_metric"].copy()
    curves["forecast_real"] = reference["metric_forecast"].copy()
    return times, curves, reference


def test_reused_and_forecast_real_are_bitwise_bound_to_reference():
    times, curves, reference = curve_fixture()
    v.validate_curves(times, curves, reference, warmup=False)
    for arm in ("cold", "truncated_metric", "forecast_real"):
        changed = copy.deepcopy(curves)
        changed[arm][200] = 0.9
        with pytest.raises(ValueError, match="reference"):
            v.validate_curves(times, changed, reference, warmup=False)
    curves["full_real"][0] = -0.0
    with pytest.raises(ValueError, match="first160"):
        v.validate_curves(times, curves, reference, warmup=False)


def test_warmup_curves_all_reuse_reference_cold():
    times, _, reference = curve_fixture()
    for key in ("innovation_metric", "metric_forecast"):
        reference[key] = reference["cold"].copy()
    curves = {a: reference["cold"].copy() for a in v.ARMS}
    v.validate_curves(times, curves, reference, warmup=True)
    curves["full_hold"][200] = 0.2
    with pytest.raises(ValueError, match="warmup"):
        v.validate_curves(times, curves, reference, warmup=True)


def trace_fixture(index=21, length=300):
    ref = Path("/reference/shots/0021_42.npz")
    arrays, updates, actions, execution = {}, {}, {}, {}
    for arm in v.ARMS:
        fresh = index > 20 and arm in v.FRESH
        ends = list(range(159, length, 128)) if fresh else []
        before = np.zeros((len(ends), 4, 64), np.float32)
        after = np.zeros_like(before)
        gradient = np.zeros((len(ends), 256), np.float32)
        events = []
        for i, end in enumerate(ends):
            if i:
                before[i] = after[i - 1]
            after[i] = before[i]
            after[i, 0, 0] += 0.25
            gradient[i, 0] = 0.5
            events.append(
                dict(
                    maturity=end,
                    origin=end - 25,
                    first_affected=end + 1,
                    forecast_sha256="f" * 64,
                    mask_sha256="a" * 64,
                    target_sha256=("b" if arm.endswith("_real") else "c") * 64,
                    observed=100,
                    accepted=True,
                    gradient_computed=True,
                    gradient_norm=0.5,
                    step_norm=float(
                        np.linalg.norm((after[i] - before[i]).astype(np.float64))
                    ),
                    reason="improved",
                    loss_before=2.0,
                    loss_proposal=1.0,
                )
            )
        arrays[arm + "_before"], arrays[arm + "_after"], arrays[arm + "_gradient"] = (
            before,
            after,
            gradient,
        )
        updates[arm] = events
        actions[arm] = dict(
            endpoints=length if fresh else 0,
            capped_endpoints=0,
            max_rms=0.1 if fresh else 0.0,
        )
        execution[arm] = dict(
            kind="reused_warmup"
            if index <= 20
            else ("fresh_uncached_gpu" if fresh else "reused_reference"),
            reference=None if fresh else str(ref),
            reference_arm="cold" if index <= 20 else v.REUSED.get(arm),
        )
    info = dict(
        reset_each_shot=True,
        temporary_parameters=256,
        updates=updates,
        actions=actions,
        execution=execution,
        statistics_scope=v.STATISTICS_SCOPE,
        reference_curve_sha256="d" * 64,
    )
    return info, arrays, ref


def check_fixture(info, arrays, ref, index=21):
    return v.check_trace(
        info, arrays, 300, index=index, reference_path=ref, reference_sha="d" * 64
    )


def test_reused_empty_receipts_mean_no_new_execution_not_no_historical_updates():
    info, arrays, ref = trace_fixture()
    totals, actions = check_fixture(info, arrays, ref)
    assert totals["truncated_metric"]["events"] == 0
    assert actions["truncated_metric"]["endpoints"] == 0
    assert info["execution"]["truncated_metric"]["reference_arm"] == "innovation_metric"
    assert totals["full_hold"]["accepted"] == 2
    assert info["updates"]["full_hold"][0]["step_norm"] > 0.1


def test_all_warmup_arms_have_no_new_events_or_actions():
    info, arrays, ref = trace_fixture(index=1)
    totals, actions = check_fixture(info, arrays, ref, index=1)
    assert all(x["events"] == 0 for x in totals.values())
    assert all(x["endpoints"] == 0 for x in actions.values())
    info["actions"]["forecast_real"]["endpoints"] = 300
    with pytest.raises(ValueError):
        check_fixture(info, arrays, ref, index=1)


@pytest.mark.parametrize(
    "defect",
    [
        "reused_event",
        "reused_action",
        "reference_path",
        "execution_kind",
        "clock",
        "origin",
        "first_effect",
        "reset",
        "stitch",
        "gradient_norm",
        "step_norm",
        "false_accept",
        "changed_reject",
        "mask",
        "real_target",
    ],
)
def test_trace_rejects_forged_execution_and_receipts(defect):
    info, arrays, ref = trace_fixture()
    a = "full_real"
    e = info["updates"][a][0]
    if defect == "reused_event":
        info["updates"]["cold"] = [dict(e)]
    elif defect == "reused_action":
        info["actions"]["truncated_metric"]["endpoints"] = 300
    elif defect == "reference_path":
        info["execution"]["cold"]["reference"] = "/different.npz"
    elif defect == "execution_kind":
        info["execution"][a]["kind"] = "reused_reference"
    elif defect == "clock":
        e["maturity"] = 158
    elif defect == "origin":
        e["origin"] = 133
    elif defect == "first_effect":
        e["first_affected"] = 159
    elif defect == "reset":
        arrays[a + "_before"][0, 0, 0] = -0.0
    elif defect == "stitch":
        arrays[a + "_before"][1, 1, 1] = 1
    elif defect == "gradient_norm":
        e["gradient_norm"] = 0.4
    elif defect == "step_norm":
        e["step_norm"] = 0.2
    elif defect == "false_accept":
        e["loss_proposal"] = 3.0
    elif defect == "changed_reject":
        e.update(accepted=False, reason="not_improved", loss_proposal=3.0)
    elif defect == "mask":
        e["mask_sha256"] = "e" * 64
    elif defect == "real_target":
        e["target_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        check_fixture(info, arrays, ref)


def test_no_target_requires_explicit_missing_gradient_and_unchanged_state():
    info, arrays, ref = trace_fixture(length=200)
    for arm in v.FRESH:
        e = info["updates"][arm][0]
        e.update(
            observed=0,
            gradient_computed=False,
            accepted=False,
            reason="no_target",
            gradient_norm=0.0,
            step_norm=0.0,
        )
        e.pop("loss_before")
        e.pop("loss_proposal")
        arrays[arm + "_after"][:] = arrays[arm + "_before"]
        arrays[arm + "_gradient"][:] = 0
    v.check_trace(
        info, arrays, 200, index=21, reference_path=ref, reference_sha="d" * 64
    )


def gate_fixture():
    base = dict(n_disrupt=180, n_safe=180, tp=125, fp=35, J=0.5)
    return {a: {"full360": dict(base)} for a in v.ARMS}, dict(base)


def test_integer_gates_ignore_epsilon_wins_and_allow_fp_tradeoff():
    scores, zero = gate_fixture()
    scores["full_real"]["full360"].update(tp=126, fp=36, J=0.5000000000000001)
    assert not v.score_gates(scores, zero)["operational_gate"]
    scores["full_real"]["full360"].update(tp=127, fp=36, J=0.0)
    assert v.score_gates(scores, zero)["stage_success_pending_verification"]
    for arm in ("full_reverse", "full_hold", "forecast_real", "truncated_metric"):
        scores[arm]["full360"].update(tp=127, fp=36)
        assert not v.score_gates(scores, zero)["control_separation_gate"]
        scores[arm]["full360"].update(tp=125, fp=35)
    zero.update(tp=127, fp=36)
    assert not v.score_gates(scores, zero)["control_separation_gate"]
    scores["full_real"]["full360"].update(tp=130, fp=37)
    assert not v.score_gates(scores, zero)["operational_gate"]


def test_forecast_requires_its_own_matched_controls_and_balanced_roster():
    scores, zero = gate_fixture()
    scores["forecast_real"]["full360"].update(fp=33)
    assert v.score_gates(scores, zero)["forecast_control_separation_gate"]
    scores["forecast_hold"]["full360"].update(fp=33)
    assert not v.score_gates(scores, zero)["forecast_control_separation_gate"]
    scores["cold"]["full360"]["n_safe"] = 179
    with pytest.raises(ValueError, match="balanced"):
        v.score_gates(scores, zero)


def training_fixture():
    ids = list(range(1, 21))
    times = np.arange(300, dtype=np.float64) / 1000
    cohort = [
        dict(shot=s, disruptive=s <= 8, down=1.0 if s <= 8 else None, points=300)
        for s in ids
    ]
    warmup = {c["shot"]: (times, c["disruptive"], c["down"]) for c in cohort}
    rng, history = np.random.default_rng(131), []
    for i in range(128):
        chosen = [ids[int(rng.integers(8))], ids[8 + int(rng.integers(12))]]
        es = [
            dict(
                shot=s,
                query=288,
                maturity=287,
                supported=True,
                label=int(s <= 8),
                risk_before=0.5,
                risk_after=0.4,
                signal=0.3,
                objective=0.2,
                replay_updates=2,
                accepted_path_updates=1,
            )
            for s in chosen
        ]
        history.append(
            dict(
                step=i + 1,
                sample_shots=chosen,
                episodes={"full_operational": es},
                gradient_norms={"full_operational": 0.1},
            )
        )
    return (
        dict(ids=ids),
        dict(support_counts={"full_operational": 256}),
        history,
        cohort,
        warmup,
    )


def test_training_checks_seeded_pairs_and_complete_replay_clock():
    args = training_fixture()
    assert v.check_training(*args) == {"full_operational": 256}
    args[2][0]["episodes"]["full_operational"][0]["replay_updates"] = 1
    with pytest.raises(ValueError, match="replay"):
        v.check_training(*args)
    args = training_fixture()
    args[2][0]["episodes"]["full_operational"][0]["accepted_path_updates"] = 3
    with pytest.raises(ValueError, match="accepted"):
        v.check_training(*args)
    args = training_fixture()
    args[2][0]["sample_shots"] = [1, 2]
    with pytest.raises(ValueError, match="sample"):
        v.check_training(*args)


def test_unsupported_training_query_cannot_acquire_replay_or_risk_change():
    args = training_fixture()
    e = args[2][0]["episodes"]["full_operational"][0]
    e.update(
        query=132,
        maturity=None,
        supported=False,
        replay_updates=0,
        accepted_path_updates=0,
        risk_after=e["risk_before"],
    )
    args[1]["support_counts"]["full_operational"] = 255
    assert v.check_training(*args) == {"full_operational": 255}
    e["replay_updates"] = 1
    with pytest.raises(ValueError, match="replay"):
        v.check_training(*args)


def test_bound_file_heads_and_completion_hashes_are_not_interchangeable(monkeypatch):
    expected = [
        v.TRAINING / "heads.json",
        v.TRAINING / "completed.json",
        v.ROOT / v.ACCEPTANCE,
    ]
    digests = {
        str(expected[0]): v.FULL_HEAD_SHA,
        str(expected[1]): v.FULL_DONE_SHA,
        str(expected[2]): "a" * 64,
    }
    monkeypatch.setattr(v, "sha_file", lambda p: digests[str(p)])
    protocol = dict(bound_files_sha256=dict(digests))
    v.check_bound_files(protocol, expected)
    protocol["bound_files_sha256"][str(expected[0])] = v.FULL_DONE_SHA
    with pytest.raises(ValueError):
        v.check_bound_files(protocol, expected)


def test_archive_contains_imported_code_and_two_plans_not_acceptance(monkeypatch):
    contents = {name: name.encode() for name in (*v.RUN_REQUIRED,)}
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        for name, value in contents.items():
            z.writestr(name, value)
    original_zip = zipfile.ZipFile
    monkeypatch.setattr(
        v.zipfile, "ZipFile", lambda *_: original_zip(io.BytesIO(data.getvalue()))
    )
    monkeypatch.setattr(v, "sha_file", lambda _: "0" * 64)
    monkeypatch.setattr(v, "ROOT", Path("/nonexistent-verifier-root"))
    declared = {
        name: hashlib.sha256(value).hexdigest() for name, value in contents.items()
    }
    p = dict(
        code_sha256=declared,
        plan_sha256=declared[v.PLAN],
        evaluation_plan_sha256=declared[v.EVALUATION],
    )
    assert v.check_snapshot(Path("/memory"), p)["archive_entries_verified"] == len(
        contents
    )
    assert v.ACCEPTANCE not in declared


def test_summary_matched_flips_and_deltas_use_integer_net():
    base = [
        dict(
            shot=i + 1,
            disruptive=i < 180,
            alarm=(i < 125 or 180 <= i < 215),
            timely=i < 125,
            lead_ms=4000.0 if i < 125 else None,
        )
        for i in range(360)
    ]
    rows = {a: copy.deepcopy(base) for a in v.ARMS}
    rows["forecast_real"][180]["alarm"] = False
    zero = v.recount(base)
    result = v.summarize(rows, zero)
    assert result["forecast_matched_flips"]["forecast_hold"]["removed_FP"] == [181]
    assert result["deltas_vs_cold"]["forecast_real"]["delta_J"] == 1 / 180
    assert result["deltas_vs_cold"]["forecast_real"]["net_favorable_flips"] == 1
    assert result["metrics"]["forecast_real"]["full360"]["early_over3000"] == 125


def test_incomplete_guard_does_not_read_other_artifacts(monkeypatch):
    calls = []

    def read(path):
        calls.append(path.name)
        return dict(status="RUNNING")

    monkeypatch.setattr(v, "read_json", read)
    with pytest.raises(ValueError, match="COMPLETED"):
        v.verify(Path("/synthetic-not-a-real-run"))
    assert calls == ["completed.json"]


def test_cli_refuses_existing_report_without_running_verifier(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["verify"])
    monkeypatch.setattr(Path, "exists", lambda _: True)
    monkeypatch.setattr(v, "verify", lambda _: pytest.fail("must not execute"))
    with pytest.raises(FileExistsError):
        v.main()


def test_cli_report_write_is_exclusive_and_explicit_only(monkeypatch):
    saved = []

    class MemoryText(io.StringIO):
        def __exit__(self, *args):
            saved.append(self.getvalue())
            return super().__exit__(*args)

    def opened(path, mode, **kwargs):
        assert path == v.OUTPUT / "verification_records.json" and mode == "x"
        return MemoryText()

    monkeypatch.setattr(sys, "argv", ["verify"])
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(
        v, "verify", lambda _: dict(status="PASS", cases=360, curves=2880)
    )
    assert not saved
    v.main()
    assert len(saved) == 1 and '"PASS"' in saved[0]
