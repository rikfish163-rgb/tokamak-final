"""CPU-only sidecar tests; no real stream, HDF signals, or report writes."""

import copy
import io
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.risk_aligned_online.armijo_feedback_20261102 import verify as v


def curve_fixture():
    times = np.arange(300, dtype=np.float64) / 1000
    cold = np.zeros(300, dtype=np.float32)
    fixed = cold.copy()
    fixed[160:] = 0.4
    reference = dict(times=times, cold=cold, full_real=fixed)
    curves = {a: cold.copy() for a in v.ARMS}
    curves["fixed_full"] = fixed.copy()
    return times, curves, reference


def test_reference_curves_and_first160_are_bitwise_but_new_forecast_is_not_archived():
    times, curves, reference = curve_fixture()
    curves["forecast_real"][160:] = 0.8
    v.validate_curves(times, curves, reference, warmup=False)
    curves["fixed_full"][200] = 0.5
    with pytest.raises(ValueError, match="reference"):
        v.validate_curves(times, curves, reference, warmup=False)
    times, curves, reference = curve_fixture()
    curves["meta_real"][0] = -0.0
    with pytest.raises(ValueError, match="first160"):
        v.validate_curves(times, curves, reference, warmup=False)


def test_warmup_requires_every_curve_identical_to_old_cold():
    times, _, reference = curve_fixture()
    curves = {a: reference["cold"].copy() for a in v.ARMS}
    v.validate_curves(times, curves, reference, warmup=True)
    curves["meta_hold"][200] = 0.7
    with pytest.raises(ValueError, match="warmup"):
        v.validate_curves(times, curves, reference, warmup=True)


def trace_fixture(index=21, length=300):
    ref = Path("/reference/shots/0021_42.npz")
    info = dict(
        reset_each_shot=True,
        temporary_parameters=256,
        updates={},
        actions={},
        execution={},
        statistics_scope=v.STATISTICS_SCOPE,
        reference_curve_sha256="d" * 64,
    )
    arrays = {}
    for a in v.ARMS:
        fresh = index > 20 and a in v.FRESH
        ends = list(range(159, length, 128)) if fresh else []
        before = np.zeros((len(ends), 4, 64), np.float32)
        after = np.zeros_like(before)
        gradient = np.zeros((len(ends), 256), np.float32)
        events = []
        for i, end in enumerate(ends):
            if i:
                before[i] = after[i - 1]
            after[i] = before[i]
            after[i, 0, 0] -= 0.25
            gradient[i, 0] = 0.5
            bound = 2.0 - 1e-4 * 0.125
            events.append(
                dict(
                    origin=end - 25,
                    maturity=end,
                    first_affected=end + 1,
                    forecast_sha256="f" * 64,
                    mask_sha256="a" * 64,
                    target_sha256=("b" if a.endswith("_real") else "c") * 64,
                    observed=100,
                    gradient_computed=True,
                    accepted=True,
                    reason="armijo",
                    alpha=1.0,
                    evaluations=1,
                    loss_before=2.0,
                    loss_after=1.0,
                    armijo_bound=bound,
                    trials=[dict(alpha=1.0, loss=1.0, armijo_bound=bound)],
                    gradient_norm=0.5,
                    step_norm=0.25,
                )
            )
        arrays[a + "_before"], arrays[a + "_after"], arrays[a + "_gradient"] = (
            before,
            after,
            gradient,
        )
        info["updates"][a] = events
        info["actions"][a] = dict(
            endpoints=length if fresh else 0,
            capped_endpoints=0,
            max_rms=0.1 if fresh else 0.0,
        )
        info["execution"][a] = dict(
            kind="reused_warmup"
            if index <= 20
            else ("fresh_uncached_gpu" if fresh else "reused_reference"),
            reference=None if fresh else str(ref),
            reference_arm="cold" if index <= 20 else v.REUSED.get(a),
        )
    return info, arrays, ref


def fake_receipt(before, after, gradient, receipt, observed):
    assert before.shape == after.shape == gradient.shape == (256,)
    assert observed == receipt["observed"]
    assert "gradient256" not in receipt and "state256_after" not in receipt
    return dict(
        events=1,
        accepted=receipt["accepted"],
        evaluations=receipt["evaluations"],
        nonzero_gradients=int(receipt["gradient_norm"] > 0),
        observed_scalars=observed,
    )


def audit(info, arrays, ref, index=21, length=300):
    return v.check_trace(
        info, arrays, length, index=index, reference_path=ref, reference_sha="d" * 64
    )


def test_six_fresh_receipts_are_delegated_and_archived_arms_have_no_new_execution(
    monkeypatch,
):
    info, arrays, ref = trace_fixture()
    calls = []

    def receipt(*args):
        calls.append(args)
        return fake_receipt(*args)

    monkeypatch.setattr(v, "verify_receipt", receipt)
    totals, actions = audit(info, arrays, ref)
    assert len(calls) == 12
    assert totals["meta_real"]["evaluations"] == 2
    assert (
        totals["fixed_full"]["events"] == 0 and actions["fixed_full"]["endpoints"] == 0
    )
    assert info["updates"]["meta_real"][0]["step_norm"] > 0.1


@pytest.mark.parametrize(
    "defect",
    [
        "kind",
        "reused_event",
        "reused_action",
        "clock",
        "origin",
        "effect",
        "reset",
        "chain",
        "mask",
        "real_target",
        "reference",
    ],
)
def test_execution_clock_and_state_tampering_rejected(monkeypatch, defect):
    monkeypatch.setattr(v, "verify_receipt", fake_receipt)
    info, arrays, ref = trace_fixture()
    e = info["updates"]["meta_real"][0]
    if defect == "kind":
        info["execution"]["meta_real"]["kind"] = "reused_reference"
    elif defect == "reused_event":
        info["updates"]["cold"] = [dict(e)]
    elif defect == "reused_action":
        info["actions"]["fixed_full"]["endpoints"] = 300
    elif defect == "clock":
        e["maturity"] = 158
    elif defect == "origin":
        e["origin"] = 133
    elif defect == "effect":
        e["first_affected"] = 159
    elif defect == "reset":
        arrays["meta_real_before"][0, 0, 0] = -0.0
    elif defect == "chain":
        arrays["meta_real_before"][1, 1, 1] = 1.0
    elif defect == "mask":
        e["mask_sha256"] = "e" * 64
    elif defect == "real_target":
        e["target_sha256"] = "e" * 64
    elif defect == "reference":
        info["execution"]["cold"]["reference"] = "/other.npz"
    with pytest.raises(ValueError):
        audit(info, arrays, ref)


def test_receipt_validation_errors_propagate(monkeypatch):
    info, arrays, ref = trace_fixture()

    def rejected(*_):
        raise ValueError("independent Armijo receipt mismatch")

    monkeypatch.setattr(v, "verify_receipt", rejected)
    with pytest.raises(ValueError, match="Armijo"):
        audit(info, arrays, ref)


def test_real_cpu_receipt_helper_accepts_stripped_export_without_mutating_arrays():
    info, arrays, ref = trace_fixture()
    before = {name: x.copy() for name, x in arrays.items()}
    totals, _ = audit(info, arrays, ref)
    assert (
        totals["meta_real"]["accepted"] == 2 and totals["meta_real"]["evaluations"] == 2
    )
    assert all(x.tobytes() == before[name].tobytes() for name, x in arrays.items())


def test_all_warmup_rows_have_empty_new_traces(monkeypatch):
    info, arrays, ref = trace_fixture(index=1)
    monkeypatch.setattr(v, "verify_receipt", lambda *_: pytest.fail("no fresh receipt"))
    totals, _ = audit(info, arrays, ref, index=1)
    assert all(x["events"] == 0 for x in totals.values())


def test_deadline_and_posthoc_frontiers_are_independent_and_not_deployed():
    times = np.array([0.0, 3.970, 4.001])
    assert not v.classify_curve(times, [0, 1, 1], True, 4.0, 0.5)["timely"]
    row = v.classify_curve(times, [1, 0, 0], True, 4.0, 0.5)
    assert row["timely"] and row["lead_ms"] == 4000.0
    flags = np.array([True] * 20 + [False] * 20)
    points = v.posthoc_caps(np.linspace(0, 1, 40), flags)
    assert [p["fpr_cap"] for p in points] == [0.10, 0.15, 0.20]
    assert all(
        p["scope"] == "POSTHOC_DEVELOPMENT_NOT_DEPLOYED" and p["fpr"] <= p["fpr_cap"]
        for p in points
    )


def gate_fixture():
    m = dict(n_disrupt=180, n_safe=180, tp=125, fp=35, J=0.5)
    return {a: {"full360": dict(m)} for a in v.ARMS}


def test_integer_gate_allows_fp_tradeoff_but_no_epsilon_tie():
    scores = gate_fixture()
    scores["meta_real"]["full360"].update(tp=126, fp=36, J=0.50000000001)
    assert not v.score_gates(scores)["operational_gate"]
    scores["meta_real"]["full360"].update(tp=127, fp=36, J=0.0)
    assert v.score_gates(scores)["stage_success_pending_verification"]
    for a in ("meta_reverse", "meta_hold", "forecast_real", "fixed_full"):
        scores[a]["full360"].update(tp=127, fp=36)
        assert not v.score_gates(scores)["control_separation_gate"]
        scores[a]["full360"].update(tp=125, fp=35)
    scores["meta_real"]["full360"].update(tp=130, fp=37)
    assert not v.score_gates(scores)["operational_gate"]


def training_fixture():
    ids = list(range(1, 21))
    times = np.arange(300, dtype=np.float64) / 1000
    cohort = [
        dict(shot=s, disruptive=s <= 8, down=1.0 if s <= 8 else None, points=300)
        for s in ids
    ]
    warmup = {c["shot"]: (times, c["disruptive"], c["down"]) for c in cohort}
    rng, history = np.random.default_rng(131), []
    for step in range(128):
        pair = [ids[int(rng.integers(8))], ids[8 + int(rng.integers(12))]]
        events = [
            dict(
                shot=s,
                query=288,
                supported=True,
                maturity=287,
                label=int(s <= 8),
                risk_before=0.5,
                risk_after=0.4,
                signal=0.3,
                objective=0.2,
                replay_updates=2,
                accepted_path_updates=1,
                proposal_evaluations=3,
            )
            for s in pair
        ]
        history.append(
            dict(
                step=step + 1,
                sample_shots=pair,
                episodes={a: copy.deepcopy(events) for a in v.HEADS},
                gradient_norms={a: 0.1 for a in v.HEADS},
            )
        )
    return (
        dict(ids=ids),
        dict(support_counts={a: 256 for a in v.HEADS}),
        history,
        cohort,
        warmup,
    )


def test_training_has_128_paired_steps_two_matched_heads_and_full_clock():
    args = training_fixture()
    assert v.check_training(*args) == args[1]["support_counts"]
    args[2][0]["episodes"]["meta"][0]["proposal_evaluations"] = 23
    with pytest.raises(ValueError, match="evaluations"):
        v.check_training(*args)
    args = training_fixture()
    args[2][0]["episodes"]["forecast"][0]["maturity"] = 159
    with pytest.raises(ValueError, match="clock"):
        v.check_training(*args)


def test_unsupported_early_query_has_no_replayed_updates():
    args = training_fixture()
    e = args[2][0]["episodes"]["meta"][0]
    e.update(
        query=132,
        supported=False,
        maturity=None,
        risk_after=e["risk_before"],
        replay_updates=0,
        accepted_path_updates=0,
        proposal_evaluations=0,
    )
    args[1]["support_counts"]["meta"] = 255
    assert v.check_training(*args)["meta"] == 255


def test_incomplete_run_is_rejected_before_other_reads(monkeypatch):
    paths = []

    def read(path):
        paths.append(path.name)
        return dict(status="RUNNING")

    monkeypatch.setattr(v, "read_json", read)
    with pytest.raises(ValueError, match="COMPLETED"):
        v.verify(Path("/synthetic"))
    assert paths == ["completed.json"]


def test_cli_never_overwrites_existing_report(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["verify", "--output", "/synthetic"])
    monkeypatch.setattr(Path, "exists", lambda _: True)
    monkeypatch.setattr(v, "verify", lambda _: pytest.fail("must not execute"))
    with pytest.raises(FileExistsError):
        v.main()


def test_cli_write_only_after_explicit_verification(monkeypatch):
    written = []

    class Memory(io.StringIO):
        def __exit__(self, *args):
            written.append(self.getvalue())
            return super().__exit__(*args)

    def opened(path, mode, **kwargs):
        assert path == Path("/synthetic/verification_records.json") and mode == "x"
        return Memory()

    monkeypatch.setattr(sys, "argv", ["verify", "--output", "/synthetic"])
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(
        v, "verify", lambda _: dict(status="PASS", cases=360, curves=2880)
    )
    assert not written
    v.main()
    assert len(written) == 1 and '"PASS"' in written[0]
