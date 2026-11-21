"""CPU-only verifier contract tests; all synthetic records stay in memory."""

import copy
import hashlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from experiments.risk_aligned_online.innovation_metric_20261102 import verify as v


def test_classifier_uses_first_preterminal_crossing_and_strict_lead():
    threshold = 0.5
    times = np.array([0.969, 0.970, 0.980, 1.001])
    assert v.classify_curve(times, [0, 1, 1, 1], True, 1.0, threshold) == {
        "alarm": True,
        "lead_ms": 30.0,
        "timely": False,
    }
    assert v.classify_curve(times, [1, 1, 1, 1], True, 1.0, threshold)["timely"]
    assert not v.classify_curve(times, [0, 0, 0, 1], True, 1.0, threshold)["alarm"]
    assert v.classify_curve(times, [0, 0, 0, 1], False, None, threshold)["alarm"]
    with pytest.raises(ValueError):
        v.classify_curve(times, [0, 0, float("nan"), 1], False, None, threshold)


def sample_curves():
    times = np.arange(200, dtype=np.float64) / 1000
    cold = np.linspace(0, 1, len(times), dtype=np.float32)
    old = cold.copy()
    old[160:] = 0.25
    curves = {arm: cold.copy() for arm in v.ARMS}
    curves["contrast_unit"] = old.copy()
    return times, cold, old, curves


def test_curve_checks_are_bitwise_and_preserve_complete_reference():
    times, cold, old, curves = sample_curves()
    v.validate_curves(times, curves, times, cold, old, warmup=False)
    changed = copy.deepcopy(curves)
    changed["innovation_metric"][0] = -0.0
    with pytest.raises(ValueError, match="first 160"):
        v.validate_curves(times, changed, times, cold, old, warmup=False)
    changed = copy.deepcopy(curves)
    changed["contrast_unit"][180] = np.float32(0.3)
    with pytest.raises(ValueError, match="contrast"):
        v.validate_curves(times, changed, times, cold, old, warmup=False)
    changed = copy.deepcopy(curves)
    changed["cold"] = changed["cold"].astype(np.float64)
    with pytest.raises(ValueError, match="cold"):
        v.validate_curves(times, changed, times, cold, old, warmup=False)
    moved_times = times.copy()
    moved_times[0] = -0.0
    with pytest.raises(ValueError, match="times"):
        v.validate_curves(moved_times, curves, times, cold, old, warmup=False)


def test_warmup_requires_all_arms_identical_to_cold():
    times, cold, _, _ = sample_curves()
    curves = {arm: cold.copy() for arm in v.ARMS}
    v.validate_curves(times, curves, times, cold, cold, warmup=True)
    curves["metric_hold"][180] = 0.25
    with pytest.raises(ValueError, match="warmup"):
        v.validate_curves(times, curves, times, cold, cold, warmup=True)


def sample_trace(length=300):
    ends = list(range(159, length, 128))
    arrays, updates, actions = {}, {}, {}
    for arm in v.ARMS:
        clock = [] if arm == "cold" else ends
        before = np.zeros((len(clock), 4, 64), dtype=np.float32)
        after = np.zeros_like(before)
        gradients = np.zeros((len(clock), 256), dtype=np.float32)
        records = []
        for i, end in enumerate(clock):
            if i:
                before[i] = after[i - 1]
            after[i] = before[i]
            # A metric update can have an arbitrarily larger parameter norm.
            after[i, 0, 0] += (
                0.05 if arm in ("contrast_unit", "innovation_unit") else 0.25
            )
            gradients[i, 0] = 0.5
            step_norm = float(np.linalg.norm((after[i] - before[i]).astype(np.float64)))
            records.append(
                dict(
                    origin=end - 25,
                    maturity=end,
                    first_affected=end + 1,
                    forecast_sha256="f" * 64,
                    mask_sha256="a" * 64,
                    target_sha256=("b" if arm in v.REAL_ARMS else "c") * 64,
                    observed=100,
                    accepted=True,
                    gradient_computed=True,
                    gradient_norm=0.5,
                    step_norm=step_norm,
                    reason="improved",
                    loss_before=2.0,
                    loss_proposal=1.0,
                )
            )
        arrays[arm + "_before"], arrays[arm + "_after"] = before, after
        arrays[arm + "_gradient"] = gradients
        updates[arm] = records
        actions[arm] = dict(endpoints=length, capped_endpoints=0, max_rms=0.1)
    return dict(
        reset_each_shot=True, temporary_parameters=256, updates=updates, actions=actions
    ), arrays


def test_metric_step_can_exceed_unit_step_and_hold_can_update():
    info, arrays = sample_trace()
    totals, _ = v.check_trace(info, arrays, 300, False)
    assert totals["innovation_metric"]["accepted"] == 2
    assert totals["metric_hold"]["accepted"] == 2
    assert info["updates"]["innovation_metric"][0]["step_norm"] > 0.1
    assert totals["cold"]["events"] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "acceptance",
        "rejected_change",
        "gradient_norm",
        "step_norm",
        "clock",
        "stitch",
        "reset",
        "mask",
        "real_target",
        "unit_step",
    ],
)
def test_trace_rejects_forged_receipts_and_broken_state(defect):
    info, arrays = sample_trace()
    arm = "innovation_metric"
    row = info["updates"][arm][0]
    if defect == "acceptance":
        row["loss_proposal"] = 3.0
    elif defect == "rejected_change":
        row.update(accepted=False, reason="not_improved", loss_proposal=3.0)
    elif defect == "gradient_norm":
        row["gradient_norm"] = 0.4
    elif defect == "step_norm":
        row["step_norm"] = 0.2
    elif defect == "clock":
        row["first_affected"] = 159
    elif defect == "stitch":
        arrays[arm + "_before"][1, 0, 0] += 1
    elif defect == "reset":
        arrays[arm + "_before"][0, 0, 0] += 1
    elif defect == "mask":
        row["mask_sha256"] = "d" * 64
    elif defect == "real_target":
        row["target_sha256"] = "d" * 64
    elif defect == "unit_step":
        arrays["innovation_unit_after"][0, 0, 0] = 0.5
        arrays["innovation_unit_before"][1] = arrays["innovation_unit_after"][0]
        arrays["innovation_unit_after"][1, 0, 0] = 0.55
        info["updates"]["innovation_unit"][0]["step_norm"] = 0.5
    with pytest.raises(ValueError):
        v.check_trace(info, arrays, 300, False)


def test_no_target_is_explicit_and_rejection_keeps_state():
    info, arrays = sample_trace(length=200)
    # All arms share the availability mask and observed count.
    for arm in v.LEARNERS:
        row = info["updates"][arm][0]
        row.update(
            observed=0,
            accepted=False,
            gradient_computed=False,
            gradient_norm=0.0,
            step_norm=0.0,
            reason="no_target",
        )
        row.pop("loss_before")
        row.pop("loss_proposal")
        arrays[arm + "_gradient"][:] = 0
        arrays[arm + "_after"][:] = arrays[arm + "_before"]
    v.check_trace(info, arrays, 200, False)
    info["updates"]["metric_hold"][0]["gradient_computed"] = True
    with pytest.raises(ValueError):
        v.check_trace(info, arrays, 200, False)


def sample_training():
    ids = list(range(1, 21))
    times = np.arange(200, dtype=np.float64) / 1000
    cohort = [
        dict(shot=i, disruptive=i <= 8, down=1.0 if i <= 8 else None, points=len(times))
        for i in ids
    ]
    warmup = {c["shot"]: (times, c["disruptive"], c["down"]) for c in cohort}
    history = []
    rng = np.random.default_rng(131)
    for step in range(1, 129):
        selected = [ids[int(rng.integers(8))], ids[8 + int(rng.integers(12))]]
        episodes = {}
        for arm in v.HEADS:
            episodes[arm] = [
                dict(
                    shot=shot,
                    query=160,
                    supported=True,
                    maturity=159,
                    label=label,
                    risk_before=0.5,
                    risk_after=0.4,
                    signal=0.3,
                    objective=0.2,
                )
                for shot, label in zip(selected, (1, 0), strict=True)
            ]
        history.append(
            dict(
                step=step,
                sample_shots=selected,
                episodes=episodes,
                gradient_norms={arm: 0.1 for arm in v.HEADS},
            )
        )
    protocol = dict(ids=ids)
    done = dict(support_counts={arm: 256 for arm in v.HEADS})
    return protocol, done, history, cohort, warmup


def test_training_recounts_all_heads_and_hdf_outcome_authority():
    p, done, history, cohort, warmup = sample_training()
    assert v.check_training(p, done, history, cohort, warmup) == done["support_counts"]
    history[0]["episodes"]["metric_forecast"][0]["label"] = 0
    with pytest.raises(ValueError, match="label"):
        v.check_training(p, done, history, cohort, warmup)


def test_training_requires_paired_classes_and_correct_support_clock():
    p, done, history, cohort, warmup = sample_training()
    history[0]["sample_shots"] = [1, 2]
    with pytest.raises(ValueError, match="sample"):
        v.check_training(p, done, history, cohort, warmup)
    p, done, history, cohort, warmup = sample_training()
    history[0]["episodes"]["metric_operational"][0]["maturity"] = 158
    with pytest.raises(ValueError, match="clock"):
        v.check_training(p, done, history, cohort, warmup)


def test_training_requires_recorded_seeded_draws_not_just_any_d_s_pair():
    p, done, history, cohort, warmup = sample_training()
    actual = history[0]["sample_shots"][0]
    other = 1 if actual != 1 else 2
    history[0]["sample_shots"][0] = other
    for arm in v.HEADS:
        history[0]["episodes"][arm][0]["shot"] = other
    with pytest.raises(ValueError, match="sample"):
        v.check_training(p, done, history, cohort, warmup)


def test_training_completion_has_two_step_fields_but_no_number_heads():
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
        independent_confirmation=False,
    )
    v.check_training_contract(done, protocol, ids, "source", "cold", "map", "plan")
    done["steps"] = 127
    with pytest.raises(ValueError, match="training completion"):
        v.check_training_contract(done, protocol, ids, "source", "cold", "map", "plan")


def test_shot_seal_is_bound_to_completion_anchored_manifest():
    def digest(n):
        return str(n) * 64

    seal = dict(
        shot=42,
        index=21,
        sha256=digest(1),
        states_sha256=digest(2),
        updates_sha256=digest(3),
        warmup_curves_reused=False,
        warmup_outcome_already_known_for_head_fit=False,
        current_outcome_used_in_prediction=False,
    )
    manifest = dict(shot=42, index=21, seal_sha256=digest(4))
    v.check_shot_seal(
        21, 42, seal, manifest, digest(4), digest(1), digest(2), digest(3)
    )
    manifest["seal_sha256"] = digest(5)
    with pytest.raises(ValueError, match="seal"):
        v.check_shot_seal(
            21, 42, seal, manifest, digest(4), digest(1), digest(2), digest(3)
        )


def test_hdf_outcome_reads_only_two_metadata_datasets(monkeypatch):
    accessed = []

    class Dataset:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, key):
            assert key == ()
            return self.value

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getitem__(self, name):
            accessed.append(name)
            return Dataset({"meta/IsDisrupt": 1, "meta/DownTime": 3.0}[name])

    monkeypatch.setitem(
        sys.modules, "h5py", SimpleNamespace(File=lambda *args: Handle())
    )
    assert v.read_outcome(71304) == (True, 3.0)
    assert accessed == ["meta/IsDisrupt", "meta/DownTime"]


def test_gate_requires_strict_superiority_to_all_four_controls():
    scores = {a: {"full360": dict(fpr=0.20, tp=100, fp=30, J=0.5)} for a in v.ARMS}
    scores["innovation_metric"]["full360"]["J"] = 0.6
    assert v.score_gates(scores)["stage_success_pending_verification"]
    for arm in v.CONTROLS:
        scores[arm]["full360"]["J"] = 0.6
        assert not v.score_gates(scores)["stage_success_pending_verification"]
        scores[arm]["full360"]["J"] = 0.5
    scores["innovation_metric"]["full360"]["fpr"] = 0.20001
    assert not v.score_gates(scores)["stage_success_pending_verification"]


def test_training_snapshot_does_not_require_unimported_runner(monkeypatch):
    module = "experiments/risk_aligned_online/innovation_metric_20261102/"
    contents = {
        name: name.encode()
        for name in (
            v.PLAN,
            module + "__init__.py",
            module + "core.py",
            module + "prepare.py",
            module + "stream.py",
        )
    }

    def archive_bytes(entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            for name, data in entries.items():
                z.writestr(name, data)
        return buffer.getvalue()

    raw = archive_bytes(contents)
    real_zip = zipfile.ZipFile
    monkeypatch.setattr(v.zipfile, "ZipFile", lambda *_: real_zip(io.BytesIO(raw)))
    monkeypatch.setattr(v, "ROOT", Path("/nonexistent-verifier-test-root"))
    monkeypatch.setattr(v, "sha_file", lambda _: "0" * 64)
    declared = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    protocol = dict(code_sha256=declared, plan_sha256=declared[v.PLAN])
    assert v.check_snapshot(Path("/nonexistent"), protocol, training=True)[
        "archive_entries_verified"
    ] == len(contents)
    with pytest.raises(ValueError, match="snapshot entries"):
        v.check_snapshot(Path("/nonexistent"), protocol, training=False)
