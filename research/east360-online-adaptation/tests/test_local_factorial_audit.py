import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.local_factorial_20261029.acceptance import search
from experiments.risk_aligned_online.local_factorial_20261029 import audit as module
from experiments.risk_aligned_online.local_factorial_20261029.learning import predict
from experiments.risk_aligned_online.tests.test_local_readout import make_head
from experiments.risk_aligned_online.tests.test_qk_matched_head import fixture
from experiments.risk_aligned_online.local_factorial_20261029.audit import (
    ADAPTIVE_ARMS,
    PREVIOUS_AUDIT_SHA256,
    auxiliary,
    audit_arm_receipt,
    check_feedback,
    check_factorial_protocol,
    check_mode_contract,
    check_prior_metadata,
    choose_replay_shots,
    decision,
    necessary_screen,
)


def accepted():
    return dict(
        observed=1050,
        trials=3,
        factor=0.25,
        reason="accepted",
        loss_before=10.0,
        loss_after=9.95,
        proposed_norm=0.05,
        accepted_norm=0.0125,
        directional_dot=-0.025,
        proposed_directional_dot=-0.1,
        gradient_norm=2.0,
        trial_losses=[10.1, 10.05, 9.95],
    )


def no_observations():
    return dict(
        observed=0,
        trials=0,
        factor=None,
        reason="no_observed_target",
        loss_before=None,
        loss_after=None,
        proposed_norm=0.0,
        accepted_norm=0.0,
        directional_dot=None,
        proposed_directional_dot=0.0,
        gradient_norm=0.0,
        trial_losses=[],
    )


def test_armijo_receipt_accepts_first_valid_step_and_optional_trial_history():
    receipt = accepted()
    audit_arm_receipt(receipt, expected_observed=1050)
    del receipt["trial_losses"]
    audit_arm_receipt(receipt, expected_observed=1050)


def test_factorial_modes_distinguish_history_from_recent_targets():
    expected = {"prior_true": ("prior", "true")}
    expected.update(
        {
            f"{history}_{target}": (history, target)
            for history in ("true", "reverse", "hold")
            for target in ("true", "reverse", "hold")
        }
    )
    check_mode_contract(expected)
    expected["reverse_true"] = ("true", "reverse")
    with pytest.raises(ValueError):
        check_mode_contract(expected)


def test_factorial_protocol_binds_candidate_modes_and_trial_sequence():
    modes = {"prior_true": ["prior", "true"]}
    modes.update(
        {
            f"{history}_{target}": [history, target]
            for history in ("true", "reverse", "hold")
            for target in ("true", "reverse", "hold")
        }
    )
    protocol = dict(
        modes=modes,
        candidate="true_true",
        comparators=[
            "cold",
            "prior_true",
            "true_reverse",
            "true_hold",
            "reverse_true",
            "hold_true",
        ],
        armijo_constant=0.0001,
        trial_factors=[2.0**-i for i in range(8)],
    )
    check_factorial_protocol(protocol)
    for field, value in (
        ("candidate", "reverse_true"),
        ("armijo_constant", 0.001),
        ("trial_factors", [1.0, 0.25]),
    ):
        altered = copy.deepcopy(protocol)
        altered[field] = value
        with pytest.raises(ValueError):
            check_factorial_protocol(altered)


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed", 1049),
        ("trials", 2),
        ("factor", 0.5),
        ("loss_after", 10.0),
        ("loss_after", 9.9999999),
        ("accepted_norm", 0.025),
        ("proposed_norm", 0.051),
        ("directional_dot", 0.025),
        ("directional_dot", -0.0125),
        ("proposed_directional_dot", -0.05),
        ("gradient_norm", float("nan")),
        ("trial_losses", [9.99, 10.05, 9.95]),
        ("trial_losses", [10.1, 10.05]),
        ("trial_losses", [float("nan"), 10.05, 9.95]),
    ],
)
def test_armijo_receipt_rejects_invalid_or_nonfirst_acceptance(field, value):
    receipt = accepted()
    receipt[field] = value
    with pytest.raises((ValueError, AssertionError)):
        audit_arm_receipt(receipt, expected_observed=1050)


def test_armijo_receipt_checks_abstention_and_nonfinite_observed_loss():
    receipt = no_observations()
    audit_arm_receipt(receipt, expected_observed=0)
    bad = copy.deepcopy(receipt)
    bad["accepted_norm"] = 0.01
    with pytest.raises(ValueError):
        audit_arm_receipt(bad, expected_observed=0)
    bad = copy.deepcopy(receipt)
    bad["reason"] = "accepted"
    with pytest.raises(ValueError):
        audit_arm_receipt(bad, expected_observed=0)
    bad = accepted()
    bad["loss_before"] = float("nan")
    with pytest.raises(ValueError):
        audit_arm_receipt(bad, expected_observed=1050)


def test_armijo_rejected_search_requires_all_eight_trials_and_zero_state():
    receipt = accepted()
    receipt.update(
        trials=8,
        factor=0.0,
        reason="no_acceptable_step",
        loss_after=10.0,
        accepted_norm=0.0,
        directional_dot=0.0,
        proposed_directional_dot=-0.1,
        trial_losses=[10.1] * 8,
    )
    audit_arm_receipt(receipt, expected_observed=1050)
    receipt["trial_losses"][3] = 9.9
    with pytest.raises(ValueError):
        audit_arm_receipt(receipt, expected_observed=1050)


def test_receipt_from_real_cpu_search_satisfies_independent_audit():
    gradient = torch.tensor([3.0, 4.0], dtype=torch.float32)
    proposal = -0.05 * gradient / (gradient.square().sum() + 1e-16).sqrt()

    def objective(x):
        return (
            10.0
            + float(gradient.double() @ x.double())
            + 20.0 * float(x.double() @ x.double())
        )

    _, receipt = search(objective, proposal, gradient, 10.0)
    receipt["observed"] = 50
    audit_arm_receipt(receipt, expected_observed=50)
    gradient.zero_()
    _, receipt = search(objective, torch.zeros_like(gradient), gradient, 10.0)
    receipt["observed"] = 50
    audit_arm_receipt(receipt, expected_observed=50)


def test_previous_audit_and_cold_readout_are_bound_to_fixed_prior():
    protocol = dict(
        order=list(range(360)),
        threshold=0.7,
        reference="ref",
        reference_hashes={"x": "refhash"},
        readout="readout",
        readout_sha256="readout_hash",
        readout_audit_sha256="readout_audit_hash",
        source_sha256="source_hash",
        previous_complete_sha256="previous_complete",
        previous_audit_sha256=PREVIOUS_AUDIT_SHA256,
    )
    old_protocol = {
        k: protocol[k]
        for k in (
            "order",
            "threshold",
            "reference",
            "reference_hashes",
            "readout",
            "readout_sha256",
            "readout_audit_sha256",
            "source_sha256",
        )
    }
    prior_audit = dict(
        complete=True,
        complete_sha256="previous_complete",
        replay=dict(prior=dict(read_ids=protocol["order"][:20], outcome_reads=0)),
    )
    readout_complete = dict(
        complete=True, artifact_sha256={"cold_readouts.npz": "readout_hash"}
    )
    readout_audit = dict(complete=True, complete_sha256="readout_complete")
    digests = dict(
        previous_complete="previous_complete",
        previous_audit=PREVIOUS_AUDIT_SHA256,
        readout="readout_hash",
        readout_audit="readout_audit_hash",
        readout_complete="readout_complete",
    )
    check_prior_metadata(
        protocol, old_protocol, prior_audit, readout_complete, readout_audit, digests
    )
    bad = copy.deepcopy(digests)
    bad["previous_audit"] = "altered"
    with pytest.raises(ValueError):
        check_prior_metadata(
            protocol, old_protocol, prior_audit, readout_complete, readout_audit, bad
        )
    bad = copy.deepcopy(prior_audit)
    bad["replay"]["prior"]["outcome_reads"] = 1
    with pytest.raises(ValueError):
        check_prior_metadata(
            protocol, old_protocol, bad, readout_complete, readout_audit, digests
        )


def feedback_fixture():
    n = 400
    signal = dict(values=np.zeros((n, 23)), available=np.ones((n, 23), bool))
    active = np.ones(23, bool)
    active[[2, 10]] = False
    records = []
    for j, clock in enumerate((256, 384)):
        records.append(
            dict(
                clock=clock,
                origins=[clock - v for v in (101, 76, 51, 26)],
                calibration_last_target=clock - 51,
                gradient_first_target=clock - 50,
                gradient_last_target=clock - 1,
                history_fit_rows=2 + 4 * j,
                fit_counts_max=2 + 4 * j,
                history_rows_after=4 + 4 * j,
                readout_delta_norm=0.1,
                arms={arm: accepted() for arm in ADAPTIVE_ARMS},
            )
        )
    updates = {
        arm: dict(
            opportunities=2,
            observed=2100,
            updates=2,
            decreased=2,
            max_norm=0.0125,
            loss_before_sum=20.0,
            loss_after_sum=19.9,
            trials=6,
            accepted_norm_sum=0.025,
        )
        for arm in ADAPTIVE_ARMS
    }
    updates["feedback"] = records
    return updates, signal, active


def test_feedback_recounts_raw_masks_history_and_all_arm_aggregates():
    updates, signal, active = feedback_fixture()
    check_feedback(updates, signal, active)
    bad = copy.deepcopy(updates)
    bad["feedback"][0]["calibration_last_target"] += 1
    with pytest.raises(ValueError):
        check_feedback(bad, signal, active)
    bad = copy.deepcopy(updates)
    bad[ADAPTIVE_ARMS[0]]["observed"] += 1
    with pytest.raises(ValueError):
        check_feedback(bad, signal, active)
    bad = copy.deepcopy(updates)
    bad[ADAPTIVE_ARMS[0]]["trials"] += 1
    with pytest.raises(ValueError):
        check_feedback(bad, signal, active)
    bad = copy.deepcopy(updates)
    bad[ADAPTIVE_ARMS[0]]["accepted_norm_sum"] += 0.01
    with pytest.raises(ValueError):
        check_feedback(bad, signal, active)
    bad = copy.deepcopy(updates)
    bad["feedback"][0]["arms"][ADAPTIVE_ARMS[1]]["observed"] -= 25
    with pytest.raises(ValueError):
        check_feedback(bad, signal, active)
    changed = copy.deepcopy(signal)
    changed["available"][207, 0] = False
    with pytest.raises(ValueError):
        check_feedback(updates, changed, active)


def test_audit_accepts_actual_cpu_factorial_feedback():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        net, sample = fixture()
        prior = make_head()
        sig = {key: sample[key][:400] for key in ("values", "available")}
        _, updates, _ = predict(net, sig, None, prior)
        totals = check_feedback(updates, sig, prior.active.numpy())
    finally:
        torch.set_num_threads(previous_threads)
    assert all(row["opportunities"] == 2 for row in totals.values())


def test_first_alarm_at_exactly_30_ms_is_not_timely():
    row = decision(7, [0.970, 0.975], [0.8, 0.9], True, 1.0, 0.5)
    assert row["alarm"] is True
    assert row["lead_ms"] == 30.0
    assert row["timely"] is False
    row = decision(7, [0.969, 0.970], [0.8, 0.9], True, 1.0, 0.5)
    assert row["lead_ms"] == 31.0 and row["timely"] is True


def test_replay_includes_every_changed_shot_and_fixed_indices():
    order = list(range(360))
    comparisons = {
        "true_true": dict(rescued=[34], harmed=[207]),
        "hold_hold": dict(rescued=[359], harmed=[34]),
    }
    assert choose_replay_shots(order, comparisons) == [20, 34, 180, 207, 359]


def test_replay_also_includes_changed_first_alarm_with_no_rescue_or_harm():
    order = list(range(360))
    cold = [dict(shot=sid, alarm=False, timely=False, lead_ms=None) for sid in order]
    adapted = copy.deepcopy(cold)
    adapted[91]["alarm"] = True
    actual = {"cold": cold, "true_true": adapted}
    comparisons = {"true_true": dict(rescued=[], harmed=[])}
    assert choose_replay_shots(order, comparisons, actual) == [20, 91, 180, 359]


def test_auxiliary_masks_use_only_available_preterminal_targets():
    n = 300
    signal = dict(values=np.zeros((n, 23)), available=np.ones((n, 23), bool))
    active = np.ones(23, bool)
    active[[2, 10]] = False
    saved = dict(aux_clock=np.array([256]), aux_last=np.zeros((1, 23)))
    saved["aux_available"] = active[None].copy()
    for arm in ADAPTIVE_ARMS:
        for part in ("before", "after"):
            saved[f"aux_{arm}_{part}"] = np.ones((1, 25, 23))
    signal["available"][256:258, 0] = False
    valid = np.arange(n) < 260
    rows = auxiliary(saved, signal, valid, np.zeros(23), np.ones(23), active)
    assert rows[0]["elements"] == 82
    assert rows[0]["losses"]["hold"]["sum"] == 0
    assert rows[0]["losses"]["true_true_after"]["sum"] == 41
    changed = copy.deepcopy(saved)
    changed["aux_last"][0, 1] = 5
    with pytest.raises(ValueError):
        auxiliary(changed, signal, valid, np.zeros(23), np.ones(23), active)
    changed = copy.deepcopy(saved)
    changed["aux_available"][0, 2] = True
    with pytest.raises(ValueError):
        auxiliary(changed, signal, valid, np.zeros(23), np.ones(23), active)


def test_necessary_screen_checks_fixed_candidate_and_all_matched_controls():
    metrics = {
        arm: {"full360": dict(tp=125, fpr=0.1, J=0.6)}
        for arm in ("cold", *ADAPTIVE_ARMS)
    }
    metrics["true_true"]["full360"]["J"] = 0.61
    assert necessary_screen(metrics) is True
    metrics["hold_true"]["full360"]["J"] = 0.61
    assert necessary_screen(metrics) is False
    metrics["hold_true"]["full360"]["J"] = 0.6
    metrics["true_true"]["full360"]["tp"] = 124
    assert necessary_screen(metrics) is False


def test_cpu_full_recount_with_sealed_synthetic_360(monkeypatch):
    """Exercise run without disk writes, real data access, or launching replay."""
    root = Path("/__factorial_audit_test_only__")
    output = root / "probe_outputs/local_factorial_east360_v1"
    previous = root / "probe_outputs/local_readout_east360_v1"
    readout = root / "readout"
    reference = root / "reference"
    order = list(range(360))
    updates, sig, active = feedback_fixture()
    times = np.arange(400) / 1000
    arrays = {
        "times": times,
        "aux_clock": np.array([256]),
        "aux_last": np.zeros((1, 23)),
        "aux_available": active[None],
    }
    for arm in module.ARMS:
        arrays[arm] = np.full(400, 0.1)
    for arm in ADAPTIVE_ARMS:
        for part in ("before", "after"):
            arrays[f"aux_{arm}_{part}"] = np.zeros((1, 25, 23))
    sample = auxiliary(
        arrays, sig, np.ones(400, bool), np.zeros(23), np.ones(23), active
    )
    aux_rows = [dict(shot=sid, samples=sample) for sid in order[20:]]
    methods = list(sample[0]["losses"])
    per_aux = [dict(shot=sid, windows=1, loss=0.0) for sid in order[20:]]
    aux_summary = {
        name: dict(shots=340, windows=340, mean=0.0, shot_sd=0.0, per_shot=per_aux)
        for name in methods
    }
    decisions = [
        decision(sid, times, arrays["cold"], bool(sid % 2), 1.0, 0.7) for sid in order
    ]
    metrics = {
        label: module.count(sub)
        for label, sub in (
            ("full360", decisions),
            ("cold20", decisions[:20]),
            ("suffix340", decisions[20:]),
        )
    }
    declared = {arm: decisions for arm in module.ARMS}
    comparisons = {arm: dict(rescued=[], harmed=[]) for arm in ADAPTIVE_ARMS}
    seals = [
        dict(
            shot=sid,
            index=i + 1,
            sha256="curve-hash",
            outcome_used=False,
            adaptation_active=i >= 20,
            updates=updates if i >= 20 else {},
            aux_samples=int(i >= 20),
        )
        for i, sid in enumerate(order)
    ]
    source_hash = "source-hash"
    filenames = [
        "experiments/risk_aligned_online/local_factorial_20261029/" + n
        for n in ("run.py", "learning.py", "acceptance.py")
    ]

    def digest(name):
        return hashlib.sha256(name.encode()).hexdigest()

    protocol = dict(
        arms=module.ARMS,
        order=order,
        source_sha256=source_hash,
        threshold=0.7,
        warmup=20,
        step=0.05,
        history_rows=16,
        prior_precision=1.0,
        inter_updates_after_cold20=0,
        alarm_updates=0,
        reset_each_feedback=True,
        reset_each_shot=True,
        modes={
            "prior_true": ["prior", "true"],
            **{
                f"{h}_{t}": [h, t]
                for h in ("true", "reverse", "hold")
                for t in ("true", "reverse", "hold")
            },
        },
        candidate="true_true",
        comparators=[
            "cold",
            "prior_true",
            "true_reverse",
            "true_hold",
            "reverse_true",
            "hold_true",
        ],
        armijo_constant=0.0001,
        trial_factors=[2.0**-i for i in range(8)],
        reference="reference",
        reference_hashes={"input_map.json": "map-hash"},
        previous=str(previous),
        previous_complete_sha256="previous-complete",
        previous_audit_sha256=PREVIOUS_AUDIT_SHA256,
        readout="readout",
        readout_sha256="readout-hash",
        readout_audit_sha256="readout-audit-hash",
        plan_sha256=digest("LOCAL_READOUT_FACTORIAL_NEXT_PLAN.md"),
        code_sha256={name: digest(name) for name in filenames},
    )
    old_protocol = {
        k: protocol[k]
        for k in (
            "order",
            "threshold",
            "reference",
            "reference_hashes",
            "readout",
            "readout_sha256",
            "readout_audit_sha256",
            "source_sha256",
        )
    }
    done = dict(
        complete=True,
        results={arm: metrics for arm in module.ARMS},
        comparisons=comparisons,
        necessary_intra_screen=False,
        update_counts={arm: 680 for arm in ADAPTIVE_ARMS},
        **{
            name + "_sha256": name + "-hash"
            for name in (
                "protocol",
                "per_shot",
                "receipts",
                "auxiliary_summary",
                "auxiliary_per_shot",
            )
        },
    )
    old_audit = dict(
        complete=True,
        complete_sha256="previous-complete",
        replay=dict(prior=dict(read_ids=order[:20], outcome_reads=0)),
    )
    files = {
        output / "protocol.json": protocol,
        output / "complete.json": done,
        output / "per_shot.json": declared,
        output / "receipts.json": seals,
        output / "auxiliary_per_shot.json": aux_rows,
        output / "auxiliary_summary.json": aux_summary,
        previous / "protocol.json": old_protocol,
        previous / "complete.json": dict(
            complete=True,
            protocol_sha256="old-protocol",
            receipts_sha256="old-receipts",
        ),
        previous / "audit.json": old_audit,
        previous / "receipts.json": seals,
        readout / "complete.json": dict(
            complete=True, artifact_sha256={"cold_readouts.npz": "readout-hash"}
        ),
        readout / "audit_with_fold_gap.json": dict(
            complete=True, complete_sha256="readout-complete"
        ),
        reference / "input_map.json": dict(calibration_ids=order[:20]),
        reference / "input_hashes.json": {str(sid): "raw-hash" for sid in order},
    }
    for i, sid in enumerate(order):
        name = f"{i + 1:04d}_{sid}_seal.json"
        files[output / "shots" / name] = seals[i]
        files[previous / "shots" / name] = seals[i]
    hashes = {
        root / "source.pt": source_hash,
        previous / "complete.json": "previous-complete",
        previous / "audit.json": PREVIOUS_AUDIT_SHA256,
        previous / "protocol.json": "old-protocol",
        previous / "receipts.json": "old-receipts",
        readout / "cold_readouts.npz": "readout-hash",
        readout / "audit_with_fold_gap.json": "readout-audit-hash",
        readout / "complete.json": "readout-complete",
        reference / "input_map.json": "map-hash",
    }
    hashes.update(
        {
            output / (name + ".json"): name + "-hash"
            for name in (
                "protocol",
                "per_shot",
                "receipts",
                "auxiliary_summary",
                "auxiliary_per_shot",
            )
        }
    )
    hashes.update(
        {
            root / name: digest(name)
            for name in filenames + ["LOCAL_READOUT_FACTORIAL_NEXT_PLAN.md"]
        }
    )

    def fake_sha(path):
        path = Path(path)
        if path.suffix == ".npz":
            return "curve-hash" if path.name != "cold_readouts.npz" else "readout-hash"
        if path.parent == root / "raw":
            return "raw-hash"
        return hashes.get(path, "audit-code-hash")

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, name):
            return name.encode()

    class Npz:
        def __enter__(self):
            return arrays

        def __exit__(self, *args):
            return False

    class Report(io.StringIO):
        def fileno(self):
            return 1

        def close(self):
            pass

    report = Report()
    monkeypatch.setattr(module.io23, "ROOT", root)
    monkeypatch.setattr(
        module.io23, "path_for", lambda device, sid: root / "raw" / str(sid)
    )
    monkeypatch.setattr(
        module.io23, "read_outcome", lambda device, sid: (bool(sid % 2), 1.0)
    )
    monkeypatch.setattr(module, "DEFAULT_PARENT", root / "source.pt")
    monkeypatch.setattr(module, "PARENT_SHA256", source_hash)
    monkeypatch.setattr(
        module, "read", lambda directory, name: files[Path(directory) / name]
    )
    monkeypatch.setattr(module, "sha", fake_sha)
    monkeypatch.setattr(module, "validate_test_ids", lambda *args: None)
    monkeypatch.setattr(module, "signal", lambda *args: sig)
    monkeypatch.setattr(
        module,
        "load_parent",
        lambda *args: (
            SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23)),
            {},
        ),
    )
    monkeypatch.setattr(
        module, "load_head", lambda *args: SimpleNamespace(active=torch.tensor(active))
    )
    monkeypatch.setattr(module.zipfile, "ZipFile", lambda *args: Archive())
    monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: Npz())
    monkeypatch.setattr(
        module, "replay", lambda root, p, selected, mapping: {"shots": selected}
    )
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: report)
    monkeypatch.setattr(module.os, "fsync", lambda *args: None)
    result = module.run(output)
    assert result["replay"]["shots"] == [20, 180, 359]
    assert result["receipt_count"] == 340 * 2 * len(ADAPTIVE_ARMS)
    assert result["all_first_passing_verified"] is True
    assert result["metrics"]["true_true"]["full360"]["n_disrupt"] == 180
    assert '"complete": true' in report.getvalue()
