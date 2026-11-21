"""Independent factorial recount and selected-shot production replay.

Outcomes are used only for retrospective scoring, never to choose an update.
Running this module is explicit: importing it does not read or write artifacts.
"""

import argparse
from collections import Counter, deque
import hashlib
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_memory_20261028.checkpoint import read
from ..fixed_test_protocol import validate_test_ids
from ..gradient_split_20261029.audit import count
from ..local_readout_20261029.learning import load_head
from ..qk_forecast_stream_20261029.audit import decision, paired, signal
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_train import sha


ADAPTIVE_ARMS = (
    "prior_true",
    "true_true",
    "true_reverse",
    "true_hold",
    "reverse_true",
    "reverse_reverse",
    "reverse_hold",
    "hold_true",
    "hold_reverse",
    "hold_hold",
)
ARMS = ("cold", *ADAPTIVE_ARMS)
STEP = 0.05
ARMIJO = 0.0001
TRIALS = 8
# Captured from the immutable local-readout v1 audit before the new run.
PREVIOUS_AUDIT_SHA256 = (
    "85ae71b19e2768b1bcd0c748041c4b09a564baa47f7966ffe2ef1643160b5d85"
)


def check_mode_contract(modes):
    kinds = ("true", "reverse", "hold")
    expected = {"prior_true": ("prior", "true")}
    expected.update(
        {
            f"{history}_{target}": (history, target)
            for history in kinds
            for target in kinds
        }
    )
    if (
        not isinstance(modes, dict)
        or any(not isinstance(value, (list, tuple)) for value in modes.values())
        or {key: tuple(value) for key, value in modes.items()} != expected
    ):
        raise ValueError("factorial history-fit and recent-target mappings differ")


def check_factorial_protocol(protocol):
    check_mode_contract(protocol["modes"])
    if (
        protocol["candidate"] != "true_true"
        or tuple(protocol["comparators"])
        != (
            "cold",
            "prior_true",
            "true_reverse",
            "true_hold",
            "reverse_true",
            "hold_true",
        )
        or protocol["armijo_constant"] != ARMIJO
        or protocol["trial_factors"] != [2.0**-i for i in range(TRIALS)]
    ):
        raise ValueError("declared candidate, controls, or line search differs")


def check_prior_metadata(
    protocol, previous, audit, readout_complete, readout_audit, digests
):
    """Bind the fixed, previously NumPy-verified cold20 prior to this run."""
    fields = (
        "order",
        "threshold",
        "reference",
        "reference_hashes",
        "readout",
        "readout_sha256",
        "readout_audit_sha256",
        "source_sha256",
    )
    if any(protocol[k] != previous[k] for k in fields):
        raise ValueError("factorial protocol changed the prior experiment contract")
    if (
        digests["previous_audit"] != PREVIOUS_AUDIT_SHA256
        or protocol["previous_audit_sha256"] != digests["previous_audit"]
        or protocol["previous_complete_sha256"] != digests["previous_complete"]
        or not audit["complete"]
        or audit["complete_sha256"] != digests["previous_complete"]
        or audit["replay"]["prior"]["read_ids"] != protocol["order"][:20]
        or audit["replay"]["prior"]["outcome_reads"] != 0
        or protocol["readout_sha256"] != digests["readout"]
        or protocol["readout_audit_sha256"] != digests["readout_audit"]
        or not readout_complete["complete"]
        or readout_complete["artifact_sha256"]["cold_readouts.npz"]
        != digests["readout"]
        or not readout_audit["complete"]
        or readout_audit["complete_sha256"] != digests["readout_complete"]
    ):
        raise ValueError("previous audit or cold20 readout hash chain differs")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _near(a, b, *, rtol=2e-7, atol=1e-9):
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _worktree_path(value):
    path = Path(value)
    return path if path.is_absolute() else io23.ROOT / path


def _passes(before, after, dot):
    return (
        math.isfinite(after)
        and dot < 0
        and after < before
        and after <= before + ARMIJO * dot
    )


def audit_arm_receipt(receipt, expected_observed):
    """Validate one optimization receipt without trusting its aggregate counters.

    If trial losses are present, reconstruct every prior Armijo test from the
    scalar gradient/proposal norms; accepted proposals are negative gradients.
    Without those losses, first-passing status is not independently verifiable.
    """
    required = {
        "observed",
        "trials",
        "factor",
        "reason",
        "loss_before",
        "loss_after",
        "proposed_norm",
        "accepted_norm",
        "directional_dot",
        "gradient_norm",
        "proposed_directional_dot",
    }
    if not required.issubset(receipt):
        raise ValueError("incomplete acceptance receipt")
    observed, trials = receipt["observed"], receipt["trials"]
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed != expected_observed
        or isinstance(trials, bool)
        or not isinstance(trials, int)
        or not 0 <= trials <= TRIALS
        or not isinstance(receipt["reason"], str)
        or not receipt["reason"]
    ):
        raise ValueError("feedback observation, trial count, or reason differs")
    proposed = _number(receipt["proposed_norm"], "proposed_norm")
    accepted = _number(receipt["accepted_norm"], "accepted_norm")
    if not 0 <= proposed <= STEP + 1e-6 or not 0 <= accepted <= STEP + 1e-6:
        raise ValueError("invalid Q/K norm")
    history = receipt.get("trial_losses")
    if history is not None and (
        not isinstance(history, list) or len(history) != trials
    ):
        raise ValueError("trial loss list does not match trial count")
    if not observed:
        if (
            trials != 0
            or proposed != 0
            or accepted != 0
            or receipt["factor"] not in (None, 0)
            or receipt["gradient_norm"] not in (None, 0)
            or receipt["directional_dot"] not in (None, 0)
            or receipt.get("proposed_directional_dot") not in (None, 0)
            or receipt["loss_before"] is not None
            or receipt["loss_after"] is not None
            or receipt["reason"] != "no_observed_target"
        ):
            raise ValueError("unobserved feedback must abstain with zero state")
        return
    before = _number(receipt["loss_before"], "loss_before")
    gradient = _number(receipt["gradient_norm"], "gradient_norm")
    if before < 0 or gradient < 0:
        raise ValueError("negative loss or gradient norm")
    proposed_dot = _number(
        receipt["proposed_directional_dot"], "proposed_directional_dot"
    )
    if not _near(proposed_dot, -gradient * proposed, rtol=2e-5, atol=1e-300):
        raise ValueError("proposal does not point along the negative gradient")
    if not accepted:
        if (
            receipt["factor"] not in (None, 0)
            or receipt["directional_dot"] not in (None, 0)
            or receipt["reason"]
            != (
                "zero_direction"
                if proposed == 0 or gradient == 0
                else "no_acceptable_step"
            )
            or receipt["loss_after"] is not None
            and not _near(_number(receipt["loss_after"], "loss_after"), before)
            or (proposed == 0 and trials != 0)
            or (gradient == 0 and (proposed != 0 or trials != 0))
            or (gradient > 0 and proposed > 0 and trials != TRIALS)
        ):
            raise ValueError("invalid zero-state abstention")
    else:
        if (
            receipt["reason"] != "accepted"
            or gradient <= 0
            or proposed <= 0
            or trials == 0
        ):
            raise ValueError("invalid accepted step")
        factor = _number(receipt["factor"], "factor")
        after = _number(receipt["loss_after"], "loss_after")
        dot = _number(receipt["directional_dot"], "directional_dot")
        if (
            factor != 2.0 ** (1 - trials)
            or not _near(accepted, proposed * factor, rtol=2e-5, atol=1e-300)
            or not _near(dot, -gradient * accepted, rtol=2e-5, atol=1e-300)
            or not _near(dot, proposed_dot * factor, rtol=2e-5, atol=1e-300)
            or not _passes(before, after, dot)
        ):
            raise ValueError("accepted step violates backtracking or Armijo")
    if history is not None:
        for index, loss in enumerate(history):
            loss = _number(loss, "trial loss")
            dot = proposed_dot * 2.0 ** (-index)
            passes = _passes(before, loss, dot)
            is_last_accepted = bool(accepted and index == trials - 1)
            if passes != is_last_accepted:
                raise ValueError("step was not the first passing Armijo trial")
            if is_last_accepted and loss != receipt["loss_after"]:
                raise ValueError("accepted trial loss differs from recorded loss")


def check_feedback(updates, sig, active):
    """Recount actual arrived complete-channel masks and feedback histories."""
    values, available = np.asarray(sig["values"]), np.asarray(sig["available"])
    active = np.asarray(active, dtype=bool)
    if (
        values.ndim != 2
        or values.shape[1] != 23
        or available.shape != values.shape
        or active.shape != (23,)
    ):
        raise ValueError("invalid signal or active-channel dimensions")
    clocks = list(range(256, len(values), 128))
    records = updates["feedback"]
    if len(records) != len(clocks):
        raise ValueError("feedback clock count differs")
    history = deque(maxlen=16)
    totals = {
        arm: dict(
            opportunities=0,
            updates=0,
            observed=0,
            decreased=0,
            max_norm=0.0,
            trials=0,
            loss_before_sum=0.0,
            loss_after_sum=0.0,
            accepted_norm_sum=0.0,
        )
        for arm in ADAPTIVE_ARMS
    }
    for clock, record in zip(clocks, records):
        origins = [clock - v for v in (101, 76, 51, 26)]
        masks = []
        for origin in origins:
            mask = (
                available[origin + 1 : origin + 26]
                & np.isfinite(values[origin + 1 : origin + 26])
            ).all(0)
            mask &= available[origin] & np.isfinite(values[origin]) & active
            masks.append(mask)
        history.extend(masks[:2])
        expected = dict(
            clock=clock,
            origins=origins,
            calibration_last_target=clock - 51,
            gradient_first_target=clock - 50,
            gradient_last_target=clock - 1,
            history_fit_rows=len(history),
            fit_counts_max=int(np.stack(history).sum(0).max()),
        )
        history.extend(masks[2:])
        expected["history_rows_after"] = len(history)
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("feedback fit/gradient timing, history, or mask differs")
        delta = record.get("readout_delta_norm")
        deltas = delta.values() if isinstance(delta, dict) else [delta]
        if any(_number(v, "readout_delta_norm") < 0 for v in deltas):
            raise ValueError("invalid readout change")
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ADAPTIVE_ARMS):
            raise ValueError("each adaptive arm needs a detailed feedback receipt")
        observed = sum(int(mask.sum()) * 25 for mask in masks[2:])
        for arm in ADAPTIVE_ARMS:
            item = arms[arm]
            audit_arm_receipt(item, observed)
            agg = totals[arm]
            agg["opportunities"] += 1
            agg["observed"] += observed
            agg["updates"] += int(item["accepted_norm"] > 0)
            agg["trials"] += item["trials"]
            agg["max_norm"] = max(agg["max_norm"], item["accepted_norm"])
            agg["accepted_norm_sum"] += item["accepted_norm"]
            if observed:
                before = item["loss_before"]
                after = item["loss_after"] if item["loss_after"] is not None else before
                agg["loss_before_sum"] += before
                agg["loss_after_sum"] += after
                agg["decreased"] += int(after < before)
    for arm, expected in totals.items():
        actual = updates[arm]
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{arm} feedback aggregate {key} differs")
            if key in ("opportunities", "updates", "observed", "decreased", "trials"):
                if type(actual[key]) is not int or actual[key] != value:
                    raise ValueError(f"{arm} feedback aggregate {key} differs")
            elif not _near(_number(actual[key], key), value):
                raise ValueError(f"{arm} feedback aggregate {key} differs")
    return totals


def choose_replay_shots(order, comparisons, actual=None):
    if len(order) != 360 or len(set(order)) != 360:
        raise ValueError("360 unique ordered shots required")
    selected = {order[i] for i in (20, 180, 359)}
    for comparison in comparisons.values():
        for key in ("rescued", "harmed"):
            selected.update(comparison[key])
    if actual is not None:
        baseline = actual["cold"]
        if len(baseline) != len(order):
            raise ValueError("cold decision count differs")
        for arm, records in actual.items():
            if len(records) != len(order):
                raise ValueError("arm decision count differs")
            for sid, cold, row in zip(order, baseline, records):
                if cold["shot"] != sid or row["shot"] != sid:
                    raise ValueError("arm decision order differs")
                if row != cold:
                    selected.add(sid)
    if not selected.issubset(set(order)):
        raise ValueError("comparison references an unknown shot")
    return [sid for sid in order if sid in selected]


def auxiliary(saved, sig, valid, center, scale, active):
    """Independently score sealed next-window predictions on raw target masks."""
    values, available = np.asarray(sig["values"]), np.asarray(sig["available"])
    valid, active = np.asarray(valid, dtype=bool), np.asarray(active, dtype=bool)
    center, scale = np.asarray(center), np.asarray(scale)
    if (
        values.ndim != 2
        or values.shape[1] != 23
        or available.shape != values.shape
        or valid.shape != (len(values),)
        or active.shape != (23,)
        or center.shape != (23,)
        or scale.shape != (23,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("invalid signal, normalization, or mask dimensions")
    rows = []
    for j, clock in enumerate(saved["aux_clock"]):
        c = int(clock)
        if c < 256 or c + 25 > len(values):
            raise ValueError("incomplete next-window sample")
        allowed = available[c - 1] & np.isfinite(values[c - 1]) & active
        if not np.array_equal(allowed, saved["aux_available"][j]):
            raise ValueError("last-observation or supported-channel mask differs")
        expected_last = np.arcsinh(
            (values[c - 1, allowed] - center[allowed]) / scale[allowed]
        )
        if not np.allclose(
            saved["aux_last"][j][allowed], expected_last, rtol=2e-6, atol=2e-6
        ):
            raise ValueError("sealed hold baseline differs from the last observation")
        raw = values[c : c + 25].astype(np.float64)
        mask = (
            available[c : c + 25] & np.isfinite(raw) & allowed & valid[c : c + 25, None]
        )
        truth = np.zeros((25, 23), dtype=np.float64)
        normalized = np.arcsinh((raw - center) / scale)
        truth[mask] = normalized[mask]
        methods = {
            arm + part: saved[f"aux_{arm}{part}"][j]
            for arm in ADAPTIVE_ARMS
            for part in ("_before", "_after")
        }
        methods["hold"] = np.broadcast_to(saved["aux_last"][j], (25, 23))
        losses = {}
        for method, prediction in methods.items():
            if np.asarray(prediction).shape != truth.shape:
                raise ValueError("auxiliary prediction shape differs")
            error = np.zeros_like(truth)
            error[mask] = prediction[mask].astype(np.float64) - truth[mask]
            if not np.isfinite(error).all():
                raise ValueError("nonfinite observed auxiliary prediction")
            magnitude = np.abs(error)
            loss = np.minimum(magnitude, 1.0) ** 2 / 2 + np.maximum(
                magnitude - 1.0, 0.0
            )
            losses[method] = dict(
                sum=float(loss.sum()), channel_sums=loss.sum(0).tolist()
            )
        rows.append(
            dict(
                clock=c,
                elements=int(mask.sum()),
                channel_counts=mask.sum(0).tolist(),
                losses=losses,
            )
        )
    return rows


def summarize_auxiliary(rows, recorded):
    methods = {
        "hold",
        *(arm + part for arm in ADAPTIVE_ARMS for part in ("_before", "_after")),
    }
    if set(recorded) != methods:
        raise ValueError("auxiliary summary methods differ")
    result = {}
    for method in sorted(methods):
        values = []
        for row in rows:
            eligible = [sample for sample in row["samples"] if sample["elements"]]
            if eligible:
                values.append(
                    dict(
                        shot=row["shot"],
                        windows=len(eligible),
                        loss=float(
                            np.mean(
                                [
                                    sample["losses"][method]["sum"] / sample["elements"]
                                    for sample in eligible
                                ]
                            )
                        ),
                    )
                )
        declaration = recorded[method]
        if not values or [r["shot"] for r in values] != [
            r["shot"] for r in declaration["per_shot"]
        ]:
            raise ValueError("auxiliary shot set differs")
        v = np.array([r["loss"] for r in values])
        np.testing.assert_allclose(
            v, [r["loss"] for r in declaration["per_shot"]], rtol=2e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            [v.mean(), v.std(ddof=1)],
            [declaration["mean"], declaration["shot_sd"]],
            rtol=2e-7,
            atol=1e-9,
        )
        if (
            len(values) != declaration["shots"]
            or sum(r["windows"] for r in values) != declaration["windows"]
        ):
            raise ValueError("auxiliary sample count differs")
        result[method] = dict(
            shots=len(values),
            windows=sum(r["windows"] for r in values),
            mean=float(v.mean()),
            shot_sd=float(v.std(ddof=1)),
            per_shot=values,
        )
    hold = np.array([r["loss"] for r in result["hold"]["per_shot"]])
    for arm in ADAPTIVE_ARMS:
        before = np.array([r["loss"] for r in result[arm + "_before"]["per_shot"]])
        after = np.array([r["loss"] for r in result[arm + "_after"]["per_shot"]])
        result[arm + "_after"].update(
            beats_hold_shots=int((after < hold).sum()),
            improved_over_before_shots=int((after < before).sum()),
            after_minus_before_mean=float((after - before).mean()),
        )
    return result


def necessary_screen(metrics):
    """Apply only the preregistered true/true screen, not post-hoc selection."""
    baseline = metrics["cold"]["full360"]
    candidate = metrics["true_true"]["full360"]
    controls = (
        "cold",
        "prior_true",
        "true_reverse",
        "true_hold",
        "reverse_true",
        "hold_true",
    )
    return bool(
        candidate["tp"] >= baseline["tp"]
        and candidate["fpr"] <= 0.2
        and candidate["J"] > max(metrics[k]["full360"]["J"] for k in controls)
    )


def replay(root, protocol, selected, mapping):
    """Replay selected sealed shots; this is the only part requiring a GPU."""
    from .learning import ARMS as production_arms, MODES, predict

    check_mode_contract(MODES)
    if tuple(production_arms) != ARMS or not torch.cuda.is_available():
        raise RuntimeError("matching factorial arms and CUDA replay are required")
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, parent = load_parent(DEFAULT_PARENT)
    if float(parent["threshold"]) != protocol["threshold"]:
        raise ValueError("source threshold changed")
    net = MatrixAttention(base.cuda())
    frozen = snapshot(net)
    slow = torch.load(
        _worktree_path(protocol["reference"]) / "models/cold.pt",
        map_location="cuda",
        weights_only=True,
    )["slow"]
    slow_before = slow.clone()
    head = load_head(_worktree_path(protocol["readout"]) / "cold_readouts.npz", "cuda")
    head_before = {
        name: getattr(head, name).clone()
        for name in ("mean", "scale", "coef", "active")
    }
    rows = []

    def deny(*args, **kwargs):
        raise AssertionError("outcome accessed during replay prediction")

    for sid in selected:
        with patch.object(io23, "read_outcome", side_effect=deny):
            curves, info, aux = predict(net, signal(sid, mapping), slow, head)
        index = protocol["order"].index(sid)
        name = f"{index + 1:04d}_{sid}"
        seal = read(root / "shots", name + "_seal.json")
        if info != seal["updates"] or len(aux) != seal["aux_samples"]:
            raise ValueError("replayed feedback or sample count differs")
        with np.load(root / "shots" / f"{name}.npz", allow_pickle=False) as saved:
            error = max(float(np.abs(saved[arm] - curves[arm]).max()) for arm in ARMS)
            if error > 2e-6 or any(
                not np.array_equal(
                    saved[arm] >= protocol["threshold"],
                    curves[arm] >= protocol["threshold"],
                )
                for arm in ARMS
            ):
                raise ValueError("replayed risk trace or threshold crossings differ")
            aux_error = 0.0
            for key in aux[0] if aux else ():
                value = np.stack([row[key] for row in aux])
                old = saved["aux_" + key]
                np.testing.assert_allclose(value, old, rtol=2e-7, atol=1e-9)
                if value.dtype.kind == "f":
                    aux_error = max(aux_error, float(np.abs(value - old).max()))
        rows.append(dict(shot=sid, max_risk_error=error, max_auxiliary_error=aux_error))
        print(
            dict(replayed=sid, risk_error=error, auxiliary_error=aux_error), flush=True
        )
    if (
        any(not torch.equal(v, frozen[k]) for k, v in snapshot(net).items())
        or not torch.equal(slow, slow_before)
        or any(not torch.equal(getattr(head, k), v) for k, v in head_before.items())
    ):
        raise ValueError("replay changed classifier, cold operator, or readout prior")
    return dict(
        shots=rows,
        source_classifier_unchanged=True,
        cold_operator_unchanged=True,
        prior_unchanged=True,
        scope="all changed first-alarm records plus indices21/181/360; not all360 neural replay",
    )


def run(root):
    """Audit the explicitly completed factorial output, then write a new report."""
    started = time.monotonic()
    root = Path(root).resolve()
    expected_root = (io23.ROOT / "probe_outputs/local_factorial_east360_v1").resolve()
    if root != expected_root:
        raise ValueError("only the declared factorial output may receive this audit")
    report = root / "audit.json"
    if report.exists():
        raise FileExistsError(report)
    protocol, done = read(root, "protocol.json"), read(root, "complete.json")
    if (
        not done["complete"]
        or tuple(protocol["arms"]) != ARMS
        or len(protocol["order"]) != 360
        or len(set(protocol["order"])) != 360
    ):
        raise ValueError("completed declared360 factorial arms required")
    check_factorial_protocol(protocol)
    validate_test_ids("EAST", protocol["order"])
    if (
        protocol["source_sha256"] != PARENT_SHA256
        or sha(DEFAULT_PARENT) != PARENT_SHA256
    ):
        raise ValueError("qualified J-TEXT source changed")
    fixed = dict(
        warmup=20,
        step=STEP,
        history_rows=16,
        prior_precision=1.0,
        inter_updates_after_cold20=0,
        alarm_updates=0,
        reset_each_feedback=True,
        reset_each_shot=True,
    )
    if any(protocol.get(k) != value for k, value in fixed.items()):
        raise ValueError("fixed online protocol differs")
    for name in (
        "protocol",
        "per_shot",
        "receipts",
        "auxiliary_summary",
        "auxiliary_per_shot",
    ):
        if sha(root / (name + ".json")) != done[name + "_sha256"]:
            raise ValueError("changed factorial artifact " + name)
    plan = "LOCAL_READOUT_FACTORIAL_NEXT_PLAN.md"
    required_code = {
        "experiments/risk_aligned_online/local_factorial_20261029/" + name
        for name in ("run.py", "learning.py", "acceptance.py")
    }
    if not required_code.issubset(protocol["code_sha256"]):
        raise ValueError("factorial implementation provenance incomplete")
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for name, digest in dict(
            protocol["code_sha256"], **{plan: protocol["plan_sha256"]}
        ).items():
            if (
                hashlib.sha256(archive.read(name)).hexdigest() != digest
                or sha(io23.ROOT / name) != digest
            ):
                raise ValueError("factorial code or preregistered plan changed")
    previous = _worktree_path(protocol["previous"])
    if (
        previous.resolve()
        != (io23.ROOT / "probe_outputs/local_readout_east360_v1").resolve()
    ):
        raise ValueError("wrong previous experiment")
    readout, reference = (
        _worktree_path(protocol["readout"]),
        _worktree_path(protocol["reference"]),
    )
    old_protocol, old_done = (
        read(previous, "protocol.json"),
        read(previous, "complete.json"),
    )
    old_audit = read(previous, "audit.json")
    readout_done, readout_audit = (
        read(readout, "complete.json"),
        read(readout, "audit_with_fold_gap.json"),
    )
    digests = dict(
        previous_complete=sha(previous / "complete.json"),
        previous_audit=sha(previous / "audit.json"),
        readout=sha(readout / "cold_readouts.npz"),
        readout_audit=sha(readout / "audit_with_fold_gap.json"),
        readout_complete=sha(readout / "complete.json"),
    )
    check_prior_metadata(
        protocol, old_protocol, old_audit, readout_done, readout_audit, digests
    )
    if (
        not old_done["complete"]
        or sha(previous / "protocol.json") != old_done["protocol_sha256"]
        or sha(previous / "receipts.json") != old_done["receipts_sha256"]
    ):
        raise ValueError("previous protocol or sealed receipt list changed")
    for name, digest in protocol["reference_hashes"].items():
        if sha(reference / name) != digest:
            raise ValueError("cold reference changed")
    mapping, hashes = (
        read(reference, "input_map.json"),
        read(reference, "input_hashes.json"),
    )
    if mapping["calibration_ids"] != protocol["order"][:20]:
        raise ValueError("cold calibration order differs")
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = base.center.numpy(), base.scale.numpy()
    active = load_head(readout / "cold_readouts.npz", "cpu").active.numpy()
    declared, receipts = read(root, "per_shot.json"), read(root, "receipts.json")
    previous_receipts = read(previous, "receipts.json")
    saved_aux = read(root, "auxiliary_per_shot.json")
    recorded_aux = {row["shot"]: row for row in saved_aux}
    if (
        set(declared) != set(ARMS)
        or any(len(declared[arm]) != 360 for arm in ARMS)
        or len(receipts) != 360
        or len(previous_receipts) != 360
        or len(recorded_aux) != len(saved_aux)
    ):
        raise ValueError("decision, receipt, or auxiliary shot cardinality differs")
    actual = {arm: [] for arm in ARMS}
    auxiliary_rows = []
    totals = {
        arm: dict(
            opportunities=0,
            updates=0,
            decreased=0,
            observed=0,
            max_norm=0.0,
            trials=0,
            loss_before_sum=0.0,
            loss_after_sum=0.0,
            accepted_norm_sum=0.0,
        )
        for arm in ADAPTIVE_ARMS
    }
    reasons = {arm: Counter() for arm in ADAPTIVE_ARMS}
    factors = {arm: Counter() for arm in ADAPTIVE_ARMS}
    ledger_count = ledger_with_trials = 0
    for index, sid in enumerate(protocol["order"]):
        if sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("raw EAST input changed")
        name = f"{index + 1:04d}_{sid}"
        seal = read(root / "shots", name + "_seal.json")
        oldseal = read(previous / "shots", name + "_seal.json")
        if (
            seal != receipts[index]
            or seal["shot"] != sid
            or seal["index"] != index + 1
            or seal["sha256"] != sha(root / "shots" / f"{name}.npz")
            or seal["outcome_used"]
            or seal["adaptation_active"] != (index >= 20)
        ):
            raise ValueError("invalid factorial prediction seal")
        if (
            oldseal != previous_receipts[index]
            or oldseal["shot"] != sid
            or oldseal["index"] != index + 1
            or oldseal["sha256"] != sha(previous / "shots" / f"{name}.npz")
        ):
            raise ValueError("previous prediction seal changed")
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(root / "shots" / f"{name}.npz", allow_pickle=False) as z,
            np.load(previous / "shots" / f"{name}.npz", allow_pickle=False) as old,
        ):
            if (
                not np.array_equal(z["times"], old["times"])
                or np.abs(z["cold"] - old["cold"]).max() > 2e-6
                or not np.array_equal(
                    z["cold"] >= protocol["threshold"],
                    old["cold"] >= protocol["threshold"],
                )
            ):
                raise ValueError("cold prediction differs from prior sealed run")
            for arm in ARMS:
                row = decision(
                    sid, z["times"], z[arm], flag, down, protocol["threshold"]
                )
                if row != declared[arm][index]:
                    raise ValueError("per-shot first-alarm decision differs")
                actual[arm].append(row)
                end = len(z["times"]) if index < 20 else 256
                if np.abs(
                    z[arm][:end] - z["cold"][:end]
                ).max() > 2e-6 or not np.array_equal(
                    z[arm][:end] >= protocol["threshold"],
                    z["cold"][:end] >= protocol["threshold"],
                ):
                    raise ValueError("cold20 or pre-feedback prediction changed")
                if index < 20 and not np.array_equal(z[arm], z["cold"]):
                    raise ValueError("cold20 arms must be identical")
            if index < 20:
                if seal["updates"] or seal["aux_samples"]:
                    raise ValueError("cold20 adaptation unexpectedly active")
                continue
            sig = signal(sid, mapping)
            if len(sig["values"]) != len(z["times"]):
                raise ValueError("raw and sealed signal lengths differ")
            shot_totals = check_feedback(seal["updates"], sig, active)
            for arm, row in shot_totals.items():
                for key, value in row.items():
                    if key == "max_norm":
                        totals[arm][key] = max(totals[arm][key], value)
                    else:
                        totals[arm][key] += value
            for record in seal["updates"]["feedback"]:
                for arm, item in record["arms"].items():
                    reasons[arm][item["reason"]] += 1
                    if item["accepted_norm"] > 0:
                        factors[arm][str(item["factor"])] += 1
                    ledger_count += 1
                    ledger_with_trials += int("trial_losses" in item)
            valid = (
                np.round((down - z["times"]) * 1000, 9) >= -1e-6
                if flag
                else np.ones(len(z["times"]), bool)
            )
            eligible = np.arange(256, len(z["times"]) - 24, 128)
            clocks = (
                eligible[
                    np.linspace(0, len(eligible) - 1, min(4, len(eligible))).astype(int)
                ]
                if len(eligible)
                else []
            )
            if not np.array_equal(
                z["aux_clock"] if len(clocks) else [], clocks
            ) or seal["aux_samples"] != len(clocks):
                raise ValueError("auxiliary sampling clocks differ")
            if len(clocks):
                rows = auxiliary(z, sig, valid, center, scale, active)
                old_rows = recorded_aux[sid]["samples"]
                if len(rows) != len(old_rows):
                    raise ValueError("auxiliary sample count differs")
                for row, recorded in zip(rows, old_rows):
                    if any(
                        row[k] != recorded[k]
                        for k in ("clock", "elements", "channel_counts")
                    ) or set(row["losses"]) != set(recorded["losses"]):
                        raise ValueError("auxiliary target masks or methods differ")
                    for method in row["losses"]:
                        for key in ("sum", "channel_sums"):
                            np.testing.assert_allclose(
                                row["losses"][method][key],
                                recorded["losses"][method][key],
                                rtol=2e-7,
                                atol=1e-9,
                            )
                auxiliary_rows.append(dict(shot=sid, samples=rows))
    if [row["shot"] for row in auxiliary_rows] != [row["shot"] for row in saved_aux]:
        raise ValueError("auxiliary shot list or order differs")
    metrics = {
        arm: {
            label: count(sub)
            for label, sub in (
                ("full360", rows),
                ("cold20", rows[:20]),
                ("suffix340", rows[20:]),
            )
        }
        for arm, rows in actual.items()
    }
    for arm, sections in metrics.items():
        for label, values in sections.items():
            for key, value in values.items():
                reported = done["results"][arm][label][key]
                if value is None or reported is None:
                    if value is not None or reported is not None:
                        raise ValueError("aggregate metric differs")
                elif not _near(value, reported, rtol=0, atol=1e-12):
                    raise ValueError("aggregate metric differs")
    comparisons = {arm: paired(actual["cold"], actual[arm]) for arm in ADAPTIVE_ARMS}
    if (
        comparisons != done["comparisons"]
        or {arm: value["updates"] for arm, value in totals.items()}
        != done["update_counts"]
        or necessary_screen(metrics) != done["necessary_intra_screen"]
    ):
        raise ValueError("paired decisions, counts, or candidate screen differ")
    aux_summary = summarize_auxiliary(
        auxiliary_rows, read(root, "auxiliary_summary.json")
    )
    selected = choose_replay_shots(protocol["order"], comparisons, actual)
    replayed = replay(root, protocol, selected, mapping)
    result = dict(
        complete=True,
        complete_sha256=sha(root / "complete.json"),
        code_sha256=sha(Path(__file__)),
        seconds=time.monotonic() - started,
        metrics=metrics,
        comparisons=comparisons,
        necessary_intra_screen=necessary_screen(metrics),
        feedback_totals=totals,
        acceptance_reasons={k: dict(v) for k, v in reasons.items()},
        accepted_factors={k: dict(v) for k, v in factors.items()},
        receipt_count=ledger_count,
        first_passing_checked_count=ledger_with_trials,
        all_first_passing_verified=ledger_with_trials == ledger_count,
        auxiliary=aux_summary,
        replay=replayed,
        previous_audit_sha256=digests["previous_audit"],
        prior_verification="previous independent NumPy fit, bound by its pinned audit and artifact hash chain; not refit here",
        independent_confirmation=False,
        scope="all360 sealed first-alarm decisions and raw masks, factorial acceptance receipts, and changed-shot neural replay",
    )
    # Exclusive creation prevents overwriting any report written in the meantime.
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        dict(
            metrics={k: v["full360"] for k, v in metrics.items()},
            comparisons=comparisons,
            replay_shots=selected,
            all_first_passing_verified=result["all_first_passing_verified"],
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
