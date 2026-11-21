"""POSTHOC_ORACLE_POINT_DIAGNOSTIC: bounded offline endpoint plasticity.

This diagnostic deliberately uses retrospective true labels for endpoint
selection AND optimization. Never use it for online policy, label queries, or
shot selection. A best VISITED point is not a global capacity bound, and a
point crossing is not a recovered whole shot or a real online metric.
Only an external, disposable fast128 delta changes; no model is saved.
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..fast_update import normalized_step
from ..simple_online import _hash
from .embedding_model import EmbeddingModel

DIAGNOSTIC = "POSTHOC_ORACLE_POINT_DIAGNOSTIC"
STEP_SIZE = .10
L2_RADIUS = 1.
MAX_STEPS = 32
DEFAULT_SHOTS = (54017, 54818, 54924, 54094, 56602, 54244, 54977)


def _threshold(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0. <= value <= 1.):
        raise ValueError("threshold must be a finite probability")
    return float(value)


def _steps(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= MAX_STEPS:
        raise ValueError("steps must be an integer in [0,32]")
    return int(value)


def _bytes(tensor):
    # torch.equal on floats does not distinguish +0/-0; compare actual bits.
    return tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().clone()


def _snapshot(model):
    tensors = dict([*model.named_parameters(), *model.named_buffers()])
    return {name: (tensor, tensor.data_ptr(), tensor._version, _bytes(tensor))
            for name, tensor in tensors.items()}


def _check_immutable(model, before):
    current = dict([*model.named_parameters(), *model.named_buffers()])
    if set(current) != set(before):
        raise AssertionError("model immutability: registered tensor names changed")
    for name, (original, pointer, version, bits) in before.items():
        tensor = current[name]
        if (tensor is not original or tensor.data_ptr() != pointer or tensor._version != version
                or not torch.equal(_bytes(tensor), bits)):
            raise AssertionError(f"model immutability: registered tensor changed: {name}")
    if (any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(module.training for module in model.modules())):
        raise AssertionError("model immutability: frozen/eval/gradient state changed")
    return dict(risk_parameters_bitwise_unchanged=True,
                source_parameters_bitwise_unchanged=True,
                wrapper_parameters_and_buffers_bitwise_unchanged=True,
                registration_and_versions_unchanged=True,
                all_registered_parameters_frozen=True,
                checked_tensors=len(before), comparison="exact uint8 bit patterns, identity, version")


def _validate_point(model, values, available, label):
    if not isinstance(model, EmbeddingModel) or model.cfg.fast_rule != "normalized":
        raise ValueError("a source-normalized EmbeddingModel wrapper is required")
    if (any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(module.training for module in model.modules())):
        raise ValueError("all registered wrapper/source parameters must be frozen, gradient-free and eval")
    if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
        raise ValueError("retrospective true label must be integer 0 or 1")
    if (not isinstance(values, torch.Tensor) or values.shape != (1, 128, 23)
            or values.dtype != model.center.dtype or values.device != model.center.device):
        raise ValueError("one [1,128,23] causal window with source dtype/device is required")
    if (not isinstance(available, torch.Tensor) or available.shape != values.shape
            or available.dtype != torch.bool or available.device != values.device
            or not available.any() or not torch.isfinite(values[available]).all()):
        raise ValueError("matching boolean availability and finite nonempty observed inputs required")
    if torch.is_inference_mode_enabled():
        raise ValueError("autograd fast path required; do not call from inference_mode")


def _logit(model, values, available, fast):
    result = model(values, available, fast=fast)
    logit = result.get("logit") if isinstance(result, dict) else None
    if (not isinstance(logit, torch.Tensor) or logit.shape != (1,)
            or logit.device != values.device or logit.dtype != values.dtype):
        raise ValueError("wrapper must return one source-dtype/device risk logit")
    if not torch.isfinite(logit).all():
        raise FloatingPointError("nonfinite risk logit")
    return logit[0]


def probe_point(model, values, available, label, threshold, *, steps=MAX_STEPS):
    """Optimize ONLY a fresh external fast128 leaf, then discard it.

    Descent on -logit for label 1, +logit for label 0 uses the exact existing
    source normalized_step, step .10, total delta L2 radius 1. Best means the
    earliest largest label-directed logit among steps 0..steps, NOT a maximum
    over all feasible fast states. The wrapper adds its NONZERO source origin.
    """
    _validate_point(model, values, available, label)
    threshold, steps = _threshold(threshold), _steps(steps)
    initial = model.initial_fast(1).detach().clone()
    if (initial.shape != (1, 128) or initial.dtype != values.dtype
            or initial.device != values.device or torch.count_nonzero(initial)):
        raise ValueError("wrapper must start from zero temporary delta, not zero source origin")
    before = _snapshot(model)
    trace, best_index, stop_reason = [], 0, "step_budget_exhausted"
    direction = 1 if label else -1
    try:
        # Match the sealed runner's differentiable fast dispatch even at zero.
        with torch.enable_grad():
            fast = initial.clone().requires_grad_(True)
            for step in range(steps + 1):
                logit = _logit(model, values.detach(), available, fast)
                numeric = float(logit.detach())
                probability = float(logit.detach().sigmoid())
                record = dict(step=step, logit=numeric, probability=probability,
                              fast_norm=float(fast.detach().double().norm()),
                              logit_change=numeric - (trace[0]["logit"] if trace else numeric),
                              threshold_reached=(probability >= threshold if label else probability < threshold))
                trace.append(record)
                if direction * numeric > direction * trace[best_index]["logit"]:
                    best_index = step
                if step == steps:
                    break
                if not logit.requires_grad:
                    stop_reason = "fast_independent_logit"
                    break
                gradient, = torch.autograd.grad(-direction * logit, fast, allow_unused=True)
                if gradient is None:
                    stop_reason = "fast_independent_logit"
                    break
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite risk gradient")
                record["gradient_norm"] = float(gradient.detach().double().norm())
                if not torch.count_nonzero(gradient):
                    stop_reason = "zero_gradient"
                    break
                fast = normalized_step(fast.detach(), initial, gradient.detach(),
                                       step=STEP_SIZE, radius=L2_RADIUS).detach().requires_grad_(True)
                if not torch.isfinite(fast).all() or float(fast.detach().double().norm()) > 1. + 1e-6:
                    raise FloatingPointError("invalid projected temporary delta")
    finally:
        immutable = _check_immutable(model, before)
    first, best, final = trace[0], trace[best_index], trace[-1]
    return dict(diagnostic=DIAGNOSTIC, label=int(label), threshold=threshold,
                retrospective_true_label_used=True, step_size=STEP_SIZE, l2_radius=L2_RADIUS,
                update_rule="source fast_update.normalized_step; epsilon=1e-12",
                objective="raise risk logit" if label else "lower risk logit",
                requested_steps=steps, completed_steps=final["step"], stop_reason=stop_reason,
                initial_probability=first["probability"], best_probability=best["probability"],
                final_probability=final["probability"], initial_logit=first["logit"],
                best_logit=best["logit"], final_logit=final["logit"],
                best_logit_change=best["logit_change"], final_logit_change=final["logit_change"],
                best_step=best_index, best_fast_norm=best["fast_norm"], final_fast_norm=final["fast_norm"],
                initial_threshold_reached=first["threshold_reached"],
                best_threshold_reached=best["threshold_reached"], final_threshold_reached=final["threshold_reached"],
                best_definition="earliest best label-directed logit among visited steps including zero",
                immutability=immutable, trace=trace,
                interpretation="local ORACLE endpoint diagnostic; no global bound or whole-shot recovery claim")


def _trace_arrays(trace):
    times, scores = np.asarray(trace["times"], dtype=float), np.asarray(trace["scores"], dtype=float)
    if (times.ndim != 1 or not len(times) or scores.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(scores).all()
            or (np.diff(times) <= 0).any() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("finite aligned causal times and sealed probabilities required")
    return times, scores


def select_endpoint(trace, outcome):
    """POSTHOC selection within a caller-fixed shot, never an online gate."""
    times, scores = _trace_arrays(trace)
    if (outcome["shot"] != trace["shot"] or outcome["device"] != "EAST"
            or float(outcome["horizon"]) != 125.
            or not np.array_equal(np.asarray(outcome["times"]), times)):
        raise ValueError("matching revealed EAST outcome with fixed horizon125 required")
    valid, labels = np.asarray(outcome["valid"]), np.asarray(outcome["y"])
    if (valid.shape != times.shape or valid.dtype != bool or labels.shape != times.shape
            or not np.isfinite(labels).all() or not np.isin(labels, [0., 1.]).all()):
        raise ValueError("aligned boolean validity and binary endpoint labels required")
    label = int(bool(outcome["is_disrupt"]))
    lead = None
    eligible = valid.copy()
    if label:
        if outcome["down"] is None or not math.isfinite(float(outcome["down"])):
            raise ValueError("finite disruptive DownTime required")
        # Match the existing nine-decimal strict lead contract at 30/125 ms.
        lead = np.round((float(outcome["down"]) - times) * 1000., 9)
        eligible &= (lead > 30.) & (lead <= 125.) & (labels == 1.)
    else:
        eligible &= labels == 0.
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError(f"no eligible retrospective endpoint for shot {trace['shot']}")
    end = int(candidates[np.argmax(scores[candidates])])
    return dict(shot=int(trace["shot"]), end_index=end, time=float(times[end]), label=label,
                lead_ms=float(lead[end]) if label else None, sealed_probability=float(scores[end]),
                endpoint_selection=("POSTHOC static maximum in 30 < lead_ms <= 125"
                                    if label else "POSTHOC static maximum over valid safe endpoints"))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_completed_static(run, shot_ids):
    """Check static completion, then read ONLY the caller's fixed shot seals.

    The existing runner has no outcomes.json. Its static summaries/audits
    record revealed outcomes; DownTime and endpoint labels are joined later
    via the existing attach_outcome, strictly after these seal checks.
    """
    if (not shot_ids or len(shot_ids) > 16 or len(set(shot_ids)) != len(shot_ids)
            or any(isinstance(sid, bool) or not isinstance(sid, Integral) or sid <= 0 for sid in shot_ids)):
        raise ValueError("1..16 distinct, positive, caller-fixed shot IDs required")
    run = Path(run)
    paths = [run / "protocol.json", run / "order.json", run / "summary.json", run / "static/summary.json"]
    hashes = {str(path): _hash(path) for path in paths}
    protocol, order, root_summary, summary = [_read_json(path) for path in paths]
    threshold = _threshold(protocol["threshold"])
    if (protocol.get("persistent_site") != "embedding" or not protocol.get("source_offline_only")
            or not protocol.get("source_gate", {}).get("qualified")
            or protocol.get("final_score_correction_head") is not False
            or protocol.get("alarm_tuning") is not False or protocol.get("east_label_ms") != 125.):
        raise ValueError("qualified frozen-source EAST embedding run without score correction required")
    if (not isinstance(order, list) or not order
            or any(type(row.get("shot")) is not int or row["shot"] <= 0 for row in order)
            or len({row["shot"] for row in order}) != len(order)):
        raise ValueError("distinct positive shots in the sealed run order required")
    limit = protocol["args"]["limit"]
    if type(limit) is not int or not 0 <= limit <= len(order):
        raise ValueError("invalid static run limit")
    expected = [row["shot"] for row in (order[:limit] if limit else order)]
    predictions = summary["metrics"]["predictions"]
    if (root_summary.get("static") != summary or summary.get("arm") != "static"
            or summary.get("observed_count") != len(expected)
            or summary.get("roster_count") != len(order)
            or [row["shot"] for row in predictions] != expected
            or _threshold(summary["metrics"]["threshold"]) != threshold):
        raise ValueError("static arm is not completed with the declared order and threshold")
    arm = run / "static"
    names = {f"{index:04d}_{sid}_seal.json" for index, sid in enumerate(expected, 1)}
    if {path.name for path in arm.glob("*_seal.json")} != names:
        raise ValueError("static completion seal count/order mismatch")
    positions = {sid: index for index, sid in enumerate(expected, 1)}
    records = []
    for sid in shot_ids:
        if sid not in positions:
            raise ValueError(f"caller-fixed shot {sid} absent from completed static run")
        index = positions[sid]
        prediction = predictions[index - 1]
        wrong = not prediction["timely"] if prediction["is_disrupt"] else prediction["alarm"]
        if not wrong:
            raise ValueError(f"caller-fixed shot {sid} is not a static error; no automatic replacement")
        stem = f"{index:04d}_{sid}"
        receipt_path, audit_path = arm / f"{stem}_seal.json", arm / f"{stem}_audit.json"
        hashes.update({str(path): _hash(path) for path in (receipt_path, audit_path)})
        receipt = _read_json(receipt_path)
        npz_path = (arm / receipt["npz"]).resolve()
        if (npz_path.parent != arm.resolve() or not npz_path.is_file()
                or receipt.get("shot") != sid or receipt.get("arm") != "static"
                or _threshold(receipt["threshold"]) != threshold):
            raise ValueError(f"static seal identity/path/threshold mismatch for {sid}")
        hashes[str(npz_path)] = _hash(npz_path)
        if hashes[str(npz_path)] != receipt["sha256"]:
            raise ValueError(f"static seal SHA mismatch for {sid}")
        # No outcome access before the receipt's SHA check.
        with np.load(npz_path, allow_pickle=False) as archive:
            if np.asarray(archive["shot"]).shape != () or int(archive["shot"]) != sid:
                raise ValueError(f"sealed NPZ shot identity mismatch for {sid}")
            trace = dict(shot=int(sid), index=index, times=np.asarray(archive["times"]).copy(),
                         scores=np.asarray(archive["scores"]).copy(), receipt=receipt)
        _trace_arrays(trace)
        audit = _read_json(audit_path)
        if (audit.get("shot") != sid or audit.get("intra") is not False
                or audit.get("persistent") is not None or audit.get("prediction") != prediction
                or audit.get("seal_sha256") != receipt["sha256"]):
            raise ValueError(f"audit is not matching zero-delta static prediction for {sid}")
        records.append(dict(trace=trace, prediction=prediction))
    return dict(protocol=protocol, threshold=threshold, records=records, input_sha256=hashes)


def write_report(output, report):
    """Write only the explicitly supplied NEW JSON file, never a checkpoint."""
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(output).open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed run ROOT containing static/ and protocol.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="same qualified J-TEXT source checkpoint")
    parser.add_argument("--prior", type=Path, required=True, help="exact same finite-step source forecast prior as the sealed run")
    parser.add_argument("--output", type=Path, required=True, help="new, user-selected JSON file; existing files are refused")
    parser.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS), help="fixed IDs, never automatically mined/replaced")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="0..32; step .10 and total L2 radius1 are fixed")
    args = parser.parse_args(argv)
    _steps(args.steps)
    if args.output.exists():
        parser.error("output already exists; no artifact may be overwritten")
    sealed = load_completed_static(args.run, args.shots)
    protocol, threshold = sealed["protocol"], sealed["threshold"]
    hashes = dict(sealed["input_sha256"])
    for path, expected in ((args.checkpoint, protocol["source_sha256"]), (args.prior, protocol["prior_sha256"])):
        actual = _hash(path)
        if actual != expected:
            raise ValueError(f"specified source/prior SHA differs from sealed run: {path}")
        hashes[str(path)] = actual
    from .run import load_parent

    model, state, gate, _ = load_parent(args.checkpoint, args.device,
                                      persistent_site="embedding", prior=args.prior)
    if (not gate["qualified"] or gate["checkpoint_sha256"] != protocol["source_sha256"]
            or state["model_config"] != protocol["model_config"] or _threshold(state["threshold"]) != threshold
            or any(torch.count_nonzero(p) for p in model.persistent_parameters())):
        raise ValueError("source/config/threshold or zero persistent embedding differs from sealed static run")
    model.eval().requires_grad_(False)
    before = _snapshot(model)
    points = []
    for record in sealed["records"]:
        trace, prediction = record["trace"], record["prediction"]
        shot = apply_input_contract(simple_data.read_shot("EAST", trace["shot"]))
        if not np.array_equal(shot["times"], trace["times"]):
            raise ValueError(f"reconstructed signal timestamps differ from seal for {trace['shot']}")
        outcome = simple_data.attach_outcome(shot, "EAST", {})
        if outcome["is_disrupt"] != prediction["is_disrupt"]:
            raise ValueError(f"revealed outcome differs from static summary for {trace['shot']}")
        point = select_endpoint(trace, outcome)
        end = point["end_index"]
        # Exactly the existing window/padding contract, with future rows hidden.
        prefix = {key: shot[key][:end + 1] for key in ("values", "available")}
        x, available = simple_data.window(prefix, end, model.cfg.window)
        x = torch.as_tensor(x, device=model.center.device, dtype=model.center.dtype)[None]
        available = torch.as_tensor(available, device=x.device, dtype=torch.bool)[None]
        with torch.enable_grad():
            zero = model.initial_fast(1).detach().requires_grad_(True)
            baseline = float(_logit(model, x, available, zero).detach().sigmoid())
        error = abs(baseline - point["sealed_probability"])
        if error > 1e-5:
            raise ValueError(f"static zero-delta mismatch for {trace['shot']}: abs error {error:.9g} > 1e-5")
        result = probe_point(model, x, available, point["label"], threshold, steps=args.steps)
        if abs(result["initial_probability"] - baseline) > 1e-7:
            raise ValueError("solver zero-delta path differs from reconstruction baseline")
        result.update(point, static_zero_delta_matches_seal=True,
                      static_zero_delta_abs_error=error, static_zero_delta_atol=1e-5)
        points.append(result)
    immutable = _check_immutable(model, before)
    if any(_hash(Path(path)) != digest for path, digest in hashes.items()):
        raise AssertionError("source/prior/sealed input files changed during diagnostic")
    report = dict(diagnostic=DIAGNOSTIC, run=str(args.run.resolve()),
                  checkpoint=str(args.checkpoint.resolve()), prior=str(args.prior.resolve()), device=args.device,
                  fixed_shots=list(args.shots), automatic_shot_selection=False,
                  label_access="retrospective true EAST outcomes AFTER static seal verification",
                  forbidden_uses=["online policy", "label querying", "shot selection"],
                  online_metrics_reported=False, threshold=threshold,
                  fast_delta_discarded=True, persistent_parameters_updated=False, checkpoints_written=False,
                  source_prior_and_sealed_files_unchanged=True, input_sha256=hashes,
                  immutability=immutable, points=points,
                  interpretation="best visited local ORACLE points; not global bounds, whole-shot recovery or online performance")
    write_report(args.output, report)
    print(json.dumps(dict(diagnostic=DIAGNOSTIC, points=len(points), output=str(args.output))), flush=True)
    return report


if __name__ == "__main__":
    main()
