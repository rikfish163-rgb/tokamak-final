"""Warm20-only, outcome-free reread: per-channel LoRA gradient reachability."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.memory import (
    direct_target,
)
from experiments.risk_aligned_online.direct_fast_memory_20261104.model import levels
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import (
    window_batch,
)
from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
)
from joint_lora_gradient_probe_20261104 import (
    centered_risk_loss,
    control_losses,
    gradient_metrics,
    sha,
    state_digest,
)


ROOT = io23.ROOT
PRIOR = ROOT / "probe_outputs/joint_lora_gradient_probe_20261104_v1"
RUN = ROOT / "probe_outputs/joint_lora_paired360_20261104_v1"
MAP = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_map.json"
OUTPUT = ROOT / "probe_outputs/joint_lora_gradient_span_20261104_v1"
PRIOR_COMPLETED_SHA = "5d9ef8ff763a6f501ebe513806e2c37df402290c89164407484061b8cabc36b5"
CHANNELS = [name for i, name in enumerate(io23.RAW23) if i not in (2, 10)]


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def channel_losses(prediction, target, observed):
    """21 additive terms, each divided by total observed 8x21 coordinates."""
    if (
        prediction.shape != (8, 21)
        or target.shape != prediction.shape
        or observed.shape != prediction.shape
        or observed.dtype != torch.bool
    ):
        raise ValueError(
            "matching 8-by-21 prediction, target and boolean mask required"
        )
    if not observed.any():
        raise ValueError("target has no support; never silently drop the case")
    return (prediction - target).square().masked_fill(~observed, 0.0).sum(
        dim=0
    ) / observed.sum()


def projection_metrics(matrix, risk):
    """Unrestricted cosine from float32-origin G, with two rank tolerances."""
    if (
        matrix.ndim != 2
        or risk.ndim != 1
        or matrix.shape[0] != risk.numel()
        or not torch.isfinite(matrix).all()
        or not torch.isfinite(risk).all()
    ):
        raise ValueError("finite column matrix and matching risk vector required")
    g, r = matrix.detach().double(), risk.detach().double()
    u, singular, _ = torch.linalg.svd(g, full_matrices=False)
    max_s = float(singular[0]) if len(singular) else 0.0
    norm = float(torch.linalg.vector_norm(r))
    values = {}
    for name, epsilon in (
        ("64", np.finfo(np.float64).eps),
        ("32", np.finfo(np.float32).eps),
    ):
        tolerance = float(max(g.shape) * float(epsilon) * max_s)
        rank = int((singular > tolerance).sum().item())
        ratio = (
            float(torch.linalg.vector_norm(u[:, :rank].T @ r).item() / norm)
            if rank and norm
            else None
        )
        if ratio is not None:
            ratio = min(1.0, max(0.0, ratio))
        values.update(
            {
                f"rank{name}": rank,
                f"rank_tolerance{name}": tolerance,
                f"projection_ratio{name}": ratio,
                f"projection_energy_fraction{name}": ratio * ratio
                if ratio is not None
                else None,
            }
        )
    values["singular_values"] = singular.tolist()
    return values


def box_oracle(scores):
    """Exact dot maximizer for 21 weights in [.5,1.5] with sum 21."""
    if scores.shape != (21,) or not torch.isfinite(scores).all():
        raise ValueError("21 finite channel inner products required")
    order = torch.argsort(scores, stable=True)
    result = torch.ones_like(scores)
    result[order[:10]] = 0.5
    result[order[-10:]] = 1.5
    return result


def adam_first_delta(theta, gradient):
    """Simulate one zero-moment Adam step on a detached copy, not a fit.

    float32, lr=1e-4, betas=(.9,.999), eps=1e-8, clip norm 1.  Reproduce
    rounding via float32 in-place addcdiv on a copy. Radius .5 must be inactive.
    """
    if (
        theta.shape != gradient.shape
        or theta.dtype != torch.float32
        or gradient.dtype != torch.float32
        or not torch.isfinite(theta).all()
        or not torch.isfinite(gradient).all()
    ):
        raise ValueError(
            "matching finite float32 parameter and gradient vectors required"
        )
    norm = torch.linalg.vector_norm(gradient)
    clipped = gradient * torch.clamp(1.0 / (norm + 1e-6), max=1.0)
    first = torch.zeros_like(clipped).lerp_(clipped, 1 - 0.9)
    second = torch.zeros_like(clipped).addcmul_(clipped, clipped, value=1 - 0.999)
    denom = second.sqrt() / math.sqrt(1 - 0.999) + 1e-8
    updated = theta.detach().clone().addcdiv_(first, denom, value=-1e-4 / (1 - 0.9))
    delta = updated - theta.detach()
    if torch.linalg.vector_norm(delta.double()).item() > 0.5:
        raise ValueError(
            "the .5 shot-local radius became active; do not report this simulation"
        )
    return delta


def _grad(loss, parameters, *, retain_graph=False):
    result = torch.cat(
        [
            v.detach().reshape(-1)
            for v in torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
        ]
    )
    if result.shape != (512,) or not torch.isfinite(result).all():
        raise ValueError("finite 512-coordinate LoRA gradient required")
    return result


def evaluate_case(model, shot, case, parameters, threshold):
    origin, future = case["origin"], case["future_risk_point"]
    x, a = window_batch(shot, [origin], device="cpu")
    future_values = torch.as_tensor(
        shot["values"][origin + 1 : origin + 9][None], device="cpu"
    )
    future_mask = torch.as_tensor(
        shot["available"][origin + 1 : origin + 9][None], device="cpu"
    )
    with torch.no_grad():
        target, observed = direct_target(
            levels(model.clone, x, a),
            a,
            levels(model.clone, future_values, future_mask),
            future_mask,
        )
    supported = int(observed.sum())
    if supported != case["supported"] or not supported:
        raise ValueError("existing case support changed; no case may be dropped")
    x_risk, a_risk = window_batch(shot, [future], device="cpu")
    risk_loss = centered_risk_loss(
        model(x_risk, a_risk)["logit"], case["label"], threshold
    )
    risk_gradient = _grad(risk_loss, parameters)
    prediction = model(x, a)["diagnostic"]
    unweighted_loss = control_losses(prediction, target, observed)["real"]
    uniform_gradient = _grad(unweighted_loss, parameters, retain_graph=True)
    terms = channel_losses(
        prediction.reshape(8, 21), target.reshape(8, 21), observed.reshape(8, 21)
    )
    matrix = torch.stack(
        [_grad(terms[i], parameters, retain_graph=i < 20) for i in range(21)], dim=1
    )
    summed = matrix.double().sum(dim=1)
    absolute_error = float((summed - uniform_gradient.double()).abs().max())
    relative_error = float(
        torch.linalg.vector_norm(summed - uniform_gradient.double())
        / torch.linalg.vector_norm(uniform_gradient.double()).clamp_min(1e-30)
    )
    if not torch.allclose(summed, uniform_gradient.double(), atol=5e-8, rtol=2e-4):
        raise ValueError(
            "sum of channel gradients does not reproduce unweighted gradient"
        )
    original = case["controls"]["real"]
    now = gradient_metrics(uniform_gradient, risk_gradient)
    for name in ("diag_norm", "risk_norm", "inner_product", "cosine"):
        if not math.isclose(now[name], original[name], rel_tol=2e-4, abs_tol=2e-7):
            raise ValueError(f"prior unweighted gradient mismatch: {name}")
    if not math.isclose(
        float(risk_loss.detach()), case["risk_loss"], rel_tol=2e-4, abs_tol=2e-7
    ):
        raise ValueError("prior risk loss mismatch")
    gd, gr = matrix.double(), risk_gradient.double()
    coefficients = gd.T @ gr
    weights = box_oracle(coefficients)
    uniform_dot = float(coefficients.sum())
    oracle_dot = float(coefficients @ weights)
    if oracle_dot + 1e-12 < uniform_dot:
        raise AssertionError(
            "box optimum cannot be worse than feasible uniform weights"
        )
    theta = torch.cat([p.detach().reshape(-1) for p in parameters])
    oracle_gradient = (gd @ weights).float()
    delta_uniform = adam_first_delta(theta, uniform_gradient)
    delta_oracle = adam_first_delta(theta, oracle_gradient)
    row = {
        key: case[key]
        for key in (
            "shot",
            "disruptive",
            "origin",
            "arrival",
            "future_risk_point",
            "actuallead",
            "label",
            "group",
            "supported",
        )
    }
    row.update(
        unweighted_norm=now["diag_norm"],
        risk_norm=now["risk_norm"],
        unweighted_cosine=now["cosine"],
        unweighted_dot_original=now["inner_product"],
        sum_gradient_max_abs_error=absolute_error,
        sum_gradient_relative_error=relative_error,
        uniform_dot=uniform_dot,
        oracle_dot=oracle_dot,
        oracle_improvement=oracle_dot - uniform_dot,
        channel_inner_products=coefficients.tolist(),
        oracle_weights=weights.tolist(),
        adam_uniform_risk_dot_delta=float(gr @ delta_uniform.double()),
        adam_oracle_risk_dot_delta=float(gr @ delta_oracle.double()),
        adam_uniform_delta_norm=float(torch.linalg.vector_norm(delta_uniform.double())),
        adam_oracle_delta_norm=float(torch.linalg.vector_norm(delta_oracle.double())),
        **projection_metrics(matrix, risk_gradient),
    )
    return row, matrix, risk_gradient, uniform_gradient


def summarize(rows):
    result = {}
    for group in ("safe", "earlier", "positive"):
        subset = [row for row in rows if row["group"] == group]
        ratios64 = [
            row["projection_ratio64"]
            for row in subset
            if row["projection_ratio64"] is not None
        ]
        ratios32 = [
            row["projection_ratio32"]
            for row in subset
            if row["projection_ratio32"] is not None
        ]
        result[group] = dict(
            cases=len(subset),
            shots=len({row["shot"] for row in subset}),
            rank64_counts={
                str(k): sum(row["rank64"] == k for row in subset)
                for k in sorted({row["rank64"] for row in subset})
            },
            rank32_counts={
                str(k): sum(row["rank32"] == k for row in subset)
                for k in sorted({row["rank32"] for row in subset})
            },
            projection64_mean=float(np.mean(ratios64)) if ratios64 else None,
            projection64_median=float(np.median(ratios64)) if ratios64 else None,
            projection64_min=min(ratios64) if ratios64 else None,
            projection64_max=max(ratios64) if ratios64 else None,
            projection32_mean=float(np.mean(ratios32)) if ratios32 else None,
            uniform_dot_nonpositive=sum(row["uniform_dot"] <= 0 for row in subset),
            oracle_dot_nonpositive=sum(row["oracle_dot"] <= 0 for row in subset),
            oracle_rescues=sum(
                row["uniform_dot"] <= 0 < row["oracle_dot"] for row in subset
            ),
            uniform_dot_mean=float(np.mean([row["uniform_dot"] for row in subset]))
            if subset
            else None,
            oracle_dot_mean=float(np.mean([row["oracle_dot"] for row in subset]))
            if subset
            else None,
            adam_uniform_decrease=sum(
                row["adam_uniform_risk_dot_delta"] < 0 for row in subset
            ),
            adam_oracle_decrease=sum(
                row["adam_oracle_risk_dot_delta"] < 0 for row in subset
            ),
            adam_oracle_worse_than_uniform=sum(
                row["adam_oracle_risk_dot_delta"] > row["adam_uniform_risk_dot_delta"]
                for row in subset
            ),
            adam_uniform_dot_delta_mean=float(
                np.mean([row["adam_uniform_risk_dot_delta"] for row in subset])
            )
            if subset
            else None,
            adam_oracle_dot_delta_mean=float(
                np.mean([row["adam_oracle_risk_dot_delta"] for row in subset])
            )
            if subset
            else None,
        )
    return result


def run():
    if OUTPUT.exists() and (
        not OUTPUT.is_dir()
        or {item.name for item in OUTPUT.iterdir()} != {"cases.first_attempt.partial"}
    ):
        raise FileExistsError("refusing to overwrite prior or ambiguous results")
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha(PRIOR / "completed.json") != PRIOR_COMPLETED_SHA:
        raise ValueError("previous completed manifest mismatch")
    completed = read_json(PRIOR / "completed.json")
    for name in ("cases.json", "summary.json"):
        if sha(PRIOR / name) != completed["artifact_sha256"][name]:
            raise ValueError("previous case or summary hash mismatch")
    prior = read_json(PRIOR / "summary.json")
    cases = read_json(PRIOR / "cases.json")
    warm_json, warm_pt = RUN / "warm.json", RUN / "warm.pt"
    input_paths = {
        "warm_json": warm_json,
        "warm_pt": warm_pt,
        "density_map": MAP,
        "source_checkpoint": DEFAULT_PARENT,
        "prior_script": ROOT / "joint_lora_gradient_probe_20261104.py",
    }
    hashes = {name: sha(path) for name, path in input_paths.items()}
    for name in ("warm_json", "warm_pt", "density_map", "source_checkpoint"):
        if hashes[name] != prior["input_sha256"][name]:
            raise ValueError(f"previous probe input changed: {name}")
    if (
        hashes["prior_script"] != completed["script_sha256"]
        or hashes["source_checkpoint"] != PARENT_SHA256
    ):
        raise ValueError("prior script or qualified source mismatch")
    warm, mapping = read_json(warm_json), read_json(MAP)
    ids = warm["label_shots"]
    if (
        len(ids) != 20
        or len(set(ids)) != 20
        or ids != prior["label_shots"]
        or ids != mapping["calibration_ids"]
        or len(cases) != 60
        or any(sum(row["shot"] == sid for row in cases) != 3 for sid in ids)
        or {row["shot"] for row in cases} != set(ids)
    ):
        raise ValueError("exact same ended20 and all 60 prior cases required")
    for row in cases:
        if (
            row["arrival"] != row["origin"] + 8
            or row["future_risk_point"] != row["origin"] + 32
            or (row["origin"] - 135) % 32
            or row["supported"] <= 0
            or (row["group"] == "positive" and not 30 < row["actuallead"] <= 125.000001)
        ):
            raise ValueError("stored causal, support, or 30ms contract failed")
    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    base_before = state_digest(base)
    state = torch.load(warm_pt, map_location="cpu", weights_only=True)
    model = JointLoRA(base, state["W0"])
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if (
        state_digest(model) != warm["fitted_state_sha256"]
        or state_digest(model) != prior["warm_state_sha256"]
    ):
        raise ValueError("warm model state receipt differs from previous probe")
    for parameter in model.neural_parameters():
        parameter.requires_grad_(True)
    parameters = model.neural_parameters()
    if sum(p.numel() for p in parameters) != 512 or {id(p) for p in parameters} != {
        id(p) for p in model.parameters() if p.requires_grad
    }:
        raise ValueError("only 512 Q/V LoRA coordinates may carry gradients")
    model_before = state_digest(model)
    saved_stats = {row["shot"]: row for row in prior["shots"]}
    signal_reader = io23.read_signals
    accesses, rows, matrices, risks, uniforms = [], [], [], [], []

    def only_ended_signal(device, sid, *args, **kwargs):
        if device != "EAST" or type(sid) is not int or sid not in ids:
            raise ValueError("only prior ended20 EAST signals are authorized")
        accesses.append(sid)
        return signal_reader(device, sid, *args, **kwargs)

    with (
        patch.object(io23, "read_signals", side_effect=only_ended_signal),
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("new HDF outcomes are forbidden"),
        ),
    ):
        for i, sid in enumerate(ids, 1):
            path = io23.path_for("EAST", sid)
            pre = path.stat()
            old_stat = saved_stats[sid]
            if (pre.st_size, pre.st_mtime_ns) != (
                old_stat["size"],
                old_stat["mtime_ns"],
            ):
                raise ValueError("ended-shot HDF stat differs from prior probe")
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            shot = prepare_actual(raw, 21, mapping)
            for case in (row for row in cases if row["shot"] == sid):
                if case["future_risk_point"] >= len(shot["values"]):
                    raise ValueError(
                        "stored coordinate is outside the same signal record"
                    )
                row, matrix, risk, uniform = evaluate_case(
                    model, shot, case, parameters, float(parent["threshold"])
                )
                rows.append(row)
                matrices.append(matrix.cpu().numpy())
                risks.append(risk.cpu().numpy())
                uniforms.append(uniform.cpu().numpy())
            post = path.stat()
            if (pre.st_size, pre.st_mtime_ns) != (post.st_size, post.st_mtime_ns):
                raise ValueError("ended-shot HDF changed while reading")
            print(f"ended shots {i}/20; retained {len(rows)}/60", flush=True)
    if (
        accesses != ids
        or len(rows) != 60
        or state_digest(base) != base_before
        or state_digest(model) != model_before
        or any(p.grad is not None for p in model.parameters())
    ):
        raise AssertionError("scope, no-drop or frozen-parameter verification failed")
    if any(sha(path) != hashes[name] for name, path in input_paths.items()):
        raise AssertionError("bound probe input changed during analysis")
    summary = dict(
        scope="TRAINED_ENDED20_60_CASE_ORACLE_DIAGNOSTIC_NOT_ONLINE_POLICY",
        cases=len(rows),
        shots=ids,
        channels=CHANNELS,
        device="cpu",
        threads=torch.get_num_threads(),
        input_sha256=hashes,
        prior_completed_sha256=PRIOR_COMPLETED_SHA,
        warm_state_sha256=model_before,
        source_state_sha256=base_before,
        column_contract="g_c=gradient of channel c masked squared error divided by all observed coordinates",
        rank_contract="SVD float64 of stored float32 G; tol=max(shape)*eps64*smax; report eps32 sensitivity",
        oracle_contract="maximize dot(Gw,g_risk): 10 smallest weights .5, median 1, 10 largest 1.5",
        adam_contract="copy-only simulated zero-moment first Adam step, lr1e-4, clip1, eps1e-8, float32; radius .5 inactive",
        max_sum_gradient_abs_error=max(
            row["sum_gradient_max_abs_error"] for row in rows
        ),
        max_sum_gradient_relative_error=max(
            row["sum_gradient_relative_error"] for row in rows
        ),
        groups=summarize(rows),
        updated_parameters=0,
        hdf_outcomes_read=0,
        independent_validation=False,
        online_policy_tested=False,
    )
    OUTPUT.mkdir(parents=False, exist_ok=True)
    with (OUTPUT / "cases.json").open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (OUTPUT / "gradients.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            G=np.stack(matrices),
            risk=np.stack(risks),
            uniform=np.stack(uniforms),
        )
    output_names = ["cases.json", "summary.json", "gradients.npz"]
    if (OUTPUT / "cases.first_attempt.partial").exists():
        output_names.append("cases.first_attempt.partial")
    result = dict(
        status="COMPLETED_WARM20_CPU_SPAN_DIAGNOSTIC",
        script_sha256=sha(__file__),
        artifact_sha256={name: sha(OUTPUT / name) for name in output_names},
    )
    with (OUTPUT / "completed.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
