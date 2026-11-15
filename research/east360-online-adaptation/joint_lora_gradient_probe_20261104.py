"""Ended-warm20-only CPU diagnostic/risk gradient alignment; no fitting."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

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


ROOT = io23.ROOT
RUN = ROOT / "probe_outputs/joint_lora_paired360_20261104_v1"
MAP = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_map.json"
OUTPUT = ROOT / "probe_outputs/joint_lora_gradient_probe_20261104_v1"
MAP_SHA256 = "43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2"
MODES = ("real", "hold", "reverse")
GROUPS = ("safe", "earlier", "positive")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(module):
    # Exactly the fixed-state receipt contract of the active producer.
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_candidates(times, disruptive, down):
    """Uniformly select at most three valid q+32 endpoints, not model errors."""
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or not np.isfinite(times).all()
        or (len(times) > 1 and not (np.diff(times) > 0).all())
    ):
        raise ValueError("finite strictly increasing shot clock required")
    if type(disruptive) is not bool or (
        disruptive and (down is None or not math.isfinite(down))
    ):
        raise ValueError("HDF disruptive flag and disruption time required")
    if not disruptive and down is not None:
        raise ValueError("safe shots do not have a disruption endpoint")
    label, valid = io23.labels_at(times, disruptive, down, 125.0)
    candidates = []
    for origin in range(135, len(times) - 32, 32):
        future = origin + 32
        if not valid[future]:
            continue
        y = int(label[future])
        lead = round(float((down - times[future]) * 1000.0), 9) if disruptive else None
        if disruptive and y and lead <= 30.0:
            continue
        candidates.append(
            dict(
                origin=origin,
                arrival=origin + 8,
                future_risk_point=future,
                actuallead=lead,
                label=y,
                group="safe" if not disruptive else ("positive" if y else "earlier"),
            )
        )
    if len(candidates) <= 3:
        return candidates
    indexes = np.linspace(0, len(candidates) - 1, 3, dtype=int)
    return [candidates[i] for i in indexes]


def centered_risk_loss(logit, label, threshold):
    if label not in (0, 1) or not 0 < threshold < 1:
        raise ValueError("binary label and threshold in (0,1) required")
    tau = math.log(threshold / (1.0 - threshold))
    return F.binary_cross_entropy_with_logits(
        logit - tau, torch.full_like(logit, float(label))
    )


def control_losses(prediction, target, observed):
    if (
        prediction.shape != target.shape
        or observed.shape != target.shape
        or observed.dtype != torch.bool
    ):
        raise ValueError("matching prediction, target and boolean mask required")
    if not observed.any():
        return None
    result = {}
    for mode, desired in (
        ("real", target),
        ("hold", torch.zeros_like(target)),
        ("reverse", -target),
    ):
        result[mode] = (prediction - desired).square().masked_fill(
            ~observed, 0.0
        ).sum() / observed.sum()
    return result


def gradient_metrics(diag, risk):
    if (
        diag.shape != risk.shape
        or not torch.isfinite(diag).all()
        or not torch.isfinite(risk).all()
    ):
        raise ValueError("matching finite gradients required")
    a, b = diag.detach().double().reshape(-1), risk.detach().double().reshape(-1)
    da, rb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    dot = float(torch.dot(a, b))
    cosine = max(-1.0, min(1.0, dot / (da * rb))) if da and rb else None
    return dict(
        diag_norm=da,
        risk_norm=rb,
        inner_product=dot,
        cosine=cosine,
        first_order_delta_per_unit_step=-dot,
    )


def summarize(rows, *, modes=MODES):
    result = {}
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        result[group] = {}
        for mode in modes:
            unsupported = sum(row["supported"] == 0 for row in subset)
            valid = [row["controls"][mode] for row in subset if row["supported"] > 0]
            if any(item is None for item in valid):
                raise ValueError("supported case missing control gradients")
            comparable = [item for item in valid if item["cosine"] is not None]
            aligned = sum(item["inner_product"] > 0 for item in comparable)
            opposed = sum(item["inner_product"] < 0 for item in comparable)
            tied = sum(item["inner_product"] == 0 for item in comparable)
            result[group][mode] = dict(
                cases=len(subset),
                unsupported=unsupported,
                zero_diag_norm=sum(item["diag_norm"] == 0 for item in valid),
                zero_risk_norm=sum(item["risk_norm"] == 0 for item in valid),
                zero_both_norms=sum(
                    item["diag_norm"] == 0 and item["risk_norm"] == 0 for item in valid
                ),
                comparable=len(comparable),
                aligned=aligned,
                opposed=opposed,
                zero_dot=tied,
                aligned_fraction=aligned / len(comparable) if comparable else None,
                mean_diag_norm=float(np.mean([item["diag_norm"] for item in valid]))
                if valid
                else None,
                median_diag_norm=float(np.median([item["diag_norm"] for item in valid]))
                if valid
                else None,
                mean_risk_norm=float(np.mean([item["risk_norm"] for item in valid]))
                if valid
                else None,
                median_risk_norm=float(np.median([item["risk_norm"] for item in valid]))
                if valid
                else None,
                mean_cosine=float(np.mean([item["cosine"] for item in comparable]))
                if comparable
                else None,
                median_cosine=float(np.median([item["cosine"] for item in comparable]))
                if comparable
                else None,
                mean_inner_product=float(
                    np.mean([item["inner_product"] for item in comparable])
                )
                if comparable
                else None,
            )
    return result


def _flat_grad(loss, parameters, *, retain_graph=False):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients])


def evaluate_case(model, shot, candidate, parameters, threshold):
    origin, future = candidate["origin"], candidate["future_risk_point"]
    x, available = window_batch(shot, [origin], device="cpu")
    future_values = torch.as_tensor(
        shot["values"][origin + 1 : origin + 9][None], device="cpu"
    )
    future_mask = torch.as_tensor(
        shot["available"][origin + 1 : origin + 9][None], device="cpu"
    )
    with torch.no_grad():
        target, observed = direct_target(
            levels(model.clone, x, available),
            available,
            levels(model.clone, future_values, future_mask),
            future_mask,
        )
    x_risk, a_risk = window_batch(shot, [future], device="cpu")
    risk_logit = model(x_risk, a_risk)["logit"]
    risk_loss = centered_risk_loss(risk_logit, candidate["label"], threshold)
    risk_gradient = _flat_grad(risk_loss, parameters)
    if risk_gradient.numel() != 512:
        raise ValueError("exactly 512 risk-gradient coordinates required")
    supported = int(observed.sum().item())
    output = dict(
        candidate,
        supported=supported,
        risk_logit=float(risk_logit.detach().item()),
        risk_loss=float(risk_loss.detach().item()),
        risk_grad_norm=float(torch.linalg.vector_norm(risk_gradient.double()).item()),
        controls={mode: None for mode in MODES},
    )
    if supported:
        prediction = model(x, available)["diagnostic"]
        losses = control_losses(prediction, target, observed)
        for index, mode in enumerate(MODES):
            gradient = _flat_grad(
                losses[mode], parameters, retain_graph=index < len(MODES) - 1
            )
            if gradient.numel() != 512:
                raise ValueError("exactly 512 diagnostic-gradient coordinates required")
            output["controls"][mode] = dict(
                diag_loss=float(losses[mode].detach().item()),
                **gradient_metrics(gradient, risk_gradient),
            )
    return output


def run():
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite prior probe results")
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warm_json, warm_pt = RUN / "warm.json", RUN / "warm.pt"
    hashes = {
        "warm_json": sha(warm_json),
        "warm_pt": sha(warm_pt),
        "density_map": sha(MAP),
        "source_checkpoint": sha(DEFAULT_PARENT),
        "script": sha(__file__),
    }
    if (
        hashes["density_map"] != MAP_SHA256
        or hashes["source_checkpoint"] != PARENT_SHA256
    ):
        raise ValueError("qualified source or historical density map hash mismatch")
    with warm_json.open(encoding="utf-8") as stream:
        warm = json.load(stream)
    with MAP.open(encoding="utf-8") as stream:
        mapping = json.load(stream)
    ids = warm["label_shots"]
    if (
        len(ids) != 20
        or len(set(ids)) != 20
        or any(type(s) is not int for s in ids)
        or mapping["calibration_ids"] != ids
    ):
        raise ValueError(
            "exactly the saved, density-map-calibration ended20 are allowed"
        )
    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_before = state_digest(base)
    state = torch.load(warm_pt, map_location="cpu", weights_only=True)
    model = JointLoRA(base, state["W0"])
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if state_digest(model) != warm["fitted_state_sha256"]:
        raise ValueError("warm.pt does not match the run warm-state receipt")
    for parameter in model.neural_parameters():
        parameter.requires_grad_(True)
    parameters = model.neural_parameters()
    if (
        sum(p.numel() for p in parameters) != 512
        or any(p.requires_grad for p in model.decoder_parameters())
        or {id(p) for p in parameters}
        != {id(p) for p in model.parameters() if p.requires_grad}
    ):
        raise ValueError("only 512 LoRA coordinates may receive gradients")
    threshold = float(parent["threshold"])
    model_before = state_digest(model)
    signal_reader, outcome_reader = io23.read_signals, io23.read_outcome
    access, rows, shots = [], [], []

    def only_ended_signal(device, sid, *args, **kwargs):
        if device != "EAST" or type(sid) is not int or sid not in ids:
            raise ValueError("only the warm.json ended20 EAST signals are authorized")
        access.append(("signal", sid))
        return signal_reader(device, sid, *args, **kwargs)

    def only_ended_outcome(device, sid):
        if device != "EAST" or type(sid) is not int or sid not in ids:
            raise ValueError("only the warm.json ended20 EAST outcomes are authorized")
        access.append(("outcome", sid))
        return outcome_reader(device, sid)

    with (
        patch.object(io23, "read_signals", side_effect=only_ended_signal),
        patch.object(io23, "read_outcome", side_effect=only_ended_outcome),
    ):
        for index, sid in enumerate(ids, 1):
            hdf = io23.path_for("EAST", sid)
            stat_before = hdf.stat()
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            disruptive, down = io23.read_outcome("EAST", sid)
            prepared = prepare_actual(raw, 21, mapping)
            prepared["times"] = raw["times"]
            selected = select_candidates(raw["times"], disruptive, down)
            for candidate in selected:
                result = evaluate_case(
                    model, prepared, candidate, parameters, threshold
                )
                rows.append(dict(shot=sid, disruptive=disruptive, **result))
            stat_after = hdf.stat()
            if (stat_before.st_size, stat_before.st_mtime_ns) != (
                stat_after.st_size,
                stat_after.st_mtime_ns,
            ):
                raise ValueError("ended-shot HDF changed while reading")
            shots.append(
                dict(
                    shot=sid,
                    disruptive=disruptive,
                    selected=len(selected),
                    size=stat_after.st_size,
                    mtime_ns=stat_after.st_mtime_ns,
                )
            )
            print(f"ended shots {index}/20; selected {len(rows)}/60", flush=True)
    if (
        access != [(kind, sid) for sid in ids for kind in ("signal", "outcome")]
        or len(rows) > 60
        or state_digest(base) != source_before
        or state_digest(model) != model_before
        or any(p.grad is not None for p in model.parameters())
    ):
        raise AssertionError("scope, <=60, frozen state or gradient isolation failed")
    if (
        sha(warm_json) != hashes["warm_json"]
        or sha(warm_pt) != hashes["warm_pt"]
        or sha(MAP) != hashes["density_map"]
        or sha(DEFAULT_PARENT) != hashes["source_checkpoint"]
    ):
        raise AssertionError("probe inputs changed during analysis")
    summary = dict(
        scope="ENDED20_TRAINING_SET_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION",
        label_shots=ids,
        device="cpu",
        torch_threads=torch.get_num_threads(),
        warm_state_sha256=model_before,
        source_state_sha256=source_before,
        input_sha256=hashes,
        threshold=threshold,
        risk_bce_center_tau=math.log(threshold / (1 - threshold)),
        candidate_policy="q=135+32k; future=q+32 valid preterminal; positive rounded lead strictly >30ms; uniform <=3/shot",
        negative_gradient_first_order="delta risk loss per unit diagnostic-gradient step = -dot(g_diag,g_risk)",
        cases=len(rows),
        shots=shots,
        groups=summarize(rows),
        updated_parameters=0,
        data_fitting=False,
        independent_validation=False,
    )
    OUTPUT.mkdir(parents=False, exist_ok=False)
    for name, value in (("cases.json", rows), ("summary.json", summary)):
        with (OUTPUT / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    completed = dict(
        status="COMPLETED_WARM20_CPU_GRADIENT_DIAGNOSTIC",
        artifact_sha256={
            name: sha(OUTPUT / name) for name in ("cases.json", "summary.json")
        },
        script_sha256=hashes["script"],
    )
    with (OUTPUT / "completed.json").open("x", encoding="utf-8") as stream:
        json.dump(completed, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    run()
