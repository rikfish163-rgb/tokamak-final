"""Finite labelled, bounded-state witnesses. This is not online evaluation."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.model import latent_forward
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .geometry import build_metric, max_rms, project_state, shift
from .model import cached_risk, direct_lora, encode, source_basis

REFERENCE = io23.ROOT / "probe_outputs/centered_gate_stream_v1"
REFERENCE_SHA = "2b28ac02039b9106627c5aca865d23bcdd340ca2eda5d55d328aea46850f0d42"
PLAN = io23.ROOT / "ADAPTIVITY_CAPACITY_PLAN_20261102.md"
FAMILIES = ("gates", "ffn_lora")


def take(pack, indices):
    return {k: v[indices] for k, v in pack.items()}


@torch.no_grad()
def cache(model, signal, basis):
    parts = []
    for lo in range(0, len(signal["values"]), 32):
        parts.append(
            encode(
                model, signal, np.arange(lo, min(lo + 32, len(signal["values"]))), basis
            )
        )
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


@torch.no_grad()
def cold_curve(model, pack):
    zs, ps = [], []
    for lo in range(0, len(pack["h"]), 32):
        p = take(pack, slice(lo, lo + 32))
        z = cached_risk(model, p, p["h"].new_zeros(len(p["h"]), 16, 64))
        zs.append(z.cpu().numpy())
        ps.append(z.sigmoid().cpu().numpy())
    return np.concatenate(zs), np.concatenate(ps)


@torch.no_grad()
def full_curve(model, signal, pack, basis, state, family, cold_z, cold_p, *, direct):
    z, probability = cold_z.copy(), cold_p.copy()
    max_hidden_rms, max_difference = 0.0, 0.0
    for lo in range(160, len(cold_z), 32):
        indices = np.arange(lo, min(lo + 32, len(cold_z)))
        p = take(pack, slice(lo, lo + len(indices)))
        delta = shift(p, state, family)
        cached = cached_risk(model, p, delta)
        if direct and family == "ffn_lora":
            actual = direct_lora(
                model,
                signal,
                indices,
                basis,
                state,
                torch.ones(len(indices), dtype=torch.bool, device=state.device),
            )
            actual_z = actual["logit"]
            actual_delta = actual["h"][:, :16] - p["h"][:, :16]
        elif direct:
            native = {k: p[k] for k in ("h", "valid", "long", "snapshot")}
            actual_z = latent_forward(model, native, state)["logit"]
            actual_delta = delta
        else:
            actual_z, actual_delta = cached, delta
        valid = p["valid"][:, :16]
        rms = (
            (actual_delta.double().square() * valid[:, :, None]).sum((1, 2))
            / (valid.sum(1) * 64).clamp_min(1)
        ).sqrt()
        max_hidden_rms = max(max_hidden_rms, float(rms.max()))
        max_difference = max(max_difference, float((cached - actual_z).abs().max()))
        z[indices] = actual_z.cpu().numpy()
        probability[indices] = actual_z.sigmoid().cpu().numpy()
    assert np.array_equal(probability[:160], cold_p[:160])
    assert max_hidden_rms <= 0.250001, max_hidden_rms
    if direct:
        assert max_difference <= 2e-5, max_difference
    return (
        z,
        probability,
        dict(
            max_hidden_rms=max_hidden_rms, cached_direct_logit_difference=max_difference
        ),
    )


def eligible_points(times, flag, down):
    mutable = np.arange(len(times)) >= 160
    if flag:
        return mutable & (np.round((down - times) * 1000, 9) > 30)
    return mutable


def assess(times, z, probability, flag, down, threshold):
    decision = classify(times, probability, flag, down, threshold)
    eligible = eligible_points(times, flag, down)
    if not eligible.any():
        raise ValueError("selected case has no mutable eligible endpoint")
    boundary = math.log(threshold / (1 - threshold))
    peak = float(z[eligible].max())
    margin = peak - boundary if flag else boundary - peak
    operation = decision["timely"] if flag else not decision["alarm"]
    return dict(
        **decision,
        logit_margin=margin,
        operational_hit=bool(operation),
        witness=bool(operation and margin >= 0.05),
        early_over3000=bool(decision["timely"] and decision["lead_ms"] > 3000),
    )


def select_points(z, eligible, flag, previous=None):
    points = np.flatnonzero(eligible)
    if not len(points):
        raise ValueError("no eligible search point")
    if flag:
        top = points[np.argsort(z[points], kind="stable")[-16:]]
        spread = points[
            np.linspace(0, len(points) - 1, min(16, len(points)), dtype=int)
        ]
        return np.unique(np.r_[top, spread])
    top = points[np.argsort(z[points], kind="stable")[-32:]]
    pool = np.unique(np.r_[top, previous]) if previous is not None else top
    return pool[np.argsort(z[pool], kind="stable")[-64:]]


def optimize(
    model,
    signal,
    times,
    pack,
    basis,
    metric,
    cold_z,
    cold_p,
    flag,
    down,
    threshold,
    family,
):
    shape = (16, 2) if family == "gates" else (4, 64)
    parameter = torch.nn.Parameter(torch.zeros(shape, device="cuda"))
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    eligible = eligible_points(times, flag, down)
    selected = select_points(cold_z, eligible, flag)
    boundary = math.log(threshold / (1 - threshold))
    best = dict(
        state=parameter.detach().clone(),
        z=cold_z.copy(),
        probability=cold_p.copy(),
        score=assess(times, cold_z, cold_p, flag, down, threshold),
        step=0,
        validation=dict(max_hidden_rms=0.0, cached_direct_logit_difference=0.0),
    )
    stages, trace = [], []
    for step in range(1, 97):
        optimizer.zero_grad(set_to_none=True)
        selected_pack = take(pack, torch.as_tensor(selected, device="cuda"))
        logits = cached_risk(
            model, selected_pack, shift(selected_pack, parameter, family)
        )
        peak = logits.max()
        gap = boundary + 0.10 - peak if flag else peak - boundary + 0.10
        loss = gap.relu().square()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite labelled capacity objective")
        loss.backward()
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite capacity gradient")
        grad_norm = float(gradient.double().norm())
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(project_state(metric, parameter, family, radius=0.25))
        trace.append(
            dict(
                step=step,
                loss=float(loss.detach()),
                gradient_norm=grad_norm,
                state_norm=float(parameter.detach().double().norm()),
            )
        )
        if step % 24:
            continue
        with torch.no_grad():
            z, probability, validation = full_curve(
                model,
                signal,
                pack,
                basis,
                parameter,
                family,
                cold_z,
                cold_p,
                direct=True,
            )
            score = assess(times, z, probability, flag, down, threshold)
            measured = float(max_rms(metric, parameter, family))
        stages.append(
            dict(
                step=step,
                selected=selected.tolist(),
                score=score,
                metric_max_rms=measured,
                **validation,
            )
        )
        if score["logit_margin"] > best["score"]["logit_margin"]:
            best = dict(
                state=parameter.detach().clone(),
                z=z,
                probability=probability,
                score=score,
                step=step,
                validation=validation,
            )
        if score["witness"]:
            break
        if not flag:
            selected = select_points(z, eligible, flag, selected)
    best["trace"], best["stages"] = trace, stages
    return best


def main(root, *, preflight=False):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    reference_protocol = read(REFERENCE / "protocol.json")
    assert read(REFERENCE / "verification.json")["status"] == "PASS"
    bounds = read(REFERENCE / "timing_bounds.json")
    assert bounds["completed_sha256"] == REFERENCE_SHA
    ids = set(
        bounds["potentially_changeable_missed_D"] + bounds["potentially_changeable_FP"]
    )
    assert len(ids) == 74
    order = [
        (i + 1, r["shot"])
        for i, r in enumerate(reference_protocol["order"])
        if r["shot"] in ids
    ]
    assert all(rank > 20 for rank, _ in order)
    model, parent, cold_path = load_model()
    assert sha(cold_path) == reference_protocol["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    mapping = read(Path(reference_protocol["reference"]) / "input_map.json")
    assert (
        sha(Path(reference_protocol["reference"]) / "input_map.json")
        == reference_protocol["input_map_sha256"]
    )
    basis = source_basis(model)
    root.mkdir(parents=True)
    (root / "cases").mkdir()
    torch.save({"basis": basis.cpu()}, root / "source_basis.pt")
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="POSTHOC_LABEL_ORACLE_FINITE_WITNESSES_NOT_ONLINE",
        preflight=preflight,
        families=FAMILIES,
        order=order[:1] if preflight else order,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        reference_protocol_sha256=sha(REFERENCE / "protocol.json"),
        clock_bounds_sha256=sha(REFERENCE / "timing_bounds.json"),
        source_sha256=reference_protocol["source_sha256"],
        cold_sha256=sha(cold_path),
        basis_sha256=sha(root / "source_basis.pt"),
        threshold=float(parent["threshold"]),
        radius=0.25,
        max_steps=96,
        learning_rate=0.05,
        certification_interval=24,
        objective_margin=0.10,
        witness_margin=0.05,
        first_mutable_index=160,
        uses_current_case_label_and_future_record=True,
        can_be_called_online_gain=False,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    outcomes = []
    for rank, sid in protocol["order"]:
        prefix = f"{rank:04d}_{sid}"
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        flag, down = io23.read_outcome("EAST", sid)
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        pack = cache(model, signal, basis)
        cold_z, cold_p = cold_curve(model, pack)
        with np.load(
            REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False
        ) as reference:
            assert np.array_equal(raw["times"], reference["times"])
            assert np.array_equal(cold_p, reference["cold"]), (sid, "cold drift")
        metric = build_metric(pack, torch.arange(len(cold_z), device="cuda") >= 160)
        folder = root / "cases" / prefix
        folder.mkdir()
        for family in FAMILIES:
            result = optimize(
                model,
                signal,
                raw["times"],
                pack,
                basis,
                metric,
                cold_z,
                cold_p,
                flag,
                down,
                float(parent["threshold"]),
                family,
            )
            state_path = folder / f"{family}_state.pt"
            torch.save({"state": result["state"].cpu()}, state_path)
            curve_path = folder / f"{family}_curves.npz"
            with curve_path.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    times=raw["times"],
                    cold=cold_p,
                    probability=result["probability"],
                    logit=result["z"],
                )
            receipt = dict(
                rank=rank,
                shot=sid,
                disruptive=flag,
                family=family,
                score=result["score"],
                best_step=result["step"],
                steps_run=len(result["trace"]),
                validation=result["validation"],
                trace=result["trace"],
                stages=result["stages"],
                raw_sha256=sha(io23.path_for("EAST", sid)),
                state_sha256=sha(state_path),
                curves_sha256=sha(curve_path),
                label_oracle=True,
                not_online=True,
            )
            _durable_json(folder / f"{family}_receipt.json", receipt)
            outcomes.append(
                {
                    k: receipt[k]
                    for k in (
                        "rank",
                        "shot",
                        "disruptive",
                        "family",
                        "score",
                        "best_step",
                        "steps_run",
                    )
                }
            )
        progress = dict(
            cases=len(outcomes) // 2,
            total=len(protocol["order"]),
            seconds=time.monotonic() - start,
        )
        _durable_json(root / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        del pack, metric
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    summary = {
        family: {
            label: dict(
                n=sum(
                    r["family"] == family and r["disruptive"] == flag for r in outcomes
                ),
                witnesses=sum(
                    r["family"] == family
                    and r["disruptive"] == flag
                    and r["score"]["witness"]
                    for r in outcomes
                ),
                early_over3000=sum(
                    r["family"] == family
                    and r["disruptive"] == flag
                    and r["score"]["witness"]
                    and r["score"]["early_over3000"]
                    for r in outcomes
                ),
            )
            for label, flag in (("D", True), ("SAFE", False))
        }
        for family in FAMILIES
    }
    completed = dict(
        status="COMPLETED_POSTHOC_CAPACITY",
        summary=summary,
        outcomes=outcomes,
        source_state_unchanged=True,
        no_online_gain_claim=True,
        method_promoted=False,
        scientific_goal_achieved=False,
        seconds=time.monotonic() - start,
    )
    _durable_json(root / "completed.json", completed)
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight=args.preflight)
