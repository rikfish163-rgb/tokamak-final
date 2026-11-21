"""Frozen source/cold forecast VJPs on only the already-ended target20."""

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.model import (
    encode_views,
    latent_forward,
    project_gates,
)
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure, read
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE
from ..warmstart_alignment_20261022.alignment import transform
from .core import centered_loss, output_derivatives

PLAN = io23.ROOT / "GATE_LOSS_AUDIT_PLAN_20261101.md"


def norm(vector):
    return float(vector.detach().double().norm())


def cosine(g, h):
    denominator = norm(g) * norm(h)
    return float((g.double() * h.double()).sum()) / denominator if denominator else None


def point(model, signal, end):
    # Preserve canonical32 rows even when only one origin VJP is inspected.
    lo = end // 32 * 32
    ends = np.arange(lo, min(lo + 32, len(signal["values"])))
    pack = encode_views(model, signal, ends)
    return {k: v[end - lo : end - lo + 1] for k, v in pack.items()}


def inspect(model, pack, future_pack, signal, origin, state, risk_label):
    beta = torch.full((16, 2), state, device="cuda", requires_grad=True)
    truth, available = targets(signal, origin, origin + 25)
    mask = torch.as_tensor(available, device="cuda")
    raw = torch.as_tensor(truth["real"], device="cuda")
    raw_reverse = torch.as_tensor(truth["reverse"], device="cuda")
    y = torch.asinh(
        (torch.where(mask, raw, model.base.center) - model.base.center)
        / model.base.scale
    )
    reverse_y = torch.asinh(
        (torch.where(mask, raw_reverse, model.base.center) - model.base.center)
        / model.base.scale
    )
    prediction = latent_forward(model, pack, beta, forecast=True)["forecast"]
    u, dc, ac = output_derivatives(prediction, y, mask)
    gradients = {}
    for name, output_gradient in (("original", u), ("DC", dc), ("AC", ac)):
        gradients[name] = torch.autograd.grad(
            prediction, beta, output_gradient, retain_graph=True
        )[0].detach()
    direct = model.base.forecast_loss(prediction, raw, mask)
    g_direct = torch.autograd.grad(direct, beta, retain_graph=True)[0].detach()
    shape = centered_loss(prediction, y, mask)
    gradients["centered"] = torch.autograd.grad(shape, beta, retain_graph=True)[
        0
    ].detach()
    reverse_loss = model.base.forecast_loss(prediction, raw_reverse, mask)
    reverse_gradient = torch.autograd.grad(reverse_loss, beta, retain_graph=True)[
        0
    ].detach()
    reverse_shape = centered_loss(prediction, reverse_y, mask)
    reverse_shape_gradient = torch.autograd.grad(reverse_shape, beta)[0].detach()
    g = gradients["original"]
    denominator = max(norm(g), norm(gradients["DC"]) + norm(gradients["AC"]), 1e-12)
    residual = norm(g - gradients["DC"] - gradients["AC"]) / denominator
    assert residual <= 2e-5, residual
    direct_error = norm(g - g_direct) / max(norm(g), norm(g_direct), 1e-12)
    assert direct_error <= 2e-5, direct_error
    with torch.no_grad():
        error = prediction.detach() - y
        observed = mask.sum((0, 1))
        saturation = ((error.abs() >= 1) & mask).sum((0, 1))
        positive = ((error >= 1) & mask).sum((0, 1))
        negative = ((error <= -1) & mask).sum((0, 1))
        constant_saturated = ((positive == observed) | (negative == observed)) & (
            observed > 0
        )
    risk_before, risk_gradient = None, None
    if risk_label is not None:
        future_logit = latent_forward(model, future_pack, beta)["logit"]
        risk = F.binary_cross_entropy_with_logits(
            future_logit, future_logit.new_full(future_logit.shape, risk_label)
        )
        risk_gradient = torch.autograd.grad(risk, beta)[0].detach()
        risk_before = float(risk.detach())
    proposals = {}
    with torch.no_grad():
        for name, gradient in gradients.items():
            magnitude = gradient.double().norm()
            direction = (
                (gradient.double() / magnitude).float()
                if magnitude
                else torch.zeros_like(gradient)
            )
            candidate = project_gates(beta.detach() - 0.1 * direction)
            risk_delta = None
            directional = None
            if risk_label is not None:
                z = latent_forward(model, future_pack, candidate)["logit"]
                loss = F.binary_cross_entropy_with_logits(
                    z, z.new_full(z.shape, risk_label)
                )
                risk_delta = float(loss) - risk_before
                directional = float(
                    (
                        risk_gradient.double() * (candidate - beta.detach()).double()
                    ).sum()
                )
            proposals[name] = dict(
                step_norm=norm(candidate - beta.detach()),
                risk_gradient_cosine=cosine(gradient, risk_gradient)
                if risk_gradient is not None
                else None,
                future_risk_loss_delta=risk_delta,
                first_order_risk_delta=directional,
                gradient32=gradient.flatten().cpu().tolist(),
            )
    return dict(
        origin=origin,
        maturity=origin + 25,
        future_endpoint=origin + 57,
        state_per_gate=state,
        observed=int(mask.sum()),
        gradient_norms={k: norm(v) for k, v in gradients.items()},
        direct_gradient_relative_error=direct_error,
        split_relative_error=residual,
        DC_to_total_cosine=cosine(gradients["DC"], g),
        AC_to_total_cosine=cosine(gradients["AC"], g),
        original_reverse_cosine=cosine(g, reverse_gradient),
        centered_reverse_cosine=cosine(gradients["centered"], reverse_shape_gradient),
        risk_label=risk_label,
        future_risk_loss_before=risk_before,
        observed_per_channel=observed.cpu().tolist(),
        saturated_per_channel=saturation.cpu().tolist(),
        all_observed_same_sign_saturated=constant_saturated.cpu().tolist(),
        proposals=proposals,
    )


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return dict(n=0, median=None, q10=None, q90=None)
    assert np.isfinite(values).all()
    return dict(
        n=len(values),
        median=float(np.median(values)),
        q10=float(np.quantile(values, 0.1)),
        q90=float(np.quantile(values, 0.9)),
    )


def summarize(rows):
    result = {}
    for state in (0.0, 0.2):
        subset = [r for r in rows if r["state_per_gate"] == state]
        stats = {
            key: distribution([r[key] for r in subset])
            for key in (
                "original_reverse_cosine",
                "centered_reverse_cosine",
                "DC_to_total_cosine",
                "AC_to_total_cosine",
            )
        }
        stats["gradient_norms"] = {
            name: distribution([r["gradient_norms"][name] for r in subset])
            for name in ("original", "DC", "AC", "centered")
        }
        stats["relative_AC_norm"] = distribution(
            [
                r["gradient_norms"]["AC"] / r["gradient_norms"]["original"]
                for r in subset
                if r["gradient_norms"]["original"] > 0
            ]
        )
        stats["risk_probes"] = {}
        for name in ("original", "DC", "AC", "centered"):
            eligible = [
                r["proposals"][name] for r in subset if r["risk_label"] is not None
            ]
            stats["risk_probes"][name] = dict(
                n=len(eligible),
                moving=sum(r["step_norm"] > 0 for r in eligible),
                improved=sum(r["future_risk_loss_delta"] < 0 for r in eligible),
                worsened=sum(r["future_risk_loss_delta"] > 0 for r in eligible),
                unchanged=sum(r["future_risk_loss_delta"] == 0 for r in eligible),
                delta=distribution([r["future_risk_loss_delta"] for r in eligible]),
                cosine=distribution([r["risk_gradient_cosine"] for r in eligible]),
            )
        result[str(state)] = stats
    return result


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    model, _, cold = load_model()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ids = [r["shot"] for r in read(REFERENCE / "order.json")[:20]]
    mapping = read(REFERENCE / "input_map.json")
    assert mapping["calibration_ids"] == ids
    root.mkdir(parents=True)
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="ENDED20_ONLY_OFFLINE_GRADIENT_DIAGNOSTIC_NOT_CONTROLLER",
        target_ids=ids,
        reference=str(REFERENCE),
        cold_state_sha256=sha(cold),
        input_map_sha256=sha(REFERENCE / "input_map.json"),
        source_path=str(
            io23.ROOT
            / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
        ),
        source_sha256=sha(
            io23.ROOT
            / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
        ),
        no_training=True,
        target_suffix_signals_read=0,
        target_suffix_outcomes_read=0,
        states=[0, 0.2],
        max_origins_per_shot=8,
        future_risk_proxy="125ms endpoint BCE, lead>30 on D",
        input_hashes={str(sid): sha(io23.path_for("EAST", sid)) for sid in ids},
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        plan_sha256=sha(PLAN),
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    rows, inventory = [], []
    for sid in ids:
        signal_raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: signal_raw[k] for k in ("values", "available")}, mapping)
        flag, down = io23.read_outcome(
            "EAST", sid
        )  # All20 already ended; offline proxy audit only.
        candidates = np.arange(159, len(signal["values"]) - 32, 128, dtype=int) - 25
        selected = (
            candidates[
                np.linspace(0, len(candidates) - 1, min(8, len(candidates)), dtype=int)
            ]
            if len(candidates)
            else []
        )
        inventory.append(
            dict(
                shot=sid,
                points=len(signal["values"]),
                origins=list(map(int, selected)),
                disruptive=flag,
            )
        )
        for o in selected:
            origin, future = int(o), int(o) + 57
            pack, future_pack = (
                point(model, signal, origin),
                point(model, signal, future),
            )
            lead = (
                round((down - signal_raw["times"][future]) * 1000, 9) if flag else None
            )
            risk_label = (
                float(lead <= 125) if flag and lead > 30 else 0.0 if not flag else None
            )
            for state in (0.0, 0.2):
                rows.append(
                    dict(
                        shot=sid,
                        disruptive=flag,
                        future_lead_ms=lead,
                        **inspect(
                            model, pack, future_pack, signal, origin, state, risk_label
                        ),
                    )
                )
        _durable_json(
            root / "progress.json",
            dict(
                shots=len(inventory),
                total=20,
                rows=len(rows),
                seconds=time.monotonic() - start,
            ),
        )
        print(json.dumps(dict(shots=len(inventory), rows=len(rows))), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="COMPLETED_AUDIT",
        summary=summarize(rows),
        rows=len(rows),
        shots_with_origins=sum(bool(r["origins"]) for r in inventory),
        input_shots=20,
        max_split_relative_error=max(r["split_relative_error"] for r in rows),
        source_state_unchanged=True,
        no_training=True,
        no_suffix_access=True,
        seconds=time.monotonic() - start,
        scientific_goal_achieved=False,
    )
    _durable_json(root / "rows.json", rows)
    _durable_json(root / "inventory.json", inventory)
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
