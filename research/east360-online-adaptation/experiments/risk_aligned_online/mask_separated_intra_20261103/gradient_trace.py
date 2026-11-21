"""Offline oracle attribution of failed feedback gradients, not an online policy.

HDF outcomes are intentionally used ONLY to diagnose whether a diagnostic
gradient helped the later operational risk loss. They cannot choose an update.
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..direct_decoder_meta_20261102.core import clone_head, prediction
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..innovation_metric_20261102.core import metric_direction
from ..lora_stream_20261102.core import encode, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..warmstart_alignment_20261022.alignment import transform
from .model import MaskSeparatedContext
from .run import LOCAL, PRIOR, PRIOR_SHA, state_digest


def channel_terms(model, predicted, raw_truth, mask):
    if (
        predicted.shape != raw_truth.shape
        or predicted.shape != mask.shape
        or predicted.shape[-2:] != (25, 23)
        or mask.dtype != torch.bool
    ):
        raise ValueError("matched25x23 direct forecasts and observed masks required")
    if (
        not torch.isfinite(predicted[mask]).all()
        or not torch.isfinite(raw_truth[mask]).all()
    ):
        raise ValueError("finite observed diagnostics required")
    target = torch.asinh(
        (torch.where(mask, raw_truth.detach(), model.base.center) - model.base.center)
        / model.base.scale
    )
    p, y = torch.where(mask, predicted, 0), torch.where(mask, target, 0)
    count = mask.sum(1, keepdim=True).clamp_min(1)
    p = p - p.sum(1, keepdim=True) / count
    y = y - y.sum(1, keepdim=True) / count
    errors = F.smooth_l1_loss(p, y, reduction="none").masked_fill(~mask, 0)
    return errors.sum((0, 1)) / mask.sum().clamp_min(1)


def metric_components(pack, gradients, full_gradient):
    """Decompose the actual step and keep reverse-mode reduction roundoff separate.

    Independent float32 backward reductions differ slightly. The inverse
    observed metric can amplify that mismatch; it is not a physical channel.
    """
    if gradients.shape != (23, 4, 64) or not torch.isfinite(gradients).all():
        raise ValueError("finite23 channel gradients in first-FFN4x64 required")
    if full_gradient.shape != (4, 64) or not torch.isfinite(full_gradient).all():
        raise ValueError("actual finite whole-loss4x64 gradient required")
    valid = pack["valid"][:, :16]
    clean = pack["phi"].double().masked_fill(~valid[:, :, None], 0)
    count = valid.sum().clamp_min(1) * 64
    flat = clean.reshape(-1, 4)
    covariance = flat.T @ flat / count
    ridge = 0.01 * covariance.trace() / 4 + 1e-8
    metric = covariance + ridge * torch.eye(4, dtype=clean.dtype, device=clean.device)
    pieces = -torch.linalg.solve(metric, gradients.double())
    full = -torch.linalg.solve(metric, full_gradient.double())
    residual = -torch.linalg.solve(
        metric, full_gradient.double() - gradients.double().sum(0)
    )
    squared_rms = (clean @ full).square().sum() / count
    scale = 0.05 / squared_rms.clamp_min(0.05**2).sqrt()
    return dict(channels=pieces * scale, reduction_residual=residual * scale)


def main(pilot, output):
    pilot, output = Path(pilot), Path(output)
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    done, protocol = read(pilot / "completed.json"), read(pilot / "protocol.json")
    if done["status"] != "COMPLETED" or done["candidate_gate"]["passed"]:
        raise ValueError("this trace diagnoses the specified failed pilot only")
    for name, expected in done["artifact_sha256"].items():
        if sha(pilot / name) != expected:
            raise ValueError("pilot artifact hash changed")
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError("source-only decoder hash changed")
    records = [r for r in read(pilot / "scored.json") if r["mode"] == "real"]
    local = read(LOCAL / "protocol.json")
    if protocol["ids"] != local["ids"]:
        raise ValueError("fixed pilot and trace20 identities must match")
    mapping = read(Path(local["calibration_reference"]) / "input_map.json")
    base, parent = load_parent(DEFAULT_PARENT)
    snapshots, rows = {}, []
    saved = torch.load(PRIOR, map_location="cuda", weights_only=True)
    code = imported_code_hashes()
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        dict(
            scope="POSTHOC_HDF_ORACLE_CHANNEL_GRADIENT_DIAGNOSIS_NOT_UPDATE_POLICY",
            pilot=str(pilot.resolve()),
            pilot_completion_sha256=sha(pilot / "completed.json"),
            samples=200,
            raw_ids=local["ids"],
            suffix340_reads=0,
            label_use="known ended20 HDF outcomes only for signed risk-gradient diagnosis",
            no_label_based_update_selection=True,
            source_unchanged=True,
            decomposition="same centered loss; per-channel gradients; shared full-step metric scaling",
            limitation="first-order local sensitivity, not physical causation or deployable policy",
            imported_code_sha256=code,
        ),
    )
    cutoff = math.log(float(parent["threshold"]) / (1 - float(parent["threshold"])))
    for stem in ("original", "reference_bt"):
        model = MaskSeparatedContext(base, policy=stem).cuda()
        basis = source_basis(model)
        decoder = clone_head(model)
        decoder.load_state_dict(saved["heads"]["forecast_only"], strict=True)
        decoder.requires_grad_(False).eval()
        snapshots[stem] = (state_digest(model), state_digest(decoder))
        for sid in local["ids"]:
            if sha(io23.path_for("EAST", sid)) != local["input_hashes"][str(sid)]:
                raise ValueError("raw ended20 hash changed")
            flag, down = io23.read_outcome("EAST", sid)
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
            for r in [v for v in records if v["stem"] == stem and v["shot"] == sid]:
                q, origin = r["query"], r["origin"]
                lead = round((down - raw["times"][q]) * 1000, 9) if flag else None
                if flag != r["disruptive"] or lead != r["actual_lead_ms"]:
                    raise ValueError("fresh HDF outcome/clock differs from pilot")
                issued = encode(
                    model,
                    {k: v[: origin + 1] for k, v in signal.items()},
                    [origin],
                    basis,
                )
                later = encode(
                    model, {k: v[: q + 1] for k, v in signal.items()}, [q], basis
                )
                target, observed = targets(
                    {k: v[:q] for k, v in signal.items()}, origin, q - 1
                )
                target = torch.as_tensor(target["real"], device="cuda")
                mask = torch.as_tensor(observed, device="cuda")
                variable = model.base.center.new_zeros(4, 64).requires_grad_(True)
                predicted = prediction(model, issued, variable, decoder)
                terms = channel_terms(model, predicted, target, mask)
                gradient = torch.autograd.grad(
                    terms.sum(), variable, retain_graph=True
                )[0]
                components = torch.stack(
                    [
                        torch.autograd.grad(terms[c], variable, retain_graph=c < 22)[0]
                        for c in range(23)
                    ]
                )
                torch.testing.assert_close(
                    components.sum(0), gradient, atol=2e-7, rtol=2e-4
                )
                recorded = gradient.new_tensor(r["receipt"]["gradient256"]).reshape(
                    4, 64
                )
                torch.testing.assert_close(gradient, recorded, atol=2e-7, rtol=2e-4)
                decomposition = metric_components(issued, components, gradient)
                pieces = decomposition["channels"]
                torch.testing.assert_close(
                    pieces.sum(0) + decomposition["reduction_residual"],
                    metric_direction(issued, gradient).double(),
                    atol=1e-7,
                    rtol=1e-5,
                )
                z = forward(model, later, variable)["logit"]
                operational_loss = F.binary_cross_entropy_with_logits(
                    z - cutoff, z.new_full(z.shape, float(flag))
                )
                risk_gradient = torch.autograd.grad(operational_loss, variable)[0]
                fraction = r["receipt"]["chosen_fraction"] or 0.0
                signed = (pieces.double() * risk_gradient.double()).sum(
                    (1, 2)
                ) * fraction
                roundoff = float(
                    (decomposition["reduction_residual"] * risk_gradient.double()).sum()
                    * fraction
                )
                rows.append(
                    dict(
                        stem=stem,
                        shot=sid,
                        query=q,
                        disruptive=flag,
                        lead_ms=lead,
                        actual_delta_bce=r["delta_bce"],
                        first_order_delta_bce=float(signed.sum()) + roundoff,
                        numerical_reduction_delta_bce=roundoff,
                        channels=[
                            dict(
                                name=name,
                                observed=int(mask[:, :, c].sum()),
                                loss=float(terms[c].detach()),
                                gradient_norm=float(components[c].double().norm()),
                                signed_first_order_delta_bce=float(signed[c]),
                            )
                            for c, name in enumerate(io23.RAW23)
                        ],
                    )
                )
            print(
                dict(stage="trace", stem=stem, shot=sid, episodes=len(rows)), flush=True
            )
        if (state_digest(model), state_digest(decoder)) != snapshots[stem]:
            raise AssertionError("diagnostic tracing modified frozen model/head")
        # Reuse the exact frozen source tensors; no parameter update or cold restore.
        base = model.base.cpu()
        del model, decoder
    if len(rows) != 200:
        raise AssertionError("both stems/all100 fixed episodes required")
    summary = {}
    for stem in ("original", "reference_bt"):
        summary[stem] = {}
        for group in ("safe", "disruptive", "danger_window"):
            local_rows = [
                r
                for r in rows
                if r["stem"] == stem
                and (
                    not r["disruptive"]
                    if group == "safe"
                    else r["disruptive"]
                    and (group == "disruptive" or 30 < r["lead_ms"] <= 125)
                )
            ]
            ids = sorted({r["shot"] for r in local_rows})

            def macro(getter):
                return float(
                    np.mean(
                        [
                            np.mean([getter(r) for r in local_rows if r["shot"] == sid])
                            for sid in ids
                        ]
                    )
                )

            summary[stem][group] = dict(
                n_shots=len(ids),
                n_points=len(local_rows),
                actual_delta=macro(lambda r: r["actual_delta_bce"]),
                first_order_delta=macro(lambda r: r["first_order_delta_bce"]),
                numerical_reduction_delta=macro(
                    lambda r: r["numerical_reduction_delta_bce"]
                ),
                sign_agreement=sum(
                    (r["actual_delta_bce"] > 0) == (r["first_order_delta_bce"] > 0)
                    for r in local_rows
                ),
                channel_contributions={
                    name: macro(
                        lambda r, c=c: r["channels"][c]["signed_first_order_delta_bce"]
                    )
                    for c, name in enumerate(io23.RAW23)
                },
            )
    write(output / "rows.json", rows)
    write(output / "summary.json", summary)
    if any(sha(Path(p)) != value for p, value in code.items()):
        raise AssertionError("imported code changed during trace")
    write(
        output / "completed.json",
        dict(
            status="COMPLETED",
            seconds=time.monotonic() - started,
            frozen_models_unchanged=True,
            neural_updates=0,
            suffix340_reads=0,
            artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    print(dict(status="COMPLETED", seconds=time.monotonic() - started), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.pilot, args.output)
