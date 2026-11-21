"""Fixed-case objective decomposition, not a new online policy evaluation."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.stream import take
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..innovation_metric_20261102.core import innovation_loss, metric_direction
from ..lora_stream_20261102.core import encode
from ..risk_bridge_forecaster_20261102.core import auxiliary
from ..warmstart_alignment_20261022.alignment import transform
from .core import select_step

IDS = (74087, 73403, 63656, 62573)
OLD = io23.ROOT / "probe_outputs/innovation_metric360_v2"
NEW = io23.ROOT / "probe_outputs/full_path_forecast_controls360_v1"


def components(model, pack, state, gradient, truth, mask, coefficient):
    with torch.enable_grad():
        before = state.detach().clone().requires_grad_(True)
        data0 = innovation_loss(
            model, auxiliary(model, pack, before, coefficient), truth, mask
        )
        anchor0 = 0.01 * before.square().mean()
        total0 = data0 + anchor0
    with torch.no_grad():
        direction = metric_direction(pack, gradient)
        proposal = before.detach() + direction
        data1 = innovation_loss(
            model, auxiliary(model, pack, proposal, coefficient), truth, mask
        )
        anchor1 = 0.01 * proposal.square().mean()
        total1 = data1 + anchor1
    return dict(
        proposal=proposal,
        direction=direction,
        current=before,
        data_before_tensor=data0,
        anchor_before_tensor=anchor0,
        data_proposal_tensor=data1,
        anchor_proposal_tensor=anchor1,
        total_before=float(total0.detach()),
        total_proposal=float(total1),
        delta_data=float(data1) - float(data0.detach()),
        delta_anchor=float(anchor1) - float(anchor0.detach()),
    )


def reason_class(accepted, data, anchor):
    if accepted:
        return (
            "accepted_while_data_fit_worsens"
            if data > 0
            else (
                "accepted_with_data_improvement"
                if data < 0
                else "accepted_no_data_change"
            )
        )
    if data < 0 and anchor > 0:
        return "anchor_overrides_data_improvement"
    if data > 0 and anchor > 0:
        return "both_terms_increase"
    if data > 0:
        return "data_term_increase"
    return "flat_or_roundoff"


def main():
    # v1 retains the original exact reconstruction. v2 additionally records the
    # selected counterfactual's data/anchor split; no scientific setting changes.
    root = io23.ROOT / "probe_outputs/armijo_rejection_audit_v2"
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    model, _, cold = load_model()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    protocols = {r: read(r / "protocol.json") for r in (OLD, NEW)}
    for r in protocols:
        done = read(r / "completed.json")
        assert (
            done["status"] == "COMPLETED"
            and read(r / "verification_records.json")["status"] == "PASS"
        )
        assert read(r / "verification_neural.json")["status"] == "PASS"
        assert sha(cold) == protocols[r]["cold_state_sha256"]
    calibration = Path(protocols[NEW]["calibration_reference"])
    mapping, hashes = (
        read(calibration / "input_map.json"),
        read(calibration / "input_hashes.json"),
    )
    tables0 = read(io23.ROOT / "probe_outputs/innovation_metric_heads20_v1/heads.json")
    tables1 = read(io23.ROOT / "probe_outputs/full_path_heads20_v1/heads.json")
    heads = {
        name: torch.tensor(value, dtype=torch.float32, device="cuda")
        for name, value in dict(
            truncated=tables0["metric_operational"],
            forecast=tables0["metric_forecast"],
            full=tables1["full_operational"],
        ).items()
    }
    specs = dict(
        truncated=(OLD, "innovation_metric"),
        forecast=(NEW, "forecast_real"),
        full=(NEW, "full_real"),
    )
    order = [r["shot"] for r in protocols[NEW]["order"]]
    basis = source_basis(model)
    rows = []
    for sid in IDS:
        index = order.index(sid) + 1
        stem = f"{index:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        saved = {}
        for name, (parent, arm) in specs.items():
            seal = read(parent / "shots" / (stem + "_seal.json"))
            path = parent / "shots" / (stem + "_states.npz")
            metadata = parent / "shots" / (stem + "_updates.json")
            assert (
                sha(path) == seal["states_sha256"]
                and sha(metadata) == seal["updates_sha256"]
            )
            with np.load(path, allow_pickle=False) as archive:
                saved[name] = dict(
                    before=archive[arm + "_before"], gradient=archive[arm + "_gradient"]
                )
            saved[name]["events"] = read(metadata)["updates"][arm]
        for event_index, event in enumerate(saved["full"]["events"]):
            origin, end = event["origin"], event["maturity"]
            lo = origin // 32 * 32
            pack = encode(
                model, signal, np.arange(lo, min(lo + 32, len(raw["times"]))), basis
            )
            point = take(pack, origin - lo, origin - lo + 1)
            observed, available = targets(signal, origin, end)
            truth, mask = [
                torch.as_tensor(v, device="cuda") for v in (observed["real"], available)
            ]
            for name in specs:
                receipt = saved[name]["events"][event_index]
                assert (
                    receipt["origin"] == origin
                    and receipt["maturity"] == end
                    and receipt["observed"] == int(mask.sum())
                )
                assert bool(mask.any()) and receipt["gradient_computed"]
                state = torch.as_tensor(
                    saved[name]["before"][event_index], device="cuda"
                )
                gradient = torch.as_tensor(
                    saved[name]["gradient"][event_index].reshape(4, 64), device="cuda"
                )
                parts = components(
                    model, point, state, gradient, truth, mask, heads[name]
                )
                assert parts["total_before"] == receipt["loss_before"], (
                    sid,
                    name,
                    origin,
                    "before objective",
                )
                assert parts["total_proposal"] == receipt["loss_proposal"], (
                    sid,
                    name,
                    origin,
                    "proposal objective",
                )

                def evaluate(candidate):
                    return (
                        innovation_loss(
                            model,
                            auxiliary(model, point, candidate, heads[name]),
                            truth,
                            mask,
                        )
                        + 0.01 * candidate.square().mean()
                    )

                selected, counterfactual = select_step(
                    parts["current"],
                    gradient,
                    parts["direction"],
                    parts["data_before_tensor"] + parts["anchor_before_tensor"],
                    evaluate,
                )
                with torch.no_grad():
                    selected_data = innovation_loss(
                        model,
                        auxiliary(model, point, selected, heads[name]),
                        truth,
                        mask,
                    )
                    selected_anchor = 0.01 * selected.square().mean()
                assert (
                    float(selected_data + selected_anchor)
                    == counterfactual["loss_after"]
                )
                rows.append(
                    dict(
                        shot=sid,
                        arm=name,
                        maturity=end,
                        accepted=receipt["accepted"],
                        data_delta=parts["delta_data"],
                        anchor_delta=parts["delta_anchor"],
                        total_before=parts["total_before"],
                        total_proposal=parts["total_proposal"],
                        direction_parameter_norm=float(
                            parts["direction"].double().norm()
                        ),
                        reason_class=reason_class(
                            receipt["accepted"],
                            parts["delta_data"],
                            parts["delta_anchor"],
                        ),
                        counterfactual_armijo_accepted=counterfactual["accepted"],
                        counterfactual_alpha=counterfactual["alpha"],
                        counterfactual_evaluations=counterfactual["evaluations"],
                        counterfactual_total_after=counterfactual["loss_after"],
                        counterfactual_data_delta=float(selected_data)
                        - float(parts["data_before_tensor"].detach()),
                        counterfactual_anchor_delta=float(selected_anchor)
                        - float(parts["anchor_before_tensor"].detach()),
                    )
                )
        print(json.dumps(dict(checked_shot=sid, rows=len(rows))), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    summaries = {}
    for name in specs:
        group = [r for r in rows if r["arm"] == name]
        labels = sorted({r["reason_class"] for r in group})
        summaries[name] = dict(
            events=len(group),
            old_accepted=sum(r["accepted"] for r in group),
            reason_counts={
                label: sum(r["reason_class"] == label for r in group)
                for label in labels
            },
            counterfactual_accepted=sum(
                r["counterfactual_armijo_accepted"] for r in group
            ),
            recovered_old_rejections=sum(
                not r["accepted"] and r["counterfactual_armijo_accepted"] for r in group
            ),
            newly_rejected_old_acceptances=sum(
                r["accepted"] and not r["counterfactual_armijo_accepted"] for r in group
            ),
            counterfactual_proposal_evaluations=sum(
                r["counterfactual_evaluations"] for r in group
            ),
            counterfactual_accepted_data_fit_worsens=sum(
                r["counterfactual_armijo_accepted"]
                and r["counterfactual_data_delta"] > 0
                for r in group
            ),
        )
    result = dict(
        status="PASS_FIXED_CASE_COUNTERFACTUAL",
        summaries=summaries,
        per_event=rows,
        raw_signal_shots_read=list(IDS),
        outcome_reads=0,
        source_tensors_unchanged=True,
        original_objective_components_and_proposals_exact=True,
        parent_completed_sha256={str(r): sha(r / "completed.json") for r in (OLD, NEW)},
        auditor_sha256=sha(Path(__file__)),
        seconds=time.monotonic() - started,
        limitations=[
            "Four fixed cases, not population-wide rejection attribution.",
            "Each line-search starts from the OLD recorded state independently; accepted counterfactuals are NOT propagated.",
            "No new full-shot trajectory, TPR/FPR or risk gain is measured here.",
            "Literal floating-point data/anchor comparisons do not imply a material change or a physical mechanism.",
        ],
    )
    root.mkdir(parents=True)
    write(root / "completed.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"], summaries=summaries, seconds=result["seconds"]
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
