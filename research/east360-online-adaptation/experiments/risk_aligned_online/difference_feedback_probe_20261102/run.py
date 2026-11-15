"""Ended20 local diagnostic; query labels never enter a candidate update."""

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import digest_array, load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure, read
from ..lora_stream_20261102.core import encode, forward, step as centered_step
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .core import propose
from .summary import summarize

REFERENCE = io23.ROOT / "probe_outputs/lora_stream_v1"
REFERENCE_SHA = "76ebb4035873ba3c6b85da9d89b995233d2ce36222237bf1733058d4c9482f4c"
PLAN = io23.ROOT / "DIFFERENCE_FEEDBACK_PROBE_PLAN_20261102.md"
ARMS = ("centered_real", "difference_real", "difference_reverse", "difference_hold")


def select_queries(times, disruptive, down):
    times = np.asarray(times)
    if len(times) <= 160:
        return [], [dict(reason="insufficient_prefix")]
    if not disruptive:
        return [
            dict(query=int(q), requested_lead_ms=None, actual_lead_ms=None)
            for q in np.unique(np.linspace(160, len(times) - 1, 5, dtype=int))
        ], []
    lead = np.round((down - times) * 1000, 9)
    eligible = np.flatnonzero((np.arange(len(times)) >= 160) & (lead > 30))
    selected, skipped, used = [], [], set()
    for requested in (250, 150, 100, 60, 35):
        if not len(eligible):
            skipped.append(
                dict(requested_lead_ms=requested, reason="requested_lead_not_covered")
            )
            continue
        q = int(eligible[np.argmin(np.abs(lead[eligible] - requested))])
        if abs(lead[q] - requested) > 2 or q in used:
            skipped.append(
                dict(requested_lead_ms=requested, reason="requested_lead_not_covered")
            )
            continue
        used.add(q)
        selected.append(
            dict(query=q, requested_lead_ms=requested, actual_lead_ms=float(lead[q]))
        )
    return selected, skipped


def risk_loss(logit, label):
    signed = (1 - 2 * label) * logit.double()
    return torch.logaddexp(torch.zeros_like(signed), signed).mean()


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / norms, -1, 1)) if norms else None


def sample(model, signal, basis, query, label):
    """Labels are used solely on the held-later query, not in inner updates."""
    origin, maturity = query - 26, query - 1
    assert origin >= 134 and maturity + 1 == query
    point = encode(model, signal, [origin], basis)
    later = encode(model, signal, [query], basis)
    truth, available = targets(signal, origin, maturity)
    device = model.base.center.device
    mask = torch.as_tensor(available, device=device)
    values = {
        key: torch.as_tensor(value, device=device) for key, value in truth.items()
    }
    zero = torch.zeros(4, 64, device=device)
    # This separate evaluation graph is never an argument to propose/centered_step.
    variable = zero.detach().requires_grad_(True)
    before_z = forward(model, later, variable)["logit"]
    before_bce = risk_loss(before_z, label)
    task_gradient = (
        torch.autograd.grad(before_bce, variable)[0].detach().cpu().flatten().tolist()
    )
    with torch.no_grad():
        before_forecast = forward(model, point, zero, forecast=True)["forecast"]
        forecast_sha = digest_array(before_forecast.cpu().numpy())
    _, hold_receipt = centered_step(model, point, zero, values["hold"], mask)
    updates = {}
    for arm in ARMS:
        if arm == "centered_real":
            after, receipt = centered_step(model, point, zero, values["real"], mask)
        else:
            mode = arm.removeprefix("difference_")
            after, receipt = propose(
                model, point, zero, values[mode], values["hold"], mask
            )
        with torch.no_grad():
            evaluated = forward(model, later, after)
            after_z = evaluated["logit"]
            after_bce = risk_loss(after_z, label)
        updates[arm] = dict(
            before_logit=float(before_z.detach()),
            after_logit=float(after_z),
            before_probability=float(before_z.detach().sigmoid()),
            after_probability=float(after_z.sigmoid()),
            before_bce=float(before_bce.detach()),
            after_bce=float(after_bce),
            delta_bce=float(after_bce - before_bce.detach()),
            risk_gradient_cosine=cosine(receipt["gradient256"], task_gradient),
            query_action_rms=float(evaluated["rms"].max()),
            receipt=receipt,
        )
    difference = np.asarray(updates["difference_real"]["receipt"]["gradient256"])
    expected = np.asarray(
        updates["centered_real"]["receipt"]["gradient256"]
    ) - np.asarray(hold_receipt["gradient256"])
    gap = float(np.max(np.abs(difference - expected)))
    relative = float(
        np.linalg.norm(difference - expected) / max(np.linalg.norm(expected), 1e-30)
    )
    # Subtracting two float32 gradients can have a large relative error near
    # zero. Use the tested absolute-plus-relative check, not a relative-only
    # assertion that would reject harmless cancellation.
    assert np.allclose(difference, expected, atol=2e-7, rtol=2e-4), (gap, relative)
    assert updates["difference_hold"]["delta_bce"] == 0
    return dict(
        origin=origin,
        target_first=origin + 1,
        target_last=maturity,
        first_affected=query,
        observed=int(mask.sum()),
        mask_sha256=digest_array(available),
        forecast_sha256=forecast_sha,
        target_sha256={key: digest_array(v) for key, v in truth.items()},
        state256_before=[0.0] * 256,
        task_gradient256=task_gradient,
        base_hold_gradient256=hold_receipt["gradient256"],
        difference_identity_max_absolute=gap,
        difference_identity_relative=relative,
        results=updates,
    )


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    assert read(REFERENCE / "verification_records.json")["status"] == "PASS"
    parent_protocol = read(REFERENCE / "protocol.json")
    ids = [r["shot"] for r in parent_protocol["order"][:20]]
    assert len(ids) == len(set(ids)) == 20
    calibration = Path(parent_protocol["calibration_reference"])
    mapping = read(calibration / "input_map.json")
    assert mapping["calibration_ids"] == ids
    hashes = read(calibration / "input_hashes.json")
    input_hashes = {str(sid): sha(io23.path_for("EAST", sid)) for sid in ids}
    assert all(input_hashes[str(sid)] == hashes[str(sid)] for sid in ids)
    model, _, cold = load_model()
    assert sha(cold) == parent_protocol["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    root.mkdir(parents=True)
    torch.save({"basis": basis.cpu()}, root / "source_basis.pt")
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="CONDITIONAL_ENDED20_LOCAL_PROBE_NOT_ONLINE",
        arms=ARMS,
        ids=ids,
        source_checkpoint=parent_protocol["source_checkpoint"],
        source_sha256=parent_protocol["source_sha256"],
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        calibration_reference=str(calibration),
        input_map_sha256=sha(calibration / "input_map.json"),
        basis_sha256=sha(root / "source_basis.pt"),
        input_hashes=input_hashes,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        outcome_authority="HDF5",
        east_horizon_ms=125,
        strict_lead_ms=30,
        label_used_for_update=False,
        query_selection_uses_ended_outcomes=True,
        reset_each_local_sample=True,
        raw_signal_ids_allowed=ids,
        step_size=0.1,
        anchor=0.01,
        radius=0.25,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    rows, shots = [], []
    for index, sid in enumerate(ids, 1):
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        flag, down = io23.read_outcome("EAST", sid)
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        queries, skipped = select_queries(raw["times"], flag, down)
        labels, valid = io23.labels_at(raw["times"], flag, down, 125)
        for selected in queries:
            q = selected["query"]
            assert valid[q]
            row = sample(model, signal, basis, q, int(labels[q]))
            rows.append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    label=int(labels[q]),
                    query_time=float(raw["times"][q]),
                    **selected,
                    **row,
                )
            )
        shots.append(dict(shot=sid, disruptive=flag, selected=queries, skipped=skipped))
        _durable_json(
            root / "progress.json",
            dict(
                completed_shots=index,
                total_shots=20,
                samples=len(rows),
                seconds=time.monotonic() - start,
            ),
        )
        print(
            json.dumps(dict(shot=sid, completed=index, samples=len(rows))), flush=True
        )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    _durable_json(root / "samples.json", rows)
    _durable_json(root / "sampling.json", shots)
    result = dict(
        status="COMPLETED_ENDED20_LOCAL_PROBE",
        n_shots=len(shots),
        n_samples=len(rows),
        raw_signal_ids_read=ids,
        raw_outcome_ids_read=ids,
        summary=summarize(rows),
        samples_sha256=sha(root / "samples.json"),
        sampling_sha256=sha(root / "sampling.json"),
        source_tensors_unchanged=True,
        suffix_raw_signals_read=0,
        online_performance_measured=False,
        independent_future_confirmation=False,
        scientific_goal_achieved=False,
        seconds=time.monotonic() - start,
    )
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
