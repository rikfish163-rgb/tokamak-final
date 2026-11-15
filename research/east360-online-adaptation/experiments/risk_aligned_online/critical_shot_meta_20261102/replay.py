"""One-update truncated meta derivative at an ended shot's critical endpoint."""

import torch

from ..centered_gate_stream_20261101.model import loss as shape_loss
from ..dual_history_ttt_20261101.stream import targets
from ..lora_stream_20261102.core import forward
from ..risk_bridge_forecaster_20261102.core import auxiliary, propose
from .objectives import last_update, margin_loss, select_critical, task_label
from .stream import take


def replay_query(model, example, simulation, arm, coefficient, query):
    maturity = last_update(query)
    if maturity is None:
        return None
    tape = next(t for t in simulation["tapes"][arm] if t["maturity"] == maturity)
    assert tape["maturity"] < query and tape["first_affected"] <= query
    truth, available = targets(example["signal"], tape["origin"], maturity)
    device = model.base.center.device
    real, hold, mask = [
        torch.as_tensor(v, device=device)
        for v in (truth["real"], truth["hold"], available)
    ]
    point = take(example["pack"], tape["origin"], tape["origin"] + 1)
    state, receipt = propose(
        model, point, tape["before"], real, hold, mask, coefficient, create_graph=True
    )
    assert torch.equal(state.detach(), tape["after"]), ("update replay mismatch", query)
    lo = query // 32 * 32
    later = take(example["pack"], lo, min(lo + 32, len(example["times"])))
    after = forward(model, later, state)["logit"][query - lo]
    assert float(after.detach()) == float(simulation["logits"][arm][query]), (
        "risk replay mismatch",
        query,
    )
    with torch.no_grad():
        before = forward(model, later, tape["before"])["logit"][query - lo]
    signal = shape_loss(
        model, auxiliary(model, point, tape["before"], coefficient), real, mask
    )
    return dict(
        after=after,
        before=before,
        signal=signal,
        receipt=receipt,
        maturity=maturity,
        origin=tape["origin"],
    )


def episode(model, example, simulation, arm, coefficient, threshold):
    query = select_critical(
        example["times"],
        simulation["logits"][arm],
        example["disruptive"],
        example["down"],
    )
    result = replay_query(model, example, simulation, arm, coefficient, query)
    supported = result is not None
    if not supported:
        # The global maximum is not replaced by a more convenient adaptable
        # point. Its frozen risk contributes no current meta derivative.
        value = coefficient.new_tensor(float(simulation["logits"][arm][query]))
        before = after = value
        if simulation["tapes"][arm]:
            tape = simulation["tapes"][arm][0]
            truth, available = targets(
                example["signal"], tape["origin"], tape["maturity"]
            )
            device = model.base.center.device
            real, mask = [
                torch.as_tensor(v, device=device) for v in (truth["real"], available)
            ]
            point = take(example["pack"], tape["origin"], tape["origin"] + 1)
            signal = shape_loss(
                model, auxiliary(model, point, tape["before"], coefficient), real, mask
            )
        else:
            signal = coefficient.sum() * 0
    else:
        before, after, signal = result["before"], result["after"], result["signal"]
    label = task_label(
        example["times"],
        query,
        example["disruptive"],
        example["down"],
        "point" if arm == "point" else "operational",
    )
    risk_after, risk_before = (
        margin_loss(after, threshold, label),
        margin_loss(before, threshold, label).detach(),
    )
    regularizer = 1e-4 * coefficient.square().mean()
    objective = (
        (signal + regularizer)
        if arm == "forecast"
        else (risk_after - risk_before) / 0.1 + 0.20 * signal + regularizer
    )
    return objective, dict(
        shot=example["shot"],
        query=query,
        supported=supported,
        maturity=result["maturity"] if supported else None,
        label=label,
        risk_before=float(risk_before),
        risk_after=float(risk_after.detach()),
        signal=float(signal.detach()),
    )
