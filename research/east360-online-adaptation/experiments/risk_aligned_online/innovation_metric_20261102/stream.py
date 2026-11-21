"""Same seal/feedback clock, no current outcome, no cross-shot W retention."""

import numpy as np
import torch

from ..critical_shot_meta_20261102.stream import take
from ..dual_history_ttt_20261101.run import digest_array
from ..dual_history_ttt_20261101.stream import feedback_origin, targets
from ..lora_stream_20261102.core import encode, forward
from ..risk_bridge_forecaster_20261102.core import auxiliary, propose as contrast
from .core import propose


def update(
    model, point, state, truth, hold, mask, head, geometry, *, create_graph=False
):
    if geometry == "contrast":
        return contrast(
            model, point, state, truth, hold, mask, head, create_graph=create_graph
        )
    return propose(
        model,
        point,
        state,
        truth,
        hold,
        mask,
        head,
        geometry=geometry,
        create_graph=create_graph,
    )


def simulate(model, signal, basis, heads, *, cached=None):
    if set(signal) != {"values", "available"}:
        raise ValueError("signals-only stream required")
    n = len(signal["values"])
    states = {a: model.base.center.new_zeros(4, 64) for a in heads}
    logits, probability = {a: [] for a in heads}, {a: [] for a in heads}
    tapes = {a: [] for a in heads}
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in heads}
    for lo in range(0, n, 32):
        hi = min(lo + 32, n)
        pack = (
            take(cached, lo, hi)
            if cached is not None
            else encode(model, signal, np.arange(lo, hi), basis)
        )
        with torch.no_grad():
            for arm in heads:
                output = forward(model, pack, states[arm])
                logits[arm].append(output["logit"].cpu().numpy())
                probability[arm].append(output["logit"].sigmoid().cpu().numpy())
                actions[arm]["endpoints"] += hi - lo
                actions[arm]["capped_endpoints"] += int(output["capped"].sum())
                actions[arm]["max_rms"] = max(
                    actions[arm]["max_rms"], float(output["rms"].max())
                )
        end = hi - 1
        origin = feedback_origin(end)
        if origin is None:
            continue
        assert lo <= origin <= end
        point = take(pack, origin - lo, origin - lo + 1)
        with torch.no_grad():
            seals = {
                a: digest_array(auxiliary(model, point, states[a], head).cpu().numpy())
                for a, (head, _, _) in heads.items()
                if head is not None
            }
        observed, available = targets(signal, origin, end)
        tensor = {
            k: torch.as_tensor(v, device=model.base.center.device)
            for k, v in observed.items()
        }
        mask = torch.as_tensor(available, device=model.base.center.device)
        for arm, (head, mode, geometry) in heads.items():
            if head is None:
                continue
            before = states[arm].detach().clone()
            states[arm], receipt = update(
                model,
                point,
                states[arm],
                tensor[mode],
                tensor["hold"],
                mask,
                head,
                geometry,
            )
            tapes[arm].append(
                dict(
                    origin=origin,
                    maturity=end,
                    first_affected=end + 1,
                    before=before,
                    after=states[arm].detach().clone(),
                    receipt=receipt,
                    forecast_sha256=seals[arm],
                    mask_sha256=digest_array(available),
                    target_sha256=digest_array(observed[mode]),
                    observed=int(mask.sum()),
                )
            )
    return dict(
        logits={a: np.concatenate(v) for a, v in logits.items()},
        probability={a: np.concatenate(v) for a, v in probability.items()},
        tapes=tapes,
        actions=actions,
    )
