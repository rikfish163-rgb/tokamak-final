"""Same frozen-parent FFN insertion, actual accumulated state and128 clock."""

import numpy as np
import torch

from ..dual_history_ttt_20261101.run import digest_array
from ..dual_history_ttt_20261101.stream import feedback_origin, targets
from ..lora_stream_20261102.core import encode, forward
from ..risk_bridge_forecaster_20261102.core import auxiliary, propose


def take(pack, start, end):
    return {k: v[start:end] for k, v in pack.items()}


@torch.no_grad()
def cache_encoding(model, signal, basis):
    parts = []
    for lo in range(0, len(signal["values"]), 32):
        ends = np.arange(lo, min(lo + 32, len(signal["values"])))
        parts.append(encode(model, signal, ends, basis))
    return {key: torch.cat([p[key] for p in parts]) for key in parts[0]}


def simulate(model, signal, basis, heads, *, cached=None):
    if set(signal) != {"values", "available"}:
        raise ValueError("signals-only stream required")
    n = len(signal["values"])
    states = {arm: torch.zeros(4, 64, device=model.base.center.device) for arm in heads}
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
                result = forward(model, pack, states[arm])
                logits[arm].append(result["logit"].cpu().numpy())
                probability[arm].append(result["logit"].sigmoid().cpu().numpy())
                actions[arm]["endpoints"] += hi - lo
                actions[arm]["capped_endpoints"] += int(result["capped"].sum())
                actions[arm]["max_rms"] = max(
                    actions[arm]["max_rms"], float(result["rms"].max())
                )
        end = hi - 1
        origin = feedback_origin(end)
        if origin is None:
            continue
        assert lo <= origin <= end
        point = take(pack, origin - lo, origin - lo + 1)
        seals = {}
        with torch.no_grad():
            for arm, (head, _) in heads.items():
                if head is not None:
                    seals[arm] = digest_array(
                        auxiliary(model, point, states[arm], head).cpu().numpy()
                    )
        # Risk and the old-origin forecast have now been sealed. Only then is
        # this exactly matured target made available to the update procedure.
        observed, available = targets(signal, origin, end)
        tensor = {
            k: torch.as_tensor(v, device=model.base.center.device)
            for k, v in observed.items()
        }
        mask = torch.as_tensor(available, device=model.base.center.device)
        for arm, (head, mode) in heads.items():
            if head is None:
                continue
            before = states[arm].detach().clone()
            states[arm], receipt = propose(
                model, point, states[arm], tensor[mode], tensor["hold"], mask, head
            )
            # Keep tensors for exact one-update replay. Full-stream export
            # serializes them compactly; no controller sees these future logs.
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
