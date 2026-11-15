"""Only ended-prefix signals/truth may be supplied to these fitting functions."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from .memory import dynamic_key, direct_target
from .model import Injection, levels, predict
from .timeline import trace, window_batch


@torch.no_grad()
def fit_initial(keys, targets, masks, *, ridge=0.01):
    if (
        keys.ndim != 2
        or keys.shape[1] != 70
        or not len(keys)
        or targets.shape != (len(keys), 168)
        or masks.shape != targets.shape
        or masks.dtype != torch.bool
        or not math.isfinite(ridge)
        or ridge <= 0
        or not torch.isfinite(keys).all()
        or not torch.isfinite(targets[masks]).all()
    ):
        raise ValueError(
            "finite observed targets, keys, masks and positive ridge required"
        )
    x = keys.double()
    y = torch.where(masks, targets, 0).double()
    counts = masks.sum(0)
    scale = counts.clamp_min(1).double()
    gram = torch.einsum("no,ni,nj->oij", masks.double(), x, x) / scale[:, None, None]
    rhs = torch.einsum("ni,no->oi", x, y) / scale[:, None]
    eye = torch.eye(70, device=x.device, dtype=x.dtype)
    weight = torch.linalg.solve(gram + ridge * eye, rhs[:, :, None]).squeeze(-1).float()
    # No observations means no fitted dynamical coefficient for that output;
    # this is represented explicitly, never fabricated from another diagnostic.
    weight[counts == 0] = 0
    pred = keys @ weight.T
    error = (pred - torch.where(masks, targets, 0)).square().masked_fill(~masks, 0)
    return weight, dict(
        observed_outputs=int((counts > 0).sum()),
        target_counts=counts.cpu().tolist(),
        fit_mse=float(error.sum() / masks.sum().clamp_min(1)),
        hold_mse=float(y.square().sum() / masks.sum().clamp_min(1)),
        ridge=ridge,
    )


def fit_prefix(base, shots, truth, ids, *, threshold, device, steps=512, rule="lms"):
    if (
        set(shots) != set(ids)
        or set(truth) != set(ids)
        or len(ids) != 20
        or len(set(ids)) != 20
        or not 0 < threshold < 1
    ):
        raise ValueError(
            "exactly the same twenty fully ended shots and fixed threshold required"
        )
    keys, targets, masks = [], [], []
    for sid in ids:
        s = shots[sid]
        queries = list(range(127, len(s["values"]) - 8, 8))
        for lo in range(0, len(queries), 256):
            selected = queries[lo : lo + 256]
            x, a = window_batch(s, selected, device=device)
            future = torch.as_tensor(
                np.stack([s["values"][q + 1 : q + 9] for q in selected]), device=device
            )
            future_mask = torch.as_tensor(
                np.stack([s["available"][q + 1 : q + 9] for q in selected]),
                device=device,
            )
            level = levels(base, x, a)
            target, target_mask = direct_target(
                level, a, levels(base, future, future_mask), future_mask
            )
            keys.append(dynamic_key(level, a))
            targets.append(target)
            masks.append(target_mask)
    if not keys:
        raise ValueError("ended prefix has no complete direct forecast pairs")
    origin, diagnostic = fit_initial(
        torch.cat(keys), torch.cat(targets), torch.cat(masks)
    )
    origin = origin.detach()
    records, signals, summaries, shot_means = [], {}, {}, []
    for sid in ids:
        signal = shots[sid]
        innovations, history = trace(
            base, signal, origin, "real", device=device, rule=rule
        )
        signals[sid] = innovations
        shot_means.append(innovations[135:].mean(0))
        summaries[str(sid)] = history
        flag, down = truth[sid]
        y, valid = io23.labels_at(signal["times"], flag, down, 125.0)
        for q in range(135, len(y), 8):
            if not valid[q]:
                continue
            if flag and y[q] and (down - signal["times"][q]) * 1000 <= 30.0:
                continue
            group = "safe" if not flag else ("positive" if y[q] else "earlier")
            records.append(dict(shot=sid, query=q, y=float(y[q]), group=group))
    pools = {g: {} for g in ("positive", "safe", "earlier")}
    for i, row in enumerate(records):
        pools[row["group"]].setdefault(row["shot"], []).append(i)
    if any(not v for v in pools.values()):
        raise ValueError("twenty-shot prefix lacks an operational training stratum")
    injection = Injection().to(device)
    optimizer = torch.optim.Adam(injection.parameters(), lr=0.01)
    rng, history = np.random.default_rng(131), []
    tau = math.log(threshold / (1 - threshold))
    for step in range(1, steps + 1):
        chosen = []
        for group, count in (("positive", 64), ("safe", 32), ("earlier", 32)):
            members = sorted(pools[group])
            for sid in rng.choice(members, size=count):
                chosen.append(int(rng.choice(pools[group][int(sid)])))
        rng.shuffle(chosen)
        windows, masks_batch, innovation_batch, labels = [], [], [], []
        for i in chosen:
            row = records[i]
            sid, q = row["shot"], row["query"]
            x, a = window_batch(shots[sid], [q], device=device)
            windows.append(x)
            masks_batch.append(a)
            innovation_batch.append(signals[sid][q])
            labels.append(row["y"])
        x, a = torch.cat(windows), torch.cat(masks_batch)
        innovation_batch = torch.stack(innovation_batch)
        y = torch.tensor(labels, device=device)
        optimizer.zero_grad(set_to_none=True)
        z = predict(base, injection, x, a, innovation_batch)
        bce = F.binary_cross_entropy_with_logits(z - tau, y)
        penalty = injection(innovation_batch).square().mean()
        loss = bce + 0.01 * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite internal connection fit")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            injection.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        history.append(
            dict(
                step=step,
                loss=float(loss.detach()),
                margin_bce=float(bce.detach()),
                fast_penalty=float(penalty.detach()),
                gradient_norm=float(norm),
            )
        )
        if step % 128 == 0:
            print(dict(stage="direct_memory_prefix_fit", **history[-1]), flush=True)
    optimizer.zero_grad(set_to_none=True)
    injection.requires_grad_(False).eval()
    return (
        origin,
        injection,
        torch.stack(shot_means).mean(0),
        dict(
            signal_pairs=sum(len(k) for k in keys),
            forecast=diagnostic,
            label_shots=ids,
            source_training_shots=0,
            suffix_training_shots=0,
            training_records=records,
            target_counts={g: sum(map(len, v.values())) for g, v in pools.items()},
            direct_history=summaries,
            connection_history=history,
            steps=steps,
            update_rule=rule,
        ),
    )
