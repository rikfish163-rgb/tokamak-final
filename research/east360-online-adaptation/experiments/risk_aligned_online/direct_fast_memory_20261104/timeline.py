"""Causal signal-only timeline: targets are used only when all eight arrive."""

import numpy as np
import torch

from .. import simple_data
from .memory import dynamic_key, direct_target, fast_step, innovation
from .model import levels
from .rls import initial_covariance, ridge_step


def window_batch(shot, queries, *, device):
    windows = [simple_data.window(shot, int(q), 128) for q in queries]
    if not windows:
        raise ValueError("nonempty causal query batch required")
    return (
        torch.as_tensor(np.stack([w[0] for w in windows]), device=device),
        torch.as_tensor(np.stack([w[1] for w in windows]), device=device),
    )


def all_keys(base, shot, *, device):
    pieces = []
    for lo in range(0, len(shot["values"]), 256):
        x, a = window_batch(
            shot, range(lo, min(lo + 256, len(shot["values"]))), device=device
        )
        pieces.append(dynamic_key(levels(base, x, a), a))
    return torch.cat(pieces)


def feedback(base, shot, origin, now, *, device):
    if not 127 <= origin < now < len(shot["values"]) or origin + 8 > now:
        raise ValueError(
            "complete historical window and eight arrived targets required"
        )
    x, a = window_batch(shot, [origin], device=device)
    actual = torch.as_tensor(
        shot["values"][origin + 1 : origin + 9][None], device=device
    )
    mask = torch.as_tensor(
        shot["available"][origin + 1 : origin + 9][None], device=device
    )
    k = dynamic_key(levels(base, x, a), a)
    target, observed = direct_target(
        levels(base, x, a), a, levels(base, actual, mask), mask
    )
    return k, target, observed


@torch.no_grad()
def trace(base, shot, initial, arm, *, device, keys=None, rule="lms"):
    if arm not in ("real", "hold", "rotated", "discard"):
        raise ValueError("explicit feedback control required")
    if rule not in ("lms", "ridge"):
        raise ValueError("explicit LMS or recursive ridge update required")
    if initial.shape != (168, 70) or not torch.isfinite(initial).all():
        raise ValueError("fixed direct prediction initialization required")
    n = len(shot["values"])
    keys = all_keys(base, shot, device=device) if keys is None else keys
    if keys.shape != (n, 70):
        raise ValueError("one causal diagnostic key per point required")
    initial = initial.to(device)
    origin_weight = initial[None]
    state = origin_weight.clone()
    covariance = initial_covariance(1, state) if rule == "ridge" else None
    result = initial.new_zeros(n, 168)
    rows = []
    for now in range(135, n, 32):
        key, target, observed = feedback(base, shot, now - 8, now, device=device)
        if not torch.equal(key[0], keys[now - 8]):
            raise AssertionError("prediction-time and matured feedback keys differ")
        before = torch.bmm(state, key[:, :, None]).squeeze(-1)
        control, control_mask = target, observed
        if arm == "hold":
            control = torch.zeros_like(target)
        elif arm == "rotated":
            control = target.reshape(1, 8, 21).roll(1, -1).reshape(1, 168)
            control_mask = observed.reshape(1, 8, 21).roll(1, -1).reshape(1, 168)
        if rule == "lms":
            candidate, _ = fast_step(
                state,
                key,
                control,
                control_mask,
                origin=origin_weight,
                lr=0.5,
                radius=2.0,
            )
        else:
            candidate, next_covariance, _ = ridge_step(
                state,
                covariance,
                key,
                control,
                control_mask,
                origin=origin_weight,
                radius=2.0,
            )
            if arm != "discard":
                covariance = next_covariance
        if arm != "discard":
            state = candidate
        after = torch.bmm(state, key[:, :, None]).squeeze(-1)
        denom = observed.sum().clamp_min(1)
        rows.append(
            dict(
                arrival=now,
                origin=now - 8,
                observed=int(observed.sum()),
                pre_update_mse=float(
                    ((before - target).square() * observed).sum() / denom
                ),
                static_mse=float(
                    ((key @ initial.T - target).square() * observed).sum() / denom
                ),
                hold_mse=float((target.square() * observed).sum() / denom),
                post_update_mse=float(
                    ((after - target).square() * observed).sum() / denom
                ),
                weight_displacement=float((state - origin_weight).norm()),
            )
        )
        hi = min(now + 32, n)
        result[now:hi] = innovation(
            state.expand(hi - now, -1, -1),
            origin_weight.expand(hi - now, -1, -1),
            keys[now:hi],
        )
    return result, rows
