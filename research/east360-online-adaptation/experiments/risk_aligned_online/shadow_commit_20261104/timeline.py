"""Issue forecasts before looking at their targets; learn only after release."""

from dataclasses import dataclass

import numpy as np
import torch

from ..direct_fast_memory_20261104.memory import innovation
from ..direct_fast_memory_20261104.rls import initial_covariance, ridge_step
from ..direct_fast_memory_20261104.timeline import all_keys, feedback
from .decision import choose_commit


@dataclass(frozen=True)
class Pending:
    origin: int
    candidate_version: int
    active_version: int
    candidate_weight: torch.Tensor
    candidate_prediction: torch.Tensor
    active_prediction: torch.Tensor


@torch.no_grad()
def issue(shadow, active, key, *, origin, candidate_version, active_version):
    """Copies prevent later learning from silently changing an old forecast."""
    return Pending(
        origin,
        candidate_version,
        active_version,
        shadow.detach().clone(),
        torch.bmm(shadow, key[:, :, None]).squeeze(-1).detach().clone(),
        torch.bmm(active, key[:, :, None]).squeeze(-1).detach().clone(),
    )


@torch.no_grad()
def trace(base, shot, initial, arm, *, policy, device, keys=None):
    if arm not in ("real", "hold", "reverse") or policy not in ("guard", "delay"):
        raise ValueError("explicit feedback arm and commit policy required")
    if initial.shape != (168, 70) or not torch.isfinite(initial).all():
        raise ValueError("same finite W0 required")
    n = len(shot["values"])
    keys = all_keys(base, shot, device=device) if keys is None else keys
    if keys.shape != (n, 70):
        raise ValueError("one causal key per point required")
    initial = initial.to(device)
    reference = initial[None]
    shadow, active = reference.clone(), reference.clone()
    covariance = initial_covariance(1, shadow)
    result = initial.new_zeros(n, 168)
    history, forecasts = [], []
    shadow_version = active_version = 0
    for origin in range(127, n, 32):
        # At this event only key[origin] and previous weights are consulted.
        pending = issue(
            shadow,
            active,
            keys[origin : origin + 1],
            origin=origin,
            candidate_version=shadow_version,
            active_version=active_version,
        )
        saved = dict(
            issued_at=origin,
            released_at=-1,
            candidate_version=shadow_version,
            active_version=active_version,
            candidate_prediction=pending.candidate_prediction[0].cpu().numpy().copy(),
            active_prediction=pending.active_prediction[0].cpu().numpy().copy(),
            target=np.zeros(168, np.float32),
            target_mask=np.zeros(168, bool),
        )
        forecasts.append(saved)
        arrival = origin + 8
        if arrival >= n:
            continue  # Keep issued/pending evidence, without fabricating truth.
        key, target, observed = feedback(base, shot, origin, arrival, device=device)
        if not torch.equal(key[0], keys[origin]) or not torch.equal(
            shadow, pending.candidate_weight
        ):
            raise AssertionError("issued key or candidate changed before release")
        accepted, candidate_mse, active_mse, support = choose_commit(
            pending.candidate_prediction,
            pending.active_prediction,
            target,
            observed,
            policy=policy,
        )
        previous_active = active
        if bool(accepted[0]):
            active = pending.candidate_weight.clone()
            active_version = pending.candidate_version
        parameter_changed = not torch.equal(active, previous_active)
        control = (
            torch.zeros_like(target)
            if arm == "hold"
            else (-target if arm == "reverse" else target)
        )
        previous_shadow = shadow
        shadow, covariance, _ = ridge_step(
            shadow, covariance, key, control, observed, origin=reference, radius=2.0
        )
        shadow_step = float((shadow - previous_shadow).norm())
        if bool(observed.any()):
            shadow_version += 1
        count = observed.sum().clamp_min(1)
        static = key @ initial.T
        history.append(
            dict(
                origin=origin,
                arrival=arrival,
                candidate_version=pending.candidate_version,
                active_version_at_issue=pending.active_version,
                accepted=bool(accepted[0]),
                parameter_changed=parameter_changed,
                support=int(support[0]),
                candidate_mse=float(candidate_mse[0]),
                active_mse=float(active_mse[0]),
                static_mse=float(
                    ((static - target).double().square() * observed).sum() / count
                ),
                hold_mse=float((target.double().square() * observed).sum() / count),
                active_displacement=float((active - reference).norm()),
                shadow_displacement=float((shadow - reference).norm()),
                shadow_step_norm=shadow_step,
            )
        )
        saved.update(
            released_at=arrival,
            target=target[0].cpu().numpy().copy(),
            target_mask=observed[0].cpu().numpy().copy(),
        )
        stop = min(arrival + 32, n)
        # Use a fixed-size readout even at record termination. CPU bmm can
        # select a different summation kernel for a one-row terminal batch;
        # zero padding here keeps an existing prefix bitwise independent of
        # whether later samples arrive. Padded outputs are never published.
        readout_keys = initial.new_zeros(32, 70)
        readout_keys[: stop - arrival] = keys[arrival:stop]
        block = innovation(
            active.expand(32, -1, -1), reference.expand(32, -1, -1), readout_keys
        )
        result[arrival:stop] = block[: stop - arrival]
    receipt = {}
    for name in ("issued_at", "released_at", "candidate_version", "active_version"):
        receipt[name] = np.array([r[name] for r in forecasts], dtype=np.int64)
    for name in ("candidate_prediction", "active_prediction", "target", "target_mask"):
        receipt[name] = (
            np.stack([r[name] for r in forecasts])
            if forecasts
            else np.empty((0, 168), dtype=bool if name == "target_mask" else np.float32)
        )
    return result, history, receipt
