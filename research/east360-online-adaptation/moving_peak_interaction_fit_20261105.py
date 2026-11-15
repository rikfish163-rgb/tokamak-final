"""Matched, fixed-window interaction fit on the revealed EAST rank-1--20 history.

This module only fits and freezes the two rules. The caller owns the online
prediction/reveal boundary and must supply the already-revealed history.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
import moving_peak_replay_20261105 as original
from moving_peak_component_audit_20261105 import component_actions
import trajectory_policy_20261105 as policy


# Bind to the historical artifact itself, not an identically named new receipt.
FIXED_RECEIPT_SHA256 = (
    "5f671797588eacae6f67d1df4a4ea7e8e3494c722f27ba80ec2dbfa65b701477"
)
INITIAL_FIT_SHA256 = "ad52728e43623c813a15525a23ab084b4345da64ed67168610acea6c9c883d4e"
ARMS = {"interaction_control": 0.0, "interaction_penalty": 1.0}
WINDOWS_PER_SHOT = 8
ENERGY_COEFFICIENT = 0.005
GRADIENT_CLIP = 5.0


def _routed_logits(base, x, a, delta):
    """Full, layer-0, layer-1, zero; retained actions are never renormalized."""
    actions = component_actions(delta)
    routes = latent.routes_all(a)
    z11 = policy.routed_pair(base, x, a, routes, actions["full"])
    z10 = policy.routed_pair(base, x, a, routes, actions["layer0"])
    z01 = policy.routed_pair(base, x, a, routes, actions["layer1"])
    with torch.no_grad():
        z00 = policy.routed_pair(base, x, a, routes, actions["zero"])
    if any(not torch.isfinite(z).all() for z in (z11, z10, z01, z00)):
        raise FloatingPointError("nonfinite routed interaction logits")
    return z11, z10, z01, z00


def _interaction_penalty(z11, z10, z01, z00, weights, safe):
    if (
        z11.ndim != 1
        or any(v.shape != z11.shape for v in (z10, z01, z00, weights, safe))
        or safe.dtype != torch.bool
    ):
        raise ValueError("aligned logits, weights and safe-example mask required")
    if any(not torch.isfinite(v).all() for v in (z11, z10, z01, z00, weights)):
        raise FloatingPointError("nonfinite interaction inputs")
    if (weights < 0).any():
        raise ValueError("interaction weights must be nonnegative")
    interaction = z11 - z10 - z01 + z00
    if not torch.isfinite(interaction).all():
        raise FloatingPointError("nonfinite routed interaction")
    value = (weights * safe.to(weights.dtype) * F.relu(interaction).square()).sum()
    if not torch.isfinite(value):
        raise FloatingPointError("nonfinite interaction penalty")
    return value


def _fixed_selections(history, initial_fit, fixed_receipt, device):
    """Validate the archived first-20 selections without mining any new curves."""
    data = Path(fixed_receipt).read_bytes()
    fixed_sha = hashlib.sha256(data).hexdigest()
    if fixed_sha != FIXED_RECEIPT_SHA256:
        raise ValueError("fixed historical receipt digest changed")
    receipt = json.loads(data)
    initial_sha = original.sha(initial_fit)
    if (
        initial_sha != INITIAL_FIT_SHA256
        or receipt["initial_fit_sha256"] != initial_sha
    ):
        raise ValueError("historical initial fit digest changed")
    if (
        len(history) != original.BUDGET
        or len(receipt["ids"]) != original.BUDGET
        or len(set(receipt["ids"])) != original.BUDGET
        or [row[0] for row in history] != receipt["ids"]
    ):
        raise ValueError("the exact chronological rank-1--20 history is required")
    if any(not isinstance(row[3], (bool, np.bool_)) for row in history):
        raise ValueError("revealed HDF5 outcomes must be boolean")
    if sum(bool(row[3]) for row in history) != 8:
        raise ValueError("exactly eight disruptive and twelve safe shots required")

    archived = receipt["arms"]["moving_peak"]
    rounds = archived["rounds"]
    if (
        archived["steps"] != 36
        or len(rounds) != original.ROUNDS
        or [rd["round"] for rd in rounds] != [1, 2, 3]
        or any(rd["steps"] != 12 for rd in rounds)
    ):
        raise ValueError("the historical three-round optimization receipt is required")
    expected_keys = {str(row[0]) for row in history}
    if any(set(rd["selection"]) != expected_keys for rd in rounds):
        raise ValueError(
            "each archived selection must contain exactly the historical IDs"
        )

    selected = []
    for rd in rounds:
        selection = {}
        for sid, ctx, times, flag, down in history:
            if len(ctx) != 7:
                raise ValueError(f"invalid native context: {sid}")
            x, a, p, _, state, past, _ = ctx
            times, p = np.asarray(times), np.asarray(p)
            n = len(times)
            anchors = (n - policy.FIRST + policy.STRIDE - 1) // policy.STRIDE
            if (
                times.ndim != 1
                or p.shape != times.shape
                or n < policy.FIRST + WINDOWS_PER_SHOT
                or not np.isfinite(times).all()
                or (np.diff(times) <= 0).any()
                or not np.isfinite(p).all()
                or ((p < 0) | (p > 1)).any()
                or x.shape != (n, 128, 23)
                or a.shape != x.shape
                or a.dtype != torch.bool
                or state.shape != (anchors, 66)
                or past.shape != (anchors, 128)
                or any(v.device != device for v in (x, a, state, past))
            ):
                raise ValueError(f"invalid clock or native fitting context: {sid}")
            if flag and (down is None or not np.isfinite(down)):
                raise ValueError(f"disruptive shot needs a finite HDF5 DownTime: {sid}")
            q_raw = rd["selection"][str(sid)]
            if (
                not isinstance(q_raw, list)
                or len(q_raw) != WINDOWS_PER_SHOT
                or any(type(q) is not int for q in q_raw)
            ):
                raise ValueError(f"eight integer archived windows required: {sid}")
            q = np.asarray(q_raw, dtype=np.int64)
            if (np.diff(q) <= 0).any() or q[0] < policy.FIRST or q[-1] >= n:
                raise ValueError(f"invalid archived window indices: {sid}")
            if flag:
                lead = np.round((down - times[q]) * 1000, 9)
                expected, _ = latent.labelled_queries(times, p, True, down)
                if ((lead <= 30) | (lead > 125)).any() or not np.array_equal(
                    q, expected
                ):
                    raise ValueError(
                        f"disruptive windows violate the fixed 30--125 ms rule: {sid}"
                    )
            else:
                cells = np.array_split(np.arange(policy.FIRST, n), WINDOWS_PER_SHOT)
                if any(
                    point < cell[0] or point > cell[-1]
                    for point, cell in zip(q, cells, strict=True)
                ):
                    raise ValueError(
                        f"safe windows must have one point per historical time cell: {sid}"
                    )
            idx = torch.as_tensor(q, device=device)
            k = (idx - policy.FIRST) // policy.STRIDE
            if (
                not torch.isfinite(state[k]).all()
                or not torch.isfinite(past[k]).all()
                or not torch.isfinite(x[idx][a[idx]]).all()
            ):
                raise FloatingPointError(f"nonfinite selected fitting input: {sid}")
            selection[str(sid)] = q.tolist()
        selected.append(dict(round=rd["round"], selection=selection))
    return selected, initial_sha, fixed_sha


def _pack(history, selection):
    parts = []
    for sid, ctx, _, flag, _ in history:
        x, a, _, _, state, past, _ = ctx
        idx = torch.as_tensor(selection[str(sid)], device=x.device)
        k = (idx - policy.FIRST) // policy.STRIDE
        parts.append(
            (
                x[idx],
                a[idx],
                torch.cat((state[k], past[k]), dim=1),
                np.full(len(idx), sid),
                np.full(len(idx), flag, dtype=bool),
            )
        )
    xx, aa, features = (torch.cat([v[j] for v in parts]) for j in range(3))
    ids, positive = (np.concatenate([v[j] for v in parts]) for j in (3, 4))
    w = torch.as_tensor(
        latent.weights_by_class_shot(ids, positive), device=xx.device, dtype=xx.dtype
    )
    signs = torch.as_tensor(
        np.where(positive, -1.0, 1.0), device=xx.device, dtype=xx.dtype
    )
    safe = torch.as_tensor(~positive, device=xx.device)
    if (
        len(xx) != 160
        or not torch.isfinite(w).all()
        or not torch.isfinite(features).all()
    ):
        raise ValueError("160 finite, class-and-shot weighted windows required")
    return xx, aa, features, w, signs, safe


def _round(base, rule, optimizer, packed, orders, coefficient):
    xx, aa, features, weights, signs, safe = packed
    threshold_logit = float(np.log(original.THRESHOLD / (1 - original.THRESHOLD)))

    def objective(idx):
        delta = rule(features[idx])
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite temporary action")
        z11, z10, z01, z00 = _routed_logits(base, xx[idx], aa[idx], delta)
        regularizer = ENERGY_COEFFICIENT * delta.square().sum(dim=1) / latent.RADIUS**2
        # Preserve the original per-example expression and sum exactly.
        matched = (
            weights[idx]
            * (F.softplus(signs[idx] * (z11 - threshold_logit)) + regularizer)
        ).sum()
        penalty = _interaction_penalty(z11, z10, z01, z00, weights[idx], safe[idx])
        # The control still evaluates all four logits, but its loss graph is
        # exactly the original graph rather than an extra zero-gradient path.
        total = matched if coefficient == 0.0 else matched + coefficient * penalty
        if not torch.isfinite(matched) or not torch.isfinite(total):
            raise FloatingPointError("nonfinite matched objective")
        return matched, penalty, total

    def full_objective():
        totals = np.zeros(3, dtype=np.float64)
        with torch.no_grad():
            for idx in torch.arange(len(xx), device=xx.device).split(latent.MINIBATCH):
                totals += [float(v) for v in objective(idx)]
        if not np.isfinite(totals).all():
            raise FloatingPointError("nonfinite full fitting objective")
        return dict(
            matched=float(totals[0]),
            interaction_penalty=float(totals[1]),
            total=float(totals[2]),
        )

    initial = full_objective()
    steps = 0
    rule.requires_grad_(True)
    for order in orders:
        for idx in order.to(xx.device).split(latent.MINIBATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(idx)[2] * len(xx) / len(idx)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite matched extra optimization")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                rule.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(v).all() for v in rule.state_dict().values()):
                raise FloatingPointError("nonfinite rule after optimizer step")
            steps += 1
    rule.requires_grad_(False)
    if steps != 12:
        raise ValueError("exactly twelve optimization steps per round required")
    return dict(steps=steps, initial=initial, final=full_objective())


def fit_interaction(base, history, output, initial_fit, fixed_receipt):
    """Fit two frozen rules on the same three pre-bound revealed selections."""
    if (
        original.BUDGET,
        original.ROUNDS,
        original.EPOCHS_PER_ROUND,
        original.SEED,
        latent.MINIBATCH,
        latent.LR,
        latent.RADIUS,
        policy.FIRST,
        policy.STRIDE,
        original.THRESHOLD,
    ) != (20, 3, 4, 131, 64, 0.002, 2.0, 135, 32, 0.7258946895599365):
        raise ValueError("historical moving-peak optimization constants changed")
    if base.training or any(p.requires_grad for p in base.parameters()):
        raise ValueError("a frozen evaluation-mode source model is required")
    if any(not torch.isfinite(v).all() for v in (*base.parameters(), *base.buffers())):
        raise FloatingPointError("nonfinite frozen source model")
    output = Path(output)
    if not output.is_dir():
        raise ValueError("the caller must create the output directory")
    paths = [output / f"fit_{arm}.npz" for arm in ARMS] + [output / "fit_receipt.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite an existing fit artifact")
    device = base.center.device
    selections, initial_sha, fixed_sha = _fixed_selections(
        history, initial_fit, fixed_receipt, device
    )
    packed = [_pack(history, rd["selection"]) for rd in selections]
    batches = original.shared_orders(160)
    if (
        len(batches) != 3
        or any(len(orders) != 4 for orders in batches)
        or any(
            not torch.equal(order.sort().values, torch.arange(160))
            for orders in batches
            for order in orders
        )
    ):
        raise ValueError("invalid shared deterministic fitting orders")
    rules = {
        arm: original.load_rule(initial_fit, device, namespace="history.")
        for arm in ARMS
    }
    for name, value in rules["interaction_control"].state_dict().items():
        if (
            not torch.equal(value, rules["interaction_penalty"].state_dict()[name])
            or not torch.isfinite(value).all()
        ):
            raise ValueError("identical finite initial rule parameters required")
    optimizers = {
        arm: torch.optim.Adam(rule.parameters(), lr=latent.LR)
        for arm, rule in rules.items()
    }
    report = {
        arm: dict(coefficient=coefficient, rounds=[])
        for arm, coefficient in ARMS.items()
    }
    for stage in range(original.ROUNDS):
        for arm, coefficient in ARMS.items():
            result = _round(
                base,
                rules[arm],
                optimizers[arm],
                packed[stage],
                batches[stage],
                coefficient,
            )
            report[arm]["rounds"].append(dict(round=stage + 1, **result))
    hashes = {}
    for arm, rule in rules.items():
        path = output / f"fit_{arm}.npz"
        original.save_arrays(
            path, **{k: v.detach().cpu().numpy() for k, v in rule.state_dict().items()}
        )
        hashes[arm] = original.sha(path)
        report[arm].update(
            steps=sum(rd["steps"] for rd in report[arm]["rounds"]),
            fit_sha256=hashes[arm],
        )
        if report[arm]["steps"] != 36:
            raise ValueError("exactly 36 optimization steps per arm required")
    original.write_new(
        output / "fit_receipt.json",
        dict(
            ids=[row[0] for row in history],
            initial_fit_sha256=initial_sha,
            fixed_receipt_sha256=fixed_sha,
            fixed_receipt=str(Path(fixed_receipt).resolve()),
            selection_source="archived moving_peak, after all rank-1--20 reveals",
            selections=selections,
            arms=report,
            constants=dict(
                rounds=3,
                epochs_per_round=4,
                steps_per_round=12,
                steps_per_arm=36,
                seed=original.SEED,
                optimizer="Adam",
                learning_rate=latent.LR,
                adam_betas=[0.9, 0.999],
                adam_epsilon=1e-8,
                adam_weight_decay=0.0,
                adam_amsgrad=False,
                minibatch=latent.MINIBATCH,
                gradient_clip=GRADIENT_CLIP,
                threshold=original.THRESHOLD,
                action_radius=latent.RADIUS,
                energy_coefficient=ENERGY_COEFFICIENT,
                interaction_coefficient=1.0,
                initial_namespace="history.",
                interaction="SAFE * relu(z11 - z10 - z01 + z00)^2",
                first=policy.FIRST,
                stride=policy.STRIDE,
                positive_lead_ms="30 < lead <= 125",
                windows_per_shot=WINDOWS_PER_SHOT,
            ),
            weighting=dict(
                examples_per_round=160,
                disruptive_shots=8,
                safe_shots=12,
                class_weight=0.5,
                disruptive_window_weight=0.5 / (8 * 8),
                safe_window_weight=0.5 / (12 * 8),
            ),
        ),
    )
    return rules, hashes
