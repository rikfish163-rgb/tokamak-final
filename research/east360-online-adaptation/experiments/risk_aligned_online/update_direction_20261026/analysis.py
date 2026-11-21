"""Signal-driven writes and their oracle, posthoc operational-loss direction.

Outcome-dependent endpoint selection is intentionally separate from the
signal-only state reconstruction. None of this module is an online policy.
"""
import numpy as np
import torch


def select_endpoints(times, scores, disruptive, down):
    times, scores = np.asarray(times), np.asarray(scores)
    if (times.ndim != 1 or scores.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(scores).all()):
        raise ValueError('finite paired time/probability arrays required')
    if disruptive and (down is None or not np.isfinite(down)):
        raise ValueError('disruptive shot requires a finite disruption time')
    lead = np.round((down-times)*1000, 9) if disruptive else None
    primary = lead > 30 if disruptive else np.ones(len(times), bool)
    masks = [('primary', primary)]
    if disruptive:
        masks.append(('late125', primary & (lead <= 125)))
    selected = {}
    for role, mask in masks:
        indices = np.flatnonzero(mask)
        if len(indices):
            end = int(indices[np.argmax(scores[indices])])
            selected.setdefault(end, []).append(role)
    return [{'end': end, 'roles': roles} for end, roles in sorted(selected.items())]


@torch.no_grad()
def selected_states(memory, means, complete, selected):
    """Reconstruct only main-arm states; no label, outcome, or policy input.

    Deliberately separate from production trace: tests and the runner compare
    its exact reads to the sealed production implementation. This diagnostic
    stops at the latest requested patch and retains only requested states.
    """
    if (means.ndim != 2 or means.shape[1] != 23 or complete.shape != means.shape
            or complete.dtype != torch.bool):
        raise ValueError('observed flat23 patches required')
    selected = set(selected)
    if any(not isinstance(j, int) or j < 0 or j >= len(means) for j in selected):
        raise ValueError('selected patch outside observed range')
    if not selected:
        return {}
    n = max(selected)+1
    # The linear key is pointwise in time. Keep the production GEMM batch shape
    # for exact CUDA reproduction; only keys[:n] can enter the recurrence below.
    # A truncated GEMM can change rounding despite mathematical equivalence.
    features = torch.cat((means*complete, complete.to(means)), -1)
    keys = torch.cat((memory.key(features).tanh(), means.new_ones(len(means), 1)), -1)
    state = memory.prior
    result = {}
    for j in range(n):
        before = state
        target = means.new_zeros(4, 23)
        mask = complete.new_zeros(4, 23)
        origin_keys = keys.new_zeros(4, 9)
        if j:
            count = min(j, 4)
            origins = torch.arange(j-1, j-count-1, -1, device=means.device)
            target[:count] = means[j]-means[origins]
            mask[:count] = complete[j] & complete[origins]
            origin_keys[:count] = keys[origins]
            if mask.any():
                columns = state.reshape(9, 4, 23).permute(1, 0, 2)
                error = ((origin_keys[:, :, None]*columns).sum(1)-target)*mask
                grad = (origin_keys[:, :, None]*error[:, None, :]
                        / (1+origin_keys.square().sum(-1))[:, None, None])
                proposal = state-memory.eta*grad.permute(1, 0, 2).reshape(9, 92)
                change = proposal-memory.prior
                state = memory.prior+change/torch.linalg.vector_norm(change).clamp_min(1.)
        if j in selected:
            residual = {}
            for name, w in (('before', before), ('after', state)):
                columns = w.reshape(9, 4, 23).permute(1, 0, 2)
                err = ((origin_keys[:, :, None]*columns).sum(1)-target).square()*mask
                residual[name] = float(err.sum()/mask.sum()) if mask.any() else None
            result[j] = dict(before=before.clone(), after=state.clone(), key=keys[j].clone(),
                             signal_residual=residual, observed_target_scalars=int(mask.sum()))
    return result


def channel_read_ablation(delta):
    if delta.shape != (1, 92):
        raise ValueError('one four-horizon flat23 read required')
    result = delta.expand(23, -1).clone().reshape(23, 4, 23)
    for channel in range(23):
        result[channel, :, channel] = 0
    return result.reshape(23, 92)


def direction_summary(gradient, step):
    g, s = gradient.detach().double().flatten(), step.detach().double().flatten()
    if g.shape != s.shape or not torch.isfinite(g).all() or not torch.isfinite(s).all():
        raise ValueError('finite paired gradient and parameter step required')
    dot, gn, sn = float(g@s), float(g.norm()), float(s.norm())
    return dict(first_order_loss_change=dot, gradient_norm=gn, step_norm=sn,
                descent_cosine=-dot/(gn*sn) if gn and sn else None)
