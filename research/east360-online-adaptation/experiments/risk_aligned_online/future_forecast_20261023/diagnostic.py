"""Issue direct future predictions using past signals; score in a separate phase."""
import numpy as np
import torch

from .. import simple_data
from ..forecast_token_scope_20261023.learning import ARMS, temporary, adapt


def issue(model, prefix, fast, *, device='cuda'):
    if (set(prefix) != {'values', 'available'} or prefix['values'].ndim != 2
            or prefix['values'].shape[1] != 23 or len(prefix['values']) < 128
            or prefix['available'].shape != prefix['values'].shape
            or prefix['available'].dtype != bool):
        raise ValueError('observed signal-only prefix with a full input window required')
    x, m = simple_data.window(prefix, len(prefix['values']) - 1, 128)
    x, m = [torch.as_tensor(a[None], device=device) for a in (x, m)]
    with torch.no_grad():
        result = model.predict_fast(x, m, fast)['forecast'][0].cpu().numpy().copy()
    if result.shape != (25, 23) or not np.isfinite(result).all():
        raise ValueError('finite direct H25 forecast required')
    return result


def forecast_error(prediction, normalized_truth, available):
    p, t, m = np.asarray(prediction), np.asarray(normalized_truth), np.asarray(available)
    if p.shape != t.shape or p.shape != m.shape or p.ndim != 2 or p.shape[1] != 23 or m.dtype != bool:
        raise ValueError('matching H-by23 prediction, transformed truth and boolean mask required')
    if not np.isfinite(p).all():
        raise ValueError('nonfinite model output')
    m = m & np.isfinite(t)
    error = np.abs(p.astype('float64') - np.where(m, t, p).astype('float64'))
    loss = np.where(error < 1., .5 * error**2, error - .5)
    counts = m.sum(axis=0)
    return dict(observed=int(m.sum()), mean=float(loss[m].mean()) if m.any() else None,
                per_channel=[float(loss[:, c][m[:, c]].mean()) if counts[c] else None for c in range(23)])


def replay(models, signal, *, device='cuda'):
    if set(models) != {'full', 'patch'} or set(signal) != {'values', 'available'}:
        raise ValueError('two diagnostic heads and signal-only input required')
    states = {a: temporary(models[a.split('_')[0]], a.split('_')[1]) for a in ARMS}
    original = {h: temporary(m, 'all') for h, m in models.items()}
    packets = {k: [] for k in ('origins', 'counts', 'last_value', 'last_available')}
    for h in models:
        packets[f'static_{h}'] = []
    for a in ARMS:
        packets[f'before_{a}'] = []; packets[f'after_{a}'] = []
    receipts = {a: [] for a in ARMS}
    n = len(signal['values'])
    for count in range(256, n, 128):
        prefix = {k: v[:count] for k, v in signal.items()}
        # Reproduce every original update, including near-end updates without
        # a complete future target. Only the separate diagnostic omits tails.
        complete = count + 25 <= n
        if complete:
            packets['origins'].append(count - 1); packets['counts'].append(count)
            packets['last_value'].append(prefix['values'][-1].copy())
            packets['last_available'].append(prefix['available'][-1].copy())
            for h, model in models.items():
                packets[f'static_{h}'].append(issue(model, prefix, original[h], device=device))
        for a in ARMS:
            model = models[a.split('_')[0]]
            if complete:
                packets[f'before_{a}'].append(issue(model, prefix, states[a], device=device))
            states[a], r = adapt(model, prefix, count, states[a], device=device)
            receipts[a].append(r)
            if complete:
                packets[f'after_{a}'].append(issue(model, prefix, states[a], device=device))
    if not packets['counts']:
        raise ValueError('no complete post-update future target in this diagnostic shot')
    return {k: np.asarray(v) for k, v in packets.items()}, receipts


def summarize(rows):
    """Equal-shot descriptions, never treating correlated windows as subjects."""
    output = {}
    for arm in sorted({r['arm'] for r in rows}):
        selected = [r for r in rows if r['arm'] == arm and r['before'] is not None and r['after'] is not None]
        ids = sorted({r['shot'] for r in selected}); per_shot = []
        for sid in ids:
            r = [v for v in selected if v['shot'] == sid]
            means = {k: float(np.mean([v[k] for v in r])) for k in ('before', 'after', 'static', 'persistence')}
            per_shot.append(dict(shot=sid, windows=len(r), **means,
                delta_step=means['after'] - means['before'], delta_static=means['after'] - means['static']))
        output[arm] = dict(n_shots=len(ids), n_windows=len(selected), per_shot=per_shot,
            mean_delta_step=float(np.mean([r['delta_step'] for r in per_shot])) if ids else None,
            mean_delta_static=float(np.mean([r['delta_static'] for r in per_shot])) if ids else None,
            improved_shots_step=sum(r['delta_step'] < 0 for r in per_shot),
            improved_shots_static=sum(r['delta_static'] < 0 for r in per_shot),
            improved_windows_step=sum(r['after'] < r['before'] for r in selected),
            mean_losses={k: float(np.mean([r[k] for r in per_shot])) if ids else None
                         for k in ('before', 'after', 'static', 'persistence')})
    return output
