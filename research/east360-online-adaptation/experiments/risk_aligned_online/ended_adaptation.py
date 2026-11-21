"""Fixed-budget diagnostic on already-ended EAST shots, not prequential replay.

The last encoder block/risk head receive supervised BCE. Afterwards ONLY the
signal head is refit, because its representation has changed. No evaluation
labels, model selection, alarm changes, or claimed intra benefit enter fitting.
"""
import numpy as np
import torch
from torch.nn import functional as F

from . import simple_train, simple_data
from .simple23 import fit_forecaster


def three_group_batch(shots, rng, size, cfg, device):
    """Diagnostic 50/25/25 sampling, not a claimed optimal class prior.

    Each group samples shots uniformly then a valid endpoint uniformly. Missing
    groups are rejected: this fixed30 diagnostic is not the first-shot learner.
    """
    if size < 4 or size % 4:
        raise ValueError('three-group batch size must be a positive multiple of four')
    pools = {name: [] for name in ('precursor', 'safe_shot', 'early_disruptive')}
    for shot in shots:
        valid = np.asarray(shot['valid'], bool)
        y = np.asarray(shot['y'])
        for name, eligible in (
                ('precursor', valid & (y > .5)),
                ('safe_shot', valid & (y <= .5) & (not shot['is_disrupt'])),
                ('early_disruptive', valid & (y <= .5) & shot['is_disrupt'])):
            ends = np.flatnonzero(eligible)
            if len(ends):
                pools[name].append((shot, ends))
    for name, group in pools.items():
        if not group:
            raise ValueError(f'no eligible {name} observations in ended shots')
    counts = dict(precursor=size//2, safe_shot=size//4, early_disruptive=size//4)
    examples = []
    for name, count in counts.items():
        group = pools[name]
        for _ in range(count):
            shot, ends = group[int(rng.integers(len(group)))]
            end = int(rng.choice(ends))
            x, mask = simple_data.window(shot, end, cfg.window)
            examples.append((x, mask, float(shot['y'][end])))
    tensors = [torch.as_tensor(np.stack(v), device=device) for v in zip(*examples)]
    return *tensors, counts


def fit_ended(model, shots, *, ended_ids, steps, forecast_steps, device='cpu',
              seed=131, lr=1e-4, batch_size=64, sampling='endpoint', update_scope='last_block'):
    if min(steps, forecast_steps, batch_size) < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('positive finite training budget required')
    ids = [int(s['shot']) for s in shots]
    if (ids != list(ended_ids) or len(set(ids)) != len(ids) or
            any(s['device'] != 'EAST' for s in shots)):
        raise ValueError('only the exact ended EAST shot manifest may train')
    groups = simple_train.pools(shots)
    if sampling not in ('endpoint', 'three_group'):
        raise ValueError('unknown sampling protocol')
    rng = np.random.default_rng(seed)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    if update_scope == 'last_block':
        selected = [*model.encoder.layers[-1].parameters(),
                    *model.risk_head.parameters(), model.fast_origin]
    elif update_scope == 'relative' and model.cfg.input_view == 'residual_window':
        selected = list(model.relative_embedding.parameters())
    else:
        raise ValueError('update scope must match model structure')
    chosen = {id(p) for p in selected}
    flags = [(p, p.requires_grad) for p in model.parameters()]
    supervised, forecast = [], []
    sample_counts = dict(precursor=0, safe_shot=0, early_disruptive=0)
    model.eval()
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in chosen)
        optimizer = torch.optim.AdamW(selected, lr=lr, weight_decay=0.)
        for _ in range(steps):
            if sampling == 'three_group':
                x, mask, y, counts = three_group_batch(shots, rng, batch_size, model.cfg, device)
                for name, count in counts.items():
                    sample_counts[name] += count
            else:
                x, mask, y, _, _ = simple_train.batch(
                    shots, groups, rng, batch_size, model.cfg, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, mask)['logit'], y.float())
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite supervised loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True)
            optimizer.step()
            supervised.append(float(loss.detach()))
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    # Same fixed probe before/after refitting: signal learning must not silently
    # alter the warning model. The broader parameter check verifies this too.
    probe_x, probe_mask, _, _, _ = simple_train.batch(
        shots, groups, rng, batch_size, model.cfg, device)
    with torch.no_grad():
        risk_before = model(probe_x, probe_mask)['logit'].clone()
    classifier_before = {n: p.detach().clone() for n, p in model.named_parameters()
                         if not n.startswith('forecast_head.')}
    optimizer = torch.optim.AdamW(model.forecast_head.parameters(), lr=lr, weight_decay=0.)
    for _ in range(forecast_steps):
        x, mask, _, future, observed = simple_train.batch(
            shots, groups, rng, batch_size, model.cfg, device)
        forecast.append(fit_forecaster(model, optimizer, x, mask, future, observed))
    with torch.no_grad():
        risk_delta = float((model(probe_x, probe_mask)['logit'] - risk_before).abs().max())
    if risk_delta != 0 or any(not torch.equal(p, classifier_before[n])
                             for n, p in model.named_parameters() if n in classifier_before):
        raise AssertionError('forecast refit changed the classifier')
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('nonfinite fitted parameter')
    return dict(ended_ids=ids, supervised_loss=supervised, forecast_loss=forecast,
                forecast_stage_risk_max_delta=risk_delta,
                changed_parameters=[n for n, p in model.named_parameters()
                                    if not torch.equal(p, before[n])],
                lr=lr, batch_size=batch_size, seed=seed,
                sampling=sampling, update_scope=update_scope,
                sample_counts=sample_counts if sampling == 'three_group' else None,
                scope='fixed30 diagnostic; not all360 online performance')
