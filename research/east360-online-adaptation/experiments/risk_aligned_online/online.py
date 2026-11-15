"""Offline causal replay. Original acquisition times are never latency-adjusted.

Run with python -m experiments.risk_aligned_online.online --checkpoint source.pt
--output NEW_DIRECTORY --device-domain EAST --limit 2.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.fast_online_idea import pipeline as P
from experiments.fast_online_idea.online import cohort_ids
from . import data
from .learning import adapt, signal_loss
from .model import ModelConfig, RiskAlignedModel


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _windows(s, ends, model, device, data_api):
    windows = [data_api.window(s, int(end), model.cfg.window) for end in ends]
    return tuple(torch.as_tensor(np.stack(items), device=device) for items in zip(*windows))


def predict_shot(model, s, *, intra, target_domain, device, data_api=data, control='learned'):
    """One update after scoring origin..origin+H, usable starting at origin+H+1."""
    if {'y', 'y125', 'valid', 'is_disrupt', 'down'} & s.keys():
        raise ValueError('outcome labels must not enter prediction before reveal')
    if target_domain not in ('EAST', 'J-TEXT') or control not in ('learned', 'fixed'):
        raise ValueError('invalid domain or control')
    n = len(s['values'])
    if n == 0:
        raise ValueError('empty shot')
    model.eval()
    model.requires_grad_(False)
    parent = state_hash(model)
    h = model.cfg.horizon
    fast = model.initial_fast(1).detach().clone()
    scores = np.empty(n, np.float32)
    shadow = np.empty(n, np.float32)
    updates = []
    _sync(device)
    started = time.perf_counter()
    update_seconds = 0.
    for origin in range(0, n, h + 1):
        stop = min(origin + h + 1, n)
        x, a = _windows(s, range(origin, stop), model, device, data_api)
        with torch.no_grad():
            head = 1 if target_domain == 'EAST' else 0
            base = model(x, a)['logits'][:, head].sigmoid()
            shadow[origin:stop] = base.cpu().numpy()
            current = model(x, a, fast=fast.expand(len(x), -1))['logits'][:, head].sigmoid() if intra else base
            scores[origin:stop] = current.cpu().numpy()
        if intra and stop == origin + h + 1:
            # Direct H-step target in the support origin's coordinates. No
            # earlier score in this block can use these observations.
            target = torch.as_tensor(s['values'][origin+1:stop][None], device=device)
            mask = torch.as_tensor(s['available'][origin+1:stop][None], device=device)
            _sync(device)
            tick = time.perf_counter()
            with torch.enable_grad():
                fast, audit = adapt(model, x[:1], a[:1], target, mask,
                                    fast=fast, create_graph=False, control=control)
            _sync(device)
            elapsed = time.perf_counter() - tick
            update_seconds += elapsed
            updates.append(dict(origin=origin, target_start=origin+1, target_end=stop-1,
                                observed_through=stop-1, effective_from=stop,
                                control=control, wall_s=elapsed,
                                **{key: float(value.item()) for key, value in audit.items()}))
    _sync(device)
    if state_hash(model) != parent:
        raise RuntimeError('prediction mutated persistent model')
    return dict(scores=scores, shadow_scores=shadow, fast_updates=updates, fast_reset=True,
                wall_s=time.perf_counter()-started, fast_update_wall_s=update_seconds)


def _active_indices(candidates, labels, errors, budget):
    """One hardest per present class, half-budget hard pool, time coverage rest."""
    candidates = np.asarray(candidates, dtype=int)
    if budget < 2:
        raise ValueError('selection budget must allow both classes')
    selected = {}
    def add(index, reason):
        selected.setdefault(int(index), reason)
    ranked = candidates[np.argsort(-errors[candidates], kind='stable')]
    for positive in (False, True):
        pool = ranked[(labels[ranked] > .5) == positive]
        if len(pool):
            add(pool[0], 'positive_floor' if positive else 'negative_floor')
    for index in ranked[:budget//2]:
        add(index, 'absolute_error')
    remaining = np.array([i for i in candidates if int(i) not in selected], dtype=int)
    count = min(budget-len(selected), len(remaining))
    if count:
        for index in remaining[np.linspace(0, len(remaining)-1, count, dtype=int)]:
            add(index, 'time_coverage')
    return np.array(sorted(selected), dtype=int), selected


def select_experience(model, s, *, sealed_scores, data_api=data, limit=32, episode_limit=8):
    """Bounded deterministic raw windows from this already revealed shot only."""
    valid = np.asarray(s['valid'], bool).copy()
    if s['is_disrupt']:
        valid &= np.asarray(s['times']) <= s['down']
    candidates = np.flatnonzero(valid)
    labels = np.asarray(s['y125'] if s.get('device', 'EAST') == 'EAST' else s['y'])
    scores = np.asarray(sealed_scores)
    if scores.shape != valid.shape or not np.isfinite(scores).all():
        raise ValueError('sealed_scores must be finite and match shot length')
    errors = np.abs(scores-labels)
    ends, reasons = _active_indices(candidates, labels, errors, limit)
    windows = [data_api.window(s, int(end), model.cfg.window) for end in ends]
    shape = (0, model.cfg.window, model.cfg.channels)
    record = dict(shot=int(s['shot']), device=s.get('device', 'EAST'),
                  risk_windows=np.stack([w[0] for w in windows]) if windows else np.empty(shape, np.float32),
                  risk_available=np.stack([w[1] for w in windows]) if windows else np.empty(shape, bool),
                  y=np.asarray(s['y'], np.float32)[ends].copy(),
                  y125=np.asarray(s['y125'], np.float32)[ends].copy(),
                  selected_endpoints=[int(end) for end in ends], meta_episodes=[],
                  selection=[dict(endpoint=int(end), reason=reasons[int(end)],
                                  absolute_error=float(errors[end])) for end in ends])
    h = model.cfg.horizon
    # All eligible queries, including rare positives between block-grid points.
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    queries = np.arange(h+1, len(valid))
    queries = queries[(invalid_prefix[queries+1]-invalid_prefix[queries-h-1]) == 0]
    queries, query_reasons = _active_indices(queries, labels, errors, episode_limit)
    origins = queries-h-1
    for t in origins:
        q = t+h+1
        sx, sa = data_api.window(s, t, model.cfg.window)
        qx, qa = data_api.window(s, q, model.cfg.window)
        record['meta_episodes'].append(dict(
            support_values=sx.copy(), support_available=sa.copy(),
            target=s['values'][t+1:t+h+1].copy(), mask=s['available'][t+1:t+h+1].copy(),
            query_values=qx.copy(), query_available=qa.copy(),
            y=float(s['y'][q]), y125=float(s['y125'][q]),
            support_end=int(t), target_end=int(t+h), query_end=int(q),
            selection_reason=query_reasons[int(q)], absolute_error=float(errors[q])))
    return record


def _risk(logits, y, y125, domain):
    if domain == 'EAST':
        return F.binary_cross_entropy_with_logits(logits[:, 1], y125)
    return F.binary_cross_entropy_with_logits(logits[:, 0], y) + .5 * F.binary_cross_entropy_with_logits(logits[:, 1], y125)


def persistent_update(model, s, replay, device, lr, steps, *, sealed_scores, data_api=data, control='learned'):
    """Supervised risk plus post-reveal second-order meta and direct forecast.

    Every episode starts at persistent fast_init; shot-temporary fast is absent
    from this API. Raw replay is re-encoded with the current persistent model.
    """
    if steps < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('positive steps and learning rate required')
    if control not in ('fixed', 'learned'):
        raise ValueError('invalid control')
    record = select_experience(model, s, sealed_scores=sealed_scores, data_api=data_api)
    experiences = [*list(replay)[-8:], record]
    if any(e['device'] != record['device'] for e in experiences):
        raise ValueError('cross-domain replay is unsupported')
    model.eval()
    model.requires_grad_(False)
    params = list(model.persistent_parameters())
    for p in params:
        p.requires_grad_(True)
    if control == 'fixed':
        model.update_controller.requires_grad_(False)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD(params, lr=lr)
    episodes = [episode for e in experiences for episode in e['meta_episodes']]
    nonempty = [e for e in experiences if len(e['risk_windows'])]
    losses, gradient_norms = [], []
    def tensor(value):
        return torch.as_tensor(value, device=device)
    _sync(device)
    started = time.perf_counter()
    try:
        for _ in range(steps if nonempty else 0):
            optimizer.zero_grad(set_to_none=True)
            x = tensor(np.concatenate([e['risk_windows'] for e in nonempty]))
            a = tensor(np.concatenate([e['risk_available'] for e in nonempty]))
            y = tensor(np.concatenate([e['y'] for e in nonempty]))
            y125 = tensor(np.concatenate([e['y125'] for e in nonempty]))
            loss = _risk(model(x, a)['logits'], y, y125, record['device'])
            if episodes:
                batch = {key: tensor(np.stack([e[key] for e in episodes])) for key in
                         ('support_values', 'support_available', 'target', 'mask', 'query_values', 'query_available')}
                sv, sa = batch['support_values'], batch['support_available']
                fast, _ = adapt(model, sv, sa, batch['target'], batch['mask'], create_graph=True, control=control)
                query = model(batch['query_values'], batch['query_available'], fast=fast)
                ey = tensor(np.array([e['y'] for e in episodes], np.float32))
                ey125 = tensor(np.array([e['y125'] for e in episodes], np.float32))
                forecast, _, _ = signal_loss(model(sv, sa), batch['target'], batch['mask'])
                loss = loss + _risk(query['logits'], ey, ey125, record['device']) + .2 * forecast.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite persistent loss')
            loss.backward()
            gradient_norms.append(float(sum(p.grad.detach().double().square().sum().item()
                                           for p in model.update_controller.parameters() if p.grad is not None) ** .5))
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
    _sync(device)
    deltas = {n: float((p.detach()-before[n]).double().norm()) for n, p in model.named_parameters()}
    controller_delta = sum(v*v for n, v in deltas.items() if n.startswith('update_controller.')) ** .5
    audit = dict(steps=len(losses), losses=losses, replay_shots=len(experiences)-1,
                 control=control, selection=record['selection'],
                 meta_episode_count=len(episodes), selected_endpoints=record['selected_endpoints'],
                 episodes=[{k: e[k] for k in ('support_end', 'target_end', 'query_end', 'selection_reason', 'absolute_error')} for e in record['meta_episodes']],
                 controller_gradient_norms=gradient_norms, controller_delta_norm=controller_delta,
                 controller_learned=controller_delta > 0, parameter_delta_norms=deltas,
                 scope=('fast_init,risk_head,forecast_head' + (',update_controller' if control == 'learned' else '') + '; risk+forecast+post-reveal meta; control=' + control),
                 wall_s=time.perf_counter()-started)
    return audit, record


def summarize_rows(rows, threshold, target_domain):
    result = P.summarize(rows, threshold, 30. if target_domain == 'EAST' else 10., strict=target_domain == 'J-TEXT')
    if not result['n_disrupt'] or not result['n_safe']:
        result.update(tpr=None, fpr=None, J=None)
    return result


def posthoc_operating_points(rows, target_domain):
    if len({bool(r['is_disrupt']) for r in rows}) < 2:
        return {}
    return {str(cap): P.select_threshold(rows, 30. if target_domain == 'EAST' else 10.,
                                        strict=target_domain == 'J-TEXT', max_fpr=cap)
            for cap in (.1, .15, .2)}


def ordered_prefix(data_api, target_domain, ids, limit=0):
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    order = data_api.build_order(target_domain, ids, order='low_to_high')
    return order[:limit] if limit else order


def run_arm(model, *, target_domain, order, arm, threshold, output_dir, advanced,
            device, data_api=data, control='learned', inter_lr=.001, inter_steps=1):
    if arm not in ('static', 'intra', 'inter', 'both'):
        raise ValueError('unknown arm')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay = deque(maxlen=8)
    rows, shadow_rows, audits = [], [], []
    initial_parent = state_hash(model)
    start = time.perf_counter()
    for number, item in enumerate(order, 1):
        tick = time.perf_counter()
        sid = int(item['shot'])
        s = data_api.read_shot(target_domain, sid)
        if int(s['shot']) != sid or s.get('device', target_domain) != target_domain:
            raise ValueError('shot identity differs from manifest')
        parent = state_hash(model)
        prediction = predict_shot(model, s, intra=arm in ('intra', 'both'),
                                  target_domain=target_domain, device=device, data_api=data_api, control=control)
        path = output_dir / f'{sid}_sealed.npz'
        with path.open('xb') as handle:
            np.savez_compressed(handle, times=s['times'], scores=prediction['scores'], shadow_scores=prediction['shadow_scores'])
            handle.flush()
            os.fsync(handle.fileno())
        sealed_hash = file_hash(path)
        write_json(output_dir / f'{sid}_seal.json', dict(sealed_sha256=sealed_hash, parent_hash=parent, threshold=threshold))
        # Both prediction file and its receipt exist durably before metadata access.
        data_api.attach_outcome(s, target_domain, advanced)
        s['device'] = target_domain
        row = dict(shot=sid, times=s['times'], scores=prediction['scores'], is_disrupt=s['is_disrupt'], down=s['down'])
        rows.append(row)
        shadow_rows.append(dict(row, scores=prediction['shadow_scores']))
        learning = None
        if arm in ('inter', 'both'):
            learning, experience = persistent_update(model, s, replay, device, inter_lr, inter_steps,
                                                     sealed_scores=prediction['scores'], data_api=data_api, control=control)
            replay.append(experience)
        cumulative = summarize_rows(rows, threshold, target_domain)
        audit = dict(index=number, shot=sid, parent_hash=parent, next_hash=state_hash(model),
                     sealed_sha256=sealed_hash, threshold=threshold, control=control, fast_lr=model.cfg.fast_lr,
                     fast_reset=True, fast_updates=prediction['fast_updates'], inter_update=learning,
                     selected_endpoints=learning['selected_endpoints'] if learning else [],
                     prediction_wall_s=prediction['wall_s'], fast_update_wall_s=prediction['fast_update_wall_s'],
                     shot_wall_s=time.perf_counter()-tick, cumulative=cumulative,
                     cumulative_revealed=number, cumulative_updates=sum(a['inter_update'] is not None for a in audits)+(learning is not None))
        write_json(output_dir / f'{sid}_audit.json', audit)
        audits.append(audit)
        if number in (1, 3, 5, 10, 20):
            torch.save(dict(model=model.state_dict(), model_config=model.cfg.to_dict(),
                            source_center=model.source_center.cpu(), source_scale=model.source_scale.cpu(),
                            revealed_shots=[r['shot'] for r in rows], threshold=threshold,
                            parent_hash=initial_parent, next_hash=state_hash(model)),
                       output_dir / f'snapshot_{number}.pt')
        print(json.dumps(dict(arm=arm, index=number, shot=sid, tpr=cumulative['tpr'], fpr=cumulative['fpr'], wall_s=time.perf_counter()-start)), flush=True)
    result = dict(deployed=summarize_rows(rows, threshold, target_domain),
                  same_parent_no_fast_shadow=summarize_rows(shadow_rows, threshold, target_domain),
                  posthoc_diagnostic_operating_points=posthoc_operating_points(rows, target_domain),
                  parent_hash=initial_parent, final_hash=state_hash(model), wall_s=time.perf_counter()-start,
                  hard_realtime_guarantee=False)
    write_json(output_dir / 'metrics.json', result)
    return dict(result, audits=audits)


def run_provenance(checkpoint, limit):
    source = checkpoint.get('evidence', checkpoint['config'].get('evidence', 'UNKNOWN'))
    smoke = 'SMOKE' in str(source).upper() or 'SMOKE' in str(checkpoint['config'].get('evidence', '')).upper()
    return dict(source_evidence=source,
                evidence='SMOKE' if smoke else ('DIAGNOSTIC_PREFIX' if limit else 'OFFLINE_COMMISSIONING_REPLAY'),
                code_sha256={name: file_hash(Path(__file__).parent/name)
                             for name in ('online.py', 'data.py', 'model.py', 'learning.py')})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-domain', choices=('EAST', 'J-TEXT'), default='EAST')
    parser.add_argument('--arms', nargs='+', choices=('static', 'intra', 'inter', 'both'), default=['static', 'intra', 'inter', 'both'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--control', choices=('learned', 'fixed'))
    parser.add_argument('--inter-lr', type=float, default=.001)
    parser.add_argument('--inter-steps', type=int, default=1)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.inter_steps < 1 or not np.isfinite(args.inter_lr) or args.inter_lr <= 0:
        parser.error('invalid limit or persistent update settings')
    if len(set(args.arms)) != len(args.arms):
        parser.error('duplicate arms')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required; no silent CPU fallback')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    mode = checkpoint['config'].get('mode')
    control = args.control or {'joint': 'fixed', 'meta': 'learned'}.get(mode)
    if control is None:
        parser.error('unknown checkpoint mode: specify --control')
    threshold = float(checkpoint['thresholds']['east' if args.device_domain == 'EAST' else 'jtext'])
    if not np.isfinite(threshold):
        raise ValueError('nonfinite source threshold')
    model = RiskAlignedModel(ModelConfig(**checkpoint['model_config']), checkpoint['source_center'], checkpoint['source_scale']).cuda()
    torch.set_num_threads(4)
    model.load_state_dict(checkpoint['model'])
    ids = cohort_ids(args.device_domain)
    full_order = ordered_prefix(data, args.device_domain, ids)
    if len(full_order) != len(ids) or {int(r['shot']) for r in full_order} != set(ids):
        raise ValueError('ordering must retain exact full roster')
    order = full_order[:args.limit] if args.limit else full_order
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'order.json', dict(full_order=full_order, selected_prefix=order, method='data.build_order:low_to_high'))
    write_json(args.output / 'config.json', dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
                arms=args.arms, domain=args.device_domain, threshold=threshold, source_mode=mode,
                control=control, control_explicit=args.control is not None, fast_lr=model.cfg.fast_lr,
                inter_lr=args.inter_lr, inter_steps=args.inter_steps, limit=args.limit,
                **run_provenance(checkpoint, args.limit),
                outcomes='automatic after sealing; no budget', replay='last 8 revealed shots, raw windows and episodes',
                hard_realtime_guarantee=False, timestamps='original acquisition times; no compute-latency adjustment',
                target_scaler=False, target_calibration=False))
    advanced = data.advanced_times() if args.device_domain == 'J-TEXT' else {}
    results = {}
    for arm in args.arms:
        model.load_state_dict(checkpoint['model'])
        results[arm] = run_arm(model, target_domain=args.device_domain, order=order, arm=arm,
                              threshold=threshold, output_dir=args.output/arm, advanced=advanced,
                              device='cuda', control=control, inter_lr=args.inter_lr, inter_steps=args.inter_steps)
        results[arm].pop('audits')
    write_json(args.output / 'summary.json', results)


if __name__ == '__main__':
    main()
