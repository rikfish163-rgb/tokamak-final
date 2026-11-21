"""Bounded J-TEXT-only joint or second-order meta source training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import data
from .learning import adapt, risk_loss, signal_loss
from .model import ModelConfig, RiskAlignedModel


def build_sampling_pools(shots, horizon=25):
    """Source TRAIN strata: main positives, auxiliary-only, background.

    Sample a stratum then a shot then an endpoint, not uniformly over all
    endpoints: long discharges must not dominate the source sampling weight.
    """
    groups = {name: [] for name in ('positive', 'auxiliary_only', 'background')}
    gap = horizon + 1
    for index, s in enumerate(shots):
        query = np.arange(gap, len(s['values']))
        query = query[s['valid'][query] & s['valid'][query-gap]]
        positive = s['y'][query] > .5
        auxiliary = s['y125'][query] > .5
        for name, mask in (('positive', positive),
                           ('auxiliary_only', ~positive & auxiliary),
                           ('background', ~positive & ~auxiliary)):
            origins = query[mask] - gap
            if len(origins):
                groups[name].append((index, origins))
    if not any(groups.values()):
        raise ValueError('no valid source sampling pools')
    return groups


def sample_episode(shots, rng, *, horizon=25, window_length=128, mode='meta',
                   force_index=None, force_origin=None, dropout_probability=.15,
                   sampling_pools=None):
    """An origin's true signal target is observed before the later query label."""
    if mode not in ('joint', 'meta') or not shots:
        raise ValueError('mode and source shots required')
    gap = horizon + 1  # same later query context in joint and meta arms
    if sampling_pools is not None and force_index is None and force_origin is None:
        groups = [items for items in sampling_pools.values() if items]
        if not groups:
            raise ValueError('empty source sampling pools')
        group = groups[int(rng.integers(len(groups)))]
        force_index, origins = group[int(rng.integers(len(group)))]
        force_origin = int(rng.choice(origins))
    for _ in range(max(8, len(shots)*2)):
        i = int(force_index if force_index is not None else rng.integers(len(shots)))
        s = shots[i]
        last = len(s['values']) - gap - 1
        if last < 0:
            # This record cannot contain both a matured target and a later
            # query. Do not fabricate future samples or use negative slicing.
            if force_index is not None or force_origin is not None:
                break
            continue
        candidates = np.flatnonzero(s['valid'][:last+1])
        candidates = candidates[(candidates + gap < len(s['valid'])) &
                                s['valid'][candidates + gap]]
        if force_origin is not None:
            candidates = candidates[candidates == force_origin]
        if len(candidates):
            # Balance available source windows by primary label when possible.
            positives = candidates[s['y'][candidates + gap] > .5]
            negatives = candidates[s['y'][candidates + gap] <= .5]
            if force_origin is None and len(positives) and len(negatives):
                candidates = positives if rng.random() < .5 else negatives
            t = int(rng.choice(candidates))
            q = t + gap
            x, a = data.window(s, t, window_length)
            qx, qa = data.window(s, q, window_length)
            target = np.zeros((horizon, 23), np.float32)
            mask = np.zeros((horizon, 23), bool)
            count = min(horizon, len(s['values'])-t-1)
            if count:
                target[:count] = s['values'][t+1:t+count+1]
                mask[:count] = s['available'][t+1:t+count+1] & s['valid'][t+1:t+count+1, None]
            # Source-only stress test: model level view may be suppressed but
            # true observed values/masks remain available for signal learning.
            level_drop = rng.random(23) < dropout_probability
            if rng.random() < dropout_probability:
                channel = int(rng.integers(23))
                a[:, channel] = False
                qa[:, channel] = False
                mask[:, channel] = False
            prefix_end = int(rng.integers(min(gap, len(s['values']))))
            px, pa = data.window(s, prefix_end, window_length)
            # Four true primary-negative windows; the hardest prediction is
            # penalized during TRAIN. No fabricated negative if none exists.
            negative = np.flatnonzero(s['valid'] & (s['y'] <= .5))
            nx = np.zeros((4, window_length, 23), np.float32)
            na = np.zeros_like(nx, bool)
            if len(negative):
                for ni, endpoint in enumerate(rng.choice(negative, 4, replace=True)):
                    nx[ni], na[ni] = data.window(s, int(endpoint), window_length)
            return dict(support_values=x, support_available=a,
                        support_target=target, support_mask=mask,
                        query_values=qx, query_available=qa,
                        y=float(s['y'][q]), y125=float(s['y125'][q]),
                        support_end=t, query_end=q, level_drop=level_drop,
                        prefix_values=px, prefix_available=pa, prefix_end=prefix_end,
                        prefix_y=float(s['y'][prefix_end]),
                        prefix_y125=float(s['y125'][prefix_end]),
                        negative_values=nx, negative_available=na)
        if force_index is not None or force_origin is not None:
            break
    raise ValueError('no valid source meta episode with post-target query')


def _batch(episodes, device):
    names = ('support_values', 'support_available', 'support_target', 'support_mask',
             'query_values', 'query_available', 'level_drop',
             'prefix_values', 'prefix_available', 'negative_values', 'negative_available')
    result = {name: torch.as_tensor(np.stack([e[name] for e in episodes]), device=device)
              for name in names}
    for name in ('y', 'y125', 'prefix_y', 'prefix_y125'):
        result[name] = torch.tensor([e[name] for e in episodes], dtype=torch.float32, device=device)
    return result


def training_step(model, optimizer, episodes, *, mode, device, prefix_weight=.25):
    if mode not in ('joint', 'meta'):
        raise ValueError('mode must be joint or meta')
    batch = _batch(episodes, device)
    sv, sa = batch['support_values'], batch['support_available']
    qv, qa = batch['query_values'], batch['query_available']
    level_drop = batch['level_drop']
    optimizer.zero_grad(set_to_none=True)
    support = model(sv, sa, level_drop=level_drop)
    forecast, count, persistence = signal_loss(support, batch['support_target'],
                                               batch['support_mask'])
    if mode == 'meta':
        fast, audit = adapt(model, sv, sa, batch['support_target'],
                            batch['support_mask'], create_graph=True,
                            level_drop=level_drop)
        query = model(qv, qa, fast=fast, level_drop=level_drop)
        with torch.no_grad():
            before = model(qv, qa, level_drop=level_drop)
            before_risk = risk_loss(before['logits'], batch['y'], batch['y125'])
        improvement = float((before_risk - risk_loss(query['logits'], batch['y'], batch['y125'])).detach())
        gate = float(audit['gate'].mean())
    else:
        query = model(qv, qa, level_drop=level_drop)
        improvement, gate = 0., 0.
    risk = risk_loss(query['logits'], batch['y'], batch['y125'])
    prefix = model(batch['prefix_values'], batch['prefix_available'], level_drop=level_drop)
    prefix_risk = risk_loss(prefix['logits'], batch['prefix_y'], batch['prefix_y125'])
    nv, nm = batch['negative_values'], batch['negative_available']
    negative_logits = model(nv.flatten(0, 1), nm.flatten(0, 1))['logits'][:, 0].reshape(len(nv), 4)
    observed_negative = nm.any(dim=(2, 3))
    negative_losses = torch.nn.functional.softplus(negative_logits).masked_fill(~observed_negative, 0.)
    per_shot = negative_losses.max(1).values
    hard_negative = per_shot.sum() / observed_negative.any(1).sum().clamp_min(1)
    loss = risk + .2 * forecast.mean() + prefix_weight * prefix_risk + .25 * hard_negative
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite source loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), risk_loss=float(risk.detach()),
                startup_risk_loss=float(prefix_risk.detach()),
                hard_negative_loss=float(hard_negative.detach()),
                signal_loss=float(forecast.detach().mean()), gate=gate,
                meta_improvement=improvement, persistence_loss=float(persistence.mean()))


def _balanced_limit(ids, limit):
    if limit is None:
        return list(ids)
    if limit < 2:
        raise ValueError('smoke limits must be >=2 to retain both classes')
    classes = {False: [], True: []}
    for sid in ids:
        d, _ = data.P.read_outcome('J-TEXT', sid)
        classes[bool(d)].append(sid)
    if not all(classes.values()):
        raise ValueError('split lacks both classes')
    n_positive = max(1, min(len(classes[True]), round(limit*len(classes[True])/len(ids))))
    n_negative = min(len(classes[False]), limit-n_positive)
    if n_negative == 0:
        n_negative = 1
        n_positive -= 1
    if n_positive+n_negative < limit:
        spare = limit-n_positive-n_negative
        n_positive += min(spare, len(classes[True])-n_positive)
    selected = classes[True][:n_positive] + classes[False][:n_negative]
    return sorted(selected)


def _load(ids, advanced):
    return [data.attach_outcome(data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
            for sid in ids]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _score_shot(model, s, *, device):
    """Static source checkpoint, every 1 ms, with independent past-only windows."""
    w = model.cfg.window
    endpoints = np.arange(len(s['values']))
    endpoints = endpoints[s['valid'][endpoints]]
    times, scores = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(endpoints), 256):
            batch = endpoints[start:start+256]
            windows = [data.window(s, int(q), w) for q in batch]
            qx = torch.as_tensor(np.stack([x for x, _ in windows]), device=device)
            qa = torch.as_tensor(np.stack([a for _, a in windows]), device=device)
            logits = model(qx, qa)['logits']
            scores.extend(logits.sigmoid().cpu().numpy())
            times.extend(s['times'][batch].tolist())
    if not times:
        raise ValueError(f'empty validation shot {s["shot"]}')
    return dict(shot=s['shot'], times=np.asarray(times), scores=np.asarray(scores),
                is_disrupt=s['is_disrupt'], down=s['down'])


def _select(rows, head, lead, strict, target_tpr=.95, max_fpr=.20):
    """VAL-only threshold with a strict FPR cap, not a TEST score search.

    If the sensitivity target is unattainable, keep the highest sensitivity
    within the FPR cap and explicitly mark target feasibility false.
    """
    if not 0 < target_tpr <= 1 or not 0 < max_fpr <= 1:
        raise ValueError('invalid validation operating targets')
    maxima, classes = [], []
    for row in rows:
        disruptive = bool(row['is_disrupt'])
        if disruptive:
            lead_ms = np.round((row['down'] - row['times']) * 1000., 9)
            eligible = lead_ms > lead if strict else lead_ms >= lead
        else:
            eligible = np.ones(len(row['times']), bool)
        values = np.asarray(row['scores'])[:, head][eligible]
        values = values[np.isfinite(values)]
        maxima.append(float(values.max()) if len(values) else -np.inf)
        classes.append(disruptive)
    classes = np.asarray(classes, bool)
    maxima = np.asarray(maxima)
    if not classes.any() or classes.all():
        raise ValueError('validation requires both classes')
    candidates = sorted({float(np.nextafter(1., 2.)), *maxima[np.isfinite(maxima)].tolist()})
    options = []
    for threshold in candidates:
        tpr = float(np.mean(maxima[classes] >= threshold))
        fpr = float(np.mean(maxima[~classes] >= threshold))
        if fpr < max_fpr:
            options.append((threshold, tpr, fpr))
    feasible = [item for item in options if item[1] >= target_tpr]
    # The no-alarm candidate makes options nonempty for finite probabilities.
    if not options:
        raise ValueError('no validation operating point satisfies FPR cap')
    chosen = (min(feasible, key=lambda x: (x[2], -x[1], -x[0])) if feasible
              else min(options, key=lambda x: (-x[1], x[2], -x[0])))
    selected = _summarize(rows, head, chosen[0], lead, strict)
    selected.update(selection_feasible=bool(feasible), target_tpr=target_tpr,
                    max_fpr_exclusive=max_fpr)
    return selected


def _summarize(rows, head, threshold, lead, strict):
    projected = [dict(r, scores=r['scores'][:, head]) for r in rows]
    result = data.P.summarize(projected, threshold, lead, strict=strict)
    timely_leads = []
    tp = 0
    for record in result['predictions']:
        if record['lead_ms'] is not None:
            record['lead_ms'] = float(round(record['lead_ms'], 9))
        value = record['lead_ms']
        record['timely'] = bool(value is not None and (value > lead if strict else value >= lead))
        if record['is_disrupt'] and record['timely']:
            tp += 1
            timely_leads.append(value)
    tpr = tp / result['n_disrupt'] if result['n_disrupt'] else None
    result.update(tp=tp, tpr=tpr,
                  J=tpr-result['fpr'] if tpr is not None and result['fpr'] is not None else None,
                  median_lead_ms=float(np.median(timely_leads)) if timely_leads else None,
                  early_over_3000=sum(v > 3000 for v in timely_leads))
    return result


def _source_acceptance(metrics):
    return bool(metrics['tpr'] is not None and metrics['fpr'] is not None and
                metrics['tpr'] >= .9 and metrics['fpr'] < .20)


def _validation_rank(jtext, east_aux):
    # Both heads must meet source-VAL timely sensitivity before minimising the
    # worse false-positive rate. A good J-TEXT head cannot hide a poor EAST head.
    return (not _source_acceptance(jtext), not jtext['selection_feasible'],
            not _source_acceptance(east_aux), not east_aux['selection_feasible'],
            max(jtext['fpr'], east_aux['fpr']), jtext['fpr'] + east_aux['fpr'],
            -jtext['tpr'], -east_aux['tpr'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--mode', choices=('joint', 'meta'), default='meta')
    ap.add_argument('--backbone', choices=('patch', 'variate'), default='patch')
    ap.add_argument('--seed', type=int, default=131)
    ap.add_argument('--train-limit', type=int)
    ap.add_argument('--val-limit', type=int)
    ap.add_argument('--test-limit', type=int)
    ap.add_argument('--window', type=int, default=128)
    ap.add_argument('--horizon', type=int, default=25)
    ap.add_argument('--forecast-space', choices=('linear', 'asinh'), default='asinh')
    ap.add_argument('--prefix-weight', type=float, default=.25)
    ap.add_argument('--val-target-tpr', type=float, default=.95)
    args = ap.parse_args(argv)
    if min(args.epochs, args.steps, args.batch_size, args.window, args.horizon) <= 0:
        ap.error('positive training dimensions required')
    if not np.isfinite(args.prefix_weight) or args.prefix_weight < 0:
        ap.error('prefix-weight must be finite and nonnegative')
    if not .9 <= args.val_target_tpr <= 1:
        ap.error('val-target-tpr must be between .9 and 1')
    if args.output.exists():
        raise FileExistsError('output exists; source runs never overwrite an earlier run')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for source training; no silent CPU fallback')
    device = 'cuda'
    roles, advanced = data.source_roles(), data.advanced_times()
    selected = {name: _balanced_limit(roles[name], getattr(args, name + '_limit'))
                for name in ('train', 'val')}
    smoke = any(getattr(args, name + '_limit') is not None for name in ('train', 'val', 'test'))
    config = dict(vars(args), output=str(args.output.resolve()), device=device,
                  source_recipe='v3_source_strata_hard_negatives_constrained_validation',
                  evidence='SMOKE' if smoke else 'DEVELOPMENT_ONLY',
                  roles={name: len(ids) for name, ids in selected.items()},
                  threshold_origin='J-TEXT VAL160' if not smoke else 'J-TEXT VAL smoke subset',
                  label_contract={'jtext': 'max(25ms,AdvancedTime)', 'east': 'fixed125ms auxiliary on J-TEXT'})
    config['source_hashes'] = {
        'canonical_roles': data.P.SRC.split_sha256(roles),
        'selected_train_val': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        'advanced_time': _sha256(data.P.SRC.JTEXT_ADVANCED_TIME_PATH),
        **{name: _sha256(Path(__file__).parent/name)
           for name in ('data.py', 'train_source.py', 'model.py', 'learning.py')},
    }
    args.output.mkdir(parents=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    train = _load(selected['train'], advanced)
    sampling_pools = build_sampling_pools(train, horizon=args.horizon)
    val = _load(selected['val'], advanced)
    scaler = data.fit_source_scaler(train)
    model_cfg = ModelConfig(window=args.window, horizon=args.horizon,
                           forecast_space=args.forecast_space, backbone=args.backbone)
    model = RiskAlignedModel(model_cfg, scaler['center'], scaler['scale']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = None
    history = []
    started = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        logs = []
        for _ in range(args.steps):
            episodes = [sample_episode(train, rng, horizon=args.horizon,
                                       window_length=args.window, mode=args.mode,
                                       sampling_pools=sampling_pools)
                        for _ in range(args.batch_size)]
            logs.append(training_step(model, optimizer, episodes, mode=args.mode, device=device,
                                      prefix_weight=args.prefix_weight))
        rows = [_score_shot(model, s, device=device) for s in val]
        j = _select(rows, 0, 10., True, target_tpr=args.val_target_tpr)
        e = _select(rows, 1, 30., False, target_tpr=args.val_target_tpr)
        rank = _validation_rank(j, e)
        rec = dict(epoch=epoch, **{k: float(np.mean([x[k] for x in logs])) for k in logs[0]},
                   val_jtext=j, val_east_aux=e,
                   auxiliary_validation_acceptance=_source_acceptance(e),
                   source_acceptance=_source_acceptance(j),
                   elapsed_s=time.time()-started)
        history.append(rec)
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        public_rec = {k: v for k, v in rec.items() if k not in ('val_jtext', 'val_east_aux')}
        public_rec['val_jtext'] = {k: v for k, v in j.items() if k != 'predictions'}
        public_rec['val_east_aux'] = {k: v for k, v in e.items() if k != 'predictions'}
        print(json.dumps(public_rec), flush=True)
        if best is None or rank < best:
            best = rank
            checkpoint = dict(model=model.state_dict(), model_config=model_cfg.to_dict(),
                              thresholds=dict(jtext=float(j['threshold']), east=float(e['threshold'])),
                              config=config, source_center=scaler['center'], source_scale=scaler['scale'],
                              epoch=epoch, validation=dict(jtext=j, east_aux=e),
                              source_acceptance=rec['source_acceptance'])
            torch.save(checkpoint, args.output/'source.pt')
    checkpoint = torch.load(args.output/'source.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    # TEST labels and even smoke class balancing are sealed until epoch and
    # both thresholds have been selected exclusively from TRAIN/VAL.
    test_ids = _balanced_limit(roles['test'], args.test_limit)
    result_provenance = dict(source_hashes=config['source_hashes'],
                             selected_test_sha256=hashlib.sha256(
                                 json.dumps(test_ids).encode()).hexdigest(),
                             selected_test_count=len(test_ids),
                             checkpoint_sha256=_sha256(args.output/'source.pt'),
                             selection_source=config['threshold_origin'],
                             scoring='static source checkpoint; 1 ms endpoints; no fast adaptation')
    test = _load(test_ids, advanced)
    rows = [_score_shot(model, s, device=device) for s in test]
    result = dict(jtext=_summarize(rows, 0, checkpoint['thresholds']['jtext'], 10., True),
                  east_aux=_summarize(rows, 1, checkpoint['thresholds']['east'], 30., False),
                  selected_epoch=checkpoint['epoch'], threshold_origin=config['threshold_origin'],
                  evidence=config['evidence'],
                  evaluation_contract='lead_ms rounded to 9 decimals then >10; TPR>=.90; FPR<.20',
                  validation_source_acceptance=checkpoint['source_acceptance'],
                  provenance=result_provenance)
    result['test_source_acceptance'] = _source_acceptance(result['jtext'])
    (args.output/'test220.json').write_text(json.dumps(result, indent=2))
    (args.output/'hashes.json').write_text(json.dumps({
        'source_pt_sha256': _sha256(args.output/'source.pt'),
        'config_sha256': _sha256(args.output/'config.json'),
        'test220_sha256': _sha256(args.output/'test220.json'),
    }, indent=2))
    print(json.dumps(dict(completed=True, output=str(args.output),
                          selected_epoch=result['selected_epoch'],
                          validation_source_acceptance=result['validation_source_acceptance'],
                          test_source_acceptance=result['test_source_acceptance'],
                          test_jtext={k: v for k, v in result['jtext'].items() if k != 'predictions'},
                          test_east_aux={k: v for k, v in result['east_aux'].items() if k != 'predictions'})), flush=True)
    return result


if __name__ == '__main__':
    main()
