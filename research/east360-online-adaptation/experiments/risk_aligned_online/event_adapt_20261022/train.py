"""Matched source-only A0/A1/A2 experiments; TEST is read after VAL selection.

This first stage establishes source competence. It does NOT claim that a
joint auxiliary loss is already a useful EAST adaptation policy. The shared
inner operator in model.py must subsequently be meta-prepared and tested.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, read_events
from ..fixed_test_protocol import score_fixed_test
from ..simple_train import select_threshold, validation_rank, pools, sha
from .data import attach_training_events, fit_coordinates, feedback_sample
from .model import Config, EventForecastModel, arrived_loss, event_loss


def load_source(ids, advanced):
    return [simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('J-TEXT', sid)),
                                      'J-TEXT', advanced) for sid in ids]


def deadline_eligible(shot, min_lead_ms):
    if min_lead_ms < 0:
        raise ValueError('nonnegative source training deadline required')
    eligible = shot['valid'].copy()
    if min_lead_ms and shot['is_disrupt']:
        eligible &= np.round((shot['down']-shot['times'])*1000., 9) > min_lead_ms
    return eligible


def risk_pools(shots, min_lead_ms):
    # Filter only sample ORIGINS. Neither source labels, coordinate statistics
    # nor legal later diagnostic targets are rewritten by this intervention.
    return pools([dict(s, valid=deadline_eligible(s, min_lead_ms)) for s in shots])


def event_pool(shots, min_lead_ms=0.):
    result = {False: [], True: []}
    for i, s in enumerate(shots):
        for e in s['event_intervals']:
            use = (np.round(s['times'], 9) >= e.start) & (np.round(s['times'], 9) < e.end) & deadline_eligible(s, min_lead_ms)
            ends = np.flatnonzero(use)
            if len(ends): result[bool(s['is_disrupt'])].append((i, ends))
    if not all(result.values()):
        raise ValueError('event comparison requires both event-safe and event-disruptive TRAIN shots')
    return result


def sample_batch(shots, groups, events, rng, cfg, batch_size, device, channel_dropout=.05):
    examples = []
    for _ in range(batch_size):
        kind = int(rng.integers(4))
        if kind == 3:
            # Same event TIMES/sampling for NONE, REAL and SHUFFLE. Outcomes
            # only balance offline supervision; they are never input features.
            group = events[bool(rng.integers(2))]
        else:
            group = groups[1 if kind == 0 else 0]
        i, ends = group[int(rng.integers(len(group)))]
        end = int(rng.choice(ends)); s = shots[i]
        item = feedback_sample(s, origin=end, now=min(end+cfg.horizon, len(s['values'])-1), cfg=cfg)
        n = min(cfg.horizon, len(s['values'])-end-1)
        item['target_mask'][:n] &= s['valid'][end+1:end+1+n, None]
        # Predeclared source augmentation for missing diagnostics, shared by
        # all arms. It does not estimate any target-device statistics.
        keep = rng.random(23) >= channel_dropout
        item['mask'] &= keep[None]
        examples.append((item['x'], item['mask'], s['y'][end], s['event_target'][end],
                         item['target'], item['target_mask']))
    return [torch.as_tensor(np.stack(v), device=device) for v in zip(*examples)]


def training_step(model, optimizer, batch, *, event_weight, forecast_weight, gradient_rule='joint'):
    x, mask, y, event, truth, tm = batch
    optimizer.zero_grad(set_to_none=True)
    out = model(x, mask)
    risk = F.binary_cross_entropy_with_logits(out['logit'], y.float())
    forecast, counts = arrived_loss(model, out['forecast'], x[:, -1], mask[:, -1], truth, tm)
    usable = counts > 0
    forecast = forecast[usable].mean() if usable.any() else forecast.sum()*0.
    events = event_loss(out['event'], event.long())
    loss = risk+forecast_weight*forecast+event_weight*events
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite source objective')
    if gradient_rule == 'joint':
        loss.backward()
        gradient_info = {}
    elif gradient_rule in ('routed', 'balanced'):
        from .gradients import assign_task_gradients
        gradient_info = assign_task_gradients(model, [risk, forecast, events],
            [1., forecast_weight, event_weight], normalize_shared=gradient_rule == 'balanced')
    else:
        raise ValueError('unknown source gradient rule')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(total=float(loss.detach()), risk=float(risk.detach()),
                forecast=float(forecast.detach()), event=float(events.detach()),
                forecast_points=int(counts.sum()), **gradient_info)


def score_source(model, shots, device):
    """Causal convolution permits whole-sequence evaluation without future use.

    Unit tests compare this path with rolling past-only windows. There is no
    batch normalization, temporal normalization, or bidirectional attention.
    """
    model.eval(); rows = []
    with torch.no_grad():
        for s in shots:
            x = torch.as_tensor(s['values'], device=device)[None]
            m = torch.as_tensor(s['available'], device=device)[None]
            scores = model.sequence(x, m)['logit'].sigmoid()[0].cpu().numpy()
            valid = s['valid']
            rows.append(dict(shot=s['shot'], is_disrupt=s['is_disrupt'], down=s['down'],
                             times=s['times'][valid], scores=scores[valid]))
    return rows


def public(metrics):
    return {k: v for k, v in metrics.items() if k != 'predictions'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--event-mode', choices=['none', 'real', 'shuffle'], required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=131)
    p.add_argument('--val-fpr-cap', type=float, default=.15)
    p.add_argument('--event-weight', type=float, default=.10)
    p.add_argument('--forecast-weight', type=float, default=.20)
    p.add_argument('--risk-past-control', action='store_true')
    p.add_argument('--gradient-rule', choices=['joint', 'routed', 'balanced'], default='joint')
    p.add_argument('--min-source-lead-ms', type=float, choices=[0., 10.], default=0.)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('never overwrite another source experiment')
    if min(args.epochs, args.steps, args.batch_size) < 1 or not 0 < args.val_fpr_cap < .20:
        p.error('positive training budget and source validation cap < .20 required')
    if args.event_weight < 0 or args.forecast_weight < 0:
        p.error('nonnegative auxiliary weights required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    if sha(DEFAULT_LABELS) != LABEL_SHA256: raise ValueError('event CSV changed')
    start = time.monotonic()
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    # Deterministic CUDA kernels for matched-arm source comparisons.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)
    cfg = Config(risk_reads_forecast=not args.risk_past_control)
    roles, advanced = io23.source_roles(), io23.advanced_times()
    print(json.dumps(dict(stage='load_source_train_val', event_mode=args.event_mode)), flush=True)
    train, val = load_source(roles['train'], advanced), load_source(roles['val'], advanced)
    grouped, _ = read_events(DEFAULT_LABELS)
    train = attach_training_events(train, grouped, allowed_train_ids=roles['train'],
        shuffle_seed=args.seed+10000 if args.event_mode == 'shuffle' else None)
    coordinates = fit_coordinates(train, allowed_train_ids=roles['train'], horizon=cfg.horizon)
    model = EventForecastModel(cfg, **coordinates).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    groups, events = risk_pools(train, args.min_source_lead_ms), event_pool(train, args.min_source_lead_ms)
    args.output.mkdir(parents=True)
    files = list(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).parents[1]/n for n in
              ('io23.py', 'simple_data.py', 'simple_train.py', 'event_prefix_probe.py',
               'causal_input.py', 'fixed_test_protocol.py')]
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for file in files: z.write(file, str(file.relative_to(io23.ROOT)))
    config = dict(format='event-forecast-source-v2', arguments={**vars(args), 'output': str(args.output.resolve())},
        model=cfg.to_dict(), role_counts={k: len(v) for k, v in roles.items()},
        roles_sha256=hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest(),
        event_csv_sha256=LABEL_SHA256, coordinates={k: v.tolist() for k, v in coordinates.items()},
        normalization='J-TEXT TRAIN only; asinh(level), one-step differences, availability',
        input_contract='23 stored features; J-TEXT ip kA->A; local CIII=-10 sentinel mask',
        sampling='1/4 risk-positive; 1/2 risk-negative; 1/4 whole-event sampling balanced by outcome',
        training_origin_deadline_ms=args.min_source_lead_ms,
        deadline_scope='TRAIN disruptive origins only; no change to labels, source scaler, or future signal targets',
        source_augmentation='independent 5% channel masking; same endpoints/RNG across event arms',
        threshold_selection='J-TEXT VAL160 only; strict lead>10ms; no TEST fitting',
        qualification='full J-TEXT TEST220 TPR>=.90 and FPR<.20',
        evidence='DEVELOPMENT ONLY; upstream event discovery overlaps TEST; not independent confirmation',
        stage='source competence; no source meta-preparation and no EAST adaptation claim yet',
        gradient_rule=args.gradient_rule,
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        parameter_count=sum(p.numel() for p in model.parameters()))
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    best, history = None, []
    event_weight = 0. if args.event_mode == 'none' else args.event_weight
    for epoch in range(1, args.epochs+1):
        model.train(); losses = []
        for _ in range(args.steps):
            batch = sample_batch(train, groups, events, rng, cfg, args.batch_size, 'cuda')
            losses.append(training_step(model, optimizer, batch, event_weight=event_weight,
                                         forecast_weight=args.forecast_weight, gradient_rule=args.gradient_rule))
        measured = select_threshold(score_source(model, val, 'cuda'), args.val_fpr_cap)
        record = dict(epoch=epoch, losses={k: float(np.mean([v[k] for v in losses])) for k in losses[0]},
                      validation=public(measured), elapsed_seconds=time.monotonic()-start)
        history.append(record)
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        rank = validation_rank(measured)
        if best is None or rank < best:
            best = rank
            torch.save(dict(model=model.state_dict(), model_config=cfg.to_dict(), coordinates=coordinates,
                            threshold=measured['threshold'], epoch=epoch, validation=measured, config=config),
                       args.output/'source.pt')
    selected = torch.load(args.output/'source.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(selected['model'])
    test = load_source(roles['test'], advanced)
    rows = score_source(model, test, 'cuda')
    result = score_fixed_test('J-TEXT', rows, selected['threshold'])
    np.savez_compressed(args.output/'test220_traces.npz', **{
        f'{r["shot"]}_{k}': r[k] for r in rows for k in ('times', 'scores')})
    report = dict(status='COMPLETED', selected_epoch=selected['epoch'], metrics=result,
        qualified=result['tpr'] >= .9 and result['fpr'] < .2,
        checkpoint_sha256=sha(args.output/'source.pt'),
        elapsed_seconds=time.monotonic()-start, independent_confirmation=False,
        source_meta_prepared=False, east_evaluated=False)
    (args.output/'test220.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, 'metrics': public(result)}), flush=True)


if __name__ == '__main__':
    main()
