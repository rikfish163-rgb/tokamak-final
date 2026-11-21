"""J-TEXT-only single-task source training; no EAST auxiliary risk labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data as data
from .simple23 import Config, Simple23, fit_forecaster


def metrics(rows, threshold, lead=10.):
    records = []
    for r in rows:
        indices = np.flatnonzero(r['scores'] >= threshold)
        alarm = bool(len(indices))
        delta = (float(round((r['down'] - r['times'][indices[0]]) * 1000., 9))
                 if r['is_disrupt'] and alarm else None)
        records.append(dict(shot=int(r['shot']), is_disrupt=bool(r['is_disrupt']),
                            alarm=alarm, lead_ms=delta,
                            timely=bool(delta is not None and delta > lead)))
    nd = sum(r['is_disrupt'] for r in records)
    ns = len(records) - nd
    tp = sum(r['timely'] for r in records)
    fp = sum(r['alarm'] and not r['is_disrupt'] for r in records)
    timely = [r['lead_ms'] for r in records if r['timely']]
    tpr, fpr = tp / nd if nd else None, fp / ns if ns else None
    return dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, tpr=tpr, fpr=fpr,
                J=tpr-fpr if nd and ns else None, threshold=float(threshold),
                median_lead_ms=float(np.median(timely)) if timely else None,
                predictions=records)


def validation_rank(m):
    # Only PRIMARY risk task selects the source epoch. Infeasible epochs must
    # not win merely because they suppress more alarms and hence lower FPR.
    return (m['tpr'] < .9, -m['tpr'], m['fpr'])


def select_threshold(rows, fpr_cap=.1):
    if not 0 < fpr_cap < 1:
        raise ValueError('FPR cap must be in (0,1)')
    maxima, labels = [], []
    for r in rows:
        scores = np.asarray(r['scores'])
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('finite probabilities required')
        eligible = (np.round((r['down'] - r['times']) * 1000., 9) > 10.
                    if r['is_disrupt'] else np.ones(len(scores), bool))
        maxima.append(float(scores[eligible].max()) if eligible.any() else -np.inf)
        labels.append(r['is_disrupt'])
    maxima, labels = np.asarray(maxima), np.asarray(labels, bool)
    if not labels.any() or labels.all():
        raise ValueError('both validation classes required')
    candidates = {float(np.nextafter(1., 2.)), *maxima[np.isfinite(maxima)].tolist()}
    feasible = []
    for threshold in candidates:
        tpr, fpr = np.mean(maxima[labels] >= threshold), np.mean(maxima[~labels] >= threshold)
        if fpr < fpr_cap:
            feasible.append((-tpr, fpr, -threshold, threshold))
    return metrics(rows, min(feasible)[-1])


def load_shots(ids, advanced):
    return [data.attach_outcome(data.read_shot('J-TEXT', sid), 'J-TEXT', advanced) for sid in ids]


def pools(shots):
    result = {0: [], 1: []}
    for i, s in enumerate(shots):
        for label in (0, 1):
            ends = np.flatnonzero(s['valid'] & ((s['y'] > .5) == bool(label)))
            if len(ends):
                result[label].append((i, ends))
    if not all(result.values()):
        raise ValueError('source training needs both endpoint classes')
    return result


def batch(shots, groups, rng, size, cfg, device):
    examples = []
    for _ in range(size):
        label = int(rng.integers(2))
        group = groups[label]
        i, ends = group[int(rng.integers(len(group)))]
        end = int(rng.choice(ends))
        s = shots[i]
        x, mask = data.window(s, end, cfg.window)
        target = np.zeros((cfg.horizon, 23), np.float32)
        tm = np.zeros_like(target, bool)
        n = min(cfg.horizon, len(s['values'])-end-1)
        if n:
            target[:n] = s['values'][end+1:end+1+n]
            tm[:n] = s['available'][end+1:end+1+n] & s['valid'][end+1:end+1+n, None]
        examples.append((x, mask, label, target, tm))
    return [torch.as_tensor(np.stack(v), device=device) for v in zip(*examples)]


def score(model, shots, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for s in shots:
            ends = np.flatnonzero(s['valid'])
            if not len(ends):
                raise ValueError(f'no valid endpoints for {s["shot"]}')
            probabilities = []
            for start in range(0, len(ends), 256):
                windows = [data.window(s, int(t), model.cfg.window) for t in ends[start:start+256]]
                x, mask = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
                probabilities.extend(model(x, mask)['logit'].sigmoid().cpu().tolist())
            rows.append(dict(shot=s['shot'], is_disrupt=s['is_disrupt'], down=s['down'],
                             times=s['times'][ends], scores=np.array(probabilities)))
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--forecast-steps', type=int, default=256)
    p.add_argument('--seed', type=int, default=131)
    p.add_argument('--val-fpr-cap', type=float, default=.1)
    p.add_argument('--input-view', choices=['source', 'window', 'hybrid'], default='source')
    args = p.parse_args()
    if min(args.epochs, args.steps, args.batch_size) < 1 or args.forecast_steps < 0:
        p.error('positive training dimensions required')
    if not 0 < args.val_fpr_cap < .2:
        p.error('source validation FPR cap must be below .20')
    if args.output.exists():
        raise FileExistsError('do not overwrite a prior experiment')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda'
    cfg = Config(input_view=args.input_view)
    roles, advanced = data.source_roles(), data.advanced_times()
    train, val = load_shots(roles['train'], advanced), load_shots(roles['val'], advanced)
    scaler = data.fit_source_scaler(train)
    model = Simple23(cfg, scaler['center'], scaler['scale']).to(device)
    # Forecast head is deliberately excluded from primary supervised training.
    forecast_ids = {id(x) for x in model.forecast_head.parameters()}
    params = [x for x in model.parameters() if id(x) not in forecast_ids]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=.01)
    groups = pools(train)
    args.output.mkdir(parents=True)
    config = dict(vars(args), output=str(args.output.resolve()), format='simple23-v1',
                  label='max(25ms,AdvancedTime)', mode='supervised_then_frozen_forecast',
                  roles={k: len(v) for k, v in roles.items()},
                  split_sha256=hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest(),
                  evidence='DEVELOPMENT_ONLY',
                  code_sha256={n: sha(Path(__file__).with_name(n)) for n in
                               ('simple23.py', 'simple_train.py', 'simple_data.py', 'io23.py')})
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    best, history = None, []
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for _ in range(args.steps):
            x, mask, y, _, _ = batch(train, groups, rng, args.batch_size, cfg, device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, mask)['logit'], y.float())
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite source loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        measured = select_threshold(score(model, val, device), args.val_fpr_cap)
        public = {k: v for k, v in measured.items() if k != 'predictions'}
        record = dict(epoch=epoch, loss=float(np.mean(losses)), validation=public)
        history.append(record)
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        rank = validation_rank(measured)
        if best is None or rank < best:
            best = rank
            torch.save(dict(model=model.state_dict(), model_config=cfg.to_dict(), config=config,
                            epoch=epoch, threshold=measured['threshold'], validation=measured,
                            source_center=scaler['center'], source_scale=scaler['scale']), args.output/'source_risk.pt')
    selected = torch.load(args.output/'source_risk.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model'])
    # No TEST access until epoch/threshold locked. Auxiliary fitting sees TRAIN
    # only and cannot change classifier parameters or predictions.
    opt_forecast = torch.optim.AdamW(model.forecast_head.parameters(), lr=3e-4)
    forecast_losses = []
    for _ in range(args.forecast_steps):
        x, mask, _, target, target_mask = batch(train, groups, rng, args.batch_size, cfg, device)
        forecast_losses.append(fit_forecaster(model, opt_forecast, x, mask, target, target_mask))
    selected['model'] = model.state_dict()
    selected['forecast_trained'] = bool(args.forecast_steps)
    torch.save(selected, args.output/'source.pt')
    test = load_shots(roles['test'], advanced)
    result = metrics(score(model, test, device), selected['threshold'])
    report = dict(jtext=result, selected_epoch=selected['epoch'],
                  test_source_acceptance=result['tpr'] >= .9 and result['fpr'] < .2,
                  evidence='DEVELOPMENT_ONLY', format='simple23-v1',
                  forecast_train_loss=float(np.mean(forecast_losses)) if forecast_losses else None,
                  provenance=dict(checkpoint_sha256=sha(args.output/'source.pt'),
                                  selected_test_count=len(test), selection_source='J-TEXT VAL160'))
    (args.output/'test220.json').write_text(json.dumps(report, indent=2))
    (args.output/'hashes.json').write_text(json.dumps(dict(source_pt_sha256=sha(args.output/'source.pt'),
        test220_sha256=sha(args.output/'test220.json'), config_sha256=sha(args.output/'config.json')), indent=2))
    print(json.dumps(dict(completed=True, selected_epoch=selected['epoch'],
                          qualified=report['test_source_acceptance'], **{k: v for k, v in result.items() if k != 'predictions'})), flush=True)


if __name__ == '__main__':
    main()
