"""J-TEXT TRAIN-only preparation of a frozen-parent internal update interface.

Three matched arms: F forecast-only; M three-step future-risk meta-objective;
E M plus weak-event supervision. The static parent NEVER changes. VAL is used
for offline paired diagnostics, not EAST tuning. No J-TEXT online deployment.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..simple_train import sha, pools
from ..fixed_test_protocol import score_fixed_test
from ..event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, read_events
from ..event_adapt_20261022.data import attach_training_events, feedback_sample
from ..event_adapt_20261022.train import load_source
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from .model import InternalDynamics, feedback_loss, feedback_target, temporary_step


def fit_change_scales(parent, shots, train_ids):
    if any(s['device'] != 'J-TEXT' or s['shot'] not in set(train_ids) for s in shots):
        raise ValueError('change statistics require declared source TRAIN')
    center, scale = parent.center.cpu().numpy(), parent.scale.cpu().numpy()
    sums, counts = np.zeros(23), np.zeros(23)
    for s in shots:
        z = np.arcsinh((s['values'].astype('float64')-center)/scale)
        for h in sorted({1, parent.cfg.horizon//2, parent.cfg.horizon}):
            m = s['available'][h:] & s['available'][:-h] & s['valid'][h:, None] & s['valid'][:-h, None]
            d = np.where(m, z[h:]-z[:-h], 0.)
            sums += (d*d).sum(0); counts += m.sum(0)
    q = np.sqrt(sums/np.maximum(counts, 1))
    active = q > 0
    active[io23.RAW23.index('mode_number_n')] = False
    return np.where(active, q, 1.).astype('float32'), active


def stack_feedback(examples, device):
    return [torch.as_tensor(np.stack([e[k] for e in examples]), device=device)
            for k in ('x', 'mask', 'target', 'target_mask')]


def sample_forecast(shots, groups, cfg, rng, batch, device):
    examples = []
    for _ in range(batch):
        group = groups[int(rng.integers(2))]
        i, ends = group[int(rng.integers(len(group)))]; end = int(rng.choice(ends))
        s = shots[i]
        e = feedback_sample(s, origin=end, now=min(end+cfg.horizon, len(s['values'])-1), cfg=cfg)
        n = min(cfg.horizon, len(s['values'])-end-1)
        e['target_mask'][:n] &= s['valid'][end+1:end+1+n, None]
        examples.append(e)
    return stack_feedback(examples, device)


def episode_pools(shots, cfg, unroll):
    groups = {0: [], 1: []}
    span = unroll*cfg.horizon+1
    for i, s in enumerate(shots):
        if len(s['times']) <= span: continue
        valid = s['valid']
        starts = np.flatnonzero(np.convolve(valid.astype(int), np.ones(span+1, int), 'valid') == span+1)
        # Offline outer targets must still be timely (strict >10 ms). Signal
        # observations remain real valid samples; no artificial event labels.
        if s['is_disrupt']:
            starts = starts[np.round((s['down']-s['times'][starts+span])*1000, 9) > 10.]
        for label in (0, 1):
            use = starts[(s['y'][starts+span] > .5) == bool(label)]
            if len(use): groups[label].append((i, use))
    if not all(groups.values()): raise ValueError('both source episode classes required')
    return groups


def sample_episode(shots, groups, cfg, rng, batch, unroll, device):
    picked = []
    for _ in range(batch):
        group = groups[int(rng.integers(2))]
        i, ends = group[int(rng.integers(len(group)))]; picked.append((shots[i], int(rng.choice(ends))))
    steps = []
    for k in range(unroll):
        feedback, queries, labels, events = [], [], [], []
        for s, start in picked:
            origin = start+k*cfg.horizon; now = origin+cfg.horizon
            feedback.append(feedback_sample(s, origin=origin, now=now, cfg=cfg))
            queries.append(simple_data.window(s, now+1, cfg.window))
            labels.append(s['y'][now+1])
            events.append(s.get('event_target', np.full(len(s['times']), -1))[origin])
        qx, qm = [torch.as_tensor(np.stack(v), device=device) for v in zip(*queries)]
        steps.append(dict(feedback=stack_feedback(feedback, device), qx=qx, qm=qm,
                          y=torch.as_tensor(labels, device=device).float(),
                          event=torch.as_tensor(events, device=device).long()))
    return steps


def task_losses(model, episode, *, create_graph):
    fast = model.initial_fast(len(episode[0]['y']))
    diagnostics, risks, events, before, improvements = [], [], [], [], []
    for e in episode:
        x, m, truth, tm = e['feedback']
        out = model(x, m, fast)
        loss, count = feedback_loss(model, out['forecast'], x[:, -1], m[:, -1], truth, tm)
        diagnostics.append(loss[count > 0].mean())
        known = e['event'] >= 0
        events.append(F.cross_entropy(out['event'][known], e['event'][known]) if known.any() else out['event'].sum()*0.)
        fast, _ = temporary_step(model, fast, x, m, truth, tm, create_graph=create_graph)
        risk = F.binary_cross_entropy_with_logits(model(e['qx'], e['qm'], fast)['logit'], e['y'])
        base = F.binary_cross_entropy_with_logits(model(e['qx'], e['qm'])['logit'], e['y']).detach()
        risks.append(risk); before.append(base); improvements.append(base-risk.detach())
    return dict(forecast=torch.stack(diagnostics).mean(), risk=torch.stack(risks).mean(),
                event=torch.stack(events).mean(), before=torch.stack(before).mean(),
                improvement=torch.stack(improvements).mean(), fast_norm=fast.detach().norm(dim=(1, 2)).mean())


def score_static(model, shots, device):
    rows = []
    for s in shots:
        ends = np.flatnonzero(s['valid']); scores = []
        for lo in range(0, len(ends), 128):
            examples = [simple_data.window(s, int(t), model.cfg.window) for t in ends[lo:lo+128]]
            x, mask = [torch.as_tensor(np.stack(v), device=device) for v in zip(*examples)]
            scores.extend(model(x, mask)['logit'].detach().sigmoid().cpu().tolist())
        rows.append(dict(shot=s['shot'], is_disrupt=s['is_disrupt'], down=s['down'], times=s['times'][ends], scores=np.asarray(scores)))
    return rows


def forecast_diagnostic(model, shots, *, limit_batches=64):
    rng = np.random.default_rng(481); groups = pools(shots)
    error, persistence, count = 0., 0., 0
    for _ in range(limit_batches):
        x, m, truth, tm = sample_forecast(shots, groups, model.cfg, rng, 32, 'cuda')
        prediction = model(x, m)['forecast'].detach()
        target, valid = feedback_target(model, x[:, -1], m[:, -1], truth, tm)
        error += float((prediction-target).square().masked_fill(~valid, 0.).sum())
        persistence += float(target.square().masked_fill(~valid, 0.).sum()); count += int(valid.sum())
    return dict(mse=error/count, persistence_mse=persistence/count, mse_skill=1.-error/persistence,
                observations=count, protocol='fixed source-VAL endpoint sample; not a shot warning metric')


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=1024)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--mode', choices=['F', 'M', 'E'], required=True)
    p.add_argument('--initial', type=Path)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    if min(args.warmup, args.steps) < 0: p.error('nonnegative budgets required')
    torch.set_num_threads(4); torch.manual_seed(131); rng = np.random.default_rng(131)
    start = time.monotonic(); parent, ck = load_parent(DEFAULT_PARENT)
    original = {k:v.clone() for k,v in parent.state_dict().items()}
    roles = io23.source_roles(); advanced = io23.advanced_times()
    print(json.dumps(dict(stage='load_source', mode=args.mode)), flush=True)
    train, val = load_source(roles['train'], advanced), load_source(roles['val'], advanced)
    if sha(DEFAULT_LABELS) != LABEL_SHA256: raise ValueError('event CSV changed')
    grouped, _ = read_events(DEFAULT_LABELS)
    train = attach_training_events(train, grouped, allowed_train_ids=roles['train'])
    q, active = fit_change_scales(parent, train, roles['train'])
    model = InternalDynamics(parent, q, active).to('cuda')
    if args.initial:
        state = torch.load(args.initial, map_location='cuda', weights_only=True)
        if state['parent_sha256'] != PARENT_SHA256: raise ValueError('initialization parent changed')
        model.load_state_dict(state['model'])
    args.output.mkdir(parents=True)
    config = dict(format='internal-dynamics-v1', arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        parent_sha256=PARENT_SHA256, threshold=float(ck['threshold']), seed=131,
        source_training_ids=roles['train'], role_counts={k:len(v) for k,v in roles.items()},
        events_sha256=LABEL_SHA256, source_labels='max(25ms,AdvancedTime)',
        input='unchanged source scaling and flat23; JTEXT ip kA->A; local CIII sentinel mask',
        temporary=dict(site='patches before last Transformer block; not CLS', rank=8, step=.1, radius=1., unroll=3),
        objective='F: forecast; M: risk-after-3-steps + 0.2 forecast; E: M + 0.1 weak-event',
        fixed_parent=True, east_accessed=False, event_caveat='A-F weak offline clusters, unknown masked; upstream TEST exposure',
        evidence='DEVELOPMENT_ONLY; no gain assumed')
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/n for n in
        ('simple23.py', 'simple_data.py', 'io23.py', 'fixed_test_protocol.py', 'causal_input.py', 'event_prefix_probe.py',
         'event_adapt_20261022/data.py', 'event_adapt_20261022/train.py', 'redesign_20261004/prepare_prior.py')]
    config['code_sha256'] = {str(f.relative_to(io23.ROOT)):sha(f) for f in files}
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, str(f.relative_to(io23.ROOT)))
    # Frozen-parent qualification is rerun before fitting any auxiliary tensor.
    test = load_source(roles['test'], advanced)
    qualified = score_fixed_test('J-TEXT', score_static(model, test, 'cuda'), ck['threshold'])
    (args.output/'static_test220.json').write_text(json.dumps(qualified, indent=2))
    print(json.dumps(dict(stage='zero_change_TEST220', **{k:v for k,v in qualified.items() if k != 'predictions'})), flush=True)
    if qualified['tp'] != 100 or qualified['fp'] != 13: raise AssertionError('parent function reproduction failed')
    def save(path):
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        parent_sha256=PARENT_SHA256, threshold=float(ck['threshold']), rank=model.rank,
                        change_scale=model.change_scale.cpu(), active=model.active.cpu(), config=config), path)
    opt = torch.optim.AdamW(model.adaptation_parameters(), lr=1e-4, weight_decay=0.)
    history = []; groups = pools(train)
    for step in range(args.warmup):
        x, m, truth, tm = sample_forecast(train, groups, model.cfg, rng, 32, 'cuda')
        opt.zero_grad(set_to_none=True)
        loss, count = feedback_loss(model, model(x, m)['forecast'], x[:, -1], m[:, -1], truth, tm)
        loss = loss[count > 0].mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.adaptation_parameters(), 1., error_if_nonfinite=True); opt.step()
        if (step+1)%128 == 0:
            print(json.dumps(dict(stage='forecast_warmup', step=step+1, loss=float(loss.detach()), elapsed=time.monotonic()-start)), flush=True)
    warm_diagnostic = forecast_diagnostic(model, val)
    save(args.output/'warm.pt')
    print(json.dumps(dict(stage='warm_VAL_diagnostic', **warm_diagnostic)), flush=True)
    train_groups = episode_pools(train, model.cfg, 3); val_groups = episode_pools(val, model.cfg, 3)
    best = float('-inf')
    # Same episodes and optimizer budget for F/M/E; all share the warm artifact.
    rng = np.random.default_rng(811)
    for step in range(args.steps):
        episode = sample_episode(train, train_groups, model.cfg, rng, 16, 3, 'cuda')
        opt.zero_grad(set_to_none=True)
        losses = task_losses(model, episode, create_graph=args.mode != 'F')
        objective = losses['forecast'] if args.mode == 'F' else losses['risk']+.2*losses['forecast']
        if args.mode == 'E': objective = objective+.1*losses['event']
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.adaptation_parameters(), 1., error_if_nonfinite=True); opt.step()
        if (step+1)%64 == 0 or step+1 == args.steps:
            val_rng = np.random.default_rng(191); measured = []
            for _ in range(16):
                e = sample_episode(val, val_groups, model.cfg, val_rng, 16, 3, 'cuda')
                v = task_losses(model, e, create_graph=False)
                measured.append({k:float(x.detach()) for k,x in v.items()})
            v = {k:float(np.mean([r[k] for r in measured])) for k in measured[0]}
            record = dict(step=step+1, train={k:float(x.detach()) for k,x in losses.items()}, validation=v,
                          elapsed_seconds=time.monotonic()-start)
            history.append(record); print(json.dumps(record), flush=True)
            if v['improvement'] > best: best=v['improvement']; save(args.output/'selected.pt')
    if not args.steps: save(args.output/'selected.pt')
    for k,v in original.items():
        if not torch.equal(model.parent.state_dict()[k].cpu(), v): raise AssertionError(f'parent changed: {k}')
    if sha(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('original checkpoint changed')
    (args.output/'history.json').write_text(json.dumps(history, indent=2))
    result = dict(status='COMPLETED', mode=args.mode, source_static_qualified=True, parent_unchanged=True,
        warm_validation_forecast=warm_diagnostic, best_source_validation_BCE_improvement=best if args.steps else None,
        selected_sha256=sha(args.output/'selected.pt'), east_evaluated=False, elapsed_seconds=time.monotonic()-start)
    (args.output/'completed.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__': run()
