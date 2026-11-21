"""Offline source reconstruction -> frozen-encoder risk-readout training.

No EAST access. event-mode is a matched weak-label ablation. Both arms see
identical event TIMES and sampled inputs, but only REAL uses their classes.
The qualified historical source remains immutable; these are new candidates.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..event_adapt_20261022.data import attach_training_events
from ..event_adapt_20261022.train import load_source, event_pool, sample_batch
from ..event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, read_events
from ..fixed_test_protocol import score_fixed_test
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_train import pools, batch, score, select_threshold, validation_rank, sha
from ..simple_online import _durable_json
from .model import MaskedState, masked_views, reconstruction_loss


def fit_level_scale(parent, shots, train_ids):
    if any(s['device'] != 'J-TEXT' or s['shot'] not in set(train_ids) for s in shots):
        raise ValueError('source TRAIN only for reconstruction loss scales')
    c, q = parent.center.cpu().numpy(), parent.scale.cpu().numpy()
    scales = []
    for channel in range(23):
        a = np.concatenate([s['values'][s['available'][:, channel] & s['valid'], channel] for s in shots]).astype('float64')
        if not len(a): raise ValueError(f'no TRAIN observations for channel {channel}')
        z = np.arcsinh((a-c[channel])/q[channel])
        # This normalizes the AUXILIARY loss, never risk inputs or EAST stats.
        # A constant source channel receives unit error scale; no /tiny-IQR.
        scales.append(max(float(z.std()), .1) if z.max() > z.min() else 1.)
    return np.asarray(scales, 'float32')


def completion_diagnostic(model, val):
    rng = np.random.default_rng(711); groups = pools(val)
    losses, zero = [], []
    with torch.no_grad():
        for _ in range(32):
            x, m, _, _, _ = batch(val, groups, rng, 32, model.cfg, 'cuda')
            xx, mm, held = masked_views(x, m)
            pred = model(xx, mm, mode='reconstruct')['reconstruction']
            losses.append(float(reconstruction_loss(model, pred, xx, held)))
            zero.append(float(reconstruction_loss(model, torch.zeros_like(pred), xx, held)))
    return dict(mean_loss=float(np.mean(losses)), source_center_loss=float(np.mean(zero)),
                reduction_vs_source_center=1.-np.mean(losses)/np.mean(zero),
                scope='VAL sampled masked signals; not a warning metric')


def public(m):
    return {k:v for k,v in m.items() if k != 'predictions'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--event-mode', choices=['none', 'real', 'shuffle'], required=True)
    p.add_argument('--reconstruction-steps', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--seed', type=int, default=131)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if min(args.reconstruction_steps, args.epochs, args.steps) < 1: p.error('positive source budgets required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed); start = time.monotonic()
    parent, ck = load_parent(DEFAULT_PARENT)
    roles, advanced = io23.source_roles(), io23.advanced_times()
    print(json.dumps(dict(stage='load_source', event_mode=args.event_mode)), flush=True)
    train, val = load_source(roles['train'], advanced), load_source(roles['val'], advanced)
    if sha(DEFAULT_LABELS) != LABEL_SHA256: raise ValueError('weak-event source changed')
    grouped, _ = read_events(DEFAULT_LABELS)
    train = attach_training_events(train, grouped, allowed_train_ids=roles['train'],
        shuffle_seed=args.seed+10000 if args.event_mode == 'shuffle' else None)
    model = MaskedState(parent, fit_level_scale(parent, train, roles['train'])).cuda().eval()
    args.output.mkdir(parents=True)
    config = dict(format='masked-state-v1', arguments={**vars(args), 'output':str(args.output.resolve())},
        parent_sha256=PARENT_SHA256, original_parent_preserved=True, seed=args.seed,
        source_ids=roles, source_label='max(25ms,AdvancedTime); missing=25ms',
        source_qualification='TEST220 strict lead>10ms; TPR>=.9 and FPR<.2',
        source_selection='VAL160 only, FPR<.15; best timely TPR then FPR',
        source_input='unchanged source coordinates; JTEXT ip*1000; CIII sentinel missing',
        stage1='masked whole-channel completion: embedding + first Transformer block; .1 weak-event if REAL/SHUFFLE',
        stage2='freeze encoder; train second block, LayerNorm and risk head on original binary labels',
        mask_partition='channel_index modulo4; technical mask schedule NOT a physical grouping',
        zero_shot='no target labels, target statistics, support110 or EAST signals used',
        event_sha256=LABEL_SHA256, event_warning='offline weak A-F clusters; unknown masked; upstream TEST exposure',
        adaptation='candidate: disposable encoder SGD on arrived signals; not yet qualified',
        evidence='DEVELOPMENT_ONLY; inherited parent and all test sets previously developed')
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/f for f in
        ('simple23.py', 'simple_data.py', 'simple_train.py', 'io23.py', 'causal_input.py',
         'fixed_test_protocol.py', 'event_prefix_probe.py', 'event_adapt_20261022/data.py',
         'event_adapt_20261022/train.py', 'redesign_20261004/prepare_prior.py')]
    config['code_sha256'] = {str(f.relative_to(io23.ROOT)):sha(f) for f in files}
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, str(f.relative_to(io23.ROOT)))
    _durable_json(args.output/'config.json', config)
    groups, events = pools(train), event_pool(train)
    model.set_stage('reconstruction')
    pars = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(pars, lr=1e-4, weight_decay=0.)
    history = []
    for step in range(args.reconstruction_steps):
        x, m, _, e, _, _ = sample_batch(train, groups, events, rng, model.cfg, 32, 'cuda', channel_dropout=0.)
        xx, mm, held = masked_views(x, m)
        opt.zero_grad(set_to_none=True)
        out = model(xx, mm, mode='reconstruct')
        reconstruction = reconstruction_loss(model, out['reconstruction'], xx, held)
        known = e.repeat_interleave(4) >= 0
        event = (F.cross_entropy(out['event'][known], e.repeat_interleave(4)[known].long())
                 if known.any() else out['event'].sum()*0.)
        loss = reconstruction+(.1*event if args.event_mode != 'none' else 0.)
        loss.backward(); torch.nn.utils.clip_grad_norm_(pars, 1., error_if_nonfinite=True); opt.step()
        if (step+1)%128 == 0:
            r = dict(stage='reconstruction', step=step+1, reconstruction=float(reconstruction.detach()),
                     event=float(event.detach()), seconds=time.monotonic()-start)
            history.append(r); print(json.dumps(r), flush=True)
    model.set_stage('frozen')
    diagnostic = completion_diagnostic(model, val)
    _durable_json(args.output/'reconstruction_val.json', diagnostic)
    encoder_snapshot = {k:v.detach().clone() for k,v in model.named_parameters() if k in set(model.encoder_names())}
    model.set_stage('risk')
    pars = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(pars, lr=3e-4, weight_decay=.01)
    best = None; selected_epoch = None
    for epoch in range(1, args.epochs+1):
        losses = []
        for _ in range(args.steps):
            x, m, y, _, _ = batch(train, groups, rng, 64, model.cfg, 'cuda')
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, m)['logit'], y.float())
            loss.backward(); torch.nn.utils.clip_grad_norm_(pars, 1., error_if_nonfinite=True); opt.step()
            losses.append(float(loss.detach()))
        measured = select_threshold(score(model, val, 'cuda'), .15)
        r = dict(stage='risk_readout', epoch=epoch, loss=float(np.mean(losses)), validation=public(measured),
                 seconds=time.monotonic()-start)
        history.append(r); print(json.dumps(r), flush=True)
        _durable_json(args.output/'history.json', history)
        rank = validation_rank(measured)
        if best is None or rank < best:
            best, selected_epoch = rank, epoch
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                model_config=model.cfg.to_dict(), threshold=measured['threshold'], config=config,
                epoch=epoch, validation=measured), args.output/'source.pt')
    if any(not torch.equal(v, dict(model.named_parameters())[k]) for k,v in encoder_snapshot.items()):
        raise AssertionError('readout training changed the reconstruction encoder')
    chosen = torch.load(args.output/'source.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(chosen['model']); model.set_stage('frozen')
    test = load_source(roles['test'], advanced)
    rows = score(model, test, 'cuda')
    measured = score_fixed_test('J-TEXT', rows, chosen['threshold'])
    np.savez_compressed(args.output/'test220_traces.npz', **{f'{r["shot"]}_{k}':r[k] for r in rows for k in ('times', 'scores')})
    qualified = measured['tpr'] >= .9 and measured['fpr'] < .2
    report = dict(status='COMPLETED', selected_epoch=selected_epoch, source_qualified=qualified,
        source_sha256=sha(args.output/'source.pt'), metrics=measured, reconstruction=diagnostic,
        seconds=time.monotonic()-start, east_accessed=False, encoder_frozen_during_risk_training=True,
        original_parent_unchanged=sha(DEFAULT_PARENT) == PARENT_SHA256)
    _durable_json(args.output/'completed.json', report)
    print(json.dumps({**report, 'metrics':public(measured)}), flush=True)


if __name__ == '__main__': main()
