"""Fixed-budget, source-only FFN update-subspace experiment.

All original source tensors, including the H25 decoder, stay unchanged.
Random, signal-trained and task-trained bases have the same rank and online
budget. Source VAL screens the last fixed step, never the best EAST result.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import fast_meta, simple_data, simple_train
from .fast_direction import episode
from .ffn_subspace import FfnSubspaceModel, one_step, meta_step
from .redesign_20261004.prepare_prior import load_parent, load_source_train, DEFAULT_PARENT
from .simple_online import _durable_json, _hash

FORMAT = 'source-ffn-update-subspace-v1'


def draw(shots, pools, cfg, rng, label, device):
    group = pools[label]
    i, origins = group[int(rng.integers(len(group)))]
    origin = int(rng.choice(origins)); shot = shots[i]
    e = episode(shot, origin, cfg, device)
    return e, dict(shot=int(shot['shot']), origin=origin, query=e['query_end'],
                   label=int(e['y'].item()))


def evaluate(model, episodes):
    rows = []
    for e, identity in episodes:
        fast, row = one_step(model, e)
        before = model(e['qx'], e['qa'], model.initial_fast(1).requires_grad_())['logit']
        after = model(e['qx'], e['qa'], fast)['logit']
        row.update(identity, risk_before=float(F.binary_cross_entropy_with_logits(before, e['y']).detach()),
                   risk_after=float(F.binary_cross_entropy_with_logits(after, e['y']).detach()),
                   probability_before=float(before.detach().sigmoid()),
                   probability_after=float(after.detach().sigmoid()))
        rows.append(row)
    groups = {}
    for label in (0, 1):
        selected = [row for row in rows if row['label'] == label]
        if not selected: raise ValueError('both source endpoint classes required')
        groups[str(label)] = dict(n=len(selected),
            risk_before=float(np.mean([row['risk_before'] for row in selected])),
            risk_after=float(np.mean([row['risk_after'] for row in selected])),
            beneficial=sum(row['risk_after'] < row['risk_before'] for row in selected))
    return dict(risk_before=float(np.mean([row['risk_before'] for row in rows])),
                risk_after=float(np.mean([row['risk_after'] for row in rows])),
                accepted=sum(row['accepted'] for row in rows), classes=groups, rows=rows)


def source_screen(reports):
    risk = reports['risk']
    reasons = []
    if risk['risk_after'] >= risk['risk_before'] - 1e-6:
        reasons.append('no_source_risk_improvement')
    for control in ('random', 'signal'):
        if risk['risk_after'] >= reports[control]['risk_after'] - 1e-6:
            reasons.append(f'not_better_than_{control}')
    for label in ('0', '1'):
        if risk['classes'][label]['risk_after'] > risk['classes'][label]['risk_before'] + 1e-6:
            reasons.append(f'source_class_{label}_risk_harmed')
    return dict(passed=not reasons, reasons=reasons,
                meaning='fragment task-loss screening ONLY; not whole-shot qualification or EAST gain')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--validation-episodes', type=int, default=256)
    p.add_argument('--risk-objective', choices=('bce', 'nonregression'), default='bce')
    p.add_argument('--control-run', type=Path,
                   help='reuse completed same-budget random/signal controls after exact validation checks')
    args = p.parse_args()
    if args.output.exists() or min(args.steps, args.batch_size, args.validation_episodes) < 1:
        raise ValueError('new directory and positive fixed budgets required')
    if args.batch_size % 2 or args.validation_episodes % 2:
        raise ValueError('exactly balanced even source batches required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT)
    models = {name: FfnSubspaceModel(deepcopy(base), rank=8, seed=131).cuda().eval()
              for name in ('random', 'signal', 'risk')}
    original = {k: v.clone() for k, v in base.state_dict().items()}
    initial = models['random'].basis.detach().clone()
    args.output.mkdir(parents=True)
    root = Path(__file__).parent
    files = [root / n for n in ('ffn_subspace.py', 'ffn_subspace_train.py', 'simple23.py',
        'simple_data.py', 'simple_train.py', 'io23.py', 'fast_meta.py', 'fast_direction.py', 'fast_update.py')]
    files += [root / 'ffn_intra_20261005/model.py', root / 'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, str(f.relative_to(root)))
    roles = simple_data.source_roles()
    protocol = dict(format=FORMAT, parent_sha256=_hash(DEFAULT_PARENT), source_only=True,
        original_model_bitwise_unchanged=True, source_classifier_training=False,
        jtext_online=False, east_access=False, test_access=False,
        train_ids=roles['train'], val_ids=roles['val'], rank=8, full_ffn_dimension=16576,
        learned_basis_parameters=16576 * 8, temporary_parameters=8,
        steps=args.steps, batch_size=args.batch_size, validation_episodes=args.validation_episodes,
        seed=131, val_seed=20261009, inner_step=.05, radius=1., stride=100, horizon=25,
        inner='masked source-coordinate SmoothL1; accepted only when signal loss improves',
        risk_outer='actual source max(25,AdvancedTime) label at origin+26',
        risk_objective=args.risk_objective,
        signal_outer='same matured H25 loss after the same proposal',
        basis_constraint='QR after each optimizer step; coefficient and FFN delta norms match',
        optimizer='Adam', learning_rate=.001, basis_gradient_clip=1.,
        selection='last fixed step; source fragment screen before any EAST replay',
        gate='risk lower than static, random and signal; neither endpoint class harmed',
        threshold=float(parent['threshold']),
        code_sha256={str(f.relative_to(root)): _hash(f) for f in files},
        code_snapshot_sha256=_hash(args.output / 'code_snapshot.zip'),
        scope='DEVELOPMENT_SOURCE_AUXILIARY; not a new JTEXT online branch')
    _durable_json(args.output / 'protocol.json', protocol)
    print(json.dumps(dict(stage='loading_source', train_count=len(roles['train']), val_count=len(roles['val']))), flush=True)
    train, train_ids = load_source_train()
    if train_ids != roles['train']: raise ValueError('source training role order mismatch')
    val = simple_train.load_shots(roles['val'], simple_data.advanced_times())
    pools, val_pools = fast_meta.episode_pools(train, base.cfg), fast_meta.episode_pools(val, base.cfg)
    vrng = np.random.default_rng(20261009)
    validation = [draw(val, val_pools, base.cfg, vrng, index % 2, 'cuda')
                  for index in range(args.validation_episodes)]
    torch.save([dict(episode={k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in e.items()},
                     identity=identity) for e, identity in validation], args.output / 'source_val_episodes.pt')
    training_modes = ('signal', 'risk')
    control_protocol = None
    if args.control_run is not None:
        control_protocol = json.loads((args.control_run / 'protocol.json').read_text())
        for key in ('parent_sha256', 'rank', 'steps', 'batch_size', 'validation_episodes',
                    'train_ids', 'val_ids', 'seed', 'val_seed', 'inner_step', 'radius'):
            if control_protocol[key] != protocol[key]: raise ValueError(f'control mismatch: {key}')
        cached_path = args.control_run / 'source_val_episodes.pt'
        if _hash(cached_path) != control_protocol['source_val_cache_sha256']:
            raise ValueError('control validation cache changed')
        cached = torch.load(cached_path, map_location='cpu', weights_only=True)
        if len(cached) != len(validation): raise ValueError('control validation count mismatch')
        for saved, (e, identity) in zip(cached, validation):
            if saved['identity'] != identity: raise ValueError('control validation identity mismatch')
            for key, value in e.items():
                old = saved['episode'][key]
                if (not torch.equal(old, value.cpu()) if isinstance(value, torch.Tensor) else old != value):
                    raise ValueError(f'control validation tensor mismatch: {key}')
        for name in ('random', 'signal'):
            path = args.control_run / f'{name}.pt'
            if _hash(path) != control_protocol['basis_sha256'][name]:
                raise ValueError('control basis changed')
            saved = torch.load(path, map_location='cuda', weights_only=True)
            if saved['parent_sha256'] != protocol['parent_sha256'] or saved['arm'] != name:
                raise ValueError('control basis provenance mismatch')
            with torch.no_grad(): models[name].basis.copy_(saved['basis'])
        training_modes = ('risk',)
        protocol['reused_control_run'] = str(args.control_run.resolve())
        protocol['reused_control_protocol_sha256'] = _hash(args.control_run / 'protocol.json')
    reports = {'random': evaluate(models['random'], validation)}
    print(json.dumps(dict(stage='random_val', risk_before=reports['random']['risk_before'],
                         risk_after=reports['random']['risk_after'])), flush=True)
    optimizers = {name: torch.optim.Adam([models[name].basis], lr=.001) for name in training_modes}
    history = []; rng = np.random.default_rng(131); manifest = []; digest = hashlib.sha256()
    started = time.monotonic()
    for step in range(args.steps):
        batch = [draw(train, pools, base.cfg, rng, slot % 2, 'cuda') for slot in range(args.batch_size)]
        identities = [identity for _, identity in batch]
        digest.update(json.dumps(identities, sort_keys=True).encode())
        manifest.append(identities)
        record = {name: meta_step(models[name], optimizers[name], [e for e, _ in batch],
                  mode=('risk_protect' if name == 'risk' and args.risk_objective == 'nonregression' else name))
                  for name in training_modes}
        history.append(record)
        if (step + 1) % 32 == 0 or step + 1 == args.steps:
            _durable_json(args.output / 'training.json', dict(history=history, episode_manifest=manifest))
            print(json.dumps(dict(stage='source_meta', step=step + 1,
                                 elapsed_s=time.monotonic() - started, **record)), flush=True)
    for name in ('signal', 'risk'): reports[name] = evaluate(models[name], validation)
    screen = source_screen(reports)
    for name, model in models.items():
        if any(not torch.equal(model.base.state_dict()[k].cpu(), v) for k, v in original.items()):
            raise RuntimeError('original classifier/forecast/scaler changed')
        error = float((model.basis.T @ model.basis - torch.eye(8, device='cuda')).detach().abs().max())
        if error > 2e-6: raise RuntimeError('subspace norm budget changed')
        torch.save(dict(basis=model.basis.detach().cpu(), format=FORMAT, arm=name,
                        parent_sha256=protocol['parent_sha256']), args.output / f'{name}.pt')
    if not torch.equal(models['random'].basis, initial): raise RuntimeError('random control changed')
    if _hash(DEFAULT_PARENT) != protocol['parent_sha256']: raise RuntimeError('parent file changed')
    protocol['episode_manifest_sha256'] = digest.hexdigest()
    if control_protocol is not None and protocol['episode_manifest_sha256'] != control_protocol['episode_manifest_sha256']:
        raise ValueError('reused controls did not have the same training episodes')
    protocol['source_val_cache_sha256'] = _hash(args.output / 'source_val_episodes.pt')
    protocol['basis_sha256'] = {name: _hash(args.output / f'{name}.pt') for name in models}
    _durable_json(args.output / 'protocol.json', protocol)
    _durable_json(args.output / 'validation.json', dict(reports=reports, screen=screen,
                                                     elapsed_s=time.monotonic() - started))
    _durable_json(args.output / 'training.json', dict(history=history, episode_manifest=manifest))
    print(json.dumps(dict(completed=True, source_screen=screen,
        risk_after={name: report['risk_after'] for name, report in reports.items()})), flush=True)


if __name__ == '__main__': main()
