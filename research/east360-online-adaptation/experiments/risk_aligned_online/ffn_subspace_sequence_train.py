"""Fixed-budget SOURCE training of cumulative, operationally supervised FFN directions.

No target data or outcome, no new classifier, no threshold optimization. The
worst source safety anchors and genuinely timely positives train ONLY B.
All original tensors remain immutable; random/signal/risk share chain manifests.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from . import simple_data, simple_train
from .fast_direction import episode
from .ffn_subspace import FfnSubspaceModel
from .ffn_subspace_sequence import chain_origins, meta_sequence_step
from .ffn_subspace_stress import distort_chain
from .ffn_subspace_train import FORMAT, evaluate, source_screen
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent, load_source_train
from .simple_online import _durable_json, _hash

REFERENCE = Path(__file__).parent / 'runs/ffn_subspace_source256_20261009'


def source_chain_spec(shot, scores):
    if shot.get('device') != 'J-TEXT': raise ValueError('only SOURCE chains are authorized')
    scores = np.asarray(scores); n = len(shot['times'])
    if scores.shape != (n,): raise ValueError('aligned frozen source scores required')
    eligible = np.asarray(shot['valid'], bool).copy()
    eligible[:26] = False
    lead = (np.round((shot['down'] - shot['times']) * 1000., 9) if shot['is_disrupt']
            else np.zeros(n))
    if shot['is_disrupt']: eligible &= (shot['y'] == 1) & (lead > 10.)
    ends = np.flatnonzero(eligible)
    if not len(ends): return None
    if not np.isfinite(scores[ends]).all(): raise ValueError('finite eligible frozen scores required')
    peak = int(ends[np.argmax(scores[ends])]); origins = chain_origins(peak)
    valid_ends = np.flatnonzero(shot['valid'])
    if not np.array_equal(valid_ends, np.arange(len(valid_ends))):
        raise ValueError('source chains require a valid preterminal prefix')
    anchors = []
    for origin in origins:
        start, stop = origin + 26, min(origin + 126, len(valid_ends))
        if stop <= start: raise ValueError('source chain must have a post-update segment')
        segment = np.arange(start, stop)
        # Source labels/risk select anchors ONLY for offline outer training.
        # They never select an update in the actual online signal-only loop.
        choices = {start, (start + stop - 1) // 2, stop - 1,
                   int(segment[np.argmax(scores[segment])])}
        timely = segment[(shot['y'][segment] == 1) & (lead[segment] > 10.)]
        if shot['is_disrupt'] and len(timely):
            choices.add(int(timely[np.argmax(scores[timely])]))
        anchors.append(sorted(choices))
    return dict(shot=int(shot['shot']), is_disrupt=bool(shot['is_disrupt']), peak=peak,
                origins=origins, anchors=anchors)


def materialize(shot, spec, cfg):
    steps = []
    for origin, anchors in zip(spec['origins'], spec['anchors']):
        e = episode(shot, origin, cfg, 'cuda')
        windows = [simple_data.window(shot, end, cfg.window) for end in anchors]
        e['segment_x'] = torch.as_tensor(np.stack([x for x, a in windows]), device='cuda')
        e['segment_a'] = torch.as_tensor(np.stack([a for x, a in windows]), device='cuda')
        e['segment_y'] = torch.as_tensor(shot['y'][anchors], device='cuda')
        lead = (np.round((shot['down'] - shot['times'][anchors]) * 1000., 9)
                if spec['is_disrupt'] else np.zeros(len(anchors)))
        e['segment_lead'] = torch.as_tensor(lead, dtype=torch.float32, device='cuda')
        steps.append(e)
    return dict(device='J-TEXT', steps=steps, is_disrupt=spec['is_disrupt'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--coordinate-stress', action='store_true',
                        help='source-only recorded-convention stress; not physical perturbations')
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1: raise ValueError('new path and positive fixed steps required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT); roles = simple_data.source_roles()
    original = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    models = {name: FfnSubspaceModel(deepcopy(base), rank=8, seed=131).cuda().eval()
              for name in ('random', 'signal', 'risk')}
    args.output.mkdir(parents=True)
    root = Path(__file__).parent
    files = [root / n for n in ('ffn_subspace.py', 'ffn_subspace_sequence.py',
        'ffn_subspace_sequence_train.py', 'ffn_subspace_stress.py', 'ffn_subspace_train.py', 'fast_direction.py',
        'simple_data.py', 'simple_train.py', 'simple23.py', 'io23.py', 'fast_update.py')]
    files += [root / 'redesign_20261004/prepare_prior.py', root / 'ffn_intra_20261005/model.py']
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, str(path.relative_to(root)))
    protocol = dict(format=FORMAT, parent_sha256=_hash(DEFAULT_PARENT), source_only=True,
        east_access=False, test_access=False, source_classifier_training=False, jtext_online=False,
        rank=8, train_ids=roles['train'], val_ids=roles['val'], steps=args.steps, batch_size=4,
        seed=131, inner_step=.05, radius=1., stride=100, horizon=25,
        training='up to four linked signal-only steps; actual safe/D shots balanced 2+2',
        risk_objective='source worst-safe / best genuinely timely positive anchored threshold-margin loss',
        anchors='first/middle/last/frozen source max; add timely positive max on disruptive segments',
        source_positive='original max(25ms,AdvancedTime); strict lead>10ms; prepositive D not safe',
        threshold=float(parent['threshold']), threshold_tuning=False,
        optimizer='Adam', learning_rate=.001, basis_gradient_clip=1.,
        original_model_bitwise_unchanged=True, control_training_budget='identical chains, rank, steps and inner budget',
        code_sha256={str(path.relative_to(root)): _hash(path) for path in files},
        selection='last fixed step; fragment diagnosis retained, whole VAL160 required before promotion',
        whole_source_gate='TP>=static and FP<=static, and strictly better than random/signal operational counts',
        scope='DEVELOPMENT_SOURCE_AUXILIARY; SOURCE anchors are not deployable outcome access')
    protocol['coordinate_stress'] = dict(enabled=args.coordinate_stress,
        gain_log_range=[-.35, .35], bias_range=[-1., 1.], clean_probability=.3,
        range_source='same source-asinh diagnostic range as affine_signal_meta, not fitted on EAST',
        convention='one convention per entire linked chain; labels/times/availability unchanged',
        physical_claim=False)
    _durable_json(args.output / 'protocol.json', protocol)
    print(json.dumps(dict(stage='source_loading_and_frozen_anchor_selection', shots=len(roles['train']))), flush=True)
    train, train_ids = load_source_train()
    if train_ids != roles['train']: raise ValueError('source TRAIN roles changed')
    static = simple_train.score(base.cuda(), train, 'cuda')
    specs, pools = [], {False: [], True: []}
    for i, (shot, row) in enumerate(zip(train, static)):
        scores = np.full(len(shot['times']), -np.inf)
        scores[shot['valid']] = row['scores']
        spec = source_chain_spec(shot, scores); specs.append(spec)
        if spec is not None: pools[spec['is_disrupt']].append(i)
    if not all(pools.values()): raise ValueError('source needs eligible safe and disruptive chains')
    _durable_json(args.output / 'source_chain_specs.json', specs)
    protocol['eligible_source_shots'] = {str(k): len(v) for k, v in pools.items()}
    protocol['source_chain_specs_sha256'] = _hash(args.output / 'source_chain_specs.json')
    reference = json.loads((REFERENCE / 'protocol.json').read_text())
    cache = REFERENCE / 'source_val_episodes.pt'
    if _hash(cache) != reference['source_val_cache_sha256'] or reference['val_ids'] != roles['val']:
        raise ValueError('fixed source validation cache changed')
    validation = [(dict((k, v.cuda() if isinstance(v, torch.Tensor) else v)
                       for k, v in item['episode'].items()), item['identity'])
                  for item in torch.load(cache, map_location='cpu', weights_only=True)]
    protocol['source_val_cache_sha256'] = _hash(cache); protocol['source_val_cache_reference'] = str(cache.resolve())
    optimizers = {name: torch.optim.Adam([models[name].basis], lr=.001) for name in ('signal', 'risk')}
    rng = np.random.default_rng(131); history, manifest = [], []; started = time.monotonic()
    tau = math.log(parent['threshold'] / (1. - parent['threshold']))
    for step in range(args.steps):
        chosen = [int(rng.choice(pools[bool(slot % 2)])) for slot in range(4)]
        identities = [dict(specs[i]) for i in chosen]
        chains = [materialize(train[i], specs[i], base.cfg) for i in chosen]
        if args.coordinate_stress:
            transformed = [distort_chain(models['random'], chain, rng) for chain in chains]
            chains = [chain for chain, receipt in transformed]
            for identity, (_, receipt) in zip(identities, transformed):
                identity['coordinate_stress'] = receipt
        manifest.append(identities)
        row = {name: meta_sequence_step(models[name], optimizers[name], chains, mode=name,
                                         threshold_logit=tau) for name in ('signal', 'risk')}
        history.append(row)
        if (step + 1) % 16 == 0 or step + 1 == args.steps:
            _durable_json(args.output / 'training.json', dict(history=history, episode_manifest=manifest))
            print(json.dumps(dict(stage='source_sequence', step=step + 1,
                elapsed_s=time.monotonic() - started, **row)), flush=True)
    reports = {name: evaluate(model, validation) for name, model in models.items()}
    for name, model in models.items():
        if any(not torch.equal(model.base.state_dict()[k].cpu(), v) for k, v in original.items()):
            raise RuntimeError('qualified original source model changed')
        error = float((model.basis.T @ model.basis - torch.eye(8, device='cuda')).detach().abs().max())
        if error > 2e-6: raise RuntimeError('orthonormal basis budget changed')
        torch.save(dict(basis=model.basis.detach().cpu(), format=FORMAT, arm=name,
                        parent_sha256=protocol['parent_sha256']), args.output / f'{name}.pt')
    if _hash(DEFAULT_PARENT) != protocol['parent_sha256']: raise RuntimeError('parent file changed')
    protocol['episode_manifest_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    protocol['basis_sha256'] = {name: _hash(args.output / f'{name}.pt') for name in models}
    _durable_json(args.output / 'protocol.json', protocol)
    _durable_json(args.output / 'validation.json', dict(reports=reports, screen=source_screen(reports),
        screen_role='old one-step fragment diagnostic ONLY; new whole-shot gate predeclared separately',
        elapsed_s=time.monotonic() - started))
    print(json.dumps(dict(completed=True, fragment_screen=source_screen(reports))), flush=True)


if __name__ == '__main__': main()
