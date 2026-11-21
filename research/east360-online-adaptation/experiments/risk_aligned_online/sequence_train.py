"""Source-only matched reset/continuous meta-training; no EAST or TEST access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .joint_meta import grad_norm
from .sequence_fast import sequence, unroll
from .simple23 import Config, Simple23


def pools(shots, cfg, depth, *, require_both=True):
    if depth < 1:
        raise ValueError('positive depth required')
    gap = depth*(cfg.horizon+1)
    result = {0: [], 1: []}
    for i, s in enumerate(shots):
        if len(s['values']) <= gap:
            continue
        starts = np.flatnonzero(np.convolve(np.asarray(s['valid'], int), np.ones(gap+1, int), 'valid') == gap+1)
        for label in (0, 1):
            group = starts[(s['y'][starts+gap] > .5) == bool(label)]
            if len(group):
                result[label].append((i, group))
    if require_both and not all(result.values()):
        raise ValueError('both source endpoint classes required')
    if not any(result.values()):
        raise ValueError('no complete observed sequence')
    return result


def sample(shots, pool, cfg, rng, depth, size, device):
    seqs, manifest = [], []
    classes = [c for c in (0, 1) if pool[c]]
    if not classes:
        raise ValueError('no observed class for sampling')
    for _ in range(size):
        group = pool[classes[int(rng.integers(len(classes)))]]
        index, starts = group[int(rng.integers(len(group)))]
        origin = int(rng.choice(starts))
        seqs.append(sequence(shots[index], origin, depth, cfg, device))
        manifest.append((int(shots[index]['shot']), origin))
    return [{k: torch.cat([s[j][k] for s in seqs]) for k in seqs[0][j] if k != 'query_end'}
            for j in range(depth)], manifest


def train_step(model, optimizer, episodes, *, mode, risk_scope='all'):
    if mode not in ('reset', 'continuous') or model.cfg.fast_site != 'tokens':
        raise ValueError('token adaptation and explicit mode required')
    if risk_scope not in ('all', 'final'):
        raise ValueError('explicit risk scope required')
    origin_flag = model.fast_origin.requires_grad
    model.fast_origin.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    try:
        if mode == 'continuous':
            result = unroll(model, episodes, create_graph=True)
            logits, signal = result['logits'], result['signal_loss']
        else:
            results = [unroll(model, [e], create_graph=True) for e in episodes]
            logits = torch.cat([r['logits'] for r in results])
            signal = torch.stack([r['signal_loss'] for r in results]).mean()
        labels = torch.stack([e['y'] for e in episodes])
        before_logits = torch.stack([model(e['qx'], e['qa'])['logit'] for e in episodes])
        if risk_scope == 'final':
            labels, before_logits, logits = labels[-1:], before_logits[-1:], logits[-1:]
        before = F.binary_cross_entropy_with_logits(before_logits, labels)
        after = F.binary_cross_entropy_with_logits(logits, labels)
        objective = .5*before + .5*after + .2*signal
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite sequence objective')
        objective.backward()
        log = dict(objective=float(objective.detach()), risk_before=float(before.detach()),
                   risk_after=float(after.detach()), signal_loss=float(signal.detach()),
                   encoder_grad_norm=grad_norm(model.encoder.parameters()),
                   head_grad_norm=grad_norm(model.forecast_head.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return log
    finally:
        model.fast_origin.requires_grad_(origin_flag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['reset', 'continuous'], required=True)
    parser.add_argument('--risk-scope', choices=['all', 'final'], default='all')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.steps, args.depth, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('positive finite training budget required')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = Config(**parent['model_config'])
    model = Simple23(cfg, parent['source_center'], parent['source_scale']).cuda()
    model.load_state_dict(parent['model'])
    roles = simple_data.source_roles()
    advanced = simple_data.advanced_times()
    shots = simple_train.load_shots(roles['train'], advanced)
    pool = pools(shots, cfg, args.depth)
    protocol = dict(mode=args.mode, steps=args.steps, depth=args.depth, batch_size=args.batch_size,
                    lr=args.lr, seed=131, risk_scope=args.risk_scope, train_ids=roles['train'], val_ids=roles['val'],
                    parent_sha256=simple_train.sha(args.checkpoint),
                    objective='0.5 static BCE + 0.5 post-update BCE + 0.2 signal SmoothL1',
                    sampling='balanced final sequence label; uniform shot within class; earlier labels not balanced',
                    evidence='DEVELOPMENT_ONLY; new source must be requalified',
                    code_sha256={n:simple_train.sha(Path(__file__).with_name(n)) for n in
                                 ('sequence_train.py', 'sequence_fast.py', 'fast_direction.py', 'fast_update.py',
                                  'simple23.py', 'simple_train.py', 'simple_data.py', 'joint_meta.py')})
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    digest = hashlib.sha256()
    history = []
    for step in range(args.steps):
        episodes, manifest = sample(shots, pool, cfg, rng, args.depth, args.batch_size, 'cuda')
        digest.update(json.dumps(manifest).encode())
        log = train_step(model, optimizer, episodes, mode=args.mode, risk_scope=args.risk_scope)
        history.append(log)
        if (step+1) % 32 == 0:
            print(json.dumps(dict(step=step+1, **log)), flush=True)
    protocol['sample_sha256'] = digest.hexdigest()
    (args.output/'training.json').write_text(json.dumps(dict(protocol=protocol, history=history), indent=2))
    val = simple_train.load_shots(roles['val'], advanced)
    scored = simple_train.score(model, val, 'cuda')
    calibrated = simple_train.select_threshold(scored, fpr_cap=.1)
    fixed = simple_train.metrics(scored, parent['threshold'])
    torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    model_config=cfg.to_dict(), source_center=parent['source_center'],
                    source_scale=parent['source_scale'], threshold=calibrated['threshold'],
                    validation=calibrated, sequence_training=protocol, forecast_trained=True), args.output/'source.pt')
    (args.output/'validation.json').write_text(json.dumps(dict(fixed_parent_threshold=fixed,
        validation_calibrated=calibrated, checkpoint_sha256=simple_train.sha(args.output/'source.pt')), indent=2))
    print(json.dumps({k:v for k,v in calibrated.items() if k != 'predictions'}), flush=True)


if __name__ == '__main__':
    main()
