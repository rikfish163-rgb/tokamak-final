"""Paired source-only diagnostic-dropout continuation; no EAST or TEST reads."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .missing_input_probe import mask_inputs
from .simple23 import Config, Simple23
from .simple_online import _durable_json, _hash


def drop_channels(x, mask, rng, *, max_drop=4):
    if x.ndim != 3 or x.shape != mask.shape or mask.dtype != torch.bool or not 0 <= max_drop < x.shape[-1]:
        raise ValueError('matching windows and bounded channel removal required')
    keep = np.ones((len(x), x.shape[-1]), bool)
    for row in keep:
        row[rng.choice(x.shape[-1], size=int(rng.integers(max_drop+1)), replace=False)] = False
    available = mask & torch.as_tensor(keep, device=x.device)[:, None, :]
    return x.masked_fill(~available, 0.), available


def train_step(model, optimizer, x, mask, y, rng, *, augmented, teacher=None, teacher_weight=0.):
    if not np.isfinite(teacher_weight) or teacher_weight < 0 or (teacher_weight and teacher is None):
        raise ValueError('nonnegative finite protection and teacher required')
    flags = [(p, p.requires_grad) for p in model.parameters()]
    excluded = {id(model.fast_origin), *(id(p) for p in model.forecast_head.parameters())}
    try:
        for p, _ in flags:
            if id(p) in excluded:
                p.requires_grad_(False)
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        xx, mm = drop_channels(x, mask, rng) if augmented else (x, mask)
        clean_logits = model(x, mask)['logit']
        clean = F.binary_cross_entropy_with_logits(clean_logits, y.float())
        perturbed = F.binary_cross_entropy_with_logits(model(xx, mm)['logit'], y.float())
        loss = .5*(clean+perturbed)
        protection = loss.new_zeros(())
        if teacher_weight:
            # Source-only functional regularization, not a pseudo-label for
            # the current EAST shot. Constrain clean inputs, leaving missing
            # inputs free to learn from their actual source labels.
            with torch.no_grad():
                target = teacher(x, mask)['logit']
            protection = F.smooth_l1_loss(clean_logits, target)
            loss = loss + teacher_weight*protection
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite source continuation')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return dict(loss=float(loss.detach()), clean=float(clean.detach()),
                    perturbed=float(perturbed.detach()), protection=float(protection.detach()))
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--teacher-weight', type=float, default=0.)
    args = p.parse_args()
    if args.steps < 1 or args.output.exists() or not np.isfinite(args.teacher_weight) or args.teacher_weight < 0:
        raise ValueError('positive budget and new output directory required')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = Config(**ckpt['model_config'])
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    train = simple_train.load_shots(roles['train'], advanced)
    val = simple_train.load_shots(roles['val'], advanced)
    groups = simple_train.pools(train)
    args.output.mkdir(parents=True)
    protocol = dict(parent_sha256=_hash(args.checkpoint), steps=args.steps, batch_size=64,
                    lr=1e-5, weight_decay=0., seed=131, augmentation_seed=4000131,
                    max_drop=4, clean_weight=.5, missing_weight=.5,
                    teacher_weight=args.teacher_weight, protection='clean-input logit SmoothL1',
                    threshold_selection='clean VAL160 only, FPR<.1; no TEST access',
                    scope='DEVELOPMENT_ONLY; source robustness, not intra adaptation',
                    code_sha256={n: _hash(Path(__file__).with_name(n)) for n in
                        ('missing_training.py', 'missing_input_probe.py', 'simple_train.py',
                         'simple_data.py', 'simple23.py', 'io23.py')})
    _durable_json(args.output/'protocol.json', protocol)
    result = {}
    for arm in ('control', 'dropout'):
        torch.manual_seed(131)
        rng, augment_rng = np.random.default_rng(131), np.random.default_rng(4000131)
        model = Simple23(cfg, ckpt['source_center'], ckpt['source_scale']).cuda()
        model.load_state_dict(ckpt['model'])
        teacher = deepcopy(model).eval().requires_grad_(False) if args.teacher_weight else None
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
        digest, history = hashlib.sha256(), []
        for step in range(args.steps):
            x, mask, y, _, _ = simple_train.batch(train, groups, rng, 64, cfg, 'cuda')
            for tensor in (x, mask, y):
                digest.update(tensor.detach().cpu().numpy().tobytes())
            history.append(train_step(model, opt, x, mask, y, augment_rng, augmented=arm == 'dropout',
                                      teacher=teacher, teacher_weight=args.teacher_weight))
            if (step+1) % 128 == 0:
                print(arm, step+1, history[-1], flush=True)
        rows = simple_train.score(model, val, 'cuda')
        selected = simple_train.select_threshold(rows, fpr_cap=.1)
        measurements = {}
        for condition, channels in [('observed', []), ('without_MNM', ['MNM']),
                                    ('without_MNM_CIII', ['MNM', 'CIII'])]:
            scored = rows if not channels else simple_train.score(model, mask_inputs(val, channels), 'cuda')
            measurements[condition] = dict(
                fixed=simple_train.metrics(scored, ckpt['threshold']),
                calibrated=simple_train.metrics(scored, selected['threshold']))
        folder = args.output/arm
        folder.mkdir()
        torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()},
                        model_config=cfg.to_dict(), source_center=ckpt['source_center'],
                        source_scale=ckpt['source_scale'], threshold=selected['threshold'],
                        validation=selected, forecast_trained=False,
                        qualification='unverified; new TEST220 required', training=protocol), folder/'source.pt')
        result[arm] = dict(sample_sha256=digest.hexdigest(), validation=measurements,
                           checkpoint_sha256=_hash(folder/'source.pt'))
        _durable_json(folder/'history.json', history)
        _durable_json(args.output/'result.json', result)
        print(arm, json.dumps({c: {t: {k: v for k, v in m.items() if k != 'predictions'}
                                  for t, m in pair.items()} for c, pair in measurements.items()}), flush=True)
    if result['control']['sample_sha256'] != result['dropout']['sample_sha256']:
        raise AssertionError('paired source samples differ')


if __name__ == '__main__':
    main()
