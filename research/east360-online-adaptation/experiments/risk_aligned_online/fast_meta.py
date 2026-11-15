"""Source-only risk-directed training of the temporary-update forecast head.

MT3-inspired outer task evaluation, not a reproduction of its image algorithm.
The inner update uses matured signals, never the later risk label. Static
classifier weights stay bitwise unchanged. This is an experiment, not evidence
of useful EAST adaptation. Compare with equally-budgeted forecast-only fitting.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .fast_direction import episode
from .simple23 import Config, Simple23
from .simple_online import FAST_STEP, FAST_GRAD_CLIP, FAST_RADIUS
from .fast_update import normalized_step


def temporary_step(model, e, *, create_graph):
    updated,loss,_,_=temporary_step_components(model,e,create_graph=create_graph)
    return updated,loss


def temporary_step_components(model, e, *, create_graph):
    fast = model.initial_fast(len(e['x'])).detach().requires_grad_(True)
    prediction = model(e['x'], e['a'], fast)['forecast']
    mask = e['mask'] & torch.isfinite(e['truth'])
    truth = torch.where(mask, e['truth'].detach(), model.center)
    target = torch.asinh((truth-model.center) / model.scale)
    loss = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~mask, 0.)
    # Each episode gets the SAME update as independent batch-size-one replay.
    per_episode = loss.sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
    gradient, = torch.autograd.grad(per_episode.sum(), fast, create_graph=create_graph,
                                   retain_graph=create_graph)
    if model.cfg.fast_rule=='normalized':
        return normalized_step(fast,fast.detach(),gradient,step=FAST_STEP,radius=FAST_RADIUS), per_episode.mean(),fast,gradient
    norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    offset = -FAST_STEP * gradient * (FAST_GRAD_CLIP / norm).clamp(max=1.)
    offset = offset * (FAST_RADIUS / offset.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
    return fast + offset, per_episode.mean(),fast,gradient


def train_step(model, optimizer, e, *, mode):
    if mode not in ('meta', 'forecast'):
        raise ValueError('meta or forecast required')
    flags = [(p, p.requires_grad) for p in model.parameters()]
    head = {id(p) for p in model.forecast_head.parameters()}
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in head)
        updated, signal_loss = temporary_step(model, e, create_graph=True)
        after = model(e['qx'], e['qa'], updated)['logit']
        risk = F.binary_cross_entropy_with_logits(after, e['y'])
        with torch.no_grad():
            before = F.binary_cross_entropy_with_logits(model(e['qx'], e['qa'])['logit'], e['y'])
        objective = risk if mode == 'meta' else signal_loss
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite training objective')
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.forecast_head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return dict(signal_loss=float(signal_loss.detach()), risk_before=float(before),
                    risk_after=float(risk.detach()), head_grad_norm=float(norm))
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def episode_pools(shots, cfg):
    pools = {0: [], 1: []}
    gap = cfg.horizon + 1
    for i, shot in enumerate(shots):
        valid = np.asarray(shot['valid'], bool)
        if len(valid) <= gap:
            continue
        origins = np.flatnonzero(np.convolve(valid.astype(int), np.ones(gap+1, int), 'valid') == gap+1)
        for label in (0, 1):
            ends = origins[(shot['y'][origins+gap] > .5) == bool(label)]
            if len(ends):
                pools[label].append((i, ends))
    if not all(pools.values()):
        raise ValueError('both source episode classes required')
    return pools


def sample(shots, pools, cfg, rng, size, device):
    episodes = []
    for _ in range(size):
        group = pools[int(rng.integers(2))]
        i, origins = group[int(rng.integers(len(group)))]
        episodes.append(episode(shots[i], int(rng.choice(origins)), cfg, device))
    return {k: torch.cat([e[k] for e in episodes]) for k in episodes[0] if k != 'query_end'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['meta', 'forecast'], required=True)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--fast-site', choices=['output', 'tokens'], default='output')
    args = p.parse_args()
    if min(args.steps, args.batch_size) < 1:
        p.error('positive training budget required')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = replace(Config(**ckpt['model_config']), fast_site=args.fast_site)
    model = Simple23(cfg, ckpt['source_center'], ckpt['source_scale'])
    model.load_state_dict(ckpt['model'])
    unchanged = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('forecast_head.')}
    model.to('cuda')
    roles = simple_data.source_roles()
    train = simple_train.load_shots(roles['train'], simple_data.advanced_times())
    pools = episode_pools(train, cfg)
    optimizer = torch.optim.AdamW(model.forecast_head.parameters(), lr=1e-4, weight_decay=0.)
    args.output.mkdir(parents=True)
    history = []
    for step in range(args.steps):
        e = sample(train, pools, cfg, rng, args.batch_size, 'cuda')
        history.append(train_step(model, optimizer, e, mode=args.mode))
        if (step+1) % 64 == 0:
            print(json.dumps(dict(step=step+1, **history[-1])), flush=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(not torch.equal(state[k], v) for k, v in unchanged.items()):
        raise AssertionError('static classifier changed')
    result = deepcopy(ckpt)
    result['model_config'] = cfg.to_dict()
    result['model'] = state
    result['fast_meta'] = dict(mode=args.mode, fast_site=args.fast_site,
                              parent_sha256=simple_train.sha(args.checkpoint),
                              steps=args.steps, batch_size=args.batch_size, lr=1e-4, seed=131,
                              static_classifier_bitwise_unchanged=True,
                              train_ids=roles['train'], evidence='DEVELOPMENT_ONLY',
                              code_sha256={n: simple_train.sha(Path(__file__).with_name(n)) for n in
                                           ('fast_meta.py', 'fast_direction.py', 'simple23.py', 'simple_data.py')})
    torch.save(result, args.output/'source.pt')
    (args.output/'training.json').write_text(json.dumps(dict(protocol=result['fast_meta'], history=history), indent=2))
    print('complete; classifier unchanged; source gate and online gains not yet re-evaluated', flush=True)


if __name__ == '__main__':
    main()
