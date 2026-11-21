"""Label-assisted offline direction diagnostic; NEVER an online algorithm.

Select among noop, uniform and 23 single-channel steps using the future label.
This finite candidate oracle is not an upper bound over all possible mixtures.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data
from .fast_direction import episode
from .fast_update import normalized_step
from .simple23 import Config, Simple23
from .simple_online import FAST_STEP, FAST_RADIUS, _durable_json, _hash


def measure(model, e):
    model.eval()
    fast = model.initial_fast(1).detach().requires_grad_(True)
    prediction = model(e['x'], e['a'], fast)['forecast']
    mask = e['mask'] & torch.isfinite(e['truth'])
    target = torch.asinh((torch.where(mask, e['truth'], model.center)-model.center)/model.scale)
    losses = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~mask, 0.)
    total = losses.sum()/mask.sum().clamp_min(1)
    uniform, = torch.autograd.grad(total, fast, retain_graph=True)
    per_channel = losses.sum((0, 1))/mask.sum((0, 1)).clamp_min(1)
    gradients = [torch.autograd.grad(per_channel[c], fast, retain_graph=c < 22)[0] for c in range(23)]
    before = F.binary_cross_entropy_with_logits(model(e['qx'], e['qa'], fast)['logit'], e['y'])
    supervised, = torch.autograd.grad(before, fast)
    initial = fast.detach()
    directions = torch.cat([uniform, *gradients, supervised])
    updated = normalized_step(initial.expand(25, -1), initial.expand(25, -1), directions,
                              step=FAST_STEP, radius=FAST_RADIUS)
    candidates = torch.cat([initial, updated], 0)
    with torch.no_grad():
        logits = model(e['qx'].expand(26, -1, -1), e['qa'].expand(26, -1, -1), candidates)['logit']
        risks = F.binary_cross_entropy_with_logits(logits, e['y'].expand(26), reduction='none')
    selected = int(risks[:25].argmin())
    cosines = F.cosine_similarity(torch.cat(gradients), supervised.expand(23, -1), dim=-1, eps=1e-12)
    return dict(before_loss=float(risks[0]), uniform_loss=float(risks[1]),
                selected_loss=float(risks[selected]), supervised_loss=float(risks[25]),
                selected_index=selected, channel_losses=risks[2:25].cpu().tolist(),
                channel_cosines=cosines.detach().cpu().tolist(),
                uniform_cosine=float(F.cosine_similarity(uniform, supervised, dim=-1, eps=1e-12)),
                available_channels=int(mask.any(1).sum()),
                uniform_improvement=float(risks[0]-risks[1]),
                selected_improvement=float(risks[0]-risks[selected]),
                supervised_improvement=float(risks[0]-risks[25]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Simple23(Config(**ckpt['model_config']), ckpt['source_center'], ckpt['source_scale']).cuda().eval()
    model.load_state_dict(ckpt['model'])
    model.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(131)
    advanced = simple_data.advanced_times()
    records = []
    for i, sid in enumerate(simple_data.source_roles()['val']):
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        gap = model.cfg.horizon+1
        origins = np.arange(max(0, len(shot['values'])-gap))
        origins = origins[np.array([shot['valid'][t:t+gap+1].all() for t in origins], bool)]
        for label in (0, 1):
            eligible = origins[(shot['y'][origins+gap] > .5) == bool(label)]
            for t in rng.choice(eligible, min(2, len(eligible)), replace=False):
                group = 'safe' if not shot['is_disrupt'] else ('precursor' if label else 'early_disruptive')
                records.append(dict(shot=int(sid), origin=int(t), group=group,
                                    **measure(model, episode(shot, int(t), model.cfg, 'cuda'))))
        if (i+1) % 20 == 0:
            print('audited', i+1, 'shots', flush=True)
    if not all(torch.equal(v, before[k]) for k, v in model.state_dict().items()):
        raise AssertionError('offline diagnostic changed source')
    summary = {}
    for group in ('safe', 'early_disruptive', 'precursor'):
        rows = [r for r in records if r['group'] == group]
        summary[group] = dict(n=len(rows), **{name: float(np.mean([r[name] for r in rows])) for name in
            ('uniform_improvement', 'selected_improvement', 'supervised_improvement', 'uniform_cosine')},
            selected_beneficial_fraction=float(np.mean([r['selected_improvement'] > 1e-7 for r in rows])),
            any_positive_channel_cosine=float(np.mean([max(r['channel_cosines']) > 0 for r in rows])))
    _durable_json(args.output, dict(scope='ORACLE_DIAGNOSTIC_ONLY; future label used; not deployable or shot metrics',
                  checkpoint_sha256=_hash(args.checkpoint), code_sha256=_hash(Path(__file__)),
                  seed=131, max_per_shot_per_class=2, model_unchanged=True, summary=summary, records=records))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
