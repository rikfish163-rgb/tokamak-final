"""Offline VAL diagnosis of last-block signal gradients, not an online algorithm.

The oracle step deliberately uses a future label and MUST NOT be deployed or
reported as performance. It tests local controllability, not a global bound.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F

from . import simple_data, simple_train
from .fast_direction import episode, measure as measure_token
from .simple23 import Config, Simple23


def measure(model, e, *, step=.001):
    if not np.isfinite(step) or step < 0:
        raise ValueError('nonnegative finite diagnostic step required')
    model.eval()
    prefix = f'encoder.layers.{model.cfg.layers-1}.'
    params = {k:p.detach().clone().requires_grad_(True) for k,p in model.named_parameters()
              if k.startswith(prefix)}
    if not params:
        raise ValueError('last Transformer block not found')
    def forward(values, mask, state):
        return functional_call(model, state, (values, mask))
    prediction = forward(e['x'], e['a'], params)['forecast']
    signal_loss = model.forecast_loss(prediction, e['truth'], e['mask'])
    sg = torch.autograd.grad(signal_loss, tuple(params.values()))
    logit = forward(e['qx'], e['qa'], params)['logit']
    risk = F.binary_cross_entropy_with_logits(logit, e['y'])
    rg = torch.autograd.grad(risk, tuple(params.values()))
    sv, rv = torch.cat([g.flatten() for g in sg]), torch.cat([g.flatten() for g in rg])
    def update(gradients):
        norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt().clamp_min(1e-12)
        return {k:p.detach()-step*g.detach()/norm for (k,p),g in zip(params.items(), gradients)}
    with torch.no_grad():
        signal = forward(e['qx'], e['qa'], update(sg))['logit']
        oracle = forward(e['qx'], e['qa'], update(rg))['logit']
        sl = F.binary_cross_entropy_with_logits(signal, e['y'])
        ol = F.binary_cross_entropy_with_logits(oracle, e['y'])
    return dict(parameter_count=sum(p.numel() for p in params.values()),
                cosine=float(F.cosine_similarity(sv, rv, dim=0, eps=1e-12)),
                signal_gradient_norm=float(sv.norm()), risk_gradient_norm=float(rv.norm()),
                risk_before=float(risk.detach()), signal_improvement=float(risk.detach()-sl),
                oracle_improvement=float(risk.detach()-ol),
                signal_probability=float(signal.sigmoid().item()),
                oracle_probability=float(oracle.sigmoid().item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Simple23(Config(**checkpoint['model_config']), checkpoint['source_center'], checkpoint['source_scale']).cuda()
    model.load_state_dict(checkpoint['model'])
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    rng = np.random.default_rng(131)
    advanced = simple_data.advanced_times()
    rows = []
    gap = model.cfg.horizon+1
    for sid in simple_data.source_roles()['val']:
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        if len(shot['values']) <= gap:
            continue
        starts = np.flatnonzero(np.convolve(np.asarray(shot['valid'], int), np.ones(gap+1, int), 'valid') == gap+1)
        for label in (0, 1):
            group = starts[(shot['y'][starts+gap] > .5) == bool(label)]
            for start in rng.choice(group, min(4, len(group)), replace=False):
                e = episode(shot, int(start), model.cfg, 'cuda')
                rows.append(dict(shot=int(sid), origin=int(start), label=label,
                                 is_disrupt=bool(shot['is_disrupt']),
                                 block=measure(model, e), token=measure_token(model, e)))
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    summary = []
    for label, disruptive in ((0, False), (0, True), (1, True)):
        group = [r for r in rows if r['label'] == label and r['is_disrupt'] == disruptive]
        if group:
            summary.append(dict(label=label, is_disrupt=disruptive, n=len(group),
                block_cosine=float(np.mean([r['block']['cosine'] for r in group])),
                token_cosine=float(np.mean([r['token']['cosine'] for r in group])),
                block_signal_improvement=float(np.mean([r['block']['signal_improvement'] for r in group])),
                block_oracle_improvement=float(np.mean([r['block']['oracle_improvement'] for r in group])),
                token_signal_improvement=float(np.mean([r['token']['risk_improvement'] for r in group]))))
    payload = dict(scope='OFFLINE JTEXT VAL160 ONLY; oracle uses future labels; not deployment',
                   step=.001, seed=131, checkpoint_sha256=simple_train.sha(args.checkpoint),
                   limitations='different parameter coordinates and step sizes; not matched update strength; no shot-level claim',
                   code_sha256={n:simple_train.sha(Path(__file__).with_name(n)) for n in
                                ('block_direction.py', 'fast_direction.py', 'simple23.py', 'fast_update.py')},
                   summary=summary, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
