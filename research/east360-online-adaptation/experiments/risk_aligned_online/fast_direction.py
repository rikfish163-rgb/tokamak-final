"""Offline mechanism audit, NOT online performance or a training algorithm.

Future risk labels are used solely by this diagnostic evaluator. The actual
temporary update consumes only a matured direct signal forecast target.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .simple23 import Config, Simple23
from .simple_online import FAST_STEP, FAST_GRAD_CLIP, FAST_RADIUS
from .fast_update import normalized_step


def episode(shot, origin, cfg, device, *, data_api=simple_data):
    query = origin + cfg.horizon + 1
    if origin < 0 or query >= len(shot['values']):
        raise ValueError('query must occur after the complete forecast target')
    if not np.asarray(shot['valid'][origin:query+1]).all():
        raise ValueError('invalid observation interval')
    x, a = data_api.window(shot, origin, cfg.window)
    qx, qa = data_api.window(shot, query, cfg.window)
    truth = shot['values'][origin+1:query]
    mask = shot['available'][origin+1:query]
    if not mask.any():
        raise ValueError('no observed signal target')
    result = {k: torch.as_tensor(v[None], device=device) for k, v in
              dict(x=x, a=a, qx=qx, qa=qa, truth=truth, mask=mask).items()}
    result.update(y=torch.tensor([float(shot['y'][query])], device=device),
                  query_end=query)
    return result


def measure(model, e):
    model.eval()
    fast = model.initial_fast(1).detach().requires_grad_(True)
    out = model(e['x'], e['a'], fast)
    signal_loss = model.forecast_loss(out['forecast'], e['truth'], e['mask'])
    signal_grad, = torch.autograd.grad(signal_loss, fast)
    logit = model(e['qx'], e['qa'], fast)['logit']
    risk_loss = F.binary_cross_entropy_with_logits(logit, e['y'])
    risk_grad, = torch.autograd.grad(risk_loss, fast)
    sn, rn = signal_grad.norm(), risk_grad.norm()
    dot = (signal_grad * risk_grad).sum()
    if model.cfg.fast_rule=='normalized':
        adapted=normalized_step(fast.detach(),fast.detach(),signal_grad,step=FAST_STEP,radius=FAST_RADIUS)
    else:
        offset = -FAST_STEP * signal_grad * min(1., FAST_GRAD_CLIP / max(float(sn), 1e-12))
        offset *= min(1., FAST_RADIUS / max(float(offset.norm()), 1e-12))
        adapted=fast.detach()+offset
    with torch.no_grad():
        after = model(e['qx'], e['qa'], adapted)['logit']
        after_loss = F.binary_cross_entropy_with_logits(after, e['y'])
    return dict(signal_loss=float(signal_loss.detach()), risk_before=float(risk_loss.detach()),
                risk_after=float(after_loss), risk_improvement=float(risk_loss.detach()-after_loss),
                probability_change=float(after.sigmoid()-logit.detach().sigmoid()),
                signal_grad_norm=float(sn), risk_grad_norm=float(rn),
                cosine=float(dot / (sn * rn).clamp_min(1e-20)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Simple23(Config(**ckpt['model_config']), ckpt['source_center'], ckpt['source_scale'])
    model.load_state_dict(ckpt['model'])
    model.to('cuda')
    roles = simple_data.source_roles()
    advanced = simple_data.advanced_times()
    rng = np.random.default_rng(131)
    records = []
    for sid in roles['val']:
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        gap = model.cfg.horizon + 1
        origins = np.arange(max(0, len(shot['values'])-gap))
        origins = origins[np.array([shot['valid'][t:t+gap+1].all() for t in origins], bool)]
        for label in (0, 1):
            eligible = origins[(shot['y'][origins+gap] > .5) == bool(label)]
            for t in rng.choice(eligible, min(4, len(eligible)), replace=False):
                e = episode(shot, int(t), model.cfg, 'cuda')
                records.append(dict(shot=int(sid), origin=int(t), query_end=e['query_end'],
                                    label=label, **measure(model, e)))
    summary = {}
    for label in (0, 1):
        rows = [r for r in records if r['label'] == label]
        summary[str(label)] = dict(n=len(rows),
            mean_cosine=float(np.mean([r['cosine'] for r in rows])),
            beneficial_fraction=float(np.mean([r['risk_improvement'] > 0 for r in rows])),
            mean_risk_improvement=float(np.mean([r['risk_improvement'] for r in rows])),
            mean_absolute_probability_change=float(np.mean([abs(r['probability_change']) for r in rows])))
    payload = dict(scope='JTEXT VAL160 offline one-step mechanism diagnostic; not shot-level performance',
                   checkpoint_sha256=simple_train.sha(args.checkpoint), seed=131,
                   code_sha256=simple_train.sha(__file__),
                   update_code_sha256=simple_train.sha(Path(__file__).with_name('fast_update.py')),
                   fast_rule=model.cfg.fast_rule,summary=summary, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
