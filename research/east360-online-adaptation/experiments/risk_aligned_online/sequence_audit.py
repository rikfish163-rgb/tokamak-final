"""VAL-only diagnosis of accumulated versus reset fast state; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .sequence_fast import sequence, unroll
from .simple23 import Config, Simple23


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        raise ValueError('positive steps and new output required')
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Simple23(Config(**ckpt['model_config']), ckpt['source_center'], ckpt['source_scale']).cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()
    rng = np.random.default_rng(131)
    roles = simple_data.source_roles()
    advanced = simple_data.advanced_times()
    gap = args.steps*(model.cfg.horizon+1)
    rows = []
    for sid in roles['val']:
        s = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        if len(s['values']) <= gap:
            continue
        starts = np.flatnonzero(np.convolve(np.asarray(s['valid'], int), np.ones(gap+1, int), 'valid') == gap+1)
        for label in (0, 1):
            pool = starts[(s['y'][starts+gap] > .5) == bool(label)]
            for start in rng.choice(pool, min(2, len(pool)), replace=False):
                episodes = sequence(s, int(start), args.steps, model.cfg, 'cuda')
                accumulated = unroll(model, episodes, create_graph=False)
                for index, e in enumerate(episodes):
                    reset = unroll(model, [e], create_graph=False)['logits'][0]
                    with torch.no_grad():
                        static = model(e['qx'], e['qa'])['logit']
                        a = accumulated['logits'][index]
                        def loss(z):
                            return float(F.binary_cross_entropy_with_logits(z, e['y']))
                        rows.append(dict(shot=int(sid), origin=int(start), step=index+1,
                                         label=int(e['y'].item()), is_disrupt=bool(s['is_disrupt']),
                                         static_loss=loss(static), reset_loss=loss(reset),
                                         accumulated_loss=loss(a),
                                         offset=float(accumulated['offsets'][index].item())))
    summary = []
    for step in range(1, args.steps+1):
        for label in (0, 1):
            group = [r for r in rows if r['step'] == step and r['label'] == label]
            if group:
                summary.append(dict(step=step, label=label, n=len(group),
                    reset_improvement=float(np.mean([r['static_loss']-r['reset_loss'] for r in group])),
                    accumulated_improvement=float(np.mean([r['static_loss']-r['accumulated_loss'] for r in group])),
                    mean_offset=float(np.mean([r['offset'] for r in group]))))
    payload = dict(scope='JTEXT VAL160 offline diagnostic, not shot-level or EAST performance',
                   checkpoint_sha256=simple_train.sha(args.checkpoint), steps=args.steps, seed=131,
                   code_sha256={n:simple_train.sha(Path(__file__).with_name(n)) for n in
                                ('sequence_audit.py', 'sequence_fast.py', 'fast_direction.py', 'fast_update.py', 'simple23.py')},
                   summary=summary, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
