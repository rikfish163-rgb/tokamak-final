"""Learn a state-dependent signal-only update rule; freeze the qualified source."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .simple23 import Config, Simple23
from .state_loss import StateLoss
from .sequence_fast import unroll
from .sequence_train import pools, sample
from .simple_online import predict_shot, _durable_json, _hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=512)
    args = p.parse_args()
    if args.steps < 1 or args.output.exists():
        raise ValueError('positive steps and new output directory required')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Simple23(Config(**parent['model_config']), parent['source_center'], parent['source_scale']).cuda()
    model.load_state_dict(parent['model'])
    model.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rule = StateLoss().cuda()
    optimizer = torch.optim.Adam(rule.parameters(), lr=.001)
    rng = np.random.default_rng(131)
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    shots = simple_train.load_shots(roles['train'], advanced)
    pool = pools(shots, model.cfg, 4)
    protocol = dict(parent_sha256=_hash(args.checkpoint), train_ids=roles['train'], val_ids=roles['val'],
                    steps=args.steps, depth=4, batch=8, lr=.001, seed=131,
                    objective='final post-update BCE; only 69-32-23 state weighting network trained',
                    context='matured signed/absolute residual and availability; detached from fast state',
                    threshold=parent['threshold'], scope='DEVELOPMENT_ONLY; no EAST or TEST data',
                    code_sha256={n: _hash(Path(__file__).with_name(n)) for n in
                        ('state_meta.py', 'state_loss.py', 'sequence_fast.py', 'sequence_train.py',
                         'simple_online.py', 'simple23.py', 'simple_data.py', 'fast_update.py')})
    args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json', protocol)
    history, digest = [], hashlib.sha256()
    for step in range(args.steps):
        episodes, manifest = sample(shots, pool, model.cfg, rng, 4, 8, 'cuda')
        digest.update(json.dumps(manifest).encode())
        optimizer.zero_grad(set_to_none=True)
        out = unroll(model, episodes, create_graph=True, loss_rule=rule)
        loss = F.binary_cross_entropy_with_logits(out['logits'][-1], episodes[-1]['y'])
        if not torch.isfinite(loss):
            raise FloatingPointError('invalid state-rule loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(rule.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(loss=float(loss.detach()), grad_norm=float(norm)))
        if (step+1) % 64 == 0:
            print(step+1, history[-1], flush=True)
    if not all(torch.equal(v, before[k]) for k, v in model.state_dict().items()):
        raise AssertionError('state learning modified base network')
    protocol['sample_sha256'] = digest.hexdigest()
    torch.save(dict(state={k: v.cpu() for k, v in rule.state_dict().items()}, protocol=protocol),
               args.output/'rule.pt')
    _durable_json(args.output/'training.json', dict(protocol=protocol, history=history))
    val = simple_train.load_shots(roles['val'], advanced)
    results = {}
    rule.eval().requires_grad_(False)
    for arm in ('static', 'uniform', 'state'):
        rows = []
        for shot in val:
            scores, _ = predict_shot(model, shot, intra=arm != 'static', device='cuda',
                                    loss_rule=rule if arm == 'state' else None)
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                             times=shot['times'][shot['valid']], scores=scores[shot['valid']]))
        results[arm] = simple_train.metrics(rows, parent['threshold'])
        _durable_json(args.output/'validation.json', dict(protocol=protocol, arms=results))
        print(arm, {k: v for k, v in results[arm].items() if k != 'predictions'}, flush=True)
    if not all(torch.equal(v, before[k]) for k, v in model.state_dict().items()):
        raise AssertionError('evaluation modified base network')


if __name__ == '__main__':
    main()
