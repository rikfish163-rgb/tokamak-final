"""First30 diagnostic: input processing, with optional ended-shot supervision.

Uses the existing locked order and fixed source threshold. Optional supervised
training is restricted to first30; evaluation persistent weights never change.
This is not a replication of EFD/PGFE-U or a full-stream prequential result.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from . import io23, simple_data
from .alignment import fit_ended_shot_statistics
from .simple23 import Config, Simple23, add_relative_input
from .simple_online import _durable_json, _hash, run_arm
from .source_gate import qualify_source
from .ended_adaptation import fit_ended


class MaskedData:
    window = staticmethod(simple_data.window)

    def __init__(self, usable):
        self.usable = usable
        self.completed = 0

    def read_shot(self, device, sid):
        s = simple_data.read_shot(device, sid)
        s['available'] &= self.usable[None, :]
        s['values'][~s['available']] = 0.
        return s

    def attach_outcome(self, s, device, advanced):
        result = simple_data.attach_outcome(s, device, advanced)
        self.completed += 1
        if self.completed % 10 == 0:
            print(f'sealed/revealed {self.completed} evaluation shots', flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--order', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--arm', choices=['mask_only', 'aligned'], required=True)
    parser.add_argument('--supervised-steps', type=int, default=0,
                        help='0=input-only; positive=diagnostic training on ended first30')
    parser.add_argument('--forecast-steps', type=int, default=256)
    parser.add_argument('--eval-arm', choices=['static', 'intra'], default='static')
    parser.add_argument('--sampling', choices=['endpoint', 'three_group'], default='endpoint')
    parser.add_argument('--relative-adapter', action='store_true')
    args = parser.parse_args()
    if args.supervised_steps < 0 or args.forecast_steps < 1:
        parser.error('nonnegative supervised steps and positive forecast steps required')
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('do not overwrite a previous diagnostic')
    gate = qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError(gate['reasons'])
    order = json.loads(args.order.read_text())
    ids = [int(r['shot']) for r in order]
    if len(ids) != 360 or len(set(ids)) != 360 or set(ids) != set(io23.cohort_ids('EAST')):
        raise ValueError('complete locked balanced360 order required')
    # Trusted local project checkpoint, never an external serialized upload.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt['config']['format'] != 'simple23-v1':
        raise ValueError('simple23-v1 checkpoint required')
    model = Simple23(Config(**ckpt['model_config']), ckpt['source_center'], ckpt['source_scale'])
    model.load_state_dict(ckpt['model'])
    if args.relative_adapter:
        model = add_relative_input(model)
    params_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    stats = fit_ended_shot_statistics(
        [simple_data.read_shot('EAST', sid) for sid in ids[:30]], ended_ids=ids[:30])
    if args.arm == 'aligned':
        model.center.copy_(torch.from_numpy(stats['center']))
        model.scale.copy_(torch.from_numpy(stats['scale']))
    args.output.mkdir(parents=True)
    _durable_json(args.output / 'statistics.json',
                  {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()})
    _durable_json(args.output / 'protocol.json', dict(
        arm=args.arm, checkpoint_sha256=_hash(args.checkpoint), order_sha256=_hash(args.order),
        calibration_ids=ids[:30], evaluation_ids=ids[30:], threshold=float(ckpt['threshold']),
        unusable_channels=[io23.RAW23[i] for i in np.flatnonzero(~stats['usable'])],
        stats_sha256=_hash(args.output / 'statistics.json'),
        code_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                     ('alignment.py', 'run_alignment_diagnostic.py', 'simple_online.py',
                      'simple_data.py', 'simple23.py', 'simple_train.py', 'io23.py')},
        supervised_steps=args.supervised_steps, forecast_steps=args.forecast_steps,
        evaluation_arm=args.eval_arm,
        supervised_sampling=args.sampling,
        relative_adapter=args.relative_adapter,
        extra_code_sha256={'ended_adaptation.py': _hash(Path(__file__).with_name('ended_adaptation.py'))},
        scope='DIAGNOSTIC_ONLY; first30 signals and optional ended labels; evaluate disjoint330'))
    if args.supervised_steps:
        data_api = MaskedData(stats['usable'])
        shots = [simple_data.attach_outcome(data_api.read_shot('EAST', sid), 'EAST', {})
                 for sid in ids[:30]]
        fitted = fit_ended(model.to(args.device), shots, ended_ids=ids[:30],
                           steps=args.supervised_steps, forecast_steps=args.forecast_steps,
                           device=args.device, sampling=args.sampling,
                           update_scope='relative' if args.relative_adapter else 'last_block')
        _durable_json(args.output / 'training.json', fitted)
        torch.save(dict(model=model.state_dict(), model_config=model.cfg.to_dict(),
                        threshold=float(ckpt['threshold']), parent_sha256=_hash(args.checkpoint)),
                   args.output / 'adapted30.pt')
        print('ended30 supervised and forecast training complete', flush=True)
    params_for_evaluation = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    print(f'{args.arm}: calibration complete; evaluating 330, masked '
          f'{[io23.RAW23[i] for i in np.flatnonzero(~stats["usable"])]}', flush=True)
    result = run_arm(model.to(args.device).eval(), order=deepcopy(order[30:]), arm=args.eval_arm,
                     threshold=float(ckpt['threshold']), output_dir=args.output / 'evaluation',
                     data_api=MaskedData(stats['usable']), device=args.device, startup_budget=0)
    changed = [name for name, p in model.named_parameters()
               if not torch.equal(p.detach().cpu(), params_before[name])]
    evaluation_changes = [name for name, p in model.named_parameters()
                          if not torch.equal(p.detach().cpu(), params_for_evaluation[name])]
    if evaluation_changes or (changed and not args.supervised_steps):
        raise AssertionError(f'unexpected persistent changes: {evaluation_changes or changed}')
    _durable_json(args.output / 'result.json', dict(
        diagnostic_arm=args.arm, neural_parameter_changes=changed,
        evaluation_persistent_changes=evaluation_changes,
        metrics=result['metrics_full']))
    print(json.dumps({k: v for k, v in result['metrics_full'].items() if k != 'predictions'}), flush=True)


if __name__ == '__main__':
    main()
