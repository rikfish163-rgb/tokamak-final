"""Fixed-source VAL160 probe for missing diagnostics, not deployment evaluation."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from . import io23, simple_data, simple_train
from .simple23 import Config, Simple23
from .simple_online import _durable_json, _hash


def mask_inputs(shots, channels):
    result = []
    indices = [io23.RAW23.index(name) for name in channels]
    for s in shots:
        copy = dict(s, values=s['values'].copy(), available=s['available'].copy())
        copy['available'][:, indices] = False
        copy['values'][:, indices] = 0.
        result.append(copy)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Simple23(Config(**ckpt['model_config']), ckpt['source_center'], ckpt['source_scale'])
    model.load_state_dict(ckpt['model'])
    model.cuda().eval()
    before = deepcopy(model.state_dict())
    shots = simple_train.load_shots(simple_data.source_roles()['val'], simple_data.advanced_times())
    result = dict(scope='DEVELOPMENT_ONLY; VAL160 fixed source threshold; no training or calibration',
                  checkpoint_sha256=_hash(args.checkpoint), threshold=ckpt['threshold'], arms={})
    for label, channels in [('observed', []), ('without_MNM', ['MNM']),
                            ('without_MNM_CIII', ['MNM', 'CIII'])]:
        rows = simple_train.score(model, mask_inputs(shots, channels), 'cuda')
        measured = simple_train.metrics(rows, ckpt['threshold'])
        result['arms'][label] = measured
        _durable_json(args.output, result)
        print(label, {k: v for k, v in measured.items() if k != 'predictions'}, flush=True)
    if not all(torch.equal(value, before[key]) for key, value in model.state_dict().items()):
        raise AssertionError('probe changed source model')
    result['model_unchanged'] = True
    _durable_json(args.output, result)


if __name__ == '__main__':
    main()
