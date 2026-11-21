"""Persist paired static/intra source replay without threshold reselection."""
import argparse
import json
from pathlib import Path

import torch

from . import simple_data, simple_train
from .simple23 import Config, Simple23
from .simple_online import predict_shot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    c = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Simple23(Config(**c['model_config']), c['source_center'], c['source_scale']).cuda()
    model.load_state_dict(c['model'])
    shots = simple_train.load_shots(simple_data.source_roles()[args.split], simple_data.advanced_times())
    result = dict(evidence='DEVELOPMENT_ONLY', split=args.split,
                  checkpoint_sha256=simple_train.sha(args.checkpoint),
                  code_sha256={n: simple_train.sha(Path(__file__).with_name(n)) for n in
                               ('evaluate_source_fast.py', 'simple_online.py', 'simple23.py','fast_update.py')}, arms={})
    for intra in (False, True):
        rows = []
        updates = 0
        for i, shot in enumerate(shots):
            scores, n = predict_shot(model, shot, intra=intra, device='cuda')
            updates += n
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                             times=shot['times'][shot['valid']], scores=scores[shot['valid']]))
            if (i+1) % 40 == 0:
                print(f'intra={intra} completed {i+1}/{len(shots)}', flush=True)
        metrics = simple_train.metrics(rows, c['threshold'], lead=10.)
        result['arms']['intra' if intra else 'static'] = dict(updates=updates, metrics=metrics)
        print(json.dumps({k:v for k,v in metrics.items() if k != 'predictions'}), flush=True)
    before = result['arms']['static']['metrics']['predictions']
    after = result['arms']['intra']['metrics']['predictions']
    result['changed_shots'] = [dict(before=a, after=b) for a,b in zip(before,after) if a != b]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
