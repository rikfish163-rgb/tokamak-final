"""Compare two completed label-policy runs without altering archived inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import flips


def read(root, name):
    return json.loads((root / name).read_text())


def compare(old, new):
    audits = [read(r, 'independent_audit.json') for r in (old, new)]
    if not all(a['complete360'] for a in audits):
        raise ValueError('two complete audited streams required')
    unchanged = ('order.json', 'input_map.json', 'source_replay.json', 'input_hashes.json')
    for name in unchanged:
        if read(old, name) != read(new, name):
            raise ValueError(f'changed comparison condition: {name}')
    protocols = [read(r, 'protocol.json') for r in (old, new)]
    for key in ('threshold', 'seed', 'source_sha256', 'source_statistics_sha256', 'contract'):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f'changed protocol: {key}')
    if protocols[0].get('target_loss', 'all_points') != 'all_points':
        raise ValueError('reference must be original point-label loss')
    if protocols[1]['target_loss'] != 'disruptive_positive_only':
        raise ValueError('candidate must be masked disruptive-early loss')
    n_values = 0
    for index, row in enumerate(read(new, 'order.json'), 1):
        filename = f"{index:04d}_{row['shot']}.npz"
        with np.load(old / 'shots' / filename) as a, np.load(new / 'shots' / filename) as b:
            for arm in ('times', 'source', 'aligned'):
                if not np.array_equal(a[arm], b[arm]):
                    raise ValueError(f'unchanged arm differs: {filename}/{arm}')
            n_values += b['source'].size
    predictions = [read(r, 'per_shot.json') for r in (old, new)]
    paired = {}
    for arm in protocols[1]['arms']:
        rows = [[dict(r, disruptive=r['is_disrupt']) for r in p[arm]] for p in predictions]
        paired[arm] = flips(*rows)
    receipts = read(new, 'receipts.json')
    output = dict(old_run=str(old.resolve()), new_run=str(new.resolve()),
                  unchanged_metadata=list(unchanged), identical_source_aligned_shots=360,
                  identical_values_per_arm=n_values, threshold=protocols[1]['threshold'],
                  old_metrics=audits[0]['metrics'], new_metrics=audits[1]['metrics'],
                  new_minus_old=paired,
                  fast_writes=sum(r['fast']['updates'] for r in receipts),
                  independent_future_validation=False)
    with (new / 'paired_label_audit.json').open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    args = parser.parse_args()
    compare(args.old, args.new)
