"""Paired readout audit; post-run only, never supplies model-selection labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import flips
from .compare_label_policy import read


def validate_protocols(old, new):
    for key in ('threshold', 'seed', 'source_sha256', 'source_statistics_sha256',
                'contract', 'warmup', 'target_loss', 'arms', 'torch', 'gpu'):
        if old[key] != new[key]:
            raise ValueError(f'changed protocol: {key}')
    # Only pre-option lifecycle archives omit readout; their code hardcoded shape.
    if old.get('readout', 'shape') != 'shape' or new['readout'] != 'full':
        raise ValueError('readout comparison requires archived shape and explicit full')


def compare(old, new):
    audits = [read(r, 'independent_audit.json') for r in (old, new)]
    if not all(a['complete360'] for a in audits):
        raise ValueError('two complete audited streams required')
    unchanged = ('order.json', 'input_map.json', 'source_replay.json', 'input_hashes.json')
    for name in unchanged:
        if read(old, name) != read(new, name):
            raise ValueError(f'changed comparison condition: {name}')
    protocols = [read(r, 'protocol.json') for r in (old, new)]
    validate_protocols(*protocols)
    n_values, changed, totals = 0, 0, 0.
    max_difference, per_shot = 0., []
    for index, row in enumerate(read(new, 'order.json'), 1):
        filename = f"{index:04d}_{row['shot']}.npz"
        with np.load(old / 'shots' / filename) as a, np.load(new / 'shots' / filename) as b:
            for arm in ('times', 'source', 'aligned'):
                if not np.array_equal(a[arm], b[arm]):
                    raise ValueError(f'unchanged arm differs: {filename}/{arm}')
            n_values += b['source'].size
            difference = np.abs(b['intra'].astype(float)-b['cold'])
            changed += np.count_nonzero(difference)
            totals += difference.sum()
            max_difference = max(max_difference, float(difference.max()))
            per_shot.append(dict(shot=row['shot'], changed=int(np.count_nonzero(difference)),
                                 max_absolute_probability_delta=float(difference.max())))
    predictions = [read(r, 'per_shot.json') for r in (old, new)]
    paired = {}
    for arm in protocols[1]['arms']:
        rows = [[dict(r, disruptive=r['is_disrupt']) for r in p[arm]] for p in predictions]
        paired[arm] = flips(*rows)
    receipts = read(new, 'receipts.json')
    energies = [r['fast']['readout_energy'] for r in receipts if r['index'] > 20]
    if len(energies) != 340:
        raise ValueError('all 340 adapted shots need readout diagnostics')
    summed = {k:sum(e[k] for e in energies) for k in ('full', 'level', 'shape')}
    if not np.isclose(summed['full'], summed['level']+summed['shape'], rtol=1e-9):
        raise ValueError('orthogonal readout energy identity failed')
    fractions = [e['level']/e['full'] for e in energies if e['full'] > 0]
    output = dict(old_run=str(old.resolve()), new_run=str(new.resolve()),
                  unchanged_metadata=list(unchanged), identical_source_aligned_shots=360,
                  identical_values_per_arm=n_values, threshold=protocols[1]['threshold'],
                  old_metrics=audits[0]['metrics'], new_metrics=audits[1]['metrics'],
                  new_minus_old=paired, fast_writes=sum(r['fast']['updates'] for r in receipts),
                  readout_energy=summed, level_energy_fraction=summed['level']/summed['full'],
                  per_shot_level_fraction_quantiles=np.quantile(fractions, [0,.25,.5,.75,1]).tolist(),
                  intra_probability_change=dict(changed_endpoints=int(changed),
                      total_endpoints=n_values, mean_absolute=float(totals/n_values),
                      max_absolute=max_difference, per_shot=per_shot),
                  evidence_scope='POSTHOC_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION')
    with (new / 'paired_readout_audit.json').open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in output.items() if k != 'intra_probability_change'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    args = parser.parse_args()
    compare(args.old, args.new)
