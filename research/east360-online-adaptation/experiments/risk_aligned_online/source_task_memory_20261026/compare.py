"""Post-run paired audit of fixed versus source-learned memory update rules."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..lifecycle_stream_20261025.audit import classify, counts, digest, flips
from ..lifecycle_stream_20261025.compare_label_policy import read


def validate_pair(fixed, learned):
    for key in fixed.keys() | learned.keys():
        if key != 'source_memory' and fixed.get(key) != learned.get(key):
            raise ValueError('changed paired protocol: '+key)
    a, b = fixed['source_memory'], learned['source_memory']
    if (a['kind'] != 'fixed' or b['kind'] != 'learned' or a['epoch'] != b['epoch']
            or a['epoch'] != 2 or a['source_protocol_sha256'] != b['source_protocol_sha256']):
        raise ValueError('source memory must be paired final-epoch fixed/learned artifacts')


def verify_source_receipt(receipt, expected_source_sha):
    path = Path(receipt['path'])
    if digest(path) != receipt['sha256']:
        raise ValueError('source memory artifact changed')
    parent = path.parent
    protocol, done, audit = [read(parent, n) for n in ('protocol.json', 'complete.json', 'audit.json')]
    if (digest(parent/'protocol.json') != receipt['source_protocol_sha256']
            or protocol['source_sha256'] != expected_source_sha
            or not done['complete'] or done['final_epoch'] != 2
            or done['checkpoints'][receipt['kind']] != receipt['sha256']
            or not audit['complete'] or not audit['source_unchanged']
            or audit['source_train'] != 1353 or audit['source_val'] != 160):
        raise ValueError('source-only teaching provenance mismatch')
    return dict(training_root=str(parent), audit_sha256=digest(parent/'audit.json'),
                checkpoint_sha256=receipt['sha256'])


def compare(fixed, learned):
    roots = [fixed, learned]
    audits = [read(root, 'independent_audit.json') for root in roots]
    if not all(a['complete360'] for a in audits):
        raise ValueError('two complete independently scored streams required')
    protocols = [read(root, 'protocol.json') for root in roots]
    validate_pair(*protocols)
    provenance = [verify_source_receipt(p['source_memory'], p['source_sha256']) for p in protocols]
    unchanged = ('order.json', 'input_map.json', 'source_replay.json', 'input_hashes.json')
    for filename in unchanged:
        if read(fixed, filename) != read(learned, filename):
            raise ValueError('changed paired input: '+filename)
    arms, threshold = protocols[0]['arms'], protocols[0]['threshold']
    rows = [{arm: [] for arm in arms} for _ in roots]
    effects = []
    changes = [{pair: dict(changed=0, total=0, absolute_sum=0., maximum=0.)
                for pair in ('intra_minus_cold', 'intra_minus_reverse')} for _ in roots]
    for index, item in enumerate(read(fixed, 'order.json'), 1):
        sid = item['shot']
        name = f'{index:04d}_{sid}.npz'
        disruptive, down = io23.read_outcome('EAST', sid)
        with np.load(fixed/'shots'/name) as a, np.load(learned/'shots'/name) as b:
            for arm in ('times', 'source', 'aligned'):
                if not np.array_equal(a[arm], b[arm]):
                    raise ValueError(f'unchanged curve differs: {name}/{arm}')
            for run_index, values in enumerate((a, b)):
                for arm in arms:
                    rows[run_index][arm].append(dict(shot=sid, disruptive=disruptive,
                        **classify(values['times'], values[arm], disruptive, down, threshold)))
                for key, reference in [('intra_minus_cold', 'cold'), ('intra_minus_reverse', 'intra_reverse')]:
                    delta = np.abs(values['intra'].astype(float)-values[reference])
                    summary = changes[run_index][key]
                    summary['changed'] += int(np.count_nonzero(delta))
                    summary['total'] += delta.size
                    summary['absolute_sum'] += float(delta.sum())
                    summary['maximum'] = max(summary['maximum'], float(delta.max()))
    for run_index, root in enumerate(roots):
        metrics = {arm: counts(r) for arm, r in rows[run_index].items()}
        if metrics != audits[run_index]['metrics']:
            raise ValueError('saved independent audit differs from reconstructed scores')
        for summary in changes[run_index].values():
            summary['mean_absolute'] = summary.pop('absolute_sum') / summary['total']
        receipts = read(root, 'receipts.json')
        effects.append(dict(kind=protocols[run_index]['source_memory']['kind'], metrics=metrics,
            flips=audits[run_index]['flips'], probability_changes=changes[run_index],
            fast_writes=sum(r['fast']['updates'] for r in receipts),
            accepted_updates=audits[run_index]['accepted_updates']))
    result = dict(fixed_run=str(fixed.resolve()), learned_run=str(learned.resolve()),
        source_provenance=provenance, identical_source_aligned_shots=360,
        unchanged_inputs=list(unchanged), threshold=threshold, effects=effects,
        learned_minus_fixed={arm: flips(rows[0][arm], rows[1][arm]) for arm in arms},
        scope='PAIRED_DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION',
        shadow_note='both_shadow disables fast state in the both trajectory; not a separate inter training run')
    with (learned/'paired_source_task_audit.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixed', type=Path)
    parser.add_argument('learned', type=Path)
    args = parser.parse_args()
    compare(args.fixed, args.learned)
