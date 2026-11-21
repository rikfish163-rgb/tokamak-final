"""Post-seal paired interpretation; never feeds labels back into prediction."""
import argparse
from pathlib import Path

import numpy as np

from ..diverse_support_20261023.run import READ
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import ARMS, METHODS


def main(root):
    output = root / 'posthoc_pairs.json'
    if output.exists():
        raise FileExistsError(output)
    done = READ(root / 'completed.json')
    audit = READ(root / 'verification.json')
    assert audit['completed_sha256'] == sha(root / 'completed.json')
    for name, digest in done['artifact_sha256'].items():
        assert sha(root / name) == digest, name
    rows = READ(root / 'per_shot.json')
    indexed = {a: {r['shot']: r for r in rows[a]['predictions']} for a in ARMS}
    assert all(len(v) == 360 and v.keys() == indexed['source'].keys() for v in indexed.values())
    pairs = {}
    comparisons = [('source', 'mapped_source_static')]
    comparisons += [(a + '_static', a + '_intra') for a in ('raw_fixed', 'raw_head', *METHODS)]
    comparisons += [('raw_' + scope + '_static', 'mapped_' + scope + '_static') for scope in ('fixed', 'head')]
    comparisons += [(prefix + '_fixed_static', prefix + '_head_static') for prefix in ('raw', 'mapped')]
    for before, after in comparisons:
        flips = {k: [] for k in ('rescued', 'lost', 'removed_fp', 'added_fp')}
        for sid, old in indexed[before].items():
            new = indexed[after][sid]
            assert old['is_disrupt'] == new['is_disrupt']
            if old['is_disrupt']:
                if not old['timely'] and new['timely']: flips['rescued'].append(sid)
                if old['timely'] and not new['timely']: flips['lost'].append(sid)
            else:
                if old['alarm'] and not new['alarm']: flips['removed_fp'].append(sid)
                if not old['alarm'] and new['alarm']: flips['added_fp'].append(sid)
        tp = len(flips['rescued']) - len(flips['lost'])
        fp = len(flips['added_fp']) - len(flips['removed_fp'])
        assert tp == rows[after]['tp'] - rows[before]['tp']
        assert fp == rows[after]['fp'] - rows[before]['fp']
        pairs[before + ' -> ' + after] = dict(**flips, delta_tp=tp, delta_fp=fp, delta_j=(tp-fp)/180.)
    interaction = {}
    for stage in ('static', 'intra'):
        def j(a): return (rows[a + '_' + stage]['tp'] - rows[a + '_' + stage]['fp']) / 180.
        interaction[stage] = j('mapped_head') - j('mapped_fixed') - j('raw_head') + j('raw_fixed')
    updates = {}
    for method in METHODS:
        records = [r for sid in indexed['source']
                   for r in READ(root / 'fold0/shots' / f'{sid}_updates.json')[method]]
        active = [r for r in records if r['gradient_steps']]
        max_difference = 0.
        for sid in indexed['source']:
            with np.load(root / 'fold0/shots' / f'{sid}.npz') as z:
                max_difference = max(max_difference, float(np.abs(z[method+'_static']-z[method+'_intra']).max()))
        updates[method] = dict(attempts=len(records), gradient_steps=sum(r['gradient_steps'] for r in active),
            nonzero_updates=sum(r['parameter_delta_norm'] > 0 for r in active),
            loss_decreased=sum(r['loss_after'] < r['loss_before'] for r in active),
            max_parameter_step=max(r['parameter_delta_norm'] for r in active), max_probability_difference=max_difference)
    result = dict(script_sha256=sha(Path(__file__)), completed_sha256=sha(root/'completed.json'),
        verification_sha256=sha(root/'verification.json'), pairs=pairs, mapping_by_head_interaction_j=interaction,
        updates=updates, warning='Post-seal development diagnosis; script added after main-run snapshot. No model or threshold selection.')
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
