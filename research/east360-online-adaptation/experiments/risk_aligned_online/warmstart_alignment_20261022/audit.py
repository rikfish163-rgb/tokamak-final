"""Independent re-score and input/statistic provenance checks for warm start."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from .alignment import SIGNAL_KEYS, fit_statistics, fit_prefix_map


def wilson(k, n):
    if not n: return None
    z = 1.959963984540054
    d = 1+z*z/n
    mid = (k/n+z*z/(2*n))/d
    width = z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/d
    return [mid-width, mid+width]


def counts(records):
    d = sum(r['is_disrupt'] for r in records); s = len(records)-d
    tp = sum(r['timely'] for r in records)
    fp = sum(r['alarm'] and not r['is_disrupt'] for r in records)
    return dict(tp=tp, fp=fp, n_disrupt=d, n_safe=s,
                tpr=tp/d if d else None, fpr=fp/s if s else None,
                J=tp/d-fp/s if d and s else None,
                tpr_wilson95=wilson(tp, d), fpr_wilson95=wilson(fp, s))


def flips(base, alternative):
    b = {r['shot']: r for r in base}; a = {r['shot']: r for r in alternative}
    if set(b) != set(a): raise AssertionError('paired IDs mismatch')
    return dict(
        rescued_TP=[s for s in b if b[s]['is_disrupt'] and not b[s]['timely'] and a[s]['timely']],
        lost_TP=[s for s in b if b[s]['is_disrupt'] and b[s]['timely'] and not a[s]['timely']],
        removed_FP=[s for s in b if not b[s]['is_disrupt'] and b[s]['alarm'] and not a[s]['alarm']],
        added_FP=[s for s in b if not b[s]['is_disrupt'] and not b[s]['alarm'] and a[s]['alarm']])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if (args.run/'verification.json').exists(): raise FileExistsError('verification already exists')
    protocol = json.loads((args.run/'protocol.json').read_text())
    completed = json.loads((args.run/'completed.json').read_text())
    original_rows = json.loads((args.run/'per_shot.json').read_text())
    contract = load_test_contract('EAST')
    if protocol['contract'] != contract: raise AssertionError('locked test changed')
    if sha(protocol['source_checkpoint']) != protocol['source_sha256']: raise AssertionError('parent changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest: raise AssertionError(f'code changed: {name}')
    source_records = [{k: s[k] for k in SIGNAL_KEYS} for s in
        (simple_data.read_shot('J-TEXT', sid) for sid in io23.source_roles()['train'])]
    source = fit_statistics(source_records, device='J-TEXT', allowed_ids=io23.source_roles()['train'])
    if source != json.loads((args.run/'source_statistics.json').read_text()):
        raise AssertionError('source statistics mismatch')
    del source_records
    order = contract['shot_ids']; threshold = protocol['threshold']
    receipts = json.loads((args.run/'prediction_receipts.json').read_text())
    if [r['shot'] for r in receipts] != order: raise AssertionError('receipt IDs mismatch')
    warm = [apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            for sid in order[:20]]
    warm = [{k: s[k] for k in SIGNAL_KEYS} for s in warm]
    maps = {}
    for name, (n, policy) in protocol['arms'].items():
        actual = json.loads((args.run/'maps'/f'{name}.json').read_text())
        check = fit_prefix_map(source, warm[:n], order=order, count=n, policy=policy)
        check['preceding_prediction_seals'] = {str(r['shot']): r['prediction_sha256'] for r in receipts[:n]}
        if check != actual: raise AssertionError(f'prefix mapping mismatch: {name}')
        maps[name] = actual
    arms = ['static', *protocol['arms']]
    fresh = {name: [] for name in arms}
    for index, sid in enumerate(order):
        path = args.run/'shots'/f'{sid}.npz'
        if sha(path) != receipts[index]['prediction_sha256']: raise AssertionError('trace hash changed')
        expected_maps = sorted(k for k, (n, _) in protocol['arms'].items() if index >= n)
        if receipts[index]['active_maps'] != expected_maps: raise AssertionError('map activation is not causal')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        d, down = io23.read_outcome('EAST', sid)
        valid = np.ones(len(raw['times']), bool) if not d else (down-raw['times'])*1000 >= -1e-6
        with np.load(path, allow_pickle=False) as trace:
            if set(trace.files) != {'times', *arms}: raise AssertionError('incomplete arms')
            if not np.array_equal(trace['times'], raw['times']): raise AssertionError('incomplete or shifted trace')
            for name in arms:
                q = trace[name]
                if (q.shape != raw['times'].shape or not np.isfinite(q).all()
                        or (q < 0).any() or (q > 1).any()): raise AssertionError('invalid probability')
                if name != 'static' and index < protocol['arms'][name][0]:
                    if not np.array_equal(q, trace['static']): raise AssertionError('warmup predictions revised')
                crossings = np.flatnonzero((q >= threshold) & valid)
                alarm = bool(len(crossings))
                lead = float(round((down-raw['times'][crossings[0]])*1000., 9)) if d and alarm else None
                row = dict(shot=sid, is_disrupt=bool(d), alarm=alarm, lead_ms=lead,
                           timely=bool(lead is not None and lead > 30.))
                fresh[name].append(row)
    for name in arms:
        if fresh[name] != original_rows[name]['predictions']: raise AssertionError('per-shot score mismatch')
        for subset, records in (('full360', fresh[name]), ('common_suffix340', fresh[name][20:]),
                                ('warmup20', fresh[name][:20])):
            c = counts(records)
            for key in ('tp', 'fp', 'n_disrupt', 'n_safe', 'tpr', 'fpr', 'J'):
                if c[key] != completed['metrics'][name][subset][key]: raise AssertionError('aggregate score mismatch')
    result = dict(status='VERIFIED', protocol_sha256=sha(args.run/'protocol.json'),
        completed_sha256=sha(args.run/'completed.json'),
        verified_predictions=len(order)*len(arms), source_statistics_recomputed=True,
        target_statistics_recomputed_from_exact_ended_prefix=True,
        warmup_class_counts={str(n):counts(fresh['static'][:n]) for n in (10, 20)},
        common_suffix340={name:dict(**counts(rows[20:]), **flips(fresh['static'][20:], rows[20:]))
                          for name, rows in fresh.items()},
        full360={name:counts(rows) for name, rows in fresh.items()},
        uncertainty='Wilson intervals are descriptive shot-level intervals; stream dependence and development selection are not removed',
        independent_confirmation=False)
    _durable_json(args.run/'verification.json', result)
    fields = ['feature', 'source_mean', 'source_std', 'source_median', 'source_iqr',
              'n10_mean', 'n10_std', 'n10_median', 'n10_iqr',
              'n20_mean', 'n20_std', 'n20_median', 'n20_iqr']
    with (args.run/'feature_statistics.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for c, name in enumerate(io23.RAW23):
            row = dict(feature=name)
            for prefix, stats in (('source', source), ('n10', maps['moments_all_n10']['target_statistics']),
                                  ('n20', maps['moments_all_n20']['target_statistics'])):
                for key in ('mean', 'std', 'median', 'iqr'): row[prefix+'_'+key] = stats[key][c]
            w.writerow(row)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
