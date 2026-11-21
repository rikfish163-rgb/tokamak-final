"""Posthoc ranking diagnostic, never a deployed threshold calibration."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23, simple_data
from ..operating_curve_audit import operational_maxima, operating_curve
from ..simple_train import sha
from ..simple_online import _durable_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    target = args.run/'operating_diagnostic.json'
    if target.exists(): raise FileExistsError(target)
    check = json.loads((args.run/'verification.json').read_text())
    if check['status'] != 'VERIFIED': raise ValueError('independently verified complete run required')
    protocol = json.loads((args.run/'protocol.json').read_text())
    if sha(args.run/'completed.json') != check['completed_sha256']: raise ValueError('results changed')
    arms = ['static', *protocol['arms']]
    rows = {k: [] for k in arms}
    receipts = {r['shot']: r for r in json.loads((args.run/'prediction_receipts.json').read_text())}
    for sid in protocol['contract']['shot_ids'][20:]:
        path = args.run/'shots'/f'{sid}.npz'
        if sha(path) != receipts[sid]['prediction_sha256']: raise ValueError('prediction changed')
        s = simple_data.attach_outcome(simple_data.read_shot('EAST', sid, rate_contract='nominal'), 'EAST', {})
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data['times'], s['times']): raise ValueError('clock mismatch')
            for arm in arms:
                rows[arm].append(dict(shot=sid, is_disrupt=s['is_disrupt'], down=s['down'],
                    valid=s['valid'], times=s['times'], scores=data[arm].copy()))
    results = {}
    for arm, values in rows.items():
        curve = operating_curve(values, protocol['threshold'])
        maxima = operational_maxima(values)
        d, s = maxima['positive'][:, None], maxima['negative'][None, :]
        auc = float(((d > s).astype(float) + .5*(d == s)).mean())
        for key in ('tp', 'fp', 'tpr', 'fpr', 'J'):
            if curve['fixed_source_threshold'][key] != check['common_suffix340'][arm][key]:
                raise AssertionError('operational maxima disagree with immutable first alarms')
        results[arm] = dict(timely_peak_ranking_auc=auc, **curve)
    output = dict(scope='POSTHOC DIAGNOSTIC ON COMMON SUFFIX340; NOT DEPLOYED OR INDEPENDENT',
        input_verification_sha256=sha(args.run/'verification.json'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in
                     (Path(__file__), Path(__file__).parents[1]/'operating_curve_audit.py')},
        threshold_search_uses_all_340_outcomes=True, thresholds_used_for_inference=False,
        inference_and_model_unchanged=True, results=results)
    _durable_json(target, output)
    print(json.dumps({arm:dict(auc=v['timely_peak_ranking_auc'],
          caps=v['posthoc_best_at_cap']) for arm,v in results.items()}), flush=True)


if __name__ == '__main__': main()
