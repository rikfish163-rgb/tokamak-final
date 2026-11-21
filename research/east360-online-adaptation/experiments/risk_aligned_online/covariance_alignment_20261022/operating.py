"""Posthoc ranking check only; never changes the deployed source threshold."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import simple_data
from ..operating_curve_audit import operational_maxima, operating_curve
from ..simple_train import sha
from ..simple_online import _durable_json


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); output = args.run/'operating_diagnostic.json'
    if output.exists(): raise FileExistsError(output)
    verified = json.loads((args.run/'verification.json').read_text())
    if verified['status'] != 'VERIFIED': raise ValueError('complete verified experiment required')
    protocol = json.loads((args.run/'protocol.json').read_text())
    receipts = {r['shot']:r for r in json.loads((args.run/'prediction_receipts.json').read_text())}
    arms = ['static', *protocol['arms']]; rows = {k:[] for k in arms}
    for sid in protocol['contract']['shot_ids'][20:]:
        path = args.run/'shots'/f'{sid}.npz'
        if sha(path) != receipts[sid]['prediction_sha256']: raise AssertionError('sealed prediction changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        s = simple_data.attach_outcome(raw, 'EAST', {})
        with np.load(path, allow_pickle=False) as trace:
            if not np.array_equal(s['times'], trace['times']): raise AssertionError('clock mismatch')
            for arm in arms:
                rows[arm].append(dict(shot=sid, is_disrupt=s['is_disrupt'], down=s['down'],
                                     valid=s['valid'], times=s['times'], scores=trace[arm].copy()))
    result = {}
    for arm, rs in rows.items():
        curve = operating_curve(rs, protocol['threshold']); maxima = operational_maxima(rs)
        d, s = maxima['positive'][:,None], maxima['negative'][None,:]
        auc = float(((d>s).astype(float)+.5*(d==s)).mean())
        for key in ('tp','fp','tpr','fpr','J'):
            if curve['fixed_source_threshold'][key] != verified['common_suffix340'][arm][key]:
                raise AssertionError('fixed operating point mismatch')
        result[arm] = dict(timely_peak_ranking_auc=auc, **curve)
    _durable_json(output, dict(scope='POSTHOC common suffix340; uses all outcomes for diagnostic threshold scan',
        threshold_search_used_for_inference=False, independent_confirmation=False,
        verification_sha256=sha(args.run/'verification.json'), completed_sha256=sha(args.run/'completed.json'),
        code_sha256=sha(Path(__file__)), results=result))
    print(json.dumps({name:dict(auc=r['timely_peak_ranking_auc'], caps=r['posthoc_best_at_cap'])
                      for name,r in result.items()}))


if __name__ == '__main__': main()
