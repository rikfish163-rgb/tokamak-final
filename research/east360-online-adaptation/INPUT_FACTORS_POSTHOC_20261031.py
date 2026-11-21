"""Posthoc ranking/cap diagnosis; thresholds never enter any online predictor."""
import json
from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.operating_curve_audit import operating_curve, operational_maxima


def main():
    root = Path(__file__).parent
    run = root/'probe_outputs/input_factors360_v1'
    done = json.loads((run/'complete.json').read_text())
    assert done['complete']
    p = json.loads((run/'protocol.json').read_text())
    rows = {n:[] for n in p['arms']}
    for i,sid in enumerate(p['order']):
        flag,down = io23.read_outcome('EAST',sid)
        with np.load(run/'shots'/f'{i+1:04d}_{sid}.npz',allow_pickle=False) as a:
            valid = np.ones(len(a['times']),bool) if not flag else (down-a['times'])*1000>=-1e-6
            for n in rows:
                rows[n].append(dict(shot=sid,is_disrupt=flag,down=down,times=a['times'].copy(),
                                    scores=a[n].copy(),valid=valid))
    result = {}
    for n,records in rows.items():
        result[n] = {}
        for label,subset in [('full360',records),('suffix340',records[20:])]:
            curve = operating_curve(subset,p['threshold'])
            maxima = operational_maxima(subset)
            pos,neg = maxima['positive'][:,None],maxima['negative'][None,:]
            auc = float(np.mean((pos>neg)+.5*(pos==neg)))
            fixed = curve['fixed_source_threshold']
            assert fixed['tp']==done['results'][n][label]['tp'] and fixed['fp']==done['results'][n][label]['fp']
            result[n][label] = dict(timely_peak_auc=auc,caps=curve['posthoc_best_at_cap'])
    out = dict(scope='EXPLORATORY_FUTURE_LABEL_ORACLE_NOT_DEPLOYED',
               outcome_authority='HDF5 meta/IsDisrupt and meta/DownTime',
               actual_threshold_unchanged=True,uses_all_evaluation_labels=True,
               no_independent_confirmation=True,results=result)
    with Path(__file__).with_suffix('.json').open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
    print({n:v['full360'] for n,v in result.items()})


if __name__=='__main__':
    main()
