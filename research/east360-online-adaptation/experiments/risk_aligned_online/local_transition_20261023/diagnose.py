"""Posthoc operating envelopes and missed-shot categories; no policy writeback."""
import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..operating_curve_audit import operating_curve
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import READ,NAMES


def diagnose(root):
    p,done=READ(root/'protocol.json'),READ(root/'completed.json')
    verified=READ(root/'verification.json');assert verified['status']=='VERIFIED'
    assert verified['completed_sha256']==sha(root/'completed.json')
    rows={a:[] for a in NAMES};full=READ(root/'per_shot.json')
    for sid in p['contract']['shot_ids']:
        disrupt,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            valid=(down-z['times'])*1000>=-1e-6 if disrupt else np.ones(len(z['times']),dtype=bool)
            for arm in NAMES:
                rows[arm].append(dict(shot=sid,times=z['times'].copy(),scores=z[arm].copy(),
                    valid=valid,is_disrupt=disrupt,down=down))
    curves={a:operating_curve(r,p['threshold']) for a,r in rows.items()}
    categories={}
    for a,c in curves.items():
        for key in ('tp','fp','tpr','fpr','J'):assert c['fixed_source_threshold'][key]==done['metrics'][a][key]
        d=[r for r in full[a]['predictions'] if r['is_disrupt']]
        categories[a]=dict(late_ids=[r['shot'] for r in d if r['alarm'] and not r['timely']],
                           never_alarm_ids=[r['shot'] for r in d if not r['alarm']])
    result=dict(scope='POSTHOC operating envelope using all TEST outcomes, NOT a deployable threshold or causal gain',
        thresholds_written_back=False,independent_confirmation=False,diagnose_sha256=sha(Path(__file__)),
        fixed_threshold_misses=categories,operating_curves=curves)
    _durable_json(root/'operating_diagnostic.json',result)
    print({a:c['posthoc_best_at_cap'] for a,c in curves.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);diagnose(p.parse_args().root)
