"""Descriptive uncertainty, early-error accounting and posthoc operating curves."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..operating_curve_audit import operating_curve
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.audit import counts
from .run import ARMS,NAMES


def diagnose(root):
    read=lambda p:json.loads(p.read_text())
    p=read(root/'protocol.json');verified=read(root/'verification.json');full=read(root/'per_shot.json');done=read(root/'completed.json')
    assert verified['status']=='VERIFIED' and sha(root/'completed.json')==verified['completed_sha256']
    records={a:[] for a in NAMES};early={a:[] for a in NAMES}
    changes={a:{k:[] for k in ('safe','disrupt_early','disrupt_timely')} for a in ARMS}
    for sid in p['contract']['shot_ids']:
        d,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            times=z['times'];lead=np.round((down-times)*1000.,9) if d else None
            valid=lead>=-1e-6 if d else np.ones(len(times),bool)
            for a in NAMES:
                records[a].append(dict(shot=sid,is_disrupt=d,down=down,valid=valid.copy(),times=times.copy(),scores=z[a].copy()))
                if not d and (z[a][:256]>=p['threshold']).any():early[a].append(sid)
            masks=({'disrupt_early':valid & (lead>125),'disrupt_timely':valid & (lead>30) & (lead<=125)} if d else {'safe':valid.copy()})
            for mask in masks.values():mask[:256]=False
            for a in ARMS:
                delta=z[a+'_intra'].astype(float)-z[a+'_static']
                for key,mask in masks.items():
                    if mask.any():changes[a][key].append(dict(shot=sid,mean_change=float(delta[mask].mean()),max_change=float(delta[mask].max()),min_change=float(delta[mask].min())))
    curves={a:operating_curve(rows,p['threshold']) for a,rows in records.items()}
    uncertainty={a:counts(full[a]['predictions']) for a in NAMES}
    for a,c in curves.items():
        for k in ('tp','fp','tpr','fpr','J'):assert c['fixed_source_threshold'][k]==done['metrics'][a][k]
    result=dict(scope='POSTHOC DEVELOPMENT DIAGNOSIS ONLY; no prediction/model/threshold modified',
        verification_sha256=sha(root/'verification.json'),code_sha256=sha(Path(__file__)),
        descriptive_uncertainty=uncertainty,
        uncertainty_caveat='Wilson intervals assume independent shots; temporal dependence, one training seed and development selection are not removed',
        false_alarms_before_first_possible_update=early,
        curve_summary={a:{k:v for k,v in c.items() if k!='curve'} for a,c in curves.items()},
        prediction_change_summary={a:{k:dict(shots=len(v),median_shot_mean_change=float(np.median([r['mean_change'] for r in v])) if v else None,
            shots_mean_increased=sum(r['mean_change']>0 for r in v)) for k,v in groups.items()} for a,groups in changes.items()},
        individual_changes=changes)
    _durable_json(root/'posthoc_diagnostic.json',result)
    print(json.dumps(dict(scope=result['scope'],descriptive_uncertainty=uncertainty,
        early_false_alarm_counts={a:len(v) for a,v in early.items()},
        posthoc_caps={a:c['posthoc_best_at_cap'] for a,c in curves.items()},
        prediction_change_summary=result['prediction_change_summary'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();diagnose(a.root)
