"""Completed-run descriptive diagnostics; never selects a deployed threshold."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..operating_curve_audit import operating_curve,operational_maxima
from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import TRAIN_ARMS


def main(root):
    read=lambda path:json.loads(path.read_text())
    verification=read(root/'verification.json');assert verification['status']=='VERIFIED'
    assert sha(root/'completed.json')==verification['completed_sha256']
    p=read(root/'protocol.json');complete=read(root/'completed.json');full=read(root/'per_shot.json')
    history=read(root/'training.json');rows={k:[] for k in full};timing={};updates={k:[] for k in p['arms']}
    for sid in p['contract']['shot_ids']:
        d,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            t=trace['times'];valid=np.ones(len(t),bool) if not d else (down-t)*1000>=-1e-6
            for name in rows:rows[name].append(dict(shot=sid,is_disrupt=d,down=down,times=t.copy(),valid=valid.copy(),scores=trace[name].copy()))
        for arm,values in read(root/'shots'/f'{sid}_updates.json').items():updates[arm].extend(values)
    operating={}
    for name,rs in rows.items():
        curve=operating_curve(rs,p['threshold']);maxima=operational_maxima(rs)
        assert curve['fixed_source_threshold']['tp']==complete['metrics'][name]['tp']
        assert curve['fixed_source_threshold']['fp']==complete['metrics'][name]['fp']
        d,s=maxima['positive'][:,None],maxima['negative'][None,:]
        operating[name]=dict(curve,timely_peak_ranking_auc=float(((d>s)+.5*(d==s)).mean()))
        disrupted=[r for r in full[name]['predictions'] if r['is_disrupt']]
        timing[name]=dict(no_alarm=[r['shot'] for r in disrupted if not r['alarm']],
            late_at_or_below30ms=[r['shot'] for r in disrupted if r['alarm'] and not r['timely']],
            tpr_if_lead_strictly_over10ms=sum(r['alarm'] and r['lead_ms']>10 for r in disrupted)/len(disrupted),
            tpr_if_any_preterminal_alarm=sum(r['alarm'] for r in disrupted)/len(disrupted),
            scope='diagnostic only; primary EAST rule remains strict >30ms')
    training={}
    for arm in TRAIN_ARMS:
        training[arm]={}
        for period,data in (('first8',history[:8]),('last8',history[-8:])):
            training[arm][period]={}
            for label in (False,True):
                selected=[r for h in data for r in h[arm]['records'] if r['is_disrupt']==label]
                training[arm][period]['disrupt' if label else 'safe']={
                    key:float(np.mean([r[key] for r in selected])) for key in ('risk_before','risk_after','forecast')}
    update_summary={a:dict(count=len(v),mean_forecast_before=float(np.mean([u['loss_before'] for u in v])),
        mean_forecast_after=float(np.mean([u['loss_after'] for u in v])),
        loss_decreased=sum(u['loss_after']<u['loss_before'] for u in v)) for a,v in updates.items()}
    result=dict(scope='POSTHOC DEVELOPMENT DIAGNOSTIC; all TEST labels used for operating curves',
        completed_sha256=sha(root/'completed.json'),verification_sha256=sha(root/'verification.json'),
        code_sha256=sha(Path(__file__)),operating=operating,training=training,
        warning_timing=timing,signal_fit_on_test=update_summary,learner_or_predictions_modified=False)
    _durable_json(root/'posthoc_diagnostic.json',result)
    print(json.dumps(dict(caps={k:v['posthoc_best_at_cap'] for k,v in operating.items()},
        warning_timing=timing,signal_fit_on_test=update_summary)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
