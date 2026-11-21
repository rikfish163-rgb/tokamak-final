"""Posthoc matched-FPR/risk-crossing diagnostics of immutable predictions."""
import argparse
from pathlib import Path
import numpy as np
from .. import io23
from ..operating_curve_audit import operating_curve
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import READ,NAMES


def main(root):
    p,done,v=[READ(root/f) for f in ('protocol.json','completed.json','verification.json')]
    assert v['status']=='VERIFIED' and v['completed_sha256']==sha(root/'completed.json')
    records={a:[] for a in NAMES};seals=READ(root/'prediction_receipts.json');details=[]
    changed={s for name,r in v['paired_contrasts'].items() if name.endswith('_fit_vs_prior') for k in ('rescued_timely','lost_timely') for s in r[k]}
    for seal in seals:
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';assert sha(path)==seal['prediction_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z:
            times=z['times'];valid=(down-times)*1000>=-1e-6 if d else np.ones(len(times),bool)
            detail=dict(shot=sid,curves={})
            for a in NAMES:
                score=z[a].copy();records[a].append(dict(shot=sid,times=times.copy(),scores=score,valid=valid,is_disrupt=d,down=down))
                if sid in changed:
                    timely=np.round((down-times)*1000,9)>30;hit=valid&(score>=p['threshold'])
                    indices=np.flatnonzero(hit);first=int(indices[0]) if len(indices) else None
                    detail['curves'][a]=dict(lead_ms=round(float((down-times[first])*1000),9) if first is not None else None,
                        timely_crossing_points=int((timely&hit).sum()),
                        timely_peak_margin=float(score[timely].max())-p['threshold'])
            if sid in changed:details.append(detail)
    curves={a:operating_curve(rs,p['threshold']) for a,rs in records.items()}
    for a,c in curves.items():
        assert all(c['fixed_source_threshold'][k]==done['metrics'][a][k] for k in ('tp','fp','tpr','fpr','J'))
    result=dict(scope='POSTHOC only, all TEST labels used to scan threshold; not deployment/independent evidence',
        code_sha256=sha(Path(__file__)),verification_sha256=sha(root/'verification.json'),
        curves=curves,changed_disruption_details=details)
    _durable_json(root/'posthoc_diagnostic.json',result)
    print({a:{cap:r['tpr'] for cap,r in c['posthoc_best_at_cap'].items()} for a,c in curves.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();main(args.root)
