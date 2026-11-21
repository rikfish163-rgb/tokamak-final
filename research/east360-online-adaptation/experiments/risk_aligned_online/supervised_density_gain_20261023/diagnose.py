"""Descriptive curves only; never writes a threshold into the learner."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..operating_curve_audit import operating_curve
from ..simple_online import _durable_json
from ..simple_train import sha


def diagnose(root):
    read=lambda p:json.loads(p.read_text())
    p=read(root/'protocol.json');verified=read(root/'verification.json')
    assert verified['status']=='VERIFIED' and sha(root/'completed.json')==verified['completed_sha256']
    reference=Path(p['reference']);full=read(root/'per_shot.json');oldfull=read(reference/'per_shot.json')
    rows={a:[] for a in p['arms']};rows['raw_source']=[]
    byclass={'disruptive':[],'safe':[]}
    for sid in p['contract']['shot_ids']:
        d,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z,np.load(reference/'shots'/f'{sid}.npz',allow_pickle=False) as old:
            valid=np.ones(len(z['times']),bool) if not d else (down-z['times'])*1000>=-1e-6
            for a in rows:
                prob=old[a] if a=='raw_source' else z[a]
                rows[a].append(dict(shot=sid,is_disrupt=d,down=down,times=z['times'].copy(),scores=prob.copy(),valid=valid))
            delta=(z['learned_static'].astype(float)-z['original_static'])[valid]
            byclass['disruptive' if d else 'safe'].append(dict(shot=sid,mean_change=float(delta.mean()),
                fraction_increased=float((delta>1e-6).mean()),fraction_decreased=float((delta< -1e-6).mean())))
    curves={a:operating_curve(rs,p['threshold']) for a,rs in rows.items()}
    result=dict(scope='POSTHOC DEVELOPMENT DIAGNOSIS: all360 outcomes used to sweep thresholds, not operational results',
        verified_sha256=sha(root/'verification.json'),diagnostic_code_sha256=sha(Path(__file__)),
        deployment_threshold_unchanged=True,selected_model_unchanged=True,
        curve_summary={a:{k:v for k,v in c.items() if k!='curve'} for a,c in curves.items()},
        raw_to_learned_static=paired_flips(oldfull['raw_source']['predictions'],full['learned_static']['predictions']),
        equal_shot_probability_change={k:dict(n=len(v),mean_of_shot_means=float(np.mean([r['mean_change'] for r in v])),
            mean_fraction_increased=float(np.mean([r['fraction_increased'] for r in v])),
            mean_fraction_decreased=float(np.mean([r['fraction_decreased'] for r in v]))) for k,v in byclass.items()},
        individual_probability_change=byclass)
    _durable_json(root/'posthoc_diagnostic.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='individual_probability_change'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();diagnose(a.root)
