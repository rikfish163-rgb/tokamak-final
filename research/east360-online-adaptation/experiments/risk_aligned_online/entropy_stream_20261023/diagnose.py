"""Sealed-trace case diagnosis only; never changes inference or chooses thresholds."""
import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ
from .learning import ARMS
from .run import NAMES


def main(root):
    p,done,verified=READ(root/'protocol.json'),READ(root/'completed.json'),READ(root/'verification.json')
    assert verified['status']=='VERIFIED' and verified['completed_sha256']==sha(root/'completed.json')
    full=READ(root/'per_shot.json');rows={a:{r['shot']:r for r in v['predictions']} for a,v in full.items()}
    cases=[]
    for arm in ARMS:
        for kind,ids in verified['paired_flips'][arm].items():
            for sid in ids:
                d,down=io23.read_outcome('EAST',sid)
                with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                    valid=np.ones(len(z['times']),bool) if not d else (down-z['times'])*1000>=-1e-6
                    # Examine the alarm that was added/rescued, or the static
                    # alarm that disappeared/became too late. No cherry-picked peak.
                    decision_arm=arm if kind in ('rescued_timely','added_fp') else 'static'
                    candidates=np.flatnonzero(valid&(z[decision_arm]>=p['threshold']))
                    assert len(candidates)
                    point=int(candidates[0])
                    preceding=[u for u in READ(root/'shots'/f'{sid}_updates.json')[arm] if u['first_affected_prediction']<=point]
                    update=preceding[-1] if preceding else None
                    timely=valid&(np.round((down-z['times'])*1000,9)>30) if d else valid
                    case=dict(arm=arm,kind=kind,shot=sid,is_disrupt=d,probe_index=point,time=float(z['times'][point]),
                        lead_ms=round(float((down-z['times'][point])*1000),9) if d else None,
                        risk={a:float(z[a][point]) for a in NAMES},
                        maximum_eligible_risk={a:float(z[a][timely].max()) if timely.any() else None for a in NAMES},
                        static_result=rows['static'][sid],new_result=rows[arm][sid],preceding_update=update,
                        updates_before_probe=len(preceding))
                    if update:
                        observed=update['last_observation']
                        case.update(observed_time=float(z['times'][observed]),
                            parent_risk_at_observed=float(z['static'][observed]),
                            delay_after_update_ms=float((z['times'][point]-z['times'][observed])*1000))
                    cases.append(case)
    summaries={}
    for arm in ARMS:
        bykind={}
        for kind in verified['paired_flips'][arm]:
            cc=[c for c in cases if c['arm']==arm and c['kind']==kind]
            bykind[kind]=dict(shots=[c['shot'] for c in cc],
                preceded_by_negative_logit=sum(c['preceding_update'] is not None and c['preceding_update'].get('observed_logit_before',0)<0 for c in cc))
        summaries[arm]=bykind
    report=dict(completed_sha256=sha(root/'completed.json'),verification_sha256=sha(root/'verification.json'),
        diagnose_sha256=sha(Path(__file__)),threshold=p['threshold'],cases=cases,summary=summaries,
        interpretation='observational paired algorithm intervention; no plasma causal mechanism identification')
    _durable_json(root/'case_diagnostic.json',report)
    print(dict(cases=len(cases),summary=summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
