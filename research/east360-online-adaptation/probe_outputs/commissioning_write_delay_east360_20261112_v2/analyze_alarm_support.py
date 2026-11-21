"""Classify sealed first alarms by whether first-block writes could reach them."""

import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[1]))

from experiments.risk_aligned_online.commissioning_coincident_support_audit_20261111 import _raw_truth
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha


def analyze():
    p,s,per = (read(RUN/k) for k in ('protocol.json','seal.json','per_shot.json'))
    waits = p['additional_wait_samples']
    bands = p['calibration']
    names = ('cold','before_first_write','unchanged_later','exposed_only')
    by = {b:{d:{c:[] for c in names} for d in waits} for b in bands}
    records = {b:{d:[] for d in waits} for b in bands}
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        path = RUN/'shots'/name
        if sha(path)!=s['curve_sha256'][name]:
            raise ValueError(f'changed curve: {name}')
        y,down,t = _raw_truth(sid)
        if y:
            continue
        with np.load(path,allow_pickle=False) as z:
            for b,cal in bands.items():
                tau = p['original'] if rank<=p['startup'] else cal
                for j,d in enumerate(waits):
                    arm = f'current_{d}'
                    v = z[arm]
                    score = score_curve(t,v,y,down,tau)
                    if dict(shot=sid,disruptive=y,down_s=down,**score)!=per[b][arm][rank-1]:
                        raise ValueError(f'raw score changed: {b}, {d}, {sid}')
                    inactive = z['dose'][j]==0.
                    np.testing.assert_array_equal(v[inactive],z['bias_only'][inactive])
                    above = np.flatnonzero(v>=tau)
                    if not len(above):
                        continue
                    frozen = above[inactive[above]]
                    first = int(above[0])
                    writes = np.flatnonzero(~inactive)
                    if rank<=p['startup']:
                        cat = 'cold'
                    elif not len(frozen):
                        cat = 'exposed_only'
                    elif not len(writes) or frozen[0]<writes[0]:
                        cat = 'before_first_write'
                    else:
                        cat = 'unchanged_later'
                    by[b][d][cat].append(sid)
                    release = z['release']+d
                    slot = np.searchsorted(release,above,side='right')
                    in_handoff = (slot>0) & (above-np.r_[0,release][slot]<8)
                    records[b][d].append(dict(shot=sid,category=cat,first_index=first,
                        first_zero_dose_crossing=int(frozen[0]) if len(frozen) else None,
                        first_write=int(writes[0]) if len(writes) else None,
                        strong_parent_alarmed=bool((z['bias_only']>=tau).any()),
                        any_crossing_outside_handoff=bool((~in_handoff).any())))
    budget = {b:{d:dict(false_alarms=sum(map(len,arms.values())),
        unchanged=sum(len(arms[c]) for c in names[:-1]), by_category=arms,
        suppressions_to_cap={k:max(0,sum(map(len,arms.values()))-cap)
                             for k,cap in (('0.10',18),('0.15',27),('0.20',36))})
        for d,arms in states.items()} for b,states in by.items()}
    return dict(scope='SEALED_FIXED_SUPPORT_FALSE_ALARM_BOUNDS',input_seal=sha(RUN/'seal.json'),
                observation='At a zero-dose point every compared curve equals its frozen strong parent. '
                            'An above-threshold unchanged point suffices for the shot to false alarm.',
                limitation='This only bounds methods restricted to these same write points and frozen parent. '
                           'Exposed alarms are not necessarily avoidable by an admissible direction.',
                summary=budget,records=records)


if __name__=='__main__':
    print(json.dumps(analyze(),ensure_ascii=False,indent=2,allow_nan=False))
