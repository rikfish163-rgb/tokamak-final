"""Recount confirmation and persistent misses from the two sealed carry replays."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logit

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[1]))

from experiments.risk_aligned_online.commissioning_coincident_support_audit_20261111 import _raw_truth
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha


def analyze():
    p,s = (read(RUN/k) for k in ('protocol.json','seal.json'))
    shadow = Path(p['carry_reference'])
    old = read(shadow/'seal.json')
    a,b = (read(f/'per_shot.json') for f in (shadow,RUN))
    hard_path = RUN.parent/'commissioning_seed_errors_20261111_v1/replay/independent_audit.json'
    hard = set(read(hard_path)['hard_shots'])
    names = [f'{rank:04d}_{sid}.npz' for rank,sid in enumerate(p['order'],1)]
    if (len(set(p['order']))!=360 or len(hard)!=33 or not hard.issubset(p['order'][p['startup']:])
            or p['carry_seal']!=sha(shadow/'seal.json')
            or any(set(x['curve_sha256'])!=set(names) for x in (s,old))):
        raise ValueError('incomplete sealed paired cohort')
    arms = ('reset_current','reset_earlier',*p['arms'])
    counts = {c:{n:{k:[0,0] for k in arms} for n in (1,2,3,5)} for c in p['calibration']}
    common_hard = {k:[] for k in arms}
    changes = {k:[] for k in ('current','earlier')}
    windows = []
    for rank,(sid,name) in enumerate(zip(p['order'],names,strict=True),1):
        f,g = (root/'shots'/name for root in (shadow,RUN))
        if sha(f)!=old['curve_sha256'][name] or sha(g)!=s['curve_sha256'][name]:
            raise ValueError(f'changed sealed curve: {name}')
        disruptive, down, t = _raw_truth(sid)
        valid = (down-t)*1000>=-1e-6 if disruptive else np.ones(len(t),bool)
        with np.load(f,allow_pickle=False) as z0,np.load(g,allow_pickle=False) as z:
            np.testing.assert_array_equal(z0['dose'],z['dose'])
            curves = {**{k:z0[k] for k in arms[:2]},**{k:z[k] for k in arms[2:]}}
            dose = z['dose']
        for k in changes:
            diff = np.abs(curves[f'own_{k}']-curves[f'carry_{k}'])
            changes[k].append((float(diff.max()),int((diff>1e-6).sum())))
        for band,cal in p['calibration'].items():
            tau = p['original'] if rank<=p['startup'] else cal
            for k,v in curves.items():
                original = score_curve(t,v,disruptive,down,tau)
                archived = (a if k in arms[:2] else b)[band][k][rank-1]
                if dict(shot=sid,disruptive=disruptive,down_s=down,**original)!=archived:
                    raise ValueError('raw first-alarm scoring differs')
                if band=='0.2' and sid in hard and original['timely']:
                    common_hard[k].append(sid)
                above = valid & (v>=tau)
                for n,values in counts[band].items():
                    confirmed = np.r_[np.zeros(n-1,bool), np.lib.stride_tricks.sliding_window_view(above,n).all(1)]
                    row = score_curve(t,confirmed,disruptive,down,.5)
                    values[k][0] += row['timely']
                    values[k][1] += not disruptive and row['alarm']
        if rank>p['startup'] and disruptive and not a['0.2']['reset_current'][rank-1]['timely']:
            lead = np.round((down-t)*1000,9)
            near = (dose>0.) & (lead>30.) & (lead<=125.)
            if not near.any():
                continue
            k = ('reset_current','carry_current','carry_earlier','own_current','own_earlier')
            x = logit(np.stack([curves[key][near] for key in k]).astype(np.float64))
            windows.append(dict(shot=sid,points=int(near.sum()),gap=float(logit(p['calibration']['0.2'])-x[0].max()),
                                **dict(zip(k[1:],(x[1:]-x[0]).max(1).tolist(),strict=True))))
    return dict(scope='SEALED_CARRY_AND_OWN_FORECAST_ATTRIBUTION',
                inputs=dict(carry_seal=sha(shadow/'seal.json'),own_seal=sha(RUN/'seal.json'),hard_audit=sha(hard_path)),
                raw_verified=360*len(arms)*len(counts), confirmation=counts,
                near_window=dict(supported=len(windows),medians={k:float(np.median([r[k] for r in windows]))
                    for k in windows[0] if k not in ('shot','points')},rows=windows),
                historical_common_miss=dict(shots=sorted(hard), timely=common_hard),
                own_vs_shadow_curve_changes={k:dict(max_abs=max(x[0] for x in v),changed_points=sum(x[1] for x in v),
                    changed_shots=sum(x[1]>0 for x in v)) for k,v in changes.items()})


if __name__=='__main__':
    print(json.dumps(analyze(),indent=2,allow_nan=False))
