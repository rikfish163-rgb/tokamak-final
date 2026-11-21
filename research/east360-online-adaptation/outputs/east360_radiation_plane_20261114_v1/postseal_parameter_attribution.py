"""Read the frozen candidates to distinguish dose, lead time and real-update flips."""

from pathlib import Path

import numpy as np
from scipy.special import logit

from experiments.risk_aligned_online import commissioning_radiation_plane_20261114 as rad
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha, write


OUT = Path(__file__).resolve().parent


def first(above, valid, count):
    if len(above)<count:
        return None
    hit = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(above & valid,count).all(1))
    return int(hit[0])+count-1 if len(hit) else None


def summarize(use):
    v = np.asarray(use,dtype=float)
    return dict(count=len(v), q10_median_q90=np.quantile(v,[.1,.5,.9]).tolist() if len(v) else None)


def main():
    p,sealed,rows = (read(OUT/f'{k}.json') for k in ('protocol','seal','per_shot'))
    archive = read(rad.history.ARCHIVE/'seal.json')
    main = rows['0.2']
    required = ('bias_only','current_5',*rad.NEW)
    confirm = {k:{n:{'tp':0,'fp':0} for n in (1,2,3,5)} for k in required}
    late,earlier_cases = {}, {}
    dose = []
    fractions = []
    differences = np.zeros((3,3),dtype=int)
    for rank,sid in enumerate(p['order'],1):
        f = OUT/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f)!=sealed['curves'][f.name]:
            raise ValueError('candidate curve changed')
        with np.load(f,allow_pickle=False) as data:
            new = {k:data[k] for k in rad.NEW}
            dose_now = data['actual']
        old = rad.history.archived_shot(rank,sid,archive)
        flag,down,t = rad.group.hdf._raw_truth(sid)
        y = np.round((down-t)*1000.,9) if flag else np.zeros(len(t))
        valid = y>=-1e-6
        tau = p['original'] if rank<=p['startup'] else p['calibration']['0.2']
        curves = {k:(new if k in new else old)[k] for k in required}
        for k,v in curves.items():
            for n in (1,2,3,5):
                hit = first(v>=tau,valid,n)
                timely = bool(flag and hit is not None and y[hit]>30)
                alarm = hit is not None
                confirm[k][n]['tp'] += timely
                confirm[k][n]['fp'] += bool(not flag and alarm)
                if n==1 and (timely!=main[k][rank-1]['timely'] or alarm!=main[k][rank-1]['alarm']):
                    raise ValueError('first alarm scorer not reproduced')
        seen = dose_now>0.
        dose.extend(dose_now[seen])
        fractions.extend(dose_now[seen]/old['dose'][seen])
        for i,a in enumerate(rad.NEW):
            for j,b in enumerate(rad.NEW):
                differences[i,j] += int(np.sum((new[a]>=tau)!=(new[b]>=tau)))
        if rank>p['startup'] and flag and not main['current_5'][rank-1]['timely']:
            near = seen & (y>30.) & (y<=125.)
            if near.any():
                log = {k:logit(v[near].clip(1e-6,1.-1e-6)) for k,v in curves.items()}
                late[str(sid)] = dict(
                    points=int(near.sum()),
                    gap=float(logit(tau)-log['current_5'].max()),
                    max_gain={k:float((log[k]-log['current_5']).max()) for k in rad.NEW},
                    crosses={k:bool((curves[k][near]>=tau).any()) for k in rad.NEW})
        if sid in rad.CASES:
            earlier_cases[str(sid)] = {k:main[k][rank-1] for k in main}
    out = dict(scope='fixed main threshold, confirmation counted when last confirming point arrives',
        arms=rad.NEW, actual_dose=summarize(dose), fraction_of_legacy_dose=summarize(fractions),
        endpoint_crossing_disagreement=differences.tolist(), confirmation=confirm,
        hard_positive=dict(supported=len(late),median_gap=summarize([x['gap'] for x in late.values()]),
            max_gain={k:summarize([x['max_gain'][k] for x in late.values()]) for k in rad.NEW},
            crosses={k:[sid for sid,r in late.items() if r['crosses'][k]] for k in rad.NEW},by_shot=late),
        individual_first_alarm=earlier_cases)
    write(OUT/'postseal_parameter_attribution.json',out)
    print({name:out[name] for name in ('actual_dose','fraction_of_legacy_dose','confirmation')})


if __name__=='__main__':
    main()
