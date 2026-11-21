"""Post-seal analysis of score shifts and equal-false-count rankings."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARMS = ('fixed','legacy_current4','response_plus_old','amplitude_plus_old','opposite_plus_old')
BUDGETS = (16,22,26,31,34,36,40)


def run():
    p,f,per = (json.loads((HERE/k).read_text()) for k in ('protocol.json','fit.json','per_shot.json'))
    audit = json.loads((HERE/'independent_audit.json').read_text())
    assert audit['scored_curves'] == 360*len(p['arms'])*3
    tau = p['calibration']['0.2']
    b = np.log(tau/(1.-tau))
    d = np.asarray(f['mask'],bool)
    cold = {}
    for key in ('old','new'):
        z = np.asarray(f[key+'_peak'])[0,1]
        cold[key] = dict(tp=int((z[d]>=b).sum()),fp=int((z[~d]>=b).sum()),
                        d_loss=float(np.logaddexp(0.,(b-z[d])/.35).mean()),
                        safe_loss=float(np.logaddexp(0.,(z[~d]-b)/.35).mean()))
    with np.load(HERE/'mapper.npz',allow_pickle=False) as map_file:
        w = map_file['signed'].astype(np.float64)
    labels = per['0.2']['legacy_current4']
    changes, maximum = defaultdict(list), defaultdict(lambda:defaultdict(list))
    for rank,row in enumerate(labels,1):
        sid,disruptive = row['shot'],row['disruptive']
        if rank<=p['startup']:
            continue
        with np.load(HERE/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as a:
            t = a['times']
            lead = (row['down_s']-t)*1000. if disruptive else np.zeros(len(t))
            for k in ARMS:
                curve = a[k].astype(np.float64)
                if disruptive:
                    maximum[k]['timely'].append(float(curve[lead>30.+1.e-6].max()))
                    maximum[k]['early'].append(float(curve[lead>3000.].max()) if (lead>3000.).any() else -1.)
                else:
                    maximum[k]['safe'].append(float(curve.max()))
            keep = a['eligible'] & (a['step_norm']>0.)
            i = a['release'][keep]
            if not len(i):
                continue
            op, rp = (a[k][i].astype(np.float64) for k in ('legacy_current4','response_plus_old'))
            z = a['compact'][keep]
            direction = np.linalg.norm(z[:,:3]@w[:3],axis=1)
            amplitude = np.linalg.norm(z[:,3:]@w[3:],axis=1)
            l1,l2 = (np.log(v.clip(1.e-7,1.-1.e-7))-np.log1p(-v.clip(1.e-7,1.-1.e-7)) for v in (op,rp))
            dt = l2-l1
            use = dict(all=np.ones(len(i),bool))
            if disruptive:
                use['timely_30_125'] = (lead[i]>30.+1.e-6)&(lead[i]<=125.+1.e-6)
                if not row['timely']:
                    use['missed_30_125'] = use['timely_30_125']
            else:
                use['safe'] = np.ones(len(i),bool)
            for group, valid in use.items():
                if valid.any():
                    changes[group].append((int(valid.sum()),int((dt[valid]<0).sum()),
                        int((dt[valid & (z[:,0]>0)]<0).sum()),int((valid&(z[:,0]>0)).sum()),
                        float(dt[valid].mean()),float(direction[valid].mean()),float(amplitude[valid].mean())))
    direction_shift = {}
    for k,v in changes.items():
        v = np.asarray(v)
        direction_shift[k] = dict(shots=len(v),release_windows=int(v[:,0].sum()),
            logit_lower_fraction=float(v[:,1].sum()/v[:,0].sum()),
            logit_lower_when_outward_fraction=float(v[:,2].sum()/v[:,3].sum()),
            mean_logit_shift_shot_equal=float(v[:,4].mean()),
            mean_direction_contribution_norm=float(v[:,5].mean()),
            mean_amplitude_contribution_norm=float(v[:,6].mean()))
    prefix = labels[:p['startup']]
    base_tp = sum(r['disruptive'] and r['timely'] for r in prefix)
    base_fp = sum(not r['disruptive'] and r['alarm'] for r in prefix)
    base_early = sum(r['disruptive'] and r['alarm'] and r['lead_ms']>3000. for r in prefix)
    budget = {}
    for limit in BUDGETS:
        budget[str(limit)] = {}
        for k,m in maximum.items():
            safety = np.sort(m['safe'])[::-1]
            threshold = np.nextafter(safety[limit-base_fp],np.inf)
            hit = np.asarray(m['timely'])>=threshold
            early = np.asarray(m['early'])>=threshold
            budget[str(limit)][k] = dict(
                tp=base_tp+int(hit.sum()),fp=base_fp+int((safety>=threshold).sum()),
                timely_30_3000=base_tp-base_early+int((hit&~early).sum()))
    output = dict(scope='postseal equal false count is a retrospective ranking, not a calibration',
                  cold_current_parent_signed=cold, signed_projection_effect=direction_shift,
                  equal_false_count=budget)
    with (HERE/'fit_bias_and_equal_fp.json').open('x') as f:
        json.dump(output,f,indent=2)
    print(dict(cold=cold,miss=direction_shift['missed_30_125'],at_26=budget['26'],at_34=budget['34']))


if __name__ == '__main__':
    run()
