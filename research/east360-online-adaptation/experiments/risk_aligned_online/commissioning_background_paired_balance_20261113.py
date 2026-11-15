"""Paired composition sensitivity of the frozen feedback contrasts."""

import argparse
from pathlib import Path

import numpy as np

from .commissioning_background_input_audit_20261113 import PREFIX, balanced_sample, quantile_pct
from .history_gradient_metric_20261102.run import read, sha, write


def run(parent, background, out):
    p,s = (read(parent/f'{k}.json') for k in ('protocol','seal'))
    q,b = (read(background/f'{k}.json') for k in ('protocol','seal'))
    balance = read(background/'postseal_bt_scale_balance.json')
    if (p['order']!=q['order'] or p['calibration']!=q['calibration']
            or len(s['curve_sha256'])!=2953 or len(b['curve_sha256'])!=2953
            or p['background_seal_sha256']!=sha(background/'seal.json')
            or balance['seal_sha256']!=p['background_seal_sha256']
            or sha(PREFIX/'order.json')!=read(PREFIX/'roster.json')['order_sha256']):
        raise ValueError('frozen paired cohorts and fixed input strata are required')
    a,z = (read(d/'per_shot.json') for d in (parent,background))
    lookup = {r['shot']:r['initial_abs_Bt_raw'] for r in read(PREFIX/'order.json')}
    take = [i for i,r in enumerate(p['order']) if r['cohort']=='extension']
    ids = [p['order'][i]['shot'] for i in take]
    y = np.array([z['0.2']['fixed'][i]['disruptive'] for i in take])
    old = np.array([p['order'][i]['old835'] for i in take])
    low = np.array([lookup[k]<balance['raw_split'] for k in ids])
    strata = []
    for g in (old,~old):
        for s0 in (low,~low):
            pools = [np.flatnonzero(g&s0&class_mask) for class_mask in (y,~y)]
            strata.append((pools,min(map(len,pools))))
    rng = np.random.default_rng(131)
    samples = np.stack([balanced_sample(strata,rng) for _ in range(1000)])
    if sorted(ids[i] for i in samples[0].ravel())!=sorted(balance['matched_ids']):
        raise ValueError('different fixed balanced cohort')
    n = samples.shape[-1]
    records = {}
    for band in ('0.2','0.15'):
        curves = a[band]|z[band]
        records[band] = {}
        for family in ('level','drift'):
            values = {}
            for arm in ('parent','earlier','current'):
                name = f'{family}_{arm}'
                selected = [curves[name][i] for i in take]
                if [r['shot'] for r in selected]!=ids or not np.array_equal([r['disruptive'] for r in selected],y):
                    raise ValueError('scored shot identity or outcome changed')
                values[arm] = tuple(np.array([r[k] for r in selected])[samples[:,j]].sum(1)
                                    for j,k in enumerate(('timely','alarm')))
            t,f = values['current']
            result = {}
            for ref in ('parent','earlier'):
                t0,f0 = values[ref]
                result[f'current_vs_{ref}'] = dict(
                    delta_tpr_pp=quantile_pct((t-t0)/n),delta_fpr_pp=quantile_pct((f-f0)/n),
                    fraction_more_timely=float((t>t0).mean()),
                    fraction_within_cap_and_more_timely={str(c):float(((f/n<=c)&(t>t0)).mean())
                                                       for c in (.10,.15,.20)})
            records[band][family] = result
    write(out,dict(scope='PAIRED_HISTORICAL_COMPOSITION_SENSITIVITY',n_each=n,n_draws=len(samples),
        bands=records,input_seals={'parent':sha(parent/'seal.json'),'background':sha(background/'seal.json')}))
    print(records,flush=True)


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('parent',type=Path)
    a.add_argument('background',type=Path)
    a.add_argument('out',type=Path)
    p = a.parse_args()
    run(p.parent.resolve(),p.background.resolve(),p.out.resolve())
