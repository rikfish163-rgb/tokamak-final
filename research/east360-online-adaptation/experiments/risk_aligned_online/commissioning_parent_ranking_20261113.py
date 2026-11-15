"""Postseal matched-false-alarm comparison of fixed readout variants."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import commissioning_parent_readout_20261113 as audit
from .history_gradient_metric_20261102.run import read, sha, write
from .operating_point_audit import best_at_count, timely_auc


def run(out, output):
    p,s,r = (read(out/f'{k}.json') for k in ('protocol','seal','result'))
    if r['status']!='COMPLETE_FROZEN_PARENT_READOUT_AUDIT' or len(s['curve_sha256'])!=len(p['order']):
        raise ValueError('completed parent-readout audit required')
    refs = {k:Path(v['path']) for k,v in p['references'].items()}
    if (any(sha(d/'seal.json')!=p['references'][k]['seal_sha256'] for k,d in refs.items())
            or sha(audit.WIDE/'balanced_cohort.json')!=p['balanced_manifest']):
        raise ValueError('sealed reference or balance changed')
    seals = {k:read(d/'seal.json') for k,d in refs.items()}
    per = read(out/'per_shot.json')['0.2']['fixed']
    balanced = {v['shot'] for v in read(audit.WIDE/'balanced_cohort.json')['order']}
    masks = {
        'fixed340': lambda row:row['cohort']=='fixed360' and row['rank']>p['startup'],
        'extension': lambda row:row['cohort']=='extension',
        'balanced': lambda row:row['shot'] in balanced,
        'old835': lambda row:row.get('old835') is True,
        'other1758': lambda row:row.get('old835') is False,
    }
    rows = {a:[] for a in audit.ARMS}
    bounds = {b:Counter() for b in ('0.2','0.15')}
    for i,entry in enumerate(p['order'],1):
        sid,rank,cohort = (entry[k] for k in ('shot','rank','cohort'))
        d,down,t = audit.group.hdf._raw_truth(sid)
        a = per[i-1]
        if (a['shot'],a['disruptive'],a['down_s'])!=(sid,d,down):
            raise ValueError('raw HDF5 disagrees with archived scoring')
        if cohort=='fixed360' and rank<=p['startup']:
            for k in rows:
                rows[k].append(None)
            continue
        u = audit.bind(refs[cohort],f'{rank:04d}_{sid}.npz',seals[cohort])
        v = audit.bind(out,f'{i:04d}_{sid}.npz',s)
        valid = np.round((down-t)*1000.,9)>30 if d else np.ones(len(t),bool)
        with np.load(u,allow_pickle=False) as old,np.load(v,allow_pickle=False) as new:
            arrays = {k:(new[k] if k in audit.GENERATED else old[k]) for k in rows}
            for k,c in arrays.items():
                if c.shape!=valid.shape:
                    raise ValueError('sealed curve is not on the HDF5 time grid')
                rows[k].append(dict(shot=sid,disruptive=d,peak=float(c[valid].max()) if valid.any() else None))
            if cohort=='extension':
                lower = np.minimum.reduce(list(arrays.values()))
                upper = np.maximum.reduce(list(arrays.values()))
                for b,receipt in bounds.items():
                    tau = p['calibration'][b]
                    if d:
                        receipt['timely_upper'] += bool(((upper>=tau)&valid).any())
                    else:
                        receipt['false_alarm_lower'] += bool((lower>=tau).any())
                        h = np.flatnonzero(arrays['current_5']>=tau)
                        if len(h):
                            receipt['candidate_false_alarms'] += 1
                            receipt['first_before_full_input'] += bool(h[0]<127)
                            receipt['static_also_crosses_at_first'] += bool(arrays['fixed'][h[0]]>=tau)
                            receipt['static_also_crosses_anywhere'] += bool((arrays['fixed']>=tau).any())
    groups = {}
    for name,keep in masks.items():
        idx = [i for i,row in enumerate(p['order']) if keep(row)]
        ns = sum(not per[i]['disruptive'] for i in idx)
        groups[name] = {}
        for k,arm in rows.items():
            take = [arm[i] for i in idx]
            bands = {}
            for b in ('0.2','0.15'):
                tau = p['calibration'][b]
                hit = [(a['disruptive'], a['peak'] is not None and np.float32(a['peak'])>=np.float32(tau)) for a in take]
                tp,fp = sum(d and x for d,x in hit),sum(not d and x for d,x in hit)
                old = r['groups'][name][b]['metrics'][k]
                if (tp,fp)!=(old['tp'],old['fp']):
                    raise ValueError(f'peak and first alarm disagree: {name}/{b}/{k}')
            for cap in (.10,.15,.20):
                at = best_at_count(take,int(cap*ns))
                bands[str(cap)] = {k:at[k] for k in ('tp','fp','tpr','fpr','max_fp')}
            groups[name][k] = dict(auc=timely_auc(take),postseal_upper_by_fp_budget=bands)
    write(output,dict(scope='POSTSEAL_MATCHED_FALSE_ALARM_RANKING_NOT_THRESHOLD_SELECTION',
        seal_sha256=sha(out/'seal.json'),groups=groups,extension_pointwise_bounds_and_first_alarm=bounds))
    for name in ('fixed340','extension','balanced'):
        print(name,{k:(round(v['auc'],5),[(c,z['tp'],z['fp']) for c,z in
                   v['postseal_upper_by_fp_budget'].items()]) for k,v in groups[name].items()},flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('out',type=Path)
    a = ap.parse_args()
    run(a.run.resolve(),a.out.resolve())
