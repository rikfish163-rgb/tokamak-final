"""Separate record-length composition from the frozen EAST false-alarm gap."""

import argparse
from pathlib import Path

import numpy as np

from . import commissioning_frozen_extension_20261113 as ext
from .history_gradient_metric_20261102.run import read, sha, write


LIMITS = (2.,5.,10.)
NAMES = ('<2s','2–<5s','5–<10s','≥10s')
ARMS = ('fixed','bias_only','current_5')
PREFIX = ext.io23.ROOT.parent/'Fuison/experiments/risk_aligned_online/runs/commissioning_current_all_20261009/full'
FIELDS = ('initial_ip_A','initial_abs_Bt_raw')


def aggregate(groups, selection=None, *, names=NAMES):
    totals = {k:np.asarray(x,dtype=np.int64) for k,x in groups.items()}
    out = {}
    for name,b in totals.items():
        if name=='fixed340':
            continue
        a = totals['fixed340']
        supported = (a[:,0]>0)&(b[:,0]>0)
        if selection is not None:
            supported &= selection
        # Use only shared support for standardization, never fill missing rates.
        joint = a[supported,0]+b[supported,0]
        w = joint/joint.sum()
        p,q = ((g[supported,1:]/g[supported,0,None])*w[:,None] for g in (a,b))
        out[name] = dict(common_bins=[k for k,ok in zip(names,supported) if ok],
            excluded=dict(fixed340=int(a[~supported,0].sum()),target=int(b[~supported,0].sum())),
            arms={k:dict(raw_reference=float(a[:,i+1].sum()/a[:,0].sum()),
                         raw_target=float(b[:,i+1].sum()/b[:,0].sum()),
                         standardized_reference=float(p[:,i].sum()),
                         standardized_target=float(q[:,i].sum()),
                         gap_pp=float(100.*(q[:,i]-p[:,i]).sum()))
                  for i,k in enumerate(ARMS)})
    return out


def run(out):
    wide = ext.io23.ROOT/'probe_outputs/commissioning_frozen_extension_east2593_20261113_v1'
    sources = {'fixed340': ext.REFERENCE,'extension': wide}
    by = {k:np.zeros((len(NAMES),1+len(ARMS)),np.int64) for k in (*sources,'old835','other1758')}
    original = {k:np.zeros_like(v) for k,v in by.items()}
    order = read(PREFIX/'order.json')
    roster = read(PREFIX/'roster.json')
    prefix_sha = sha(PREFIX/'order.json')
    if prefix_sha!=roster['order_sha256'] or [r['shot'] for r in order]!=roster['shot_ids']:
        raise ValueError('changed sealed prefix measurements')
    coordinates = np.asarray([[r[k] for k in FIELDS] for r in order],dtype=float)
    breaks = np.quantile(coordinates,[.25,.5,.75],axis=0).T
    cells = {r['shot']:4*np.searchsorted(breaks[0],ip,side='right')+
             np.searchsorted(breaks[1],bt,side='right') for r,(ip,bt) in zip(order,coordinates)}
    by_prefix = {k:np.zeros((16,1+len(ARMS)),np.int64) for k in by}
    for group,root in sources.items():
        p,s,per = (read(root/f'{k}.json') for k in ('protocol','seal','per_shot'))
        archived = per['0.2']
        for i,row in enumerate(p['order'],1):
            if group=='fixed340' and i<=p['startup']:
                continue
            sid = row if group=='fixed340' else row['shot']
            saved = [archived[a][i-1] for a in ARMS]
            if saved[0]['disruptive']:
                continue
            d,_,t = ext.group.hdf._raw_truth(sid)
            if d or not len(t):
                raise ValueError('archived safe shot conflicts with original HDF5')
            f = root/'shots'/f'{i:04d}_{sid}.npz'
            if sha(f)!=s['curve_sha256'][f.name]:
                raise ValueError('safe reference curve changed')
            with np.load(f,allow_pickle=False) as score:
                if group=='extension':
                    np.testing.assert_array_equal(t,score['times'])
                call = [bool((score[a]>=p['calibration']['0.2']).any()) for a in ARMS]
                base_call = [bool((score[a]>=p['original']).any()) for a in ARMS]
            if any(x['shot']!=sid or x['disruptive'] or x['alarm']!=c for x,c in zip(saved,call)):
                raise ValueError('sealed curve and score disagree')
            j = np.searchsorted(LIMITS,t[-1]-t[0],side='right')
            data = np.asarray([1,*call],dtype=np.int64)
            control = np.asarray([1,*base_call],dtype=np.int64)
            by[group][j] += data
            original[group][j] += control
            by_prefix[group][cells[sid]] += data
            if group=='extension':
                k = 'old835' if row['old835'] else 'other1758'
                by[k][j] += data
                original[k][j] += control
                by_prefix[k][cells[sid]] += data
    if not np.array_equal(by['extension'],by['old835']+by['other1758']):
        raise ValueError('historical groups do not partition the extension')
    summary = aggregate(by)
    names = tuple(f'Ip{a+1}_Bt{b+1}' for a in range(4) for b in range(4))
    write(out,dict(scope='POSTSEAL_EAST_SAFE_DURATION_STANDARDIZATION',
        bins=NAMES, columns=('n_safe',*ARMS), counts={k:v.tolist() for k,v in by.items()},
        original_threshold_counts={k:v.tolist() for k,v in original.items()},
        weights='pooled safe-shot duration distribution on pairwise shared bins',
        limitation='recorded duration only, not verified physical shot length or adjustment for regime',
        thresholds_unchanged=True, comparison=summary,central_bins=aggregate(by,np.array([False,True,True,False])),
        prefix=dict(input_sha256=prefix_sha,fields=FIELDS,breaks=breaks.tolist(),cells=names,
                    counts={k:v.tolist() for k,v in by_prefix.items()},comparison=aggregate(by_prefix,names=names)),
        reference_seals={k:sha(v/'seal.json') for k,v in sources.items()}))
    print(dict(duration=summary,prefix=aggregate(by_prefix,names=names)),flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    run(ap.parse_args().out.resolve())
