"""Post-seal support for difficult windows under the original EAST reveal clock."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.spatial import cKDTree

from . import commissioning_attention_reweight_20261114 as attention, io23
from .history_gradient_metric_20261102.run import read, write


REFERENCE = io23.ROOT/'outputs/east360_attention_reweight_20261114_v1'
QUERY = io23.ROOT/'probe_outputs/layer0_direct_transfer360_20261112_v2'
COLS = tuple(io23.RAW23.index(k) for k in ('ip','Bt','CIII'))


def kind(row):
    if row['disruptive']:
        return 'timely' if row['timely'] else 'late' if row['alarm'] else 'never'
    return 'false' if row['alarm'] else 'safe'


def ago16(x):
    return np.r_[np.full(15,np.nan),np.convolve(x,np.ones(16)/16,mode='valid')]


def window_support(sid,name,row,threshold,clock,rank):
    raw = io23.read_signals('EAST',sid,rate_contract='nominal')
    y,down = io23.read_outcome('EAST',sid)
    if (y,down)!=(row['disruptive'],row['down_s']):
        raise ValueError('sealed outcome differs from raw HDF5')
    with np.load(QUERY/'shots'/name,allow_pickle=False) as a, np.load(
            REFERENCE/'shots'/name,allow_pickle=False) as b:
        np.testing.assert_allclose(raw['times'],a['times'],atol=1e-9,rtol=0.)
        score,dose = b['current_5'],b['dose']
    t = raw['times']
    ip,bt,ciii = raw['v'][:,COLS].astype(np.float64).T
    avail = raw['a'][:,COLS]
    ip,bt = (ago16(np.where(avail[:,j],x,np.nan)) for j,x in enumerate((ip,bt)))
    coverage = ago16(avail[:,2] & ~np.isclose(ciii,-10.,atol=.01,rtol=0.))
    c = np.where(coverage==1.,1,np.where(coverage==0.,0,-1))
    coords = np.stack((np.arcsinh(abs(ip)/1e5)/.35,np.arcsinh(abs(bt))/.35,
                       (logit(score.astype(np.float64).clip(1e-6,1.-1e-6))-logit(threshold))/.5),axis=1)
    usable = (np.arange(len(t))>=127)&np.isfinite(coords).all(1)&(c>=0)
    ms = np.round((down-t)*1e3,9) if y else np.zeros(len(t))
    near = (ms>30.) & (ms<=125.)
    return dict(shot=sid,rank=rank,available_from=clock,y=y,kind=kind(row),score=score,dose=dose,
                coords=coords,mask_state=c,valid=usable,near=near,threshold=threshold)


def make_donors(rows,cold):
    bank = {k:[] for k in ('strict_query','strict_all','any_query','any_all','safe_query','safe_all')}
    records = []
    for v in rows:
        sid,r = v['shot'],v['rank']
        good = v['valid']
        above = v['score']>=v['threshold']
        if v['y']:
            near = v['near']
            small = near & ~above
            use = small & good
            peak = float(v['score'][near].max()) if near.any() else None
            strict = bool(near.any() and not (near&above).any())
            tasks = ['any']+(['strict'] if strict else [])
            d = dict(near_points=int(near.sum()),below=int(small.sum()),strict=strict,near_peak=peak)
        else:
            use = above & good
            tasks = ['safe']
            d = dict(above=int(above.sum()),any_false=bool(above.any()))
        records.append(dict(shot=sid,rank=r,revealed_for=v['available_from'],kind=v['kind'],**d))
        if not use.any():
            continue
        item = (sid,v['available_from'],
                {c:cKDTree(v['coords'][use & (v['mask_state']==c)]) for c in (0,1)})
        for group in tasks:
            bank[f'{group}_all'].append(item)
            if r>cold:
                bank[f'{group}_query'].append(item)
    return bank,records


def matching(shot, group, use):
    ids = []
    for sid,first,trees in group:
        if shot['rank']<first:
            continue
        for c,tree in trees.items():
            points = shot['coords'][use & (shot['mask_state']==c)]
            if len(points) and tree.n and np.isfinite(tree.query(
                    points,k=1,p=np.inf,distance_upper_bound=1.+1e-10)[0]).any():
                ids.append(sid)
                break
    return ids


def audit():
    p = read(REFERENCE/'protocol.json')
    s = read(attention.QUERY/'seal.json')
    start,order = p['startup'],p['order']
    rows = read(REFERENCE/'per_shot.json')['0.2']['current_5']
    if start!=20 or len(order)!=360 or order!=[r['shot'] for r in rows]:
        raise ValueError('only the original fixed stream is supported')
    release = dict.fromkeys(order[:start],start+1)
    for n in range(2,18):
        chosen = attention.checked_selection(n,order,s['blocks'])[0]
        release[chosen] = 20*n+1
    if list(release)[20:]!=read(QUERY/'result.json')['queried']:
        raise ValueError('historical acquisition records disagree')
    cache = {}
    for rank,(sid,row) in enumerate(zip(order,rows,strict=True),1):
        if sid not in release and not (rank>start and kind(row) in ('late','never','false')):
            continue
        threshold = p['original'] if rank<=start else p['calibration']['0.2']
        cache[sid] = window_support(sid,f'{rank:04d}_{sid}.npz',row,threshold,release.get(sid),rank)
    bank,donors = make_donors([cache[s] for s in release],start)
    coverage = []
    for v in cache.values():
        k = v['kind']
        if v['rank']<=start or k not in ('late','never','false'):
            continue
        use = v['near'].copy()
        keys = ('strict_query','strict_all','any_query','any_all')
        if k=='false':
            use[np.flatnonzero(v['score']>=v['threshold'])[0]] = True
            keys = ('safe_query','safe_all')
        at = use & v['valid'] & (v['dose']>0.)
        coverage.append(dict(shot=v['shot'],rank=v['rank'],kind=k,effective_points=int(at.sum()),
            by_policy={j:matching(v,bank[j],at) for j in keys}))
    summary = {}
    for k in ('late','never','false'):
        a = [x for x in coverage if x['kind']==k]
        summary[k] = dict(n=len(a),clock_support=sum(x['effective_points']>0 for x in a),
            coverage={b:sum(bool(x['by_policy'][b]) for x in a) for b in a[0]['by_policy']})
    return dict(scope='postseal audit only; no new alarms or fitting',raw_verified=len(cache),
        cold=start,queried=len(release)-start,distance='max of scaled 16-point |ip|/|Bt| means and '
        'relative threshold logit, within same CIII validity; fixed match radius 1',
        matching_support=summary,all_pairs=coverage,donors=donors)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit()
    print(result['matching_support'],flush=True)
    write(args.output,result)
