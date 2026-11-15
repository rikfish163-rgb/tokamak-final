"""Verify raw warning clocks and contrast harder versus broader earlier supervision."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from . import commissioning_coincident_innovation_20261111 as joint
from . import commissioning_excursion_order_20261111 as excursion
from . import io23
from .history_gradient_metric_20261102.run import read, sha, write


def recent_pairs(rank, pairs, budget):
    """Prioritize pairs whose later member has the highest rank, not their outcomes."""
    if not 0<=budget<=len(pairs):
        raise ValueError('equal-budget history exceeds available disjoint matches')
    return pairs[np.argsort(-rank[pairs].max(1),kind='stable')[:budget]]


def _raw_truth(sid):
    with h5py.File(io23.path_for('EAST',sid),'r') as f:
        flag = np.asarray(f['meta/IsDisrupt'][()]).reshape(-1)
        if len(flag)!=1 or flag[0] not in (0,1):
            raise ValueError('IsDisrupt is not an authoritative binary scalar')
        disrupted = bool(flag[0])
        when = np.asarray(f['meta/DownTime'][()]).reshape(-1) if disrupted else []
        if disrupted and (len(when)!=1 or not np.isfinite(when[0])):
            raise ValueError('disruptive shot must have a finite DownTime')
        ip = f['data/ip']
        start,rate = (float(np.asarray(ip.attrs[k]).reshape(-1)[0]) for k in ('StartTime','SampleRate'))
        return disrupted, float(when[0]) if disrupted else None, start+np.arange(len(ip))/rate


def _verified_rows(out, p, result):
    parent = Path(p['reference'])
    if sha(parent/'seal.json')!=p['reference_seal']:
        raise ValueError('fixed parent has changed')
    ref = read(parent/'seal.json')['curve_sha256']
    with np.load(out/'frames.npz',allow_pickle=False) as f:
        data = dict(f)
    positive, missed, safe, fp = [np.zeros(len(data['rank']),dtype=bool) for _ in range(4)]
    records = []
    for rank,sid in enumerate(p['order'],1):
        d,down,t = _raw_truth(sid)
        name = f'{rank:04d}_{sid}.npz'
        path = parent/'shots'/name
        if sha(path)!=ref[name]:
            raise ValueError('sealed fixed score changed')
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            prob = f['fixed']
        level = p['original'] if rank<=p['startup'] else p['main_threshold']
        above = prob>=level
        if d:
            above &= (down-t)*1000>=-1e-6
        crossing = np.flatnonzero(above)
        first = int(crossing[0]) if len(crossing) else None
        lead = round(float((down-t[first])*1000.),9) if d and first is not None else None
        i = np.flatnonzero(data['rank']==rank)
        r = data['release'][i]
        np.testing.assert_array_equal(data['time'][i],t[r])
        np.testing.assert_array_equal(data['shot'][i],np.full(len(r),sid))
        if rank>p['startup']:
            np.testing.assert_array_equal(r,np.arange(p['release'],len(t),p['stride']))
        late = bool(d and (lead is None or lead<=30.))
        wrong = bool(not d and first is not None)
        release_lead = np.round((down-t[r])*1000.,9) if d else np.zeros(len(r))
        positive[i] = (release_lead>30.) & (release_lead<=125.)
        missed[i] = positive[i] & late
        safe[i] = not d
        if wrong:
            safe[i] &= r<=first
        fp[i] = safe[i] & wrong
        records.append(dict(rank=rank,shot=sid,disruptive=d,first=first,first_lead_ms=lead,late=late,
                            fp=wrong,pre_feedback_fp=bool(wrong and not safe[i].any()),
                            any_timely_support=bool(positive[i].any())))
    if records!=result['per_shot_clock'] or sum(r['disruptive'] for r in records)!=180:
        raise ValueError('raw HDF5 outcome or warning clocks differ from the completed analysis')
    return dict(**data,positive=positive,missed=missed,safe=safe,fp=fp)


def run(out):
    p,seal,result = (read(out/k) for k in ('protocol.json','observation_seal.json','result.json'))
    if any(sha(out/k)!=h for k,h in seal['assets'].items()):
        raise ValueError('the observation archive changed')
    data = _verified_rows(out,p,result)
    rank = data['rank']
    x = joint._designs(data)
    verified, support = 0, []
    for b in result['blocks']:
        lo,hi = b['first_rank'],b['last_rank']
        f = out/b['forecasts']
        if sha(f)!=b['sha256']:
            raise ValueError('the chronological prediction archive changed')
        ii = np.flatnonzero((rank>=lo)&(rank<=hi))
        historical = rank<lo
        earlier = {k:v[historical] for k,v in data.items()}
        d,s = joint._pair(earlier,earlier['missed'],earlier['fp'])
        hard = b['tests']['hard']
        held = np.array(hard['source_indices'],dtype=int).reshape(-1,2)
        tested = ii[held]
        if not (data['missed'][tested[:,0]].all() and data['fp'][tested[:,1]].all()):
            raise ValueError('held matches are not eligible hard errors')
        if np.any(np.abs(data['match'][tested[:,0]]-data['match'][tested[:,1]])>joint.CALIPER):
            raise ValueError('held pair outside the locked risk and operating-state match')
        with np.load(f,allow_pickle=False) as saved:
            for k in ('rank','shot','release'):
                np.testing.assert_array_equal(saved[k],data[k][ii])
            for k,v in hard['ordering'].items():
                assert int((saved[k][held[:,0]]>saved[k][held[:,1]]).sum())==v['correct']
            if not len(d):
                assert b['fit'] is None and all(k not in saved for k in joint.MODELS)
                continue
            np.testing.assert_array_equal(b['fit']['ranks'],np.c_[earlier['rank'][d],earlier['rank'][s]])
            for k in joint.MODELS:
                np.testing.assert_allclose(saved[k],x[k][ii]@np.array(b['fit'][k]),rtol=1e-12,atol=1e-12)
                verified+=len(ii)
        a,c = joint._pair(earlier,earlier['positive'],earlier['safe'])
        broad = np.c_[a,c]
        budget = len(d)
        comparison = {}
        for name,pairs in (('hard_only',np.c_[d,s]),('broad_equal',recent_pairs(earlier['rank'],broad,budget)),
                           ('broad_full',broad)):
            i,j = pairs.T
            before = joint._designs(earlier)
            good = {}
            for model in joint.MODELS:
                w = excursion.pair_fit(before[model][i],before[model][j])
                gap = (x[model][tested[:,0]]-x[model][tested[:,1]])@w
                good[model] = dict(correct=int((gap>0.).sum()),ties=int((gap==0.).sum()),
                                   median=float(np.median(gap)) if len(gap) else None)
            if name=='hard_only':
                assert all(good[k]['correct']==hard['ordering'][k]['correct'] for k in joint.MODELS)
            comparison[name] = dict(train_pairs=len(i),ordering=good)
        support.append(dict(first_rank=lo,last_rank=hi,hard_test_pairs=len(tested),regimes=comparison))
    sums = {r:{m:sum(b['regimes'][r]['ordering'][m]['correct'] for b in support) for m in joint.MODELS}
            for r in ('hard_only','broad_equal','broad_full')}
    payload = dict(status='PASS',scope='raw HDF5 first-alarm audit and postseal support sensitivity; '
                   'supervised fits use only earlier blocks, unchanged test pairs, not alarm outputs',
                   shots=360,observed_releases=len(rank),verified_learned_scores=verified,
                   hard_test_pairs=sum(b['hard_test_pairs'] for b in support),fixed_clock=result['raw_counts'],
                   blocks=support,correct=sums,observation_sha256=seal['assets']['frames.npz'],
                   result_sha256=sha(out/'result.json'),audit_code_sha256=sha(Path(__file__)))
    write(out/'hdf5_and_support_audit.json',payload)
    print(dict(rows=verified,correct=sums,counts=[b['regimes']['broad_full']['train_pairs'] for b in support]),
          flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
