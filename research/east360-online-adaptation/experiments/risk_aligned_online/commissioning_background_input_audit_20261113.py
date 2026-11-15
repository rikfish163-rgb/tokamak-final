"""Audit raw EAST input strata against already sealed warning predictions."""

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from . import commissioning_parent_readout_20261113 as parent
from . import io23
from .commissioning_extension_order_budget_20261113 import PREFIX
from .commissioning_stream import _record
from .history_gradient_metric_20261102.run import read, sha, write


def balanced_sample(strata, rng):
    selected = [[p if len(p)==n else rng.choice(p,n,replace=False) for p in pools]
                for pools,n in strata]
    return np.stack([np.concatenate(c) for c in zip(*selected,strict=True)])


def quantile_pct(x):
    return (100*np.percentile(x,[2.5,50,97.5])).tolist()


def input_profiles(protocol):
    source = PREFIX/'order.json'
    roster = read(PREFIX/'roster.json')
    order = read(source)
    if sha(source)!=roster['order_sha256'] or [r['shot'] for r in order]!=roster['shot_ids']:
        raise ValueError('archived EAST prefix measurement changed')
    archived = {r['shot']:r for r in order}
    ids = [r['shot'] for r in protocol['order']]
    if len(ids)!=len(archived) or set(ids)!=set(archived):
        raise ValueError('EAST HDF5 roster and score roster differ')
    rows = []
    attributes = Counter()
    for sid in ids:
        with h5py.File(io23.path_for('EAST',sid),'r') as f:
            bt,ip = (f[f'data/{k}'] for k in ('Bt','ip'))
            def clock(ds,k):
                return float(np.asarray(ds.attrs[k]).reshape(-1)[0])
            if len(bt)!=len(ip) or any(clock(bt,k)!=clock(ip,k) for k in ('SampleRate','StartTime')):
                raise ValueError(f'Bt/Ip time base mismatch: {sid}')
            v,i = (np.asarray(s[:],np.float64) for s in (bt,ip))
            prefix = _record(sid,dict(planning_ip=i,planning_Bt=v,rate=clock(ip,'SampleRate')))
            for k in ('initial_abs_Bt_raw','initial_ip_A','prefix_samples','actual_rate_hz'):
                if prefix[k]!=archived[sid][k]:
                    raise ValueError(f'archived initial measurement differs: {sid} {k}')
            attributes[tuple(sorted(bt.attrs))] += 1
            v = abs(v[np.isfinite(v)])
            i = abs(i[np.isfinite(i)])
            rows.append((prefix['initial_abs_Bt_raw'],np.median(v),np.max(v),np.median(i)))
    return np.asarray(rows), {', '.join(k):n for k,n in attributes.items()}


def compare(folder, profiles, order, low):
    protocol,seal,scored = (read(folder/f'{k}.json') for k in ('protocol','seal','per_shot'))
    ids = [r['shot'] for r in order]
    if protocol['order']!=order or len(seal['curve_sha256'])!=len(ids):
        raise ValueError('comparison needs the same completely sealed roster')
    groups = {}
    reference = scored['0.2']['fixed']
    labels = np.array([r['disruptive'] for r in reference])
    warm = np.arange(len(order))>=protocol['startup']
    old = np.array([r.get('old835') is True for r in order])
    ext = np.array([r['cohort']=='extension' for r in order])
    balanced_ids = {r['shot'] for r in read(parent.WIDE/'balanced_cohort.json')['order']}
    groups['fixed360'] = ~ext
    groups['fixed340'] = ~ext & warm
    groups['extension'] = ext
    groups['old835'] = ext & old
    groups['other1758'] = ext & ~old
    groups['historical_1to1'] = np.array([sid in balanced_ids for sid in ids])
    rng = np.random.default_rng(131)
    strata = []
    for history in (old,~old):
        for scale in (low,~low):
            pools = [np.flatnonzero(ext & history & scale & mask) for mask in (labels,~labels)]
            strata.append((pools,min(map(len,pools))))
    sample = balanced_sample(strata,rng)
    paired = np.zeros(len(order),dtype=bool)
    paired[sample.ravel()] = True
    groups['history_and_raw_scale_1to1'] = paired
    draws = np.stack([sample]+[balanced_sample(strata,rng) for _ in range(999)])
    split = {f'{key}/{name}':mask & scale for key,mask in groups.items()
             for name,scale in (('all',np.ones(len(order),bool)),('smaller',low),('larger',~low))}
    stats = {}
    for key,mask in split.items():
        s = profiles[mask]
        stats[key] = dict(n_disruptive=int((mask & labels).sum()),n_safe=int((mask & ~labels).sum()),
                          profiles_p10_p50_p90=np.quantile(s,[.1,.5,.9],axis=0).T.tolist() if len(s) else None)
    counts,resampling = {},{}
    for band,arms in scored.items():
        counts[band] = {k:{} for k in split}
        resampling[band] = {}
        base_tp = np.asarray([r['timely'] for r in arms['fixed']])[draws[:,0]].sum(1)
        for a,rows in arms.items():
            if [r['shot'] for r in rows]!=ids or not np.array_equal([r['disruptive'] for r in rows],labels):
                raise ValueError('score identity or outcome mismatch')
            tp,fp = (np.array([r[k] for r in rows]) & mask
                     for k,mask in (('timely',labels),('alarm',~labels)))
            for key,mask in split.items():
                counts[band][key][a] = [int((tp & mask).sum()),int((fp & mask).sum())]
            t,f = (v[draws[:,j]].sum(1) for j,v in enumerate((tp,fp)))
            fpr = f/draws.shape[-1]
            delta = (t-base_tp)/draws.shape[-1]
            resampling[band][a] = dict(fpr_pct=quantile_pct(fpr),delta_tpr_pp_vs_fixed=quantile_pct(delta),
                fraction_feasible={str(c):float((fpr<=c).mean()) for c in (.10,.15,.20)},
                fraction_feasible_and_more_timely={str(c):float(((fpr<=c)&(t>base_tp)).mean())
                                                   for c in (.10,.15,.20)})
    sizes = np.asarray([[stats[f'{g}/{s}']['n_safe'] for s in ('smaller','larger')]
                        for g in ('fixed340','extension')])
    weights = sizes/sizes.sum(axis=1,keepdims=True)
    decomposition = {}
    for band,rows in counts.items():
        decomposition[band] = {}
        for arm in scored[band]:
            fp = np.asarray([[rows[f'{g}/{s}'][arm][1] for s in ('smaller','larger')]
                             for g in ('fixed340','extension')])
            rate = fp/sizes
            observed = float(np.diff((weights*rate).sum(axis=1))[0])
            composition = float((weights[1]-weights[0])@rate.mean(axis=0))
            within = float(weights.mean(axis=0)@(rate[1]-rate[0]))
            np.testing.assert_allclose(observed,composition+within,atol=1e-12,rtol=0.)
            decomposition[band][arm] = dict(raw_gap_pp=100*observed,composition_pp=100*composition,
                                            within_strata_pp=100*within)
    return dict(groups=stats, counts=counts, fixed340_to_extension_fpr_decomposition=decomposition,
                balance_resampling=dict(draws=len(draws),sample_per_class=draws.shape[-1],
                    interpretation='retrospective balance sensitivity, not prospective confidence',
                    bands=resampling),
                seal_sha256=sha(folder/'seal.json'),
                matched_ids=[sid for sid,m in zip(ids,paired,strict=True) if m])


def run(output, frozen):
    protocol = read(frozen/'protocol.json')
    profiles,attributes = input_profiles(protocol)
    v = np.sort(profiles[:,0])
    gap = int(np.argmax(v[1:]/v[:-1]))
    threshold = float(np.sqrt(v[gap]*v[gap+1]))
    low = profiles[:,0]<threshold
    result = compare(frozen,profiles,protocol['order'],low)
    write(output,dict(
        scope='POSTSEAL_RAW_INPUT_STRATIFICATION_NOT_A_PHYSICAL_CALIBRATION',
        stratum_rule='geometric midpoint of the largest adjacent ratio in positive archived raw prefix medians',
        gap=v[gap:gap+2].tolist(),raw_split=threshold,bt_attributes=attributes,
        profile_columns=('first_100ms_median_abs_Bt','whole_median_abs_Bt','whole_max_abs_Bt','whole_median_abs_Ip'),
        whole_median_crossovers=dict(smaller_to_larger=int((low & (profiles[:,1]>=threshold)).sum()),
                                     larger_to_smaller=int((~low & (profiles[:,1]<threshold)).sum())),
        whole_max_crossovers=int((low & (profiles[:,2]>=threshold)).sum()),
        balancing='same shot subset for all arms/bands; seed 131; 1:1 within historical membership x raw magnitude group',
        **result))
    print('raw input gap',v[gap:gap+2], 'Bt attributes',attributes,flush=True)
    print({k:(g['n_disruptive'],g['n_safe'],result['counts']['0.2'][k]['fixed'])
           for k,g in result['groups'].items()},flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('frozen',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    run(a.output.resolve(),a.frozen.resolve())
