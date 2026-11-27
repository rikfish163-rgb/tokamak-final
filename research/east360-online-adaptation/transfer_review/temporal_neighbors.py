"""CPU-only, sealed EAST360 neighbor diagnostics. No model/HDF5/network access."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ORDER = '3fefd51a0a1e16bb82735f59e8b332857eecd1c0d66102e6b4b413371a93ce97'
NAMES = ('log_ip', 'mask_hamming', 'label_mismatch', 'kind_mismatch',
         'normal_z', 'normal_margin', 'normal_drop')


def summary(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
                quantiles=dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max'),
                                   map(float, np.percentile(x, (0, 25, 50, 75, 90, 95, 99, 100))))))


def pairs(values, normal, mask, logip, mode):
    """Return one latest donor and uniformly weighted older alternatives per target."""
    selected = []
    for i in range(1, len(normal)):
        if mode.startswith('normal') and not normal[i]:
            continue
        eligible = np.arange(i)
        if mode.startswith('normal'):
            eligible = eligible[normal[:i]]
        if mode.endswith('matched'):
            ok = (np.abs(logip[i] - logip[eligible]) <= np.log(1.25))
            ok &= (mask[eligible] == mask[i]).all(1)
            eligible = eligible[ok]
        if len(eligible) < 2:
            continue
        donor = int(eligible[-1])
        if mode.startswith('immediate') and donor != i - 1:
            continue
        selected.append((i, donor, eligible[:-1]))
    return selected


def measure(values, normal, mask, logip, mode, detailed=False):
    selected = pairs(values, normal, mask, logip, mode)
    if not selected:
        return None
    first, base = [], []
    for i, j, older in selected:
        first.append(abs(values[i] - values[j]))
        base.append(np.abs(values[i] - values[older]).mean())
    first, base = np.array(first), np.array(base)
    result = dict(n=len(first), predecessor=float(first.mean()),
                  random_one_expected=float(base.mean()), advantage=float((base - first).mean()))
    if detailed:
        result.update(predecessor_sum=float(first.sum()), other_history_expected_sum=float(base.sum()),
                      strict_wins=int((first < base).sum()), ties=int((first == base).sum()),
                      losses=int((first > base).sum()),
                      rank_gaps=summary([i - j for i, j, _ in selected]),
                      alternative_counts=summary([len(k) for _, _, k in selected]))
    return result


def all_metrics(v, normal, mask, logip, detailed=False):
    # Categorical mismatch must not use numeric code distances.
    results = {}
    selected = pairs(v, normal, mask, logip, 'immediate')
    for name, distance in (
        ('log_ip', abs(logip[:, None] - logip[None, :])),
        ('mask_hamming', (mask[:, None] != mask[None, :]).sum(2)),
        ('label_mismatch', normal[:, None] != normal[None, :]),
        ('kind_mismatch', v[:, 0, None] != v[None, :, 0])):
        first = np.array([distance[i, j] for i, j, _ in selected], float)
        base = np.array([distance[i, older].mean() for i, _, older in selected])
        results[name] = dict(n=len(first), predecessor=float(first.mean()),
                             random_one_expected=float(base.mean()), advantage=float((base-first).mean()))
        if detailed:
            results[name].update(predecessor_sum=float(first.sum()), other_history_expected_sum=float(base.sum()))
    for name, k in zip(NAMES[4:], (1, 2, 3)):
        results[name] = measure(v[:, k], normal, mask, logip, 'normal', detailed)
    return results


def aggregate(blocks):
    out = {}
    for name in NAMES:
        valid = [b[name] for b in blocks if b.get(name) is not None]
        n = sum(r['n'] for r in valid)
        out[name] = dict(n=n, **{k:sum(r[k]*r['n'] for r in valid)/n
                                for k in ('predecessor', 'random_one_expected', 'advantage')})
    return out


def null_summary(samples, observed):
    x = np.asarray(samples)
    return dict(repetitions=len(x), mean=float(x.mean()), null95=list(map(float, np.percentile(x, (2.5, 97.5)))),
                p_upper=(1+int((x >= observed).sum()))/(1+len(x)),
                mc_se=float(np.sqrt(((1+(x >= observed).sum())/(1+len(x)))*(1-(1+(x >= observed).sum())/(1+len(x)))/len(x))))


def run(path, repeats, seed):
    parent = json.loads((path/'paired_forensics.json').read_text())
    drops = json.loads((path/'current_drop_check.json').read_text())
    rows = parent['rows']
    ids = [r['shot'] for r in rows]
    assert len(rows) == 360 and len(set(ids)) == 360 and ids == sorted(ids)
    assert sha256(','.join(map(str, ids)).encode()).hexdigest() == parent['order_sha256'] == ORDER
    assert [r['rank'] for r in rows] == list(range(1, 361))
    assert [r['block'] for r in rows] == [i//90 for i in range(360)]
    assert set(drops['rows']) == set(map(str, ids))
    assert Counter(r['kind'] for r in rows) == dict(TP_timely=135, TARDY=23, MISS=22, FP=31, TN=149)
    assert all(r['ip'] is not None and r['ip'] > 0 and len(r['mask0']) == 23 for r in rows)
    assert all((r['down'] is None) == (r['kind'] in ('TN', 'FP')) for r in rows)
    assert all(r['first'] == drops['rows'][str(r['shot'])]['parent_first'] for r in rows)
    kinds = {'TN':0, 'FP':1, 'TP_timely':2, 'TARDY':3, 'MISS':4}
    normal = np.array([r['down'] is None for r in rows])
    mask = np.array([r['mask0'] for r in rows])
    ip = np.array([r['ip'] for r in rows])
    logip = np.log(ip)
    v = np.array([[kinds[r['kind']], r['eligible_peak_z'], r['timely_peak_margin'],
                   drops['rows'][str(r['shot'])]['full_peak']] for r in rows])
    assert np.isfinite(v).all()
    # Normal peaks are whole-record values; disruption peaks never enter normal metrics.
    assert all(r['eligible_n'] == r['n'] for r in rows if r['down'] is None)
    chunks = [(v[s:s+90], normal[s:s+90], mask[s:s+90], logip[s:s+90]) for s in range(0,360,90)]
    metrics = [all_metrics(*b, detailed=True) for b in chunks]
    observed = aggregate(metrics)
    gap = np.diff(ids)
    out = dict(seed=seed, repetitions=repeats, order_sha256=ORDER,
               input_sha256={n:sha256((path/n).read_bytes()).hexdigest()
                             for n in ('paired_forensics.json', 'current_drop_check.json')},
               gaps=summary(gap), gap_counts={str(t):int((gap<=t).sum()) for t in (1,5,10,25,50,100,500,1000)},
               blocks=[], pooled=observed, conditional_past=[], immediate_normal=[])
    for b, (chunk, metric) in enumerate(zip(chunks, metrics)):
        vv, nn, mm, ll = chunk
        rr = rows[90*b:90*(b+1)]
        same = (np.abs(np.diff(ll)) <= np.log(1.25)) & (mm[1:] == mm[:-1]).all(1)
        starts = np.r_[0, np.flatnonzero(~same)+1, 90]
        ratios = [float(np.exp(ll[s:e].max()-ll[s:e].min())) for s,e in zip(starts[:-1],starts[1:])]
        out['blocks'].append(dict(block=b+1, shot_range=[rr[0]['shot'],rr[-1]['shot']],
                                  kinds=dict(Counter(r['kind'] for r in rr)), gaps=summary(np.diff(ids[90*b:90*(b+1)])),
                                  metrics=metric, same_proxy_pairs=int(same.sum()),
                                  proxy_runs=summary(np.diff(starts)),
                                  proxy_chain_singletons=int((np.diff(starts)==1).sum()),
                                  proxy_chains_exceeding_25pct_span=sum(x>1.25 for x in ratios),
                                  max_proxy_chain_ip_ratio=max(ratios)))
        matched = {}
        sel = pairs(vv, nn, mm, ll, 'immediate_matched')
        for name, labels in [('label_mismatch', nn), ('kind_mismatch', vv[:,0])]:
            first = np.array([labels[i] != labels[j] for i,j,_ in sel], float)
            base = np.array([(labels[i] != labels[older]).mean() for i,_,older in sel])
            matched[name] = dict(n=len(first), predecessor=float(first.mean()),
                                 random_one_expected=float(base.mean()), advantage=float((base-first).mean()))
        for name,k in zip(NAMES[4:], (1,2,3)):
            matched[name] = measure(vv[:,k],nn,mm,ll,'normal_matched',True)
        out['conditional_past'].append(matched)
        # Keep the true immediate predecessor only when both shots are normal.
        nm = {}
        for name,k in zip(NAMES[4:], (1,2,3)):
            sel = [(i,j,older) for i,j,older in pairs(vv,nn,mm,ll,'normal') if j == i-1]
            first = np.array([abs(vv[i,k]-vv[j,k]) for i,j,_ in sel])
            base = np.array([abs(vv[i,k]-vv[older,k]).mean() for i,_,older in sel])
            nm[name] = dict(n=len(first), predecessor=float(first.mean()),
                            random_one_expected=float(base.mean()), advantage=float((base-first).mean()))
        out['immediate_normal'].append(nm)
    # Shuffle whole rows within each original 90-shot block, preserving the joint distribution.
    rng = np.random.default_rng(seed)
    samples = {name:[] for name in NAMES}
    for _ in range(repeats):
        trial = []
        for vv, nn, mm, ll in chunks:
            idx = rng.permutation(90)
            trial.append(all_metrics(vv[idx],nn[idx],mm[idx],ll[idx]))
        scores = aggregate(trial)
        for name in NAMES:
            samples[name].append(scores[name]['advantage'])
    out['block_row_permutation'] = {name:null_summary(x,observed[name]['advantage']) for name,x in samples.items()}
    out['conditional_permutation'] = {}
    # Coarse measured-condition strata, fixed before looking at outcomes.
    # Labels shuffle within block/mask/rounded-current bands. Normal peak shuffles also keep FP/TN.
    for width in (50000,100000,200000):
        groups = defaultdict(list)
        peak_groups = defaultdict(list)
        for i in range(360):
            key = (i//90, tuple(mask[i]), int(np.floor(ip[i]/width+.5)))
            groups[key].append(i)
            if normal[i]:
                peak_groups[(*key, int(v[i,0]))].append(i)
        label_samples = {name:[] for name in NAMES[2:4]}
        peak_samples = {name:[] for name in NAMES[4:]}
        for _ in range(repeats):
            nv = v.copy()
            for ix in groups.values():
                nv[ix,0] = v[rng.permutation(ix),0]
            nn = nv[:,0] < 2
            scores = aggregate([all_metrics(nv[s:s+90],nn[s:s+90],mask[s:s+90],logip[s:s+90]) for s in range(0,360,90)])
            for name in label_samples:
                label_samples[name].append(scores[name]['advantage'])
            pv = v.copy()
            for ix in peak_groups.values():
                pv[np.array(ix),1:] = v[rng.permutation(ix),1:]
            scores = aggregate([all_metrics(pv[s:s+90],normal[s:s+90],mask[s:s+90],logip[s:s+90]) for s in range(0,360,90)])
            for name in peak_samples:
                peak_samples[name].append(scores[name]['advantage'])
        out['conditional_permutation'][str(width)] = dict(
            strata=len(groups), singleton_strata=sum(len(g)==1 for g in groups.values()),
            rows_in_mixed_label_strata=sum(len(g) for g in groups.values() if len(set(normal[g]))>1),
            movable_normal_peak_rows=sum(len(g) for g in peak_groups.values() if len(g)>1),
            tests={name:null_summary(x,observed[name]['advantage']) for name,x in {**label_samples,**peak_samples}.items()})
    out['largest_gaps'] = [dict(before=ids[i],after=ids[i+1],gap=int(gap[i]),
                               rank_before=int(i+1),same_block=bool(i//90==(i+1)//90),
                               ip_ratio=float(max(ip[i:i+2])/min(ip[i:i+2])),
                               same_label=bool(normal[i]==normal[i+1]))
                           for i in np.argsort(gap)[-8:][::-1]]
    breaks = [0]+[i for i in range(1,360) if not np.array_equal(mask[i],mask[i-1])]+[360]
    out['mask_segments'] = [dict(ranks=[s+1,e],shots=[ids[s],ids[e-1]],n=e-s,
                                 first_two_mask=mask[s,:2].tolist())
                            for s,e in zip(breaks[:-1],breaks[1:])]
    out['gap_groups'] = {}
    within = np.array([i//90==(i+1)//90 for i in range(359)])
    for name, select in [('le10',gap<=10),('11to100',(gap>10)&(gap<=100)),('gt100',gap>100)]:
        ix = np.flatnonzero(select & within)
        out['gap_groups'][name] = dict(n=len(ix),same_label=int((normal[ix]==normal[ix+1]).sum()),
                                      same_mask=int((mask[ix]==mask[ix+1]).all(1).sum()),
                                      mean_log_ip=float(abs(logip[ix]-logip[ix+1]).mean()))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=1999)
    parser.add_argument('--seed',type=int,default=20261127)
    args = parser.parse_args()
    assert args.repeats > 0
    print(json.dumps(run(Path(__file__).resolve().parent,args.repeats,args.seed),ensure_ascii=False,indent=2,allow_nan=False))
