"""Descriptive paired diagnostics; folds never become independent shots."""
import numpy as np


def partition(rows, folds, test_ids):
    ids=[r['shot'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)&set(test_ids):
        raise ValueError('distinct historical rows, disjoint from test required')
    used=set()
    for fold in folds:
        fit,held=fold['fit'],fold['held']
        if len(fit)!=len(set(fit)) or len(held)!=len(set(held)) or set(fit)&set(held):
            raise ValueError('invalid preparation fold')
        used.update(fit+held)
    if not used<=set(ids):raise ValueError('preparation identity outside historical pool')
    return sorted((r for r in rows if r['shot'] not in used),key=lambda r:r['shot'])


def paired_changes(before, after):
    b={r['shot']:r for r in before};a={r['shot']:r for r in after}
    if len(b)!=len(before) or len(a)!=len(after) or b.keys()!=a.keys():
        raise ValueError('unique matching identities required')
    result=dict(rescued=[],lost=[],removed_fp=[],added_fp=[])
    for sid in sorted(b):
        x,y=b[sid],a[sid]
        if x['is_disrupt']!=y['is_disrupt']:raise ValueError('outcome mismatch')
        if x['is_disrupt']:
            if y['timely'] and not x['timely']:result['rescued'].append(sid)
            if x['timely'] and not y['timely']:result['lost'].append(sid)
        else:
            if x['alarm'] and not y['alarm']:result['removed_fp'].append(sid)
            if y['alarm'] and not x['alarm']:result['added_fp'].append(sid)
    return result


def interval_peaks(times, scores, disruptive, down):
    t,p=np.asarray(times),np.asarray(scores)
    if (t.ndim!=1 or p.shape!=t.shape or not len(t) or not np.isfinite(t).all()
            or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
            or (disruptive and (down is None or not np.isfinite(down)))):
        raise ValueError('finite matching times/probabilities and outcome required')
    if not disruptive:return dict(max_valid=float(p.max()),max_timely=None,max_label_timely=None)
    lead=np.round((down-t)*1000.,9);valid=lead>=-1e-6
    masks=(valid,valid&(lead>30.),valid&(lead>30.)&(lead<=125.+1e-6))
    return {k:float(p[m].max()) if m.any() else None
            for k,m in zip(('max_valid','max_timely','max_label_timely'),masks)}
