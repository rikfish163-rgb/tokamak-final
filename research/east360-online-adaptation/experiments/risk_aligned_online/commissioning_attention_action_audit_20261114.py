"""Postseal fixed-clock opportunity diagnosis for the tested attention actions."""

import argparse
from pathlib import Path

import numpy as np

from . import commissioning_attention_reweight_20261114 as study
from .history_gradient_metric_20261102.run import read, write
from .operating_point_audit import best_at_count, peak_for_shot


def confirmation(t, prob, y, down, tau, required):
    hits = np.asarray(prob >= tau, np.int64)
    if y:
        hits *= np.round((down-t)*1e3,9)>30.
    cumulative = np.r_[0,np.cumsum(hits)]
    return bool((cumulative[required:]-cumulative[:-required]==required).any())


def counts(bands, shots):
    return {k:dict(timely=sum(x['y'] and x[k] for x in shots),
                   fp=sum(not x['y'] and x[k] for x in shots))
            for k in bands}


def diagnostic(out):
    p,s,per = (read(out/k) for k in ('protocol.json','seal.json','per_shot.json'))
    names = [f'{rank:04d}_{sid}.npz' for rank,sid in enumerate(p['order'],1)]
    if len(names)!=360 or names!=s['curves']:
        raise ValueError('requires fully sealed attention comparison')
    keep = (*study.ARMS[:4],*study.NEW)
    budgets = {k:[] for k in keep}
    positive, negative, confirmed = [], [], {n:[] for n in (1,2,3,5)}
    tau = p['calibration']['0.2']
    ref_arms = ('attention_current','attention_earlier','attention_reversed')
    joint_arms = ('inter_only','joint_current','joint_earlier','joint_reversed')
    menus = dict(parent_and_intra=('bias_only',*ref_arms),
                 parent_old_and_intra=('bias_only','current_5','earlier_5',*ref_arms),
                 all_methods=keep)
    shots = []
    cold_fp = 0
    near_log = []
    for rank,(sid,name) in enumerate(zip(p['order'],names,strict=True),1):
        y,down,t = study.group.hdf._raw_truth(sid)
        fixed_score = p['original'] if rank<=p['startup'] else tau
        with np.load(out/'shots'/name,allow_pickle=False) as archive:
            data = dict(archive)
        considered = [k for k in (*ref_arms,*joint_arms) if not (
            k=='inter_only' and np.array_equal(data[k],data['bias_only']))]
        recorded = {k:per['0.2'][k][rank-1] for k in keep}
        cold_fp += bool(rank<=p['startup'] and not y and recorded['bias_only']['alarm'])
        seen = data['dose']>0.
        if rank>p['startup']:
            for k in keep:
                budgets[k].append(dict(shot=sid,disruptive=y,peak=peak_for_shot(t,data[k],y,down)))
        for n,r in confirmed.items():
            r.append(dict(y=y,**{k:confirmation(t,data[k],y,down,fixed_score,n) for k in keep}))
        cuts = dict(shot=sid,rank=rank,writable=int(seen.sum()))
        if y and not recorded['current_5']['timely'] and rank>p['startup']:
            ms = np.round((down-t)*1e3,9)
            scope = (ms>30.) & (ms<=125.) & seen
            # Compare in probability units at the same legal endpoint.
            point = None
            if scope.any():
                active = np.flatnonzero(scope)
                i = active[np.argmax(data['bias_only'][scope])]
                point = dict(sample=int(i),lead_ms=float(ms[i]),gap=float(tau-data['bias_only'][i]),
                    lift={k:float(data[k][i]-data['bias_only'][i]) for k in considered},
                    best_lift={k:float((data[k][scope]-data['bias_only'][scope]).max()) for k in considered},
                    near_hits={k:int((data[k][scope]>=tau).sum()) for k in considered})
            cuts.update(stratum='late' if recorded['current_5']['alarm'] else 'never',
                legal_near=int(scope.sum()),near=point,
                fixed_rescued=[k for k in considered if recorded[k]['timely']])
            positive.append(cuts)
        elif not y and recorded['bias_only']['alarm']:
            u = int(np.flatnonzero(data['bias_only']>=fixed_score)[0])
            cuts.update(first_false_index=u,first_false_writable=bool(seen[u]),
                immutable_crossings=int(((data['bias_only']>=fixed_score) & ~seen).sum()),
                fixed_removed=[k for k in considered if not recorded[k]['alarm']])
            negative.append(cuts)
        elif not y and rank>p['startup']:
            near_log.append(dict(shot=sid,added=[k for k in considered if recorded[k]['alarm']]))
        row = {k:bool(recorded[k]['timely']) for k in keep}
        if y:
            shots.append(dict(y=y,**{k:any(row[m] for m in v) for k,v in menus.items()}))
        else:
            # Pointwise lower envelope only as a deliberately overpermissive
            # exclusion test; a real controller cannot choose after the fact.
            shots.append(dict(y=y,**{k:bool((np.min(np.stack([data[m] for m in v]),axis=0)>=
                                           fixed_score).any()) for k,v in menus.items()}))
    points = {str(n):{k:{x:v[x] for x in ('tp','fp')} for k,r in budgets.items()
                     for v in [best_at_count(r,n-cold_fp)]} for n in (18,27,36)}
    if counts(keep,confirmed[1])!={
            k:dict(timely=sum(x['timely'] for x in v),
                   fp=sum(x['alarm'] and not x['disruptive'] for x in v))
            for k,v in per['0.2'].items()}:
        raise AssertionError('one-point confirmation must give the original score')
    output = dict(cold_false=cold_fp,suffix_ranking=points,late_and_never=positive,
        original_false=negative,original_safe_new=near_log,optimistic_menu=counts(menus,shots),
        confirmation={str(n):counts(keep,r) for n,r in confirmed.items()},
        interpretation='Postseal score ordering and action opportunities; not alternative deployed thresholds.')
    write(out/'postseal_action_audit.json',output)
    print(dict(menu=output['optimistic_menu'],confirm2=output['confirmation']['2'],
               ranking36=points['36']),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    diagnostic(cli.parse_args().output.resolve())
