"""Attribute early, sustained, and overlapping warnings in sealed EAST curves."""

import argparse
from pathlib import Path

import numpy as np

from . import commissioning_hidden_background_20261113 as background
from .history_gradient_metric_20261102.run import read, sha, write


def support(curve, tau, first):
    above = curve>=tau
    edges = np.flatnonzero(np.diff(np.r_[False,above,False]))
    lengths = edges[1::2]-edges[::2]
    return dict(alarm=bool(len(edges)),first=int(edges[0]) if len(edges) else None,
                early=bool(above[:first].any()),later=bool(above[first:].any()),
                crossing_points=int(above.sum()),longest_run=int(lengths.max()) if len(lengths) else 0)


def run(folder, output):
    par = background.prior
    previous = background.PARENT
    p,s,scores = (read(folder/f'{k}.json') for k in ('protocol','seal','per_shot'))
    pp,ps = (read(previous/f'{k}.json') for k in ('protocol','seal'))
    if (len(s['curve_sha256'])!=len(p['order']) or p['parent_seal']!=sha(previous/'seal.json')
            or pp['order']!=p['order'] or pp['calibration']!=p['calibration']):
        raise ValueError('completed paired references are required')
    paths = {c:Path(v['path']) for c,v in pp['references'].items()}
    seals = {c:read(path/'seal.json') for c,path in paths.items()}
    if any(sha(path/'seal.json')!=pp['references'][c]['seal_sha256'] for c,path in paths.items()):
        raise ValueError('source archive changed')
    raw_balance = read(folder/'postseal_bt_scale_balance.json')
    if raw_balance['seal_sha256']!=sha(folder/'seal.json'):
        raise ValueError('raw scale balance must bind this experiment')
    sample = set(raw_balance['matched_ids'])
    rows = {b:[] for b in ('0.2','0.15')}
    refs = background.BASES+('current_5',)
    old_scores = read(previous/'per_shot.json')
    for i,entry in enumerate(p['order'],1):
        sid = entry['shot']
        if scores['0.2']['fixed'][i-1]['disruptive']:
            continue
        rank,c = entry['rank'],entry['cohort']
        original = par.bind(paths[c],f'{rank:04d}_{sid}.npz',seals[c])
        prior = par.bind(previous,f'{i:04d}_{sid}.npz',ps)
        new = par.bind(folder,f'{i:04d}_{sid}.npz',s)
        with np.load(original,allow_pickle=False) as f,np.load(prior,allow_pickle=False) as g,\
                np.load(new,allow_pickle=False) as h:
            curves = {k:f[k] if k in ('fixed','current_5') else g[k] for k in refs}
            curves.update({k:h[k] for k in p['arms'] if k not in refs})
            for b in rows:
                tau = p['original'] if i<=p['startup'] else p['calibration'][b]
                calls = {k:support(v,tau,p['first']) for k,v in curves.items()}
                for k,value in calls.items():
                    expected = (old_scores if k=='current_5' else scores)[b][k][i-1]
                    if expected['shot']!=sid or expected['disruptive'] or value['alarm']!=expected['alarm']:
                        raise ValueError('sealed support and HDF5 score disagree')
                rows[b].append(dict(shot=sid,cohort=c,balanced=sid in sample,arms=calls))
    summary = {}
    for b,band in rows.items():
        summary[b] = {}
        for name,selected in (('fixed360',[r for r in band if r['cohort']=='fixed360']),
                              ('extension',[r for r in band if r['cohort']=='extension']),
                              ('balance',[r for r in band if r['balanced']])):
            summary[b][name] = {}
            for k in curves:
                v = [r['arms'][k] for r in selected]
                summary[b][name][k] = dict(n_safe=len(v),fp=sum(r['alarm'] for r in v),
                    pre_first=sum(r['early'] for r in v),late_only=sum(r['later'] and not r['early'] for r in v),
                    crossing_over_125=sum(r['crossing_points']>125 for r in v),
                    longest_over_125=sum(r['longest_run']>125 for r in v))
    write(output,dict(scope='POSTSEAL_EAST_ALARM_SUPPORT',first=p['first'],
        safe_counts=summary,per_safe_shot=rows,seal_sha256=sha(folder/'seal.json')))
    print({b:s['extension'] for b,s in summary.items()},flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path)
    ap.add_argument('output',type=Path)
    a = ap.parse_args()
    run(a.folder.resolve(),a.output.resolve())
