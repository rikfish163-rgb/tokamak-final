"""How well do symmetric neural actions expose a direction to real EAST feedback?"""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logit

from . import commissioning_opposed_feedback_20261114 as experiment
from .history_gradient_metric_20261102.run import read, write
from .hdf5_score_recheck_20261103 import score_curve


TRIALS = ('pos','neg','observation_current','observation_earlier','predicted_only',
          'history_fixed','joint_current','joint_earlier','joint_reversed')


def effect(z, changed, i):
    return float(logit(np.clip(changed[i],1e-7,1.-1e-7))-logit(np.clip(z[i],1e-7,1.-1e-7)))


def orientation_agreement(positive,negative,selected,should_rise):
    """A zero write or identical two actions has no decision orientation."""
    wins = np.sign(positive-negative)*(1. if should_rise else -1.)
    chosen = np.sign(np.abs(selected-negative)-np.abs(selected-positive))
    use = (wins!=0.) & (chosen!=0.)
    return dict(known=int(use.sum()),support=len(use),
                proportion=float((wins[use]==chosen[use]).mean()) if use.any() else None)


def summarize(out):
    p,s,matrix = (read(out/k) for k in ('protocol.json','seal.json','per_shot.json'))
    old = matrix['0.2']
    result = read(out/'result.json')
    threshold = p['calibration']['0.2']
    hfields = ('positive', 'negative', 'segments')
    coefs = {k:[] for k in hfields}
    queries = []
    block_exposure = []
    for block in range(1,19):
        c = read(out/'blocks'/f'{block:02d}_commit.json')
        r = read(out/'blocks'/f'{block:02d}_reveal.json')
        a = read(out/'blocks'/f'{block:02d}_learning.json')
        if c['sealed']!=s['curves'][(block-1)*20:block*20] or c['rank_end']!=block*20:
            raise ValueError('saved predictions and block commit differ')
        if c['selected'] != [k['shot'] for k in r['outcomes']] or c['selected'] != [k['shot'] for k in a['query']]:
            raise ValueError('query IDs changed across seal, reveal, and training')
        if any(x['read_start_ns']<c['committed_ns'] for x in r['outcomes']):
            raise ValueError('outcome read early')
        for row in a['query']:
            if row['available_from']!=block*20+1:
                raise ValueError('invalid future eligibility of the teacher row')
            queries.append(row)
        block_exposure.append(dict(block=block,weights=a['applied'],historical_records=a['active_history']))
    if len(queries)!=16:
        raise ValueError('this is the fixed 16-query replay')
    post_query = [q['shot'] for q in queries]
    truth = {r['shot']:r['disruptive'] for r in old['bias_only']}
    if any(x['disruptive']!=truth[x['shot']] for x in queries):
        raise ValueError('query label differs from contemporaneous HDF5')
    group_rows = {k:[] for k in ('never_near','late_near','false_editable')}
    print('sealed history',len(queries),flush=True)
    example = {}
    for i,(sid,fn) in enumerate(zip(s['order'],s['curves'],strict=True)):
        if i<20:
            continue
        y,down,t = experiment.group.hdf._raw_truth(sid)
        with np.load(out/'shots'/fn,allow_pickle=False) as f:
            a = dict(f)
        z = a['bias_only']
        active = a['dose']>0.
        if y:
            ms = np.round((down-t)*1e3,9)
            chosen = active & (ms>30.) & (ms<=125.)
            if old['current_5'][i]['timely']:
                continue
            g = 'late_near' if old['current_5'][i]['alarm'] else 'never_near'
        elif old['bias_only'][i]['alarm']:
            first = np.flatnonzero(z>=threshold)[0]
            # Count immutable first alarms separately. The later suffix does
            # not retroactively earn protection.
            if not active[first]:
                continue
            chosen = (t>=t[first]-.125) & (t<=t[first]) & active
            ms = (t[first]-t)*1e3
            g = 'false_editable'
        else:
            continue
        if not chosen.any():
            continue
        pool = np.flatnonzero(chosen)
        at = int(pool[np.argmax(z[pool])])
        row = dict(shot=sid,rank=i+1,example_query=sid in post_query,writable=int(chosen.sum()),
                   eval_at=at,time_to_event_or_original_fp_ms=float(ms[at]),
                   base_prob=float(z[at]),effects={k:effect(z,a[k],at) for k in TRIALS})
        # At matched-dose endpoints each decision is a genuine selected sign.
        # Track how often it follows the better sign for the *actual* outcome
        # without using that outcome to pick which window to score online.
        history_active = any(block_exposure[i//20]['weights'])
        row['sign_alignment'] = {k:orientation_agreement(a['pos'][chosen],a['neg'][chosen],a[k][chosen],y)
                                 for k in TRIALS[2:] if k!='history_fixed' or history_active}
        row['near_opportunity'] = bool(y and ((np.maximum(a['pos'],a['neg'])>=threshold)&chosen).any())
        row['actual_correction'] = {k:score_curve(t,a[k],y,down,threshold) for k in (
            'pos','neg','observation_current','observation_earlier','joint_current','joint_reversed')}
        group_rows[g].append(row)
        if sid in experiment.CASES:
            example[str(sid)]=row
    for q in queries:
        coefs['positive' if q['disruptive'] else 'negative'].extend(q['targets'])
        coefs['segments'].append(len(q['targets']))
    reduced = {}
    for k,rows in group_rows.items():
        reduced[k] = dict(n=len(rows),writeable=sum(r['writable'] for r in rows),
            median_effect={a:float(np.median([r['effects'][a] for r in rows])) for a in TRIALS},
            median_direction_agreement={a:float(np.median(v)) for a in TRIALS[2:] if (v:=[
                x['proportion'] for r in rows if a in r['sign_alignment']
                and (x:=r['sign_alignment'][a])['known']>0])})
    write(out/'orientation_attribution.json',dict(exposure=block_exposure,query_rows=queries,query_support=coefs,
        strata=reduced,per_shot=group_rows,cases=example,
        finite_action_opportunities=result['postseal']['near_potential']))
    print(reduced,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    summarize(cli.parse_args().output.resolve())
