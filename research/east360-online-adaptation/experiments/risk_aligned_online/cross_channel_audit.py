"""Descriptive mechanism checks on completed, sealed factorial experiments.

No result below is used as an online acceptor. An alarm following an update is
only timing evidence, not the same-input counterfactual used in a prior audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .simple_online import _durable_json,_hash


def quantiles(values):
    return np.quantile(values,[0,.25,.5,.75,.95,1]).tolist() if len(values) else None


def update_summary(audits,*,cap):
    attempts=[r for a in audits for r in a['attempts']]
    accepted=[r for r in attempts if r['accepted']]
    measured=[r for r in attempts if 'actual_encoding_rms' in r]
    if cap is not None and any(r['actual_encoding_rms']>cap*(1+3e-5)+1e-8 for r in measured):
        raise AssertionError('observed encoding-step cap violated')
    return dict(shots=len(audits),attempted=len(attempts),accepted=len(accepted),
        limited_attempts=sum(r['limited'] for r in measured),
        accepted_some_view_worsens=sum(bool(np.any(np.array(r['views_after'])>
            np.array(r['views_before'])+1e-7)) for r in accepted),
        accepted_mean_view_loss_increases=sum(bool(np.mean(r['views_after'])>
            np.mean(r['views_before'])+1e-7) for r in accepted),
        encoding_step_quantiles=quantiles([r['actual_encoding_rms'] for r in measured]),
        accepted_encoding_step_quantiles=quantiles([r['actual_encoding_rms'] for r in accepted]),
        step_scale_quantiles=quantiles([r['scale'] for r in measured]),
        final_fast_norm_quantiles=quantiles([a['fast_final']['l2'] for a in audits]),
        final_radius_ge095=sum(a['fast_final']['l2']>=.95 for a in audits),
        all_measured_steps_within_cap=True if cap is not None else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();out=args.run/'mechanism_audit.json'
    if out.exists():p.error('new analysis artifact required')
    completed=json.loads((args.run/'completed.json').read_text())
    protocol=json.loads((args.run/'protocol.json').read_text())
    order=json.loads((args.run/'order.json').read_text())
    threshold=protocol['threshold'];allstats={};static=None
    for arm,result in completed['results'].items():
        folder=args.run/arm
        decisions=json.loads((folder/'verification.json').read_text())['decisions']
        if [r['shot'] for r in decisions]!=[r['shot'] for r in order]:raise ValueError('identity mismatch')
        audits=[];first_safe=[];means=[];new_fp_next_step=[];first_new=[]
        if arm=='static':static=decisions
        if static is None:raise ValueError('static must precede adapted arms')
        for i,(row,base) in enumerate(zip(decisions,static),1):
            stem=f'{i:04d}_{row["shot"]}'
            audit=json.loads((folder/'stream'/f'{stem}_audit.json').read_text());audits.append(audit)
            if row['is_disrupt']:continue
            with np.load(folder/'stream'/f'{stem}_sealed.npz',allow_pickle=False) as z:
                k=np.flatnonzero(z['scores']>=threshold)
                first=int(k[0]) if len(k) else None
                first_safe.append(dict(shot=row['shot'],index=first))
                means.append(float(np.mean(z['scores'])))
            if first is not None and not base['alarm']:
                first_new.append(row['shot'])
                if any(r['accepted'] and r['effective_index']==first for r in audit['attempts']):
                    new_fp_next_step.append(row['shot'])
        bound=protocol['encoding_step_cap'] if arm.endswith('_limited') else None
        stats=update_summary(audits,cap=bound)
        if stats['accepted']!=result['accepted_updates']:raise AssertionError('accepted count mismatch')
        stats.update(safe_count=len(first_safe),safe_mean_of_shot_mean_probability=float(np.mean(means)),
            first_safe_alarm_before_point={str(q):sum(r['index'] is not None and r['index']<q
                for r in first_safe) for q in (1,2,4,8,16,32,64,128,153)},
            safe_alarm_before_first_possible_update=[r for r in first_safe
                if r['index'] is not None and r['index']<8],
            new_fp_shots=first_new,new_fp_first_alarm_immediately_after_update=new_fp_next_step,
            post_update_timing_is_not_causal_proof=True)
        allstats[arm]=stats
    results=completed['results']
    contrasts={}
    for metric in ('tp','fp','tpr','fpr','J'):
        r={k:v[metric] for k,v in results.items()}
        contrasts[metric]=dict(
            joint_minus_rotate_unbounded=r['joint_unbounded']-r['rotate_unbounded'],
            joint_minus_rotate_limited=r['joint_limited']-r['rotate_limited'],
            limit_minus_unbounded_rotate=r['rotate_limited']-r['rotate_unbounded'],
            limit_minus_unbounded_joint=r['joint_limited']-r['joint_unbounded'],
            interaction=r['joint_limited']-r['rotate_limited']-r['joint_unbounded']+r['rotate_unbounded'])
    _durable_json(out,dict(scope='DEVELOPMENT DESCRIPTIVE MECHANISM AUDIT; NO INDEPENDENT INFERENCE',
        stats=allstats,factorial_contrasts=contrasts,quantile_levels=[0,.25,.5,.75,.95,1],
        repeated_seeds=False,no_target_outcome_driven_policy=True,code_sha256=_hash(__file__),
        completed_sha256=_hash(args.run/'completed.json')))
    print(json.dumps(dict(arms=list(allstats),output=str(out))),flush=True)


if __name__=='__main__':main()
