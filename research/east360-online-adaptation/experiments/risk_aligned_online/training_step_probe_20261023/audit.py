"""Independently aggregate diagnostic traces; no new model selection."""
import argparse
import math
from pathlib import Path
import statistics
import time

from .. import io23
from ..credit_horizon_20261023.run import READ
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import PREVIOUS,PREVIOUS_SHA,PLAN


def distance(a,b):
    pairs=[(x,y) for x,y in zip(a['asinh_window_median'],b['asinh_window_median']) if x is not None and y is not None]
    if not pairs:raise ValueError('no common observed channels')
    return math.sqrt(sum((x-y)**2 for x,y in pairs)/len(pairs))


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    assert sha(PREVIOUS/'completed.json')==PREVIOUS_SHA==p['reference_sha256']==done['reference_sha256']
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256
    for n,h in done['artifact_sha256'].items():assert sha(root/n)==h,n
    for n,h in p['code_sha256'].items():assert sha(io23.ROOT/n)==h,n
    index={(r['fold'],r['arm'],r['shot']):r for r in done['rows']};checks=0;cases=[]
    for plan in p['folds']:
        f=plan['fold'];points=READ(root/f'fold{f}'/'anchors.json');rows=READ(root/f'fold{f}'/'trace.json')
        schedule=READ(PREVIOUS/f'fold{f}'/'schedule.json');old=READ(PREVIOUS/f'fold{f}'/'training.json')
        assert len(rows)==64 and [r['shot'] for r in points]==plan['held']
        for step,r in enumerate(rows):
            assert step==r['step']==schedule[step]['step']
            assert r['episodes']==schedule[step]['episodes'] and r['source_examples']==old[step]['source_examples']
            assert all(e['shot'] in plan['fit'] for e in r['episodes'])
            for a in ('joint','meta'):
                if step:assert r[a]['before_logits']==rows[step-1][a]['after_logits']
                assert r[a]['after_logits']==r[a]['hypothetical_logits']['full'];checks+=1
        for a in ('joint','meta'):
            for j,pt in enumerate(points):
                rec=index[f,a,pt['shot']]
                first=rows[0][a]['before_logits'][j];last=rows[-1][a]['after_logits'][j]
                assert first==rec['initial_logit'] and last==rec['final_logit']
                assert math.isclose(1/(1+math.exp(-first)),pt['original_probability'],rel_tol=2e-6,abs_tol=2e-6)
                changes=[r[a]['after_logits'][j]-r[a]['before_logits'][j] for r in rows]
                assert sum(changes)==rec['true_logit_change'] and sum(x<0 for x in changes)==rec['down_steps']
                for name,stats in rec['omission_minus_full'].items():
                    key=name if name=='zero_current' else 'omit_'+name
                    delta=[r[a]['hypothetical_logits'][key][j]-r[a]['after_logits'][j] for r in rows]
                    assert statistics.median(delta)==stats['median']
                    assert math.isclose(sum(delta),stats['sum'],abs_tol=1e-10)
                    assert sum(x>0 for x in delta)==stats['higher_count'] and sum(x<0 for x in delta)==stats['lower_count']
                order=sorted(range(64),key=lambda k:changes[k])[:5]
                assert order==rec['largest_decrease_steps']
                for k in order:
                    r=rows[k]
                    cases.append(dict(fold=f,arm=a,shot=pt['shot'],step_one_based=k+1,
                        change=changes[k],before_probability=1/(1+math.exp(-r[a]['before_logits'][j])),
                        after_probability=1/(1+math.exp(-r[a]['after_logits'][j])),
                        zero_current_change=r[a]['hypothetical_logits']['zero_current'][j]-r[a]['before_logits'][j],
                        training=[dict(shot=e['shot'],disruptive=e['disruptive'],first_target=e['ends'][0],
                            last_target=e['ends'][-1],representative_time=c['time'],
                            normalized_median_rms_distance=distance(pt['context'],c))
                            for e,c in zip(r['episodes'],r['endpoint_contexts'])]))
    assert checks==512 and len(index)==40 and done['exact_checkpoint_matches']==8
    access=READ(root/'access.json');train=set(io23.source_roles()['train'])
    for r in access:
        if r['device']=='J-TEXT':assert r['phase']=='source_train' and r['shot'] in train
        else:
            pl=p['folds'][r['fold']];assert r['phase'] in ('fit','posthoc_anchor')
            assert r['shot'] in pl['fit' if r['phase']=='fit' else 'held']
    _durable_json(root/'largest_decrease_cases.json',cases)
    result=dict(status='VERIFIED_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        verification_sha256=sha(Path(__file__)),checked_steps=checks,checked_anchor_summaries=len(index),
        full_checkpoint_matches_recorded=8,seconds=time.monotonic()-start,
        limitation='Exact real checkpoint replay is in run.py; this audit independently recomputes scalar summaries, not all Adam counterfactuals. Not a policy or test performance.')
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
