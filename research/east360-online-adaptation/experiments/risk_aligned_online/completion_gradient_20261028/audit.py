"""Independent coverage/arithmetic checks and outcome-stratified diagnosis."""
import argparse
from pathlib import Path

import numpy as np

from ..context_memory_20261028.checkpoint import read
from ..simple_online import _durable_json
from ..simple_train import sha


def run(root,previous):
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    p=read(root,'protocol.json');done=read(root,'complete.json');rows=read(root,'records.json');coverage=read(root,'coverage.json')
    if not done['complete'] or sha(root/'records.json')!=done['records_sha256'] or sha(root/'coverage.json')!=done['coverage_sha256']:
        raise ValueError('completed unchanged attribution required')
    if sha(Path(__file__).with_name('analysis.py'))!=p['code_sha256']:raise ValueError('attribution code changed')
    if [r['shot'] for r in coverage]!=p['order'][20:] or len(coverage)!=340:raise ValueError('incomplete340 coverage')
    expected=[]
    for c in coverage:
        ends=list(range(127,c['samples'],128));n=min(4,len(ends))
        selected=[ends[int(v)] for v in np.linspace(0,len(ends)-1,n)] if n else []
        if c['ends']!=selected or len(set(selected))!=len(selected):raise ValueError('incorrect short-record selection')
        expected.extend((c['shot'],v) for v in selected)
    if [(r['shot'],r['end']) for r in rows]!=expected:raise ValueError('window coverage mismatch')
    for r in rows:
        q=r['loss'];frac=r['fraction_along_full']
        if not np.isclose(q['mean']+q['dynamic']+q['cls'],q['full'],rtol=1e-5,atol=1e-7):raise ValueError('MSE decomposition mismatch')
        for keys in (('mean','dynamic','cls'),('prediction_path','target_path')):
            if all(frac[k] is not None for k in keys) and not np.isclose(sum(frac[k] for k in keys),1.,rtol=1e-5,atol=1e-6):
                raise ValueError('gradient projection decomposition mismatch')
    predictions=read(previous,'per_shot.json');old_done=read(previous,'complete.json')
    if not old_done['complete']:raise ValueError('old360 incomplete')
    cold={r['shot']:r for r in predictions['cold']}
    valid=[r for r in rows if r['preterminal']]
    # Match the strict30ms endpoint contract, avoiding float-neighbor inclusion.
    near=[r for r in valid if r['disruptive'] and 30<round(r['lead_ms'],9)<=125]
    groups={'timely_positive':near,'miss_timely_positive':[r for r in near if not cold[r['shot']]['timely']],
            'safe':[r for r in valid if not r['disruptive']]}
    summary={g:dict(windows=len(rs),shots=len({r['shot'] for r in rs}),directions={
        k:dict(median=float(np.median([r['normalized_step_logit_derivative'][k] for r in rs])),
               positive=sum(r['normalized_step_logit_derivative'][k]>0 for r in rs))
        for k in ('full','prediction_path','target_path','dynamic')}) for g,rs in groups.items() if rs}
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),coverage340=True,distinct_windows=len(rows),
        posthoc_groups=summary,previous_complete_sha256=sha(previous/'complete.json'),
        no_online_performance_claim=True,zero_state_local_derivative_only=True,
        note='Outcome groups explain diagnosed directions; never an online selector or inferred TP/FP gain.')
    _durable_json(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path)
    a=p.parse_args();run(a.root,a.previous)
