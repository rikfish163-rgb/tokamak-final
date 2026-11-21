"""Audit the sequential B/S sampling controls without relabeling failures."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..event_adapt_20261022.train import load_source
from ..event_adapt_20261022.data import attach_training_events
from ..event_prefix_probe import read_events,DEFAULT_LABELS
from ..simple_online import _durable_json
from .analyze import load_model,cluster_interval,gradient_direction
from .prepare import validation_examples,diagnose
from .sampling import source_groups,balanced_pair


def main():
    torch.set_num_threads(2)
    root=Path(__file__).parents[1]/'runs';roles=io23.source_roles();adv=io23.advanced_times()
    train=load_source(roles['train'],adv);val=load_source(roles['val'],adv)
    grouped,_=read_events(DEFAULT_LABELS)
    train=attach_training_events(train,grouped,allowed_train_ids=roles['train'])
    results={}
    for arm,policy in [('B','balanced'),('S','safe_negative')]:
        directory=root/f'paired_feedback_{arm}_20261022';model,done,ck=load_model(directory)
        saved=json.loads((directory/'reporting_probe.json').read_text())
        examples=validation_examples(model,val,128,922);rerun=diagnose(model,examples)
        maxdiff=0.
        for old,new in zip(saved['records'],rerun['records']):
            if (old['shot'],old['end'],old['y'],old['condition'])!=(new['shot'],new['end'],new['y'],new['condition']):
                raise AssertionError('probe identities differ')
            for key in ('before_logit','after_logit','before_completion','after_completion','before_representation','after_representation'):
                maxdiff=max(maxdiff,abs(old[key]-new[key]))
            for key in ('before','after'):
                z=old[key+'_logit'];np.testing.assert_allclose(old[key+'_bce'],np.logaddexp(0.,z)-old['y']*z,atol=1e-12,rtol=0)
        if maxdiff>2e-6 or saved['probe_gate']!=rerun['probe_gate'] or done['allow_east_candidate']!=saved['probe_gate']:
            raise AssertionError('probe replay/gate mismatch')
        history=json.loads((directory/'history.json').read_text())
        if min(history,key=lambda h:h['criterion'])['step']!=done['selected_step']: raise AssertionError('selection mismatch')
        groups=source_groups(train,policy=policy);rng=np.random.default_rng(811)
        digest=hashlib.sha256();receipts=[]
        for _ in range(512):
            x,m,y,info=balanced_pair(train,groups,rng,model.cfg,'cpu');receipts.extend(info)
            for i in range(2):
                gain=np.exp(rng.uniform(-.35,.35,23)).astype('float32');bias=rng.uniform(-1.,1.,23).astype('float32')
                for a in (x[i:i+1],m[i:i+1],y[i:i+1]):digest.update(a.numpy().tobytes())
                digest.update(gain.tobytes());digest.update(bias.tobytes())
        saved_receipts=json.loads((directory/'sampling_receipts.json').read_text())
        inputs=json.loads((directory/'training_inputs.json').read_text())
        if receipts!=saved_receipts or digest.hexdigest()!=inputs['sequence_sha256']: raise AssertionError('sampling replay mismatch')
        if sum(r['y']==1 for r in receipts)!=512 or sum(r['y']==0 for r in receipts)!=512:
            raise AssertionError('class balance broken')
        if policy=='safe_negative' and any(r['y']==0 and r['is_disrupt'] for r in receipts):
            raise AssertionError('disruptive negative reached safe-negative arm')
        directions=gradient_direction(model,examples[:32]);_durable_json(directory/'gradient_directions.json',directions)
        results[arm]=dict(policy=policy,selected_step=done['selected_step'],summary=saved['summary'],
            gate=saved['probe_gate'],input_audit=inputs,replay_max_abs_diff=maxdiff,
            uncertainty={c:cluster_interval(saved['records'],c) for c in ('clean','shift')},seconds=done['seconds'],
            source_parent_unchanged=True)
        print(json.dumps(dict(stage='sampling_replay',arm=arm,maxdiff=maxdiff,inputs=inputs)),flush=True)
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/paired_sampling_summary.json'
    _durable_json(output,dict(status='VERIFIED_REPLAY_AND_RECOUNT',arms=results,probe_records=512,
        source_role='J-TEXT TRAIN only for optimization; VAL-only selection and diagnosis',
        independent_confirmation=False,east_accessed=False,sequential_exploration=True))
    print(json.dumps(dict(output=str(output))),flush=True)


if __name__=='__main__':main()
