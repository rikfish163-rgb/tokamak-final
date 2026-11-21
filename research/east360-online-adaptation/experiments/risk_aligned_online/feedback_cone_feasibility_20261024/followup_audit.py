"""Recompute policy fits/decisions and replay complete fixed calibration shots."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.learning import observed
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import feedback_prefix,collect_geometry,finite_step
from .policy import fit,predicted_alignment,decide
from .policy_run import summarize_policy,eligible
from .run import (READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,signal,transform,
                  REFERENCE,DEFAULT_PARENT,PARENT_SHA256)
from .stream import ARMS,predict,gate


def numeric_equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:numeric_equal(a[k],b[k])
    elif isinstance(a,(str,bool)) or a is None:
        assert a==b
    else:np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-9)


def verify_hashes(root):
    done=READ(root/'completed.json');assert done['status']=='completed'
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    return done


def main(previous,policy_root,stream_root):
    pd=verify_hashes(policy_root);sd=verify_hashes(stream_root)
    contexts={(r['shot'],r['query']):r['context'] for r in READ(policy_root/'contexts.json')}
    p=READ(policy_root/'protocol.json');history=p['preparation_ids'];cal=p['calibration_ids']
    teacher=[dict(r,context=contexts[(r['shot'],r['query'])]) for sid in history for r in READ(previous/'probes'/f'{sid}.json')]
    policies=READ(policy_root/'policies.json')
    for key,arms in policies.items():
        excluded=None if key=='None' else int(key);ids=[s for s in history if s!=excluded]
        rows=[r for r in teacher if r['shot'] in ids]
        for arm,archived in arms.items():numeric_equal(fit(rows,ids,shuffle=arm=='shuffle'),archived)
    results=READ(policy_root/'probes.json');old={}
    for sid in history+cal:
        for r in READ(previous/'probes'/f'{sid}.json'):old[(sid,r['query'])]=r
    for r in results:
        key=(r['shot'],r['query']);a=policies[str(r['shot'] if r['shot'] in history else None)]
        assert r['shot'] not in a['context']['allowed_ids']
        for arm,decision in r['decisions'].items():
            prediction=np.array(a['context']['target_mean']) if arm=='mean' else predicted_alignment(a[arm],contexts[key])
            w,receipt=decide(prediction,old[key]['signal_alignment'])
            numeric_equal(dict(predicted=prediction.tolist(),weights=w.tolist(),receipt=receipt),decision)
    summary=summarize_policy(results);numeric_equal(summary,READ(policy_root/'summary.json'))
    assert eligible(summary)==pd['eligible_for_full10_stream']
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');prior=np.array(policies['None']['context']['target_mean'])
    replay=[]
    def deny(*args,**kwargs):raise AssertionError('no additional outcomes in audit')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (89272,89284,89584):
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            r=max((r for r in results if r['shot']==sid),key=lambda v:v['query']);query=r['query'];key=(sid,query)
            prefix,count=feedback_prefix(mapped,query)
            *_,context=observed(model,prefix,count,initial_encoder(model),device='cuda')
            np.testing.assert_array_equal(context,contexts[key])
            record,state=collect_geometry(model,mapped,query,old[key]['label'],device='cuda')
            for arm,d in r['decisions'].items():
                direction=torch.as_tensor(d['weights'],dtype=state['parts'].dtype,device='cuda')@state['parts']
                assert [finite_step(model,state,direction,v) for v in (.001,.01)]==r['finite'][arm]
            curves,updates=predict(model,mapped,prior)
            with np.load(stream_root/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'],z['times'])
                for arm in ('off',)+ARMS:np.testing.assert_array_equal(curves[arm],z[arm])
            assert updates==READ(stream_root/'updates'/f'{sid}.json')
            replay.append(dict(shot=sid,query=query,complete_stream_replayed=True))
    # Independent shot counting and loss arithmetic, not the evaluator helper.
    outcomes={r['shot']:r for r in READ(stream_root/'outcomes.json')}
    threshold=READ(stream_root/'protocol.json')['threshold'];metrics={}
    for arm in ('off',)+ARMS:
        tp=fp=0;danger=[];safe=[]
        for sid in cal:
            o=outcomes[sid]
            with np.load(stream_root/'shots'/f'{sid}.npz') as z:
                lead=np.round((o['down']-z['times'])*1000,9) if o['is_disrupt'] else None
                valid=lead>=-1e-6 if o['is_disrupt'] else np.ones(len(z['times']),bool)
                scores=z[arm][valid].astype(np.float64)
                alarms=np.flatnonzero(scores>=threshold)
                if o['is_disrupt']:
                    timely=bool(len(alarms) and lead[valid][alarms[0]]>30.)
                    tp+=int(timely);danger.append(float(-np.log(scores[(lead[valid]>30)&(lead[valid]<=125)].clip(1e-12,1)).mean()))
                else:
                    fp+=int(bool(len(alarms)));safe.append(float(-np.log1p(-scores.max().clip(0,1-1e-12))))
        metrics[arm]=dict(tp=tp,fp=fp,danger_loss=float(np.mean(danger)),safe_loss=float(np.mean(safe)),balanced_loss=float((np.mean(danger)+np.mean(safe))/2))
        for k,v in metrics[arm].items():numeric_equal(v,sd['results'][arm][k])
    assert gate(sd['results'])==sd['eligible_for360']
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='passed',policy_completed_sha256=sha(policy_root/'completed.json'),stream_completed_sha256=sha(stream_root/'completed.json'),
        ridge_refits=42,policy_decisions=3*len(results),stream_decisions=70,replayed=replay,independent_stream_metrics=metrics,
        source_and_parent_unchanged=True,audit_code_sha256=sha(Path(__file__)))
    _durable_json(stream_root/'followup_audit.json',report);print({k:v for k,v in report.items() if k!='independent_stream_metrics'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--stream',type=Path,required=True)
    a=p.parse_args();main(a.previous,a.policy,a.stream)
