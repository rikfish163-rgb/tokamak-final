"""Predeclared causal policy-function screen on offline selected probes."""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.learning import observed
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import feedback_prefix, collect_geometry, finite_step
from .policy import fit, predicted_alignment, decide
from .run import (READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
                  signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256)

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/历史通道收益学习_事前约定_20261024.md'


def summarize_policy(rows):
    result={}
    for group in ('preparation','calibration'):
        result[group]={}
        for flag in (False,True):
            take=[r for r in rows if r['group']==group and r['is_disrupt']==flag]
            ids=sorted({r['shot'] for r in take}); arms={}
            for arm in ('context','mean','shuffle'):
                arms[arm]={}
                for length in (.001,.01):
                    pairs=[(r,next(s for s in r['finite'][arm] if s['step_norm']==length)) for r in take]
                    metrics={}
                    for name,get in {
                        'risk_delta':lambda r,s:s['risk_after']-r['risk_before'],
                        'signal_delta':lambda r,s:s['signal_after']-r['signal_before'],
                        'risk_improvement_fraction':lambda r,s:float(s['risk_after']<r['risk_before']-1e-7),
                        'joint_improvement_fraction':lambda r,s:float(s['risk_after']<r['risk_before']-1e-7 and s['signal_after']<r['signal_before']-1e-7),
                        'zero_direction_fraction':lambda r,s:float(s['zero_direction']),
                    }.items():
                        metrics[name]=float(np.mean([np.mean([get(r,s) for r,s in pairs if r['shot']==sid]) for sid in ids]))
                    arms[arm][str(length)]=metrics
            result[group]['disruptive' if flag else 'safe']=dict(shots=len(ids),probes=len(take),arms=arms)
    return result


def eligible(summary):
    for kind in ('safe','disruptive'):
        arms=summary['calibration'][kind]['arms']
        for length in ('0.001','0.01'):
            delta=arms['context'][length]['risk_delta']
            if not (delta<0 and all(delta<arms[a][length]['risk_delta']-1e-7 for a in ('mean','shuffle'))):
                return False
    return True


def main(previous,root):
    if root.exists(): raise FileExistsError(root)
    assert READ(previous/'audit.json')['status']=='passed'
    done=READ(previous/'completed.json')
    for f,digest in done['artifact_sha256'].items(): assert sha(previous/f)==digest
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic(); parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed']
    original=state_cpu(model);mapping=READ(REFERENCE/'input_map.json')
    old=READ(previous/'protocol.json');history=old['preparation_ids'];cal=old['calibration_ids']
    root.mkdir(parents=True)
    records=[r for sid in history+cal for r in READ(previous/'probes'/f'{sid}.json')]
    # Build context strictly from the exact observed prefix; endpoint selection
    # was offline and is NOT passed off as a prospective acquisition policy.
    def deny(*args,**kwargs):raise AssertionError('no additional outcomes for policy context or execution')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in history+cal:
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            for r in [v for v in records if v['shot']==sid]:
                prefix,count=feedback_prefix(mapped,r['query'])
                *_,context=observed(model,prefix,count,initial_encoder(model),device='cuda')
                r['context']=context.tolist()
    _durable_json(root/'contexts.json',[{k:r[k] for k in ('shot','query','group','context')} for r in records])
    teacher=[r for r in records if r['group']=='preparation']
    policies={}
    for sid in history+[None]:
        ids=[s for s in history if s!=sid];training=[r for r in teacher if r['shot'] in ids]
        policies[str(sid)]={arm:fit(training,ids,shuffle=arm=='shuffle') for arm in ('context','shuffle')}
    _durable_json(root/'policies.json',policies)
    protocol=dict(previous_completed_sha256=sha(previous/'completed.json'),plan_sha256=sha(PLAN),
        policies_sha256=sha(root/'policies.json'),source_sha256=PARENT_SHA256,
        preparation_ids=history,calibration_ids=cal,scope='OFFLINE_SELECTED_PROBE_POLICY_SCREEN_NOT360_OR_ONLINE_GAIN',
        prior_parent_saw_all20=True,policy_leave_shot_out=True,step_norms=[.001,.01],
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')},
        test_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_feedback_cone_policy.py'))
    _durable_json(root/'protocol.json',protocol)
    outputs=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in history+cal:
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            p=policies[str(sid if sid in history else None)]
            assert sid not in p['context']['allowed_ids']
            for r in [v for v in records if v['shot']==sid]:
                # Decide BEFORE re-creating the future-gradient diagnostic state.
                decisions={}
                for arm in ('context','mean','shuffle'):
                    pred=(np.array(p['context']['target_mean']) if arm=='mean' else predicted_alignment(p[arm],r['context']))
                    w,receipt=decide(pred,r['signal_alignment'])
                    decisions[arm]=dict(predicted=pred.tolist(),weights=w.tolist(),receipt=receipt)
                fresh,state=collect_geometry(model,mapped,r['query'],r['label'],device='cuda')
                for key in ('risk_alignment','signal_alignment','risk_before','signal_before'):
                    np.testing.assert_array_equal(fresh[key],r[key])
                out={k:copy.deepcopy(r[k]) for k in ('shot','query','group','is_disrupt','risk_before','signal_before')}
                out.update(decisions=decisions,finite={})
                for arm,d in decisions.items():
                    direction=torch.tensor(d['weights'],device='cuda',dtype=state['parts'].dtype)@state['parts']
                    out['finite'][arm]=[finite_step(model,state,direction,length) for length in (.001,.01)]
                outputs.append(out)
            print(dict(stage='policy_screen',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True)
            if time.monotonic()-start>900:raise TimeoutError('15minute declared limit exceeded')
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    summary=summarize_policy(outputs);_durable_json(root/'probes.json',outputs);_durable_json(root/'summary.json',summary)
    result=dict(status='completed',seconds=time.monotonic()-start,probes=len(outputs),
                eligible_for_full10_stream=eligible(summary),source_and_parent_unchanged=True,
                artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.previous,args.out)
