"""Post-hoc finite-step/generalization diagnosis; never fits a deployable policy."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..history_tangent_20261023.learning import apply_step
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import GROUPS,collect_shot,future_groups,bounded_step,_risk_gradient

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/部署时钟方向_有限步与泛化诊断约定_20261025.md'


def summarize(rows):
    summary={};per_shot=[]
    for scope in ('preparation','calibration'):
        summary[scope]={}
        for group in GROUPS:
            take=[r for r in rows if r['scope']==scope and r['group']==group]
            ids=sorted({r['shot'] for r in take})
            records=[]
            for sid in ids:
                rs=[r for r in take if r['shot']==sid]
                record=dict(scope=scope,group=group,shot=sid,blocks=len(rs))
                for key in ('predicted_risk_delta','actual_risk_delta','signal_delta','step_norm'):
                    record[key]=float(np.mean([r[key] for r in rs]))
                record['risk_improvement_fraction']=float(np.mean([r['actual_risk_delta']<0 for r in rs]))
                records.append(record);per_shot.append(record)
            summary[scope][group]=dict(shots=len(ids),blocks=len(take),
                **{key:float(np.mean([r[key] for r in records])) for key in
                   ('predicted_risk_delta','actual_risk_delta','signal_delta','step_norm','risk_improvement_fraction')})
    return summary,per_shot


def main(previous,root):
    if root.exists():raise FileExistsError(root)
    done=READ(previous/'completed.json');assert done['status']=='completed' and done['results'] is not None
    for file,digest in done['artifact_sha256'].items():assert sha(previous/file)==digest,file
    p=READ(previous/'protocol.json');policy=READ(previous/'policy.json');w=np.array(policy['block'])
    outcomes={r['shot']:r for r in READ(previous/'outcomes.json')}
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');root.mkdir(parents=True);start=time.monotonic()
    _durable_json(root/'protocol.json',dict(previous_completed_sha256=sha(previous/'completed.json'),
        plan_sha256=sha(PLAN),code_sha256=sha(Path(__file__)),policy_sha256=sha(previous/'policy.json'),
        scope='POSTHOC_DIAGNOSTIC_NO_POLICY_CHANGE_NO_NEW360',step_bound=.01,new_outcomes_accessed=0))
    rows=[]
    def deny(*args,**kwargs):raise AssertionError('only previously archived labels for this diagnostic')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for scope,ids in (('preparation',p['preparation_ids']),('calibration',p['shot_ids'])):
            for sid in ids:
                raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
                if scope=='preparation':
                    old=READ(previous/'geometry'/f'{sid}.json');flag,down=old['is_disrupt'],old['down'];geometry=old['rows']
                else:
                    flag,down=outcomes[sid]['is_disrupt'],outcomes[sid]['down']
                    geometry,_=collect_shot(model,s,raw['times'],sid,flag,down)
                for count in sorted({r['count'] for r in geometry}):
                    fast=initial_encoder(model)
                    parts,_=signal_parts(model,s,count,fast,hold=False,device='cuda')
                    if parts is None:raise AssertionError('archived geometry requires measured targets')
                    step=bounded_step(parts,w);changed=apply_step(fast,step)
                    x,m,target,tm,_=forecast_examples({k:v[:count] for k,v in s.items()},count,model.cfg,'cuda')
                    valid=target_mask(x,m,target,tm)
                    before=model.base.forecast_loss(model.predict_fast(x,m,fast)['forecast'],target,valid)
                    after=model.base.forecast_loss(model.predict_fast(x,m,changed)['forecast'],target,valid)
                    groups=future_groups(raw['times'],count,flag,down)
                    for r in [r for r in geometry if r['count']==count]:
                        _,base=_risk_gradient(model,s,groups[r['group']],fast,r['label'],device='cuda')
                        assert base==r['risk']
                        _,updated=_risk_gradient(model,s,groups[r['group']],changed,r['label'],device='cuda')
                        rows.append(dict(scope=scope,shot=sid,count=count,group=r['group'],
                            predicted_risk_delta=float(-.01*np.array(r['a'])@w),
                            actual_risk_delta=updated['loss']-base['loss'],signal_delta=float((after-before).detach()),
                            before=base,after=updated,step_norm=float(step.double().norm())))
                print(dict(stage='finite_diagnostic',scope=scope,shot=sid,seconds=round(time.monotonic()-start,2)),flush=True)
                if time.monotonic()-start>1200:raise TimeoutError('declared20minute diagnosis limit exceeded')
    summary,per_shot=summarize(rows)
    _durable_json(root/'rows.json',rows);_durable_json(root/'summary.json',summary);_durable_json(root/'per_shot.json',per_shot)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',rows=len(rows),seconds=time.monotonic()-start,summary=summary,
        source_and_parent_unchanged=True,policy_changed=False,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result)
    print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.previous,a.out)
