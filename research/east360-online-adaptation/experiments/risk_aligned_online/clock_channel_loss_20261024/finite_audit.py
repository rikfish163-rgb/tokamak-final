"""Recount finite-step summaries independently and replay two diagnostic blocks."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..history_tangent_20261023.learning import apply_step,logit
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import future_groups,bounded_step


def main(previous,root):
    done=READ(root/'completed.json');assert done['status']=='completed'
    for file,digest in done['artifact_sha256'].items():assert sha(root/file)==digest,file
    rows=READ(root/'rows.json');summary=READ(root/'summary.json')
    fields=('predicted_risk_delta','actual_risk_delta','signal_delta','step_norm','risk_improvement_fraction')
    for scope,groups in summary.items():
        for group,old in groups.items():
            take=[r for r in rows if r['scope']==scope and r['group']==group]
            ids={r['shot'] for r in take};assert len(ids)==old['shots'] and len(take)==old['blocks']
            for key in fields:
                total=0.
                for sid in ids:
                    local=[r for r in take if r['shot']==sid]
                    value=sum(float(r['actual_risk_delta']<0) if key=='risk_improvement_fraction' else r[key] for r in local)/len(local)
                    total+=value/len(ids)
                np.testing.assert_allclose(total,old[key],atol=1e-12,rtol=1e-10)
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');w=READ(previous/'policy.json')['block']
    targets=[next(r for r in rows if r['shot']==63226 and r['count']==3712 and r['group']=='early'),
             max((r for r in rows if r['shot']==89584 and r['group']=='danger'),key=lambda r:r['count'])]
    replays=[]
    def deny(*args,**kwargs):raise AssertionError('no new labels in diagnosis audit')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for r in targets:
            sid=r['shot'];raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            o=(READ(previous/'geometry'/f'{sid}.json') if r['scope']=='preparation'
               else next(o for o in READ(previous/'outcomes.json') if o['shot']==sid))
            ends=future_groups(raw['times'],r['count'],o['is_disrupt'],o['down'])[r['group']]
            fast=initial_encoder(model);parts,_=signal_parts(model,s,r['count'],fast,hold=False,device='cuda')
            step=bounded_step(parts,w);changed=apply_step(fast,step)
            measured={}
            for name,state in (('before',fast),('after',changed)):
                scores=[]
                for lo in range(0,len(ends),32):
                    windows=[simple_data.window(s,int(e),128) for e in ends[lo:lo+32]]
                    x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
                    scores.extend(logit(model,x,m,state).detach().cpu().tolist())
                end=int(ends[int(np.argmax(scores))])
                x,m=[torch.as_tensor(v[None],device='cuda') for v in simple_data.window(s,end,128)]
                z=logit(model,x,m,state)
                label=float(r['group']=='danger')
                loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.full_like(z,label))
                measured[name]=float(loss.detach());assert end==r[name]['end']
                assert float(z.detach()[0])==r[name]['logit'] and measured[name]==r[name]['loss']
            assert measured['after']-measured['before']==r['actual_risk_delta']
            replays.append(dict(shot=sid,count=r['count'],group=r['group'],actual_risk_delta=r['actual_risk_delta']))
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='passed',completed_sha256=sha(root/'completed.json'),rows=len(rows),
        independently_recounted_group_metrics=30,replayed_blocks=replays,source_and_parent_unchanged=True,
        audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.previous,a.run)
