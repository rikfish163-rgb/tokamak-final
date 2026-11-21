"""Historical-only neural gradient screen; TEST is never opened."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import REFERENCE,READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,encode,step,sample_queries,summarize

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/已到达后段表示反馈_事前约定_20261023.md'


def probe(model,signal,row,label,*,device='cuda'):
    q,count=row['query'],row['count']
    if q!=count+24:raise ValueError('fixed later query required')
    # Supervised diagnostic gradient is NEVER passed to step(). It is only
    # compared with the already-defined unlabeled direction after inference.
    origin=initial_encoder(model)
    h=encode(model,signal,[q],origin,device)
    logit=model.base.risk_head(h).reshape(())
    target=logit.new_tensor(float(label))
    before=F.binary_cross_entropy_with_logits(logit,target)
    true_grads=torch.autograd.grad(before,tuple(origin.values()))
    risk_norm=sum(g.double().square().sum() for g in true_grads).sqrt()
    results=[]
    for arm in ARMS:
        fast,receipt,gradient=step(model,signal,count,arm,device=device)
        h=encode(model,signal,[q],fast,device)
        after_logit=model.base.risk_head(h).reshape(())
        after=F.binary_cross_entropy_with_logits(after_logit,target)
        dot=sum(g.double().mul(gradient[k].double()).sum() for k,g in zip(origin,true_grads))
        aux_norm=sum(g.double().square().sum() for g in gradient.values()).sqrt()
        # Undefined angle for a zero gradient is recorded as zero, together
        # with both actual norms and a flag, never treated as improvement.
        undefined=bool(risk_norm==0 or aux_norm==0)
        cosine=0. if undefined else float(dot/(risk_norm*aux_norm))
        results.append(dict(row,arm=arm,label=int(label),logit_before=float(logit.detach()),
            logit_after=float(after_logit.detach()),bce_before=float(before.detach()),
            bce_after=float(after.detach()),actual_delta=float((after-before.detach()).detach()),
            risk_gradient_norm=float(risk_norm),aux_gradient_norm=float(aux_norm),
            gradient_dot=float(dot),cosine=cosine,cosine_undefined=undefined,
            first_order_delta=-.001*receipt.get('clip_factor',1.)*float(dot),update=receipt))
    return results


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    rp,rv=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    for file,digest in rp['code_sha256'].items():assert sha(io23.ROOT/file)==digest,file
    ids=rp['selected_ids'];assert len(ids)==20 and not set(ids)&set(rp['contract']['shot_ids'])
    assert sha(REFERENCE/'conditional_moments.pt')==READ(REFERENCE/'preparation_receipts.json')['conditional_moments']['checkpoint_sha256']
    model,threshold=load_model();frozen=state_cpu(model);mapping=READ(REFERENCE/'input_map.json')
    output.mkdir(parents=True)
    paths={io23.ROOT/f for f in rp['code_sha256']};paths.update(Path(__file__).parent.glob('*.py'))
    for f in ('task_geometry_20261023/run.py','task_geometry_20261023/learning.py',
              'tests/test_arrived_teacher.py','tests/test_task_geometry.py'):
        paths.add(io23.ROOT/'experiments/risk_aligned_online'/f)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(paths):z.write(file,str(file.relative_to(io23.ROOT)))
    protocol=dict(reference=str(REFERENCE),parent_sha256=sha(REFERENCE/'conditional_moments.pt'),
        source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,selected_ids=ids,
        threshold=threshold,parameter_scope=model.encoder_names(),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths},
        reference_hashes={n:sha(REFERENCE/n) for n in ('protocol.json','verification.json','completed.json','input_map.json')},
        plan_path=str(PLAN),plan_sha256=sha(PLAN),learning_rate=.001,clip_norm=1.,lag=25,query_after_update=25,
        gate='advance: safe and timely-positive equal-shot BCE nonworse,balanced strictly better,>=3/4D better',
        scope='historical TRAIN screen; parent already fitted these20; NO TEST performance or independent validation',
        sample='32 endpoints/safe;16early-negative+16timely-positive/disruptive; duplicates explicit',
        no_new_label_identities=True,no_extra_offline_optimization=True,no_alarm_change=True,no_inter=True,
        test_signals_accessed=False,test_outcomes_accessed=False,neural_updates=True,
        gradient_diagnostic='query true gradient compared only; never used by step or any deployed parameter')
    _durable_json(output/'protocol.json',protocol)
    access=[];outcome=io23.read_outcome;readshot=simple_data.read_shot
    def allowed_outcome(device,sid):
        if device!='EAST' or sid not in ids:raise AssertionError('not an authorized historical identity')
        access.append(dict(device=device,shot=int(sid)));return outcome(device,sid)
    def allowed_signal(device,sid,**kw):
        if device!='EAST' or sid not in ids:raise AssertionError('TEST signal forbidden in screen')
        return readshot(device,sid,**kw)
    rows=[];manifest=[]
    with patch.object(io23,'read_outcome',side_effect=allowed_outcome),patch.object(simple_data,'read_shot',side_effect=allowed_signal):
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            queries=sample_queries(s)
            for i,r in enumerate(queries):
                label=int(s['y'][r['query']]);record=dict(r,shot=sid,index=i)
                manifest.append(dict(record,label=label,time=float(s['times'][r['query']]),
                    lead_ms=round(float((s['down']-s['times'][r['query']])*1000),9) if s['is_disrupt'] else None))
                rows.extend(probe(model,signal,record,label))
            print(dict(shot=sid,probes=len(manifest),seconds=time.monotonic()-started),flush=True)
    assert_unchanged(model,frozen);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'sampling.json',manifest);_durable_json(output/'probes.json',rows)
    _durable_json(output/'label_access.json',access)
    summary=summarize(rows)
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,summary=summary,
        recommend_full_stream=summary['advance']['passes_screen'],neural_steps=sum(r['update']['gradient_steps'] for r in rows),
        unique_queries=len({(r['shot'],r['query']) for r in manifest}),total_queries=len(manifest),
        parent_unchanged=True,source_unchanged=True,test_accessed=False,protocol_sha256=sha(output/'protocol.json'),
        probes_sha256=sha(output/'probes.json'))
    _durable_json(output/'completed.json',done)
    print(json.dumps(dict(status='COMPLETED',recommend_full_stream=done['recommend_full_stream'],
        summary={a:{k:v for k,v in r.items() if k!='per_shot'} for a,r in summary.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
