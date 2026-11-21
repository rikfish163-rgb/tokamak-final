"""Historical leave-one-shot-out retrieval screen; no TEST access."""
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
from ..arrived_teacher_20261023.learning import sample_queries as memory_queries
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..history_tangent_20261023.learning import logit,flat
from ..learned_update_loss_20261023.learning import summarize
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import REFERENCE,READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,normalize,step_all,sample_queries

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/历史片段检索微调_事前约定_20261023.md'


def prepare(model,ids,mapping,*,device='cuda'):
    windows=[];manifest=[];records={};signals={}
    for sid in ids:
        shot=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
        signal=transform({k:shot[k] for k in ('values','available')},mapping)
        records[sid]=shot;signals[sid]=signal
        for row in memory_queries(shot):
            end=row['query'];windows.append(simple_data.window(signal,end,128))
            manifest.append(dict(shot=sid,end=end,label=int(shot['y'][end]),group=row['group']))
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
    keys=[]
    for start in range(0,len(x),32):
        keys.append(model.predict_fast(x[start:start+32],m[start:start+32],initial_encoder(model),
            mode='risk_state')['risk_state'].detach().cpu().numpy())
    memory=dict(x=x,mask=m,keys=normalize(np.concatenate(keys)),
                shots=np.array([r['shot'] for r in manifest]),labels=np.array([r['label'] for r in manifest]))
    return memory,manifest,records,signals


def probe(model,signal,row,label,memory,*,device='cuda'):
    if row['query']!=row['count']+24:raise ValueError('fixed future25-point endpoint required')
    fast=initial_encoder(model)
    x,m=[torch.as_tensor(v[None],device=device) for v in simple_data.window(signal,row['query'],128)]
    z=logit(model,x,m,fast).reshape(())
    loss=F.binary_cross_entropy_with_logits(z.double(),z.new_tensor(float(label)).double())
    risk_gradient=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach().double()
    states,receipts,gradients=step_all(model,signal,row['count'],memory,excluded_shot=row['shot'],device=device)
    result=[]
    for arm in ARMS:
        after=logit(model,x,m,states[arm]).reshape(()).detach()
        after_loss=F.binary_cross_entropy_with_logits(after.double(),after.new_tensor(float(label)).double())
        g=gradients[arm].double();norm=float(g.norm());risk_norm=float(risk_gradient.norm())
        dot=float(g@risk_gradient);undefined=norm==0 or risk_norm==0
        result.append(dict(row,arm=arm,label=int(label),before=float(loss.detach()),after=float(after_loss),
            logit_before=float(z.detach()),logit_after=float(after),gradient_dot=dot,
            cosine=0. if undefined else dot/(norm*risk_norm),cosine_undefined=undefined,
            risk_gradient_norm=risk_norm,update=receipts[arm]))
    return result


def summarize_rows(rows):
    summary={arm:summarize([r for r in rows if r['arm']==arm]) for arm in ARMS}
    candidate=summary['nearest']
    gate=bool(candidate['passes_screen'] and all(candidate['balanced_after']<summary[a]['balanced_after']
                                               for a in ('initial','centroid')))
    return summary,gate


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    rp,rv=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    for f,digest in rp['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    ids=rp['selected_ids'];assert len(ids)==20 and not set(ids)&set(rp['contract']['shot_ids'])
    parent=REFERENCE/'conditional_moments.pt'
    assert sha(parent)==READ(REFERENCE/'preparation_receipts.json')['conditional_moments']['checkpoint_sha256']
    model,threshold=load_model();original=state_cpu(model);mapping=READ(REFERENCE/'input_map.json')
    output.mkdir(parents=True)
    paths={io23.ROOT/f for f in rp['code_sha256']};paths.update(Path(__file__).parent.glob('*.py'))
    for f in ('arrived_teacher_20261023/learning.py','task_geometry_20261023/run.py',
              'task_geometry_20261023/learning.py','history_tangent_20261023/learning.py',
              'learned_update_loss_20261023/learning.py','tests/test_history_retrieval.py'):
        paths.add(io23.ROOT/'experiments/risk_aligned_online'/f)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(paths):z.write(f,str(f.relative_to(io23.ROOT)))
    protocol=dict(reference=str(REFERENCE),parent=str(parent),parent_sha256=sha(parent),
        source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,selected_ids=ids,
        threshold=threshold,code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths},
        plan_path=str(PLAN),plan_sha256=sha(PLAN),
        reference_hashes={n:sha(REFERENCE/n) for n in ('protocol.json','completed.json','verification.json','input_map.json')},
        parameter_scope=model.encoder_names(),arms=list(ARMS),learning_rate=.001,clip_norm=1.,
        history_per_query='2binary-positive+2binary-negative windows; distinct identities within each class',
        loss='mean historical BCE; original fixed125ms labels; no current-shot outcome in update',
        retrieval='frozen parent CLS64 cosine; current or first128 prefix; class centroid control',
        controls='same requested step length as nearest; no current-query class supplied',
        screen='one step then future25-point endpoint; temporary state reset per probe',
        scope='leave-current-shot-out MEMORY only; parent/mapping trained all20; NOT independent validation',
        no_test_access=True,no_new_label_identities=True,no_alarm_change=True,no_inter=True,
        gate='nearest two main groups nonworse;balanced better;>=3/4D better;better than both controls',
        seed=131,torch_version=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json',protocol)
    access=[];outcome=io23.read_outcome;readshot=simple_data.read_shot
    def allowed_outcome(device,sid):
        if device!='EAST' or sid not in ids:raise AssertionError('unauthorized outcome')
        access.append(dict(device=device,shot=sid));return outcome(device,sid)
    def allowed_signal(device,sid,**kwargs):
        if device!='EAST' or sid not in ids:raise AssertionError('TEST signals forbidden')
        return readshot(device,sid,**kwargs)
    with patch.object(io23,'read_outcome',side_effect=allowed_outcome),patch.object(simple_data,'read_shot',side_effect=allowed_signal):
        memory,manifest,records,signals=prepare(model,ids,mapping)
        torch.save({k:torch.as_tensor(v).cpu() for k,v in memory.items()},output/'memory.pt')
        _durable_json(output/'memory_manifest.json',manifest)
        rows=[];sampling=[]
        for sid in ids:
            shot=records[sid]
            for index,row in enumerate(sample_queries(shot)):
                query=row['query'];label=int(shot['y'][query]);entry=dict(row,shot=sid,index=index)
                sampling.append(dict(entry,label=label,lead_ms=round(float((shot['down']-shot['times'][query])*1000),9)
                                     if shot['is_disrupt'] else None))
                rows.extend(probe(model,signals[sid],entry,label,memory))
            progress=dict(shot=sid,queries=len(sampling),seconds=time.monotonic()-started)
            _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(parent)==protocol['parent_sha256']
    _durable_json(output/'sampling.json',sampling);_durable_json(output/'probes.json',rows)
    _durable_json(output/'label_access.json',access)
    summary,gate=summarize_rows(rows)
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,summary=summary,
        recommend_full_stream=gate,parent_unchanged=True,source_unchanged=True,test_accessed=False,
        historical_windows=len(manifest),unique_history_windows=len({(r['shot'],r['end']) for r in manifest}),
        queries=len(sampling),unique_queries=len({(r['shot'],r['query']) for r in sampling}),
        neural_steps=sum(r['update']['gradient_steps'] for r in rows),
        artifact_sha256={f.name:sha(f) for f in output.iterdir() if f.suffix in ('.json','.pt','.zip')})
    _durable_json(output/'completed.json',done)
    print(json.dumps(dict(status='COMPLETED',recommend_full_stream=gate,
        summary={a:{k:v for k,v in s.items() if k!='per_shot'} for a,s in summary.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);main(parser.parse_args().output)
