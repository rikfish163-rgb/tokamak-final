"""Six-arm history-only screen; risk ranking is not a calibrated alarm claim."""
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
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..history_retrieval_20261023.learning import sample_queries
from ..history_retrieval_20261023.run import prepare
from ..history_tangent_20261023.learning import flat,logit
from ..learned_update_loss_20261023.learning import summarize
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model,REFERENCE
from .learning import ARMS,step_all,trajectory_vectors

OLD=io23.ROOT/'experiments/risk_aligned_online/runs/history_retrieval20_screen_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/有序轨迹检索与相对风险学习_事前约定_20261023.md'


def build_trajectories(model,memory,manifest,signals,*,device='cuda'):
    windows=[]
    for r in manifest:
        assert r['end']>=177
        for offset in (-50,-25,0):windows.append(simple_data.window(signals[r['shot']],r['end']+offset,128))
    all_states=[]
    for start in range(0,len(windows),32):
        x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows[start:start+32])]
        all_states.append(model.predict_fast(x,m,initial_encoder(model),mode='risk_state')['risk_state'].detach().cpu().numpy())
    states=np.concatenate(all_states).reshape(len(manifest),3,-1)
    vectors=[trajectory_vectors(h) for h in states]
    return dict(memory,trajectory_keys=np.stack([v['trajectory'] for v in vectors]),
                pooled_keys=np.stack([v['pooled'] for v in vectors])),states


def probe(model,signal,row,label,memory,*,device='cuda'):
    assert row['query']==row['count']+24
    fast=initial_encoder(model)
    x,m=[torch.as_tensor(v[None],device=device) for v in simple_data.window(signal,row['query'],128)]
    z=logit(model,x,m,fast).reshape(())
    loss=F.binary_cross_entropy_with_logits(z.double(),z.new_tensor(float(label)).double())
    q=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach().double()
    states,receipts,gradients=step_all(model,signal,row['count'],memory,excluded_shot=row['shot'],device=device)
    rows=[]
    for arm in ARMS:
        after=logit(model,x,m,states[arm]).reshape(()).detach()
        after_loss=F.binary_cross_entropy_with_logits(after.double(),after.new_tensor(float(label)).double())
        g=gradients[arm].double();dot=float(g@q);gn=float(g.norm());qn=float(q.norm());undefined=gn==0 or qn==0
        rows.append(dict(row,arm=arm,label=int(label),before=float(loss.detach()),after=float(after_loss),
            logit_before=float(z.detach()),logit_after=float(after),gradient_dot=dot,
            risk_gradient_norm=qn,cosine=0. if undefined else dot/(gn*qn),cosine_undefined=undefined,update=receipts[arm]))
    return rows


def summary_and_gate(rows):
    results={a:summarize([r for r in rows if r['arm']==a]) for a in ARMS}
    c=results['trajectory_rank']
    return results,bool(c['passes_screen'] and all(c['balanced_after']<results[a]['balanced_after']
                                                 for a in ARMS if a!='trajectory_rank'))


def verify_old_row(row,old):
    for k in old:
        if k not in ('arm','update'):assert row[k]==old[k],k
    for k in ('selected_indices','selected_labels','selected_shots','distances','excluded_shot',
              'last_observation','first_affected_prediction','gradient_steps','gradient_norm',
              'requested_step_norm','parameter_delta_norm','loss_before','loss_after'):
        assert row['update'][k]==old['update'][k],k


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    oldp,oldd,oldv=READ(OLD/'protocol.json'),READ(OLD/'completed.json'),READ(OLD/'verification.json')
    assert oldv['status']=='VERIFIED' and oldv['completed_sha256']==sha(OLD/'completed.json')
    for f,digest in oldp['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    for f,digest in oldd['artifact_sha256'].items():assert sha(OLD/f)==digest,f
    ids=oldp['selected_ids'];assert len(set(ids))==20 and not set(ids)&set(READ(REFERENCE/'protocol.json')['contract']['shot_ids'])
    for f,k in (('parent','parent_sha256'),('source_checkpoint','source_sha256')):assert sha(Path(oldp[f]))==oldp[k]
    model,threshold=load_model();origin=state_cpu(model);assert threshold==oldp['threshold']
    output.mkdir(parents=True)
    files={io23.ROOT/f for f in oldp['code_sha256']};files.update(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_trajectory_rank.py')
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):archive.write(f,str(f.relative_to(io23.ROOT)))
    protocol=dict(selected_ids=ids,old_reference=str(OLD),preparation_reference=str(REFERENCE),
        parent=oldp['parent'],parent_sha256=oldp['parent_sha256'],source_checkpoint=oldp['source_checkpoint'],
        source_sha256=oldp['source_sha256'],threshold=threshold,parameter_scope=model.encoder_names(),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},plan_path=str(PLAN),plan_sha256=sha(PLAN),
        old_hashes={f:sha(OLD/f) for f in ('protocol.json','completed.json','verification.json','memory.pt','memory_manifest.json','sampling.json','probes.json')},
        input_map_sha256=sha(REFERENCE/'input_map.json'),arms=list(ARMS),primary='trajectory_rank',
        retrieval_offsets=[-50,-25,0],history_per_class=2,query_horizon_points=25,
        step='each direction matched to state_bce SGD.001 clip1 length',
        reset='each diagnostic endpoint resets parent; no persistent updates',
        scope='history20 only; same-shot excluded from memory, parent trained all20; development diagnostic',
        gate='two primary groups nonworse;balanced better;>=3/4D;better than all5 controls',
        no_test_access=True,no_alarm_change=True,no_new_label_identities=True,
        seed=131,torch_version=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json',protocol)
    access=[];readshot,outcome=simple_data.read_shot,io23.read_outcome
    def allow_signal(device,sid,**kwargs):
        assert device=='EAST' and sid in ids,'non-history signal forbidden'
        return readshot(device,sid,**kwargs)
    def allow_outcome(device,sid):
        assert device=='EAST' and sid in ids,'non-history outcome forbidden'
        access.append(dict(device=device,shot=sid));return outcome(device,sid)
    oldrows={(r['shot'],r['query']):r for r in READ(OLD/'probes.json') if r['arm']=='nearest'}
    with patch.object(simple_data,'read_shot',side_effect=allow_signal),patch.object(io23,'read_outcome',side_effect=allow_outcome):
        memory,manifest,records,signals=prepare(model,ids,READ(REFERENCE/'input_map.json'))
        assert manifest==READ(OLD/'memory_manifest.json')
        cached=torch.load(OLD/'memory.pt',map_location='cpu',weights_only=True)
        for k,v in memory.items():torch.testing.assert_close(torch.as_tensor(v).cpu(),cached[k],rtol=0,atol=0)
        memory,states=build_trajectories(model,memory,manifest,signals)
        torch.save(dict(memory={k:torch.as_tensor(v).cpu() for k,v in memory.items()},states=torch.as_tensor(states)),output/'memory.pt')
        rows=[];sampling=[]
        for sid in ids:
            shot=records[sid]
            for index,r in enumerate(sample_queries(shot)):
                query=r['query'];y=int(shot['y'][query]);entry=dict(r,shot=sid,index=index)
                sampling.append(dict(entry,label=y,lead_ms=round(float((shot['down']-shot['times'][query])*1000),9)
                                     if shot['is_disrupt'] else None))
                values=probe(model,signals[sid],entry,y,memory)
                verify_old_row(values[0],oldrows[(sid,query)]);rows.extend(values)
            progress=dict(shot=sid,queries=len(sampling),seconds=time.monotonic()-started)
            _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert sampling==READ(OLD/'sampling.json');assert_unchanged(model,origin)
    for f,k in (('parent','parent_sha256'),('source_checkpoint','source_sha256')):assert sha(Path(oldp[f]))==oldp[k]
    for name,value in (('sampling',sampling),('probes',rows),('label_access',access)):_durable_json(output/(name+'.json'),value)
    summary,gate=summary_and_gate(rows)
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,summary=summary,recommend_full_stream=gate,
        parent_unchanged=True,source_unchanged=True,test_accessed=False,queries=len(sampling),historical_windows=len(manifest),
        exact_old_baseline_rows=len(oldrows),neural_steps=sum(r['update']['gradient_steps'] for r in rows),
        artifact_sha256={f.name:sha(f) for f in output.iterdir() if f.suffix in ('.json','.pt','.zip')})
    _durable_json(output/'completed.json',done)
    print(json.dumps(dict(status='COMPLETED',gate=gate,summary={a:{k:v for k,v in s.items() if k!='per_shot'}
                                                             for a,s in summary.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);main(parser.parse_args().output)
