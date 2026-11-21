"""Four source-only arms, fixed ten epochs and shared 22-channel targets."""
import argparse
import hashlib
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import validate_roles
from .forecast import direct_examples, forecast_screen
from .anchored import ARMS, AnchoredForecast


def summary(rows):
    out={}
    for group in ('all','D','S'):
        sub=[r for r in rows if group=='all' or r['disruptive']==(group=='D')]
        pred=float(np.mean([r['prediction'] for r in sub]));held=float(np.mean([r['persistence'] for r in sub]))
        out[group]=dict(shots=len(sub),prediction=pred,persistence=held,relative_reduction=1-pred/held,
                        improved=sum(r['prediction']<r['persistence'] for r in sub))
    return out


def run(output, reference):
    if output.absolute().parent!=io23.ROOT/'probe_outputs' or output.exists():
        raise ValueError('new output in worktree required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    roles=simple_data.source_roles();validate_roles(roles)
    rp=read(reference,'protocol.json');rd=read(reference,'complete.json');cache=Path(rp['cache'])
    weights_path=Path(rp['reference'])/'forecast_weights.json'
    if (not rd['complete'] or rp['canonical_train_ids']!=roles['train'] or rp['val_ids']!=roles['val']
            or sha(cache/'inventory.json')!=rp['cache_inventory_sha256']
            or sha(weights_path)!=rp['forecast_weights_sha256'] or sha(DEFAULT_PARENT)!=PARENT_SHA256):
        raise ValueError('verified fixed source cache and metric required')
    weights=torch.tensor(read(weights_path.parent,weights_path.name)['weights']);weights[:,2]=0
    examples={};exclusions=[]
    for r in read(cache,'inventory.json'):
        sid,role=r['shot'],r['role'];path=cache/'cache'/f'{sid}.pt'
        if role not in ('train','val') or sid not in roles[role] or sha(path)!=r['cache_sha256']:
            raise ValueError('fixed TRAIN/VAL examples only')
        e=torch.load(path,map_location='cpu',weights_only=True)
        if len(e['means'])<=4:
            exclusions.append(sid);continue
        x,y,m=direct_examples(e['means'],e['complete']);w=m*weights
        if w.sum()<=0:raise ValueError('no eligible common22 forecast target')
        examples[sid]=dict(x=x,y=y,w=w,role=role,disruptive=bool(e['flag']))
    if set(exclusions)!={1052590,1053119,1053639,1057940}:raise ValueError('source eligibility changed')
    train=[sid for sid in roles['train'] if sid in examples]
    if len(train)!=1349 or not set(roles['val'])<=set(examples):raise ValueError('source split incomplete')
    counts={flag:sum(examples[s]['disruptive']==flag for s in train) for flag in (False,True)}
    pieces={k:[] for k in ('x','y','w')}
    for sid in train:
        e=examples[sid];pieces['x'].append(e['x']);pieces['y'].append(e['y'])
        pieces['w'].append(e['w']/e['w'].sum()*(.5/counts[e['disruptive']]))
    dataset={k:torch.cat(v).cuda() for k,v in pieces.items()};n=len(dataset['x'])
    if abs(float(dataset['w'].sum())-1)>1e-5:raise ValueError('shot/class weighting differs')
    models={k:AnchoredForecast(anchored=a,omit_mnm=m).cuda() for k,(a,m) in ARMS.items()}
    initial={k:v.clone() for k,v in models['direct_native'].state_dict().items()}
    for model in models.values():model.load_state_dict(initial)
    output.mkdir(parents=True);plan=io23.ROOT/'ANCHORED_FORECAST_PLAN.md'
    files=[Path(__file__),Path(__file__).parent/'anchored.py',Path(__file__).parent/'forecast.py']
    protocol=dict(profile='anchored_forecast_factorial_v1_NOT_A_RISK_MODEL',arms=ARMS,
        candidate='anchor_no_mnm',seed=131,epochs=10,learning_rate=.0003,batch_size=512,gradient_clip=1.,
        source_sha256=PARENT_SHA256,reference=str(reference),reference_protocol_sha256=sha(reference/'protocol.json'),
        cache=str(cache),cache_inventory_sha256=sha(cache/'inventory.json'),train_ids=train,
        canonical_train_ids=roles['train'],val_ids=roles['val'],excluded=exclusions,forecast_origins=n,
        target_channels=[c for c in io23.RAW23 if c!='MNM'],weights=weights.tolist(),
        source_classifier_updates=0,east_records_read=0,test220_records_read=0,phase_labels_read=0,
        same_initialization=True,effective_capacity_equal=False,nominal_parameters=11932,
        plan_sha256=sha(plan),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(plan,plan.name)
    start=time.monotonic();results={}

    @torch.no_grad()
    def evaluate(model):
        rows=[]
        for sid in roles['val']:
            e=examples[sid];x,y,w=(e[k].cuda() for k in ('x','y','w'))
            pred=torch.cat([model(x[i:i+512]) for i in range(0,len(x),512)])
            rows.append(dict(shot=sid,disruptive=e['disruptive'],
                prediction=float(((pred-y).square()*w).sum()/w.sum()),persistence=float((y.square()*w).sum()/w.sum())))
        return dict(rows=rows,summary=summary(rows),gate=forecast_screen(rows))

    for name,model in models.items():
        arm=output/name;arm.mkdir();rng=np.random.default_rng(131)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003);history=[]
        for epoch in range(1,11):
            cpu_order=rng.permutation(n);order=torch.tensor(cpu_order,device='cuda');losses=[];grads=[]
            for lo in range(0,n,512):
                ids=order[lo:lo+512];pred=model(dataset['x'][ids])
                loss=((pred-dataset['y'][ids]).square()*dataset['w'][ids]).sum()*n/len(ids)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)))
                optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,steps=len(losses),nonzero_gradient_steps=sum(g>0 for g in grads),
                loss=float(np.mean(losses)),order_sha256=hashlib.sha256(cpu_order.tobytes()).hexdigest()))
        result=evaluate(model);_durable_json(arm/'val_epoch10.json',result);_durable_json(arm/'training.json',history)
        torch.save(dict(profile=protocol['profile'],arm=name,protocol_sha256=sha(output/'protocol.json'),
                        model={k:v.detach().cpu() for k,v in model.state_dict().items()}),arm/'forecast_epoch10.pt')
        results[name]=dict(summary=result['summary'],gate=result['gate'],checkpoint_sha256=sha(arm/'forecast_epoch10.pt'))
        print(dict(arm=name,summary=result['summary'],seconds=time.monotonic()-start),flush=True)
    orders=[[r['order_sha256'] for r in read(output/k,'training.json')] for k in ARMS]
    if any(v!=orders[0] for v in orders):raise ValueError('training orders differ')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during experiment')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or sha(plan)!=protocol['plan_sha256']:raise ValueError('parent/plan changed')
    _durable_json(output/'complete.json',dict(complete=True,results=results,seconds=time.monotonic()-start,
        source_classifier_unchanged=True,east_records_read=0,test220_records_read=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.reference)
