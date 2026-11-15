"""Two source-only head fits with matched episodes and fixed VAL diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import GROUPS,STEP,query_pools,assemble,train_step,paired_risk


def load_split(ids,advanced):
    original_signals=io23.read_signals;original_outcome=io23.read_outcome;allowed=set(ids);calls=[]
    def signal(device,sid,*args,**kwargs):
        if device!='J-TEXT' or sid not in allowed:raise ValueError('out-of-role signal access')
        calls.append(sid);return original_signals(device,sid,*args,**kwargs)
    def outcome(device,sid):
        if device!='J-TEXT' or sid not in allowed:raise ValueError('out-of-role outcome access')
        return original_outcome(device,sid)
    with patch.object(io23,'read_signals',side_effect=signal),patch.object(io23,'read_outcome',side_effect=outcome):
        shots=[simple_data.attach_outcome(simple_data.read_shot('J-TEXT',sid),'J-TEXT',advanced) for sid in ids]
    if calls!=ids:raise ValueError('unexpected read sequence')
    return shots


def snapshot(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items() if 'forecast_head.' not in k}


def group_summary(rows):
    summary={}
    for group in GROUPS:
        selected=[r for r in rows if r['group']==group]
        ids=sorted({r['shot'] for r in selected});shot_delta=[];shot_cos=[];shot_signal=[]
        if not ids:raise ValueError('all predeclared validation strata required')
        for sid in ids:
            sub=[r for r in selected if r['shot']==sid]
            shot_delta.append(np.mean([r['after']-r['before'] for r in sub]))
            shot_cos.append(np.mean([r['cosine'] for r in sub]))
            shot_signal.append(np.mean([r['signal'] for r in sub]))
        summary[group]=dict(shots=len(ids),endpoints=len(selected),mean_risk_change=float(np.mean(shot_delta)),
            mean_cosine=float(np.mean(shot_cos)),mean_signal_loss=float(np.mean(shot_signal)),
            improved_shots=int(np.sum(np.array(shot_delta)<-1e-6)),worsened_shots=int(np.sum(np.array(shot_delta)>1e-6)))
    summary['macro_change']=float(np.mean([summary[g]['mean_risk_change'] for g in GROUPS]))
    return summary


def evaluate(nets,shots):
    pools,excluded=query_pools(shots);choices=[];groups=[]
    for group in GROUPS:
        for i,q in pools[group]:
            take=q[np.linspace(0,len(q)-1,min(4,len(q)),dtype=int)]
            choices.extend((i,int(end)) for end in take);groups.extend([group]*len(take))
    arms=dict(original_tokens=('original','tokens','true'),original_qk=('original','qk','true'),
        forecast_qk=('forecast','qk','true'),matched_qk=('matched','qk','true'),
        matched_reverse=('matched','qk','reverse'),matched_hold=('matched','qk','hold'))
    for net in nets.values():net.requires_grad_(False)
    rows={k:[] for k in arms};receipts=[]
    for lo in range(0,len(choices),12):
        e=assemble(shots,choices[lo:lo+12],'cuda');receipts.extend(e['receipts']);baseline=None
        for arm,(name,site,mode) in arms.items():
            result=paired_risk(nets[name],e,site=site,mode=mode)
            if baseline is None:baseline=result['before'].detach()
            elif not torch.allclose(result['before'],baseline,atol=2e-6,rtol=1e-6):raise ValueError('static risk changed across heads/sites')
            values={k:v.detach().cpu().numpy() for k,v in result.items() if k!='alignment'}
            for j,r in enumerate(e['receipts']):
                rows[arm].append(dict(shot=r['shot'],query=r['query'],group=groups[lo+j],**{k:float(v[j]) for k,v in values.items()}))
    summary={k:group_summary(v) for k,v in rows.items()};m=summary['matched_qk']
    screen=all(m[g]['mean_risk_change']<=0 for g in ('safe','danger')) and all(
        m['macro_change']<summary[k]['macro_change']-1e-6 for k in ('original_qk','forecast_qk','matched_reverse','matched_hold'))
    return dict(summary=summary,per_endpoint=rows,receipts=receipts,excluded=excluded,
        necessary_source_screen=bool(screen),not_shot_tpr_fpr=True,not_east_validation=True)


def run(output):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    roles=simple_data.source_roles()
    if ([len(roles[k]) for k in ('train','val','test')]!=[1353,160,220]
            or len(set(sum([roles[k] for k in ('train','val','test')],[])))!=1733):raise ValueError('canonical disjoint roles required')
    advanced=simple_data.advanced_times();nets={};original={};optimizers={}
    for name in ('original','forecast','matched'):
        base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=MatrixAttention(base)
        nets[name]=net;original[name]=snapshot(net)
        if name!='original':
            base.forecast_head.requires_grad_(True)
            optimizers[name]=torch.optim.AdamW(base.forecast_head.parameters(),lr=1e-4,weight_decay=0)
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='JTEXT_TRAIN_ONLY_MATCHED_QK_AUX_HEAD_NOT_JTEXT_ONLINE',source_sha256=PARENT_SHA256,
        train_ids=roles['train'],val_ids=roles['val'],test_access=False,east_access=False,
        steps=512,batch_size=12,seed=131,lr=1e-4,weight_decay=0,gradient_clip=1,inner_step=STEP,
        training_groups=GROUPS,threshold_unchanged=parent['threshold'],source_fast_site=base.cfg.fast_site,
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in files},
        plan_sha256=sha(io23.ROOT/'QK_MATCHED_HEAD_PLAN.md'),gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    output.mkdir();_durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):archive.write(p,str(p.relative_to(io23.ROOT)))
        archive.write(io23.ROOT/'QK_MATCHED_HEAD_PLAN.md','QK_MATCHED_HEAD_PLAN.md')
    started=time.monotonic();train=load_split(roles['train'],{i:advanced[i] for i in roles['train'] if i in advanced})
    pools,excluded=query_pools(train)
    if not all(pools.values()):raise ValueError('all three TRAIN strata required')
    _durable_json(output/'train_population.json',dict(excluded=excluded,groups={g:dict(shots=len(v),queries=sum(len(q) for _,q in v)) for g,v in pools.items()}))
    print(dict(train_loaded=len(train),groups={k:len(v) for k,v in pools.items()},seconds=time.monotonic()-started),flush=True)
    rng=np.random.default_rng(131);history=[];digest=hashlib.sha256();sampled=[]
    for step in range(512):
        choices=[]
        for group in GROUPS:
            for _ in range(4):
                i,q=pools[group][int(rng.integers(len(pools[group])))];choices.append((i,int(rng.choice(q))))
        e=assemble(train,choices,'cuda');sampled.extend(e['receipts'])
        digest.update(json.dumps(e['receipts'],sort_keys=True).encode())
        row=dict(step=step+1)
        for name,opt in optimizers.items():row[name]=train_step(nets[name],opt,e,name)
        history.append(row)
        if (step+1)%64==0:
            print(dict(**row,seconds=time.monotonic()-started),flush=True)
            _durable_json(output/'progress.json',dict(step=step+1,seconds=time.monotonic()-started))
    fit_seconds=time.monotonic()-started
    for name in optimizers:
        if any(not torch.equal(v,original[name][k]) for k,v in snapshot(nets[name]).items()):raise ValueError('classifier modified')
        torch.save({k:v.detach().cpu() for k,v in nets[name].base.forecast_head.state_dict().items()},output/f'{name}_head.pt')
    _durable_json(output/'training.json',dict(history=history,episode_sha256=digest.hexdigest(),receipts=sampled))
    del train,pools
    val=load_split(roles['val'],{i:advanced[i] for i in roles['val'] if i in advanced})
    result=evaluate(nets,val);_durable_json(output/'validation.json',result)
    for name in nets:
        if any(not torch.equal(v,original[name][k]) for k,v in snapshot(nets[name]).items()):raise ValueError('classifier modified during evaluation')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('original source checkpoint changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('implementation changed during run')
    _durable_json(output/'input_hashes.json',{str(i):sha(io23.path_for('J-TEXT',i)) for i in roles['train']+roles['val']})
    complete=dict(complete=True,seconds=time.monotonic()-started,fit_seconds=fit_seconds,
        necessary_source_screen=result['necessary_source_screen'],source_unchanged=True,target_or_test_signals_read=0,
        protocol_sha256=sha(output/'protocol.json'),validation_sha256=sha(output/'validation.json'),
        training_sha256=sha(output/'training.json'),input_hashes_sha256=sha(output/'input_hashes.json'),
        heads={k:sha(output/f'{k}_head.pt') for k in optimizers},scientific_goal_achieved=False)
    _durable_json(output/'complete.json',complete);print(result['summary'],flush=True);print(complete,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
