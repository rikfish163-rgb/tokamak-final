"""Two paired source-only imputation objectives; qualified classifier frozen."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import validate_roles
from .model import Imputer,tokens,masked,latent_loss,risk_kl,readout

CACHE=io23.ROOT/'experiments/risk_aligned_online/runs/source_task_memory_v2_20261026'


def source_windows(ids,inventory,rng):
    xs=[];ms=[];flags=[];records=[]
    for sid in ids:
        r=inventory[sid]
        if sha(io23.path_for('J-TEXT',sid))!=r['data_sha256']:
            raise ValueError('source raw input changed')
        raw=apply_input_contract(simple_data.read_shot('J-TEXT',sid))
        ends=np.asarray(r['ends'])
        chosen=ends[np.linspace(0,len(ends)-1,4,dtype=int)] if rng is None else rng.choice(ends,4,replace=len(ends)<4)
        for e in chosen:
            x,m=simple_data.window(raw,int(e),128);xs.append(x);ms.append(m);flags.append(r['disruptive'])
        records.append(dict(shot=sid,ends=chosen.tolist()))
    return torch.tensor(np.stack(xs)),torch.tensor(np.stack(ms)),np.asarray(flags),records


@torch.no_grad()
def evaluate(base,nets,data):
    x,a,flags,_=data;rows={k:[] for k in ('identity',*nets)}
    for lo in range(0,len(x),32):
        xx=x[lo:lo+32].cuda();aa=a[lo:lo+32].cuda()
        h,v=tokens(base,xx,aa,base.center.new_zeros(128));z=readout(base,h,v,None)
        mx,ma=masked(xx,aa);mh,mv=tokens(base,mx,ma,base.center.new_zeros(128))
        target=h.repeat_interleave(2,0);valid=v.repeat_interleave(2,0);teacher=z.repeat_interleave(2)
        predictions={'identity':mh.masked_fill(~mv[...,None],0.),**{k:m(mh,mv) for k,m in nets.items()}}
        for k,p in predictions.items():
            mse=latent_loss(p,target,valid).reshape(-1,2).mean(-1)
            kl=risk_kl(readout(base,p,valid,None),teacher).reshape(-1,2).mean(-1)
            rows[k].extend(zip(mse.cpu().tolist(),kl.cpu().tolist()))
    return {k:{g:dict(mse=float(np.asarray(v)[mask,0].mean()),kl=float(np.asarray(v)[mask,1].mean()))
                 for g,mask in (('all',np.ones(len(flags),bool)),('D',flags),('S',~flags))}
            for k,v in rows.items()}


def run(root):
    if root.exists() or root.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();original=classifier_state(base)
    roles=simple_data.source_roles();validate_roles(roles)
    inventory={r['shot']:r for r in read(CACHE,'inventory.json')}
    for role in ('train','val'):
        if {s for s,r in inventory.items() if r['role']==role}!=set(roles[role]):raise ValueError('source cache role changed')
    net=Imputer().cuda();nets={'plain':net,'task':deepcopy(net)}
    opts={k:torch.optim.Adam(m.parameters(),lr=.001) for k,m in nets.items()}
    root.mkdir(parents=True);start=time.monotonic()
    files=[Path(__file__),Path(__file__).with_name('model.py')]
    protocol=dict(scope='SOURCE_ONLY_FIXED_CLASSIFIER_COMPLETION',source_sha256=PARENT_SHA256,
        threshold=float(parent['threshold']),train_ids=roles['train'],val_ids=roles['val'],
        test_ids_not_read=roles['test'],epochs=2,seed=131,source_windows_per_shot=4,batch_shots=8,
        source_learning_rate=.001,task_kl_coefficient=1.,selected_epoch='FIXED_FINAL_2',
        cache_inventory_sha256=sha(CACHE/'inventory.json'),plan_sha256=sha(io23.ROOT/'LATENT_IMPUTATION_PLAN.md'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(root/'protocol.json',protocol)
    validation=source_windows(roles['val'],inventory,None)
    _durable_json(root/'val_windows.json',validation[3])
    rng=np.random.default_rng(131);history=[]
    counts={f:sum(inventory[s]['disruptive']==f for s in roles['train']) for f in (False,True)}
    for epoch in (1,2):
        ids=rng.permutation(roles['train']).tolist();data=source_windows(ids,inventory,rng)
        _durable_json(root/f'windows_epoch{epoch}.json',data[3]);x,a,flags,_=data
        for lo in range(0,len(x),32):
            xx=x[lo:lo+32].cuda();aa=a[lo:lo+32].cuda()
            with torch.no_grad():
                h,v=tokens(base,xx,aa,base.center.new_zeros(128));z=readout(base,h,v,None)
                mx,ma=masked(xx,aa);mh,mv=tokens(base,mx,ma,base.center.new_zeros(128))
            target=h.repeat_interleave(2,0);valid=v.repeat_interleave(2,0);teacher=z.repeat_interleave(2)
            w=torch.tensor([.5*len(ids)/counts[f] for f in flags[lo:lo+32]],device='cuda').repeat_interleave(2)
            record=dict(epoch=epoch,shots=min((lo+32)//4,len(ids)),arms={})
            for name,m in nets.items():
                opts[name].zero_grad(set_to_none=True);p=m(mh,mv)
                mse=latent_loss(p,target,valid);kl=risk_kl(readout(base,p,valid,None),teacher)
                loss=((mse+kl if name=='task' else mse)*w).mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite source training')
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opts[name].step()
                record['arms'][name]=dict(loss=float(loss.detach()),mse=float(mse.mean().detach()),kl=float(kl.mean().detach()),gradient_norm=float(gn))
            history.append(record)
            if len(history)%25==0:print(dict(epoch=epoch,shots=record['shots'],seconds=time.monotonic()-start,**record['arms']),flush=True)
        result=evaluate(base,nets,validation);_durable_json(root/f'val_epoch{epoch}.json',result)
        print(dict(epoch=epoch,validation=result,seconds=time.monotonic()-start),flush=True)
    checkpoints={}
    for name,m in nets.items():
        path=root/f'{name}.pt';torch.save(dict(model={k:v.cpu() for k,v in m.state_dict().items()},protocol_sha256=sha(root/'protocol.json')),path)
        checkpoints[name]=sha(path)
    assert_classifier_unchanged(base,original)
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('qualified source changed')
    _durable_json(root/'training.json',history)
    _durable_json(root/'complete.json',dict(complete=True,source_unchanged=True,checkpoints=checkpoints,
        val=result,seconds=time.monotonic()-start,east_signals_read=0,test220_signals_read=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
