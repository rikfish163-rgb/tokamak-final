"""Source-only temporal relation head; no source-classifier or EAST update."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..context_memory_20261028.checkpoint import read
from ..latent_completion_20261028.train import CACHE,source_windows
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256,classifier_state,assert_classifier_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import validate_roles
from .model import OrderJudge,permutation,margin,mean_view


def summary(records):
    by_shot={}
    for r in records:
        if r['eligible']:by_shot.setdefault(r['shot'],[]).append(r)
    result={}
    for group in ('all','D','S'):
        rows=[rs for rs in by_shot.values() if group=='all' or rs[0]['flag']==(group=='D')]
        if not rows:raise ValueError('both source validation classes need temporal pairs')
        result[group]=dict(shots=len(rows),windows=sum(map(len,rows)),
            accuracy=float(np.mean([np.mean([float(r['margin']>0)+.5*float(r['margin']==0) for r in rs]) for rs in rows])),
            roughness_accuracy=float(np.mean([np.mean([float(r['roughness_margin']>0)+.5*float(r['roughness_margin']==0) for r in rs]) for rs in rows])),
            mean_margin=float(np.mean([np.mean([r['margin'] for r in rs]) for rs in rows])))
    return result


@torch.no_grad()
def evaluate(base,net,data):
    x,a,flags,windows=data;shots=np.repeat([r['shot'] for r in windows],4)
    output={};maximum_mean_margin=0.
    for mode in ('random','interleave'):
        rng=np.random.default_rng(2718);records=[]
        for lo in range(0,len(x),32):
            xx,aa=x[lo:lo+32].cuda(),a[lo:lo+32].cuda();perm=permutation(len(xx),rng,mode)
            delta,eligible,tv=margin(base,net,xx,aa,perm,base.center.new_zeros(128))
            control,_,_=margin(base,net,mean_view(xx,aa),aa,perm,base.center.new_zeros(128))
            maximum_mean_margin=max(maximum_mean_margin,float(control.abs().max()))
            for j,(d,e,t) in enumerate(zip(delta.cpu().tolist(),eligible.cpu().tolist(),tv.cpu().tolist())):
                records.append(dict(shot=int(shots[lo+j]),flag=bool(flags[lo+j]),eligible=e,margin=d,roughness_margin=t))
        output[mode]=dict(groups=summary(records),records=records)
    output['maximum_mean_only_margin']=maximum_mean_margin
    return output


def run(root):
    if root.exists() or root.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();initial=classifier_state(base)
    roles=simple_data.source_roles();validate_roles(roles)
    inventory={r['shot']:r for r in read(CACHE,'inventory.json')}
    for role in ('train','val'):
        if {s for s,r in inventory.items() if r['role']==role}!=set(roles[role]):raise ValueError('source roles changed')
    net=OrderJudge().cuda();opt=torch.optim.Adam(net.parameters(),lr=.001)
    root.mkdir();start=time.monotonic()
    files=[Path(__file__),Path(__file__).with_name('model.py')]
    protocol=dict(scope='SOURCE_ONLY_TEMPORAL_RELATION_AUXILIARY_NOT_NEW_CLASSIFIER',source_sha256=PARENT_SHA256,
        train_ids=roles['train'],val_ids=roles['val'],test_ids_not_read=roles['test'],epochs=6,seed=131,
        batch_windows=32,windows_per_shot=4,learning_rate=.001,selected_epoch='FIXED_FINAL_6',
        plan_sha256=sha(io23.ROOT/'TEMPORAL_RELATION_PLAN.md'),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        parent_threshold=float(parent['threshold']),parameters=sum(p.numel() for p in net.parameters()),
        raw_window_loader_sha256=sha(Path(source_windows.__code__.co_filename)),cache_inventory_sha256=sha(CACHE/'inventory.json'),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(root/'protocol.json',protocol)
    training=source_windows(roles['train'],inventory,np.random.default_rng(131));validation=source_windows(roles['val'],inventory,None)
    _durable_json(root/'train_windows.json',training[3]);_durable_json(root/'val_windows.json',validation[3])
    rng=np.random.default_rng(131);x,a,_,_=training;history=[]
    for epoch in range(1,7):
        order=rng.permutation(len(x));eligible_count=0;loss_sum=0.;steps=0
        for lo in range(0,len(order),32):
            ids=order[lo:lo+32];xx,aa=x[ids].cuda(),a[ids].cuda();perm=permutation(len(ids),rng)
            delta,eligible,_=margin(base,net,xx,aa,perm,base.center.new_zeros(128))
            if not eligible.any():continue  # Explicitly no temporal comparison; count coverage below.
            opt.zero_grad(set_to_none=True);loss=F.softplus(-delta[eligible]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite relation learning')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            eligible_count+=int(eligible.sum());loss_sum+=float(loss.detach())*int(eligible.sum());steps+=1
        if not eligible_count:raise ValueError('no source temporal comparisons')
        validation_result=evaluate(base,net,validation)
        _durable_json(root/f'val_epoch{epoch}.json',validation_result)
        record=dict(epoch=epoch,steps=steps,eligible_windows=eligible_count,loss=loss_sum/eligible_count,
            validation={m:validation_result[m]['groups'] for m in ('random','interleave')},seconds=time.monotonic()-start)
        history.append(record);print(record,flush=True)
    path=root/'judge.pt';torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},protocol_sha256=sha(root/'protocol.json')),path)
    assert_classifier_unchanged(base,initial)
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('source classifier changed')
    _durable_json(root/'training.json',history)
    passes=all(validation_result[m]['groups']['all']['accuracy']>.60 and
        all(validation_result[m]['groups'][g]['accuracy']>.55 for g in ('D','S')) for m in ('random','interleave'))
    passes=bool(passes and validation_result['maximum_mean_only_margin']<1e-6)
    _durable_json(root/'complete.json',dict(complete=True,source_unchanged=True,source_only_gate=passes,
        checkpoint_sha256=sha(path),val={m:validation_result[m]['groups'] for m in ('random','interleave')},
        maximum_mean_only_margin=validation_result['maximum_mean_only_margin'],seconds=time.monotonic()-start,
        east_signals_read=0,test220_signals_read=0,no_online_performance_claim=True))
    print(dict(complete=True,source_only_gate=passes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
