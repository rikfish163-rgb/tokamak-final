"""Train two finite-step loss policies on20 past shots, then seal seven arms."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..feedback_cone_feasibility_20261024.learning import solve_weights
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import predict_many
from ..signal_driven_span_20261024 import run as evaluator
from .learning import ARMS,feedback_context,train_policy,predict,eligible

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/clock_channel_loss_cal10_20261025'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/有限步条件损失学习_事前约定_20261025.md'


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    done=READ(PREVIOUS/'completed.json');assert READ(PREVIOUS/'audit.json')['status']=='passed'
    for name,digest in done['artifact_sha256'].items():assert sha(PREVIOUS/name)==digest,name
    old=READ(PREVIOUS/'protocol.json');history=old['preparation_ids'];ids=old['shot_ids']
    assert not set(history)&set(ids) and not set(history+ids)&set(old['excluded360_ids'])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');start=time.monotonic()
    root.mkdir(parents=True)
    for sub in ('shots','updates','training'): (root/sub).mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_finite_conditional_loss.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(shot_ids=ids,preparation_ids=history,excluded360_ids=old['excluded360_ids'],threshold=old['threshold'],
        arms=('off',)+ARMS,parent_sha256=old['parent_sha256'],source_sha256=PARENT_SHA256,
        mapping_sha256=sha(REFERENCE/'input_map.json'),previous_completed_sha256=sha(PREVIOUS/'completed.json'),
        plan_sha256=sha(PLAN),epochs=10,optimizer='Adam',outer_lr=.05,step_bound=.01,
        global_trainable_parameters=24,context_trainable_parameters=47,update_every=128,first_update=256,
        parameter_reset='parent every update',inter_steps=0,alarm_updates=0,
        labels='archived20 historical outcomes for meta-training; seal all predictions before new outcome reads',
        scope='DEVELOPMENT_ONLY_NOT_COMPLETE_CONTINUAL_SYSTEM',independent_confirmation=False,
        event_labels_added=False,code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256=old['input_sha256'],torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('no unsealed/new outcomes in training or prediction')
    cache=[];shots={};all_rows=[];bytes_used=0;cache_receipts=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in history:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            oldg=READ(PREVIOUS/'geometry'/f'{sid}.json');rows=oldg['rows'];all_rows.extend(rows)
            shots[sid]=dict(signal=s,times=raw['times'],flag=oldg['is_disrupt'],down=oldg['down'])
            for count in sorted({r['count'] for r in rows}):
                fast=initial_encoder(model);parts,receipt=signal_parts(model,s,count,fast,hold=False,device='cuda')
                if parts is None:raise AssertionError('archived rows require valid targets')
                scale=float(parts.double().norm(dim=1).max());c,mask=feedback_context(model,s,count,fast,hold=False,device='cuda')
                selected=[r for r in rows if r['count']==count]
                assert all(r['scale']==scale and r['observations']==receipt['observed'] for r in selected)
                cpu=parts.cpu();bytes_used+=cpu.numel()*cpu.element_size()
                if bytes_used>5*1024**3:raise MemoryError('declared historical cache exceeds5GiB')
                cache.append(dict(shot=sid,count=count,parts=cpu,scale=scale,context=c.cpu(),observed=mask.cpu(),rows=selected))
                cache_receipts.append(dict(shot=sid,count=count,scale=scale,context=c.cpu().tolist(),observed=mask.cpu().tolist(),
                                           channel_gram=(parts.double()@parts.double().T).cpu().tolist()))
            print(dict(stage='cache',shot=sid,blocks=len(cache),GiB=round(bytes_used/1024**3,3),seconds=round(time.monotonic()-start,2)),flush=True)
        _durable_json(root/'cache_receipts.json',cache_receipts)
        keys=sorted({(r['shot'],r['group']) for r in all_rows})
        a=np.array([np.mean([r['a'] for r in all_rows if (r['shot'],r['group'])==key],axis=0) for key in keys])
        b=np.array([np.mean([r['b'] for r in all_rows if (r['shot'],r['group'])==key],axis=0) for key in keys])
        w,fit=solve_weights(a,b)
        _durable_json(root/'per_shot_first_order_feasibility.json',dict(keys=keys,weights=w.tolist(),fit=fit,
                        scope='restricted nonnegative channel-gradient cone, not all neural policies'))
        records={}
        def progress(kind,record):
            _durable_json(root/'training'/f'{kind}_{record["epoch"]:02d}.json',record)
            print(dict(stage='training',kind=kind,epoch=record['epoch'],loss=record['objective'],
                       strength=record['strength'],seconds=round(time.monotonic()-start,2)),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('declared30minute run limit exceeded')
        for kind in ('global','context'):
            best,logs=train_policy(model,kind,cache,shots,progress=progress)
            records[kind]=best
            _durable_json(root/f'{kind}_selection.json',dict(epoch=min(logs,key=lambda r:r['objective'])['epoch'],
                            objective=min(r['objective'] for r in logs),selection='historical training only'))
        _durable_json(root/'policies.json',records)
        _durable_json(root/'policy_seal.json',dict(sha256=sha(root/'policies.json'),new_outcomes_accessed=0))
        del cache,shots
        linear=READ(PREVIOUS/'policy.json')['block'];seals=[]
        for sid in ids:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s,records,linear)
            with torch.no_grad():baseline=predict_many(model,dict(parent=s),device='cuda')['parent']
            np.testing.assert_array_equal(curves['off'],baseline)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:np.testing.assert_array_equal(curves['linear'],z['block'])
            for arm in ARMS:np.testing.assert_array_equal(curves[arm][:256],baseline[:256])
            out=root/'shots'/f'{sid}.npz';np.savez_compressed(out,times=raw['times'],**curves)
            up=root/'updates'/f'{sid}.json';_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(out),updates_sha256=sha(up)))
            print(dict(stage='prediction',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('declared30minute run limit exceeded')
    _durable_json(root/'prediction_seals.json',seals)
    with patch.object(evaluator,'ARMS',ARMS):results=evaluator.evaluate(root,p)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',seconds=time.monotonic()-start,cache_bytes=bytes_used,results=results,
        eligible_for360=eligible(results),source_and_parent_unchanged=True,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
