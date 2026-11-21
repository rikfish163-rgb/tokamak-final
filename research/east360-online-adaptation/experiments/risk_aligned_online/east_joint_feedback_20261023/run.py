"""Small labeled-EAST preparation, then strictly label-free intra-shot replay."""
import argparse
import copy
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE,REF_SHA,STEPS,load_model,schedule
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import enable,episode,objective

ARMS=('joint','meta')
CURVES=('warm_static','warm_intra','joint_static','joint_intra','meta_static','meta_intra')
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/EAST表示与反馈联合准备_事前约定_20261023.md'


def train(warm,shots,entries,threshold,folder,fold):
    models={a:copy.deepcopy(warm) for a in ARMS};before=state_cpu(warm)
    byid={s['shot']:s for s in shots};opts={};allowed={};params={}
    for a,m in models.items():
        allowed[a]=enable(m);named=dict(m.named_parameters())
        params[a]=[v for k,v in named.items() if k in allowed[a]]
        opts[a]=torch.optim.Adam([
            dict(params=[v for k,v in named.items() if k in m.encoder_names()],lr=1e-5),
            dict(params=list(m.base.forecast_head.parameters()),lr=1e-4)])
    boundary=math.log(threshold/(1-threshold));history=[];start=time.monotonic()
    for item in entries:
        row=dict(step=item['step'])
        for a,m in models.items():
            opts[a].zero_grad(set_to_none=True);receipts=[];arm_start=time.monotonic()
            for e in item['episodes']:
                s=byid[e['shot']];signal={k:s[k] for k in ('values','available')}
                r=episode(m,signal,e['count'],e['ends'],e['disruptive'],boundary=boundary,track=a=='meta',device='cuda')
                loss=objective(r,a)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint preparation')
                (loss/2).backward()
                receipts.append(dict(**e,risk_before=float(r['before'].detach()),risk_after=float(r['after'].detach()),
                    forecast=float(r['forecast'].detach()),objective=float(loss.detach()),all_counts=r['all_counts']))
                del r,loss
            assert all(p.grad is None for k,p in m.named_parameters() if k not in allowed[a])
            gradients={group:float(sum(p.grad.detach().double().square().sum() for k,p in m.named_parameters()
                if k in allowed[a] and p.grad is not None and (k.startswith('base.forecast_head.')==(group=='head'))).sqrt())
                for group in ('encoder','head')}
            norm=torch.nn.utils.clip_grad_norm_(params[a],1.,error_if_nonfinite=True)
            opts[a].step()
            row[a]=dict(episodes=receipts,grad_norm=float(norm),group_grad_norms=gradients,
                        seconds=time.monotonic()-arm_start)
        history.append(row)
        if (item['step']+1)%8==0:
            _durable_json(folder/'training.json',history)
            print(dict(stage='train',fold=fold,step=item['step']+1,seconds=time.monotonic()-start,
                       peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20),flush=True)
    for a,m in models.items():
        m.set_stage('frozen');assert_unchanged(m,before,tuple(allowed[a]))
        torch.save(dict(model=state_cpu(m),config=m.cfg.to_dict()),folder/f'{a}.pt')
    return models,history


def gate(summary):
    a,b,c,w=(summary[k] for k in ('meta_intra','meta_static','joint_intra','warm_static'))
    return bool(a['J']>b['J'] and a['J']>w['J'] and a['J']>c['J'] and a['tp']>=w['tp'] and a['fpr']<=.2
                and a['J']-b['J']>c['J']-summary['joint_static']['J'])


def evaluate(root,plans,threshold):
    rows={a:[] for a in CURVES};access=[]
    for plan in plans:
        for sid in plan['held']:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            access.append(dict(shot=sid,phase='after_all_seals'));v=s['valid']
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(s['times'],z['times'])
                for a in CURVES:rows[a].append(dict(shot=sid,times=s['times'][v],scores=z[a][v],is_disrupt=s['is_disrupt'],down=s['down']))
    result={a:metrics(v,threshold,lead=30.) for a,v in rows.items()}
    _durable_json(root/'per_shot.json',result);_durable_json(root/'evaluation_access.json',access)
    return {a:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in result.items()}


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env local CUDA required')
    assert sha(REFERENCE/'completed.json')==REF_SHA and READ(REFERENCE/'verification.json')['status']=='VERIFIED'
    reference=READ(REFERENCE/'completed.json');p0=READ(REFERENCE/'protocol.json')
    for name,digest in reference['artifact_sha256'].items():assert sha(REFERENCE/name)==digest,name
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    root.mkdir(parents=True)
    paths=set(Path(__file__).parent.glob('*.py'));paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_east_joint_feedback.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):z.write(path,str(path.relative_to(io23.ROOT)))
    p=dict(reference=str(REFERENCE),reference_sha256=REF_SHA,source_sha256=PARENT_SHA256,
        folds=p0['folds'],threshold=p0['threshold'],steps=STEPS,seed=131,sampling_seed=6131,
        encoder_lr=1e-5,head_lr=1e-4,forecast_weight=.2,meta_risk_weight=.5,inner_lr=.001,
        derivative='full observed-prefix graph; exact initialization connection',
        code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in paths},plan_sha256=sha(PLAN),
        test360_accessed=False,independent_confirmation=False,matched_flops=False,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p);access=[];seals=[];counts=dict(offline_forward=0,online=0)
    original_signal,original_outcome=simple_data.read_shot,io23.read_outcome
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';folder.mkdir();(folder/'shots').mkdir();phase='fit'
        def signal_guard(device,sid,**kwargs):
            assert device=='EAST' and sid in plan['fit']+plan['held']
            access.append(dict(fold=f,kind='signal',shot=sid,phase=phase));return original_signal(device,sid,**kwargs)
        def outcome_guard(device,sid):
            assert device=='EAST' and sid in plan['fit'] and phase=='fit'
            access.append(dict(fold=f,kind='outcome',shot=sid,phase=phase));return original_outcome(device,sid)
        warm=load_model(f);mapping=READ(REFERENCE/f'fold{f}'/'input_map.json');_durable_json(folder/'input_map.json',mapping)
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries,eligible=schedule(shots,plan['fit'],f)
            _durable_json(folder/'schedule.json',entries);_durable_json(folder/'eligibility.json',eligible)
            models,history=train(warm,shots,entries,p['threshold'],folder,f)
            counts['offline_forward']+=sum(len(e['all_counts']) for r in history for a in ARMS for e in r[a]['episodes'])
            frozen={a:state_cpu(m) for a,m in models.items()};phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping);updates={}
                with np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as old:
                    curves=dict(warm_static=old['static'].copy(),warm_intra=old['increment'].copy())
                for a,m in models.items():
                    scores,updates[a],packets=predict(m,signal)
                    curves.update({f'{a}_{key}':value for key,value in scores.items()})
                    counts['online']+=sum(r['gradient_steps'] for r in updates[a])
                    np.savez_compressed(folder/'shots'/f'{sid}_{a}_forecasts.npz',**packets)
                path=folder/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**curves)
                _durable_json(folder/'shots'/f'{sid}_updates.json',updates)
                seals.append(dict(fold=f,shot=sid,sha256=sha(path)))
            for a,m in models.items():assert_unchanged(m,frozen[a])
        _durable_json(root/'access.json',access)
        print(dict(stage='fold_complete',fold=f,shots=len(seals),seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'seals.json',seals);summary=evaluate(root,p['folds'],p['threshold'])
    assert summary['warm_static']==reference['metrics']['static'] and summary['warm_intra']==reference['metrics']['increment']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=summary,passes_screen=gate(summary),counts=counts,
        seconds=time.monotonic()-start,source_unchanged=True,test360_accessed=False,
        peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
