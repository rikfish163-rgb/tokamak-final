"""Matched encoder-momentum ablation; archived controls are read-only."""
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
from ..east_joint_feedback_20261023.learning import enable,objective
from ..asymmetric_feedback_20261023.learning import routed_episode as episode
from .learning import optimizer
from ..east_joint_feedback_20261023.run import ARMS,CURVES,evaluate,gate
from ..increment_forecast_20261023.learning import predict
from ..labeled_alignment_20261023.learning import balanced_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import pools,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..east_joint_source_replay_20261023.learning import replay_loss,validate_source

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/asymmetric_feedback_cv20_20261023'
PREVIOUS_SHA='7f4bda909ac80ad9c3346738737cb7ca47a4555338283af60e8d5f1effcbb6db'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/编码器动量对照_事前约定_20261023.md'


def train(warm,shots,entries,threshold,folder,fold,source):
    validate_source(source,io23.source_roles()['train'])
    models={a:copy.deepcopy(warm) for a in ARMS};before=state_cpu(warm)
    byid={s['shot']:s for s in shots};opts={};allowed={};params={}
    groups=pools(source);rng=np.random.default_rng(9131+fold)
    for a,m in models.items():
        allowed[a]=enable(m);named=dict(m.named_parameters())
        params[a]=[v for k,v in named.items() if k in allowed[a]]
        opts[a]=optimizer(m)
    boundary=math.log(threshold/(1-threshold));history=[];start=time.monotonic();batches=[]
    for item in entries:
        sx,sm,sy,sr=balanced_batch(source,groups,rng,warm.cfg)
        batches.append((sx.cpu(),sm.cpu(),sy.cpu()))
        row=dict(step=item['step'],source_examples=sr)
        with torch.no_grad():row['warm_source_bce']=float(replay_loss(warm,sx,sm,sy))
        for a,m in models.items():
            opts[a].zero_grad(set_to_none=True);receipts=[];arm_start=time.monotonic()
            for e in item['episodes']:
                s=byid[e['shot']];signal={k:s[k] for k in ('values','available')}
                r=episode(m,signal,e['count'],e['ends'],e['disruptive'],boundary=boundary,track=a=='meta',device='cuda')
                loss=objective(r,a)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite target objective')
                (loss/2).backward()
                receipts.append(dict(**e,risk_before=float(r['before'].detach()),risk_after=float(r['after'].detach()),
                    forecast=float(r['forecast'].detach()),objective=float(loss.detach()),all_counts=r['all_counts']))
                del r,loss
            sl=replay_loss(m,sx,sm,sy)
            if not torch.isfinite(sl):raise FloatingPointError('nonfinite source objective')
            sl.backward()
            assert all(p.grad is None for k,p in m.named_parameters() if k not in allowed[a])
            gradients={group:float(sum(p.grad.detach().double().square().sum() for k,p in m.named_parameters()
                if k in allowed[a] and p.grad is not None and (k.startswith('base.forecast_head.')==(group=='head'))).sqrt())
                for group in ('encoder','head')}
            norm=torch.nn.utils.clip_grad_norm_(params[a],1.,error_if_nonfinite=True);opts[a].step()
            row[a]=dict(episodes=receipts,source_bce=float(sl.detach()),
                objective=float(sl.detach())+sum(r['objective'] for r in receipts)/2,
                grad_norm=float(norm),group_grad_norms=gradients,seconds=time.monotonic()-arm_start)
        history.append(row)
        if (item['step']+1)%8==0:
            _durable_json(folder/'training.json',history)
            print(dict(stage='train',fold=fold,step=item['step']+1,seconds=time.monotonic()-start),flush=True)
    retention={}
    for a,m in models.items():
        m.set_stage('frozen');assert_unchanged(m,before,tuple(allowed[a]))
        torch.save(dict(model=state_cpu(m),config=m.cfg.to_dict()),folder/f'{a}.pt')
        with torch.no_grad():retention[a]=[float(replay_loss(m,*(t.cuda() for t in b))) for b in batches]
    _durable_json(folder/'source_training_retention.json',dict(warm=[r['warm_source_bce'] for r in history],**retention))
    return models,history


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(PREVIOUS/'completed.json')==PREVIOUS_SHA and READ(PREVIOUS/'verification.json')['status']=='VERIFIED'
    previous=READ(PREVIOUS/'completed.json');prior=READ(PREVIOUS/'protocol.json')
    for name,digest in previous['artifact_sha256'].items():assert sha(PREVIOUS/name)==digest,name
    assert sha(REFERENCE/'completed.json')==REF_SHA and sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    root.mkdir(parents=True);paths=set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_encoder_momentum.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):z.write(path,str(path.relative_to(io23.ROOT)))
    source_ids=io23.source_roles()['train']
    p=dict(previous=str(PREVIOUS),previous_sha256=PREVIOUS_SHA,reference_sha256=REF_SHA,source_sha256=PARENT_SHA256,
        folds=prior['folds'],threshold=prior['threshold'],steps=STEPS,source_replay_seed=9131,
        source_train_ids=source_ids,source_replay_weight=1.,source_batch_size=32,matched_flops=False,
        direct_outer_forecast_gradient_to_encoder=False,inner_update_unchanged=True,
        encoder_betas=[0.,.999],head_betas=[.9,.999],
        encoder_lr=1e-5,head_lr=1e-4,code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in paths},
        plan_sha256=sha(PLAN),test360_accessed=False,independent_confirmation=False,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p)
    original_outcome,original_signal=io23.read_outcome,simple_data.read_shot;source_access=[]
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        source_access.append(sid);return original_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,ids=load_source_train()
    assert ids==source_access==source_ids;_durable_json(root/'source_label_access.json',source_access)
    access=[];seals=[];counts=dict(offline_forward=0,online=0,source_windows=0)
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';folder.mkdir();(folder/'shots').mkdir();phase='fit'
        def signal_guard(device,sid,**kwargs):
            assert device=='EAST' and sid in plan['fit']+plan['held']
            access.append(dict(fold=f,kind='signal',shot=sid,phase=phase));return original_signal(device,sid,**kwargs)
        def outcome_guard(device,sid):
            assert device=='EAST' and sid in plan['fit'] and phase=='fit'
            access.append(dict(fold=f,kind='outcome',shot=sid,phase=phase));return original_outcome(device,sid)
        warm=load_model(f);original=state_cpu(warm);mapping=READ(REFERENCE/f'fold{f}'/'input_map.json')
        _durable_json(folder/'input_map.json',mapping)
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries,eligible=schedule(shots,plan['fit'],f)
            assert entries==READ(PREVIOUS/f'fold{f}'/'schedule.json')
            _durable_json(folder/'schedule.json',entries);_durable_json(folder/'eligibility.json',eligible)
            models,history=train(warm,shots,entries,p['threshold'],folder,f,source)
            counts['offline_forward']+=sum(len(e['all_counts']) for r in history for a in ARMS for e in r[a]['episodes'])
            counts['source_windows']+=sum(len(r['source_examples'])*len(ARMS) for r in history)
            frozen={a:state_cpu(m) for a,m in models.items()};phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping);updates={}
                with np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as old:
                    curves=dict(warm_static=old['static'].copy(),warm_intra=old['increment'].copy())
                for a,m in models.items():
                    scores,updates[a],packets=predict(m,signal);curves.update({f'{a}_{k}':v for k,v in scores.items()})
                    counts['online']+=sum(r['gradient_steps'] for r in updates[a])
                    np.savez_compressed(folder/'shots'/f'{sid}_{a}_forecasts.npz',**packets)
                path=folder/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**curves)
                _durable_json(folder/'shots'/f'{sid}_updates.json',updates);seals.append(dict(fold=f,shot=sid,sha256=sha(path)))
            for a,m in models.items():assert_unchanged(m,frozen[a])
            assert_unchanged(warm,original)
        _durable_json(root/'access.json',access)
        print(dict(stage='fold_complete',fold=f,shots=len(seals),seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'seals.json',seals);summary=evaluate(root,p['folds'],p['threshold'])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=summary,passes_screen=gate(summary),counts=counts,
        previous_metrics=previous['metrics'],seconds=time.monotonic()-start,source_unchanged=True,test360_accessed=False,
        peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k not in ('artifact_sha256','previous_metrics')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)

