"""Same15-shot preparation, replace only safe pre-adaptation target locations."""
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
from ..asymmetric_feedback_20261023.learning import routed_episode
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE,load_model,schedule
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..east_joint_feedback_20261023.learning import enable,objective
from ..east_joint_source_replay_20261023.learning import replay_loss,validate_source
from ..external65_20261023.analysis import paired_changes
from ..increment_forecast_20261023.learning import predict
from ..labeled_alignment_20261023.learning import balanced_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import pools,sha,metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import prefix_risk,paired_objective

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/asymmetric_feedback_cv20_20261023'
PREVIOUS_SHA='7f4bda909ac80ad9c3346738737cb7ca47a4555338283af60e8d5f1effcbb6db'
EXTERNAL=io23.ROOT/'experiments/risk_aligned_online/runs/external65_frozen_20261023'
EXTERNAL_SHA='1dbc3b674371c87fd65e12617f55d6866338dc8b1d054e3ff2880fefbc9f93c3'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/初始窗口监督覆盖_配对约定_20261023.md'
ARMS=('old_static','old_intra','prefix_static','prefix_intra')


def train(warm,shots,entries,threshold,folder,fold,source,*,arm='prefix'):
    if arm not in ('prefix','baseline'):raise ValueError('declared training arm required')
    validate_source(source,io23.source_roles()['train']);model=copy.deepcopy(warm);frozen=state_cpu(warm)
    allowed=enable(model);named=dict(model.named_parameters());params=[p for k,p in named.items() if k in allowed]
    opt=torch.optim.Adam([dict(params=[v for k,v in named.items() if k in model.encoder_names()],lr=1e-5),
                          dict(params=list(model.base.forecast_head.parameters()),lr=1e-4)])
    groups=pools(source);rng=np.random.default_rng(9131+fold);byid={s['shot']:s for s in shots}
    boundary=math.log(threshold/(1-threshold));history=[];start=time.monotonic()
    for item in entries:
        sx,sm,sy,sr=balanced_batch(source,groups,rng,warm.cfg)
        opt.zero_grad(set_to_none=True);receipts=[]
        for e in item['episodes']:
            s=byid[e['shot']];signal={k:s[k] for k in ('values','available')}
            r=routed_episode(model,signal,e['count'],e['ends'],e['disruptive'],boundary=boundary,track=True,device='cuda')
            early=None;prefix=[]
            if arm=='prefix' and not e['disruptive']:
                assert not s['is_disrupt'] and np.all(s['y'][:256]==0) and s['valid'][:256].all()
                early,prefix=prefix_risk(model,signal,len(e['ends']),boundary=boundary,device='cuda')
            loss=paired_objective(r,early,disruptive=e['disruptive']) if arm=='prefix' else objective(r,'meta')
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite target objective')
            (loss/2).backward()
            receipts.append(dict(**e,prefix_ends=prefix,risk_before=float(r['before'].detach()),
                risk_after=float(r['after'].detach()),prefix_risk=None if early is None else float(early.detach()),
                forecast=float(r['forecast'].detach()),objective=float(loss.detach()),all_counts=r['all_counts']))
            del r,loss,early
        sl=replay_loss(model,sx,sm,sy)
        if not torch.isfinite(sl):raise FloatingPointError('nonfinite source loss')
        sl.backward()
        assert all(p.grad is None for k,p in model.named_parameters() if k not in allowed)
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        history.append(dict(step=item['step'],episodes=receipts,source_examples=sr,source_bce=float(sl.detach()),
            objective=float(sl.detach())+sum(r['objective'] for r in receipts)/2,grad_norm=float(norm)))
        if (item['step']+1)%8==0:
            _durable_json(folder/'training.json',history)
            print(dict(stage='train',fold=fold,arm=arm,step=item['step']+1,seconds=round(time.monotonic()-start,2)),flush=True)
    model.set_stage('frozen');assert_unchanged(model,frozen,tuple(allowed))
    torch.save(dict(model=state_cpu(model),config=model.cfg.to_dict()),folder/f'{arm}.pt')
    return model,history


def evaluate(root,p):
    summaries={};changes={};early={}
    for fplan in p['folds']:
        f=fplan['fold'];folder=root/f'fold{f}';summaries[str(f)]={};changes[str(f)]={};early[str(f)]={}
        for group,ids in (('held',fplan['held']),('external',p['external_ids'])):
            rows={a:[] for a in ARMS};timing={a:dict(padded=[],full_before_update=[],after_update=[]) for a in ARMS}
            for sid in ids:
                d,down=io23.read_outcome('EAST',sid)
                with np.load(folder/'shots'/f'{sid}.npz') as z:
                    valid=io23.labels_at(z['times'],d,down,125.)[1]
                    for a in ARMS:
                        rows[a].append(dict(shot=sid,times=z['times'][valid],scores=z[a][valid],is_disrupt=d,down=down))
                        if not d:
                            crossed=np.flatnonzero(valid&(z[a]>=p['threshold']))
                            if len(crossed):
                                idx=int(crossed[0]);key='padded' if idx<127 else 'full_before_update' if idx<256 else 'after_update'
                                timing[a][key].append(dict(shot=sid,index=idx,time=float(z['times'][idx])))
            results={a:metrics(rs,p['threshold'],lead=30.) for a,rs in rows.items()}
            _durable_json(folder/f'{group}_per_shot.json',results)
            summaries[str(f)][group]={a:dict(**public(r),very_early_over3000=sum(
                x['timely'] and x['lead_ms']>3000 for x in r['predictions'])) for a,r in results.items()}
            changes[str(f)][group]={f'{b}->{a}':paired_changes(results[b]['predictions'],results[a]['predictions'])
                for b,a in (('old_static','prefix_static'),('old_intra','prefix_intra'),('prefix_static','prefix_intra'))}
            early[str(f)][group]=timing
    _durable_json(root/'first_false_alarms.json',early)
    return summaries,changes


def main(root):
    if root.exists():raise FileExistsError(root)
    for ref,h in ((PREVIOUS,PREVIOUS_SHA),(EXTERNAL,EXTERNAL_SHA)):
        assert sha(ref/'completed.json')==h
        for name,digest in READ(ref/'completed.json')['artifact_sha256'].items():assert sha(ref/name)==digest
    prior=READ(PREVIOUS/'protocol.json');ep=READ(EXTERNAL/'protocol.json')
    assert prior['folds']==ep['folds'] and sha(DEFAULT_PARENT)==PARENT_SHA256
    for path,h in ep['checkpoints'].items():assert sha(path)==h
    ids=sorted({sid for plan in prior['folds'] for sid in plan['fit']+plan['held']}|set(ep['shot_ids']))
    assert len(ids)==85 and not set(ids)&set(io23.cohort_ids('EAST'))
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic();root.mkdir(parents=True)
    paths=set(Path(__file__).parent.glob('*.py'));paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_prefix_coverage.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for pp in sorted(paths):z.write(pp,str(pp.relative_to(io23.ROOT)))
    p=dict(folds=prior['folds'],threshold=prior['threshold'],external_ids=ep['shot_ids'],
        previous_sha256=PREVIOUS_SHA,external_sha256=EXTERNAL_SHA,plan_sha256=sha(PLAN),
        code_sha256={str(pp.relative_to(io23.ROOT)):sha(pp) for pp in paths},checkpoints=ep['checkpoints'],
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in ids},
        source_sha256=PARENT_SHA256,source_train_ids=io23.source_roles()['train'],
        test360_accessed=False,independent_confirmation=False,inter_steps=0,steps=64,matched_flops=False)
    _durable_json(root/'protocol.json',p);original_outcome=io23.read_outcome;original_signal=simple_data.read_shot;access=[]
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in p['source_train_ids']
        return original_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    counts=dict(offline_steps=0,intra_steps=0)
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';folder.mkdir();(folder/'shots').mkdir();phase='fit'
        def signal_guard(device,sid,**kwargs):
            assert device=='EAST' and sid in (plan['fit'] if phase=='fit' else plan['held']+p['external_ids'])
            return original_signal(device,sid,**kwargs)
        def label_guard(device,sid):
            assert device=='EAST' and phase=='fit' and sid in plan['fit']
            access.append(dict(fold=f,shot=sid,phase=phase));return original_outcome(device,sid)
        warm=load_model(f);before=state_cpu(warm);mapping=READ(REFERENCE/f'fold{f}'/'input_map.json')
        _durable_json(folder/'input_map.json',mapping)
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=label_guard):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries,_=schedule(shots,plan['fit'],f);assert entries==READ(PREVIOUS/f'fold{f}'/'schedule.json')
            _durable_json(folder/'schedule.json',entries)
            model,history=train(warm,shots,entries,p['threshold'],folder,f,source);counts['offline_steps']+=len(history)
            oldhistory=READ(PREVIOUS/f'fold{f}'/'training.json')
            assert [r['source_examples'] for r in history]==[r['source_examples'] for r in oldhistory]
            frozen=state_cpu(model);phase='predict'
            for i,sid in enumerate(plan['held']+p['external_ids'],1):
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                scores,updates,_=predict(model,signal);counts['intra_steps']+=sum(r['gradient_steps'] for r in updates)
                ref=PREVIOUS if sid in plan['held'] else EXTERNAL;prefix='meta' if sid in plan['held'] else 'prepared'
                with np.load(ref/f'fold{f}'/'shots'/f'{sid}.npz') as z:
                    np.testing.assert_array_equal(z['times'],s['times'])
                    curves={f'old_{a}':z[f'{prefix}_{a}'].copy() for a in ('static','intra')}
                curves.update({f'prefix_{a}':v for a,v in scores.items()})
                np.testing.assert_array_equal(scores['static'][:256],scores['intra'][:256])
                np.savez_compressed(folder/'shots'/f'{sid}.npz',times=s['times'],**curves)
                _durable_json(folder/'shots'/f'{sid}_updates.json',updates)
                if i%20==0 or i==70:print(dict(stage='predict',fold=f,shots=i,seconds=round(time.monotonic()-start,2)),flush=True)
            assert_unchanged(model,frozen);assert_unchanged(warm,before)
        _durable_json(root/'fit_label_access.json',access)
    summary,changes=evaluate(root,p)
    for path,h in p['checkpoints'].items():assert sha(path)==h
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=summary,changes=changes,counts=counts,source_unchanged=True,
        test360_accessed=False,seconds=time.monotonic()-start,
        artifact_sha256={str(pp.relative_to(root)):sha(pp) for pp in root.rglob('*') if pp.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
