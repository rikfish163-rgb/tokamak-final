"""Recount and reconstruct preparation plus stateful policy-learning segments."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..continual_episode_20261023.audit import recount, read_record
from ..continual_episode_20261023.learning import episode_indices, select_pair, trim_memory
from ..continual_episode_20261023.run import labeled
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import models_for_calibration, REFERENCE
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import WeightPolicy, ARMS, predict, train_policy
from .run import ALL_ARMS, normalization, prepare_pair, train_ended, read_signal, PLAN


def main(root):
    torch.set_num_threads(2);torch.manual_seed(131)
    p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in p['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    for f,h in p['input_sha256'].items():assert sha(f)==h,f
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256
    out=READ(root/'outcomes.json');stored=READ(root/'per_shot.json')
    assert [r['shot'] for r in out]==p['shot_ids']
    mapping=READ(REFERENCE/'input_map.json');memory={};by_id={}
    for sid in p['preparation_ids']:
        outcome=io23.read_outcome('EAST',sid);by_id[sid]=outcome
        s=read_record(sid,outcome,mapping);memory[sid]=(s,episode_indices(s))
    initial_memory=memory.copy();authorized=set(memory)
    checks=0;updates=0;training_steps=0;counts={}
    for i,row in enumerate(out):
        sid=row['shot'];path=root/'shots'/f'{sid}.npz'
        assert READ(root/'shots'/f'{sid}_seal.json')==dict(shot=sid,index=i,sha256=sha(path),policy_version=i)
        assert row['prediction_sha256']==sha(path)
        outcome=io23.read_outcome('EAST',sid);by_id[sid]=outcome
        assert outcome==(row['is_disrupt'],row['down'])
        with np.load(path) as z:
            for arm in ALL_ARMS:
                for label,threshold in (('calibrated',p['threshold']),('source_threshold',p['source_threshold'])):
                    key=arm+'/'+label;r=dict(shot=sid,**recount(z['times'],z[arm],*outcome,threshold))
                    assert r==stored[key]['predictions'][i],(sid,key)
                    acc=counts.setdefault(key,dict(tp=0,fp=0,very_early_over3000=0))
                    acc['tp']+=int(r['timely']);acc['fp']+=int(r['alarm'] and not outcome[0])
                    acc['very_early_over3000']+=int(r['timely'] and r['lead_ms']>3000);checks+=1
            for arm in ARMS:np.testing.assert_array_equal(z[arm][:256],z['static'][:256])
            for rs in READ(root/'shots'/f'{sid}_intra.json').values():
                for receipt in rs:
                    count=receipt['first_affected_prediction']
                    assert receipt['last_observation']==count-1 and 256<=count<len(z['times'])
                    assert all(e>=127 and e+25<count for e in receipt['forecast_origins'])
                    w=np.asarray(receipt['weights'])
                    assert w.shape==(23,) and np.isfinite(w).all() and (w>=np.exp(-2)-1e-6).all() and (w<=np.exp(2)+1e-6).all()
                    updates+=receipt['gradient_steps']
        authorized.add(sid);s=read_record(sid,outcome,mapping)
        memory[sid]=(s,episode_indices(s));trim_memory(memory)
        tr=READ(root/'shots'/f'{sid}_inter.json')
        assert tr['memory_ids']==list(memory) and set(memory)<=authorized
        rng=np.random.default_rng(131+i)
        for step in tr['steps']:
            pair=select_pair(memory,sid,rng)
            if not pair:assert step['reason']=='no_authorized_ordered_episode';continue
            expected=[dict(shot=s['shot'],**e) for s,e in pair]
            assert [{k:e[k] for k in ('shot','end','count','label')} for e in step['episodes']]==expected
            training_steps+=1
        if (i+1)%100==0:print(dict(stage='recount',shots=i+1),flush=True)
    for key,values in counts.items():
        for name,value in values.items():assert done['metrics'][key][name]==value
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    center,scale,norm_rows=normalization(model,initial_memory,device='cuda')
    np.testing.assert_array_equal(center,p['normalization']['center'])
    np.testing.assert_array_equal(scale,p['normalization']['scale'])
    assert norm_rows==p['normalization']['rows']
    policy=WeightPolicy(center,scale).cuda();opt=torch.optim.Adam(policy.parameters(),lr=.01)
    preparation=READ(root/'preparation.json');rng=np.random.default_rng(131)
    def deny(*args,**kwargs):raise AssertionError('replay may use only supplied authorized outcomes')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for i,saved in enumerate(preparation):
            actual=train_policy(model,policy,opt,prepare_pair(initial_memory,rng),device='cuda')
            assert actual==saved,i
    first=torch.load(root/'checkpoints/before000.pt',map_location='cpu',weights_only=True)
    assert_unchanged(policy,first['online']);assert_unchanged(policy,first['fixed'])
    for path in sorted((root/'checkpoints').glob('*.pt')):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,ck['fixed'][k]) for k,v in first['fixed'].items())
    replayed=[];ranges=[(0,min(2,len(out)))]
    if len(out)>=40:ranges.append((20,22))
    if len(out)>=360:ranges.append((340,360))
    for start,end in ranges:
        ck=torch.load(root/'checkpoints'/f'before{start:03}.pt',map_location='cuda',weights_only=True)
        fixed,online=WeightPolicy(center,scale).cuda(),WeightPolicy(center,scale).cuda()
        fixed.load_state_dict(ck['fixed']);online.load_state_dict(ck['online'])
        optimizer=torch.optim.Adam(online.parameters(),lr=.01);optimizer.load_state_dict(ck['optimizer'])
        memory={};allowed=set(p['preparation_ids'])|set(p['shot_ids'][:start])
        assert set(ck['memory_ids'])<=allowed
        for sid in ck['memory_ids']:
            s=read_record(sid,by_id[sid],mapping);memory[sid]=(s,episode_indices(s))
        with patch.object(io23,'read_outcome',side_effect=deny):
            for index in range(start,end):
                sid=p['shot_ids'][index];raw=read_signal(sid)
                signal=transform({k:raw[k] for k in ('values','available')},mapping)
                curves,receipts=predict(model,fixed,online,signal)
                with np.load(root/'shots'/f'{sid}.npz') as z:
                    for a in ARMS:np.testing.assert_array_equal(curves[a],z[a])
                assert receipts==READ(root/'shots'/f'{sid}_intra.json')
                s=labeled(raw,by_id[sid],mapping);memory[sid]=(s,episode_indices(s));trim_memory(memory)
                actual=train_ended(model,online,optimizer,memory,sid,index)
                assert actual==READ(root/'shots'/f'{sid}_inter.json')['steps']
                replayed.append(sid)
                if len(replayed)%4==0:print(dict(stage='stateful_replay',shots=len(replayed)),flush=True)
        end_path=root/'checkpoints'/f'before{end:03}.pt'
        if end_path.exists():
            saved=torch.load(end_path,map_location='cpu',weights_only=True)
            assert_unchanged(online,saved['online']);assert list(memory)==saved['memory_ids']
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    flips={}
    for other in ('static','uniform','fixed','hold','mean'):
        changes=[]
        for a,b in zip(stored[other+'/calibrated']['predictions'],stored['online/calibrated']['predictions']):
            ga=a['timely'] if a['is_disrupt'] else not a['alarm']
            gb=b['timely'] if b['is_disrupt'] else not b['alarm']
            if ga!=gb:changes.append(dict(shot=a['shot'],beneficial=gb,before=a,after=b))
        flips[other]=changes
    result=dict(status='VERIFIED_COUNTS_PREPARATION_AND_STATEFUL_POLICY_REPLAY',
        completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        counts=counts,shot_arm_checks=checks,intra_updates=updates,post_shot_steps=training_steps,
        preparation_replayed=len(preparation),stateful_shots_replayed=replayed,flips=flips,
        persistent_classifier_unchanged=True,source_unchanged=True,independent_confirmation=False)
    _durable_json(root/'verification.json',result)
    print({k:v for k,v in result.items() if k not in ('counts','flips')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
