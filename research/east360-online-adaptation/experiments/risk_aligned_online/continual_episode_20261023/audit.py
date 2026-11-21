"""Independent counts, reveal/replay identities, and stateful CUDA reconstruction."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import assert_unchanged,state_cpu
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import models_for_calibration,REFERENCE
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import episode_indices,trim_memory,select_pair,predict
from .run import ARMS,labeled,optimizer,train_current


def recount(times,scores,disruptive,down,threshold):
    times=np.asarray(times);scores=np.asarray(scores)
    valid=np.ones(len(times),bool) if not disruptive else (down-times)*1000>=-1e-6
    alarm_indices=np.flatnonzero(valid & (scores>=threshold))
    alarm=bool(len(alarm_indices))
    lead=round(float((down-times[alarm_indices[0]])*1000),9) if alarm and disruptive else None
    return dict(is_disrupt=bool(disruptive),alarm=alarm,lead_ms=lead,timely=bool(lead is not None and lead>30))


def read_record(sid,outcome,mapping):
    s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
    return labeled(s,outcome,mapping)


def main(root):
    torch.set_num_threads(2);torch.manual_seed(131)
    p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
    outcomes=READ(root/'outcomes.json');assert [r['shot'] for r in outcomes]==p['shot_ids']
    stored=READ(root/'per_shot.json');mapping=READ(REFERENCE/'input_map.json')
    counts={};flips={};checks=0;intra_steps=0;inter_steps=0;episode_count=0
    memory={};outcome_by_id={};reader=io23.read_outcome
    for sid in p['preparation_ids']:
        outcome=reader('EAST',sid);outcome_by_id[sid]=outcome
        s=read_record(sid,outcome,mapping);memory[sid]=(s,episode_indices(s))
    authorized=set(p['preparation_ids'])
    for i,row in enumerate(outcomes):
        sid=row['shot'];path=root/'shots'/f'{sid}.npz'
        seal=READ(root/'shots'/f'{sid}_seal.json')
        assert seal==dict(shot=sid,index=i,sha256=sha(path),model_version=i)
        assert row['prediction_sha256']==seal['sha256']
        outcome=reader('EAST',sid);assert outcome==(row['is_disrupt'],row['down'])
        outcome_by_id[sid]=outcome
        with np.load(path) as z:
            for arm in ARMS:
                for label,threshold in (('calibrated',p['threshold']),('source_threshold',p['source_threshold'])):
                    key=arm+'/'+label
                    result=dict(shot=sid,**recount(z['times'],z[arm],*outcome,threshold))
                    assert result==stored[key]['predictions'][i],(sid,key)
                    acc=counts.setdefault(key,dict(tp=0,fp=0,very_early_over3000=0))
                    acc['tp']+=int(result['timely']);acc['fp']+=int(result['alarm'] and not outcome[0])
                    acc['very_early_over3000']+=int(result['timely'] and result['lead_ms']>3000);checks+=1
            for arm in ('meta_intra','meta_hold'):np.testing.assert_array_equal(z[arm][:256],z['meta_static'][:256])
            for arm,rs in READ(root/'shots'/f'{sid}_intra.json').items():
                for r in rs:
                    count=r['first_affected_prediction']
                    assert count==r['last_observation']+1 and count>=256 and count<len(z['times'])
                    assert all(e+25<count and e>=127 for e in r['forecast_origins'])
                    intra_steps+=r['gradient_steps']
        authorized.add(sid);s=read_record(sid,outcome,mapping)
        memory[sid]=(s,episode_indices(s));trim_memory(memory)
        tr=READ(root/'shots'/f'{sid}_inter.json');assert tr['memory_ids']==list(memory)
        assert set(memory)<=authorized
        rng=np.random.default_rng(131+i)
        for r in tr['steps']:
            selected=select_pair(memory,sid,rng)
            expected=[dict(shot=ss['shot'],**e) for ss,e in selected]
            if not selected:assert r['reason']=='no_authorized_ordered_episode';continue
            for arm in ('inter','meta'):
                assert [{k:e[k] for k in ('shot','end','count','label')} for e in r[arm]['episodes']]==expected
                assert all(e['shot'] in authorized and e['count']<=e['end'] for e in r[arm]['episodes'])
                if arm=='meta':
                    assert all(e['last_feedback_observation']==e['count']-1 and max(e['forecast_origins'])+25<e['count'] for e in r[arm]['episodes'])
                inter_steps+=1;episode_count+=len(expected)
        if (i+1)%100==0:print(dict(stage='audit_counts',shots=i+1),flush=True)
    for k,c in counts.items():
        for metric,value in c.items():assert done['metrics'][k][metric]==value,(k,metric)
    for comparator in ('static','fixed_intra','inter_only','meta_static','meta_hold'):
        a=stored[comparator+'/calibrated']['predictions'];b=stored['meta_intra/calibrated']['predictions']
        diff=[]
        for old,new in zip(a,b):
            was=old['timely'] if old['is_disrupt'] else not old['alarm']
            now=new['timely'] if new['is_disrupt'] else not new['alarm']
            if was!=now:diff.append(dict(shot=old['shot'],beneficial=now,before=old,after=new))
        flips[comparator]=diff
    parent,_=make_parent();origin=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(origin)
    initial=torch.load(root/'checkpoints'/'before000.pt',map_location='cpu',weights_only=True)
    for arm in ('meta','inter'):
        assert all(torch.equal(v,initial[arm][k]) for k,v in original.items())
    for path in sorted((root/'checkpoints').glob('before*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=True)
        for arm in ('meta','inter'):
            assert all(torch.equal(v,state[arm][k]) for k,v in original.items() if k not in p['parameter_names'])
    replayed=[]
    # Reconstruct whole shots AND the intervening persistent/Adam state. The
    # final block verifies the complete saved end state, not isolated curves.
    ranges=[(0,min(2,len(outcomes)))]
    if len(outcomes)>=40:ranges.append((20,22))
    if len(outcomes)>=360:ranges.append((340,360))
    def deny(*args,**kwargs):raise AssertionError('replay uses only explicitly authorized ended outcomes')
    for start,end in ranges:
        checkpoint=torch.load(root/'checkpoints'/f'before{start:03}.pt',map_location='cuda',weights_only=True)
        meta=models_for_calibration(parent)['mapped_fixed'];inter=models_for_calibration(parent)['mapped_fixed']
        meta.load_state_dict(checkpoint['meta']);inter.load_state_dict(checkpoint['inter'])
        opts=dict(meta=optimizer(meta),inter=optimizer(inter))
        for name in opts:opts[name].load_state_dict(checkpoint['optimizers'][name])
        allowed=set(p['preparation_ids'])|set(p['shot_ids'][:start]);mem={}
        assert set(checkpoint['memory_ids'])<=allowed
        for sid in checkpoint['memory_ids']:
            s=read_record(sid,outcome_by_id[sid],mapping);mem[sid]=(s,episode_indices(s))
        with patch.object(io23,'read_outcome',side_effect=deny):
            for index in range(start,end):
                sid=p['shot_ids'][index]
                raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:raw[k] for k in ('values','available')},mapping)
                curves,rows=predict(meta,inter,signal)
                with np.load(root/'shots'/f'{sid}.npz') as z:
                    for arm,curve in curves.items():np.testing.assert_array_equal(curve,z[arm])
                assert rows==READ(root/'shots'/f'{sid}_intra.json'),sid
                # Only after complete prediction matching is this shot's
                # stored outcome made available to the training replay.
                s=labeled(raw,outcome_by_id[sid],mapping);mem[sid]=(s,episode_indices(s));trim_memory(mem)
                training=train_current(meta,inter,opts,mem,sid,index)
                assert training==READ(root/'shots'/f'{sid}_inter.json')['steps'],sid
                replayed.append(sid)
                if len(replayed)%4==0:print(dict(stage='stateful_replay',shots=len(replayed)),flush=True)
        final=root/'checkpoints'/f'before{end:03}.pt'
        if final.exists():
            saved=torch.load(final,map_location='cpu',weights_only=True)
            for arm,net in (('meta',meta),('inter',inter)):
                assert_unchanged(net,saved[arm])
            assert list(mem)==saved['memory_ids']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    evidence=dict(status='VERIFIED_PREQUENTIAL_COUNTS_IDENTITIES_AND_STATEFUL_REPLAY',
        completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),counts=counts,
        shot_arm_checks=checks,intra_updates=intra_steps,inter_steps=inter_steps,episode_checks=episode_count,
        replays=replayed,flips=flips,source_unchanged=True,independent_confirmation=False)
    _durable_json(root/'verification.json',evidence)
    print({k:v for k,v in evidence.items() if k not in ('counts','flips')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
