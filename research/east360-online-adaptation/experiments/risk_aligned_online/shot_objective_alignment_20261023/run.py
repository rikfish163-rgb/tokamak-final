"""Mean vs exact peak EAST preparation; static vs temporary intra per parent."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import SIGNAL_KEYS,eligible_records,natural_batch
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged,predict_pairs
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..labeled_alignment_20261023.learning import balanced_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent,load_source_train
from ..simple23 import fit_forecaster
from ..simple_online import _durable_json
from ..simple_train import pools,sha,metrics
from ..temporal_dependence_20261023.run import REFERENCE,load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public,predict_many
from .learning import candidate_ends,backward_bag,aggregation_for

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/整炮聚合表示准备_事前约定_20261023.md'
ARMS=('mean','safe_peak','peak')
NAMES=[f'{a}_{m}' for a in ('original',*ARMS) for m in ('static','intra')]
STEPS=64


def prepare(parent,source,target,threshold,output):
    initial=state_cpu(parent);models={a:copy.deepcopy(parent) for a in ARMS}
    opts={}
    for a,m in models.items():
        names=set(m.encoder_names())
        for k,p in m.named_parameters():p.requires_grad_(k in names)
        opts[a]=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    candidates={s['shot']:candidate_ends(s) for s in target}
    groups={d:[s for s in target if bool(s['is_disrupt'])==d] for d in (False,True)}
    _durable_json(output/'candidate_manifest.json',{str(sid):ends.tolist() for sid,ends in candidates.items()})
    rs,rt=np.random.default_rng(131),np.random.default_rng(2131);source_groups=pools(source)
    boundary=math.log(threshold/(1-threshold));logs=[];elapsed={a:0. for a in ARMS}
    with (output/'risk_training_manifest.jsonl').open('x') as manifest:
        for step in range(STEPS):
            sx,sm,sy,sreceipt=balanced_batch(source,source_groups,rs,parent.cfg)
            selected=[groups[d][int(rt.integers(len(groups[d])))] for d in (False,True)]
            manifest.write(json.dumps(dict(step=step,source=sreceipt,target_ids=[s['shot'] for s in selected]))+'\n')
            record=dict(step=step)
            for a,m in models.items():
                t0=time.monotonic();m.zero_grad(set_to_none=True)
                h=m(sx,sm,mode='risk_state')['risk_state']
                bce=F.binary_cross_entropy_with_logits(m.base.risk_head(h).squeeze(-1),sy)
                if not torch.isfinite(bce):raise FloatingPointError('nonfinite source risk')
                (.5*bce).backward();bags=[]
                for s in selected:
                    r=backward_bag(m,{k:s[k] for k in ('values','available')},candidates[s['shot']],
                        s['is_disrupt'],boundary,aggregation_for(a,s['is_disrupt']),weight=.25,device='cuda')
                    bags.append(dict(shot=s['shot'],is_disrupt=s['is_disrupt'],**r))
                norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                opts[a].step();elapsed[a]+=time.monotonic()-t0
                record[a]=dict(source_bce=float(bce.detach()),bags=bags,gradient_norm=float(norm),seconds=elapsed[a])
            logs.append(record)
            if (step+1)%8==0:
                print(json.dumps(dict(stage='representation',step=step,arms={a:dict(source_bce=record[a]['source_bce'],
                    target_losses=[r['loss'] for r in record[a]['bags']],seconds=elapsed[a]) for a in ARMS})),flush=True)
    _durable_json(output/'risk_training.json',logs)
    source_base,_=load_parent(DEFAULT_PARENT)
    for a,m in models.items():
        assert_unchanged(m,initial,m.encoder_names());m.set_stage('frozen')
        m.base.forecast_head.load_state_dict(source_base.forecast_head.state_dict())
    del source_base
    before={a:state_cpu(m) for a,m in models.items()}
    sf,se=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in source],parent.cfg,future=True)
    tf,te=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in target],parent.cfg,future=True)
    _durable_json(output/'forecast_exclusions.json',dict(source=se,target=te))
    opts={a:torch.optim.Adam(m.base.forecast_head.parameters(),lr=3e-4) for a,m in models.items()}
    rs,rt=np.random.default_rng(4131),np.random.default_rng(5131);history=[]
    previous=[json.loads(line) for line in (REFERENCE/'forecast_manifest.jsonl').read_text().splitlines()]
    with (output/'forecast_manifest.jsonl').open('x') as manifest:
        for step in range(256):
            source_batch=natural_batch(sf,rs,32,parent.cfg,'cuda',future=True)
            target_batch=natural_batch(tf,rt,32,parent.cfg,'cuda',future=True)
            receipt=dict(step=step,source=source_batch[-1],target=target_batch[-1])
            assert receipt==previous[step];manifest.write(json.dumps(receipt)+'\n')
            x,m,t,tm=[torch.cat((v,w)) for v,w in zip(source_batch[:-1],target_batch[:-1])]
            history.append(dict(step=step,**{a:fit_forecaster(model.base,opts[a],x,m,t,tm) for a,model in models.items()}))
    _durable_json(output/'forecast_training.json',history);receipts={}
    for a,m in models.items():
        assert_unchanged(m,before[a],('base.forecast_head.weight','base.forecast_head.bias'));m.set_stage('frozen')
        state=state_cpu(m);torch.save(dict(model=state,config=m.cfg.to_dict()),output/f'{a}.pt')
        receipts[a]=dict(checkpoint_sha256=sha(output/f'{a}.pt'),changed_parameters=[k for k in state if not torch.equal(state[k],initial[k])],
            representation_training_seconds=elapsed[a],encoder_delta_norm=float(sum((state[k]-initial[k]).double().square().sum() for k in m.encoder_names()).sqrt()))
    _durable_json(output/'preparation_receipts.json',receipts)
    return models


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    read=lambda path:json.loads(path.read_text())
    contract=load_test_contract('EAST');rp=read(REFERENCE/'protocol.json');rv=read(REFERENCE/'verification.json')
    assert contract==rp['contract'] and rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    mapping=read(REFERENCE/'input_map.json');ids=rp['selected_ids']
    assert ids==mapping['calibration_ids'] and len(ids)==20 and not set(ids)&set(contract['shot_ids'])
    parent,threshold=load_model();original=state_cpu(parent)
    output.mkdir(parents=True);(output/'shots').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[io23.ROOT/'experiments/risk_aligned_online'/n for n in (
        'simple23.py','simple_data.py','io23.py','causal_input.py','simple_online.py','simple_train.py','fixed_test_protocol.py',
        'paired_feedback_20261022/model.py','masked_state_20261022/model.py','redesign_20261004/prepare_prior.py',
        'warmstart_alignment_20261022/alignment.py','warmstart_alignment_20261022/run.py',
        'deepcoral_prefix_20261022/run.py','deepcoral_prefix_20261022/learning.py',
        'labeled_alignment_20261023/learning.py','temporal_dependence_20261023/run.py','tests/test_shot_objective_alignment.py')]
    hashes={str(f.relative_to(io23.ROOT)):sha(f) for f in paths}
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in paths:z.write(f,str(f.relative_to(io23.ROOT)))
    protocol=dict(source_sha256=PARENT_SHA256,threshold=threshold,contract=contract,arms=NAMES,selected_ids=ids,
        reference=str(REFERENCE),reference_hashes={f:sha(REFERENCE/f) for f in (
            'protocol.json','completed.json','verification.json','input_only.pt','input_map.json','prediction_receipts.json','forecast_manifest.jsonl')},
        plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256=hashes,steps=STEPS,forecast_steps=256,
        parameter_scope=parent.encoder_names(),learning_rate=1e-4,gradient_cap=1.,risk_head_frozen=True,
        risk_objective='.5 source BCE + .25 full safe bag + .25 (30,125]ms disruptive bag; mean/mean vs peak/mean vs peak/peak',
        different_compute_costs=True,independent_confirmation=False,source_endpoint_labels='unchanged max25 AdvancedTime',
        target_endpoint_labels='unchanged fixed125; only positive30..125 bag and safe fullshot selected for preparation',
        label_gate='No TEST HDF outcome/DownTime until all curves sealed; class split manifest is known',
        preparation='historical TRAIN20,4D16N; not blind-query first20;88 metadata previously checked',
        intra='temporary firstencoder,H25,four arrived targets,SGD.001,cap1,every128,first256; reset per shot')
    _durable_json(output/'protocol.json',protocol);_durable_json(output/'input_map.json',mapping)
    read_outcome=io23.read_outcome;source_ids=set(io23.source_roles()['train']);access=[]
    def guarded(device,sid):
        allowed=source_ids if device=='J-TEXT' else set(ids) if device=='EAST' else set()
        if sid not in allowed:raise AssertionError('nontraining outcome access before seals')
        access.append(dict(device=device,shot=int(sid)));return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guarded):
        source,_=load_source_train();target=[]
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            target.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
        assert sum(s['is_disrupt'] for s in target)==4
        models=prepare(parent,source,target,threshold,output);del source
        in_sample={a:[] for a in ARMS}
        for s in target:
            signal={k:s[k] for k in ('values','available')}
            for a,m in models.items():
                probs=predict_many(m,dict(mapped=signal),device='cuda')['mapped']
                in_sample[a].append(dict(shot=s['shot'],times=s['times'][s['valid']],scores=probs[s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
        _durable_json(output/'preparation_in_sample.json',{a:public(metrics(rows,threshold,lead=30.)) for a,rows in in_sample.items()})
        before={a:state_cpu(m) for a,m in models.items()};seals=[]
        refseals={r['shot']:r for r in read(REFERENCE/'prediction_receipts.json')}
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred,updates=predict_pairs(models,transform({k:s[k] for k in ('values','available')},mapping))
            refpath=REFERENCE/'shots'/f'{sid}.npz';assert sha(refpath)==refseals[sid]['prediction_sha256']
            with np.load(refpath,allow_pickle=False) as z:
                np.testing.assert_array_equal(s['times'],z['times'])
                pred.update(original_static=z['input_only_static'].copy(),original_intra=z['input_only_intra'].copy())
            for a in ARMS:np.testing.assert_array_equal(pred[a+'_static'][:256],pred[a+'_intra'][:256])
            path=output/'shots'/f'{sid}.npz';up=output/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**pred);_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up),reference_sha256=sha(refpath)))
            if (index+1)%30==0:
                progress=dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(output/'prediction_receipts.json',seals);_durable_json(output/'pre_eval_label_access.json',access)
    rows={a:[] for a in NAMES}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in NAMES:rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()}
    for a,m in models.items():assert_unchanged(m,before[a])
    assert_unchanged(parent,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'per_shot.json',results)
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,source_unchanged=True,
        metrics={a:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])) for a,r in results.items()})
    _durable_json(output/'completed.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
