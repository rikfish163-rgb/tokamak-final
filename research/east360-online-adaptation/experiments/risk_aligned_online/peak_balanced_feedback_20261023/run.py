"""Pre-registered 2x2 follow-up to target-feedback failure; no new risk module."""
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

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..fixed_test_protocol import score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..target_feedback_meta_20261023.learning import episode_pool,block_episode
from ..target_feedback_meta_20261023.run import scored_row
from ..target_feedback_meta_20261023.stream import predict_five
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public,predict_many
from .learning import peak_manifest,balanced_head_gradient

REFERENCE=Path(__file__).parents[1]/'runs/target_feedback15_holdout5_test360_20261023'
ARMS=('warm','random_original','random_balanced','peak_original','peak_balanced')
TRAIN_ARMS=ARMS[2:]
HEAD=('base.forecast_head.weight','base.forecast_head.bias')


def prepare(warm,shots,root,threshold,old):
    byid={s['shot']:s for s in shots};fit=[byid[sid] for sid in old['fit_ids']]
    pools=episode_pool(fit,old['fit_ids']);_durable_json(root/'eligibility.json',pools)
    original=[json.loads(line) for line in (REFERENCE/'training_manifest.jsonl').read_text().splitlines()]
    assert len(original)==64
    safe_scores={}
    for s in fit:
        if s['is_disrupt']:continue
        signal={k:s[k] for k in ('values','available')}
        score=predict_many(warm,dict(static=signal),device='cuda')['static']
        safe_scores[s['shot']]=score
        np.savez_compressed(root/'calibration'/f"{s['shot']}.npz",times=s['times'],static=score)
    peak,peaks=peak_manifest(original,pools,safe_scores)
    _durable_json(root/'manifests.json',dict(random=original,peak=peak,peaks=peaks))
    models={k:copy.deepcopy(warm) for k in ARMS};before=state_cpu(warm)
    models['random_original'].load_state_dict(torch.load(REFERENCE/'maximum_meta.pt',map_location='cpu',weights_only=True)['model'])
    opts={}
    for arm in TRAIN_ARMS:
        models[arm].base.forecast_head.requires_grad_(True)
        opts[arm]=torch.optim.Adam(models[arm].base.forecast_head.parameters(),lr=1e-4)
    history=[];started=time.monotonic();boundary=math.log(threshold/(1-threshold))
    for step in range(64):
        row=dict(step=step)
        for arm in TRAIN_ARMS:
            model=models[arm];opt=opts[arm];opt.zero_grad(set_to_none=True);records=[]
            episodes=(peak if arm.startswith('peak') else original)[step]['episodes']
            for e in episodes:
                s=byid[e['shot']];signal={k:s[k] for k in ('values','available')}
                r=block_episode(model,signal,e['count'],e['ends'],e['is_disrupt'],boundary=boundary,
                                objective='maximum',device='cuda',create_graph=True)
                record=dict(**e,risk_before=float(r['risk_before']),risk_after=float(r['risk_after'].detach()),
                    forecast=float(r['forecast'].detach()),update_counts=r['all_update_counts'],
                    differentiated_counts=r['differentiated_counts'])
                if arm.endswith('balanced'):
                    grads,info=balanced_head_gradient(r,tuple(model.base.forecast_head.parameters()))
                    for parameter,g in zip(model.base.forecast_head.parameters(),grads):
                        if parameter.grad is None:parameter.grad=g/2
                        else:parameter.grad.add_(g/2)
                    record['gradient_balance']=info
                else:
                    loss=r['forecast']+(r['risk_after']-r['risk_before'])/(.001*len(r['differentiated_counts']))
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite original objective')
                    (loss/2).backward();record['objective']=float(loss.detach())
                records.append(record);del r
            assert all(p.grad is None for k,p in model.named_parameters() if k not in HEAD)
            norm=torch.nn.utils.clip_grad_norm_(model.base.forecast_head.parameters(),1.,error_if_nonfinite=True)
            opt.step();assert_unchanged(model,before,HEAD)
            row[arm]=dict(records=records,head_gradient_norm=float(norm))
        history.append(row)
        if (step+1)%8==0:
            _durable_json(root/'training.json',history)
            print(json.dumps(dict(stage='training',step=step+1,seconds=time.monotonic()-started)),flush=True)
    receipts={}
    for arm,m in models.items():
        m.set_stage('frozen');assert_unchanged(m,before,HEAD if arm!='warm' else ())
        torch.save(dict(model=state_cpu(m),config=m.cfg.to_dict()),root/f'{arm}.pt')
        receipts[arm]=dict(sha256=sha(root/f'{arm}.pt'),new_training_steps=64 if arm in TRAIN_ARMS else 0,
            reused_64step_model=arm=='random_original',classifier_bitwise_unchanged=True)
    _durable_json(root/'preparation_receipts.json',receipts)
    return models


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    root=args.output
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    read=lambda path:json.loads(path.read_text())
    old=read(REFERENCE/'protocol.json');verified=read(REFERENCE/'verification.json')
    assert verified['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==verified['completed_sha256']
    assert sha(REFERENCE/'protocol.json')==verified['protocol_sha256']
    for f,digest in old['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    receipts=read(REFERENCE/'preparation_receipts.json')
    for arm in ('warm','maximum_meta'):assert sha(REFERENCE/f'{arm}.pt')==receipts[arm]['sha256']
    base,source=load_parent(DEFAULT_PARENT);warm=PairedFeedback(base,np.ones(23)).cuda()
    warm.load_state_dict(torch.load(REFERENCE/'warm.pt',map_location='cpu',weights_only=True)['model']);warm.set_stage('frozen')
    threshold=float(source['threshold']);assert threshold==old['threshold']
    mapping=read(REFERENCE/'input_map.json');contract=old['contract']
    root.mkdir(parents=True)
    for name in ('calibration','holdout','shots'):(root/name).mkdir()
    files={io23.ROOT/f for f in old['code_sha256']}
    files.update(Path(__file__).parent.glob('*.py'));files.add(Path(__file__).parents[1]/'tests/test_peak_balanced_feedback.py')
    hashes={str(f.relative_to(io23.ROOT)):sha(f) for f in sorted(files)}
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/高风险片段覆盖与梯度平衡_事前约定_20261023.md'
    p=dict(source_sha256=PARENT_SHA256,reference=str(REFERENCE),reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        reference_completed_sha256=sha(REFERENCE/'completed.json'),reference_warm_sha256=sha(REFERENCE/'warm.pt'),
        reference_maximum_sha256=sha(REFERENCE/'maximum_meta.pt'),code_sha256=hashes,plan_path=str(plan),plan_sha256=sha(plan),
        contract=contract,threshold=threshold,arms=ARMS,new_training_arms=TRAIN_ARMS,steps=64,seed=131,
        selected_ids=old['selected_ids'],fit_ids=old['fit_ids'],holdout_ids=old['holdout_ids'],
        persistent_optimized_parameters=HEAD,head_warmup_and_input_mapping_used_all20=True,
        static_peak_selection='historical safe TRAIN only, fixed warm classifier, alternating with original random per shot',
        risk_gradient_cap='per episode risk-gradient norm <= forecast-gradient norm; Adam is not a monotonicity guarantee',
        metadata_boundary='class-listed split known from prior experiment; no TEST HDF outcomes before all seals',
        threshold_selection=False,checkpoint_selection=False,independent_confirmation=False,
        equal_optimizer_steps=True,equal_total_compute=False)
    _durable_json(root/'protocol.json',p);_durable_json(root/'input_map.json',mapping)
    access=[];real_outcome=io23.read_outcome;allowed=set(old['selected_ids'])
    def guarded(device,sid):
        if device!='EAST' or sid not in allowed:raise AssertionError('unauthorized HDF label access before sealing')
        access.append(sid);return real_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guarded):
        shots=[]
        for sid in old['selected_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            s=simple_data.attach_outcome(s,'EAST',{})
            shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
        models=prepare(warm,shots,root,threshold,old);frozen={k:state_cpu(m) for k,m in models.items()}
        held={}
        for s in shots:
            if s['shot'] not in old['holdout_ids']:continue
            preds,updates=predict_five(models,{k:s[k] for k in ('values','available')})
            np.savez_compressed(root/'holdout'/f"{s['shot']}.npz",times=s['times'],**preds)
            _durable_json(root/'holdout'/f"{s['shot']}_updates.json",updates)
            for name,v in preds.items():held.setdefault(name,[]).append(scored_row(s,v))
        _durable_json(root/'holdout_metrics.json',{k:metrics(v,threshold,lead=30.) for k,v in held.items()})
        seals=[]
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred,updates=predict_five(models,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(REFERENCE/'shots'/f'{sid}.npz',allow_pickle=False) as ref:
                for name,old_name in (('static','static'),('warm_intra','warm_intra'),('random_original_intra','maximum_meta_intra')):
                    np.testing.assert_array_equal(pred[name],ref[old_name])
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**pred);_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            if (index+1)%40==0:
                progress=dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(root/'prediction_receipts.json',seals);_durable_json(root/'pre_eval_label_access.json',access)
    rows={}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['times'],s['times'])
            for name in ['static']+[f'{k}_intra' for k in ARMS]:rows.setdefault(name,[]).append(scored_row(s,trace[name]))
    results={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    for k,m in models.items():assert_unchanged(m,frozen[k])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'per_shot.json',results)
    summary={k:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for k,v in results.items()}
    _durable_json(root/'completed.json',dict(status='COMPLETED',metrics=summary,seconds=time.monotonic()-started,
        static_unchanged=True,reused_arms_exact=True,source_unchanged=True,temporary_resets=True))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
