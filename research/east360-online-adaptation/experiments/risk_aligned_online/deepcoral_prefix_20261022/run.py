"""Predeclared six-arm warm-prefix representation and intra-shot experiment.

No target outcome is accessed until all360 prediction traces are sealed.
The first20 shots are common unadapted predictions, not removed from scoring.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder
from ..redesign_20261004.prepare_prior import load_parent, load_source_train, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, pools, sha
from ..simple23 import fit_forecaster
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from ..warmstart_alignment_20261022.run import predict_many, public, snapshot
from .learning import (SIGNAL_KEYS, coral_loss, natural_batch, supervised_batch,
                       validate_prefix, adapt_arrived, eligible_records)

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022'
ARMS=('density', 'replay', 'coral')
STEPS=512
FORECAST_STEPS=256


def state_cpu(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def assert_unchanged(model, before, allowed=()):
    for k,v in model.state_dict().items():
        if k not in allowed and not torch.equal(v.cpu(),before[k]):
            raise AssertionError(f'frozen state changed: {k}')


def prepare_models(parent, prefix, order, mapping, output):
    validate_prefix(prefix,order,20)
    source,ids=load_source_train()
    source_natural=[{k:s[k] for k in SIGNAL_KEYS} for s in source]
    target=[dict(shot=s['shot'],device='EAST',**transform(
        {k:s[k] for k in ('values','available')},mapping)) for s in prefix]
    validate_prefix(target,order,20)
    source_forecast,sf_excluded=eligible_records(source_natural,parent.cfg,future=True)
    target_forecast,tf_excluded=eligible_records(target,parent.cfg,future=True)
    source_natural,sn_excluded=eligible_records(source_natural,parent.cfg)
    target,tn_excluded=eligible_records(target,parent.cfg)
    _durable_json(output/'window_eligibility.json',dict(
        source_alignment=dict(eligible_ids=[s['shot'] for s in source_natural],excluded=sn_excluded),
        target_alignment=dict(eligible_ids=[s['shot'] for s in target],excluded=tn_excluded),
        source_forecast=dict(eligible_ids=[s['shot'] for s in source_forecast],excluded=sf_excluded),
        target_forecast=dict(eligible_ids=[s['shot'] for s in target_forecast],excluded=tf_excluded)))
    groups=pools(source)
    models={k:copy.deepcopy(parent) for k in ARMS}
    before=state_cpu(parent)
    rngs=[np.random.default_rng(s) for s in (131,2131,3131)]
    optimizers={}
    for name in ('replay','coral'):
        model=models[name]; allowed=set(model.encoder_names())
        for k,p in model.named_parameters(): p.requires_grad_(k in allowed)
        optimizers[name]=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    train_log=[]
    with (output/'training_manifest.jsonl').open('x') as manifest:
        for step in range(STEPS):
            x,m,y,supervision=supervised_batch(source,groups,rngs[0],32,parent.cfg,'cuda')
            sx,sm,source_draws=natural_batch(source_natural,rngs[1],64,parent.cfg,'cuda')
            tx,tm,target_draws=natural_batch(target,rngs[2],64,parent.cfg,'cuda')
            manifest.write(json.dumps(dict(step=step,supervision=supervision,
                source_alignment=source_draws,target_alignment=target_draws))+'\n')
            record=dict(step=step)
            for name in ('replay','coral'):
                model=models[name]; opt=optimizers[name]; opt.zero_grad(set_to_none=True)
                h=model(x,m,mode='risk_state')['risk_state']
                bce=F.binary_cross_entropy_with_logits(model.base.risk_head(h).squeeze(-1),y)
                hs=model(sx,sm,mode='risk_state')['risk_state']
                ht=model(tx,tm,mode='risk_state')['risk_state']
                covariance=coral_loss(hs,ht)
                loss=bce+(1. if name=='coral' else 0.)*covariance
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite representation training')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                   1.,error_if_nonfinite=True)
                opt.step()
                record[name]=dict(bce=float(bce.detach()),coral=float(covariance.detach()),gradient_norm=float(norm))
            train_log.append(record)
            if (step+1)%128==0: print(json.dumps(dict(stage='representation',**record)),flush=True)
    _durable_json(output/'representation_training.json',train_log)
    for name,model in models.items():
        assert_unchanged(model,before,model.encoder_names() if name!='density' else ())
        model.set_stage('frozen')
    # Auxiliary predictor calibration is identical in sampling and budget for
    # all three encoder states. Only forecast_head changes, hence its static
    # classifier is checked bitwise unchanged. No extra inference risk head.
    pre_forecast={k:state_cpu(m) for k,m in models.items()}
    forecast_opts={k:torch.optim.Adam(m.base.forecast_head.parameters(),lr=3e-4) for k,m in models.items()}
    rng_source,rng_target=np.random.default_rng(4131),np.random.default_rng(5131)
    forecast_log=[]
    with (output/'forecast_manifest.jsonl').open('x') as manifest:
        for step in range(FORECAST_STEPS):
            a=natural_batch(source_forecast,rng_source,32,parent.cfg,'cuda',future=True)
            b=natural_batch(target_forecast,rng_target,32,parent.cfg,'cuda',future=True)
            x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
            manifest.write(json.dumps(dict(step=step,source=a[-1],target=b[-1]))+'\n')
            record=dict(step=step)
            for name,model in models.items():
                record[name]=fit_forecaster(model.base,forecast_opts[name],x,m,t,tm)
            forecast_log.append(record)
            if (step+1)%128==0: print(json.dumps(dict(stage='forecast_head',**record)),flush=True)
    _durable_json(output/'forecast_training.json',forecast_log)
    receipts={}
    for name,model in models.items():
        assert_unchanged(model,pre_forecast[name],('base.forecast_head.weight','base.forecast_head.bias'))
        model.set_stage('frozen')
        state=state_cpu(model)
        changed=[k for k in state if not torch.equal(state[k],before[k])]
        torch.save(dict(model=state,config=model.cfg.to_dict(),threshold=None),output/f'{name}.pt')
        receipts[name]=dict(changed_parameters=changed,checkpoint_sha256=sha(output/f'{name}.pt'),
            encoder_delta_norm=float(sum((state[k]-before[k]).double().square().sum()
                for k in model.encoder_names()).sqrt()))
    _durable_json(output/'preparation_receipts.json',receipts)
    return models


def predict_pairs(models,signal,*,device='cuda',interval=128):
    if set(signal)!={'values','available'} or type(interval) is not int or interval<100 or interval%32:
        raise ValueError('signal-only input; interval multiple32 and >=four H25 targets')
    n=len(signal['values'])
    states={k:initial_encoder(m) for k,m in models.items()}
    fast={k:initial_encoder(m) for k,m in models.items()}
    predictions={f'{k}_{mode}':[] for k in models for mode in ('static','intra')}
    updates={k:[] for k in models}
    for lo in range(0,n,interval):
        hi=min(lo+interval,n)
        for start in range(lo,hi,32):
            end=min(start+32,hi)
            windows=[simple_data.window(signal,t,next(iter(models.values())).cfg.window) for t in range(start,end)]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            for name,model in models.items():
                for mode,state in (('static',states[name]),('intra',fast[name])):
                    h=model.predict_fast(x,m,state,mode='risk_state')['risk_state']
                    p=model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy()
                    predictions[f'{name}_{mode}'].append(p)
        # Predictions through hi-1 have already been emitted. The four direct
        # forecasts originate inside this block, during which fast was fixed;
        # their recomputation uses exactly the state available at each origin.
        for name,model in models.items():
            if hi<model.cfg.window+model.cfg.horizon or hi==n: continue
            fast[name],receipt=adapt_arrived(model,signal,hi,fast[name],device=device)
            updates[name].append(receipt)
    result={k:np.concatenate(v) for k,v in predictions.items()}
    if not all(np.isfinite(v).all() for v in result.values()): raise FloatingPointError('nonfinite inference')
    return result,updates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    start=time.monotonic(); contract=load_test_contract('EAST'); order=contract['shot_ids']
    base,checkpoint=load_parent(DEFAULT_PARENT)
    parent=PairedFeedback(base,np.ones(23)).cuda(); parent.set_stage('frozen')
    threshold=float(checkpoint['threshold']); original=state_cpu(parent)
    args.output.mkdir(parents=True); (args.output/'shots').mkdir()
    hashes=snapshot(args.output)
    with zipfile.ZipFile(args.output/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as archive:
        for f in Path(__file__).parent.glob('*.py'):
            archive.write(f,str(f.relative_to(io23.ROOT))); hashes[str(f.relative_to(io23.ROOT))]=sha(f)
    source_path=REFERENCE/'source_statistics.json'; statistics=json.loads(source_path.read_text())
    if statistics['shot_ids']!=io23.source_roles()['train']: raise AssertionError('source IDs changed')
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/DeepCORAL与到达信号反馈_事前约定_20261022.md'
    protocol=dict(source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,threshold=threshold,
        plan_path=str(plan),plan_sha256=sha(plan),
        contract=contract,code_sha256=hashes,arms=list(ARMS),steps=STEPS,forecast_steps=FORECAST_STEPS,
        source_statistics_sha256=sha(source_path),prefix_count=20,coral_lambda=1.,learning_rate=1e-4,
        risk_training='source BCE only, vs source BCE + centered64dim CORAL; common batches32/64/64',
        covariance_sampling='equal shot then full-window endpoint; no label or event stratification',
        parameter_scope=parent.encoder_names(),forecast_training='head only,32source+32endedEAST,Adam3e-4',
        intra='temporary encoder,1SGD step1e-3,global gradient cap1,interval128,H25x4arrived targets',
        first_affected_shot_index=20,head_and_threshold_frozen=True,target_labels_for_training=0,
        independent_confirmation=False,original_paper_reproduction=False,
        limitations=['density20 selected during previous development','lambda1 fixed engineering choice',
                     'marginal state mixtures remain different','all data development-exposed'])
    _durable_json(args.output/'protocol.json',protocol)
    pending=[]; receipts=[]; models=None; mapping=None
    frozen={}
    for index,sid in enumerate(order):
        shot=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal={k:shot[k] for k in ('values','available')}
        if index<20:
            p=predict_many(parent,dict(raw=signal),device='cuda')['raw']
            predictions={f'{k}_{mode}':p.copy() for k in ARMS for mode in ('static','intra')}
            updates={k:[] for k in ARMS}
        else:
            predictions,updates=predict_pairs(models,transform(signal,mapping))
        with np.load(REFERENCE/'shots'/f'{sid}.npz',allow_pickle=False) as ref:
            if not np.array_equal(predictions['density_static'],ref['density_gain_n20']):
                raise AssertionError(f'density baseline no longer identical: {sid}')
        path=args.output/'shots'/f'{sid}.npz'
        np.savez_compressed(path,times=shot['times'],**predictions)
        _durable_json(args.output/'shots'/f'{sid}_updates.json',updates)
        receipts.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(args.output/'shots'/f'{sid}_updates.json')))
        if index<20: pending.append({k:shot[k] for k in SIGNAL_KEYS})
        if index==19:
            mapping=fit_prefix_map(statistics,pending,order=order,count=20,policy='density_gain')
            mapping['preceding_prediction_seals']=receipts.copy()
            _durable_json(args.output/'density_map.json',mapping)
            models=prepare_models(parent,pending,order,mapping,args.output)
            frozen={k:state_cpu(m) for k,m in models.items()}; pending=[]
        if (index+1)%20==0:
            progress=dict(stage='stream',shots=index+1,total=len(order),seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json',progress); print(json.dumps(progress),flush=True)
    # Evaluation is physically separated from adaptation. No outcome was read
    # for any EAST shot above, even for the unlabeled warm-prefix preparation.
    _durable_json(args.output/'prediction_receipts.json',receipts)
    rows={f'{k}_{mode}':[] for k in ARMS for mode in ('static','intra')}
    for sid in order:
        shot=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(args.output/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for name in rows:
                valid=shot['valid']
                rows[name].append(dict(shot=sid,times=shot['times'][valid],scores=trace[name][valid],
                                       is_disrupt=shot['is_disrupt'],down=shot['down']))
    full={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    summary={k:dict(full360=public(full[k]),common_suffix340=public(metrics(v[20:],threshold,lead=30.)))
             for k,v in rows.items()}
    for k,m in models.items(): assert_unchanged(m,frozen[k])
    assert_unchanged(parent,original)
    if sha(DEFAULT_PARENT)!=PARENT_SHA256: raise AssertionError('archive changed')
    _durable_json(args.output/'per_shot.json',full)
    _durable_json(args.output/'completed.json',dict(status='COMPLETED',metrics=summary,
        parent_bitwise_unchanged=True,temporary_updates_discarded=True,
        baseline_traces_bitwise_equal=True,seconds=time.monotonic()-start))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
