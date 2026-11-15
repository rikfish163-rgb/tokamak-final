"""Few-labelled-shot preparation, then paired static / temporary EAST adaptation.

New user-authorized protocol: calibration is from the existing EAST training
role, not the TEST360 prefix. Role-disjoint does not imply historical chronology:
calibration IDs can exceed earlier test IDs. This is retrospective few-shot
transfer plus causal within-stream updates, NOT a zero-label commissioning run.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import SIGNAL_KEYS, natural_batch, eligible_records
from ..deepcoral_prefix_20261022.run import predict_pairs, state_cpu, assert_unchanged
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256, load_source_train
from ..simple23 import fit_forecaster
from ..simple_train import pools, sha
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import fit_statistics, make_map, transform
from ..warmstart_alignment_20261022.run import predict_many, public, snapshot
from .learning import choose_budget, balanced_batch, moment_loss, conditional_moment_loss

ARMS = ('input_only','supervised','global_moments','conditional_moments')
SPLIT = io23.ROOT/'config/splits/east_paper131_split_with_val.json'
STEPS, FORECAST_STEPS = 512, 256


def historical_pool(test_ids):
    split = json.loads(SPLIT.read_text())
    train = split['train']; val = split['val']
    ids = sorted(train['disruption_shots']+train['non_disruption_shots'])
    val_ids = set(val['disruption_shots']+val['non_disruption_shots'])
    if len(ids)!=88 or len(set(ids))!=88 or set(ids)&(set(test_ids)|val_ids):
        raise ValueError('canonical88 TRAIN must be distinct from VAL and TEST360')
    d = set(json.loads((io23.ROOT/'shot_list/EAST/Disruption_EAST_TV.json').read_text()))
    n = set(json.loads((io23.ROOT/'shot_list/EAST/Non-disruption_EAST_TV.json').read_text()))
    accepted,excluded = [],[]
    for sid in ids:
        outcome,down = io23.read_outcome('EAST',sid)
        if (sid in d)==(sid in n):
            reason = 'absent_or_ambiguous_shotlist_membership'
        elif outcome != (sid in d):
            reason = 'shotlist_HDF5_outcome_disagreement'
        else:
            accepted.append(dict(shot=sid,is_disrupt=outcome,down=down,shotlist_is_disrupt=sid in d))
            continue
        excluded.append(dict(shot=sid,is_disrupt=outcome,reason=reason))
    return accepted,dict(all_training_role_ids=ids,accepted=accepted,excluded=excluded,
        selection='known class-stratified oldest IDs, NOT blind querying',
        metadata_budget=len(ids),signal_and_gradient_budget='only selected10/20')


def prepare(parent, selected, output, source):
    source_signals = [{k:s[k] for k in SIGNAL_KEYS} for s in source]
    target=[]
    for r in selected:
        s=apply_input_contract(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'))
        s=simple_data.attach_outcome(s,'EAST',{})
        if s['is_disrupt'] != r['is_disrupt']:
            raise ValueError('selected historical label changed')
        target.append(s)
    source_stats=fit_statistics(source_signals,device='J-TEXT',allowed_ids=io23.source_roles()['train'])
    target_signals=[{k:s[k] for k in SIGNAL_KEYS} for s in target]
    target_stats=fit_statistics(target_signals,device='EAST',allowed_ids=[r['shot'] for r in selected])
    mapping=make_map(source_stats,target_stats,'density_gain')
    _durable_json(output/'input_map.json',mapping)
    target=[dict(s,**transform({k:s[k] for k in ('values','available')},mapping)) for s in target]
    groups_source,groups_target=pools(source),pools(target)
    models={k:copy.deepcopy(parent) for k in ARMS}
    before=state_cpu(parent); opts={}
    for name in ARMS[1:]:
        model=models[name]; names=set(model.encoder_names())
        for k,p in model.named_parameters(): p.requires_grad_(k in names)
        opts[name]=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rs,rt=np.random.default_rng(131),np.random.default_rng(2131)
    log=[]
    with (output/'risk_training_manifest.jsonl').open('x') as manifest:
        for step in range(STEPS):
            sx,sm,sy,sreceipt=balanced_batch(source,groups_source,rs,parent.cfg)
            tx,tm,ty,treceipt=balanced_batch(target,groups_target,rt,parent.cfg)
            manifest.write(json.dumps(dict(step=step,source=sreceipt,target=treceipt))+'\n')
            record=dict(step=step)
            for name in ARMS[1:]:
                model=models[name]; opt=opts[name]; opt.zero_grad(set_to_none=True)
                hs=model(sx,sm,mode='risk_state')['risk_state']
                ht=model(tx,tm,mode='risk_state')['risk_state']
                bs=F.binary_cross_entropy_with_logits(model.base.risk_head(hs).squeeze(-1),sy)
                bt=F.binary_cross_entropy_with_logits(model.base.risk_head(ht).squeeze(-1),ty)
                if name=='conditional_moments': align=conditional_moment_loss(hs,ht,sy,ty)
                elif name=='global_moments': align=moment_loss(hs,ht)
                else: align=hs.new_zeros(())
                loss=.5*(bs+bt)+align
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite labelled alignment')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                opt.step()
                record[name]=dict(source_bce=float(bs.detach()),target_bce=float(bt.detach()),
                    alignment=float(align.detach()),gradient_norm=float(norm))
            log.append(record)
            if (step+1)%128==0: print(json.dumps(dict(stage='labelled_alignment',**record)),flush=True)
    _durable_json(output/'risk_training.json',log)
    for name,model in models.items():
        assert_unchanged(model,before,model.encoder_names() if name!='input_only' else ())
        model.set_stage('frozen')
    preforecast={k:state_cpu(m) for k,m in models.items()}
    sf,se=eligible_records(source_signals,parent.cfg,future=True)
    tf,te=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in target],parent.cfg,future=True)
    _durable_json(output/'forecast_exclusions.json',dict(source=se,target=te))
    optimizers={k:torch.optim.Adam(m.base.forecast_head.parameters(),lr=3e-4) for k,m in models.items()}
    rs,rt=np.random.default_rng(4131),np.random.default_rng(5131); log=[]
    with (output/'forecast_manifest.jsonl').open('x') as manifest:
        for step in range(FORECAST_STEPS):
            a=natural_batch(sf,rs,32,parent.cfg,'cuda',future=True)
            b=natural_batch(tf,rt,32,parent.cfg,'cuda',future=True)
            x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
            manifest.write(json.dumps(dict(step=step,source=a[-1],target=b[-1]))+'\n')
            record=dict(step=step)
            for name,model in models.items():
                record[name]=fit_forecaster(model.base,optimizers[name],x,m,t,tm)
            log.append(record)
    _durable_json(output/'forecast_training.json',log)
    receipts={}
    for name,model in models.items():
        assert_unchanged(model,preforecast[name],('base.forecast_head.weight','base.forecast_head.bias'))
        model.set_stage('frozen')
        state=state_cpu(model)
        torch.save(dict(model=state,config=model.cfg.to_dict()),output/f'{name}.pt')
        receipts[name]=dict(checkpoint_sha256=sha(output/f'{name}.pt'),
            changed_parameters=[k for k in state if not torch.equal(state[k],before[k])])
    _durable_json(output/'preparation_receipts.json',receipts)
    return models,mapping


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',type=int,choices=(10,20),default=20); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131); started=time.monotonic()
    base,parent_info=load_parent(DEFAULT_PARENT)
    parent=PairedFeedback(base,np.ones(23)).cuda(); parent.set_stage('frozen'); original=state_cpu(parent)
    contract=load_test_contract('EAST'); order=contract['shot_ids']; threshold=float(parent_info['threshold'])
    pool,audit=historical_pool(order); selected=choose_budget(pool,args.budget,test_ids=order)
    args.output.mkdir(parents=True); (args.output/'shots').mkdir()
    _durable_json(args.output/'selection.json',dict(audit=audit,selected=selected))
    hashes=snapshot(args.output)
    dependencies=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/n for n in (
        'deepcoral_prefix_20261022/run.py','deepcoral_prefix_20261022/learning.py','tests/test_labeled_alignment.py')]
    with zipfile.ZipFile(args.output/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as z:
        for f in dependencies:
            name=str(f.relative_to(io23.ROOT));z.write(f,name);hashes[name]=sha(f)
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/少量有标签EAST对齐_事前约定_20261023.md'
    protocol=dict(source_sha256=PARENT_SHA256,source_checkpoint=str(DEFAULT_PARENT),contract=contract,
        threshold=threshold,arms=ARMS,budget=args.budget,selected_ids=[r['shot'] for r in selected],
        plan_sha256=sha(plan),code_sha256=hashes,steps=STEPS,forecast_steps=FORECAST_STEPS,
        input='source asinh unchanged, historical-calibration density-only positive gain',
        source_labels='max25ms AdvancedTime',target_labels='fixed125ms existing contract',
        target_event_labels_used=False,target_AdvancedTime_used=False,
        metadata_selection_budget=88,model_fitting_shots=args.budget,
        chronological_caveat='role-disjoint retrospective calibration; historical IDs are not all earlier than TEST',
        risk_training='.5*(source_BCE+target_BCE)+global_or_labelconditional_mean_covariance',
        mean_weight=1.,covariance_scaling='DeepCORAL 1/(4*d*d)',
        learning_rate=1e-4,parameter_scope=parent.encoder_names(),threshold_and_risk_head_frozen=True,
        intra='existing direct H25 SGD1e-3 cap1 every128; first updateafter256; reset eachshot',
        no_test_outcomes_before_all_prediction_seals=True,independent_confirmation=False,
        original_paper_reproduction=False,no_test_driven_checkpoint_selection=True)
    _durable_json(args.output/'protocol.json',protocol)
    read_outcome=io23.read_outcome
    source_ids=set(io23.source_roles()['train']); target_ids=set(protocol['selected_ids']); accesses=[]
    def guarded(device,sid):
        if sid not in (source_ids if device=='J-TEXT' else target_ids):
            raise AssertionError('nontraining outcome access before evaluation')
        accesses.append(dict(device=device,shot=int(sid)))
        return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guarded):
        source,_=load_source_train()
        models,mapping=prepare(parent,selected,args.output,source)
        del source
        frozen={k:state_cpu(m) for k,m in models.items()}; seals=[]
        for index,sid in enumerate(order):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal={k:s[k] for k in ('values','available')}
            predictions,updates=predict_pairs(models,transform(signal,mapping))
            predictions['raw_source']=predict_many(parent,dict(raw=signal),device='cuda')['raw']
            path=args.output/'shots'/f'{sid}.npz'; up=args.output/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**predictions);_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            if (index+1)%40==0:
                progress=dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(args.output/'prediction_receipts.json',seals)
        _durable_json(args.output/'pre_eval_label_access.json',accesses)
    # Labels are accessed for scoring only after all arms and shots are sealed.
    names=['raw_source']+[f'{k}_{mode}' for k in ARMS for mode in ('static','intra')]
    rows={k:[] for k in names}
    for sid in order:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(args.output/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['times'],s['times'])
            for name in names:
                rows[name].append(dict(shot=sid,times=s['times'][s['valid']],scores=trace[name][s['valid']],
                    is_disrupt=s['is_disrupt'],down=s['down']))
    results={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    for k,m in models.items():assert_unchanged(m,frozen[k])
    assert_unchanged(parent,original)
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('qualified archive changed')
    _durable_json(args.output/'per_shot.json',results)
    summary={k:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for k,v in results.items()}
    _durable_json(args.output/'completed.json',dict(status='COMPLETED',metrics=summary,seconds=time.monotonic()-started,
        parent_bitwise_unchanged=True,temporary_updates_discarded=True))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
