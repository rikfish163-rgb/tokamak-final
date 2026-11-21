"""Source-only FFN step screening and four-event learned signal-loss weights."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..redesign_20261004.prepare_prior import load_source_train
from ..redesign_20261004.run import load_parent, DEFAULT_SOURCE
from ..simple_online import _durable_json, _hash
from .loss import SignalRule
from .model import FfnModel
from .source import STRIDE, split_shots, make_episode, episode_loss
from .stream import predict_shot
from ..simple_train import metrics

DEFAULT_PRIOR=Path(__file__).parents[1]/'runs/redesign_finite_prior128_20261004/prior.pt'
BUDGETS=((.01,.10),(.03,.30),(.10,1.0))


def build_pairs(shots,rng,count):
    pools=[]
    for label in (False,True):
        pool=[]
        for index in rng.permutation(len(shots)):
            shot=shots[index]
            if shot['is_disrupt']==label:
                episode=make_episode(shot,rng)
                if episode is not None:
                    pool.append((shot,episode))
            if len(pool)==count:
                break
        if len(pool)!=count:
            raise ValueError('too few distinct eligible TRAIN shots for paired episodes')
        pools.append(pool)
    return [item for pair in zip(*pools) for item in pair]


def evaluate(model,episodes,*,step,radius,rule,boundary,adapt=True):
    records=[]
    for shot,episode in episodes:
        _,row=episode_loss(model,shot,episode,step=step,radius=radius,rule=rule,
                           boundary=boundary,adapt=adapt)
        records.append(row)
    return dict(mean_loss=float(np.mean([r['outer_segment_loss'] for r in records])),records=records)


def full_source(model,episodes,*,step,radius,rule,threshold,adapt=True):
    rows=[];losses={0:[],1:[]};updates=at_radius=0
    for shot,_ in episodes:
        scores,audit=predict_shot(model,shot,intra=adapt,device=model.center.device,
            stride=STRIDE,step=max(step,1e-12),radius=radius,rule=rule,offline_source=True)
        valid=shot['valid'].copy()
        if shot['is_disrupt']:
            valid &= shot['times']<=shot['down']
        rows.append(dict(shot=shot['shot'],is_disrupt=shot['is_disrupt'],down=shot['down'],
                         times=shot['times'][valid],scores=scores[valid]))
        timely=valid.copy()
        if shot['is_disrupt']:
            timely &= np.round((shot['down']-shot['times'])*1000.,9)>30.
        if timely.any():
            p=float(np.clip(scores[timely].max(),1e-7,1-1e-7))
            margin=math.log(p/(1-p))-math.log(threshold/(1-threshold))
            losses[int(shot['is_disrupt'])].append(float(np.logaddexp(0.,-margin if shot['is_disrupt'] else margin)))
        updates+=audit['accepted_updates']
        at_radius+=sum(e['delta_norm']>=radius*(1-1e-5) for e in audit['events'])
    return dict(metrics30=metrics(rows,threshold,lead=30.),metrics10=metrics(rows,threshold,lead=10.),
        balanced_loss=float(np.mean([np.mean(v) for v in losses.values()])),updates=updates,
        at_radius=at_radius,scope='auxiliary rule whole-shot SOURCE TRAIN development; not JTEXT online branch')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--prior',type=Path,default=DEFAULT_PRIOR)
    p.add_argument('--steps',type=int,default=180)
    p.add_argument('--calibration-per-class',type=int,default=24)
    p.add_argument('--validation-per-class',type=int,default=16)
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=131)
    args=p.parse_args()
    if min(args.steps,args.calibration_per_class,args.validation_per_class)<1:
        p.error('positive episode/step counts required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    parent,state,gate,prior_protocol=load_parent(args.checkpoint,args.device,persistent_site='embedding',prior=args.prior)
    model=FfnModel(parent.base)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    threshold=float(state['threshold']); boundary=math.log(threshold/(1-threshold))
    prior_hash=_hash(args.prior)
    shots,ids=load_source_train()
    train,validation=split_shots(shots,args.seed)
    calibration=build_pairs(train,rng,args.calibration_per_class)
    val=build_pairs(validation,rng,args.validation_per_class)
    protocol=dict(source_only=True,parent_sha256=gate['checkpoint_sha256'],prior_sha256=prior_hash,
        source_train_ids=ids,fit_ids=[s['shot'] for s in train],validation_ids=[s['shot'] for s in validation],
        calibration_episodes=[e for _,e in calibration],validation_episodes=[e for _,e in val],
        budgets=BUDGETS,threshold=threshold,alarm_tuning=False,steps=args.steps,
        seed=args.seed,stride=STRIDE,learning_rate=.001,outer='balanced source-segment softplus about frozen VAL threshold',
        outer_positive='source D: >30 and <=125ms; auxiliary goal NOT original source point labels',
        outer_negative='sampled source safe segment including intermediate updates, NOT whole-shot FPR',
        inner='four matured H25 signal updates, last FFN only; no outcome in inner loss',
        original_source_labels_unchanged=True,source_classifier_frozen=True,
        source_episode_coverage={name:{str(label):dict(total=sum(s['is_disrupt']==label for s in part),
            eligible=sum(make_episode(s,np.random.default_rng(args.seed)) is not None for s in part if s['is_disrupt']==label))
            for label in (False,True)} for name,part in (('fit',train),('selection',validation))},
        code_sha256={f.name:_hash(f) for f in Path(__file__).parent.glob('*.py')},
        interpretation='development screening; all source and target evidence development-exposed')
    _durable_json(args.output/'protocol.json',protocol)
    baseline=evaluate(model,calibration,step=0.,radius=1.,rule=None,boundary=boundary,adapt=False)
    whole_baseline=full_source(model,calibration,step=0.,radius=1.,rule=None,threshold=threshold,adapt=False)
    budget_results=[]
    for step,radius in BUDGETS:
        result=evaluate(model,calibration,step=step,radius=radius,rule=None,boundary=boundary)
        result['whole_source']=full_source(model,calibration,step=step,radius=radius,rule=None,threshold=threshold)
        result.update(step=step,radius=radius)
        budget_results.append(result)
        print(json.dumps(dict(stage='source_budget',step=step,radius=radius,loss=result['mean_loss'],
                              baseline_loss=baseline['mean_loss'])),flush=True)
    selected=min(budget_results,key=lambda r:r['whole_source']['balanced_loss'])
    step,radius=selected['step'],selected['radius']
    _durable_json(args.output/'budget.json',dict(step=step,radius=radius,source_only=True,
        choice='minimum balanced WHOLE-SHOT SOURCE calibration loss among three declared budgets',
        baseline=baseline,whole_baseline=whole_baseline,results=budget_results,
        any_improvement=selected['whole_source']['balanced_loss']<whole_baseline['balanced_loss']))
    rule=SignalRule().to(args.device)
    optimizer=torch.optim.Adam(rule.parameters(),lr=.001)
    fixed_val=evaluate(model,val,step=step,radius=radius,rule=None,boundary=boundary)
    validation_reports=[dict(iteration=0,**fixed_val)]
    # Initial rule is exactly uniform; save it too, so no learned improvement
    # is fabricated if every fitted candidate is worse than the fixed rule.
    best_loss=fixed_val['mean_loss'];best_iteration=0
    best={k:v.detach().cpu().clone() for k,v in rule.state_dict().items()}
    trace=[]; started=time.monotonic()
    for iteration in range(1,args.steps+1):
        pair=build_pairs(train,rng,1)
        optimizer.zero_grad(set_to_none=True)
        rows=[]
        for shot,episode in pair:
            loss,row=episode_loss(model,shot,episode,step=step,radius=radius,rule=rule,
                                  boundary=boundary,create_graph=True)
            (loss/2).backward()
            rows.append(row)
        grad=torch.nn.utils.clip_grad_norm_(rule.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        record=dict(iteration=iteration,mean_loss=float(np.mean([r['outer_segment_loss'] for r in rows])),
            rule_grad_norm=float(grad),episodes=[e for _,e in pair],elapsed_seconds=time.monotonic()-started)
        trace.append(record)
        if iteration%20==0 or iteration==args.steps:
            print(json.dumps(dict(stage='source_meta',**{k:v for k,v in record.items() if k!='episodes'})),flush=True)
        if iteration%60==0 or iteration==args.steps:
            report=evaluate(model,val,step=step,radius=radius,rule=rule,boundary=boundary)
            validation_reports.append(dict(iteration=iteration,**report))
            if report['mean_loss']<best_loss:
                best_loss=report['mean_loss'];best_iteration=iteration
                best={k:v.detach().cpu().clone() for k,v in rule.state_dict().items()}
            _durable_json(args.output/'training.json',dict(trace=trace,validation=validation_reports,
                best_iteration=best_iteration,best_loss=best_loss))
    unchanged=all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    if not unchanged or _hash(args.checkpoint)!=gate['checkpoint_sha256'] or _hash(args.prior)!=prior_hash:
        raise RuntimeError('fixed classifier/decoder or parent artifacts changed')
    protocol.update(step=step,radius=radius,best_iteration=best_iteration,
        rule_selected='initial uniform' if best_iteration==0 else 'source-heldout fitted rule',
        classifier_and_decoder_unchanged=True)
    fixed_whole=full_source(model,val,step=step,radius=radius,rule=None,threshold=threshold)
    rule.load_state_dict(best)
    learned_whole=full_source(model,val,step=step,radius=radius,rule=rule,threshold=threshold)
    static_whole=full_source(model,val,step=0.,radius=radius,rule=None,threshold=threshold,adapt=False)
    torch.save(dict(rule=best,protocol=protocol),args.output/'rule.pt')
    _durable_json(args.output/'summary.json',dict(protocol=protocol,rule_sha256=_hash(args.output/'rule.pt'),
        fixed_validation_loss=fixed_val['mean_loss'],selected_validation_loss=best_loss,
        whole_source_selection=dict(static=static_whole,fixed=fixed_whole,learned=learned_whole)))


if __name__=='__main__':
    main()
