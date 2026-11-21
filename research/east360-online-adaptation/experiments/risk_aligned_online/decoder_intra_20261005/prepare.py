"""One source-only anchored forecast-head refit; unchanged source risk model.

Future-dependent activity bins are OFFLINE sampling, never EAST update gates.
The whole qualified classifier is frozen. Cached features are computed with
zero FFN delta; head selection is by a source-only signal objective, not EAST.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import simple_data,io23
from ..simple_online import _durable_json,_hash
from ..redesign_20261004.intra import _batch_windows
from ..redesign_20261004.prepare_prior import (
    load_parent,load_source_train,classifier_state,assert_classifier_unchanged,DEFAULT_PARENT,
)
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.source import split_shots
from ..ffn_intra_20261005.prepare import DEFAULT_PRIOR
from ..redesign_20261004.run import load_parent as load_prior_parent
from .model import last_observation
from .source import (source_origins,activity_thresholds,sampling_pools,
                     sample_indices,normalized_level_loss)


def cache_events(model,shots):
    """Cache fixed-source tokens for this linear-head diagnostic, not online use."""
    rows=[];pieces={k:[] for k in ('features','anchor','seen','target','mask')}
    excluded=0;device=model.center.device
    for shot in shots:
        origins=source_origins(shot)
        if not len(origins):
            continue
        x,a=_batch_windows(simple_data,{k:shot[k] for k in ('values','available')},origins,model,device)
        captured=[]
        handle=model.base.forecast_head.register_forward_pre_hook(
            lambda module,args:captured.append(args[0].detach()))
        try:
            out=model(x,a,model.initial_delta())
        finally:
            handle.remove()
        if len(captured)!=1:
            raise RuntimeError('one direct forecast head invocation required')
        del out
        anchor,seen=last_observation(x,a,model.center,model.scale)
        raw=torch.as_tensor(np.stack([shot['values'][o+1:o+26] for o in origins]),device=device)
        mask=torch.as_tensor(np.stack([shot['available'][o+1:o+26] for o in origins]),device=device)
        mask=mask & torch.isfinite(raw)
        target=torch.asinh((torch.where(mask,raw,model.center)-model.center)/model.scale)
        common=mask & seen[:,None]
        activity=normalized_level_loss(anchor[:,None].expand_as(target),target,common)
        valid=common.flatten(1).any(1)
        for j,origin in enumerate(origins):
            if not bool(valid[j]):
                excluded+=1
                continue
            rows.append(dict(shot=int(shot['shot']),label=int(shot['is_disrupt']),
                             origin=int(origin),activity=float(activity[j])))
            for key,tensor in zip(pieces,(captured[0],anchor,seen,target,mask)):
                pieces[key].append(tensor[j].detach().cpu())
    if not rows:
        raise ValueError('no eligible source events with real anchors and mature targets')
    return {k:torch.stack(v) for k,v in pieces.items()},rows,excluded


def report(head,cache,rows,thresholds,*,anchored):
    values=[];common_values=[];per_channel=[]
    with torch.no_grad():
        for start in range(0,len(rows),256):
            sl=slice(start,start+256)
            prediction=head(cache['features'][sl]).reshape(-1,25,23)
            if anchored:
                prediction=prediction+cache['anchor'][sl,None]
            values.extend(normalized_level_loss(prediction,cache['target'][sl],cache['mask'][sl]).tolist())
            common=cache['mask'][sl] & cache['seen'][sl,None]
            common_values.extend(normalized_level_loss(prediction,cache['target'][sl],common).tolist())
            loss=torch.nn.functional.smooth_l1_loss(prediction,cache['target'][sl],reduction='none')
            per_channel.append((loss*common).sum(1)/common.sum(1).clamp_min(1))
    groups={}
    for key,pool in sampling_pools(rows,thresholds).items():
        idx=[i for entries in pool.values() for i in entries]
        groups[f'{key[0]}_{key[1]}']=dict(shots=len(pool),events=len(idx),
            loss=float(np.mean([np.mean([values[i] for i in entries]) for entries in pool.values()])),
            common_anchor_loss=float(np.mean([np.mean([common_values[i] for i in entries]) for entries in pool.values()])))
    ordinary=[]
    for label in (0,1):
        ids={r['shot'] for r in rows if r['label']==label}
        ordinary.append(float(np.mean([np.mean([values[i] for i,r in enumerate(rows) if r['shot']==sid]) for sid in ids])))
    channels=torch.cat(per_channel).cpu()
    valid=(cache['mask'] & cache['seen'][:,None]).any(1).cpu()
    return dict(balanced_class_activity_loss=float(np.mean([g['loss'] for g in groups.values()])),
        class_balanced_ordinary_loss=float(np.mean(ordinary)),strata=groups,
        common_anchor_channel_event_mean={name:float(channels[:,i][valid[:,i]].mean()) if valid[:,i].any() else None
                                          for i,name in enumerate(io23.RAW23)},
        weighting='main: class/activity/shot/event uniform; channel diagnostic: eligible-event uniform')


def risk_probe(model,shot):
    # Match online input access even in this offline immutability test: metadata
    # and outcome scalars are not tensors/windows and must never be sliced.
    signals={k:shot[k] for k in ('values','available')}
    x,a=_batch_windows(simple_data,signals,[0,min(128,len(shot['times'])-1)],model,model.center.device)
    return model(x,a,model.initial_delta())['logit'].detach()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seed',type=int,default=131)
    args=p.parse_args()
    if args.steps<1 or args.batch_size<1:p.error('positive steps and batch required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);started=time.monotonic()
    base,state=load_parent(DEFAULT_PARENT)
    base=base.to(args.device);before=classifier_state(base)
    original={k:v.detach().clone() for k,v in base.forecast_head.state_dict().items()}
    parent_hash=_hash(DEFAULT_PARENT);prior_hash=_hash(DEFAULT_PRIOR)
    source,ids=load_source_train();fit,selection=split_shots(source,args.seed)
    model=FfnModel(base)
    fit_cache,fit_rows,fit_excluded=cache_events(model,fit)
    selection_cache,selection_rows,selection_excluded=cache_events(model,selection)
    thresholds=activity_thresholds(fit_rows)
    pools=sampling_pools(fit_rows,thresholds)
    # Validation bins use FIT thresholds, never fit their own activity scale.
    sampling_pools(selection_rows,thresholds)
    protocol=dict(source_only=True,parent_sha256=parent_hash,finite_prior_sha256=prior_hash,
        decoder_kind='last_observation_plus_direct_increment',source_train_ids=ids,
        fit_ids=[s['shot'] for s in fit],selection_ids=[s['shot'] for s in selection],
        fit_events=fit_rows,selection_events=selection_rows,
        excluded_anchorless_events=dict(fit=fit_excluded,selection=selection_excluded),
        eligible_shots=dict(fit=len({r['shot'] for r in fit_rows}),selection=len({r['shot'] for r in selection_rows})),
        sampling='class 1/2; activity bin 1/2; shot uniform; event uniform',
        activity_thresholds=thresholds,activity='masked persistence SmoothL1, source-asinh coordinates',
        activity_threshold_fit='shot-balanced 75th percentile within source fit class only',
        event_clock='0,26,...; H25 maturity; first affected point o+26 must exist',
        disruptive_source_cutoff='first affected point strict lead >30ms, not classifier relabeling',
        source_classifier_frozen=True,source_scaler_frozen=True,original_source_labels_unchanged=True,
        no_east_reads=True,seed=args.seed,steps=args.steps,batch_size=args.batch_size,lr=.0003,
        objective='only channel-balanced masked SmoothL1 forecast level; no risk or difference term',
        online_comparison='fixed level + .25 difference; normalized step .1 radius1 stride26 unchanged',
        selected_by='minimum source selection class/activity/shot-balanced forecast loss, including epoch0',
        hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
        concurrent_gpu_run='fixed2953; timings are throughput logs, not real-time benchmarks',
        code_sha256={f.name:_hash(f) for f in Path(__file__).parent.glob('*.py')},
        interpretation='auxiliary-head development holdout; all canonical TRAIN previously seen by qualified source')
    _durable_json(args.output/'protocol.json',protocol)
    fit_cache={k:v.to(args.device) for k,v in fit_cache.items()}
    selection_cache={k:v.to(args.device) for k,v in selection_cache.items()}
    head=base.forecast_head
    references={'parent_original':report(head,selection_cache,selection_rows,thresholds,anchored=False)}
    prior_parent,_,_,_=load_prior_parent(DEFAULT_PARENT,args.device,persistent_site='embedding',prior=DEFAULT_PRIOR)
    references['finite_fast128_prior']=report(prior_parent.base.forecast_head,selection_cache,selection_rows,thresholds,anchored=False)
    del prior_parent
    with torch.no_grad():head.weight.zero_();head.bias.zero_()
    head.requires_grad_(True)
    initial=report(head,selection_cache,selection_rows,thresholds,anchored=True)
    references['persistence']=initial
    best_loss=initial['balanced_class_activity_loss'];best_step=0
    best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    history=[dict(step=0,selection=initial)]
    optimizer=torch.optim.Adam(head.parameters(),lr=.0003)
    for step in range(1,args.steps+1):
        indices=sample_indices(pools,rng,args.batch_size)
        prediction=head(fit_cache['features'][indices]).reshape(-1,25,23)+fit_cache['anchor'][indices,None]
        loss=normalized_level_loss(prediction,fit_cache['target'][indices],fit_cache['mask'][indices]).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step%100==0 or step==args.steps:
            result=report(head,selection_cache,selection_rows,thresholds,anchored=True)
            if result['balanced_class_activity_loss']<best_loss:
                best_loss=result['balanced_class_activity_loss'];best_step=step
                best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            history.append(dict(step=step,training_loss=float(loss.detach()),gradient_norm=float(norm),selection=result))
            print(json.dumps(dict(stage='anchored_head_only',step=step,selection_loss=result['balanced_class_activity_loss'],
                                  best_step=best_step,elapsed_seconds=time.monotonic()-started)),flush=True)
            _durable_json(args.output/'training.json',dict(references=references,history=history))
    head.load_state_dict(best);head.requires_grad_(False)
    chosen=report(head,selection_cache,selection_rows,thresholds,anchored=True)
    assert_classifier_unchanged(base,before)
    if _hash(DEFAULT_PARENT)!=parent_hash or _hash(DEFAULT_PRIOR)!=prior_hash:
        raise RuntimeError('immutable source/prior artifact changed')
    # Restore the old head just for exact risk-path equality on cached source
    # input windows; the fitted head has no direct connection to the risk output.
    s=selection[0];risk_new=risk_probe(model,s)
    head.load_state_dict(original)
    risk_old=risk_probe(model,s)
    if not torch.equal(risk_new,risk_old):raise RuntimeError('head refit altered static risk output')
    head.load_state_dict(best)
    torch.save(dict(forecast_head=best,protocol=protocol,selected_step=best_step),args.output/'decoder.pt')
    _durable_json(args.output/'summary.json',dict(selected_step=best_step,references=references,selected=chosen,
        classifier_unchanged=True,static_risk_probe_bit_exact=True,parent_sha256=parent_hash,
        decoder_sha256=_hash(args.output/'decoder.pt'),elapsed_seconds=time.monotonic()-started,
        high_activity_beats_persistence=all(chosen['strata'][f'{c}_1']['common_anchor_loss']<
            initial['strata'][f'{c}_1']['common_anchor_loss'] for c in (0,1)),
        risk_adaptation_evidence='not yet evaluated; forecast competence alone is not promotion'))


if __name__=='__main__':main()
