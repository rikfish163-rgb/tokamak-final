"""Frozen-model temporal-use diagnosis; never an online performance claim.

Keep the latest eight samples, and intervene ONLY on already observed history.
Reversing earlier patch order preserves contemporaneous channel rows; holding
earlier values does not. Both are artificial predictor diagnostics, not plasma
interventions or evidence that a particular physical disruption cause exists.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .redesign_20261004.intra import _batch_windows
from .redesign_20261004.run import DEFAULT_SOURCE, load_parent
from .simple_online import _durable_json, _hash

VIEWS=('identity','reverse_old_patches','hold_old_at_last')
DEFAULT_STATIC=Path(__file__).parent/'runs/redesign_static_all2953_20261004/static'


def history_view(values, available, view):
    if (view not in VIEWS or values.ndim!=3 or values.shape[1:]!=(128,23)
            or not values.is_floating_point() or available.shape!=values.shape
            or available.dtype!=torch.bool or available.device!=values.device
            or not torch.isfinite(values[available]).all()):
        raise ValueError('declared view and finite observed W128/23 boolean data required')
    result,mask=values.clone(),available.clone()
    if view=='reverse_old_patches':
        # Only complete observed eight-row patches can move. Partially padded
        # patches remain in place, so startup padding is never reinterpreted.
        full=available[:,:120].any(-1).reshape(len(values),15,8).all(-1)
        for b in range(len(values)):
            patches=torch.nonzero(full[b],as_tuple=False).flatten()
            src=patches.flip(0)
            result[b,:120].reshape(15,8,23)[patches]=values[b,:120].reshape(15,8,23)[src]
            mask[b,:120].reshape(15,8,23)[patches]=available[b,:120].reshape(15,8,23)[src]
    elif view=='hold_old_at_last':
        index=torch.arange(128,device=values.device)[None,:,None].expand_as(available)
        last=index.masked_fill(~available,-1).amax(1)
        observed=last>=0
        anchor=values.gather(1,last.clamp_min(0)[:,None]).detach()
        replace=available[:,:120] & observed[:,None]
        result[:,:120]=torch.where(replace,anchor,values[:,:120])
    return result,mask


def changed_patch_counts(before,mask,after,after_mask):
    changed=(mask!=after_mask) | ((before!=after) & (mask | after_mask))
    return changed[:,:120].reshape(len(before),15,-1).any(-1).sum(-1)


def choose_endpoints(scores, threshold):
    scores=np.asarray(scores)
    if (scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all()
            or not np.isfinite(threshold) or not 0<threshold<1):
        raise ValueError('finite nonempty archived scores and fixed threshold required')
    chosen={0,len(scores)-1,*range(0,len(scores),500)}
    chosen.update(i for i in (5,25,127) if i<len(scores))
    hits=np.flatnonzero(scores>=threshold)
    if len(hits):chosen.add(int(hits[0]))
    return sorted(chosen)


def measure_views(model, values, mask):
    # Same differentiable evaluation path for reference and interventions.
    # A read-only risk-head pre-hook does not modify the Transformer backend.
    flags=[(p,p.requires_grad) for p in model.parameters()]
    modes=[(m,m.training) for m in model.modules()]
    captured=[]
    handle=model.base.risk_head.register_forward_pre_hook(
        lambda _module,inp:captured.append(inp[0].detach().clone()))
    try:
        model.eval()
        for p,_ in flags:p.requires_grad_(False)
        outputs={}
        with torch.inference_mode(False),torch.enable_grad():
            for view in VIEWS:
                x,a=history_view(values,mask,view)
                fast=model.initial_fast(len(values)).detach().requires_grad_(True)
                out=model(x,a,fast)
                h=captured[-1]
                if (h.shape!=(len(values),64) or not torch.isfinite(h).all()
                        or not torch.isfinite(out['logit']).all()
                        or not torch.isfinite(out['forecast']).all()):
                    raise FloatingPointError('nonfinite or unexpected frozen-model output')
                outputs[view]=dict(probability=out['logit'].detach().sigmoid().cpu().numpy(),
                    logit=out['logit'].detach().cpu().numpy(),representation=h.cpu().numpy(),
                    forecast=out['forecast'].detach().cpu().numpy(),
                    changed_patches=changed_patch_counts(values,mask,x,a).cpu().numpy())
        return outputs
    finally:
        handle.remove()
        for p,flag in flags:p.requires_grad_(flag)
        for m,mode in modes:m.training=mode


def summarize_records(records, threshold):
    groups=defaultdict(list)
    for r in records:
        groups[(r['kind'],r['view'],r['category'])].append(r)
    result={}
    for (kind,view,category),rows in groups.items():
        by_shot=defaultdict(list)
        for r in rows:by_shot[r['shot']].append(r)
        # Scheduled windows overlap: first average within a shot. No endpoint
        # is silently counted as an independent experimental replication.
        deltas=[np.mean([r['changed']-r['base'] for r in rs]) for rs in by_shot.values()]
        valid_shots=sum(any(r['changed_patches']>0 for r in rs) for rs in by_shot.values())
        result.setdefault(kind,{}).setdefault(view,{})[category]=dict(
            shots=len(by_shot),endpoints=len(rows),valid_shots=valid_shots,
            median_probability_change=float(np.median(deltas)),
            mean_probability_change=float(np.mean(deltas)),
            lost_crossings=sum(any(r['base']>=threshold and r['changed']<threshold for r in rs)
                               for rs in by_shot.values()),
            gained_crossings=sum(any(r['base']<threshold and r['changed']>=threshold for r in rs)
                                 for rs in by_shot.values()),
            warning='crossings at sampled endpoints, NOT removed whole-shot FP or rescued whole-shot TP')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static',type=Path,default=DEFAULT_STATIC)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.output.exists() or args.limit<0:p.error('new output and nonnegative limit required')
    if not torch.cuda.is_available():raise RuntimeError('local CUDA required')
    parent=json.loads((args.static.parent/'protocol.json').read_text())
    archive=json.loads((args.static/'summary.json').read_text())
    predictions=archive['metrics']['predictions']
    if args.limit:predictions=predictions[:args.limit]
    if not predictions or args.limit>archive['observed_count']:p.error('invalid archive prefix')
    torch.set_num_threads(4);torch.manual_seed(131)
    model,state,gate,_=load_parent(DEFAULT_SOURCE,'cuda',persistent_site='embedding')
    if (parent['source_sha256']!=gate['checkpoint_sha256'] or
            parent['threshold']!=float(state['threshold'])):
        raise ValueError('same qualified parent and threshold required')
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    threshold=float(state['threshold']);records=[];seals={};max_baseline_error=0.
    args.output.mkdir(parents=True)
    root=Path(__file__).parent
    files=[root/n for n in ('sequence_usage_audit.py','simple23.py','simple_data.py','io23.py','causal_input.py','simple_online.py')]
    files.extend(root/'redesign_20261004'/n for n in ('run.py','embedding_model.py','intra.py'))
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for file in files:z.write(file,str(file.relative_to(root)))
    protocol=dict(scope='RETROSPECTIVE FROZEN PREDICTOR DIAGNOSTIC; NOT ONLINE TPR/FPR',
        source_sha256=gate['checkpoint_sha256'],threshold=threshold,views=VIEWS,
        last_samples_unchanged=8,window=128,patch=8,selected_shots=len(predictions),
        archive=str(args.static.resolve()),archive_summary_sha256=_hash(args.static/'summary.json'),
        archive_protocol_sha256=_hash(args.static.parent/'protocol.json'),
        source_training=False,model_updates=False,alarm_tuning=False,
        outcome_access='previously sealed archive; evaluator only; not used by history_view or model',
        endpoint_selection='fixed 0/5/25/127, every500, last, and first archived preterminal crossing',
        baseline_probability_tolerance=1e-6,
        code_sha256={str(f.relative_to(root)):_hash(f) for f in files},
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json',protocol)
    started=time.monotonic()
    for position,pred in enumerate(predictions,1):
        sid=pred['shot'];stem=f'{position:04d}_{sid}'
        receipt=json.loads((args.static/f'{stem}_seal.json').read_text())
        path=args.static/receipt['npz']
        if receipt['shot']!=sid or _hash(path)!=receipt['sha256']:
            raise ValueError('original archived seal mismatch')
        shot=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        with np.load(path,allow_pickle=False) as z:
            if int(z['shot'])!=sid or not np.array_equal(z['times'],shot['times']):
                raise ValueError('original sampling grid changed')
            archived=z['scores'].copy()
        disrupt,down=io23.read_outcome('EAST',sid)
        if disrupt!=pred['is_disrupt']:raise ValueError('archived outcome differs')
        eligible=np.ones(len(archived),bool) if not disrupt else shot['times']<=down
        hits=np.flatnonzero(eligible & (archived>=threshold))
        first=int(hits[0]) if len(hits) else None
        lead=float(round((down-shot['times'][first])*1000,9)) if disrupt and first is not None else None
        if dict(shot=sid,is_disrupt=disrupt,alarm=first is not None,lead_ms=lead,
                timely=bool(lead is not None and lead>30))!=pred:
            raise ValueError('archived first alarm contract differs')
        category=('TP' if pred['timely'] else 'TARDY' if pred['alarm'] else 'FN') if disrupt else ('FP' if pred['alarm'] else 'TN')
        selection=np.where(eligible,archived,0.)
        ends=[e for e in choose_endpoints(selection,threshold) if eligible[e]]
        if not ends:raise ValueError('no preterminal diagnostic endpoint')
        x,a=_batch_windows(simple_data,{'values':shot['values'],'available':shot['available']},ends,model,'cuda')
        outputs=measure_views(model,x,a)
        base=outputs['identity'];error=float(np.max(np.abs(base['probability']-archived[ends])))
        max_baseline_error=max(max_baseline_error,error)
        if error>protocol['baseline_probability_tolerance']:
            raise ValueError(f'sparse reference differs from archive: {sid} {error}')
        entry=[]
        for j,end in enumerate(ends):
            for view in VIEWS[1:]:
                q=outputs[view]
                entry.append(dict(shot=sid,position=position,end=end,time=float(shot['times'][end]),
                    kind='first_alarm' if end==first else 'scheduled',category=category,
                    base=float(base['probability'][j]),changed=float(q['probability'][j]),view=view,
                    changed_patches=int(q['changed_patches'][j]),
                    logit_change=float(q['logit'][j]-base['logit'][j]),
                    representation_change=float(np.linalg.norm(q['representation'][j]-base['representation'][j])),
                    forecast_change=float(np.mean(np.abs(q['forecast'][j]-base['forecast'][j])))))
        seals[str(sid)]=receipt['sha256'];records.extend(entry)
        _durable_json(args.output/f'{stem}_diagnostic.json',dict(archived_seal_sha256=receipt['sha256'],
            baseline_max_error=error,records=entry))
        if position%100==0 or position==len(predictions):
            print(json.dumps(dict(observed=position,total=len(predictions),endpoints=len(records)//2,
                baseline_max_error=max_baseline_error,elapsed_seconds=time.monotonic()-started)),flush=True)
    if any(not torch.equal(model.state_dict()[k],v) for k,v in original.items()):
        raise RuntimeError('frozen model changed')
    if _hash(DEFAULT_SOURCE)!=gate['checkpoint_sha256']:raise RuntimeError('source checkpoint changed')
    result=dict(protocol=protocol,original_model_bitwise_unchanged=True,seals=seals,
        max_baseline_probability_error=max_baseline_error,elapsed_seconds=time.monotonic()-started,
        summary=summarize_records(records,threshold),records=records)
    _durable_json(args.output/'result.json',result)
    print(json.dumps(dict(completed=True,first_alarm=result['summary'].get('first_alarm',{})),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
