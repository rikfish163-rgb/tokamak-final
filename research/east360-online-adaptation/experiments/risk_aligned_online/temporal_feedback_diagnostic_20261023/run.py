"""POSTHOC timing-conditioned audit of the exact sealed feedback experiment.

Samples FIRST/MIDDLE/LAST update per shot, without outcome-based selection.
Future observations/outcomes enter only discarded diagnostic gradients.
The actual trajectory is rebuilt exclusively with the original signal-only
update and checked against every saved update receipt. No policy is trained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback,initial_encoder
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..gradient_forensics_20261022.run import gradient_geometry
from ..lag_bias_feedback_20261022.learning import adapt_arrived,split_residuals,feedback_loss
from ..lag_bias_feedback_20261022.run import MAP_PATH
from ..warmstart_alignment_20261022.alignment import transform


def timing_record(times,scores,valid,disruptive,down,threshold):
    times=np.asarray(times);scores=np.asarray(scores);valid=np.asarray(valid)
    if (times.ndim!=1 or scores.shape!=times.shape or valid.shape!=times.shape
            or valid.dtype!=bool or not np.isfinite(times).all() or not np.isfinite(scores).all()
            or np.any(np.diff(times)<0) or not np.isfinite(threshold)):
        raise ValueError('matching finite ordered trace and boolean validity required')
    ix=np.flatnonzero(valid & (scores>=threshold));first=int(ix[0]) if len(ix) else None
    lead=None
    if not disruptive: category='safe_fp' if first is not None else 'safe_tn'
    elif first is None: category='never'
    else:
        if down is None or not np.isfinite(down):raise ValueError('disruption time required')
        lead=round(float(down-times[first])*1000.,9)
        if lead<0:raise ValueError('postterminal prediction in a valid disruption trace')
        category=next(name for upper,name in ((10,'late_0_10'),(25,'late_10_25'),
            (30,'late_25_30'),(125,'timely_30_125'),(3000,'timely_125_3000'),
            (float('inf'),'early_over3000')) if lead<=upper)
    bins={}
    if disruptive:
        leads=np.round((down-times)*1000,9)
        for lo,hi in ((0,10),(10,25),(25,30),(30,50),(50,125),(125,500),(500,float('inf'))):
            selected=valid & (leads>=0 if lo==0 else leads>lo) & (leads<=hi)
            bins[f'{lo}_{hi}']=dict(observed=int(selected.sum()),
                peak=float(scores[selected].max()) if selected.any() else None)
    return dict(category=category,alarm_index=first,lead_ms=lead,lead_bin_peaks=bins)


def loss_components(prediction,truth,mask,bias,valid,*,correct):
    # Reuse the tested loss validator and preserve its single global denominator.
    total=feedback_loss(prediction,truth,mask,bias,valid,correct=correct)
    selected=mask[2:] & valid[None,None,:]
    residual=prediction[2:]-truth[2:].detach()-(bias if correct else 0.)
    safe=torch.where(selected,residual,0.)
    parts=F.smooth_l1_loss(safe,torch.zeros_like(safe),reduction='none').sum((0,1))/selected.sum()
    torch.testing.assert_close(parts.sum(),total,rtol=1e-6,atol=1e-7)
    return parts


def source_timing():
    roles=io23.source_roles();advanced=io23.advanced_times();rows=[]
    parent_protocol=DEFAULT_PARENT.parent/'protocol.json'
    assert json.loads(parent_protocol.read_text())['train_ids']==roles['train']
    for sid in roles['train']:
        s=simple_data.attach_outcome(simple_data.read_shot('J-TEXT',sid),'J-TEXT',advanced)
        if not s['is_disrupt']:continue
        lead=np.round((s['down']-s['times'])*1000,9);positive=s['valid'] & (s['y']>.5)
        horizon,raw,missing=io23.effective_horizon_ms(sid,advanced)
        rows.append(dict(shot=sid,horizon_ms=horizon,raw_advanced_ms=raw,missing=missing,
            observed=int(s['valid'].sum()),positive=int(positive.sum()),
            positive_gt10=int((positive & (lead>10)).sum()),positive_gt30=int((positive & (lead>30)).sum()),
            eligible_gt30=int((s['valid'] & (lead>30)).sum())))
    return dict(scope='current canonical TRAIN label contract; not reconstruction of historical random batches',
        advanced_path=str(io23.JTEXT_ADVANCED_TIME_PATH),advanced_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        parent_protocol_sha256=sha(parent_protocol),n_train=len(roles['train']),n_disrupt=len(rows),
        missing_advanced=sum(r['missing'] for r in rows),horizon_gt30=sum(r['horizon_ms']>30 for r in rows),
        positive=sum(r['positive'] for r in rows),positive_gt10=sum(r['positive_gt10'] for r in rows),
        positive_gt30=sum(r['positive_gt30'] for r in rows),
        disruptions_with_positive_gt30=sum(r['positive_gt30']>0 for r in rows),rows=rows)


def inspect_update(model,signal,outcome,count,before,after,arm,saved,threshold):
    stop=min(count+128,len(signal['values']))
    q_indices=np.arange(count,stop)
    valid=outcome['valid'][count:stop].copy()
    leads=None
    if outcome['is_disrupt']:
        leads=np.round((outcome['down']-outcome['times'][count:stop])*1000,9)
        valid &= leads>30.
    r=dict(count=count,query_stop_exclusive=stop,eligible_query_points=int(valid.sum()),
        alarm_already_emitted=bool(np.any(saved[:count][outcome['valid'][:count]]>=threshold)),
        is_disrupt=bool(outcome['is_disrupt']),arm=arm)
    if not valid.any():
        r['status']='no_future_timely_or_valid_point';return r
    # Keep the original batch32 numerical path. Labels are used ONLY by this
    # diagnostic query loss, never by adapt_arrived or a deployed selection.
    before_logits=[];after_logits=[]
    for start in range(count,stop,32):
        end=min(stop,start+32)
        windows=[simple_data.window(signal,e,128) for e in range(start,end)]
        x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
        h=model.predict_fast(x,m,before,mode='risk_state')['risk_state']
        z=model.base.risk_head(h).squeeze(-1);before_logits.append(z)
        hh=model.predict_fast(x,m,after,mode='risk_state')['risk_state']
        zz=model.base.risk_head(hh).squeeze(-1);after_logits.append(zz)
        np.testing.assert_array_equal(zz.sigmoid().detach().cpu().numpy(),saved[start:end])
    bz=torch.cat(before_logits);az=torch.cat(after_logits);v=torch.as_tensor(valid,device='cuda')
    before_peak=bz[v].max();after_peak=az[v].max();label=float(outcome['is_disrupt'])
    risk=F.binary_cross_entropy_with_logits(before_peak,before_peak.new_tensor(label))
    later=F.binary_cross_entropy_with_logits(after_peak,after_peak.new_tensor(label))
    risk_gradient=torch.autograd.grad(risk,tuple(before.values()))
    selected=int(q_indices[valid][int(bz[v].detach().argmax())])
    r.update(status='measured',query_peak_index=selected,
        query_peak_lead_ms=round(float(outcome['down']-outcome['times'][selected])*1000,9) if label else None,
        point_label125_at_query=float(outcome['y'][selected]),
        before_peak_probability=float(before_peak.sigmoid().detach()),
        actual_after_peak_probability=float(after_peak.sigmoid().detach()),
        actual_risk_change=float((later-risk).detach()),operational_label=label,
        point_and_operational_label_disagree=bool(outcome['y'][selected]!=label))
    x,m,target,tm,origins=forecast_examples(signal,count,model.cfg,'cuda')
    order=np.argsort(origins).tolist();x,m,target,tm=[a[order] for a in (x,m,target,tm)]
    tm=tm & torch.isfinite(target)
    truth=torch.asinh((torch.where(tm,target,model.base.center)-model.base.center)/model.base.scale)
    prediction=model.predict_fast(x,m,before)['forecast'];bias,eligible=split_residuals(prediction,truth,tm)
    candidates={}
    for correct in (False,True):
        pieces=loss_components(prediction,truth,tm,bias,eligible,correct=correct)
        total=pieces.sum();grads=torch.autograd.grad(total,tuple(before.values()),retain_graph=True)
        norm=float(sum(g.square().sum() for g in grads).sqrt());step=.001/max(norm,1.)
        components=[];summed=[torch.zeros_like(g) for g in grads]
        for c,name in enumerate(io23.RAW23):
            cg=torch.autograd.grad(pieces[c],tuple(before.values()),retain_graph=True)
            for a,b in zip(summed,cg):a.add_(b)
            components.append(dict(feature=name,loss=float(pieces[c].detach()),
                old_bias=float(bias[c]),old_available=bool(eligible[c]),
                **gradient_geometry(cg,risk_gradient,step)))
        for a,b in zip(summed,grads):torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-6)
        candidates['lag_bias' if correct else 'recent']=dict(
            loss=float(total.detach()),**gradient_geometry(grads,risk_gradient,step),channels=components)
    r['same_state_candidates']=candidates
    return r


def summarize(cases):
    groups={}
    for arm in ('recent','lag_bias'):
        for opened in (True,False):
            for phase in ('safe','early_disruptive_gt125','actionable_30_125'):
                rs=[r for r in cases if r['status']=='measured' and r['arm']==arm
                    and (not r['alarm_already_emitted'])==opened
                    and ('safe' if not r['is_disrupt'] else 'early_disruptive_gt125'
                         if r['query_peak_lead_ms']>125 else 'actionable_30_125')==phase]
                if not rs:continue
                measured=np.array([r['actual_risk_change'] for r in rs])
                channels={}
                for method in ('recent','lag_bias'):
                    totals={c:0. for c in io23.RAW23}
                    for r in rs:
                        for c in r['same_state_candidates'][method]['channels']:
                            totals[c['feature']]+=c['predicted_risk_loss_change']
                    channels[method]=dict(sorted(totals.items(),key=lambda x:x[1],reverse=True))
                groups[f'{arm}/decision_{"open" if opened else "already_alarm"}/{phase}']=dict(
                    sampled_updates=len(rs),distinct_shots=len({r['shot'] for r in rs}),
                    risk_increased=int((measured>1e-7).sum()),risk_decreased=int((measured< -1e-7).sum()),
                    numerically_negligible=int((abs(measured)<=1e-7).sum()),
                    mean_change=float(measured.mean()),median_change=float(np.median(measured)),
                    same_state_mean_first_order_change={k:float(np.mean([r['same_state_candidates'][k]['predicted_risk_loss_change'] for r in rs])) for k in ('recent','lag_bias')},
                    summed_first_order_change_by_channel=channels)
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    verified=json.loads((args.run/'verification.json').read_text());assert verified['status']=='VERIFIED'
    protocol=json.loads((args.run/'protocol.json').read_text());order=protocol['contract']['shot_ids']
    seals=json.loads((args.run/'prediction_receipts.json').read_text())
    assert [s['shot'] for s in seals]==order
    args.output.mkdir(parents=True);start=time.monotonic()
    _durable_json(args.output/'protocol.json',dict(scope='POSTHOC all360 diagnostic, not a new deployed experiment',
        run=str(args.run.resolve()),verification_sha256=sha(args.run/'verification.json'),
        code_sha256=sha(Path(__file__)),selection='first/middle/last scheduled update per shot, independent of outcome',
        query='max risk in next128 observed endpoints; disruption points require lead>30; all safe valid points',
        local_loss='BCE of query peak with shot outcome; not point125 label; future-label diagnostic only',
        no_new_training=True,no_changed_emitted_predictions=True))
    _durable_json(args.output/'source_timing.json',source_timing())
    torch.set_num_threads(2);torch.manual_seed(131)
    base,_=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda();model.set_stage('frozen')
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    mapping=json.loads(MAP_PATH.read_text());assert sha(MAP_PATH)==protocol['density_mapping_sha256']
    timing={k:[] for k in protocol['arms']};cases=[];n_verified_updates=0
    for index,(sid,seal) in enumerate(zip(order,seals)):
        path=args.run/'shots'/f'{sid}.npz';upath=args.run/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(upath)==seal['updates_sha256']
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        outcome=simple_data.attach_outcome(raw,'EAST',{})
        with np.load(path,allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['times'],raw['times'])
            predictions={k:trace[k].copy() for k in protocol['arms']}
        for arm,values in predictions.items():
            timing[arm].append(dict(shot=sid,**timing_record(raw['times'],values,outcome['valid'],
                outcome['is_disrupt'],outcome['down'],protocol['threshold'])))
        if index<20:continue
        signal=transform({k:raw[k] for k in ('values','available')},mapping)
        expected=json.loads(upath.read_text());counts=list(range(256,len(signal['values']),128))
        if not counts:continue
        selected={counts[0],counts[len(counts)//2],counts[-1]}
        for arm in ('recent','lag_bias'):
            fast=initial_encoder(model)
            assert len(expected[arm])==len(counts)
            for j,count in enumerate(counts):
                new,r=adapt_arrived(model,signal,count,fast,device='cuda',correct=arm=='lag_bias')
                assert r==expected[arm][j];n_verified_updates+=1
                if count in selected:
                    diagnostic=inspect_update(model,signal,outcome,count,fast,new,arm,
                                              predictions[arm],protocol['threshold'])
                    diagnostic['shot']=sid;cases.append(diagnostic)
                fast=new
        if (index+1)%40==0:
            progress=dict(shots=index+1,sampled_updates=len(cases),seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),frozen[k],rtol=0,atol=0)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='COMPLETED',verified_update_receipts=n_verified_updates,
        timing_counts={k:dict(Counter(r['category'] for r in v)) for k,v in timing.items()},
        sampled_updates=len(cases),measured=sum(r['status']=='measured' for r in cases),
        summaries=summarize(cases),seconds=time.monotonic()-start,
        source_unchanged=True,independent_confirmation=False)
    _durable_json(args.output/'timing.json',timing);_durable_json(args.output/'cases.json',cases)
    _durable_json(args.output/'completed.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
