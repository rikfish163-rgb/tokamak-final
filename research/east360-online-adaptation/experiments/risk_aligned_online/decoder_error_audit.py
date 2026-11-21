"""Posthoc forensic gradients for the ACTUAL ended-EAST learned decoder.

Rebuild the changing first30 decoder using only each already-sealed historical
outcome. Freeze the exact pre-shot decoder, then replay selected errors. Labels
are used only to explain discarded numerical interventions, NEVER to modify
the original causal update. This is not a policy or a plasma-causality test.
"""
import argparse
from collections import deque
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .channel_gradient_audit import measure_event
from .continual_decoder import RollingDecoder, head_digest
from .ended_decoder import DecoderFfnModel
from .ended_density_replay import EndedDensityReplay
from .redesign_20261004 import intra
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .verify_operational_memory import load_curve


def diagnostic_anchors(shot,left,right,threshold):
    times=np.asarray(shot['times']);valid=np.asarray(shot['valid']);left=np.asarray(left);right=np.asarray(right)
    if (times.ndim!=1 or valid.shape!=times.shape or valid.dtype!=bool
            or left.shape!=times.shape or right.shape!=times.shape
            or not np.isfinite(times).all() or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.isfinite(threshold) or not 0<threshold<1):
        raise ValueError('matching finite sealed curves and boolean valid time axis required')
    eligible=valid.copy()
    if shot['is_disrupt']:
        if shot['down'] is None or not np.isfinite(shot['down']):raise ValueError('finite down time required')
        eligible &= np.round((shot['down']-times)*1000.,9)>30.
    indices=np.flatnonzero(eligible)
    if not len(indices):raise ValueError('no eligible diagnostic point; not deleting this shot')
    out={}
    for name,scores in (('static',left),('candidate',right)):
        above=indices[scores[indices]>=threshold]
        query=int(above[0]) if len(above) else int(indices[np.argmax(scores[indices])])
        kind=('first_timely_alarm' if shot['is_disrupt'] else 'first_false_alarm') if len(above) else (
            'eligible_peak_without_timely_alarm' if shot['is_disrupt'] else 'eligible_peak_without_false_alarm')
        out[name]=dict(query=query,kind=kind,time=float(times[query]),probability=float(scores[query]))
    return out


def preceding_updates(attempts,query,*,count=2):
    if type(query) is not int or query<0 or type(count) is not int or count<1:
        raise ValueError('existing diagnostic query and positive event budget required')
    return [r['origin_index'] for r in attempts if r['accepted'] and r['effective_index'] is not None
            and r['effective_index']<=query][-count:]


def reconstruct_heads(model,folder,order,needed,threshold):
    provider=EndedDensityReplay(model.center.cpu().numpy(),model.scale.cpu().numpy(),
        output_dir=folder/'stream',order=[r['shot'] for r in order],preserve_observed_channels=True)
    learner=RollingDecoder(model,mode='early_fixed');history=deque(maxlen=32)
    heads={};records=[];steps=0
    for index,item in enumerate(order[:30],1):
        if index in needed:
            heads[index]={k:v.detach().cpu().clone() for k,v in model.decoder.state_dict().items()}
        shot=provider.read_shot('EAST',item['shot'])
        scores,times,audit=load_curve(folder,index,item['shot'])
        if not np.array_equal(times,shot['times']):raise AssertionError('prefix signal/seal identity differs')
        old=audit['persistent']['head']
        if old.get('head_before_sha256') is not None and head_digest(model)!=old['head_before_sha256']:
            raise AssertionError('wrong pre-shot auxiliary head')
        # Read the HISTORICAL outcome only after its full archived seal was
        # checked. This reconstructs a completed run; no new seal is written.
        ended=provider.attach_outcome(shot,'EAST',{});ended['_sealed_scores']=scores
        receipt=learner(model,ended,history,provider=provider,threshold=threshold,device='cuda')
        if (receipt['actual_steps']!=old['actual_steps'] or receipt['accepted']!=old['accepted']
                or receipt.get('head_after_sha256')!=old.get('head_after_sha256')):
            raise AssertionError('historical decoder reconstruction is not bitexact')
        steps+=receipt['actual_steps'];history.append(ended)
        records.append(dict(index=index,shot=item['shot'],actual_steps=receipt['actual_steps'],
                            head_sha256=head_digest(model),matches_archived_head=True))
        if index%5==0:print(json.dumps(dict(reconstructed_head_through=index)),flush=True)
    final=torch.load(folder/'final_state.pt',map_location='cpu',weights_only=True)
    for name,value in model.decoder.state_dict().items():
        torch.testing.assert_close(value.cpu(),final['decoder'][name],rtol=0,atol=0)
    if model.memory.any() or steps!=152 or learner.optimizer_steps!=40:
        raise AssertionError('changed persistent state or wrong historical learning schedule')
    heads[31]={k:v.detach().cpu().clone() for k,v in model.decoder.state_dict().items()}
    return heads,provider.statistics,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new forensic output required')
    verified=json.loads((args.run/'independent_verification.json').read_text())
    if verified['verification']!='PASS' or verified['source_sha256']!=PARENT_SHA256:
        raise AssertionError('completed matched240 reference required')
    folder=args.run/'intra_control';static=args.run/'mapped_static'
    protocol=json.loads((folder/'protocol.json').read_text());threshold=protocol['threshold']
    if not protocol['auxiliary_head_frozen_after30'] or protocol['persistent_enabled']:
        raise AssertionError('forensic reconstruction requires the declared frozen-after30, zero-memory candidate')
    order=json.loads((folder/'order.json').read_text());indices={r['shot']:i+1 for i,r in enumerate(order)}
    groups=verified['versus_mapped_static']
    cases=[(sid,group) for group in ('lost_disruptive','new_false_alarm') for sid in groups[group]]
    # A small explicit beneficial comparison, not a representative population
    # sample: last rescued D and last removed FP, both post30 frozen phase.
    cases += [(groups['rescued_disruptive'][-1],'rescued_disruptive'),
              (groups['removed_false_alarm'][-1],'removed_false_alarm')]
    torch.set_num_threads(4);torch.manual_seed(131)
    base,state=load_parent(DEFAULT_PARENT);model=DecoderFfnModel(base).to('cuda').eval().requires_grad_(False)
    source={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json',dict(scope='POSTHOC SELECTED ERROR FORENSICS, NOT ONLINE POLICY',
        reference=str(args.run.resolve()),verification_sha256=_hash(args.run/'independent_verification.json'),
        parent_sha256=PARENT_SHA256,code_sha256=_hash(Path(__file__)),threshold=threshold,
        cases=[dict(shot=s,index=indices[s],category=g) for s,g in cases],
        events_per_anchor=2,changing_prefix_heads_reconstructed=True,current_labels_only_explanation=True,
        oracle_never_deployed=True,no_new_training_rule=True,no_physical_causality_claim=True))
    heads,statistics,reconstruction=reconstruct_heads(model,folder,order,
        needed={indices[s] for s,g in cases if indices[s]<=30},threshold=threshold)
    _durable_json(args.output/'head_reconstruction.json',dict(records=reconstruction,
        cumulative_steps=152,final_adam_steps=40,all_historical_heads_bitexact=True))
    provider=EndedDensityReplay(state['source_center'],state['source_scale'],output_dir=folder/'stream',
        order=[r['shot'] for r in order],preserve_observed_channels=True)
    original_attempt=intra._attempt;rows=[]
    for sid,group in cases:
        index=indices[sid];model.decoder.load_state_dict(heads[index if index<=30 else 31]);model.requires_grad_(False)
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        provider.statistics=statistics if index>30 else None
        shot=provider.reencode(raw);ended=simple_data.attach_outcome(shot,'EAST',{})
        expected,times,audit=load_curve(folder,index,sid);left,left_times,_=load_curve(static,index,sid)
        if not np.array_equal(times,shot['times']) or not np.array_equal(times,left_times):
            raise AssertionError('forensic curve times differ')
        anchors=diagnostic_anchors(ended,left,expected,threshold)
        origins=sorted(set(o for anchor in anchors.values() for o in preceding_updates(audit['attempts'],anchor['query'])))
        events=[]

        def explained(model,pending,signals,times,fast,step,radius,objective,device,previous=None):
            updated,receipt=original_attempt(model,pending,signals,times,fast,step,radius,objective,device,previous=previous)
            origin=receipt['origin_index'];query=receipt['maturity_index']+1
            if origin in origins:
                if not receipt['accepted'] or query>=len(times) or not ended['valid'][query]:
                    raise AssertionError('selected archived mature update no longer qualifies')
                if ended['is_disrupt'] and round((ended['down']-times[query])*1000.,9)<=30.:
                    raise AssertionError('operational derivative must remain timely eligible')
                _,x,a,_=pending;maturity=receipt['maturity_index']
                truth=torch.as_tensor(signals['values'][origin+1:maturity+1],device=device,dtype=model.center.dtype)[None]
                observed=torch.as_tensor(signals['available'][origin+1:maturity+1],device=device)[None]
                qx,qa=intra._batch_windows(simple_data,signals,[query],model,device)
                event=measure_event(model,x,a,truth,observed,qx,qa,fast,int(ended['is_disrupt']),step=step,radius=radius)
                if (event['actual']['candidate_sha256']!=model.summarize_fast(updated)['sha256']
                        or abs(event['actual']['signal_gain']-receipt['loss_gain'])>2e-6):
                    raise AssertionError('forensic candidate differs from actual causal update')
                event.update(origin=origin,query=query,category=group,
                    operational_label=int(ended['is_disrupt']),point_label=float(ended['y'][query]),
                    lead_ms=None if not ended['is_disrupt'] else round((ended['down']-times[query])*1000.,9),
                    fast_norm_before=receipt['fast_norm_before'],fast_norm_after=receipt['fast_norm_after'],
                    used_by_anchors=[name for name,anchor in anchors.items()
                        if origin in preceding_updates(audit['attempts'],anchor['query'])])
                events.append(event)
            return updated,receipt

        with patch.object(intra,'_attempt',explained):
            actual,replayed=intra.predict_shot(model,shot,intra=True,device='cuda',data_api=provider)
        error=float(np.max(np.abs(actual-expected)))
        if error!=0. or [r['origin'] for r in events]!=origins:
            raise AssertionError('forensic replay changed the archived trajectory or omitted selected events')
        for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),frozen[k],rtol=0,atol=0)
        row=dict(shot=sid,index=index,category=group,head_sha256=head_digest(model),anchors=anchors,
            chosen_origins=origins,measured_events=len(events),curve_max_error=error,
            actual_accepted_updates=replayed['accepted_updates'],events=events,
            no_update_before_anchor={name:not preceding_updates(audit['attempts'],anchor['query'])
                                     for name,anchor in anchors.items()})
        _durable_json(args.output/f'{sid}_diagnostic.json',row);rows.append(row)
        print(json.dumps({k:row[k] for k in ('shot','index','category','measured_events','curve_max_error')}),flush=True)
    for k,v in model.base.state_dict().items():torch.testing.assert_close(v.cpu(),source[k],rtol=0,atol=0)
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256 or model.memory.any():raise AssertionError('source or memory changed')
    flat=[e for r in rows for e in r['events']]
    summary=dict(selected_shots=len(rows),actual_curves_reproduced=len(rows),measured_updates=len(flat),
        operational_risk_worsened=sum(e['actual']['risk_change']>0 for e in flat),
        local_signal_and_risk_improvement_available=sum(e['plain_bounded_weight_oracle']['risk_change']<0
            and e['plain_bounded_weight_oracle']['would_pass_original_signal_acceptance'] for e in flat),
        source_and_memory_unchanged=True,all_curve_max_error=0.,historical_heads_bitexact=True,
        groups={g:dict(shots=sum(r['category']==g for r in rows),
            events=sum(e['category']==g for e in flat),
            operational_risk_worsened=sum(e['category']==g and e['actual']['risk_change']>0 for e in flat))
            for g in sorted({r['category'] for r in rows})},
        dominant_gradient_feature_counts={name:sum(max(e['channels'],key=lambda c:c['gradient_norm'])['feature']==name
            for e in flat) for name in io23.RAW23},
        no_population_harm_rate=True,not_physical_causal_effect=True,
        no_new_policy_deployed=True,oracle_used_hidden_labels=True)
    _durable_json(args.output/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
