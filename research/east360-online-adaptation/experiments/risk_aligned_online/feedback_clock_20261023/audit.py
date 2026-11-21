"""Independent full-trace scoring plus sampled CUDA and timing audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from .learning import predict_clock,update_counts


def exposure(times,down,counts):
    times=np.asarray(times)
    if (times.ndim!=1 or not len(times) or not np.isfinite(times).all()
            or (np.diff(times)<=0).any() or not np.isfinite(down)
            or any(type(c) is not int or not 0<c<len(times) for c in counts)
            or counts!=sorted(set(counts))):
        raise ValueError('finite increasing times and distinct valid update boundaries required')
    lead=np.round((down-times)*1000,9);eligible=np.flatnonzero(lead>30)
    last=int(eligible[-1]) if len(eligible) else None
    applied=[c for c in counts if last is not None and c<=last]
    return dict(last_timely_index=last,updates_in_30_125ms=int(sum(30<lead[c]<=125 for c in counts)),
        last_update_count=applied[-1] if applied else None,
        last_update_age_ms=float((times[last]-times[applied[-1]-1])*1000) if applied else None)


def audit(root,replay):
    read=lambda p:json.loads(p.read_text())
    p=read(root/'protocol.json');done=read(root/'completed.json');ref=Path(p['reference'])
    assert done['status']=='COMPLETED'
    assert sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for path,h in p['code_sha256'].items():assert sha(io23.ROOT/path)==h,path
    for f,key in (('protocol.json','reference_protocol_sha256'),('completed.json','reference_completed_sha256'),
                  ('warm.pt','reference_warm_sha256'),('input_map.json','reference_mapping_sha256'),
                  ('prediction_receipts.json','reference_receipts_sha256')):
        assert sha(ref/f)==p[key]
    assert read(root/'pre_eval_label_access.json')==[]
    assert read(root/'input_map.json')==read(ref/'input_map.json')
    full=read(root/'per_shot.json');seals=read(root/'prediction_receipts.json');ids=p['contract']['shot_ids']
    assert [x['shot'] for x in seals]==ids and set(full)==set(p['arms'])
    counts={a:dict(tp=0,fp=0,d=0,s=0,leads=[],early=0) for a in p['arms']}
    totals={a:dict(updates=0,nonzero=0,loss_before=[],loss_after=[],path_length=0.) for a in p['arms'] if a!='static'}
    timing=[]
    for index,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as old:
            np.testing.assert_array_equal(z['static'],old['static'])
            np.testing.assert_array_equal(z['clock128_four'],old['warm_intra'])
            for a in p['arms']:
                row=recount(sid,z['times'],z[a],d,down,p['threshold']);assert row==full[a]['predictions'][index]
                c=counts[a];c['d' if d else 's']+=1;c['tp']+=row['timely'];c['fp']+=not d and row['alarm']
                if row['timely']:c['leads'].append(row['lead_ms']);c['early']+=row['lead_ms']>3000
                np.testing.assert_array_equal(z[a][:256],z['static'][:256])
            for a,records in read(up).items():
                interval=128 if a.startswith('clock128') else 25;segments=4 if a.endswith('four') else 1
                expected=update_counts(len(z['times']),interval)
                assert [r['first_affected_prediction'] for r in records]==expected
                for r,cnt in zip(records,expected):
                    assert r['last_observation']==cnt-1 and r['learning_rate']==.001*interval/128
                    assert r['target_segments']==segments and r['clock_interval']==interval
                    assert r['forecast_origins']==[cnt-1-25*(j+1) for j in range(segments)]
                    assert max(r['forecast_origins'])+25==cnt-1
                    assert r['parameter_delta_norm']<=r['learning_rate']*1.001
                    totals[a]['updates']+=r['gradient_steps'];totals[a]['nonzero']+=r['parameter_delta_norm']>0
                    totals[a]['path_length']+=r['parameter_delta_norm']
                    if r['gradient_steps']:
                        totals[a]['loss_before'].append(r['loss_before']);totals[a]['loss_after'].append(r['loss_after'])
            if d:
                baseline=full['static']['predictions'][index]
                timing.append(dict(shot=sid,static_category='timely' if baseline['timely'] else 'late' if baseline['alarm'] else 'never',
                    clocks={str(interval):exposure(z['times'],down,update_counts(len(z['times']),interval)) for interval in (128,25)}))
    for a,c in counts.items():
        m=done['metrics'][a];assert c['d']==c['s']==180
        assert c['tp']==m['tp'] and c['fp']==m['fp'] and c['early']==m['very_early_over3000']
        assert c['tp']/180==m['tpr'] and c['fp']/180==m['fpr'] and m['J']==m['tpr']-m['fpr']
        assert float(np.median(c['leads']))==m['median_lead_ms']
    summary={}
    for a,v in totals.items():
        before=np.asarray(v.pop('loss_before'));after=np.asarray(v.pop('loss_after'))
        summary[a]=dict(v,mean_fit_before=float(before.mean()),mean_fit_after=float(after.mean()),
                        fit_decreased=int((after<before).sum()))
    replayed=[]
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131)
        base,_=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda()
        model.load_state_dict(torch.load(ref/'warm.pt',map_location='cpu',weights_only=True)['model'])
        model.set_stage('frozen');frozen=state_cpu(model);mapping=read(root/'input_map.json')
        for sid in (ids[0],ids[len(ids)//2],ids[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping);actual=read(root/'shots'/f'{sid}_updates.json')
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as saved:
                for interval in (128,25):
                    values,records=predict_clock(model,signal,interval=interval)
                    for name in values:
                        arm=f'clock{interval}_{name}';np.testing.assert_array_equal(values[name],saved[arm])
                        assert records[name]==actual[arm]
            replayed.append(dict(shot=sid,arms=4,max_abs_error=0.,update_receipts_exact=True))
        assert_unchanged(model,frozen)
    time_summary={}
    for group in ('timely','late','never'):
        rs=[r for r in timing if r['static_category']==group];time_summary[group]={}
        for interval in (128,25):
            t=[r['clocks'][str(interval)] for r in rs];ages=[v['last_update_age_ms'] for v in t if v['last_update_age_ms'] is not None]
            time_summary[group][str(interval)]=dict(shots=len(t),no_update_30_125=sum(v['updates_in_30_125ms']==0 for v in t),
                median_age_ms=float(np.median(ages)) if ages else None,p90_age_ms=float(np.quantile(ages,.9)) if ages else None)
    j=lambda a:done['metrics'][a]['J']
    out=dict(status='VERIFIED',protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        audit_code_sha256=sha(Path(__file__)),all_shot_traces_recounted=len(ids)*len(p['arms']),
        static_reused_exact=True,old_intra_replayed_exact=True,updates=summary,
        paired_vs_static={a:paired_flips(full['static']['predictions'],full[a]['predictions']) for a in full if a!='static'},
        factorial_J=dict(latest_at128=j('clock128_latest')-j('clock128_four'),latest_at25=j('clock25_latest')-j('clock25_four'),
            faster_four=j('clock25_four')-j('clock128_four'),faster_latest=j('clock25_latest')-j('clock128_latest')),
        timing=time_summary,sampled_cuda_replays=replayed,full_independent_reinference=False)
    _durable_json(root/'timing_exposure.json',timing);_durable_json(root/'verification.json',out)
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();audit(args.root,args.replay)
