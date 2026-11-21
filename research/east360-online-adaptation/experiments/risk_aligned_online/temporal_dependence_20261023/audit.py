"""Independent scoring and declared posthoc channel/time sensitivity analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from .controls import flatten_window,predict_controls
from .run import load_model


def eligible_peak(times,scores,disruptive,down):
    t,p=np.asarray(times),np.asarray(scores)
    if t.ndim!=1 or p.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError('matching finite one-dimensional times and probabilities required')
    valid=np.round((down-t)*1000,9)>30 if disruptive else np.ones(len(t),bool)
    indices=np.flatnonzero(valid)
    return int(indices[np.argmax(p[indices])]) if len(indices) else None


def transition_groups(raw,mapped):
    if [r['shot'] for r in raw]!=[r['shot'] for r in mapped]:raise ValueError('paired identities required')
    result=dict(lost_timely=[],removed_fp=[],other=[])
    for a,b in zip(raw,mapped):
        if a['is_disrupt']!=b['is_disrupt']:raise ValueError('paired outcomes required')
        group=('lost_timely' if a['is_disrupt'] and a['timely'] and not b['timely'] else
               'removed_fp' if not a['is_disrupt'] and a['alarm'] and not b['alarm'] else 'other')
        result[group].append(a['shot'])
    return result


def lead_median(values):
    return float(np.median(values)) if values else None


def channel_controls(model,x,m,device='cuda'):
    # One original + 23 single-channel flattenings; pad to the reference batch32.
    variants=[x]+[flatten_window(x[None],m[None],(c,))[0] for c in range(23)]
    values=np.stack(variants+[x]*8);masks=np.broadcast_to(m,values.shape).copy()
    h=model.predict_fast(torch.as_tensor(values,device=device),torch.as_tensor(masks,device=device),
                         initial_encoder(model),mode='risk_state')['risk_state']
    z=model.base.risk_head(h).squeeze(-1)
    return z[:24].detach().cpu().numpy(),z[:24].sigmoid().detach().cpu().numpy()


def audit(root,*,replay,channels):
    read=lambda path:json.loads(path.read_text())
    p=read(root/'protocol.json');done=read(root/'completed.json');full=read(root/'per_shot.json');ref=Path(p['reference'])
    assert done['status']=='COMPLETED' and sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for name,h in p['code_sha256'].items():assert sha(io23.ROOT/name)==h,name
    for name,h in p['reference_hashes'].items():assert sha(ref/name)==h,name
    ids=p['contract']['shot_ids'];seals=read(root/'prediction_receipts.json');assert [r['shot'] for r in seals]==ids
    groups=transition_groups(full['raw_original']['predictions'],full['mapped_original']['predictions'])
    assert len(groups['lost_timely'])==29 and len(groups['removed_fp'])==67
    label={sid:k for k,vs in groups.items() for sid in vs};counts={a:dict(tp=0,fp=0,d=0,s=0,leads=[],early=0) for a in p['arms']}
    summary={g:{a:dict(tp=0,fp=0,alarm=0) for a in p['arms']} for g in groups};anchors=[];anchor_error=0.
    curve_change={a:dict(max_abs_probability_change=0.,sum_shot_mean_abs_change=0.,shots_with_any_change=0)
                  for a in p['arms'] if not a.endswith('original')}
    if channels or replay:
        torch.set_num_threads(2);torch.manual_seed(131);model,threshold=load_model(ref)
        assert threshold==p['threshold'];frozen=state_cpu(model)
    mapping=read(root/'input_map.json')
    for index,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';assert sha(path)==seal['prediction_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as old:
            for a in p['arms']:
                r=recount(sid,z['times'],z[a],d,down,p['threshold']);assert r==full[a]['predictions'][index]
                c=counts[a];c['d' if d else 's']+=1;c['tp']+=r['timely'];c['fp']+=not d and r['alarm']
                if r['timely']:c['leads'].append(r['lead_ms']);c['early']+=r['lead_ms']>3000
                for key,value in dict(tp=r['timely'],fp=not d and r['alarm'],alarm=r['alarm']).items():
                    summary[label[sid]][a][key]+=value
            np.testing.assert_array_equal(z['raw_original'],old['raw_source'])
            np.testing.assert_array_equal(z['mapped_original'],old['input_only_static'])
            for a,c in curve_change.items():
                difference=np.abs(z[a].astype(float)-z[a.split('_')[0]+'_original'])
                c['max_abs_probability_change']=max(c['max_abs_probability_change'],float(difference.max()))
                c['sum_shot_mean_abs_change']+=float(difference.mean());c['shots_with_any_change']+=bool(difference.max()>0)
            # Outcome-based selection is confined to this retrospective diagnosis.
            points={name:eligible_peak(z['times'],z[name+'_original'],d,down) for name in ('raw','mapped')}
            if channels:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                raw={k:s[k] for k in ('values','available')};mapped=transform(raw,mapping)
                for end in sorted({v for v in points.values() if v is not None}):
                    record=dict(shot=sid,group=label[sid],end=end,time=float(z['times'][end]),
                        peak_of=[k for k,v in points.items() if v==end],lead_ms=float((down-z['times'][end])*1000) if d else None,views={})
                    for view,signal in (('raw',raw),('mapped',mapped)):
                        x,m=simple_data.window(signal,end,128);zz,pp=channel_controls(model,x,m)
                        error=abs(float(pp[0])-float(z[view+'_original'][end]));anchor_error=max(anchor_error,error)
                        assert error<2e-6,(sid,end,view,error)
                        flat=flatten_window(x[None],m[None],tuple(range(23)))[0]
                        center=model.base.center.detach().cpu().numpy();scale=model.base.scale.detach().cpu().numpy()
                        clean=np.where(m,x,center);flat_clean=np.where(m,flat,center)
                        dz=np.arcsinh((flat_clean.astype(float)-center)/scale)-np.arcsinh((clean.astype(float)-center)/scale)
                        observed=m.sum(axis=0);rms=np.sqrt((dz*dz).sum(axis=0)/observed.clip(1))
                        record['views'][view]=dict(original_probability=float(pp[0]),original_logit=float(zz[0]),
                            flattened_channel_probability=pp[1:].tolist(),flattened_channel_logit_delta=(zz[1:]-zz[0]).tolist(),
                            all_flat_probability=float(z[view+'_all_flat'][end]),observed_counts=observed.tolist(),
                            changed_counts=((x!=flat)&m).sum(axis=0).tolist(),normalized_input_rms_change=rms.tolist())
                    anchors.append(record)
        if channels and (index+1)%60==0:print(json.dumps(dict(audit_shots=index+1,anchor_windows=len(anchors))),flush=True)
    for a,c in counts.items():
        m=done['metrics'][a];assert c['d']==c['s']==180 and c['tp']==m['tp'] and c['fp']==m['fp']
        assert c['early']==m['very_early_over3000'] and c['tp']/180==m['tpr'] and c['fp']/180==m['fpr']
        assert m['J']==m['tpr']-m['fpr'] and lead_median(c['leads'])==m['median_lead_ms']
    replays=[]
    if replay:
        for sid in (ids[0],ids[len(ids)//2],ids[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            raw={k:s[k] for k in ('values','available')}
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for view,signal in (('raw',raw),('mapped',transform(raw,mapping))):
                    for kind,v in predict_controls(model,signal).items():
                        np.testing.assert_array_equal(v,z[view+'_'+kind])
            replays.append(dict(shot=sid,arms=6,max_abs_error=0.))
    if channels or replay:assert_unchanged(model,frozen)
    channel_summary={}
    for group in groups:
        channel_summary[group]={}
        # Exactly one mapped-peak anchor per shot, not double-weighting shots
        # whose raw and mapped peaks differ. Single-column effects are nonadditive.
        rs=[r for r in anchors if r['group']==group and 'mapped' in r['peak_of']]
        for view in ('raw','mapped'):
            if not rs:continue
            delta=np.array([r['views'][view]['flattened_channel_logit_delta'] for r in rs])
            top=np.argsort(-np.median(abs(delta),axis=0))[:6]
            channel_summary[group][view]=dict(shots=len(rs),top_by_median_absolute_logit_effect=[
                dict(channel=io23.RAW23[c],median_abs=float(np.median(abs(delta[:,c]))),median_signed=float(np.median(delta[:,c]))) for c in top])
    lost=[r for r in full['raw_original']['predictions'] if r['shot'] in groups['lost_timely']]
    result=dict(status='VERIFIED',protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        audit_code_sha256=sha(Path(__file__)),all_shot_traces_recounted=len(ids)*len(p['arms']),
        sampled_cuda_replays=replays,originals_reference_exact=True,models_unchanged=True,
        groups=groups,group_counts=summary,channel_summary=channel_summary,channel_anchor_max_abs_error=anchor_error,
        curve_changes={a:dict(max_abs_probability_change=c['max_abs_probability_change'],
            equal_shot_mean_abs_change=c['sum_shot_mean_abs_change']/len(ids),shots_with_any_change=c['shots_with_any_change'])
            for a,c in curve_change.items()},
        lost_raw_warning_timing=dict(n=len(lost),lead_gt125=sum(r['lead_ms']>125 for r in lost),
                                    lead_gt3000=sum(r['lead_ms']>3000 for r in lost),median_ms=float(np.median([r['lead_ms'] for r in lost]))),
        paired={a:paired_flips(full[a.split('_')[0]+'_original']['predictions'],full[a]['predictions'])
                for a in p['arms'] if not a.endswith('original')},
        physical_causality_identified=False,full_independent_reinference=False)
    if channels:_durable_json(root/'channel_anchor_diagnostics.json',anchors)
    _durable_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--replay',action='store_true');p.add_argument('--channels',action='store_true')
    a=p.parse_args();audit(a.root,replay=a.replay,channels=a.channels)
