"""Frozen model diagnosis on historical85; never a deployment result."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..external65_20261023.run import models_for,PREPARED
from ..prefix_coverage_20261023.run import EXTERNAL,EXTERNAL_SHA
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .analysis import descriptor,covariance,nearest_sq,boundary_shifts

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/准备工况覆盖_诊断约定_20261023.md'
FIXED=(0,63,127,255)
VIEWS=('input','early_state','risk_state')


def extract(model,signal,ends):
    windows=[simple_data.window(signal,e,128) for e in ends]
    x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
    with torch.no_grad():
        h=model(x,m,mode='risk_state')['risk_state'];p=model.base.risk_head(h).sigmoid().flatten()
        e,v=model.encode(x,m);e=(e*v[:,:,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
    center=model.base.center.detach().cpu().numpy();scale=model.base.scale.detach().cpu().numpy()
    return dict(input=np.stack([descriptor(xx,mm,center,scale) for xx,mm in windows]),
        early_state=e.cpu().numpy(),risk_state=h.cpu().numpy(),prob=p.cpu().numpy())


def weights(records):
    ids=np.array([r['shot'] for r in records]);return np.array([1./sum(ids==s) for s in ids])


def reference_indices(records,kind,exclude=None):
    return [i for i,r in enumerate(records) if r['group']=='fit_safe' and r['shot']!=exclude
        and (r['fixed'] if kind=='prefix' else r['reference'])]


def geometry(records,arrays):
    details=[]
    for i,r in enumerate(records):
        if not r['fixed'] and not r['case']:continue
        out=dict(**r,prob=float(arrays['prob'][i]),distances={})
        for kind in ('prefix','full'):
            idx=reference_indices(records,kind,r['shot'] if r['group']=='fit_safe' else None)
            refrecords=[records[j] for j in idx];w=weights(refrecords)
            out['distances'][kind]={}
            for view in VIEWS:
                ref=arrays[view][idx];_,cov=covariance(ref,w)
                out['distances'][kind][view]=float(nearest_sq(arrays[view][i:i+1],ref,cov)[0])
        details.append(out)
    return details


def fold(model,signals,outcomes,plan,threshold):
    f=plan['fold'];records=[];parts={k:[] for k in (*VIEWS,'prob')};max_difference=0.
    for sid,s in signals.items():
        d,down=outcomes[sid];isfit=sid in plan['fit'];isheld=sid in plan['held']
        group=('fit_disrupt' if d else 'fit_safe') if isfit else ('held_disrupt' if d else 'held_safe') if isheld else ('external_disrupt' if d else 'external_safe')
        fixed=set(e for e in FIXED if e<len(s['values']));ref=set()
        if group=='fit_safe':ref=fixed|set(np.linspace(127,len(s['values'])-1,16,dtype=int).tolist())
        cases={};old=None
        if not isfit:
            path=(PREPARED if isheld else EXTERNAL)/f'fold{f}'/'shots'/f'{sid}.npz'
            with np.load(path) as z:old=z['meta_static' if isheld else 'prepared_static'].copy()
            if not d and not isheld:
                above=np.flatnonzero(old>=threshold)
                if len(above) and above[0]<256:cases[int(above[0])]='first_fp'
        if d and not isfit:
            timely=np.flatnonzero(np.round((down-s['times'])*1000,9)>30.)
            if len(timely):cases[int(timely[-1])]='timely_anchor'
        ends=sorted(fixed|ref|set(cases));features=extract(model,s,ends)
        if old is not None:
            delta=float(np.max(abs(features['prob']-old[ends])));max_difference=max(max_difference,delta)
            np.testing.assert_allclose(features['prob'],old[ends],rtol=0,atol=2e-6)
        early_fp=bool(old is not None and not d and np.any(old[:256]>=threshold))
        for e in ends:records.append(dict(shot=sid,end=e,time=float(s['times'][e]),group=group,
            fixed=e in fixed,reference=e in ref,case=cases.get(e),early_fp=early_fp))
        for k,v in features.items():parts[k].append(v)
    arrays={k:np.concatenate(v) for k,v in parts.items()}
    distances=geometry(records,arrays)
    idx=reference_indices(records,'full');desc=arrays['input'][idx];low=np.full(23,np.nan);high=low.copy()
    for c in range(23):
        observed=desc[:,46+c]>0
        if observed.any():low[c]=desc[observed,c].min();high[c]=desc[observed,c].max()
    center=model.base.center.detach().cpu().numpy();scale=model.base.scale.detach().cpu().numpy()
    interventions=[]
    for i,r in enumerate(records):
        if not r['case']:continue
        x,m=simple_data.window(signals[r['shot']],r['end'],128)
        probes,shifts=boundary_shifts(x,m,low,high,center,scale)
        xx=torch.as_tensor(probes,device='cuda');mm=torch.as_tensor(np.repeat(m[None],24,0),device='cuda')
        with torch.no_grad():
            h=model(xx,mm,mode='risk_state')['risk_state'];prob=model.base.risk_head(h).sigmoid().flatten().cpu().numpy()
        interventions.append(dict(**r,original=float(arrays['prob'][i]),joint=float(prob[-1]),
            channels=[dict(channel=io23.RAW23[c],shift=float(shifts[c]),prob=float(prob[c]),
                original_median=float(arrays['input'][i,c]),reference_low=float(low[c]),reference_high=float(high[c]))
                for c in range(23) if shifts[c]!=0]))
    return records,arrays,distances,interventions,max_difference


def summarize(details,interventions,threshold):
    groups={}
    for group in sorted({r['group'] for r in details}):
        for category in ('all','early_fp','not_early_fp'):
            rows=[r for r in details if r['fixed'] and r['group']==group and
                (category=='all' or r['early_fp']==(category=='early_fp'))]
            if not rows:continue
            groups[f'{group}:{category}']=dict(windows=len(rows),shots=len({r['shot'] for r in rows}),
                median_distance={kind:{v:float(np.median([r['distances'][kind][v] for r in rows])) for v in VIEWS}
                    for kind in ('prefix','full')})
    probes={}
    for kind in ('first_fp','timely_anchor'):
        rows=[r for r in interventions if r['case']==kind]
        probes[kind]=dict(n=len(rows),original_above=sum(r['original']>=threshold for r in rows),
            joint_above=sum(r['joint']>=threshold for r in rows),
            channel_responses={c:dict(outside=sum(any(e['channel']==c for e in r['channels']) for r in rows),
                above_to_below=sum(r['original']>=threshold and any(e['channel']==c and e['prob']<threshold for e in r['channels']) for r in rows),
                median_probability_change=float(np.median([next((e['prob'] for e in r['channels'] if e['channel']==c),r['original'])-r['original'] for r in rows])))
                for c in io23.RAW23} if rows else {})
    return dict(groups=groups,probes=probes)


def main(root):
    if root.exists():raise FileExistsError(root)
    assert sha(EXTERNAL/'completed.json')==EXTERNAL_SHA and sha(DEFAULT_PARENT)==PARENT_SHA256
    p=READ(EXTERNAL/'protocol.json');ids=sorted(set(p['shot_ids'])|{s for f in p['folds'] for s in f['fit']+f['held']})
    assert len(ids)==85 and not set(ids)&set(io23.cohort_ids('EAST'))
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(131);root.mkdir(parents=True)
    paths=set(Path(__file__).parent.glob('*.py'));paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_support_geometry.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    protocol=dict(folds=p['folds'],shot_ids=ids,external_ids=p['shot_ids'],threshold=p['threshold'],checkpoints=p['checkpoints'],
        source_sha256=PARENT_SHA256,plan_sha256=sha(PLAN),parent_completed_sha256=EXTERNAL_SHA,
        code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in paths},input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in ids},
        retrospective_diagnostic=True,online_steps=0,offline_steps=0,test360_accessed=False)
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for v in sorted(paths):z.write(v,str(v.relative_to(io23.ROOT)))
    outcomes={sid:io23.read_outcome('EAST',sid) for sid in ids}
    raw={sid:apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')) for sid in ids}
    summaries={};maxdelta=0.
    def deny(*args,**kwargs):raise AssertionError('model extraction cannot access labels')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for plan in p['folds']:
            f=plan['fold'];folder=root/f'fold{f}';folder.mkdir()
            model=models_for(f)['prepared'];initial=state_cpu(model)
            mapping=READ(REFERENCE/f'fold{f}'/'input_map.json')
            signals={sid:dict(times=s['times'],**transform({k:s[k] for k in ('values','available')},mapping)) for sid,s in raw.items()}
            records,arrays,distances,probes,delta=fold(model,signals,outcomes,plan,p['threshold']);maxdelta=max(maxdelta,delta)
            _durable_json(folder/'records.json',records);np.savez_compressed(folder/'features.npz',**arrays)
            _durable_json(folder/'distances.json',distances);_durable_json(folder/'probes.json',probes)
            summaries[str(f)]=summarize(distances,probes,p['threshold']);assert_unchanged(model,initial)
            print(dict(fold=f,windows=len(records),cases=len(probes),seconds=round(time.monotonic()-start,2)),flush=True)
    for name,h in p['checkpoints'].items():assert sha(name)==h
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='COMPLETED_DIAGNOSTIC',summary=summaries,max_cached_probability_difference=maxdelta,
        seconds=time.monotonic()-start,neural_updates=0,test360_accessed=False,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',result);print(dict(status=result['status'],seconds=result['seconds'],max_difference=maxdelta),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
