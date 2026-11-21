"""Paired source-TRAIN update utility with the deployed missing-MNM condition."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory
from risk_feature_intra_20261107 import FeatureIntra,update
from run_joint_motion_20261107 import FIT


PARENT=j.r.ROOT/'probe_outputs/risk_feature_source_utility_20261107_v1/result.json'


def masked_view(e,query_mask):
    masked=dict(e,a=e['a'].clone(),mask=e['mask'].clone())
    masked['a'][...,2]=False
    masked['mask'][...,2]=False
    available=query_mask.clone()
    available[...,2]=False
    return masked,available


def measure(base,decoder,e,qx,qa,current_y,future_y):
    model=FeatureIntra(base,decoder)
    opt=torch.optim.AdamW([model.weight],lr=j.INTRA_LR,weight_decay=0.)
    current=torch.tensor([current_y],dtype=qx.dtype,device=qx.device)
    future=torch.tensor([future_y],dtype=qx.dtype,device=qx.device)
    logit=model.risk(qx,qa)['logit']
    before_future=F.binary_cross_entropy_with_logits(logit,future)
    before_current=F.binary_cross_entropy_with_logits(logit,current)
    risk_gradient,=torch.autograd.grad(before_future,model.weight)
    signal=model.signal_loss(e)
    signal_gradient,=torch.autograd.grad(signal,model.weight)
    norm=risk_gradient.norm()*signal_gradient.norm()
    cosine=float((risk_gradient*signal_gradient).sum()/norm) if float(norm)>0 else None
    initial=model.weight.detach().clone()
    update(model,opt,e)
    with torch.no_grad():
        after=model.risk(qx,qa)['logit']
        after_future=F.binary_cross_entropy_with_logits(after,future)
        after_current=F.binary_cross_entropy_with_logits(after,current)
    return dict(cosine=cosine,
        actual_step_dot_future_gradient=float((risk_gradient*(model.weight-initial)).sum().detach()),
        future_risk_before=float(before_future.detach()),future_risk_after=float(after_future),
        current_risk_before=float(before_current.detach()),current_risk_after=float(after_current))


def run(out):
    j.r.configure()
    prior=j.r.read(PARENT)
    cases=prior['records']
    roles=j.simple_data.source_roles()
    train=set(roles['train'])
    if any(v['shot'] not in train for v in cases):
        raise ValueError('only the previous source TRAIN cases are authorized')
    base,_=j.r.load_parent(j.r.DEFAULT_PARENT)
    base.cuda()
    decoder=OrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    decoder.load_state_dict(torch.load(FIT/'source_predictor.pt',map_location='cuda',weights_only=True))
    advanced=j.simple_data.advanced_times()
    shots={sid:j.simple_data.attach_outcome(j.simple_data.read_shot('J-TEXT',sid),'J-TEXT',advanced)
           for sid in dict.fromkeys(v['shot'] for v in cases)}
    records=[]
    for old in cases:
        s=shots[old['shot']];o,q=old['origin'],old['query']
        if q!=o+26 or old['label_time']!=q+25 or not s['valid'][o-127:q+26].all():
            raise ValueError('the same complete causal source episode is required')
        current_y,future_y=float(s['y'][q]),float(s['y'][q+25])
        if old['group']!=f'{int(current_y)}_to_{int(future_y)}':
            raise ValueError('source native labels changed since the paired diagnostic')
        x,a=j.simple_data.window(s,o,128);qx,qa=j.simple_data.window(s,q,128)
        e={k:torch.as_tensor(v[None],device='cuda') for k,v in
            dict(x=x,a=a,truth=s['values'][o+1:o+26],mask=s['available'][o+1:o+26]).items()}
        qx,qa=torch.as_tensor(qx[None],device='cuda'),torch.as_tensor(qa[None],device='cuda')
        native=measure(base,decoder,e,qx,qa,current_y,future_y)
        for key,value in native.items():
            if (value is None)!=(old[key] is None) or (value is not None and abs(value-old[key])>1e-7):
                raise ValueError('native update no longer reproduces the previous diagnostic')
        masked,masked_query=masked_view(e,qa)
        missing=measure(base,decoder,masked,qx,masked_query,current_y,future_y)
        records.append(dict(shot=s['shot'],origin=o,query=q,label_time=q+25,
                            group=old['group'],native=native,no_MNM=missing))
    summary={}
    for group in dict.fromkeys(v['group'] for v in records):
        selected=[v for v in records if v['group']==group]
        summary[group]={}
        for view in ('native','no_MNM'):
            rows=[v[view] for v in selected]
            cosine=[v['cosine'] for v in rows if v['cosine'] is not None]
            summary[group][view]=dict(n=len(rows),
                future_improved=sum(v['future_risk_after']<v['future_risk_before'] for v in rows),
                current_improved=sum(v['current_risk_after']<v['current_risk_before'] for v in rows),
                mean_future_before=float(np.mean([v['future_risk_before'] for v in rows])),
                mean_future_delta=float(np.mean([v['future_risk_after']-v['future_risk_before'] for v in rows])),
                median_cosine=float(np.median(cosine)) if cosine else None)
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'result.json',dict(status='MATCHED_SOURCE_MISSING_MNM_UPDATE_UTILITY',
        parent=str(PARENT),summary=summary,records=records,source_TRAIN_only=True,
        no_east_data=True,source_classifier_unchanged=True,first_step_only=True,
        native_reproduced=True,stratified_not_population_estimate=True,
        source_files={str(p):p.read_text() for p in (Path(__file__),j.r.ROOT/'risk_feature_intra_20261107.py')}))
    print(summary,flush=True)


if __name__=='__main__':
    run(j.r.ROOT/'probe_outputs/risk_feature_masked_utility_20261107_v1')
