"""Numerical stationarity, CPU refit and independent metric checks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .run import (READ,io23,sha,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,shot_weights,
    _durable_json,REPRESENTATIONS,coordinates,feedback,predict,load_parent,simple_data,apply_input_contract)
from .learning import fit_normalization,logistic_fit,linear_score
from ..arrived_evidence_20261024.model import rotate_within_shot


def main(root):
    p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(Path(name))==digest,name
    torch.set_num_threads(2)
    outcomes={(r['fold'],r['role'],r['shot']):r for r in READ(root/'evaluation_access.json')}
    expected={(r['fold'],r['role'],r['shot'],r['arm']):r for r in READ(root/'per_shot.json')}
    base,_=load_parent(DEFAULT_PARENT);center=base.center.numpy();scale=base.scale.numpy()
    count=0;replayed=0;recounts=[];stationarity=[];cpu_refits=0;numpy_prefix_checks=0
    for fold in p['folds']:
        f=fold['fold'];folder=root/f'fold{f}';data={}
        for role in ('fit','held'):
            for sid in fold[role]:
                with np.load(folder/f'{role}_{sid}_inputs.npz') as z:data[role,sid]={k:z[k].copy() for k in z.files}
        train={k:np.concatenate([data['fit',sid][k] for sid in fold['fit']]) for k in data['fit',fold['fit'][0]]}
        w=shot_weights(train['ids'],train['y']);models={}
        wrong,wrongm=rotate_within_shot(train['delta'],train['mask'],train['ids'])
        for name in REPRESENTATIONS:
            with np.load(folder/f'{name}_model.npz') as z:model={k:z[k].copy() for k in z.files}
            n=fit_normalization(train[name],train['delta'],train['mask'],w)
            for k,v in n.items():np.testing.assert_array_equal(v,model[k])
            models[name]=model;x=coordinates(train,name,model)
            d=feedback(train['delta'],train['mask'],model);wd=feedback(wrong,wrongm,model)
            off=linear_score(x,model['prior'])
            for arm,xx,offset in (('prior',np.c_[x,np.ones(len(x))],np.zeros(len(x))),('real',d,off),('mispaired',wd,off)):
                b=model[arm];z=xx@b+offset;pr=torch.as_tensor(z).sigmoid().numpy()
                reg=np.full(len(b),.1)
                if arm=='prior':reg[-1]=0.
                gradient=xx.T@(w*(pr-train['y']))+reg*b
                maximum=float(np.max(np.abs(gradient)))
                assert maximum<=1.1e-8,(f,name,arm,maximum)
                stationarity.append(dict(fold=f,representation=name,arm=arm,gradient_max=maximum))
            if f==0 and name=='hidden':
                b,_=logistic_fit(x,train['y'],w,device='cpu')
                u,_=logistic_fit(d,train['y'],w,offset=linear_score(x,b),intercept=False,device='cpu')
                np.testing.assert_allclose(b,model['prior'],atol=1e-8,rtol=1e-8)
                np.testing.assert_allclose(u,model['real'],atol=1e-8,rtol=1e-8)
                cpu_refits+=2
        with np.load(folder/'native_head.npz') as z:native={k:z[k] for k in z.files}
        for (role,sid),a in data.items():
            curves=predict(a,models,native)
            with np.load(folder/f'{role}_{sid}_predictions.npz') as z:
                for arm,v in curves.items():np.testing.assert_array_equal(v,z[arm])
            replayed+=1
            for name in REPRESENTATIONS:np.testing.assert_array_equal(curves[name+'_prior'],curves[name+'_constant'])
            r=outcomes[f,role,sid];dis=r['disrupted'];down=r['down']
            if role=='held' and dis:
                raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                # First/middle/last cached origins; direct NumPy formula, not causal_summary.
                for index in sorted({0,len(a['origins'])//2,len(a['origins'])-1}):
                    end=int(a['origins'][index]);x,m=simple_data.window(raw,end,128)
                    valid=m&np.isfinite(x);q=np.arcsinh((np.where(valid,x,center).astype(float)-center)/scale)
                    n=valid.sum(0).clip(1);mean=q.sum(0)/n
                    sd=np.sqrt((((q-mean)**2)*valid).sum(0)/n)
                    last=np.array([q[np.flatnonzero(valid[:,c])[-1],c] if valid[:,c].any() else 0. for c in range(23)])
                    change=q[64:].sum(0)/valid[64:].sum(0).clip(1)-q[:64].sum(0)/valid[:64].sum(0).clip(1)
                    target=np.concatenate((last,mean,sd,change,valid.mean(0)))
                    np.testing.assert_allclose(target,a['raw'][index],atol=1e-12,rtol=1e-12)
                    numpy_prefix_checks+=1
            use=(np.round((down-a['times'])*1000,9)<=125)&(np.round((down-a['arrived_times'])*1000,9)>30) if dis else np.ones(len(a['times']),bool)
            for arm,z in curves.items():
                zz=torch.as_tensor(z[use],dtype=torch.float64)
                bce=float(F.binary_cross_entropy_with_logits(zz,torch.full_like(zz,float(dis))))
                inc=z[use]-curves[arm.split('_')[0]+'_prior'][use] if arm!='native' else np.zeros(len(zz))
                wanted=expected[f,role,sid,arm]
                np.testing.assert_allclose(bce,wanted['bce'],rtol=1e-12,atol=1e-12)
                assert wanted['n']==len(zz) and wanted['median_increment']==float(np.median(inc))
                recounts.append({**wanted,'bce':bce});count+=1
    s=READ(root/'summary.json')
    for role,arms in s['results'].items():
        for arm,v in arms.items():
            rows=[r for r in recounts if r['role']==role and r['arm']==arm]
            safe=[r for r in rows if not r['disrupted']];danger=[r for r in rows if r['disrupted']]
            a=np.mean([r['bce'] for r in safe]);b=np.mean([r['bce'] for r in danger])
            np.testing.assert_allclose([a,b,(a+b)/2],[v['safe_bce'],v['danger_bce'],v['balanced_bce']],atol=1e-12,rtol=1e-12)
            assert v['safe_direction']==sum(r['median_increment']<0 for r in safe)
            assert v['danger_direction']==sum(r['median_increment']>0 for r in danger)
    for name,gate in s['neural_trial_gates'].items():
        v=s['results']['held'];a=v[name+'_real'];b=v[name+'_prior']
        actual=(a['safe_bce']<b['safe_bce'] and a['danger_bce']<b['danger_bce'] and a['safe_direction']>=12 and a['danger_direction']>=3
            and a['balanced_bce']<min(v[name+'_earlier']['balanced_bce'],v[name+'_mispaired']['balanced_bce']))
        assert bool(actual)==gate
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_CAPACITY_REPRESENTATION_CV20',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),shot_fold_replays=replayed,independent_bce_recounts=count,
        cpu_refits=cpu_refits,independent_numpy_prefix_checks=numpy_prefix_checks,stationarity=stationarity,source_unchanged=True,
        scope='Numerical reproduction only, not independent scientific confirmation')
    _durable_json(root/'verification.json',result);print({k:v for k,v in result.items() if k!='stationarity'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
