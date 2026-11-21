"""Fixed-capacity crossed representation diagnostic on historical CV20."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from ..arrived_evidence_20261024.run import (READ,REFERENCE,REF_SHA,io23,simple_data,
    apply_input_contract,sha,load_parent,DEFAULT_PARENT,PARENT_SHA256,shot_weights,_durable_json)
from ..arrived_evidence_20261024.model import feedback_arrays,rotate_within_shot
from .learning import prefix_statistics,fit_normalization,logistic_fit,linear_score

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/反馈容量与表示_配对检验事前约定_20261024.md'
PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/arrived_evidence_cv20_20261024'
PREV_SHA='69e0840e659fb8455951b7e73e01a524adb596da2d9c0cb3017ada73d805549c'
REPRESENTATIONS=('hidden','raw')
MODES=('prior','real','constant','reverse','earlier','mispaired')


def coordinates(a,representation,n):
    return (a[representation]-n['mean'])/n['scale']


def feedback(d,m,n):
    return (np.where(m,d,0.)/n['dscale']).reshape(len(d),-1)


def fit(a,representation):
    w=shot_weights(a['ids'],a['y']);n=fit_normalization(a[representation],a['delta'],a['mask'],w)
    x=coordinates(a,representation,n);d=feedback(a['delta'],a['mask'],n)
    wrong,wrongm=rotate_within_shot(a['delta'],a['mask'],a['ids'])
    b,bl=logistic_fit(x,a['y'],w,device='cuda')
    prior=linear_score(x,b)
    u,ul=logistic_fit(d,a['y'],w,offset=prior,intercept=False,device='cuda')
    q,ql=logistic_fit(feedback(wrong,wrongm,n),a['y'],w,offset=prior,intercept=False,device='cuda')
    return dict(**n,prior=b,real=u,mispaired=q),dict(prior=bl,real=ul,mispaired=ql)


def predict(a,models,native):
    out={'native':a['hidden']@native['weight']+native['bias']}
    for name,model in models.items():
        p=linear_score(coordinates(a,name,model),model['prior'])
        d=feedback(a['delta'],a['mask'],model)
        curves=dict(prior=p,real=p+d@model['real'],constant=p.copy(),
            reverse=p+feedback(a['delta'][:,::-1],a['mask'][:,::-1],model)@model['real'],
            earlier=p+feedback(a['earlier'],a['earlier_mask'],model)@model['real'],
            mispaired=p+d@model['mispaired'])
        out.update({name+'_'+k:v for k,v in curves.items()})
    if not all(np.isfinite(v).all() for v in out.values()):raise FloatingPointError('nonfinite score')
    return out


def evaluate(root,folds):
    rows=[];access=[]
    for fold in folds:
        folder=root/f'fold{fold["fold"]}'
        for role in ('fit','held'):
            for sid in fold[role]:
                disrupted,down=io23.read_outcome('EAST',sid)
                access.append(dict(fold=fold['fold'],role=role,shot=sid,phase='after_all_seals',disrupted=bool(disrupted),down=down))
                with np.load(folder/f'{role}_{sid}_predictions.npz') as z:
                    use=(np.round((down-z['times'])*1000,9)<=125)&(np.round((down-z['arrived_times'])*1000,9)>30) if disrupted else np.ones(len(z['times']),bool)
                    if not use.any():raise ValueError('missing evaluation interval')
                    for arm in z.files:
                        if arm in ('times','arrived_times'):continue
                        zz=z[arm][use];bce=np.logaddexp(0,zz)-float(disrupted)*zz
                        increment=zz-z[arm.split('_')[0]+'_prior'][use] if arm!='native' else np.zeros(len(zz))
                        rows.append(dict(role=role,fold=fold['fold'],shot=sid,disrupted=bool(disrupted),arm=arm,
                            n=len(zz),bce=float(bce.mean()),median_increment=float(np.median(increment))))
    summary={}
    for role in ('fit','held'):
        summary[role]={}
        for arm in sorted(set(r['arm'] for r in rows)):
            r=[a for a in rows if a['arm']==arm and a['role']==role]
            safe=[a for a in r if not a['disrupted']];danger=[a for a in r if a['disrupted']]
            s=np.mean([a['bce'] for a in safe]);d=np.mean([a['bce'] for a in danger])
            summary[role][arm]=dict(safe_bce=float(s),danger_bce=float(d),balanced_bce=float((s+d)/2),
                safe_direction=sum(a['median_increment']<0 for a in safe),danger_direction=sum(a['median_increment']>0 for a in danger))
    gates={}
    for name in REPRESENTATIONS:
        s=summary['held'];a=s[name+'_real'];b=s[name+'_prior']
        gates[name]=bool(a['safe_bce']<b['safe_bce'] and a['danger_bce']<b['danger_bce'] and
            a['safe_direction']>=12 and a['danger_direction']>=3 and
            a['balanced_bce']<min(s[name+'_earlier']['balanced_bce'],s[name+'_mispaired']['balanced_bce']))
    _durable_json(root/'per_shot.json',rows);_durable_json(root/'evaluation_access.json',access)
    _durable_json(root/'summary.json',dict(results=summary,neural_trial_gates=gates,
        scope='DEV_CV20 diagnostic; no TPR/FPR, online learning or independent confirmation'))
    return summary,gates


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env local CUDA required')
    assert sha(REFERENCE/'completed.json')==REF_SHA and sha(PREVIOUS/'completed.json')==PREV_SHA
    old=READ(REFERENCE/'completed.json');previous=READ(PREVIOUS/'completed.json')
    for base,manifest in ((REFERENCE,old['artifacts']),(PREVIOUS,previous['artifact_sha256'])):
        for name,digest in manifest.items():assert sha(base/name)==digest,name
    folds=READ(REFERENCE/'protocol.json')['folds'];older=Path(READ(REFERENCE/'protocol.json')['reference'])
    base,_=load_parent(DEFAULT_PARENT);center=base.center.numpy();scale=base.scale.numpy()
    torch.set_num_threads(2);start=time.monotonic();root.mkdir(parents=True)
    paths=list(Path(__file__).parent.glob('*.py'))+[io23.ROOT/'experiments/risk_aligned_online/tests/test_feedback_capacity.py',
        io23.ROOT/'experiments/risk_aligned_online/event_prefix_probe.py',
        io23.ROOT/'experiments/risk_aligned_online/arrived_evidence_20261024/run.py']
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(io23.ROOT)))
    _durable_json(root/'protocol.json',dict(folds=folds,reference_sha256=REF_SHA,previous_sha256=PREV_SHA,
        source_sha256=PARENT_SHA256,plan_sha256=sha(PLAN),l2=.1,fit='float64 full-batch Newton',
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for f in folds for sid in f['held']},
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in paths},
        test360_accessed=False,eventlabels_used=False,online_neural_updates=0,gpu=torch.cuda.get_device_name(),torch=torch.__version__))
    seals=[];logs=[]
    def deny(*args,**kwargs):raise AssertionError('no new outcome reads before all prediction seals')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for fold in folds:
            f=fold['fold'];folder=root/f'fold{f}';folder.mkdir();data={}
            assert not set(fold['fit'])&set(fold['held'])
            with np.load(REFERENCE/f'fold{f}'/'fit.npz') as z:cached={k:z[k].copy() for k in ('features','ids','y','origins','target','valid')}
            for role in ('fit','held'):
                for sid in fold[role]:
                    if role=='fit':
                        use=cached['ids']==sid;h=cached['features'][use];ends=cached['origins'][use]
                        a=dict(hidden=h,origins=ends,ids=cached['ids'][use],y=cached['y'][use])
                    else:
                        with np.load(PREVIOUS/f'fold{f}'/f'{sid}_inputs.npz') as z:
                            a={k:z[k].copy() for k in z.files};a['hidden']=a.pop('features')
                        ends=a['origins']
                    raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                    signal={k:raw[k] for k in ('values','available')}
                    d,m,od,om=feedback_arrays(signal,ends,center,scale)
                    m[:,:,io23.RAW23.index('mode_number_n')]=False;om[:,:,io23.RAW23.index('mode_number_n')]=False
                    if role=='fit':
                        np.testing.assert_array_equal(d,cached['target'][use,:5]);np.testing.assert_array_equal(m,cached['valid'][use,:5])
                    else:
                        for key,v in dict(delta=d,mask=m,earlier=od,earlier_mask=om).items():np.testing.assert_array_equal(a[key],v)
                    a.update(delta=d,mask=m,earlier=od,earlier_mask=om,times=raw['times'][ends],arrived_times=raw['times'][ends+5],
                        raw=prefix_statistics(signal,ends,center,scale,device='cuda'))
                    data[role,sid]=a;np.savez_compressed(folder/f'{role}_{sid}_inputs.npz',**a)
            train={k:np.concatenate([data['fit',s][k] for s in fold['fit']]) for k in data['fit',fold['fit'][0]]}
            models={}
            for name in REPRESENTATIONS:
                models[name],trace=fit(train,name);np.savez_compressed(folder/f'{name}_model.npz',**models[name])
                _durable_json(folder/f'{name}_optimization.json',trace)
                logs.append(dict(fold=f,representation=name,rows=len(train['y']),positive_rows=int(train['y'].sum()),
                    gradients={k:t[-1]['gradient_max'] for k,t in trace.items()}))
            cp=older/f'fold{f}'/'increment.pt'
            assert sha(cp)==READ(REFERENCE/f'fold{f}'/'fit_receipt.json')['parent_sha256']
            state=torch.load(cp,map_location='cpu',weights_only=True)['model']
            native=dict(weight=state['base.risk_head.weight'].numpy().reshape(-1).astype(float),bias=state['base.risk_head.bias'].numpy().astype(float))
            np.savez_compressed(folder/'native_head.npz',**native)
            for (role,sid),a in data.items():
                predictions=predict(a,models,native)
                path=folder/f'{role}_{sid}_predictions.npz'
                np.savez_compressed(path,times=a['times'],arrived_times=a['arrived_times'],**predictions)
                seals.append(dict(fold=f,role=role,shot=sid,path=str(path.relative_to(root)),sha256=sha(path)))
            print(dict(fold=f,seconds=time.monotonic()-start,fit_logs=logs[-2:]),flush=True)
    _durable_json(root/'prediction_seals.json',seals);_durable_json(root/'fit_receipts.json',logs)
    summary,gates=evaluate(root,folds)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'completed.json',dict(status='COMPLETED_CAPACITY_REPRESENTATION_CV20',seconds=time.monotonic()-start,
        source_unchanged=True,neural_trial_gates=gates,artifact_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}))
    print(dict(held=summary['held'],gates=gates,seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
