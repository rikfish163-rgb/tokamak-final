"""Cross-shot feedback feasibility. No neural online update and no TEST360."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..simple_train import sha
from ..simple_online import _durable_json
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..conditional_forecast_20261023.learning import shot_weights
from .model import EvidenceProbe,feedback_arrays,rotate_within_shot

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/conditional_forecast_cv20_20261023'
REF_SHA='8e19be2b65eddbfdaae146d8d09acedfd7b60d6f1aed2907cba4b7a172f0adf5'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/到达信息判别反馈_事前约定_20261024.md'
READ=lambda p:json.loads(p.read_text())
SEEDS=(131,132,133)
ARMS=('prior','real','constant','reverse','earlier','mispaired_prior','mispaired_real')


def normalize(features,d,mask,w):
    mean=np.einsum('n,nd->d',w,features)
    scale=np.sqrt(np.einsum('n,nd->d',w,(features-mean)**2)).clip(.001)
    counts=np.einsum('n,ntc->c',w,mask)
    squares=np.einsum('n,ntc->c',w,np.where(mask,d,0.)**2)
    # Wholly absent channels remain masked; identity scale does not create observations.
    ds=np.sqrt(np.divide(squares,counts,out=np.ones(23),where=counts>0)).clip(.001)
    return dict(mean=mean,scale=scale,dscale=ds)


def tensors(h,d,m,n):
    return (torch.as_tensor((h-n['mean'])/n['scale'],dtype=torch.float32,device='cuda'),
            torch.as_tensor(np.where(m,d,0.)/n['dscale'],dtype=torch.float32,device='cuda'),
            torch.as_tensor(m,dtype=torch.bool,device='cuda'))


def train_pair(fit,seed,n):
    h,d,m,ids,y=(fit[k] for k in ('features','delta','mask','ids','y'))
    wrong,wrongm=rotate_within_shot(d,m,ids)
    x=tensors(h,d,m,n);wx=tensors(h,wrong,wrongm,n)
    yy=torch.as_tensor(y,dtype=torch.float32,device='cuda')
    torch.manual_seed(seed)
    net=EvidenceProbe().cuda();nets={'real':net,'mispaired':copy.deepcopy(net)}
    opt={k:torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.001) for k,a in nets.items()}
    gen=np.random.default_rng(seed);w=shot_weights(ids,y);receipt=hashlib.sha256();log=[]
    for step in range(1024):
        idx=gen.choice(len(y),64,replace=True,p=w);receipt.update(idx.astype('<i8').tobytes())
        ii=torch.as_tensor(idx,device='cuda');record=dict(step=step)
        for key,data in (('real',x),('mispaired',wx)):
            opt[key].zero_grad(set_to_none=True)
            prior,post=nets[key](*(a[ii] for a in data))
            loss=.5*(F.binary_cross_entropy_with_logits(prior,yy[ii])+F.binary_cross_entropy_with_logits(post,yy[ii]))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite probe training')
            loss.backward();opt[key].step();record[key]=float(loss.detach())
        log.append(record)
    return nets,log,receipt.hexdigest()


@torch.no_grad()
def predict(nets,h,d,m,old,oldm,n):
    result={a:[] for a in ARMS}
    for start in range(0,len(h),128):
        sl=slice(start,start+128);x,dd,mm=tensors(h[sl],d[sl],m[sl],n)
        _,od,om=tensors(h[sl],old[sl],oldm[sl],n)
        prior,post=nets['real'](x,dd,mm)
        constant=nets['real'](x,torch.zeros_like(dd),mm)[1]
        torch.testing.assert_close(prior,constant,atol=0,rtol=0)
        wp,wr=nets['mispaired'](x,dd,mm)
        values=dict(prior=prior,real=post,constant=constant,
            reverse=nets['real'](x,dd.flip(1),mm.flip(1))[1],
            earlier=nets['real'](x,od,om)[1],mispaired_prior=wp,mispaired_real=wr)
        for k,v in values.items():result[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def evaluate(root,folds):
    rows=[];access=[]
    for fold in folds:
        for sid in fold['held']:
            disrupted,down=io23.read_outcome('EAST',sid)
            access.append(dict(fold=fold['fold'],shot=sid,role='after_all_prediction_seals',disrupted=bool(disrupted),down=down))
            for seed in SEEDS:
                with np.load(root/f'fold{fold["fold"]}'/f'{sid}_{seed}.npz') as z:
                    lead=np.round((down-z['times'])*1000,9) if disrupted else None
                    arrived=np.round((down-z['arrived_times'])*1000,9) if disrupted else None
                    critical=(lead<=125)&(lead>=0)&(arrived>30) if disrupted else np.ones(len(z['times']),bool)
                    for arm in ARMS:
                        values=z[arm][critical].astype(float);y=float(disrupted)
                        bce=np.logaddexp(0,values)-y*values
                        inc=z[arm][critical]-z['prior' if not arm.startswith('mispaired') else 'mispaired_prior'][critical]
                        rows.append(dict(fold=fold['fold'],shot=sid,seed=seed,arm=arm,disrupted=bool(disrupted),
                            n=len(values),bce=float(bce.mean()) if len(values) else None,
                            median_increment=float(np.median(inc)) if len(values) else None))
    summary={}
    for arm in ARMS:
        ss=[r for r in rows if r['arm']==arm and not r['disrupted']]
        ds=[r for r in rows if r['arm']==arm and r['disrupted']]
        if any(r['n']==0 for r in ss+ds):raise ValueError('missing declared evaluation group')
        per_seed={str(seed):dict(safe_bce=float(np.mean([r['bce'] for r in ss if r['seed']==seed])),
            danger_bce=float(np.mean([r['bce'] for r in ds if r['seed']==seed]))) for seed in SEEDS}
        details=[]
        for sid in sorted(set(r['shot'] for r in ss+ds)):
            rr=[r for r in ss+ds if r['shot']==sid]
            details.append(dict(shot=sid,disrupted=rr[0]['disrupted'],mean_bce=float(np.mean([r['bce'] for r in rr])),
                mean_median_increment=float(np.mean([r['median_increment'] for r in rr]))))
        safe=float(np.mean([r['bce'] for r in ss]));danger=float(np.mean([r['bce'] for r in ds]))
        summary[arm]=dict(safe_bce=safe,danger_bce=danger,balanced_bce=(safe+danger)/2,
            safe_direction=sum(not r['disrupted'] and r['mean_median_increment']<0 for r in details),
            danger_direction=sum(r['disrupted'] and r['mean_median_increment']>0 for r in details),
            per_seed=per_seed,per_shot=details)
    a,b=summary['real'],summary['prior']
    gate=bool(a['safe_bce']<b['safe_bce'] and a['danger_bce']<b['danger_bce'] and
        a['safe_direction']>=12 and a['danger_direction']>=3 and
        a['balanced_bce']<min(summary['earlier']['balanced_bce'],summary['mispaired_real']['balanced_bce']))
    _durable_json(root/'per_shot.json',rows);_durable_json(root/'evaluation_access.json',access)
    _durable_json(root/'summary.json',dict(arms=summary,neural_trial_gate=gate,
        scope='DEVELOPMENT_CV20 endpoint information diagnostic, NOT TPR/FPR or intra adaptation gain'))
    return gate,summary


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REF_SHA
    old=READ(REFERENCE/'completed.json');folds=READ(REFERENCE/'protocol.json')['folds']
    consumed=(['protocol.json']+[f'fold{f["fold"]}/fit.npz' for f in folds]+
        [f'fold{f["fold"]}/{sid}.npz' for f in folds for sid in f['held']])
    for name in consumed:assert sha(REFERENCE/name)==old['artifacts'][name],name
    base,_=load_parent(DEFAULT_PARENT);center=base.center.numpy();scale=base.scale.numpy()
    torch.set_num_threads(2);start=time.monotonic();root.mkdir(parents=True)
    paths=list(Path(__file__).parent.glob('*.py'))+[io23.ROOT/'experiments/risk_aligned_online/tests/test_arrived_evidence.py']
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,str(path.relative_to(io23.ROOT)))
    _durable_json(root/'protocol.json',dict(folds=folds,seeds=SEEDS,steps=1024,batch=64,learning_rate=.001,
        weight_decay=.001,source_sha256=PARENT_SHA256,reference_sha256=REF_SHA,plan_sha256=sha(PLAN),
        reference=str(REFERENCE),consumed_sha256={k:sha(REFERENCE/k) for k in consumed},
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in paths},
        online_neural_updates=0,test360_accessed=False,threshold_updates=0,eventlabels_used=False,
        cuda=torch.cuda.get_device_name(),torch=torch.__version__))
    seals=[];fit_receipts=[]
    def deny(*args,**kwargs):raise AssertionError('held outcomes cannot enter feature generation or fitting')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for fold in folds:
            f=fold['fold'];folder=root/f'fold{f}';folder.mkdir()
            with np.load(REFERENCE/f'fold{f}'/'fit.npz') as z:
                fit={k:z[k].copy() for k in ('features','ids','y','origins')}
                fit['delta']=z['target'][:,:5].copy();fit['mask']=z['valid'][:,:5].copy()
            assert set(np.unique(fit['ids']))==set(fold['fit'])
            w=shot_weights(fit['ids'],fit['y']);n=normalize(fit['features'],fit['delta'],fit['mask'],w)
            np.savez_compressed(folder/'normalization.npz',**n)
            held={}
            for sid in fold['held']:
                with np.load(REFERENCE/f'fold{f}'/f'{sid}.npz') as z:
                    h=z['features'].copy();ends=z['origins'].copy();times=z['times'].copy()
                raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal={k:raw[k] for k in ('values','available')}
                np.testing.assert_array_equal(raw['times'][ends],times)
                d,m,od,om=feedback_arrays(signal,ends,center,scale)
                m[:,:,io23.RAW23.index('mode_number_n')]=False;om[:,:,io23.RAW23.index('mode_number_n')]=False
                held[sid]=(h,d,m,od,om,times,raw['times'][ends+5],ends)
                np.savez_compressed(folder/f'{sid}_inputs.npz',features=h,delta=d,mask=m,earlier=od,earlier_mask=om,
                    times=times,arrived_times=raw['times'][ends+5],origins=ends)
            for seed in SEEDS:
                nets,log,receipt=train_pair(fit,seed,n)
                torch.save({k:net.cpu().state_dict() for k,net in nets.items()},folder/f'weights_{seed}.pt')
                for net in nets.values():net.cuda().eval()
                _durable_json(folder/f'training_{seed}.json',log)
                fit_receipts.append(dict(fold=f,seed=seed,samples_sha256=receipt,fit_ids=fold['fit'],
                    rows=len(fit['y']),positive_rows=int(fit['y'].sum()),parameter_count=sum(p.numel() for p in nets['real'].parameters())))
                for sid,(h,d,m,od,om,times,arrived,ends) in held.items():
                    predictions=predict(nets,h,d,m,od,om,n)
                    if not all(np.isfinite(v).all() for v in predictions.values()):raise FloatingPointError('nonfinite held score')
                    path=folder/f'{sid}_{seed}.npz'
                    np.savez_compressed(path,times=times,arrived_times=arrived,origins=ends,**predictions)
                    seals.append(dict(fold=f,shot=sid,seed=seed,path=str(path.relative_to(root)),sha256=sha(path)))
                print(dict(fold=f,seed=seed,seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'prediction_seals.json',seals);_durable_json(root/'fit_receipts.json',fit_receipts)
    gate,summary=evaluate(root,folds)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    complete=dict(status='COMPLETED_ARRIVED_EVIDENCE_CV20',seconds=time.monotonic()-start,neural_trial_gate=gate,
        source_unchanged=True,artifact_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    _durable_json(root/'completed.json',complete)
    print({k:{a:b for a,b in v.items() if a not in ('per_seed','per_shot')} for k,v in summary.items()},flush=True)
    print(dict(neural_trial_gate=gate,seconds=complete['seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
