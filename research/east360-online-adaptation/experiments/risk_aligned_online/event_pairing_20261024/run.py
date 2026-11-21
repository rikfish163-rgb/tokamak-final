"""SOURCE event/hazard metric experiment and EAST20 same-history pair screen.

No TEST220 evaluation, no TEST360 data read, no risk-network update. The
auxiliary metric can qualify for a later neural adaptation experiment only.
"""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .. import simple_data,io23
from ..causal_input import apply_input_contract
from ..event_prefix_probe import DEFAULT_LABELS,LABEL_SHA256
from ..event_content_20261023.probe import CACHE
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..curvature_memory_20261024.run import PREVIOUS,READ
from .learning import features,mean_features,reverse_features,pair_candidates,sample_triplets,Encoder,pair_loss,neighbour_score
from .evaluation import pair_out_auc,qualifies

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/同类事件危险轨迹配对_事前约定_20261024.md'
AMENDMENT=io23.ROOT/'docs/research/source_event_rebuild_20261023/事件配对筛查_评价修订_20261024.md'
MODES=('risk','mean','shuffle','family')


def build_data():
    old=READ(CACHE/'completed.json');assert sha(CACHE/'features.npz')==old['feature_cache_sha256']
    assert sha(DEFAULT_LABELS)==LABEL_SHA256
    with np.load(CACHE/'features.npz') as z:
        use=np.isin(z['role'],['train','val'])
        meta={k:z[k][use].copy() for k in ('shot','endpoint','event_id','target','role','outcome')}
    base,_=load_parent(DEFAULT_PARENT)
    center=base.center.detach().numpy();scale=base.scale.detach().numpy()
    roles=simple_data.source_roles();adv=simple_data.advanced_times();collected=[];inputs={};inventory={}
    allowed=set(roles['train']+roles['val']);reader=io23.read_outcome
    def source_only(device,sid):
        assert device=='J-TEXT' and sid in allowed
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_only):
        for number,sid in enumerate(np.unique(meta['shot'])):
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('J-TEXT',int(sid))),'J-TEXT',adv)
            inputs[str(sid)]=sha(io23.path_for('J-TEXT',sid))
            rows=np.flatnonzero(meta['shot']==sid)
            assert np.all(meta['outcome'][rows]==int(s['is_disrupt']))
            for i in rows:
                end=int(meta['endpoint'][i]);role=str(meta['role'][i]);family=int(meta['target'][i])
                if end<127 or not s['valid'][end]:continue
                lead=None if not s['is_disrupt'] else float(round((s['down']-s['times'][end])*1000,9))
                if lead is not None and lead<=10:continue
                x,m=simple_data.window(s,end,128);label=int(s['y'][end])
                collected.append(dict(x=features(x,m,center,scale),shot=int(sid),end=end,event=int(meta['event_id'][i]),
                    family=family,role=role,y=label,time=float(s['times'][end]),lead=-1. if lead is None else lead))
            if (number+1)%80==0:print(dict(stage='source_descriptors',shots=number+1),flush=True)
    data={k:np.array([r[k] for r in collected]) for k in collected[0]}
    for role in ('train','val'):
        group=data['role']==role;inventory[role]={}
        for family in range(6):
            inventory[role][chr(65+family)]={str(y):dict(windows=int(np.sum(group&(data['family']==family)&(data['y']==y))),
                shots=len(np.unique(data['shot'][group&(data['family']==family)&(data['y']==y)]))) for y in (0,1)}
    with np.load(PREVIOUS/'bank.npz') as z:
        east=dict(x=np.stack([features(x,m,center,scale) for x,m in zip(z['x'],z['mask'])]),
            shot=z['shots'].copy(),y=z['labels'].copy())
    assert set(east['shot'])==set(READ(PREVIOUS/'protocol.json')['preparation_ids'])
    return data,east,inputs,inventory


def shot_weights(shot,y):
    weight=np.zeros(len(y),float)
    for label in (0,1):
        ids=np.unique(shot[y==label])
        if not len(ids):raise ValueError('both states needed for weighted AUC')
        for sid in ids:
            rows=(shot==sid)&(y==label);weight[rows]=1./(len(ids)*rows.sum())
    return weight


def metrics(data,score):
    weight=shot_weights(data['shot'],data['y'])
    out=dict(auc=float(roc_auc_score(data['y'],score,sample_weight=weight)),n=len(score),
             n_shots=len(np.unique(data['shot'])))
    if 'family' in data:
        family={}
        for f in range(6):
            use=data['family']==f
            if len(np.unique(data['y'][use]))!=2:raise ValueError(f'family{f} lacks validation support')
            w=shot_weights(data['shot'][use],data['y'][use])
            family[chr(f+65)]=float(roc_auc_score(data['y'][use],score[use],sample_weight=w))
        out.update(family_auc=family,conditional_macro_auc=float(np.mean(list(family.values()))))
    return out


def fit(x,pairs,seed,steps):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=Encoder().cuda();xx=torch.as_tensor(x,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    trace=[]
    for step in range(steps):
        ix=sample_triplets(pairs,rng,128)
        loss=pair_loss(model(xx[ix]))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite contrastive loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%128==0 or step==steps-1:trace.append(dict(step=step+1,loss=float(loss.detach())))
    return model.eval(),trace


def encode(model,x):
    with torch.no_grad():return np.concatenate([model(torch.as_tensor(x[i:i+1024],device='cuda')).cpu().numpy()
                                               for i in range(0,len(x),1024)])


def main(root,steps=1024):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();root.mkdir(parents=True)
    data,east,inputs,inventory=build_data();train=data['role']=='train';val=data['role']=='val'
    assert not set(data['shot'][train])&set(data['shot'][val])
    assert not set(data['shot'])&set(simple_data.source_roles()['test'])
    tr={k:v[train] for k,v in data.items()};va={k:v[val] for k,v in data.items()}
    # Equal weight per source TRAIN shot for scaling; no VAL/EAST moments.
    _,inv,counts=np.unique(tr['shot'],return_inverse=True,return_counts=True);w=1./counts[inv]
    mean=np.average(tr['x'].astype('float64'),axis=0,weights=w)
    scale=np.maximum(np.sqrt(np.average((tr['x']-mean)**2,axis=0,weights=w)),.1)
    norm=lambda x:((x-mean)/scale).astype('float32')
    np.savez_compressed(root/'source.npz',**data);np.savez_compressed(root/'east20.npz',**east)
    np.savez_compressed(root/'standardizer.npz',mean=mean,scale=scale)
    _durable_json(root/'inventory.json',inventory)
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_event_pairing.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(scope='DEVELOPMENT_ONLY_EVENT_CONDITIONAL_SIMILARITY',modes=MODES,seeds=[131,132,133],steps=steps,
        parent_sha256=PARENT_SHA256,labels_sha256=LABEL_SHA256,cache_sha256=sha(CACHE/'features.npz'),
        plan_sha256=sha(PLAN),evaluation_amendment_sha256=sha(AMENDMENT),source_input_sha256=inputs,train_ids=sorted(set(tr['shot'].tolist())),
        val_ids=sorted(set(va['shot'].tolist())),east_ids=sorted(set(east['shot'].tolist())),
        east_bank_sha256=sha(PREVIOUS/'bank.npz'),source_online=False,source_risk_changed=False,
        test220_evaluated=False,test360_read=False,east_training_steps=0,risk_outputs_modified=False,
        pair_sampling='event x time-point-risk x shot balanced; nearest4 other-shot mean states',
        representation='391->64SiLU->16unit; MNM masked in both domains for this auxiliary representation',
        optimizer='AdamW lr=.001 wd=.001 clip1 batch128; fixed final step',
        source_labels='max(25ms,AdvancedTime), only endpoints lead>10ms',
        evaluation='mean nearest3 different shots per risk class; EAST same-history leave-one-D-one-S-out AUC',
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),independent_confirmation=False)
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    results={};predictions={};heads={};traces={}
    def score(name,training,validation,target):
        vs=neighbour_score(validation,training,tr['shot'],tr['y'],exclude=np.full(len(validation),-1))
        es=neighbour_score(target,target,east['shot'],east['y'],exclude=east['shot'])
        ea,_=pair_out_auc(target,target,east['shot'],east['y'])
        results[name]=dict(val=metrics(va,vs),east=dict(auc=ea,estimator='same-history shot-pair AUC'),
                           pooled_LOO_not_used_for_gate=metrics(east,es))
        predictions[name+'_val']=vs;predictions[name+'_east']=es
        print(dict(stage='screen',name=name,**results[name],seconds=round(time.monotonic()-start,1)),flush=True)
    score('raw',norm(tr['x']),norm(va['x']),norm(east['x']))
    score('raw_mean',norm(mean_features(tr['x'])),norm(mean_features(va['x'])),norm(mean_features(east['x'])))
    means=norm(tr['x'])[:,:23].astype('float64')
    fixed={m:pair_candidates(means,tr['shot'],tr['y'],tr['family'],mode=m) for m in ('risk','family')}
    for mode,pairs in fixed.items():np.savez_compressed(root/f'pairs_{mode}.npz',**pairs,y=tr['y'])
    def deny(*args,**kwargs):raise AssertionError('no outcome reading during metric training or evaluation')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for seed in p['seeds']:
            for mode in MODES:
                if time.monotonic()-start>1800:raise TimeoutError('declared30min exceeded')
                y=tr['y'].copy();pairs=fixed['family' if mode=='family' else 'risk']
                if mode=='shuffle':
                    rng=np.random.default_rng(seed)
                    for f in range(6):
                        use=tr['family']==f;y[use]=rng.permutation(y[use])
                    pairs=pair_candidates(means,tr['shot'],y,tr['family'],mode='risk')
                    np.savez_compressed(root/f'pairs_shuffle_{seed}.npz',**pairs,y=y)
                view=mean_features if mode=='mean' else lambda x:x
                model,trace=fit(norm(view(tr['x'])),pairs,seed,steps)
                name=f'{mode}_{seed}';heads[name]={k:v.cpu().clone() for k,v in model.state_dict().items()};traces[name]=trace
                a,b,c=[encode(model,norm(view(s['x']))) for s in (tr,va,east)]
                score(name,a,b,c)
                if mode=='risk':
                    rv=encode(model,norm(reverse_features(va['x'])));re=encode(model,norm(reverse_features(east['x'])))
                    vs=neighbour_score(rv,a,tr['shot'],tr['y'],exclude=np.full(len(rv),-1))
                    es=neighbour_score(re,c,east['shot'],east['y'],exclude=east['shot'])
                    ea,_=pair_out_auc(re,c,east['shot'],east['y'])
                    key=f'reverse_{seed}';results[key]=dict(val=metrics(va,vs),east=dict(auc=ea,estimator='same-history shot-pair AUC'),
                        pooled_LOO_not_used_for_gate=metrics(east,es))
                    predictions[key+'_val']=vs;predictions[key+'_east']=es
                _durable_json(root/'results.json',results);_durable_json(root/'training.json',traces)
    aggregate={mode:dict(val_conditional_auc=float(np.mean([results[f'{mode}_{s}']['val']['conditional_macro_auc'] for s in p['seeds']])),
                        east_auc=float(np.mean([results[f'{mode}_{s}']['east']['auc'] for s in p['seeds']]))) for mode in (*MODES,'reverse')}
    accepted=qualifies(aggregate,results['raw']['east']['auc'],[results[f'risk_{s}']['east']['auc'] for s in p['seeds']])
    torch.save(heads,root/'encoders.pt');np.savez_compressed(root/'scores.npz',**predictions)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='completed',aggregate=aggregate,eligible_for_online_trial=accepted,
        candidate_promoted=False,scientific_goal_achieved=False,seconds=time.monotonic()-start,
        source_unchanged=True,artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.iterdir() if f.is_file()})
    _durable_json(root/'completed.json',done);print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--steps',type=int,default=1024)
    args=parser.parse_args()
    try:main(args.out,args.steps)
    except Exception as exc:
        if args.out.exists():_durable_json(args.out/'failed.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
