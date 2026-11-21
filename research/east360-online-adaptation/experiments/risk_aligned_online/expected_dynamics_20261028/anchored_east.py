"""Complete360 transfer screen of the preregistered four forecast components."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .anchored import ARMS, AnchoredForecast
from .anchored_train import summary
from .forecast import observed_features


@torch.no_grad()
def run(output, source_root, stream):
    if output.absolute().parent!=io23.ROOT/'probe_outputs' or output.exists():
        raise ValueError('new worktree output required')
    torch.set_num_threads(2)
    source=read(source_root,'protocol.json');trained=read(source_root,'complete.json')
    if not trained['complete'] or source['candidate']!='anchor_no_mnm':raise ValueError('completed declared four arms required')
    weights=torch.tensor(source['weights']);base,_=load_parent(DEFAULT_PARENT)
    models={}
    for name,(a,m) in ARMS.items():
        checkpoint=source_root/name/'forecast_epoch10.pt'
        if sha(checkpoint)!=trained['results'][name]['checkpoint_sha256']:raise ValueError('source checkpoint changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['protocol_sha256']!=sha(source_root/'protocol.json'):raise ValueError('source protocol changed')
        model=AnchoredForecast(anchored=a,omit_mnm=m).eval().requires_grad_(False)
        model.load_state_dict(state['model']);models[name]=model
    prior=read(stream,'protocol.json');done=read(stream,'completed.json')
    order=read(stream,'order.json');ids=[r['shot'] for r in order]
    mapping=read(stream,'input_map.json');hashes=read(stream,'input_hashes.json')
    if (done['status']!='COMPLETED' or len(ids)!=360 or len(set(ids))!=360
            or prior['source_sha256']!=PARENT_SHA256 or prior['warmup']!=20
            or mapping['calibration_ids']!=ids[:20] or mapping['policy']!='density_gain'
            or sha(stream/'order.json')!=prior['order_sha256']):
        raise ValueError('complete fixed commissioning360 required')
    output.mkdir(parents=True);(output/'seals').mkdir()
    plan=io23.ROOT/'ANCHORED_FORECAST_PLAN.md'
    files=[Path(__file__),Path(__file__).parent/'anchored.py',Path(__file__).parent/'forecast.py',Path(__file__).parent/'anchored_train.py']
    protocol=dict(scope='COMPLETE360_AUXILIARY_FORECAST_NOT_RISK',arms=ARMS,candidate='anchor_no_mnm',
        source_root=str(source_root),source_protocol_sha256=sha(source_root/'protocol.json'),
        source_complete_sha256=sha(source_root/'complete.json'),stream=str(stream),order=ids,warmup=20,
        weights=source['weights'],plan_sha256=sha(plan),source_classifier_sha256=PARENT_SHA256,
        reference_hashes={f:sha(stream/f) for f in ('protocol.json','completed.json','order.json','input_hashes.json','input_map.json')},
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},optimizer_steps=0,
        classification_predictions=False,all_models_frozen=True)
    _durable_json(output/'protocol.json',protocol)
    rows={k:[] for k in ARMS};prefix=[];start=time.monotonic()
    for index,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('input changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal={k:raw[k] for k in ('values','available')}
        if index>=20:signal=transform(signal,mapping)
        x,a=patch_signal(signal,base.center,base.scale)
        if len(x)<=4:raise ValueError('no complete horizon; do not omit a stream shot')
        features=observed_features(x,a);predictions={k:m(features) for k,m in models.items()}
        _durable_json(output/'seals'/f'{index+1:04d}_{sid}.json',dict(shot=sid,index=index+1,outcome_used=False,
            predictions_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in predictions.items()}))
        flag,down=io23.read_outcome('EAST',sid)
        y=torch.stack([x[h:len(x)-4+h]-x[:-4] for h in range(1,5)],1)
        valid=torch.stack([a[h:len(x)-4+h]&a[:-4] for h in range(1,5)],1)
        times=np.stack([raw['times'][(np.arange(len(x)-4)+h+1)*8-1] for h in range(1,5)],1)
        within=times<=down if flag else np.ones_like(times,dtype=bool)
        w=valid*torch.tensor(within)[...,None]*weights
        if w.sum()<=0:raise ValueError('no observed scored target')
        persistence=float((y.square()*w).sum()/w.sum())
        for k,pred in predictions.items():
            error=float(((pred[:-4]-y).square()*w).sum()/w.sum())
            if not np.isfinite(error):raise ValueError('nonfinite forecast error')
            rows[k].append(dict(shot=sid,index=index+1,disruptive=flag,prediction=error,persistence=persistence))
        if index<20:prefix.append({k:raw[k] for k in ('shot','device','values','available')})
        if index==19:
            rebuilt=fit_prefix_map(mapping['source_statistics'],prefix,order=ids,count=20,policy='density_gain')
            if rebuilt!=mapping:raise ValueError('ended-prefix map reconstruction failed')
            prefix=[]
        if (index+1)%60==0:print(dict(shots=index+1,seconds=time.monotonic()-start),flush=True)
    if sum(r['disruptive'] for r in rows['anchor_no_mnm'])!=180:raise ValueError('class balance changed')
    results={k:{section:summary(sub) for section,sub in (('full360',v),('cold20',v[:20]),('suffix340',v[20:]))} for k,v in rows.items()}
    c=results['anchor_no_mnm']['full360']
    checks={g+'_forecast_better':c[g]['prediction']<c[g]['persistence'] for g in ('all','D','S')}
    checks.update(majority=c['all']['improved']>=181,safe_majority=c['S']['improved']>=91,
                  source_screen=trained['results']['anchor_no_mnm']['gate']['eligible'])
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during experiment')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or sha(plan)!=protocol['plan_sha256']:raise ValueError('parent/plan changed')
    completed=dict(complete=True,results=results,rows=rows,seconds=time.monotonic()-start,
        gate=dict(eligible=all(checks.values()),checks=checks,scope='COMPONENT_ONLY_NOT_INTRA_RISK_ACCEPTANCE'),
        source_unchanged=True,optimizer_steps=0,risk_predictions_computed=False)
    _durable_json(output/'complete.json',completed)
    for k,result in results.items():print(k,result['full360'],flush=True)
    print(completed['gate'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--stream',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.source_root,args.stream)
