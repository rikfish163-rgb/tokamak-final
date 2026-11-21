"""Re-evaluate both sealed source candidates in the SAME TRAIN-defined metric."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import read, load_context_memory, FORECAST_PROFILE, SCALED_PROFILE
from .model import ContextAdapter
from .scaled import validate_weights


def errors(x, observed, forecast, weights):
    validate_weights(weights)
    if (x.ndim!=2 or x.shape[1]!=23 or len(x)<=4 or observed.shape!=x.shape
            or observed.dtype!=torch.bool or forecast.shape!=(len(x),4,23)):
        raise ValueError('complete four-horizon records required')
    if not torch.isfinite(x[observed]).all():
        raise ValueError('finite observed diagnostics required')
    clean=torch.where(observed,x,0.).double()
    target=torch.stack([clean[h:len(x)-4+h]-clean[:-4] for h in range(1,5)],1)
    mask=torch.stack([observed[h:len(x)-4+h]&observed[:-4] for h in range(1,5)],1)
    if not mask.any() or not torch.isfinite(forecast[:-4][mask]).all():
        raise ValueError('observed targets and finite predictions required')
    square=torch.where(mask,(forecast[:-4].double()-target).square(),0.)
    w=mask*weights.double()
    channel=(square*w).sum((0,1))/w.sum()
    return dict(raw=float(square.sum()/mask.sum()),weighted=float(channel.sum()),
                persistence=float((target.square()*w).sum()/w.sum()),
                channel_contributions=channel.cpu().tolist())


@torch.no_grad()
def run(reference,candidate):
    output=candidate/'same_metric_pair.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    p,q=read(reference,'protocol.json'),read(candidate,'protocol.json')
    if p['profile']!=FORECAST_PROFILE or q['profile']!=SCALED_PROFILE:
        raise ValueError('unweighted forecast v2 versus TRAIN-scaled v3 required')
    matched=('source_sha256','train_ids','val_ids','seed','epochs','learning_rate','accumulation_shots',
             'gradient_clip','threshold','feedback','readout','cache_root','cache_inventory_sha256',
             'forecast_weight','reference_hashes')
    if any(p[k]!=q[k] for k in matched):
        raise ValueError('mismatched source training contract')
    for epoch in (1,2):
        if read(reference,f'order_epoch{epoch}.json')!=read(candidate,f'order_epoch{epoch}.json'):
            raise ValueError('source training order changed')
    if read(reference,'val_epoch0.json')['summary']!=read(candidate,'val_epoch0.json')['summary']:
        raise ValueError('initial source risk changed')
    base,_=load_parent(DEFAULT_PARENT)
    models=[ContextAdapter(base.cuda()).cuda().eval() for _ in range(2)]
    receipts=[load_context_memory(m,r/'context8_epoch2.pt') for m,r in zip(models,(reference,candidate),strict=True)]
    for m in models:
        m.requires_grad_(False)
    weights=models[1].memory.forecast_weights
    cache=Path(q['cache_root'])
    if sha(cache/'inventory.json')!=q['cache_inventory_sha256']:
        raise ValueError('source cache changed')
    inv={r['shot']:r for r in read(cache,'inventory.json')}
    rows=[]
    for sid in q['val_ids']:
        path=cache/'cache'/f'{sid}.pt'
        if inv[sid]['role']!='val' or sha(path)!=inv[sid]['cache_sha256']:
            raise ValueError('source VAL cache changed')
        e=torch.load(path,map_location='cpu',weights_only=True)
        x,a=e['means'].cuda(),e['complete'].cuda()
        forecasts=[m.memory.trace(x,a,mode='memory')['forecast'] for m in models]
        old,new=[errors(x,a,f,weights) for f in forecasts]
        if old['persistence']!=new['persistence']:
            raise ValueError('forecast baselines differ')
        rows.append(dict(shot=sid,disruptive=e['flag'],old=old,new=new))
    summary={}
    for label,sub in (('all',rows),('D',[r for r in rows if r['disruptive']]),('S',[r for r in rows if not r['disruptive']])):
        summary[label]=dict(shots=len(sub),
            weighted_persistence=float(np.mean([r['new']['persistence'] for r in sub])),
            old_weighted=float(np.mean([r['old']['weighted'] for r in sub])),
            new_weighted=float(np.mean([r['new']['weighted'] for r in sub])),
            old_raw=float(np.mean([r['old']['raw'] for r in sub])),
            new_raw=float(np.mean([r['new']['raw'] for r in sub])),
            new_better_than_old=sum(r['new']['weighted']<r['old']['weighted'] for r in sub),
            new_better_than_persistence=sum(r['new']['weighted']<r['new']['persistence'] for r in sub))
    _durable_json(output,dict(scope='PAIRED_SOURCE_VAL160_NOT_EAST',model_updates=0,east_records_read=0,
        matched_fields=list(matched),source_receipts=receipts,
        same_metric='both forecast outputs scored with fixed v3 TRAIN weights and common valid masks',
        code_sha256=sha(Path(__file__)),forecast_weights_sha256=q['forecast_weights_sha256'],summary=summary,rows=rows))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path);parser.add_argument('candidate',type=Path)
    args=parser.parse_args();run(args.reference,args.candidate)
