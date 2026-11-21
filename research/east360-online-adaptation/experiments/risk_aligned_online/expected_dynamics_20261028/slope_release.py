"""Single-factor release-lag diagnostic, same fitted source prior and signals."""
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
from ..warmstart_alignment_20261022.alignment import transform
from .matured_slope import SlopeMemory
from .slope_probe import errors


def delayed_forecast(trace):
    # before_delta uses the previous coefficient on the CURRENT observed
    # change. It is not the previous origin's forecast, which has another base.
    return trace['baseline']+trace['before_delta'].reshape_as(trace['baseline'])


def summary(rows):
    result={}
    for g in ('all','D','S'):
        sub=[r for r in rows if g=='all' or r['disruptive']==(g=='D')]
        result[g]=dict(shots=len(sub),errors={m:float(np.mean([r['errors'][m] for r in sub]))
            for m in sub[0]['errors']},improved_persistence=sum(r['errors']['delayed']<r['errors']['persistence'] for r in sub),
            improved_sync=sum(r['errors']['delayed']<r['errors']['memory'] for r in sub))
    return result


@torch.no_grad()
def run(reference,output):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':
        raise ValueError('new worktree output required')
    torch.set_num_threads(2);start=time.monotonic()
    p=read(reference,'protocol.json');done=read(reference,'complete.json')
    model=SlopeMemory(64,torch.tensor(p['prior']),torch.tensor(p['ridge']),torch.tensor(p['weights']))
    weights=torch.tensor(p['weights']);base,_=load_parent(DEFAULT_PARENT)
    stream=Path(p['stream']);mapping=read(stream,'input_map.json');hashes=read(stream,'input_hashes.json')
    if not done['complete'] or sha(DEFAULT_PARENT)!=PARENT_SHA256:
        raise ValueError('completed immutable reference required')
    plan=io23.ROOT/'SLOPE_RELEASE_PLAN.md';output.mkdir(parents=True);(output/'seals').mkdir()
    protocol=dict(scope='ONE_PATCH_RELEASE_LAG_NOT_RISK',reference=str(reference),
        reference_complete_sha256=sha(reference/'complete.json'),plan_sha256=sha(plan),
        code_sha256=sha(Path(__file__)),source_sha256=PARENT_SHA256,window=16,release_lag=1,
        frozen_parameters=True,source_cache=p['cache'],stream=p['stream'],order=p['order'])
    _durable_json(output/'protocol.json',protocol)
    cache=Path(p['cache']);inventory={r['shot']:r for r in read(cache,'inventory.json')}
    val=[]
    for sid in p['val_ids']:
        path=cache/'cache'/f'{sid}.pt'
        if inventory[sid]['role']!='val' or sha(path)!=inventory[sid]['cache_sha256']:
            raise ValueError('VAL identity changed')
        e=torch.load(path,map_location='cpu',weights_only=True);x,a=e['means'],e['complete']
        t=model.trace(x,a,mode='memory',return_before=True)
        traces=dict(memory=t,delayed={'forecast':delayed_forecast(t)},off={'forecast':t['baseline']})
        val.append(dict(shot=sid,disruptive=e['flag'],errors=errors(x,a,traces,weights)))
    _durable_json(output/'source_val.json',dict(summary=summary(val),rows=val))
    rows=[]
    for old in done['rows']:
        sid,i=old['shot'],old['index']
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('EAST input changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        sig={k:raw[k] for k in ('values','available')}
        if i>20:sig=transform(sig,mapping)
        x,a=patch_signal(sig,base.center,base.scale);t=model.trace(x,a,mode='memory',return_before=True)
        old_seal=read(reference/'seals',f'{i:04d}_{sid}.json')
        if hashlib.sha256(t['forecast'].numpy().tobytes()).hexdigest()!=old_seal['sha256']['memory']:
            raise ValueError('synchronous control not reproduced')
        delayed=delayed_forecast(t)
        _durable_json(output/'seals'/f'{i:04d}_{sid}.json',dict(shot=sid,index=i,outcome_used=False,
            sha256=hashlib.sha256(delayed.numpy().tobytes()).hexdigest()))
        flag,down=io23.read_outcome('EAST',sid)
        times=np.stack([raw['times'][(np.arange(len(x)-4)+h+1)*8-1] for h in range(1,5)],1)
        within=times<=down if flag else np.ones_like(times,dtype=bool)
        traces=dict(memory=t,delayed={'forecast':delayed},off={'forecast':t['baseline']})
        rows.append(dict(shot=sid,index=i,disruptive=flag,errors=errors(x,a,traces,weights,within)))
    result={k:summary(sub) for k,sub in (('full360',rows),('cold20',rows[:20]),('suffix340',rows[20:]))}
    checks={g+'_'+b:result['full360'][g]['errors']['delayed']<result['full360'][g]['errors'][b]
            for g in ('all','D','S') for b in ('off','persistence')}
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or sha(plan)!=protocol['plan_sha256']:
        raise ValueError('source or plan changed')
    _durable_json(output/'complete.json',dict(complete=True,results=result,rows=rows,
        gate=dict(eligible=all(checks.values()),checks=checks),seconds=time.monotonic()-start,
        source_classifier_unchanged=True,risk_predictions_computed=False))
    print(dict(source_val=summary(val),east_full360=result['full360'],checks=checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.reference,args.output)
