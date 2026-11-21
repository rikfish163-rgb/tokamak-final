"""Source-only operational preparation of the short-prefix update rule.

Unlike endpoint BCE, the outer task uses safe-shot risk peaks and timely
disruptive segments. The actual 7,39,71,... update clock is replayed, with
truncated derivatives only through the last four updates (not a late reset).
This adapts the existing operational-objective idea in ffn_intra_20261005;
the experimental differences are cross-channel feedback and value embedding.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data
from .cross_channel_prefix import CrossChannelModel
from .cross_channel_shot_screen import HEADS
from .fast_update import normalized_step
from .ffn_intra_20261005.source import split_shots
from .redesign_20261004.intra import _batch_windows
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent,load_source_train
from .simple_online import _durable_json,_hash


def make_episode(shot,rng):
    if shot['device']!='J-TEXT':raise ValueError('source TRAIN offline episodes only')
    n=len(shot['times']);ticks=list(range(7,n-1,32));queries=[]
    for end in ticks:
        q=np.arange(end+1,min(end+33,n))
        q=q[np.asarray(shot['valid'])[q]]
        if shot['is_disrupt']:
            lead=np.round((shot['down']-shot['times'][q])*1000.,9)
            q=q[(lead>10.)&(lead<=125.)]
        queries.append(q.tolist())
    eligible=[i for i,q in enumerate(queries) if q]
    if not eligible:return None  # No source training target; never an EAST stream filter.
    stop=int(rng.choice(eligible))+1
    return dict(shot=int(shot['shot']),label=int(shot['is_disrupt']),
                ticks=ticks[:stop],queries=queries[:stop],differentiate_from=max(0,stop-4))


def episode_loss(model,shot,episode,*,mode,threshold):
    if mode not in ('signal','risk') or not 0<threshold<1:raise ValueError('declared mode/threshold required')
    if shot['device']!='J-TEXT' or int(shot['shot'])!=episode['shot']:
        raise ValueError('offline source episode identity required')
    device=model.center.device;raw={k:shot[k] for k in ('values','available')}
    fast=model.initial_fast(1).requires_grad_();origin=fast.detach().clone()
    logits=[];original=[];losses=[];accepted=0
    for i,(end,queries) in enumerate(zip(episode['ticks'],episode['queries'])):
        differentiable=mode=='risk' and i>=episode['differentiate_from']
        x,a=_batch_windows(simple_data,raw,[end],model,device)
        loss=model.reconstruction_loss(x,a,fast,phase=i%4)
        gradient,=torch.autograd.grad(loss,fast,create_graph=differentiable,retain_graph=True)
        candidate=normalized_step(fast,origin,gradient)
        after=model.reconstruction_loss(x,a,candidate,phase=i%4)
        if not torch.isfinite(loss+after):raise FloatingPointError('nonfinite source inner loss')
        if bool(after<loss.detach()):fast=candidate;accepted+=1
        if i>=episode['differentiate_from']:
            losses.append(loss)
            if queries:
                qx,qa=_batch_windows(simple_data,raw,queries,model,device)
                logits.append(model(qx,qa,fast.expand(len(queries),-1))['logit'])
                with torch.no_grad():original.append(model(qx,qa)['logit'])
        if not differentiable:fast=fast.detach().requires_grad_()
    if not logits:raise ValueError('at least one eligible post-update source query required')
    boundary=float(np.log(threshold/(1-threshold)))
    peak=torch.cat(logits).max();before=torch.cat(original).max()
    target=peak.new_tensor(float(episode['label']))
    risk=F.binary_cross_entropy_with_logits(peak-boundary,target)
    old=F.binary_cross_entropy_with_logits(before-boundary,target)
    signal=torch.stack(losses).mean()
    objective=signal if mode=='signal' else risk+.20*signal
    if not torch.isfinite(objective):raise FloatingPointError('nonfinite source outer objective')
    return objective,dict(shot=episode['shot'],label=episode['label'],
        before=float(old),after=float(risk.detach()),signal=float(signal.detach()),
        peak_before=float(before.sigmoid()),peak_after=float(peak.detach().sigmoid()),
        queries=sum(len(q) for q in episode['queries'][episode['differentiate_from']:]),
        updates=len(episode['ticks']),accepted=accepted,final_fast=model.summarize_fast(fast))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('new output directory required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(131)
    base,parent=load_parent(DEFAULT_PARENT);initial=deepcopy(base.state_dict())
    source=torch.load(HEADS/'signal_head.pt',map_location='cpu',weights_only=True)
    if source['parent_sha256']!=PARENT_SHA256 or source['branch']!='signal':raise ValueError('head provenance')
    shots,_=load_source_train();train,heldout=split_shots(shots)
    rng=np.random.default_rng(5131)
    pools={label:[s for s in train if int(s['is_disrupt'])==label and make_episode(s,rng) is not None]
           for label in (0,1)}
    if not all(pools.values()):raise ValueError('both source shot classes required')
    rng=np.random.default_rng(6131);schedule=[]
    for iteration in range(256):
        schedule.append([make_episode(pools[label][int(rng.integers(len(pools[label])))],rng) for label in (0,1)])
    byshot={int(s['shot']):s for s in train}
    _durable_json(args.output/'source_episodes.json',schedule)
    root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in ('cross_channel_operational.py','cross_channel_prefix.py','simple_data.py','io23.py',
            'simple23.py','fast_update.py','ffn_intra_20261005/source.py',
            'redesign_20261004/intra.py','redesign_20261004/prepare_prior.py'):
            z.write(root/name,name)
    protocol=dict(scope='SOURCE-ONLY OFFLINE UPDATE-RULE PREPARATION FOR EAST',
        parent_sha256=PARENT_SHA256,base_changed=False,east_data_used=False,
        initializer_sha256=_hash(HEADS/'signal_head.pt'),threshold=float(parent['threshold']),
        episode_sha256=_hash(args.output/'source_episodes.json'),train_ids=list(byshot),
        heldout_ids=[int(s['shot']) for s in heldout],steps=256,batch=2,seed=131,
        shot_balanced=True,query='safe observed windows; disruptive 10<lead<=125 ms on source',
        source_training_point_labels_changed=False,source_only_outer_warning_objective=True,
        update_clock='7,39,71,...; same replay acceptor; no late reset',
        gradient_truncation='last4 updates; earlier updates replayed with same head then detached',
        branch_budget='identical512 source episodes; risk branch uses more compute for higher derivatives',
        branches=['signal','risk_embedding'],outer_risk='BCE(max segment logit - original alarm logit, shot outcome)',
        risk_objective='outer_risk +.20 masked loss',signal_objective='masked loss only',
        checkpoint_selection='last step only; no EAST or VAL selection',
        optimizer='AdamW lr1e-4 wd0 clip1',code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json',protocol)
    started=time.monotonic();hashes={}
    for branch in ('signal','risk_embedding'):
        base,_=load_parent(DEFAULT_PARENT)
        model=CrossChannelModel(base,site='embedding').cuda().eval()
        model.decoder.load_state_dict(source['decoder']);model.decoder.requires_grad_(True)
        optimizer=torch.optim.AdamW(model.decoder.parameters(),lr=1e-4,weight_decay=0.)
        history=[]
        for iteration,episodes in enumerate(schedule):
            model.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True);rows=[]
            for e in episodes:
                objective,info=episode_loss(model,byshot[e['shot']],e,
                    mode='risk' if branch.startswith('risk') else 'signal',threshold=float(parent['threshold']))
                (objective/len(episodes)).backward();rows.append(info)
            norm=torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();history.append(dict(step=iteration+1,episodes=rows,gradient_norm=float(norm)))
            if (iteration+1)%32==0:
                print(json.dumps(dict(branch=branch,step=iteration+1,
                    before=float(np.mean([r['before'] for r in rows])),
                    after=float(np.mean([r['after'] for r in rows])))),flush=True)
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v.cpu(),initial[k],rtol=0,atol=0)
        _durable_json(args.output/f'{branch}_training.json',history)
        with (args.output/f'{branch}_head.pt').open('xb') as h:
            torch.save(dict(decoder=model.decoder.state_dict(),branch=branch,
                            parent_sha256=PARENT_SHA256,protocol=protocol),h)
        hashes[branch]=_hash(args.output/f'{branch}_head.pt')
    assert _hash(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'completed.json',dict(base_bitwise_unchanged=True,east_data_used=False,
        head_hashes=hashes,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
