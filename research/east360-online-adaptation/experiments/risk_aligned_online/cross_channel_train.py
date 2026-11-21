"""Source-only preparation of short-history cross-diagnostic update rules.

All classifier weights and risk thresholds are immutable. A common masked head
is prepared on TRAIN, followed by equal sample-budget signal and risk-directed
head fitting. Risk-directed fitting differentiates through one internal step.
The source screen is a development triage, not evidence of EAST generalization.
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

from . import fast_meta
from .cross_channel_prefix import CrossChannelModel, episode_windows
from .fast_update import normalized_step
from .ffn_intra_20261005.source import split_shots
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent,load_source_train
from .simple_online import _durable_json,_hash

LENGTHS=(1,8,16,32,64,128)


def train_step(model,optimizer,e,*,mode,phase,length):
    if mode not in ('signal','risk'):raise ValueError('declared source objective required')
    expected={id(p) for p in model.decoder.parameters()}
    actual=[id(p) for g in optimizer.param_groups for p in g['params']]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('optimizer may change only auxiliary decoder')
    x,a,qx,qa=episode_windows(e,length)
    model.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True)
    total=dict(signal=0.,before=0.,after=0.,objective=0.)
    for i in range(len(x)):
        fast=model.initial_fast(1).requires_grad_()
        loss=model.reconstruction_loss(x[i:i+1],a[i:i+1],fast,phase=phase)
        gradient,=torch.autograd.grad(loss,fast,create_graph=mode=='risk',retain_graph=True)
        candidate=normalized_step(fast,fast.detach(),gradient)
        after=F.binary_cross_entropy_with_logits(model(qx[i:i+1],qa[i:i+1],candidate)['logit'],e['y'][i:i+1])
        with torch.no_grad():
            before=F.binary_cross_entropy_with_logits(model(qx[i:i+1],qa[i:i+1])['logit'],e['y'][i:i+1])
        objective=loss if mode=='signal' else after+.20*loss
        if not torch.isfinite(objective):raise FloatingPointError('nonfinite source auxiliary training')
        (objective/len(x)).backward()
        for k,v in dict(signal=loss,before=before,after=after,objective=objective).items():
            total[k]+=float(v.detach())/len(x)
    norm=torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(total,gradient_norm=float(norm),length=length)


def source_screen(rows):
    if not rows or {int(r['label']) for r in rows}!={0,1}:
        raise ValueError('both original source endpoint-label classes required')
    groups={}
    for label in (0,1):
        subset=[r for r in rows if int(r['label'])==label]
        groups[str(label)]=dict(n=len(subset),before=float(np.mean([r['before'] for r in subset])),
            after=float(np.mean([r['after'] for r in subset])))
    before=float(np.mean([r['before'] for r in rows]));after=float(np.mean([r['after'] for r in rows]))
    passed=after<before-1e-6 and all(g['after']<=g['before']+1e-6 for g in groups.values())
    return dict(passed=bool(passed),n=len(rows),before=before,after=after,classes=groups,
                criterion='paired mean BCE decreases >1e-6 and neither endpoint-label class worsens >1e-6',
                not_a_whole_shot_or_EAST_gain=True)


def evaluate(model,shots,pools):
    rng=np.random.default_rng(912);rows=[]
    for iteration in range(128):
        e=fast_meta.sample(shots,pools,model.cfg,rng,1,'cuda')
        length=LENGTHS[iteration%len(LENGTHS)];phase=iteration%4
        x,a,qx,qa=episode_windows(e,length)
        fast=model.initial_fast(1).requires_grad_()
        loss=model.reconstruction_loss(x,a,fast,phase=phase)
        gradient,=torch.autograd.grad(loss,fast)
        candidate=normalized_step(fast.detach(),fast.detach(),gradient.detach()).requires_grad_()
        newloss=model.reconstruction_loss(x,a,candidate,phase=phase)
        accept=bool(newloss<loss.detach())
        with torch.no_grad():
            before=F.binary_cross_entropy_with_logits(model(qx,qa)['logit'],e['y'])
            after=F.binary_cross_entropy_with_logits(model(qx,qa,candidate if accept else fast)['logit'],e['y'])
        rows.append(dict(length=length,phase=phase,label=int(e['y'].item()),accepted=accept,
            before=float(before),after=float(after),signal_before=float(loss.detach()),signal_after=float(newloss.detach())))
    return dict(screen=source_screen(rows),episodes=rows,
        by_length={str(length):dict(n=sum(r['length']==length for r in rows),
            before=float(np.mean([r['before'] for r in rows if r['length']==length])),
            after=float(np.mean([r['after'] for r in rows if r['length']==length]))) for length in LENGTHS})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('new output directory required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(4);torch.manual_seed(131)
    args.output.mkdir(parents=True)
    base,parent=load_parent(DEFAULT_PARENT)
    model=CrossChannelModel(base,site='embedding').cuda().eval()
    initial={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    model.decoder.requires_grad_(True)
    shots,ids=load_source_train()
    train,holdout=split_shots(shots)
    pools=fast_meta.episode_pools(train,model.cfg);validpools=fast_meta.episode_pools(holdout,model.cfg)
    root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in ('cross_channel_train.py','cross_channel_prefix.py','masked_prefix_train.py',
            'masked_prefix_feedback.py','fast_meta.py','fast_direction.py','fast_update.py',
            'ffn_intra_20261005/source.py','simple23.py','simple_data.py','io23.py',
            'redesign_20261004/prepare_prior.py','redesign_20261004/intra.py'):
            z.write(root/name,name)
    protocol=dict(scope='SOURCE TRAIN ONLY AUXILIARY PREPARATION, NO JTEXT ONLINE BRANCH',
        parent_sha256=PARENT_SHA256,source_classifier_changed=False,east_data_used=False,
        train_ids=[s['shot'] for s in train],holdout_ids=[s['shot'] for s in holdout],all_train_ids=ids,
        label='original JTEXT max25ms AdvancedTime endpoint labels',
        prefix_lengths=list(LENGTHS),query='one real time point after the observed inner window',
        query_history='same synthetic record start: inner L points, query min(L+1,128) points',
        masked_channels='index modulo4; selected channels hidden over whole observed window',
        original_missing_values_are_not_targets=True,
        warmup_steps=512,warmup_batch=32,branch_steps=512,branch_batch=4,
        branches=['signal','risk_embedding','risk_ffn'],shared_signal_head_across_sites=True,
        optimizer='AdamW lr1e-4 weight_decay0 clip1',seed=131,
        auxiliary_decoder='linear64 ->8x23 on original normalized patch states',
        signal_objective='masked observed SmoothL1',risk_objective='query BCE after one inner step +.20 masked loss',
        inner_step=.05,inner_radius=1.,unroll_steps=1,hard_acceptance_not_differentiated=True,
        source_screen='128 fixed TRAIN-held-out episodes, mean BCE improves and neither class worsens',
        no_EAST_candidate_selection=True,source_screen_not_whole_shot_qualification=True,
        new_hypotheses=['short-history source preparation','cross-channel rather than whole-time-patch mask',
                        'original input-value encoding rather than only lastFFN'],
        no_domain_scale_augmentation=True,not_pretrained_MAE_backbone=True,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json',protocol)
    started=time.monotonic();rng=np.random.default_rng(131)
    opt=torch.optim.AdamW(model.decoder.parameters(),lr=1e-4,weight_decay=0.)
    history=[]
    for iteration in range(512):
        e=fast_meta.sample(train,pools,model.cfg,rng,32,'cuda')
        length=int(rng.choice(LENGTHS));x,a,_,_=episode_windows(e,length)
        opt.zero_grad(set_to_none=True)
        loss=model.reconstruction_loss(x,a,model.initial_fast(len(x)),phase=iteration%4)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.,error_if_nonfinite=True)
        opt.step();history.append(dict(step=iteration+1,length=length,loss=float(loss.detach())))
        if (iteration+1)%64==0:print(json.dumps(dict(stage='warmup',**history[-1])),flush=True)
    _durable_json(args.output/'warmup.json',history)
    common=deepcopy(model.decoder.state_dict());evaluations={};hashes={};signal=None
    for branch in ('signal','risk_embedding','risk_ffn'):
        site='ffn' if branch.endswith('ffn') else 'embedding'
        base,_=load_parent(DEFAULT_PARENT)
        model=CrossChannelModel(base,site=site).cuda().eval()
        model.decoder.load_state_dict(common);model.decoder.requires_grad_(True)
        opt=torch.optim.AdamW(model.decoder.parameters(),lr=1e-4,weight_decay=0.)
        rng=np.random.default_rng(3131);rows=[]
        for iteration in range(512):
            e=fast_meta.sample(train,pools,model.cfg,rng,4,'cuda')
            row=train_step(model,opt,e,mode='signal' if branch=='signal' else 'risk',
                           phase=iteration%4,length=int(rng.choice(LENGTHS)))
            rows.append(dict(step=iteration+1,**row))
            if (iteration+1)%64==0:print(json.dumps(dict(stage=branch,**rows[-1])),flush=True)
        model.decoder.requires_grad_(False)
        evaluations[branch]=evaluate(model,holdout,validpools)
        _durable_json(args.output/f'{branch}_training.json',rows)
        with (args.output/f'{branch}_head.pt').open('xb') as h:
            torch.save(dict(decoder=model.decoder.state_dict(),branch=branch,
                            parent_sha256=PARENT_SHA256,protocol=protocol),h)
        hashes[branch]=_hash(args.output/f'{branch}_head.pt')
        if branch=='signal':signal=deepcopy(model.decoder.state_dict())
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v.detach().cpu(),initial[k],rtol=0,atol=0)
        assert not model.memory.any()
    # The signal-only decoder is identical at zero state for either site;
    # evaluate its induced FFN step separately, not a duplicate training run.
    model.decoder.load_state_dict(signal)
    evaluations['signal_ffn']=evaluate(model,holdout,validpools)
    assert _hash(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'evaluation.json',evaluations)
    _durable_json(args.output/'completed.json',dict(base_bitwise_unchanged=True,east_data_used=False,
        source_head_sha256=hashes,elapsed_seconds=time.monotonic()-started,
        screens={k:v['screen'] for k,v in evaluations.items()}))
    print(json.dumps(dict(completed=True,screens={k:v['screen'] for k,v in evaluations.items()})),flush=True)


if __name__=='__main__':main()
