"""Paired inter/both with postshot forecast-head maintenance and token fast state.

Temporary adaptation observes only matured H25 signal targets. It is reset per
shot and never committed into the classifier. No labels guide within-shot steps.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import io23, simple_data
from .simple23 import Simple23, Config, fit_forecaster
from .replay_retraining import ReplayRetrainer
from .simple_online import run_arm, _durable_json, _hash
from .source_gate import qualify_source
from .fast_direction import episode
from .joint_meta import train_step as aligned_train_step


def forecast_batch(shots,rng,size,cfg,device,data_api=simple_data):
    pools=[]
    for shot in shots:
        valid=np.asarray(shot['valid'],bool)
        if len(valid) <= cfg.horizon:
            continue
        starts=np.flatnonzero(np.convolve(valid.astype(int),np.ones(cfg.horizon+1,int),'valid') == cfg.horizon+1)
        if len(starts):
            pools.append((shot,starts))
    if not pools:
        raise ValueError('no complete observed preterminal H25 target')
    examples=[]
    digest=hashlib.sha256()
    for _ in range(size):
        s,starts=pools[int(rng.integers(len(pools)))]
        end=int(rng.choice(starts))
        x,mask=data_api.window(s,end,cfg.window)
        future=s['values'][end+1:end+cfg.horizon+1]
        observed=s['available'][end+1:end+cfg.horizon+1]
        examples.append((x,mask,future,observed))
        digest.update(f'{s["shot"]}:{end};'.encode())
    return (*[torch.as_tensor(np.stack(v),device=device) for v in zip(*examples)],digest.hexdigest())


def classifier_sha(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not name.startswith('forecast_head.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def aligned_batch(shots,rng,size,cfg,device,data_api=simple_data):
    pools={0:[],1:[]}
    gap=cfg.horizon+1
    for shot in shots:
        valid=np.asarray(shot['valid'],bool)
        if len(valid)<=gap:
            continue
        starts=np.flatnonzero(np.convolve(valid.astype(int),np.ones(gap+1,int),'valid')==gap+1)
        for label in (0,1):
            eligible=starts[(np.asarray(shot['y'])[starts+gap]>.5)==bool(label)]
            if len(eligible):
                pools[label].append((shot,eligible))
    classes=[c for c in (0,1) if pools[c]]
    if not classes:
        raise ValueError('no observed preterminal alignment episode')
    examples=[]
    digest=hashlib.sha256()
    for _ in range(size):
        # Only classes already revealed in ended history can be sampled.
        group=pools[classes[int(rng.integers(len(classes)))]]
        shot,starts=group[int(rng.integers(len(group)))]
        origin=int(rng.choice(starts))
        examples.append(episode(shot,origin,cfg,device,data_api=data_api))
        digest.update(f'{shot["shot"]}:{origin};'.encode())
    e={k:torch.cat([x[k] for x in examples]) for k in examples[0] if k!='query_end'}
    return e,digest.hexdigest()


def fit_aligned_head(model,optimizer,e):
    flags=[(p,p.requires_grad) for p in model.parameters()]
    head={id(p) for p in model.forecast_head.parameters()}
    try:
        for p,_ in flags:
            p.requires_grad_(id(p) in head)
        return aligned_train_step(model,optimizer,e,mode='aligned')
    finally:
        for p,flag in flags:
            p.requires_grad_(flag)


class ForecastRetrainer:
    supports_intra=True

    def __init__(self,model,*,supervised_steps=16,forecast_steps=16,batch_size=64,maintenance='forecast'):
        if forecast_steps < 1:
            raise ValueError('forecast maintenance required')
        if maintenance not in ('forecast','aligned'):
            raise ValueError('unknown forecast maintenance')
        self.maintenance=maintenance
        self.supervised=ReplayRetrainer(model,mode='continue',steps=supervised_steps,
                                       scope='full',batch_size=batch_size)
        self.forecast_steps=forecast_steps
        self.batch_size=batch_size

    def __call__(self,model,shot,*,device='cpu',data_api=simple_data):
        before=deepcopy(model.state_dict())
        history=deepcopy(self.supervised.history)
        seen=set(self.supervised.seen)
        round_before=self.supervised.round
        try:
            result=self.supervised(model,shot,device=device,data_api=data_api)
            signature=classifier_sha(model)
            rng=np.random.default_rng(1000131+round_before)
            optimizer=torch.optim.AdamW(model.forecast_head.parameters(),lr=1e-4,weight_decay=0.)
            losses=[]
            digest=hashlib.sha256()
            for _ in range(self.forecast_steps):
                if self.maintenance=='aligned':
                    e,receipt=aligned_batch(list(self.supervised.history),rng,self.batch_size,model.cfg,device,data_api)
                    x,mask=e['x'],e['a']
                else:
                    x,mask,future,available,receipt=forecast_batch(
                        list(self.supervised.history),rng,self.batch_size,model.cfg,device,data_api)
                digest.update(receipt.encode())
                if not losses:
                    probe=(x,mask)
                    with torch.no_grad():
                        risk_before=model(*probe)['logit'].clone()
                losses.append(fit_aligned_head(model,optimizer,e) if self.maintenance=='aligned' else
                              fit_forecaster(model,optimizer,x,mask,future,available))
            with torch.no_grad():
                risk_delta=float((model(*probe)['logit']-risk_before).abs().max())
            if signature != classifier_sha(model) or risk_delta != 0:
                raise AssertionError('forecast maintenance changed classifier')
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite maintained model')
        except Exception:
            model.load_state_dict(before)
            self.supervised.history=history
            self.supervised.seen=seen
            self.supervised.round=round_before
            raise
        result.update(forecast_losses=losses,forecast_sample_sha256=digest.hexdigest(),
                      classifier_sha256=signature,forecast_risk_max_delta=risk_delta,
                      maintenance=self.maintenance)
        if self.supervised.round % 10 == 0:
            print(f'sealed/revealed/classifier+forecast fitted {self.supervised.round}',flush=True)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--order',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maintenance',choices=['forecast','aligned'],default='forecast')
    parser.add_argument('--forecast-steps',type=int,default=16)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior experiment')
    gate=qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError(gate['reasons'])
    order=json.loads(args.order.read_text())
    ids=[r['shot'] for r in order]
    if len(ids)!=360 or len(set(ids))!=360 or set(ids)!=set(io23.cohort_ids('EAST')):
        raise ValueError('complete locked360 order required')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=replace(Config(**checkpoint['model_config']),fast_site='tokens')
    model=Simple23(cfg,checkpoint['source_center'],checkpoint['source_scale'])
    model.load_state_dict(checkpoint['model'])
    model.cuda().eval()
    args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json',dict(
        parent_sha256=_hash(args.checkpoint),order_sha256=_hash(args.order),source_gate=gate,
        steps=16,forecast_steps=args.forecast_steps,forecast_lr=1e-4,fast_site='tokens',
        maintenance=args.maintenance,fast_rule=cfg.fast_rule,
        threshold=checkpoint['threshold'],scope='DEVELOPMENT_ONLY paired inter vs both',
        code_sha256={name:_hash(Path(__file__).with_name(name)) for name in
                     ('forecast_replay.py','replay_retraining.py','simple_online.py','simple23.py',
                      'simple_data.py','io23.py','simple_train.py','stage_metrics.py',
                      'fast_direction.py','fast_meta.py','fast_update.py','joint_meta.py')}))
    results={}
    for arm in ('inter','both'):
        net=deepcopy(model)
        updater=ForecastRetrainer(net,maintenance=args.maintenance,forecast_steps=args.forecast_steps)
        print(f'starting {arm}',flush=True)
        results[arm]=run_arm(net,order=order,arm=arm,threshold=float(checkpoint['threshold']),
                             output_dir=args.output/arm,device='cuda',startup_budget=30,
                             persistent_updater=updater)
        torch.save(net.state_dict(),args.output/f'{arm}_final.pt')
        print(arm,json.dumps({k:v for k,v in results[arm]['metrics_full'].items()
                              if k!='predictions'}),flush=True)
    for a,b in zip(results['inter']['rows'],results['both']['rows']):
        for field in ('sample_sha256','forecast_sample_sha256','classifier_sha256'):
            if a['persistent'][field]!=b['persistent'][field]:
                raise AssertionError(f'paired persistent training differs: {field}')
    _durable_json(args.output/'summary.json',results)


if __name__=='__main__':
    main()
