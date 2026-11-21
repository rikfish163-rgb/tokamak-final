"""Matched source-only controller preparation. Never trains the risk model."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..event_adapt_20261022.data import attach_training_events
from ..event_adapt_20261022.train import load_source,public
from ..event_prefix_probe import DEFAULT_LABELS,LABEL_SHA256,read_events
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics,score
from ..fixed_test_protocol import score_fixed_test
from .model import ContextFeedback,operational_loss
from .stream import predict_pair

WARM=io23.ROOT/'experiments/risk_aligned_online/runs/paired_feedback_warm_20261022/warm.pt'
WARM_SHA='68139ff39a6275bfc7e15a9f2df3289750dd712f070318c865b3a3d1e48983cc'


def model_from_warm():
    if sha(WARM)!=WARM_SHA: raise ValueError('warm reconstruction artifact changed')
    ck=torch.load(WARM,weights_only=True,map_location='cpu')
    base,parent=load_parent(DEFAULT_PARENT)
    model=ContextFeedback(base,ck['model']['level_scale'])
    incompatible=model.load_state_dict(ck['model'],strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys)!={
        'controller.0.weight','controller.0.bias','controller.2.weight','controller.2.bias'}:
        raise ValueError('warm artifact has unexplained differences')
    model.cuda(); model.freeze_base()
    return model,float(parent['threshold'])


def snapshot(output):
    files=list(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).parents[1]/n for n in ('simple23.py','simple_data.py','simple_train.py',
        'io23.py','causal_input.py','fixed_test_protocol.py','event_prefix_probe.py',
        'masked_state_20261022/model.py','paired_feedback_20261022/model.py',
        'event_adapt_20261022/data.py','event_adapt_20261022/train.py',
        'redesign_20261004/prepare_prior.py')]
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f,str(f.relative_to(io23.ROOT)))
    return {str(f.relative_to(io23.ROOT)):sha(f) for f in files}


def prepare_events(model,shots,steps):
    rng=np.random.default_rng(9881)
    event_shots=[s for s in shots if (s['event_target']>=0).any()]
    ids=rng.permutation([s['shot'] for s in event_shots])
    holdout=set(int(i) for i in ids[:max(1,len(ids)//5)])
    groups={c:[] for c in range(6)}; validation=[]
    for s in event_shots:
        for c in groups:
            ends=np.flatnonzero((s['event_target']==c)&s['valid'])
            if len(ends):
                if s['shot'] in holdout: validation.append((s,ends,c))
                else: groups[c].append((s,ends))
    if not all(groups.values()): raise ValueError('all six weak classes need TRAIN examples')
    model.event_head.requires_grad_(True)
    opt=torch.optim.AdamW(model.event_head.parameters(),lr=.003,weight_decay=0.)
    receipts=[]
    for step in range(steps):
        examples=[]; labels=[]
        for _ in range(32):
            c=int(rng.integers(6)); s,ends=groups[c][int(rng.integers(len(groups[c])))]
            end=int(rng.choice(ends)); examples.append(simple_data.window(s,end,model.cfg.window));labels.append(c)
            receipts.append((s['shot'],end,c))
        x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*examples)]
        with torch.no_grad():
            h,v=model.encode(x,m)
            h=(h*v[:,:,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
        logits=model.event_head(h)
        loss=F.cross_entropy(logits,torch.tensor(labels,device='cuda'))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    truth=[]; predicted=[]
    with torch.no_grad():
        for s,ends,c in validation:
            examples=[simple_data.window(s,int(t),model.cfg.window) for t in ends[::8]]
            for lo in range(0,len(examples),128):
                x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*examples[lo:lo+128])]
                h,v=model.encode(x,m); h=(h*v[:,:,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
                predicted.extend(model.event_head(h).argmax(-1).cpu().tolist()); truth.extend([c]*len(x))
    confusion=np.zeros((6,6),int)
    for a,b in zip(truth,predicted): confusion[a,b]+=1
    model.freeze_base()
    return dict(training_examples=len(receipts),training_sha256=hashlib.sha256(json.dumps(receipts).encode()).hexdigest(),
        heldout_TRAIN_shots=sorted(holdout),heldout_window_count=len(truth),confusion=confusion.tolist(),
        accuracy=float(np.trace(confusion)/confusion.sum()),
        limitation='weak algorithmic A-F classes, not physics causes; correlated windows; TRAIN-subsplit only')


def evaluate(model,shots,threshold,*,use_event,uniform=False):
    rows={k:[] for k in ('static','intra')}; traces={}; updates=[]
    for s in shots:
        scores,info=predict_pair(model,{k:s[k] for k in ('values','available')},
                                device='cuda',use_event=use_event,uniform=uniform)
        for arm in rows:
            v=s['valid']
            rows[arm].append(dict(shot=s['shot'],times=s['times'][v],scores=scores[arm][v],
                                  is_disrupt=s['is_disrupt'],down=s['down']))
            traces[f'{s["shot"]}_{arm}']=scores[arm]
        traces[f'{s["shot"]}_times']=s['times'];traces[f'{s["shot"]}_valid']=s['valid']
        updates.append(dict(shot=s['shot'],**info))
    return {k:metrics(v,threshold) for k,v in rows.items()},traces,updates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['warm','point','shot','event'],required=True)
    p.add_argument('--initial',type=Path)
    p.add_argument('--steps',type=int,default=512)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.steps<1: p.error('positive training budget required')
    if args.mode!='warm' and args.initial is None: p.error('matched event-head preparation required')
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);begin=time.monotonic()
    model,threshold=model_from_warm()
    roles=io23.source_roles();adv=io23.advanced_times()
    print(json.dumps(dict(stage='load_source',mode=args.mode)),flush=True)
    train=load_source(roles['train'],adv)
    if sha(DEFAULT_LABELS)!=LABEL_SHA256: raise ValueError('event CSV changed')
    grouped,_=read_events(DEFAULT_LABELS)
    train=attach_training_events(train,grouped,allowed_train_ids=roles['train'])
    source={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    if args.initial is not None:
        receipt=json.loads((args.initial.parent/'completed.json').read_text())
        if sha(args.initial)!=receipt['checkpoint_sha256']: raise ValueError('initial artifact changed')
        state=torch.load(args.initial,map_location='cuda',weights_only=True)
        if state['config']['source_parent_sha256']!=PARENT_SHA256: raise ValueError('parent mismatch')
        model.load_state_dict(state['model'])
    args.output.mkdir(parents=True)
    config=dict(format='context-feedback-v1',mode=args.mode,steps=args.steps,seed=131,
        source_parent_sha256=PARENT_SHA256,warm_sha256=WARM_SHA,event_sha256=LABEL_SHA256,
        initial_sha256=sha(args.initial) if args.initial else None,
        source_roles=roles,threshold=threshold,source_label='max(25ms,AdvancedTime); unchanged',
        outer='point: original endpoint BCE; shot/event: max eligible risk relative to frozen threshold, D strict>10ms',
        learned='only channel-weight controller; optional frozen inferred A-F probability context',
        inner='2 SGD .01 on disposable embedding+firstblock from arrived128 samples; next32 samples only',
        sampling='512 complete TRAIN shots, alternating disruptive/safe, seed1622; no deletion of zero-gradient shots',
        selection='fixed last512; no VAL/TEST checkpoint or threshold selection',
        followup='prospective matched diagnostic after fifth-round failures; all arms reported even if VAL worse',
        east_accessed=False,independent_confirmation=False,code_sha256=snapshot(args.output))
    _durable_json(args.output/'config.json',config)
    if args.mode=='warm':
        event_report=prepare_events(model,train,args.steps)
        _durable_json(args.output/'events.json',event_report)
        test=score(model,load_source(roles['test'],adv),'cuda')
        static=score_fixed_test('J-TEXT',test,threshold)
        if (static['tp'],static['fp'])!=(100,13): raise AssertionError('qualified source reproduction failed')
        _durable_json(args.output/'static_test220.json',static)
        checkpoint=args.output/'warm.pt';torch.save(dict(model=model.cpu().state_dict(),config=config),checkpoint)
        done=dict(status='COMPLETED',event=event_report,static_test220=public(static))
    else:
        opt=torch.optim.AdamW(model.controller.parameters(),lr=.001,weight_decay=0.)
        rng=np.random.default_rng(1622);pools={y:[s for s in train if bool(s['is_disrupt'])==y] for y in (False,True)}
        receipts=[];history=[]
        for step in range(args.steps):
            group=pools[bool(step%2)];s=group[int(rng.integers(len(group)))];receipts.append(s['shot'])
            out,info=predict_pair(model,{k:s[k] for k in ('values','available')},device='cuda',
                use_event=args.mode=='event',training=True)
            if args.mode=='point':
                use=torch.as_tensor(s['valid'],device='cuda')
                target=torch.as_tensor(s['y'],device='cuda').float()
                loss=F.binary_cross_entropy_with_logits(out['intra'][use],target[use])
            else:
                loss=operational_loss(out['intra'],s['times'],s['valid'],
                    is_disrupt=s['is_disrupt'],down=s['down'],threshold=threshold)
            opt.zero_grad(set_to_none=True);loss.backward()
            g=torch.nn.utils.clip_grad_norm_(model.controller.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            history.append(dict(step=step+1,shot=s['shot'],loss=float(loss.detach()),grad_norm=float(g),
                                updates=len(info['updates'])))
            if (step+1)%64==0:
                print(json.dumps(dict(stage='source_controller',mode=args.mode,step=step+1,
                    loss=float(np.mean([r['loss'] for r in history[-64:]])),
                    zero_grad=sum(r['grad_norm']==0 for r in history[-64:]),seconds=time.monotonic()-begin)),flush=True)
        _durable_json(args.output/'history.json',history)
        _durable_json(args.output/'sampling.json',dict(shots=receipts,unique=len(set(receipts)),
            sha256=hashlib.sha256(json.dumps(receipts).encode()).hexdigest()))
        checkpoint=args.output/'last.pt';torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),checkpoint)
        print(json.dumps(dict(stage='VAL160_diagnostic',mode=args.mode)),flush=True)
        val,traces,updates=evaluate(model,load_source(roles['val'],adv),threshold,use_event=args.mode=='event')
        _durable_json(args.output/'val160.json',val)
        np.savez_compressed(args.output/'val160_traces.npz',**traces)
        _durable_json(args.output/'val160_updates.json',updates)
        done=dict(status='COMPLETED',val160={k:public(v) for k,v in val.items()},
                  training_zero_grad=sum(r['grad_norm']==0 for r in history))
    if any(not torch.equal(v,model.base.state_dict()[k].cpu()) for k,v in source.items()):
        raise AssertionError('immutable source changed')
    done.update(checkpoint_sha256=sha(checkpoint),source_unchanged=True,east_evaluated=False,
                seconds=time.monotonic()-begin,evidence='diagnostic, not independent confirmation')
    _durable_json(args.output/'completed.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':main()
