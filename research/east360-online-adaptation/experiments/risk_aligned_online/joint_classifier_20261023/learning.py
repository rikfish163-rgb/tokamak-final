"""Supervise the existing classifier jointly; no added output correction."""
import json
import numpy as np
import torch
from torch.nn import functional as F

from ..deepcoral_prefix_20261022.learning import SIGNAL_KEYS,eligible_records,natural_batch
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.learning import observed_pools,target_batch
from ..increment_forecast_20261023.learning import prepare_head,train_head_step
from ..labeled_alignment_20261023.learning import balanced_batch
from ..simple_train import pools
from ..simple_online import _durable_json


def parameter_names(model,scope):
    if scope not in ('first_head','full_classifier'):raise ValueError('explicit joint classifier scope required')
    names=set(model.encoder_names())|{'base.risk_head.weight','base.risk_head.bias'}
    if scope=='full_classifier':
        names|={n for n,_ in model.named_parameters()
                if n.startswith(('base.encoder.layers.1.','base.norm.')) or n=='base.fast_origin'}
    existing={n for n,_ in model.named_parameters()}
    if not names<=existing:raise ValueError('expected existing source classifier parameters')
    return tuple(n for n,_ in model.named_parameters() if n in names)


def configure(model,scope):
    names=parameter_names(model,scope);allowed=set(names);model.eval()
    for n,p in model.named_parameters():p.requires_grad_(n in allowed)
    return names


def prepare(model,source,target,output,*,scope):
    origin=state_cpu(model);names=configure(model,scope)
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(parameters,lr=1e-4);gs,gt=pools(source),observed_pools(target)
    rs,rt=np.random.default_rng(131),np.random.default_rng(2131);history=[]
    with (output/'risk_manifest.jsonl').open('x') as manifest:
        for step in range(512):
            sx,sm,sy,sr=balanced_batch(source,gs,rs,model.cfg)
            tx,tm,ty,tr=target_batch(target,gt,rt,model.cfg)
            manifest.write(json.dumps(dict(step=step,source=sr,target=tr))+'\n')
            opt.zero_grad(set_to_none=True)
            hs=model(sx,sm,mode='risk_state')['risk_state'];ht=model(tx,tm,mode='risk_state')['risk_state']
            bs=F.binary_cross_entropy_with_logits(model.base.risk_head(hs).squeeze(-1),sy)
            bt=F.binary_cross_entropy_with_logits(model.base.risk_head(ht).squeeze(-1),ty)
            loss=.5*(bs+bt)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint classification loss')
            loss.backward()
            if any(v.grad is None for v in parameters):raise AssertionError('selected parameter missing classification gradient')
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            opt.step();history.append(dict(step=step,source_bce=float(bs.detach()),target_bce=float(bt.detach()),gradient_norm=float(norm)))
            if (step+1)%128==0:print(dict(stage='risk',scope=scope,**history[-1]),flush=True)
    _durable_json(output/'risk_training.json',history);assert_unchanged(model,origin,names)
    # Refit exactly the existing direct predictor to the now-adapted features.
    # Its fitting must not change any classifier parameter.
    model.set_stage('frozen');prepare_head(model);before=state_cpu(model)
    sf,se=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in source],model.cfg,future=True)
    tf,te=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in target],model.cfg,future=True)
    _durable_json(output/'forecast_exclusions.json',dict(source=se,target=te))
    opt=torch.optim.Adam(model.base.forecast_head.parameters(),lr=3e-4)
    rs,rt=np.random.default_rng(4131),np.random.default_rng(5131);history=[]
    with (output/'forecast_manifest.jsonl').open('x') as manifest:
        for step in range(256):
            a=natural_batch(sf,rs,32,model.cfg,'cuda',future=True);b=natural_batch(tf,rt,32,model.cfg,'cuda',future=True)
            manifest.write(json.dumps(dict(step=step,source=a[-1],target=b[-1]))+'\n')
            x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
            loss=train_head_step(model,opt,x,m,t,tm);history.append(dict(step=step,loss=loss))
            if (step+1)%128==0:print(dict(stage='forecast',scope=scope,**history[-1]),flush=True)
    _durable_json(output/'forecast_training.json',history)
    assert_unchanged(model,before,('base.forecast_head.weight','base.forecast_head.bias'));model.set_stage('frozen')
    torch.save(dict(model=state_cpu(model),config=model.cfg.to_dict()),output/'prepared.pt')
    _durable_json(output/'preparation.json',dict(scope=scope,fit_ids=[s['shot'] for s in target],
        n_disrupt=sum(s['is_disrupt'] for s in target),trainable_names=names,trainable_count=sum(v.numel() for v in parameters),
        intra_names=model.encoder_names(),risk_steps=512,forecast_steps=256,
        changed_parameters=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),origin[k])]))
