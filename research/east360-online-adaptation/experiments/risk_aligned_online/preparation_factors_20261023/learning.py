"""Paired source-on/off preparation; keep old executed experiment code intact."""
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
from .design import risk_objective,gradient_relation


def prepare(model,source,target,output,*,use_source):
    origin=state_cpu(model);names=set(model.encoder_names())
    model.eval()
    for k,p in model.named_parameters():p.requires_grad_(k in names)
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(parameters,lr=1e-4)
    gs,gt=pools(source),observed_pools(target)
    rs,rt=np.random.default_rng(131),np.random.default_rng(2131);history=[];relations=[]
    with (output/'risk_manifest.jsonl').open('x') as manifest:
        for step in range(512):
            sx,sm,sy,sr=balanced_batch(source,gs,rs,model.cfg)
            tx,tm,ty,tr=target_batch(target,gt,rt,model.cfg)
            manifest.write(json.dumps(dict(step=step,source=sr,target=tr))+'\n')
            opt.zero_grad(set_to_none=True)
            hs=model(sx,sm,mode='risk_state')['risk_state'];ht=model(tx,tm,mode='risk_state')['risk_state']
            bs=F.binary_cross_entropy_with_logits(model.base.risk_head(hs).squeeze(-1),sy)
            bt=F.binary_cross_entropy_with_logits(model.base.risk_head(ht).squeeze(-1),ty)
            if (step+1)%64==0:relations.append(dict(step=step,**gradient_relation(bs,bt,parameters)))
            loss=risk_objective(bs,bt,use_source)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite support BCE')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            opt.step();history.append(dict(step=step,source_bce=float(bs.detach()),target_bce=float(bt.detach()),gradient_norm=float(norm)))
            if (step+1)%128==0:print(dict(stage='risk',directory=str(output),**history[-1]),flush=True)
    _durable_json(output/'risk_training.json',history);_durable_json(output/'gradient_relations.json',relations)
    assert_unchanged(model,origin,names)
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
            if (step+1)%128==0:print(dict(stage='forecast',directory=str(output),**history[-1]),flush=True)
    _durable_json(output/'forecast_training.json',history)
    assert_unchanged(model,before,('base.forecast_head.weight','base.forecast_head.bias'))
    model.set_stage('frozen')
    torch.save(dict(model=state_cpu(model),config=model.cfg.to_dict()),output/'prepared.pt')
    _durable_json(output/'preparation.json',dict(fit_ids=[s['shot'] for s in target],use_source=use_source,
        n_disrupt=sum(s['is_disrupt'] for s in target),available_endpoint_classes=[k for k in gt if gt[k]],
        changed_parameters=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),origin[k])],
        risk_steps=512,forecast_steps=256))
