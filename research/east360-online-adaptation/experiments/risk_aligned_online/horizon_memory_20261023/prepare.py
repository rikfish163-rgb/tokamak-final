"""Fit one fold from the qualified J-TEXT parent, never the all20 checkpoint."""
import copy
import json

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import SIGNAL_KEYS,natural_batch,eligible_records
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..labeled_alignment_20261023.learning import balanced_batch,conditional_moment_loss
from ..simple23 import fit_forecaster
from ..simple_train import pools,sha
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import fit_statistics,make_map,transform


def prepare(parent,fit_ids,source,source_stats,output):
    if len(fit_ids)!=15 or len(set(fit_ids))!=15:raise ValueError('distinct fifteen-shot fit required')
    model=copy.deepcopy(parent);origin=state_cpu(parent)
    target=[]
    for sid in fit_ids:
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        target.append(simple_data.attach_outcome(s,'EAST',{}))
    if sum(r['is_disrupt'] for r in target)!=3:raise ValueError('three disruptive training shots required')
    stats=fit_statistics([{k:r[k] for k in SIGNAL_KEYS} for r in target],device='EAST',allowed_ids=fit_ids)
    mapping=make_map(source_stats,stats,'density_gain')
    _durable_json(output/'input_map.json',mapping)
    target=[dict(r,**transform({k:r[k] for k in ('values','available')},mapping)) for r in target]
    groups_source,groups_target=pools(source),pools(target)
    names=set(model.encoder_names())
    for k,p in model.named_parameters():p.requires_grad_(k in names)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rs,rt=np.random.default_rng(131),np.random.default_rng(2131);logs=[]
    with (output/'risk_training_manifest.jsonl').open('x') as manifest:
        for step in range(512):
            sx,sm,sy,sr=balanced_batch(source,groups_source,rs,parent.cfg)
            tx,tm,ty,tr=balanced_batch(target,groups_target,rt,parent.cfg)
            manifest.write(json.dumps(dict(step=step,source=sr,target=tr))+'\n')
            opt.zero_grad(set_to_none=True)
            hs=model(sx,sm,mode='risk_state')['risk_state'];ht=model(tx,tm,mode='risk_state')['risk_state']
            bs=F.binary_cross_entropy_with_logits(model.base.risk_head(hs).squeeze(-1),sy)
            bt=F.binary_cross_entropy_with_logits(model.base.risk_head(ht).squeeze(-1),ty)
            align=conditional_moment_loss(hs,ht,sy,ty);loss=.5*(bs+bt)+align
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite fold preparation')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step()
            logs.append(dict(step=step,source_bce=float(bs.detach()),target_bce=float(bt.detach()),
                             alignment=float(align.detach()),gradient_norm=float(norm)))
            if (step+1)%128==0:print(dict(stage='fit_risk',fold=output.name,**logs[-1]),flush=True)
    _durable_json(output/'risk_training.json',logs)
    assert_unchanged(model,origin,names);model.set_stage('frozen');pref=state_cpu(model)
    sf,se=eligible_records([{k:r[k] for k in SIGNAL_KEYS} for r in source],parent.cfg,future=True)
    tf,te=eligible_records([{k:r[k] for k in SIGNAL_KEYS} for r in target],parent.cfg,future=True)
    _durable_json(output/'forecast_exclusions.json',dict(source=se,target=te))
    opt=torch.optim.Adam(model.base.forecast_head.parameters(),lr=3e-4)
    rs,rt=np.random.default_rng(4131),np.random.default_rng(5131);logs=[]
    with (output/'forecast_manifest.jsonl').open('x') as manifest:
        for step in range(256):
            a=natural_batch(sf,rs,32,parent.cfg,'cuda',future=True)
            b=natural_batch(tf,rt,32,parent.cfg,'cuda',future=True)
            x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
            manifest.write(json.dumps(dict(step=step,source=a[-1],target=b[-1]))+'\n')
            loss=fit_forecaster(model.base,opt,x,m,t,tm);logs.append(dict(step=step,loss=loss))
            if (step+1)%128==0:print(dict(stage='fit_forecast',fold=output.name,**logs[-1]),flush=True)
    _durable_json(output/'forecast_training.json',logs)
    assert_unchanged(model,pref,('base.forecast_head.weight','base.forecast_head.bias'))
    model.set_stage('frozen');state=state_cpu(model)
    torch.save(dict(model=state,config=model.cfg.to_dict()),output/'parent.pt')
    _durable_json(output/'preparation_receipt.json',dict(fit_ids=fit_ids,
        checkpoint_sha256=sha(output/'parent.pt'),changed_parameters=[k for k in state if not torch.equal(state[k],origin[k])],
        encoder_parameters=sum(dict(model.named_parameters())[k].numel() for k in names),
        risk_steps=512,forecast_steps=256,held_signals_used=False))
    return model,mapping,target
