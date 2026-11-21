"""Fit the same raw-BCE parent, changing only disputed EAST negative draws."""
import copy
import json

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import SIGNAL_KEYS,natural_batch,eligible_records
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..labeled_alignment_20261023.learning import balanced_batch
from ..preparation_factorial_20261023.learning import mapping_for,objective
from ..simple23 import fit_forecaster
from ..simple_train import pools,sha
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import transform
from .learning import paired_batch


def prepare(parent,fit_ids,source,source_stats,output,policy):
    model=copy.deepcopy(parent);origin=state_cpu(parent)
    target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in fit_ids]
    if sum(r['is_disrupt'] for r in target)!=3:raise ValueError('three disruptive preparation shots required')
    mapping=mapping_for(source_stats,target,fit_ids,'raw_bce');_durable_json(output/'input_map.json',mapping)
    target=[dict(r,**transform({k:r[k] for k in ('values','available')},mapping)) for r in target]
    groups_source,groups_target=pools(source),pools(target);names=set(model.encoder_names())
    for k,p in model.named_parameters():p.requires_grad_(k in names)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rs,rt,rr=[np.random.default_rng(i) for i in (131,2131,6131)];logs=[];replaced=0
    with (output/'risk_training_manifest.jsonl').open('x') as manifest,(output/'replacement_manifest.jsonl').open('x') as ledger:
        for step in range(512):
            sx,sm,sy,sr=balanced_batch(source,groups_source,rs,parent.cfg)
            tx,tm,ty,tr,change=paired_batch(target,groups_target,rt,rr,parent.cfg,policy)
            manifest.write(json.dumps(dict(step=step,source=sr,target=tr))+'\n')
            ledger.write(json.dumps(dict(step=step,**change))+'\n');replaced+=len(change['replaced'])
            opt.zero_grad(set_to_none=True)
            hs=model(sx,sm,mode='risk_state')['risk_state'];ht=model(tx,tm,mode='risk_state')['risk_state']
            loss,components=objective(model.base.risk_head,hs,ht,sy,ty,'raw_bce')
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite preparation')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step();logs.append(dict(step=step,**components,gradient_norm=float(norm)))
            if (step+1)%256==0:print(dict(stage='risk',fold=output.parent.name,policy=policy,**logs[-1]),flush=True)
    _durable_json(output/'risk_training.json',logs)
    assert_unchanged(model,origin,names);model.set_stage('frozen');pref=state_cpu(model)
    sf,se=eligible_records([{k:r[k] for k in SIGNAL_KEYS} for r in source],parent.cfg,future=True)
    tf,te=eligible_records([{k:r[k] for k in SIGNAL_KEYS} for r in target],parent.cfg,future=True)
    _durable_json(output/'forecast_exclusions.json',dict(source=se,target=te))
    opt=torch.optim.Adam(model.base.forecast_head.parameters(),lr=3e-4)
    rs,rt=np.random.default_rng(4131),np.random.default_rng(5131);logs=[]
    with (output/'forecast_manifest.jsonl').open('x') as manifest:
        for step in range(256):
            a=natural_batch(sf,rs,32,parent.cfg,'cuda',future=True);b=natural_batch(tf,rt,32,parent.cfg,'cuda',future=True)
            x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
            manifest.write(json.dumps(dict(step=step,source=a[-1],target=b[-1]))+'\n')
            loss=fit_forecaster(model.base,opt,x,m,t,tm);logs.append(dict(step=step,loss=loss))
    _durable_json(output/'forecast_training.json',logs)
    assert_unchanged(model,pref,('base.forecast_head.weight','base.forecast_head.bias'))
    model.set_stage('frozen');state=state_cpu(model)
    torch.save(dict(model=state,config=model.cfg.to_dict()),output/'parent.pt')
    _durable_json(output/'preparation_receipt.json',dict(policy=policy,fit_ids=fit_ids,checkpoint_sha256=sha(output/'parent.pt'),
        changed_parameters=[k for k in state if not torch.equal(state[k],origin[k])],replaced_negative_draws=replaced,
        encoder_parameters=sum(dict(model.named_parameters())[k].numel() for k in names),risk_steps=512,forecast_steps=256))
    return model,mapping,target
