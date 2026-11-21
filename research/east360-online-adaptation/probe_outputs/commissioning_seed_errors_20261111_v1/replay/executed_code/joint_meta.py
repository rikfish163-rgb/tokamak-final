"""Source-only joint learning of useful signal-driven temporary updates.

MT3-inspired mechanism experiment, not reproduction or deployment evidence.
The two arms share episodes, initialization and budget. Only the outer risk
objective differs. No EAST or TEST data is accessed here.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .fast_meta import episode_pools, sample, temporary_step,temporary_step_components
from .simple23 import Config, Simple23


def grad_norm(parameters):
    gradients=[p.grad.detach().norm().square() for p in parameters if p.grad is not None]
    return float(torch.stack(gradients).sum().sqrt()) if gradients else 0.


def alignment_loss(signal_gradient,risk_gradient):
    target=risk_gradient.detach()
    valid=target.norm(dim=-1)>1e-8
    cosine=F.cosine_similarity(signal_gradient,target,dim=-1,eps=1e-8)
    # Exactly solved/saturated risk examples have no supervised direction.
    # They contribute no alignment target, rather than an arbitrary vector.
    return ((1.-cosine)*valid).sum()/valid.sum().clamp_min(1)


def train_step(model, optimizer, e, *, mode, signal_weight=.2, risk_after_weight=None):
    if mode not in ('meta','joint','aligned') or not np.isfinite(signal_weight) or signal_weight<0:
        raise ValueError('invalid joint objective')
    if model.cfg.fast_site!='tokens':
        raise ValueError('token adaptation with double-backward math attention required')
    if risk_after_weight is not None and (not np.isfinite(risk_after_weight)
                                         or not 0 <= risk_after_weight <= 1):
        raise ValueError('risk_after_weight must be finite and in [0, 1]')
    model.eval()
    model.zero_grad(set_to_none=True)
    origin_flag=model.fast_origin.requires_grad
    # Both arms freeze the origin. In token mode the temporary offset is
    # fast-origin; do not train the reference through a detached fast copy.
    model.fast_origin.requires_grad_(False)
    try:
        if mode=='aligned':
            updated,signal,fast,signal_gradient=temporary_step_components(model,e,create_graph=True)
            losses=F.binary_cross_entropy_with_logits(model(e['qx'],e['qa'],fast)['logit'],e['y'],reduction='none')
            risk_gradient,=torch.autograd.grad(losses.sum(),fast,retain_graph=True)
            alignment=alignment_loss(signal_gradient,risk_gradient)
            before=losses.mean()
        else:
            updated,signal=temporary_step(model,e,create_graph=True)
            before=F.binary_cross_entropy_with_logits(model(e['qx'],e['qa'])['logit'],e['y'])
            alignment=before.new_zeros(())
        after=F.binary_cross_entropy_with_logits(model(e['qx'],e['qa'],updated)['logit'],e['y'])
        # An alignment penalty alone does not optimize performance AFTER the
        # temporary step. Both adaptation-aware modes backpropagate that task
        # loss; joint remains the explicit non-meta control.
        after_weight=(.5 if mode in ('meta','aligned') else 0.) if risk_after_weight is None else float(risk_after_weight)
        risk=(1.-after_weight)*before+after_weight*after if after_weight else before
        objective=risk+signal_weight*signal+.1*alignment
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite joint objective')
        objective.backward()
        log=dict(objective=float(objective.detach()),risk_before=float(before.detach()),
                 risk_after=float(after.detach()),signal_loss=float(signal.detach()),
                 alignment_loss=float(alignment.detach()),risk_after_weight=after_weight,
                 encoder_grad_norm=grad_norm(model.encoder.parameters()),
                 head_grad_norm=grad_norm(model.forecast_head.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        return log
    finally:
        model.fast_origin.requires_grad_(origin_flag)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['meta','joint','aligned'],required=True)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--fast-rule',choices=['raw','normalized'],default='raw')
    parser.add_argument('--risk-after-weight',type=float,default=None,
                        help='outer task weight after the temporary step; aligned defaults to 0.5')
    args=parser.parse_args()
    if min(args.steps,args.batch_size)<1 or not np.isfinite(args.lr) or args.lr<=0:
        parser.error('positive finite budget required')
    if args.risk_after_weight is not None and (not np.isfinite(args.risk_after_weight)
                                              or not 0 <= args.risk_after_weight <= 1):
        parser.error('risk-after-weight must be in [0, 1]')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    rng=np.random.default_rng(131)
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=replace(Config(**parent['model_config']),fast_site='tokens',fast_rule=args.fast_rule)
    model=Simple23(cfg,parent['source_center'],parent['source_scale']).cuda()
    model.load_state_dict(parent['model'])
    roles=simple_data.source_roles()
    advanced=simple_data.advanced_times()
    train=simple_train.load_shots(roles['train'],advanced)
    pools=episode_pools(train,cfg)
    protocol=dict(mode=args.mode,steps=args.steps,batch_size=args.batch_size,lr=args.lr,fast_rule=args.fast_rule,
                  seed=131,signal_weight=.2,
                  meta_risk_weight=(.5 if args.mode in ('meta','aligned') else 0.) if args.risk_after_weight is None else args.risk_after_weight,
                  alignment_weight=.1 if args.mode=='aligned' else 0.,
                  parent_sha256=simple_train.sha(args.checkpoint),
                  train_ids=roles['train'],val_ids=roles['val'],
                  evidence='DEVELOPMENT_ONLY; no inherited source qualification',
                  code_sha256={n:simple_train.sha(Path(__file__).with_name(n)) for n in
                               ('joint_meta.py','fast_meta.py','fast_update.py','fast_direction.py','simple23.py',
                                'simple_train.py','simple_data.py')})
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    history=[]
    digest=hashlib.sha256()
    for step in range(args.steps):
        e=sample(train,pools,cfg,rng,args.batch_size,'cuda')
        for key in sorted(e):
            digest.update(key.encode())
            digest.update(e[key].detach().cpu().numpy().tobytes())
        log=train_step(model,opt,e,mode=args.mode,risk_after_weight=args.risk_after_weight)
        history.append(log)
        if (step+1)%64==0:
            print(json.dumps(dict(step=step+1,**log)),flush=True)
    protocol['episode_sha256']=digest.hexdigest()
    (args.output/'training.json').write_text(json.dumps(dict(protocol=protocol,history=history),indent=2))
    val=simple_train.load_shots(roles['val'],advanced)
    scored=simple_train.score(model,val,'cuda')
    fixed=simple_train.metrics(scored,parent['threshold'])
    calibrated=simple_train.select_threshold(scored,fpr_cap=.1)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # Build a fresh checkpoint: inherited validation/TEST acceptance would be
    # invalid because the classifier changed. This script never reads TEST.
    result=dict(model=state,model_config=cfg.to_dict(),source_center=parent['source_center'],
                source_scale=parent['source_scale'],threshold=calibrated['threshold'],
                validation=calibrated,joint_training=protocol,forecast_trained=True)
    torch.save(result,args.output/'source.pt')
    report=dict(fixed_parent_threshold=fixed,validation_calibrated=calibrated,
                checkpoint_sha256=simple_train.sha(args.output/'source.pt'))
    (args.output/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{n:v for n,v in m.items() if n!='predictions'}
                      for k,m in report.items() if isinstance(m,dict)}),flush=True)


if __name__=='__main__':
    main()
