"""Same-capacity level versus level-and-change trajectory learning."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory, components
from joint_trajectory_20261107 import JointTrajectory


FIT = j.r.ROOT/'probe_outputs/joint_motion_source_20261107_v1'


def calibrate_weight(base, model, shots, *, batches=16, batch=j.FIT_BATCH):
    """One fixed source-TRAIN gradient-scale ratio; no target score selection."""
    pools = j.eligible_origins(shots)
    rng = np.random.default_rng(j.SEED+2)
    parameters = tuple(model.parameters())
    records = []
    for _ in range(batches):
        e = j.sample_signals(shots, pools, rng, batch, base.center.device)
        token, valid, _ = j.features(base, e['x'], e['a'])
        output = model(token, valid, e['x'][:, -1], e['a'][:, -1])
        mask = e['mask'] & output['available'] & torch.isfinite(e['truth'])
        target = torch.asinh((torch.where(mask,e['truth'],model.center)-model.center)/model.scale)
        level, change = components(output['encoded'], output['probabilities'], target, mask)
        gl = torch.autograd.grad(level.mean(), parameters, retain_graph=True)
        gc = torch.autograd.grad(change.mean(), parameters)
        nl = torch.stack([v.square().sum() for v in gl]).sum().sqrt()
        nc = torch.stack([v.square().sum() for v in gc]).sum().sqrt()
        if not torch.isfinite(nl+nc) or nl <= 0 or nc <= 0:
            raise ValueError('positive finite source level/change gradients required')
        records.append(dict(level_norm=float(nl),change_norm=float(nc),ratio=float(nl/nc)))
    with torch.no_grad():
        model.motion_weight.fill_(np.median([v['ratio'] for v in records]))
    return dict(weight=float(model.motion_weight),batches=batches,batch=batch,
                seed=j.SEED+2,records=records)


def fit_step(models, optimizers, e):
    losses = {}
    for name, model in models.items():
        optimizers[name].zero_grad(set_to_none=True)
        objective = j.energy(model, e)
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite paired level/change fitting objective')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizers[name].step()
        losses[name] = float(objective.detach())
    return losses


def fit_both(base, models, shots, steps, seed, label):
    pools, rng = j.eligible_origins(shots), np.random.default_rng(seed)
    opts = {k:torch.optim.AdamW(v.parameters(),lr=j.FIT_LR,weight_decay=0.) for k,v in models.items()}
    history = {k:[] for k in models}
    for step in range(steps):
        e = j.sample_signals(shots,pools,rng,j.FIT_BATCH,base.center.device)
        e['tokens'],e['token_valid'],_ = j.features(base,e['x'],e['a'])
        losses = fit_step(models,opts,e)
        for name,value in losses.items():
            history[name].append(value)
        if (step+1)%64 == 0:
            print(label,step+1,losses,flush=True)
    return dict(steps=steps,eligible_shots=len(pools),history=history)


def same_state(model, path):
    expected = torch.load(path,map_location='cpu',weights_only=True)
    actual = model.state_dict()
    if set(actual) != set(expected) or any(not torch.equal(actual[k].cpu(),v) for k,v in expected.items()):
        raise ValueError('matched level control differs from the previous same-source predictor')


def fit(out):
    started = time.monotonic()
    j.r.configure()
    torch.manual_seed(j.SEED)
    base,_ = j.r.load_parent(j.r.DEFAULT_PARENT)
    base.cuda()
    frozen = {k:v.detach().clone() for k,v in base.state_dict().items()}
    level = JointTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).cuda().eval()
    ordered = OrderedTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).cuda().eval()
    loaded = ordered.load_state_dict(level.state_dict(),strict=False)
    if loaded.missing_keys != ['motion_weight'] or loaded.unexpected_keys:
        raise ValueError('only the nontrainable objective weight may differ at initialization')
    models = dict(level=level,motion=ordered)
    train,ids = j.load_source_train()
    calibration = calibrate_weight(base,ordered,train)
    _,cfg,_,(_,_,warm) = j.cohort('fixed360')
    mapping = j.r.read(j.r.COLD/'input_map.json')
    if warm != 20 or mapping['calibration_ids'] != cfg['order'][:20]:
        raise ValueError('original20 and fixed input coordinates required')
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'protocol.json',dict(status='MATCHED_LEVEL_AND_ORDERED_SOURCE',
        parent_sha256=j.PARENT_SHA256,cold20=cfg['order'][:20],source_train_ids=ids,
        source_steps=j.SOURCE_STEPS,cold_steps=j.COLD_STEPS,batch=j.FIT_BATCH,lr=j.FIT_LR,
        seed=j.SEED,shared_initial_network=True,shared_episodes=True,
        objective=dict(level='joint L2 energy score',change='lag8 joint difference energy score',
            lag=8,motion_weight=calibration['weight'],coefficient_selection='source initial gradient scale only'),
        calibration=calibration,source_classifier_training=False,target_label_fitting=False,
        source_val_or_test_selection=False,
        source_files={str(p):p.read_text() for p in (Path(__file__),
            j.r.ROOT/'joint_motion_20261107.py',j.r.ROOT/'joint_trajectory_20261107.py',
            j.r.ROOT/'run_joint_trajectory_20261107.py')}))
    print('fixed source gradient weight',calibration['weight'],flush=True)
    source = fit_both(base,models,train,j.SOURCE_STEPS,j.SEED,'source-paired')
    same_state(level,j.FIT/'source_predictor.pt')
    with (out/'source_predictor.pt').open('xb') as handle:
        torch.save(ordered.state_dict(),handle)
    cold = []
    with patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('EAST outcomes unavailable for preparation')):
        for sid in cfg['order'][:20]:
            raw = j.raw_for(sid,cfg['raw_sha256'])
            cold.append(j.r.prepare_actual(raw,21,mapping))
        preparation = fit_both(base,models,cold,j.COLD_STEPS,j.SEED+1,'ended20-paired')
    same_state(level,j.FIT/'prepared_predictor.pt')
    if any(not torch.equal(v,frozen[k]) for k,v in base.state_dict().items()):
        raise ValueError('ordered-objective fitting changed the qualified source classifier')
    with (out/'prepared_predictor.pt').open('xb') as handle:
        torch.save(ordered.state_dict(),handle)
    j.r.write_new(out/'fit.json',dict(status='FIT_COMPLETE',source=source,cold=preparation,
        source_train_ids=ids,coefficient=calibration['weight'],
        previous_level_source_exact=True,previous_level_prepared_exact=True,
        source_classifier_unchanged=True,seconds=time.monotonic()-started))
    print('completed matched ordered fit',round(time.monotonic()-started,2),'s',flush=True)


def predict(out):
    j.predict(out,fit_root=FIT,predictor_type=OrderedTrajectory)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','predict','score'))
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    {'fit':fit,'predict':predict,'score':j.recount}[args.phase](args.out)
