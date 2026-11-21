"""Matched level/change key preparation and eight causal TEST curves."""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..labeled_alignment_20261023.learning import balanced_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import pools,sha
from ..task_geometry_20261023.run import REFERENCE,READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .model import ChangeConditioned,patches,fixed_scale

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/变化条件局部适配_事前约定_20261023.md'
VARIANTS={name+'_'+mode:(name,mode) for name in ('level','change') for mode in ('prior','fit','shuffled','mean')}
NAMES=('parent',*VARIANTS)


def batches(source,target,cfg,steps):
    rs,rt=np.random.default_rng(131),np.random.default_rng(2131)
    gs,gt=pools(source),pools(target)
    for step in range(steps):
        sx,sm,sy,sr=balanced_batch(source,gs,rs,cfg);tx,tm,ty,tr=balanced_batch(target,gt,rt,cfg)
        yield torch.cat((sx,tx)),torch.cat((sm,tm)),torch.cat((sy,ty)),dict(step=step,source=sr,target=tr)


def fit_models(parent,source,target,output):
    rows=[];manifest=[]
    with torch.no_grad():
        for x,m,y,r in batches(source,target,parent.cfg,32):
            rows.append(patches(parent.base,x,m));manifest.append(r)
        scale,count=fixed_scale(rows)
    _durable_json(output/'scale_fit.json',dict(scale=scale.cpu().tolist(),observations=count.cpu().tolist(),
        manifest=manifest,scope='first32 training batches only; fixed across all inference windows'))
    del rows
    first=ChangeConditioned(copy.deepcopy(parent.base),scale,'level').cuda().eval()
    models={'level':first,'change':copy.deepcopy(first)};models['change'].key_kind='change'
    assert sum(v.numel() for v in first.parameters() if v.requires_grad)==2055
    opts={k:torch.optim.Adam([p for p in v.parameters() if p.requires_grad],lr=.001) for k,v in models.items()}
    base=state_cpu(parent.base);history=[]
    with (output/'training_manifest.jsonl').open('x') as f:
        for x,m,y,r in batches(source,target,parent.cfg,512):
            if r['step']<32:assert r==manifest[r['step']]
            f.write(json.dumps(r)+'\n');record=dict(step=r['step'])
            for name,model in models.items():
                opts[name].zero_grad(set_to_none=True)
                before=model(x,m,mode='prior');after=model(x,m,mode='fit')
                risk=.5*(F.binary_cross_entropy_with_logits(before['logit'],y)+F.binary_cross_entropy_with_logits(after['logit'],y))
                dynamic=.5*(before['forecast_loss']+after['forecast_loss']);loss=risk+.1*dynamic
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                opts[name].step();record[name]=dict(risk=float(risk.detach()),dynamic=float(dynamic.detach()),gradient_norm=float(norm))
            history.append(record)
            if (r['step']+1)%128==0:print(dict(stage='fit',**record),flush=True)
    _durable_json(output/'training.json',history)
    for name,model in models.items():
        assert_unchanged(model.base,base);model.requires_grad_(False);model.eval()
        torch.save(dict(model=state_cpu(model),key_kind=name),output/f'{name}.pt')
    return models


def predict(models,signal,device='cuda'):
    if set(signal)!={'values','available'}:raise ValueError('signals only')
    curves={a:[] for a in VARIANTS}
    stats={a:dict(points=0,fit_observed=0,fast_changed=0,fast_delta_sum=0.,held_error_sum=0.,held_targets=0,correction_norm_sum=0.) for a in VARIANTS}
    for lo in range(0,len(signal['values']),32):
        windows=[simple_data.window(signal,t,128) for t in range(lo,min(lo+32,len(signal['values'])))]
        x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        for arm,(name,mode) in VARIANTS.items():
            model=models[name];r=model(x,m,mode=mode)
            curves[arm].append(r['logit'].sigmoid().detach().cpu().numpy())
            with torch.no_grad():
                delta=(r['fast']-model.prior[None]).norm(dim=(1,2))
                s=stats[arm];s['points']+=len(x);s['fit_observed']+=int((r['local_observations']>0).sum())
                s['fast_changed']+=int((delta>0).sum());s['fast_delta_sum']+=float(delta.sum())
                s['held_error_sum']+=float(r['forecast_error_sum']);s['held_targets']+=int(r['target_mask'].sum())
                s['correction_norm_sum']+=float(r['correction'].norm(dim=-1).sum())
    result={k:np.concatenate(v) for k,v in curves.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('nonfinite probability')
    return result,stats


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    rp,rv=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and rv['completed_sha256']==sha(REFERENCE/'completed.json')
    for f,h in rp['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    contract=load_test_contract('EAST');assert contract==rp['contract']
    ids=rp['selected_ids'];assert not set(ids)&set(contract['shot_ids'])
    parent,threshold=load_model();initial=state_cpu(parent);mapping=READ(REFERENCE/'input_map.json')
    output.mkdir(parents=True);(output/'shots').mkdir()
    paths={io23.ROOT/f for f in rp['code_sha256']};paths.update(Path(__file__).parent.glob('*.py'))
    for f in ('task_geometry_20261023/run.py','task_geometry_20261023/learning.py','local_transition_20261023/model.py',
              'context_separation_20261023/model.py','tests/test_change_conditioned.py'):
        paths.add(io23.ROOT/'experiments/risk_aligned_online'/f)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):z.write(path,str(path.relative_to(io23.ROOT)))
    p=dict(source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,reference=str(REFERENCE),
        parent_sha256=sha(REFERENCE/'conditional_moments.pt'),threshold=threshold,contract=contract,selected_ids=ids,
        source_train_ids=io23.source_roles()['train'],variants=VARIANTS,steps=512,learning_rate=.001,ridge=.1,
        dynamics_weight=.1,slow_parameters=['key.weight','key.bias','prior','readout.weight'],slow_parameter_count=2055,
        plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths},
        reference_hashes={f:sha(REFERENCE/f) for f in ('protocol.json','verification.json','completed.json','input_map.json','prediction_receipts.json')},
        fit_contract='13 changes->nextchanges; query last change held out, masks require three complete patches',
        source_and_target_preparation_same=True,temporary_state='per-window closed form; no cross-shot carry',
        independent_confirmation=False,no_alarm_change=True,no_inter=True,
        test_label_access='class split metadata known; no HDF outcome before all new curves sealed',
        budgets='same20 label identities; extra32 scale batches and512 matched adapter preparation batches disclosed')
    _durable_json(output/'protocol.json',p)
    original=io23.read_outcome;access=[];source_ids=set(p['source_train_ids'])
    def guarded(device,sid):
        allowed=source_ids if device=='J-TEXT' else set(ids) if device=='EAST' else set()
        if sid not in allowed:raise AssertionError('TEST outcome before seals')
        access.append(dict(device=device,shot=int(sid)));return original(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guarded):
        source,_=load_source_train();target=[]
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            s.update(transform({k:s[k] for k in ('values','available')},mapping));target.append(s)
        models=fit_models(parent,source,target,output);del source,target
        frozen={k:state_cpu(m) for k,m in models.items()};seals=[]
        refseals={r['shot']:r for r in READ(REFERENCE/'prediction_receipts.json')}
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            curves,stats=predict(models,transform({k:s[k] for k in ('values','available')},mapping))
            old=REFERENCE/'shots'/f'{sid}.npz';assert sha(old)==refseals[sid]['prediction_sha256']
            with np.load(old,allow_pickle=False) as z:
                np.testing.assert_array_equal(s['times'],z['times']);curves['parent']=z['conditional_moments_static'].copy()
            for a in VARIANTS:np.testing.assert_array_equal(curves[a][:23],curves['parent'][:23])
            path=output/'shots'/f'{sid}.npz';sp=output/'shots'/f'{sid}_stats.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(sp,stats)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),stats_sha256=sha(sp),reference_sha256=sha(old)))
            if (index+1)%60==0:
                progress=dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(output/'progress.json',progress);print(progress,flush=True)
        _durable_json(output/'prediction_receipts.json',seals);_durable_json(output/'pre_eval_label_access.json',access)
    results={a:[] for a in NAMES}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in NAMES:results[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    scored={a:score_fixed_test('EAST',r,threshold) for a,r in results.items()}
    for name,model in models.items():assert_unchanged(model,frozen[name])
    assert_unchanged(parent,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'per_shot.json',scored)
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,original_parameters_unchanged=True,
        checkpoint_sha256={name:sha(output/f'{name}.pt') for name in models},protocol_sha256=sha(output/'protocol.json'),
        metrics={a:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])) for a,r in scored.items()})
    _durable_json(output/'completed.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
