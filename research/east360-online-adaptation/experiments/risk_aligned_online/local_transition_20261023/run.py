"""Learn two equally sized internal adapters, seal six full EAST curves."""
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
from .model import LocalTransition

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/局部动力学快参数_事前约定_20261023.md'
VARIANTS={'static_prior':('static_trained','prior'),'static_fit':('static_trained','fit'),
          'adaptive_prior':('adaptive_trained','prior'),'adaptive_fit':('adaptive_trained','fit'),
          'adaptive_shuffled':('adaptive_trained','shuffled')}
NAMES=('parent',*VARIANTS)


def fit_models(parent,source,target,output):
    first=LocalTransition(copy.deepcopy(parent.base)).cuda().eval()
    models={'static_trained':first,'adaptive_trained':copy.deepcopy(first)}
    assert sum(p.numel() for p in first.parameters() if p.requires_grad)==2055
    opts={k:torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001) for k,m in models.items()}
    initial=state_cpu(parent.base); rs,rt=np.random.default_rng(131),np.random.default_rng(2131)
    groups_s,groups_t=pools(source),pools(target); log=[]
    with (output/'training_manifest.jsonl').open('x') as manifest:
        for step in range(512):
            sx,sm,sy,sr=balanced_batch(source,groups_s,rs,parent.cfg)
            tx,tm,ty,tr=balanced_batch(target,groups_t,rt,parent.cfg)
            x,mask,y=torch.cat((sx,tx)),torch.cat((sm,tm)),torch.cat((sy,ty))
            manifest.write(json.dumps(dict(step=step,source=sr,target=tr))+'\n')
            record=dict(step=step)
            for name,model in models.items():
                modes=('prior','fit') if name=='adaptive_trained' else ('prior','prior')
                opts[name].zero_grad(set_to_none=True); bce=[]; dynamics=[]
                for mode in modes:
                    result=model(x,mask,mode=mode)
                    bce.append(F.binary_cross_entropy_with_logits(result['logit'],y))
                    dynamics.append(result['forecast_loss'])
                risk=torch.stack(bce).mean(); dynamic=torch.stack(dynamics).mean()
                loss=risk+.1*dynamic
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite outer loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                opts[name].step()
                record[name]=dict(risk=float(risk.detach()),dynamics=float(dynamic.detach()),gradient_norm=float(norm))
            log.append(record)
            if (step+1)%128==0:print(json.dumps(dict(stage='fit',**record)),flush=True)
    _durable_json(output/'training.json',log)
    for name,model in models.items():
        assert_unchanged(model.base,initial);model.requires_grad_(False);model.eval()
        torch.save(dict(model=state_cpu(model)),output/f'{name}.pt')
    return models


def predict(models,signal,device='cuda'):
    if set(signal)!={'values','available'}:raise ValueError('signals only')
    curves={a:[] for a in VARIANTS}
    stats={a:dict(points=0,fit_observed=0,fast_changed=0,fast_delta_sum=0.,
                  held_error_sum=0.,held_targets=0,correction_norm_sum=0.) for a in VARIANTS}
    for lo in range(0,len(signal['values']),32):
        windows=[simple_data.window(signal,t,128) for t in range(lo,min(lo+32,len(signal['values'])))]
        x,mask=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        for arm,(name,mode) in VARIANTS.items():
            model=models[name]; r=model(x,mask,mode=mode)
            curves[arm].append(r['logit'].sigmoid().detach().cpu().numpy())
            with torch.no_grad():
                delta=(r['fast']-model.prior[None]).square().sum((1,2)).sqrt()
                s=stats[arm];s['points']+=len(x);s['fit_observed']+=int((r['local_observations']>0).sum())
                s['fast_changed']+=int((delta>0).sum());s['fast_delta_sum']+=float(delta.sum())
                s['held_error_sum']+=float(r['forecast_error_sum']);s['held_targets']+=int(r['target_mask'].sum())
                s['correction_norm_sum']+=float(r['correction'].norm(dim=-1).sum())
    result={k:np.concatenate(v) for k,v in curves.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('nonfinite probabilities')
    return result,stats


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    rp,rv=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and rv['completed_sha256']==sha(REFERENCE/'completed.json')
    for file,digest in rp['code_sha256'].items():
        assert sha(io23.ROOT/file)==digest,file
    assert sha(REFERENCE/'conditional_moments.pt')==READ(REFERENCE/'preparation_receipts.json')['conditional_moments']['checkpoint_sha256']
    contract=load_test_contract('EAST');assert contract==rp['contract']
    ids=rp['selected_ids'];assert not set(ids)&set(contract['shot_ids'])
    parent,threshold=load_model();initial=state_cpu(parent)
    mapping=READ(REFERENCE/'input_map.json');output.mkdir(parents=True);(output/'shots').mkdir()
    paths={io23.ROOT/f for f in rp['code_sha256']}
    paths.update(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/task_geometry_20261023/run.py')
    paths.add(io23.ROOT/'experiments/risk_aligned_online/task_geometry_20261023/learning.py')
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_local_transition.py')
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):z.write(path,str(path.relative_to(io23.ROOT)))
    protocol=dict(source_sha256=PARENT_SHA256,source_checkpoint=str(DEFAULT_PARENT),reference=str(REFERENCE),
        parent_sha256=sha(REFERENCE/'conditional_moments.pt'),threshold=threshold,contract=contract,
        selected_ids=ids,source_train_ids=io23.source_roles()['train'],variants=VARIANTS,
        steps=512,learning_rate=.001,rank=8,ridge=.1,dynamics_weight=.1,
        slow_parameters=['key.weight','key.bias','prior','readout.weight'],slow_parameter_count=2055,
        plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in paths},
        reference_hashes={f:sha(REFERENCE/f) for f in ('protocol.json','completed.json','verification.json','input_map.json','prediction_receipts.json')},
        fit_origins='patch0..13 -> delta1..14; heldout transition14->15 never used in inner fit',
        temporary_state='recomputed within each causal128point window; no persistent cross-window/cross-shot state',
        independent_confirmation=False,no_alarm_change=True,no_inter_shot_update=True,
        outcome_gate='source TRAIN+selected20 only until all360x5 new curves sealed',
        budget='same20 label identities, extra512 slow-adapter training steps disclosed; equal capacity control included')
    _durable_json(output/'protocol.json',protocol)
    original_read=io23.read_outcome;access=[];source_ids=set(protocol['source_train_ids'])
    def guarded(device,sid):
        allowed=source_ids if device=='J-TEXT' else set(ids) if device=='EAST' else set()
        if sid not in allowed:raise AssertionError('outcome access before seals')
        access.append(dict(device=device,shot=int(sid)));return original_read(device,sid)
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
            predictions,stats=predict(models,transform({k:s[k] for k in ('values','available')},mapping))
            refpath=REFERENCE/'shots'/f'{sid}.npz';assert sha(refpath)==refseals[sid]['prediction_sha256']
            with np.load(refpath,allow_pickle=False) as z:
                np.testing.assert_array_equal(s['times'],z['times']);predictions['parent']=z['conditional_moments_static'].copy()
            for arm in VARIANTS:np.testing.assert_array_equal(predictions[arm][:15],predictions['parent'][:15])
            path=output/'shots'/f'{sid}.npz';receipt=output/'shots'/f'{sid}_stats.json'
            np.savez_compressed(path,times=s['times'],**predictions);_durable_json(receipt,stats)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),stats_sha256=sha(receipt),reference_sha256=sha(refpath)))
            if (index+1)%30==0:
                progress=dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(output/'prediction_receipts.json',seals);_durable_json(output/'pre_eval_label_access.json',access)
    rows={a:[] for a in NAMES}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in NAMES:rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()}
    for k,m in models.items():assert_unchanged(m,frozen[k])
    assert_unchanged(parent,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'per_shot.json',results)
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,original_parameters_unchanged=True,
        checkpoint_sha256={k:sha(output/f'{k}.pt') for k in models},metrics={a:dict(**public(r),
            very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])) for a,r in results.items()})
    _durable_json(output/'completed.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
