"""Cross-fitted auxiliary-head training, then full causal intra-only replay."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..increment_forecast_20261023.learning import IncrementFeedback, predict
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..target_feedback_meta_20261023.learning import episode_pool
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import episode

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/increment_forecast_cv20_20261023'
REF_SHA = '3f5e0d0003104b1c54e413badfa5d85147a59b377b566d1d8a532bf20ff23e61'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/连续更新梯度追溯_配对约定_20261023.md'
READ = lambda p: json.loads(p.read_text())
ARMS = ('tail2', 'tail8')
STEPS = 64


def load_model(fold):
    base, _ = load_parent(DEFAULT_PARENT)
    model = IncrementFeedback(base, 'increment').cuda()
    model.load_state_dict(torch.load(REFERENCE/f'fold{fold}'/'increment.pt', map_location='cuda', weights_only=True)['model'])
    model.set_stage('frozen')
    return model


def schedule(shots, fit_ids, fold):
    eligible = episode_pool(shots, fit_ids)
    groups = {d:[r for r in eligible if r['is_disrupt']==d and r['blocks']] for d in (False, True)}
    if len(groups[False]) != 12 or len(groups[True]) != 3:
        raise ValueError('all fifteen preparation identities must have eligible blocks')
    rng = np.random.default_rng(6131+fold); result = []
    for step in range(STEPS):
        pair = []
        for d in (False, True):
            row = groups[d][int(rng.integers(len(groups[d])))]; block = row['blocks'][int(rng.integers(len(row['blocks'])))]
            pair.append(dict(shot=row['shot'], disruptive=d, **block))
        result.append(dict(step=step, episodes=pair))
    return result, eligible


def train(warm, shots, entries, threshold, directory, fold):
    byid = {s['shot']:s for s in shots}; boundary = math.log(threshold/(1-threshold))
    frozen = state_cpu(warm); models = {a:copy.deepcopy(warm) for a in ARMS}
    opts = {}
    for a, model in models.items():
        model.base.forecast_head.requires_grad_(True)
        opts[a] = torch.optim.Adam(model.base.forecast_head.parameters(), lr=1e-4)
    history = []; start = time.monotonic()
    for entry in entries:
        row = dict(step=entry['step'])
        for arm, model in models.items():
            opts[arm].zero_grad(set_to_none=True); receipts = []
            for e in entry['episodes']:
                s = byid[e['shot']]; signal = {k:s[k] for k in ('values','available')}
                r = episode(model, signal, e['count'], e['ends'], e['disruptive'], boundary=boundary,
                            tail=int(arm[-1]), device='cuda')
                objective = r['forecast']+(r['risk']-r['before'])/.001
                if not torch.isfinite(objective): raise FloatingPointError('nonfinite head objective')
                (objective/2).backward()
                receipts.append(dict(**e, risk_before=float(r['before']), risk_after=float(r['risk'].detach()),
                    forecast=float(r['forecast'].detach()), objective=float(objective.detach()),
                    all_counts=r['all_counts'], differentiated=r['differentiated']))
                del objective, r
            assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith('base.forecast_head.'))
            norm = torch.nn.utils.clip_grad_norm_(model.base.forecast_head.parameters(), 1., error_if_nonfinite=True)
            opts[arm].step()
            row[arm] = dict(episodes=receipts, outer_gradient_norm=float(norm))
        history.append(row)
        if (entry['step']+1)%8 == 0:
            _durable_json(directory/'training.json', history)
            print(dict(stage='training',fold=fold,step=entry['step']+1,seconds=time.monotonic()-start),flush=True)
    for arm, model in models.items():
        model.set_stage('frozen')
        assert_unchanged(model, frozen, ('base.forecast_head.weight','base.forecast_head.bias'))
        torch.save(dict(model=state_cpu(model), config=model.cfg.to_dict()), directory/f'{arm}.pt')
    return models, history


def evaluate(root, plans, threshold):
    rows = {a:[] for a in ('static', 'warm_intra')+ARMS}; access = []
    for plan in plans:
        for sid in plan['held']:
            s = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')), 'EAST', {})
            access.append(dict(shot=sid,phase='after_all_seals'))
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(z['times'],s['times'])
                valid = s['valid']
                for a in rows:
                    rows[a].append(dict(shot=sid,times=s['times'][valid],scores=z[a][valid],is_disrupt=s['is_disrupt'],down=s['down']))
    result = {a:metrics(r, threshold, lead=30.) for a,r in rows.items()}
    _durable_json(root/'evaluation_access.json',access); _durable_json(root/'per_shot.json',result)
    summary = {a:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])) for a,r in result.items()}
    a,b,c = [summary[k] for k in ('tail8','static','tail2')]
    gate = bool(a['J'] > b['J'] and a['J'] > c['J'] and a['tp'] >= b['tp'] and a['fpr'] <= .2)
    return summary, gate


def main(root):
    if root.exists(): raise FileExistsError(root)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env local CUDA required')
    assert sha(REFERENCE/'completed.json') == REF_SHA
    old, protocol = READ(REFERENCE/'completed.json'), READ(REFERENCE/'protocol.json')
    assert READ(REFERENCE/'verification.json')['status']=='VERIFIED'
    for name,digest in old['artifact_sha256'].items(): assert sha(REFERENCE/name)==digest,name
    torch.set_num_threads(2); torch.manual_seed(131); start = time.monotonic()
    root.mkdir(parents=True); paths=set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_credit_horizon.py')
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None): paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths): z.write(p,str(p.relative_to(io23.ROOT)))
    p = dict(reference=str(REFERENCE),reference_sha256=REF_SHA,source_sha256=PARENT_SHA256,folds=protocol['folds'],
        threshold=protocol['threshold'],steps=STEPS,seed=131,sampling_seed=6131,tail_lengths=[2,8],
        inner_lr=.001,outer_lr=.0001,risk_difference_scale=.001,plan_sha256=sha(PLAN),
        code_sha256={str(path.relative_to(io23.ROOT)):sha(path) for path in paths},
        persistent_parameters=['base.forecast_head.weight','base.forecast_head.bias'],test360_accessed=False,
        independent_confirmation=False,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p); seals=[]; access=[]; neural_steps=0
    original_signal, original_outcome = simple_data.read_shot, io23.read_outcome
    for plan in p['folds']:
        f=plan['fold']; directory=root/f'fold{f}';directory.mkdir();(directory/'shots').mkdir()
        phase='fit'
        def read_signal(device,sid,**kwargs):
            if device!='EAST' or sid not in plan['fit']+plan['held']:raise AssertionError('out-of-fold signal')
            access.append(dict(fold=f,kind='signal',shot=sid,phase=phase));return original_signal(device,sid,**kwargs)
        def read_outcome(device,sid):
            if device!='EAST' or sid not in plan['fit'] or phase!='fit':raise AssertionError('outcome unavailable before seals')
            access.append(dict(fold=f,kind='outcome',shot=sid,phase=phase));return original_outcome(device,sid)
        warm=load_model(f); mapping=READ(REFERENCE/f'fold{f}'/'input_map.json');_durable_json(directory/'input_map.json',mapping)
        with patch.object(simple_data,'read_shot',side_effect=read_signal),patch.object(io23,'read_outcome',side_effect=read_outcome):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries, eligibility = schedule(shots,plan['fit'],f)
            _durable_json(directory/'schedule.json',entries);_durable_json(directory/'eligibility.json',eligibility)
            models,history=train(warm,shots,entries,p['threshold'],directory,f)
            neural_steps += sum(len(e['all_counts']) for r in history for a in ARMS for e in r[a]['episodes'])
            frozen={a:state_cpu(m) for a,m in models.items()};phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                with np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as prior:
                    curves=dict(static=prior['static'].copy(),warm_intra=prior['increment'].copy())
                updates={}
                for arm,model in models.items():
                    predictions,updates[arm],packets=predict(model,signal)
                    np.testing.assert_array_equal(predictions['static'],curves['static'])
                    curves[arm]=predictions['intra'];neural_steps+=sum(r['gradient_steps'] for r in updates[arm])
                    np.savez_compressed(directory/'shots'/f'{sid}_{arm}_forecasts.npz',**packets)
                path=directory/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**curves)
                _durable_json(directory/'shots'/f'{sid}_updates.json',updates)
                seals.append(dict(fold=f,shot=sid,sha256=sha(path)))
            for a,m in models.items():assert_unchanged(m,frozen[a])
        _durable_json(root/'access.json',access)
        print(dict(stage='fold_complete',fold=f,shots=len(seals),seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'seals.json',seals)
    summary,gate=evaluate(root,p['folds'],p['threshold'])
    assert summary['static']==old['metrics']['static'] and summary['warm_intra']==old['metrics']['increment']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=summary,passes_screen=gate,seconds=time.monotonic()-start,
        neural_steps=neural_steps,source_unchanged=True,test360_accessed=False,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
