"""Matched zero-initialized level/change heads on the same fold classifiers."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import natural_batch, eligible_records, SIGNAL_KEYS
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..future_forecast_20261023.diagnostic import forecast_error, summarize
from ..horizon_memory_20261023.learning import FoldAccess
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, load_source_train, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import IncrementFeedback, prepare_head, train_head_step, predict

READ=lambda p:json.loads(p.read_text())
REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/forecast_token_scope_cv20_20261023'
REF_SHA='7e4ba1b6a51e69eb1eccc74b32d6dd6c905eec568d03f9f2af390fdb46dcf89d'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/变化量诊断反馈_事前约定_20261023.md'
HEADS=('absolute','increment')


def fit_heads(fit, source, target, out, reference):
    original=state_cpu(fit);cfg=fit.cfg
    sf,se=eligible_records([{k:r[k] for k in SIGNAL_KEYS} for r in source],cfg,future=True)
    tf,te=eligible_records([{k:r[k] for k in SIGNAL_KEYS} for r in target],cfg,future=True)
    _durable_json(out/'exclusions.json',dict(source=se,target=te));models={}
    for head in HEADS:
        model=IncrementFeedback(copy.deepcopy(fit.base),head).cuda()
        model.load_state_dict(original);prepare_head(model)
        opt=torch.optim.Adam(model.base.forecast_head.parameters(),lr=3e-4)
        rs,rt=np.random.default_rng(4131),np.random.default_rng(5131);log=[]
        with (out/f'{head}_manifest.jsonl').open('x') as manifest:
            for step in range(256):
                a=natural_batch(sf,rs,32,cfg,'cuda',future=True)
                b=natural_batch(tf,rt,32,cfg,'cuda',future=True)
                x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
                manifest.write(json.dumps(dict(step=step,source=a[-1],target=b[-1]))+'\n')
                loss=train_head_step(model,opt,x,m,t,tm);log.append(dict(step=step,loss=loss))
        assert (out/f'{head}_manifest.jsonl').read_bytes()==(reference/'full_manifest.jsonl').read_bytes()
        _durable_json(out/f'{head}_training.json',log)
        assert_unchanged(model,original,('base.forecast_head.weight','base.forecast_head.bias'))
        torch.save(dict(model=state_cpu(model),forecast_mode=head,config=cfg.to_dict()),out/f'{head}.pt')
        models[head]=model
    return models


def score(root,plans,threshold,center,scale):
    results={a:[] for a in ('static','legacy','absolute','increment')};forecast_rows=[];access=[]
    for plan in plans:
        folder=root/f'fold{plan["fold"]}';mapping=READ(folder/'input_map.json')
        for sid in plan['held']:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            access.append(dict(shot=sid,phase='after_all_seals'))
            with np.load(folder/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace['times'],s['times']);valid=s['valid']
                for a in results:
                    results[a].append(dict(shot=sid,times=s['times'][valid],scores=trace[a][valid],is_disrupt=s['is_disrupt'],down=s['down']))
            for h in HEADS:
                with np.load(folder/'shots'/f'{sid}_{h}_forecast.npz',allow_pickle=False) as pack:
                    for j,count in enumerate(pack['counts']):
                        raw=signal['values'][count:count+25];mask=signal['available'][count:count+25]&np.isfinite(raw)
                        lastvalid=pack['last_available'][j]&np.isfinite(pack['last_value'][j]);mask&=lastvalid[None]
                        truth=np.arcsinh((np.where(mask,raw,center).astype('float64')-center)/scale)
                        persistence=np.broadcast_to(np.arcsinh((np.where(lastvalid,pack['last_value'][j],center).astype('float64')-center)/scale),truth.shape)
                        static=forecast_error(pack['static'][j],truth,mask)['mean']
                        after=forecast_error(pack['intra'][j],truth,mask)['mean']
                        forecast_rows.append(dict(shot=sid,arm=h,origin=int(count)-1,observed=int(mask.sum()),
                            is_disrupt=bool(s['is_disrupt']),future_all_valid=bool(valid[count:count+25].all()),
                            future_any_positive=bool((s['y'][count:count+25]>.5).any()),
                            before=static,static=static,after=after,persistence=forecast_error(persistence,truth,mask)['mean']))
    _durable_json(root/'evaluation_access.json',access)
    per_shot={a:metrics(r,threshold,lead=30.) for a,r in results.items()}
    _durable_json(root/'per_shot.json',per_shot);_durable_json(root/'future_windows.json',forecast_rows)
    _durable_json(root/'future_summary.json',summarize(forecast_rows))
    return {a:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])) for a,r in per_shot.items()}


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REF_SHA
    old,p=READ(REFERENCE/'completed.json'),READ(REFERENCE/'protocol.json')
    for name,digest in old['artifact_sha256'].items():assert sha(REFERENCE/name)==digest,name
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    base,info=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
    origin=state_cpu(parent);source_ids=io23.source_roles()['train'];threshold=float(info['threshold'])
    root.mkdir(parents=True);paths=set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_increment_forecast.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):archive.write(path,str(path.relative_to(io23.ROOT)))
    protocol=dict(reference=str(REFERENCE),reference_sha256=REF_SHA,source_sha256=PARENT_SHA256,folds=p['folds'],
        threshold=threshold,plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256={str(path.relative_to(io23.ROOT)):sha(path) for path in paths},
        heads=HEADS,forecast_steps=256,seed=131,test360_accessed=False,independent_confirmation=False,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    _durable_json(root/'protocol.json',protocol)
    read_signal,read_outcome=simple_data.read_shot,io23.read_outcome;source_access=[]
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        source_access.append(sid);return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    _durable_json(root/'source_label_access.json',source_access);seals=[];steps=0
    for plan in p['folds']:
        f=plan['fold'];out=root/f'fold{f}';out.mkdir();(out/'shots').mkdir();ref=REFERENCE/f'fold{f}'
        fit=copy.deepcopy(parent);fit.load_state_dict(torch.load(ref/'full.pt',map_location='cuda',weights_only=True)['model']);fit.set_stage('frozen')
        mapping=READ(ref/'input_map.json');_durable_json(out/'input_map.json',mapping);access=FoldAccess(plan,source_ids)
        def signal_guard(device,sid,**kwargs):
            access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
        def forbidden_outcome(device,sid):raise AssertionError(f'no outcome needed before seals: {device}/{sid}')
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=forbidden_outcome):
            records=[apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')) for sid in plan['fit']]
            target=[dict(r,**transform({k:r[k] for k in ('values','available')},mapping)) for r in records]
            models=fit_heads(fit,source,target,out,ref);frozen={h:state_cpu(m) for h,m in models.items()};access.phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping);predictions={};updates={}
                with np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                    predictions.update(static=trace['static'].copy(),legacy=trace['full_all'].copy())
                for h,m in models.items():
                    scores,updates[h],future=predict(m,signal)
                    np.testing.assert_array_equal(scores['static'],predictions['static'])
                    predictions[h]=scores['intra'];np.savez_compressed(out/'shots'/f'{sid}_{h}_forecast.npz',**future)
                    steps+=sum(r['gradient_steps'] for r in updates[h])
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**predictions);_durable_json(up,updates)
                seals.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            for h,m in models.items():assert_unchanged(m,frozen[h])
        _durable_json(out/'access.json',access.log)
        print(dict(stage='fold_complete',fold=f,shots=len(seals),seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'prediction_receipts.json',seals)
    results=score(root,p['folds'],threshold,parent.base.center.cpu().numpy(),parent.base.scale.cpu().numpy())
    assert results['static']==old['metrics']['static'] and results['legacy']==old['metrics']['full_all']
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',seconds=time.monotonic()-start,metrics=results,neural_steps=steps,
        source_unchanged=True,test360_accessed=False,independent_confirmation=False,
        artifact_sha256={str(path.relative_to(root)):sha(path) for path in root.rglob('*') if path.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
