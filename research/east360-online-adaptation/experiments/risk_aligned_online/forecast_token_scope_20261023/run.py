"""Keep four fitted classifiers; vary diagnostic readout and temporary scope."""
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

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import natural_batch,eligible_records,SIGNAL_KEYS
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..horizon_memory_20261023.learning import FoldAccess
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple23 import fit_forecaster
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import PatchForecast,predict,ARMS

READ=lambda p:json.loads(p.read_text())
ROOT=io23.ROOT
REFERENCE=ROOT/'experiments/risk_aligned_online/runs/east_negative_sampling_cv20_20261023'
REFERENCE_SHA='ae808b9a661149677e62991e5aef989efe3df03a2a0e3984fa1fd3f3210b5cff'
PLAN=ROOT/'docs/research/source_event_rebuild_20261022/诊断头读取CLS与更新范围_事前约定_20261023.md'


def snapshot(root):
    files=set(Path(__file__).parent.glob('*.py'))
    files.add(ROOT/'experiments/risk_aligned_online/tests/test_forecast_token_scope.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):
            path=Path(module.__file__)
            if path.suffix=='.py':files.add(path)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):archive.write(path,str(path.relative_to(ROOT)))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def prepare(source_parent,fit_parent,source,target,out,reference):
    cfg=fit_parent.cfg;origin=state_cpu(fit_parent)
    sf,se=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in source],cfg,future=True)
    tf,te=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in target],cfg,future=True)
    _durable_json(out/'forecast_exclusions.json',dict(source=se,target=te))
    models={}
    for mode in ('full','patch'):
        model=copy.deepcopy(fit_parent)
        model.base.forecast_head=copy.deepcopy(source_parent.base.forecast_head)
        if mode=='patch':model.base.forecast_head=PatchForecast(model.base.forecast_head,cfg.dim)
        model.set_stage('frozen')
        optimizer=torch.optim.Adam(model.base.forecast_head.parameters(),lr=3e-4)
        rs,rt=np.random.default_rng(4131),np.random.default_rng(5131);logs=[]
        with (out/f'{mode}_manifest.jsonl').open('x') as manifest:
            for step in range(256):
                a=natural_batch(sf,rs,32,cfg,'cuda',future=True);b=natural_batch(tf,rt,32,cfg,'cuda',future=True)
                x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
                manifest.write(json.dumps(dict(step=step,source=a[-1],target=b[-1]))+'\n')
                loss=fit_forecaster(model.base,optimizer,x,m,t,tm);logs.append(dict(step=step,loss=loss))
        _durable_json(out/f'{mode}_training.json',logs)
        assert (out/f'{mode}_manifest.jsonl').read_bytes()==(reference/'forecast_manifest.jsonl').read_bytes()
        model.set_stage('frozen');assert_unchanged(model,origin,('base.forecast_head.weight','base.forecast_head.bias'))
        if mode=='full':
            assert_unchanged(model,origin)
            assert logs==READ(reference/'forecast_training.json')
        else:
            # Masked columns receive no gradient, not a hidden trained shortcut.
            torch.testing.assert_close(model.base.forecast_head.weight[:,-cfg.dim:],source_parent.base.forecast_head.weight[:,-cfg.dim:],rtol=0,atol=0)
        models[mode]=model
        torch.save(dict(model=state_cpu(model),config=cfg.to_dict(),readout=mode),out/f'{mode}.pt')
    return models


def score(root,plans,threshold):
    rows={a:[] for a in ('static',)+ARMS};access=[]
    for plan in plans:
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            access.append(dict(shot=sid,phase='score_after_all_seals'));valid=s['valid']
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                np.testing.assert_array_equal(s['times'],trace['times'])
                for a in rows:rows[a].append(dict(shot=sid,times=s['times'][valid],scores=trace[a][valid],is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:metrics(r,threshold,lead=30.) for a,r in rows.items()}
    _durable_json(root/'per_shot.json',results);_durable_json(root/'evaluation_access.json',access)
    return {a:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])) for a,r in results.items()}


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REFERENCE_SHA
    old=READ(REFERENCE/'completed.json');oldp=READ(REFERENCE/'protocol.json');plans=oldp['folds']
    for f in plans:
        for name in ('parent.pt','input_map.json','forecast_manifest.jsonl','forecast_training.json','access.json'):
            path=REFERENCE/f'fold{f["fold"]}'/'legacy'/name
            assert sha(path)==old['artifact_sha256'][str(path.relative_to(REFERENCE))]
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    base,info=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
    origin=state_cpu(parent);source_ids=io23.source_roles()['train'];threshold=float(info['threshold']);root.mkdir(parents=True)
    protocol=dict(reference=str(REFERENCE),reference_sha256=REFERENCE_SHA,source_sha256=PARENT_SHA256,source_checkpoint=str(DEFAULT_PARENT),
        folds=plans,threshold=threshold,plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256=snapshot(root),
        arms=ARMS,seed=131,forecast_steps=256,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        input_policy='same necessary handling, no empirical density map',test360_accessed=False,
        scope='historical CV20, not independent future confirmation',classifier_preparation='unchanged legacy fold parents')
    _durable_json(root/'protocol.json',protocol)
    read_signal,read_outcome=simple_data.read_shot,io23.read_outcome;source_access=[]
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        source_access.append(sid);return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    _durable_json(root/'source_label_access.json',source_access);seals=[];update_count=0
    for plan in plans:
        f=plan['fold'];out=root/f'fold{f}';out.mkdir();(out/'shots').mkdir();ref=REFERENCE/f'fold{f}'/'legacy'
        fit=copy.deepcopy(parent);fit.load_state_dict(torch.load(ref/'parent.pt',map_location='cuda',weights_only=True)['model']);fit.set_stage('frozen')
        mapping=READ(ref/'input_map.json');_durable_json(out/'input_map.json',mapping);access=FoldAccess(plan,source_ids)
        def signal_guard(device,sid,**kwargs):
            access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
        def outcome_guard(device,sid):
            access.check('outcome',device,sid);return read_outcome(device,sid)
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            # Forecast preparation has no use for EAST outcomes, so none are read.
            records=[apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')) for sid in plan['fit']]
            target=[dict(r,**transform({k:r[k] for k in ('values','available')},mapping)) for r in records]
            models=prepare(parent,fit,source,target,out,ref);frozen={h:state_cpu(m) for h,m in models.items()};access.phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                predictions,updates=predict(models,transform({k:s[k] for k in ('values','available')},mapping))
                with np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                    np.testing.assert_array_equal(predictions['static'],trace['static']);np.testing.assert_array_equal(predictions['full_all'],trace['intra'])
                reference_updates=READ(ref/'shots'/f'{sid}_updates.json')
                assert len(reference_updates)==len(updates['full_all'])
                for r,u in zip(reference_updates,updates['full_all']):assert r=={k:u[k] for k in r}
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**predictions);_durable_json(up,updates)
                update_count+=sum(r['gradient_steps'] for rows in updates.values() for r in rows)
                seals.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            for h,m in models.items():assert_unchanged(m,frozen[h])
        _durable_json(out/'access.json',access.log);assert_unchanged(parent,origin)
        print(dict(stage='fold_complete',fold=f,shots=len(seals),seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'prediction_receipts.json',seals);results=score(root,plans,threshold)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',seconds=time.monotonic()-start,metrics=results,neural_steps=update_count,
              exact_reference_reproduced=True,source_unchanged=True,test360_accessed=False,independent_confirmation=False,
              artifact_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    _durable_json(root/'completed.json',done);print(json.dumps({k:v for k,v in done.items() if k!='artifact_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
