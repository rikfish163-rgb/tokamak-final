"""Frozen checkpoints on all eligible historical shots outside their CV20 pool.

No training or model selection. Metadata selection is retrospective and explicit;
only values/masks enter the unchanged intra-shot predictor. Not TEST360.
"""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE,REF_SHA,load_model
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..increment_forecast_20261023.learning import predict
from ..labeled_alignment_20261023.run import historical_pool,SPLIT
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .analysis import partition,paired_changes,interval_peaks

PREPARED=io23.ROOT/'experiments/risk_aligned_online/runs/asymmetric_feedback_cv20_20261023'
PREPARED_SHA='7f4bda909ac80ad9c3346738737cb7ca47a4555338283af60e8d5f1effcbb6db'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/固定模型跨准备集诊断_事前约定_20261023.md'
MODELS=('warm','prepared')
CURVES=tuple(f'{m}_{s}' for m in MODELS for s in ('static','intra'))


def models_for(fold):
    warm=load_model(fold);other=copy.deepcopy(warm)
    other.load_state_dict(torch.load(PREPARED/f'fold{fold}'/'meta.pt',map_location='cuda',weights_only=True)['model'])
    other.set_stage('frozen')
    return dict(warm=warm,prepared=other)


def evaluate(root,p):
    summaries={};changes={};details=[];access=[]
    for f in range(4):
        records={a:[] for a in CURVES}
        for sid in p['shot_ids']:
            disruptive,down=io23.read_outcome('EAST',sid)
            access.append(dict(fold=f,shot=sid,phase='all_predictions_sealed'))
            with np.load(root/f'fold{f}'/'shots'/f'{sid}.npz') as z:
                times=z['times'];valid=io23.labels_at(times,disruptive,down,125.)[1]
                for a in CURVES:
                    r=dict(shot=sid,times=times[valid],scores=z[a][valid],is_disrupt=disruptive,down=down)
                    records[a].append(r)
                    details.append(dict(fold=f,arm=a,shot=sid,is_disrupt=disruptive,
                        **interval_peaks(times,z[a],disruptive,down)))
        results={a:metrics(rs,p['threshold'],lead=30.) for a,rs in records.items()}
        summaries[str(f)]={a:dict(**public(v),very_early_over3000=sum(
            r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in results.items()}
        _durable_json(root/f'fold{f}'/'per_shot.json',results)
        changes[str(f)]={m:paired_changes(results[f'{m}_static']['predictions'],results[f'{m}_intra']['predictions']) for m in MODELS}
        changes[str(f)]['preparation_static']=paired_changes(results['warm_static']['predictions'],results['prepared_static']['predictions'])
    _durable_json(root/'evaluation_access.json',access)
    _durable_json(root/'peak_details.json',details)
    return summaries,changes


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    for ref,digest in ((REFERENCE,REF_SHA),(PREPARED,PREPARED_SHA)):
        assert sha(ref/'completed.json')==digest
        assert READ(ref/'verification.json')['status']=='VERIFIED'
        done=READ(ref/'completed.json')
        for name,h in done['artifact_sha256'].items():assert sha(ref/name)==h,name
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    prior=READ(REFERENCE/'protocol.json');folds=prior['folds']
    assert folds==READ(PREPARED/'protocol.json')['folds']
    pool,audit=historical_pool(io23.cohort_ids('EAST'));rows=partition(pool,folds,io23.cohort_ids('EAST'))
    assert len(rows)==65 and sum(r['is_disrupt'] for r in rows)==4
    ids=[r['shot'] for r in rows]
    torch.set_num_threads(2);torch.manual_seed(131);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    root.mkdir(parents=True)
    paths=set(Path(__file__).parent.glob('*.py'));paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_external65.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):z.write(path,str(path.relative_to(io23.ROOT)))
    checkpoints={str(ref/f'fold{f}'/name):sha(ref/f'fold{f}'/name) for ref,name in
        ((REFERENCE,'increment.pt'),(PREPARED,'meta.pt')) for f in range(4)}
    p=dict(folds=folds,shot_ids=ids,roster=rows,threshold=prior['threshold'],source_sha256=PARENT_SHA256,
        checkpoints=checkpoints,reference_sha256=REF_SHA,prepared_sha256=PREPARED_SHA,
        plan_sha256=sha(PLAN),code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in paths},
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in ids},
        split_sha256=sha(SPLIT),metadata_inspected=88,offline_steps=0,inter_steps=0,
        retrospective_selection=True,independent_confirmation=False,test360_accessed=False,
        cuda=torch.version.cuda,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p);_durable_json(root/'historical_pool.json',audit)
    seals=[];access=[];update_counts={m:0 for m in MODELS};original_signal=simple_data.read_shot
    def guard_signal(device,sid,**kwargs):
        if device!='EAST' or sid not in ids:raise AssertionError('signal outside declared65')
        access.append(dict(kind='signal',shot=sid));return original_signal(device,sid,**kwargs)
    def guard_outcome(*args,**kwargs):raise AssertionError('predictor must not access outcome')
    with patch.object(simple_data,'read_shot',side_effect=guard_signal),patch.object(io23,'read_outcome',side_effect=guard_outcome):
        for f in range(4):
            folder=root/f'fold{f}';(folder/'shots').mkdir(parents=True)
            models=models_for(f);frozen={a:state_cpu(m) for a,m in models.items()}
            mapping=READ(REFERENCE/f'fold{f}'/'input_map.json')
            assert mapping==READ(PREPARED/f'fold{f}'/'input_map.json')
            _durable_json(folder/'input_map.json',mapping)
            for number,sid in enumerate(ids,1):
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping);curves={};updates={}
                for name,model in models.items():
                    pred,receipt,packets=predict(model,signal)
                    curves.update({f'{name}_{k}':v for k,v in pred.items()});updates[name]=receipt
                    update_counts[name]+=sum(r['gradient_steps'] for r in receipt)
                path=folder/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**curves)
                _durable_json(folder/'shots'/f'{sid}_updates.json',updates)
                seals.append(dict(fold=f,shot=sid,sha256=sha(path)))
                if number%5==0 or number==len(ids):
                    _durable_json(root/'progress.json',dict(fold=f,shots=number,seconds=time.monotonic()-start))
                    print(dict(stage='predict',fold=f,shots=number,seconds=round(time.monotonic()-start,2)),flush=True)
            for name,model in models.items():assert_unchanged(model,frozen[name])
            del model,models,frozen
    _durable_json(root/'seals.json',seals);_durable_json(root/'signal_access.json',access)
    summary,changes=evaluate(root,p)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    for path,digest in checkpoints.items():assert sha(path)==digest
    result=dict(status='COMPLETED',metrics=summary,changes=changes,updates=update_counts,
        seconds=time.monotonic()-start,peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20,
        source_unchanged=True,test360_accessed=False,independent_confirmation=False,
        distinct_shots=65,folds=4,offline_steps=0,inter_steps=0,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',result)
    print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
