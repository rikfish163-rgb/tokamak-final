"""Learn feedback weights on held shots; deploy only on the original20-shot parent."""
import argparse
import copy
import gc
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile
import numpy as np
import torch

from .. import io23
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import predict_many
from ..operational_bag_loss_20261025 import run as previous_run
from .prepare import PREVIOUS,PLAN
from .learning import ARMS,train_policy,predict,eligible

TEACHERS=io23.ROOT/'experiments/risk_aligned_online/runs/crossfit_teachers20_20261025'


def load_teachers(parent):
    result=[]
    for fold in READ(TEACHERS/'protocol.json')['folds']:
        folder=TEACHERS/f'fold{fold["fold"]}'
        model=copy.deepcopy(parent)
        checkpoint=torch.load(folder/'prepared.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(checkpoint['model']);model.set_stage('frozen');model.eval()
        result.append(dict(fold=fold['fold'],model=model,held=fold['held'],mapping=READ(folder/'input_map.json')))
    return result


def cache_teachers(teachers,threshold,progress=None):
    receipts=[];baselines={};bytes_used=0
    for teacher in teachers:
        shots,rows,size=previous_run.build_history(teacher['model'],teacher['held'],teacher['mapping'],threshold,progress=progress)
        teacher['shots']=shots;bytes_used+=size
        if bytes_used>5*1024**3:raise MemoryError('combined teacher cache exceeds5GiB')
        receipts.extend(dict(fold=teacher['fold'],**r) for r in rows)
        baselines.update({str(sid):dict(fold=teacher['fold'],**s['baseline']) for sid,s in shots.items()})
    return receipts,baselines,bytes_used


def evaluate(root,protocol):
    with patch.object(previous_run,'ARMS',ARMS):return previous_run.evaluate(root,protocol)


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    for folder in (PREVIOUS,TEACHERS):
        done=READ(folder/'completed.json');assert done['status']=='completed'
        for name,digest in done['artifact_sha256'].items():assert sha(folder/name)==digest,name
    assert READ(TEACHERS/'completed.json')['proceed_to_policy_fit']
    assert READ(PREVIOUS/'audit.json')['status']=='passed'
    old=READ(PREVIOUS/'protocol.json');history=old['preparation_ids'];ids=old['shot_ids']
    assert not set(history)&set(ids) and not set(history+ids)&set(old['excluded360_ids'])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==old['parent_sha256']
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    teachers=load_teachers(parent);teacher_states=[state_cpu(t['model']) for t in teachers]
    mapping=READ(REFERENCE/'input_map.json');start=time.monotonic()
    root.mkdir(parents=True)
    for sub in ('shots','updates','training'):(root/sub).mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_crossfit_policy.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(shot_ids=ids,preparation_ids=history,excluded360_ids=old['excluded360_ids'],threshold=old['threshold'],
        folds=READ(TEACHERS/'protocol.json')['folds'],arms=('off',)+ARMS,
        parent_sha256=old['parent_sha256'],source_sha256=PARENT_SHA256,mapping_sha256=sha(REFERENCE/'input_map.json'),
        previous_completed_sha256=sha(PREVIOUS/'completed.json'),teachers_completed_sha256=sha(TEACHERS/'completed.json'),
        plan_sha256=sha(PLAN),epochs=10,optimizer='Adam',outer_lr=.05,step_bound=.01,
        global_trainable_parameters=24,context_trainable_parameters=47,update_every=128,first_update=256,
        parameter_reset='parent every update',inter_steps=0,alarm_updates=0,
        outer_objective='same whole-shot max threshold margin; four15-shot teachers evaluated on disjoint held5 each',
        labels='20 historical labels for teaching; all7/all10 predictions sealed before new outcome reads',
        scope='DEVELOPMENT_ONLY_NOT_COMPLETE_CONTINUAL_SYSTEM',independent_confirmation=False,event_labels_added=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in history+ids},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        training_backend='grad-enabled math, exact32-chunk replay',deployment_backend='no_grad fused, unchanged')
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('new outcomes forbidden during policy fitting/prediction')
    def check_time():
        if time.monotonic()-start>1800:raise TimeoutError('declared30minute experiment limit')
    with patch.object(io23,'read_outcome',side_effect=deny):
        def cache_progress(sid,blocks,size):
            print(dict(stage='cache',shot=sid,fold_blocks=blocks,fold_GiB=round(size/1024**3,3),
                       seconds=round(time.monotonic()-start,2)),flush=True);check_time()
        receipts,baselines,bytes_used=cache_teachers(teachers,p['threshold'],cache_progress)
        _durable_json(root/'cache_receipts.json',receipts)
        _durable_json(root/'historical_baselines.json',baselines)
        records={};selections={}
        def progress(kind,record):
            _durable_json(root/'training'/f'{kind}_{record["epoch"]:02d}.json',record)
            print(dict(stage='training',kind=kind,epoch=record['epoch'],loss=record['objective'],
                history_gate=record['history_gate']['passed'],seconds=round(time.monotonic()-start,2)),flush=True);check_time()
        for kind in ('global','context'):
            best,_=train_policy(teachers,kind,p['threshold'],progress=progress)
            records[kind]=best['policy'];selections[kind]={k:best[k] for k in ('epoch','objective','history_gate')}
            _durable_json(root/f'{kind}_selection.json',dict(selections[kind],selection='minimum20-held-shot training objective only'))
        for teacher,state in zip(teachers,teacher_states):assert_unchanged(teacher['model'],state)
        _durable_json(root/'policies.json',records)
        _durable_json(root/'policy_seal.json',dict(sha256=sha(root/'policies.json'),new_outcomes_accessed=0))
        del teachers;gc.collect()
        insample=READ(PREVIOUS/'policies.json')['context'];seals=[]
        for sid in ids:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s,records,insample)
            with torch.no_grad():baseline=predict_many(model,dict(parent=s),device='cuda')['parent']
            np.testing.assert_array_equal(curves['off'],baseline)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:np.testing.assert_array_equal(curves['insample'],z['bag_context'])
            for arm in ARMS:np.testing.assert_array_equal(curves[arm][:256],baseline[:256])
            out=root/'shots'/f'{sid}.npz';np.savez_compressed(out,times=raw['times'],**curves)
            up=root/'updates'/f'{sid}.json';_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(out),updates_sha256=sha(up)))
            print(dict(stage='prediction',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True);check_time()
    _durable_json(root/'prediction_seals.json',seals)
    results=evaluate(root,p)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',seconds=time.monotonic()-start,cache_bytes=bytes_used,results=results,
        eligible_for360=eligible(results,selections['context']['history_gate']['passed']),
        source_parent_and_teachers_unchanged=True,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
