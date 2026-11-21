"""Prepare four15-shot teachers, then seal all20 held-shot causal trajectories."""
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
from ..diverse_support_20261023.run import make_parent,READ
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..preparation_factors_20261023.learning import prepare
from ..redesign_20261004.prepare_prior import load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..heldout_threshold_20261023.run import REFERENCE
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.alignment import fit_statistics,transform,SIGNAL_KEYS
from ..warmstart_alignment_20261022.run import predict_many,public
from ..operational_bag_loss_20261025.learning import operational_losses
from .design import folds,fit_fold_map

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/operational_bag_loss_cal10_20261025'
GEOMETRY=io23.ROOT/'experiments/risk_aligned_online/runs/clock_channel_loss_cal10_20261025'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/按炮隔离纠错教学_事前约定_20261025.md'


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    assert READ(PREVIOUS/'audit.json')['status']=='passed'
    prior=READ(PREVIOUS/'protocol.json');ids=prior['preparation_ids']
    outcomes={sid:READ(GEOMETRY/'geometry'/f'{sid}.json') for sid in ids}
    plans=folds(ids,{sid:outcomes[sid]['is_disrupt'] for sid in ids})
    assert not set(ids)&set(prior['shot_ids']+prior['excluded360_ids'])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==prior['parent_sha256']
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();parent,_=make_parent();original=state_cpu(parent)
    root.mkdir(parents=True)
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_crossfit_teacher.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(folds=plans,preparation_ids=ids,excluded_calibration_ids=prior['shot_ids'],excluded360_ids=prior['excluded360_ids'],
        source_sha256=PARENT_SHA256,deployed_parent_sha256=prior['parent_sha256'],threshold=prior['threshold'],
        previous_completed_sha256=sha(PREVIOUS/'completed.json'),plan_sha256=sha(PLAN),
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in ids},
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        risk_steps_per_fold=512,forecast_steps_per_fold=256,seed=131,mapping='density_gain fit only15 training shots',
        deployed_model_changed=False,independent_confirmation=False,scope='historical20 crossfit teaching design; not independent validation',
        label_contract='JTEXT max25-AdvancedTime; EAST125; strict EAST timely lead>30',
        inter_steps=0,alarm_updates=0,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    reader=io23.read_outcome;source_ids=io23.source_roles()['train']
    access=[]
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return reader(device,sid)
    def deny(*args,**kwargs):raise AssertionError('outcomes forbidden in signal fitting or held prediction')
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    print(dict(stage='source_loaded',shots=len(source),seconds=round(time.monotonic()-start,2)),flush=True)
    with patch.object(io23,'read_outcome',side_effect=deny):
        source_statistics=fit_statistics([{k:s[k] for k in SIGNAL_KEYS} for s in source],device='J-TEXT',allowed_ids=source_ids)
        raw={sid:apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')) for sid in ids}
        for plan in plans:
            folder=root/f'fold{plan["fold"]}';folder.mkdir();(folder/'shots').mkdir()
            mapping=fit_fold_map(source_statistics,[{k:s[k] for k in SIGNAL_KEYS} for s in raw.values()],plan['train'])
            _durable_json(folder/'input_map.json',mapping)
    seals=[]
    for plan in plans:
        if time.monotonic()-start>1800:raise TimeoutError('declared30minute teacher limit')
        folder=root/f'fold{plan["fold"]}';mapping=READ(folder/'input_map.json')
        def train_guard(device,sid):
            assert device=='EAST' and sid in plan['train'] and sid not in plan['held']
            access.append(dict(fold=plan['fold'],shot=sid,phase='teacher_preparation'))
            return reader(device,sid)
        with patch.object(io23,'read_outcome',side_effect=train_guard):
            target=[]
            for sid in plan['train']:
                s=simple_data.attach_outcome(raw[sid],'EAST',{})
                target.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            model=copy.deepcopy(parent);prepare(model,source,target,folder,use_source=True)
        training_ids=set()
        for name in ('risk_manifest.jsonl','forecast_manifest.jsonl'):
            import json
            for line in (folder/name).read_text().splitlines():
                row=json.loads(line)
                for r in row['target']:training_ids.add(r['shot'])
        assert training_ids<=set(plan['train']) and not training_ids&set(plan['held'])
        _durable_json(folder/'identity_check.json',dict(status='passed',observed_target_ids=sorted(training_ids),held_ids=plan['held']))
        model_state=state_cpu(model)
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in plan['held']:
                signal=transform({k:raw[sid][k] for k in ('values','available')},mapping)
                # This is teaching geometry, so also retain the grad-enabled
                # math path used by the next finite-step outer objective.
                math_scores=predict_many(model,{'static':signal},device='cuda')['static']
                with torch.no_grad():scores=predict_many(model,{'static':signal},device='cuda')['static']
                path=folder/'shots'/f'{sid}.npz'
                np.savez_compressed(path,times=raw[sid]['times'],math_scores=math_scores,scores=scores)
                seals.append(dict(fold=plan['fold'],shot=sid,sha256=sha(path)))
        assert_unchanged(model,model_state);assert_unchanged(parent,original)
        print(dict(stage='teacher_prepared',fold=plan['fold'],held=plan['held'],seconds=round(time.monotonic()-start,2)),flush=True)
    _durable_json(root/'prediction_seals.json',seals);_durable_json(root/'preparation_label_access.json',access)
    rows={a:[] for a in ('math_scores','scores')};held_access=[]
    for plan in plans:
        for sid in plan['held']:
            flag,down=io23.read_outcome('EAST',sid);held_access.append(dict(shot=sid,phase='after_all20_seals'))
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
                _,valid=io23.labels_at(z['times'],flag,down,125.)
                for arm in rows:rows[arm].append(dict(shot=sid,is_disrupt=flag,down=down,times=z['times'][valid],scores=z[arm][valid]))
    result={};details={};bags={}
    for arm,rs in rows.items():
        m=metrics(rs,p['threshold'],lead=30.);loss,bag=operational_losses(rs,p['threshold'])
        result[arm]=dict(public(m),**loss,very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in m['predictions']))
        details[arm]=m;bags[arm]=bag
    _durable_json(root/'per_shot.json',details);_durable_json(root/'bag_losses.json',bags);_durable_json(root/'held_label_access.json',held_access)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256 and sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==p['deployed_parent_sha256']
    done=dict(status='completed',seconds=time.monotonic()-start,results=result,
        proceed_to_policy_fit=result['math_scores']['tp']<4,
        preparation_steps=4*(512+256),source_and_deployed_parent_unchanged=True,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
