"""Fixed centered-target comparison; reuse audited sampling and evaluation."""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile
import numpy as np
import torch

from ..internal_fast_window_20261023.run import (
    io23,READ,make_parent,models_for_calibration,REFERENCE,read_signal,labeled,
    transform,state_cpu,assert_unchanged,_durable_json,sha,DEFAULT_PARENT,PARENT_SHA256,
    training_pools,sample_batch,risk_step,predict,outcome_record,evaluate)
from .model import CenteredAdapter

PREVIOUS = io23.ROOT/'experiments/risk_aligned_online/runs/curvature_fast_window360_20261023'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/窗口内变化学习_事前约定.md'


def main(root, smoke=False):
    if root.exists(): raise FileExistsError(root)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    prior=READ(PREVIOUS/'protocol.json'); done=READ(PREVIOUS/'completed.json')
    assert READ(PREVIOUS/'verification.json')['completed_sha256']==sha(PREVIOUS/'completed.json')
    for f,h in done['artifact_sha256'].items(): assert sha(PREVIOUS/f)==h,f
    torch.set_num_threads(2); torch.manual_seed(131); start=time.monotonic()
    base,_=make_parent(); parent=models_for_calibration(base)['mapped_fixed']
    adaptive=CenteredAdapter(copy.deepcopy(parent.base)).cuda().eval(); capacity=copy.deepcopy(adaptive)
    original=state_cpu(parent.base); initial=state_cpu(adaptive.layer)
    old_layers=torch.load(PREVIOUS/'layers.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,old_layers['initial'][k]) for k,v in initial.items())
    root.mkdir(parents=True); (root/'shots').mkdir(); mapping=READ(REFERENCE/'input_map.json')
    ids=prior['shot_ids'][:2] if smoke else prior['shot_ids']; steps=4 if smoke else 512
    files=set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_centered_fast_window.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None): files.add(Path(module.__file__))
    protocol=dict(prior,shot_ids=ids,additional_preparation_steps=steps,
        variant='centered_auxiliary_target_with_trace_curvature_control',
        scope='SMOKE_ONLY' if smoke else 'DEVELOPMENT_ONLY; auxiliary mean component removed',
        previous_completed_sha256=sha(PREVIOUS/'completed.json'),plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files): archive.write(f,str(f.relative_to(io23.ROOT)))
    reader=io23.read_outcome; records=[]; access=[]
    def prep_reader(device,sid):
        assert device=='EAST' and sid in prior['preparation_ids']
        access.append(dict(shot=sid,phase='historical_preparation')); return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=prep_reader):
        for sid in prior['preparation_ids']: records.append(labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping))
    _durable_json(root/'preparation_outcomes.json',[dict(shot=s['shot'],is_disrupt=s['is_disrupt'],down=s['down']) for s in records])
    pools=training_pools(records); rng=np.random.default_rng(131)
    models=dict(capacity=capacity,adaptive=adaptive)
    opts={name:torch.optim.Adam(m.layer.parameters(),lr=.001) for name,m in models.items()}
    train=[]; old_train=READ(PREVIOUS/'training.json')
    def deny(*args,**kwargs): raise AssertionError('stream outcomes inaccessible before all predictions sealed')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for step in range(steps):
            x,m,y,samples=sample_batch(records,pools,rng,'cuda'); assert samples==old_train[step]['samples']
            with torch.no_grad(): h,valid=adaptive.tokens(x,m)
            row=dict(step=step,samples=samples)
            for name,net in models.items(): row[name]=risk_step(net,opts[name],h,valid,y,mode='on' if name=='adaptive' else 'off')
            assert row['capacity']==old_train[step]['capacity']; train.append(row)
            if (step+1)%128==0: print(dict(stage='prepare',step=step+1,seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'training.json',train)
    torch.save(dict(initial=initial,capacity=state_cpu(capacity.layer),adaptive=state_cpu(adaptive.layer)),root/'layers.pt')
    if not smoke: assert all(torch.equal(v,old_layers['capacity'][k]) for k,v in state_cpu(capacity.layer).items())
    states={name:state_cpu(m) for name,m in models.items()}; seals=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for index,sid in enumerate(ids):
            s=read_signal(sid); signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,traces=predict(adaptive,capacity,signal)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(s['times'],z['times']); np.testing.assert_array_equal(curves['parent'],z['parent'])
                if not smoke: np.testing.assert_array_equal(curves['capacity'],z['capacity'])
            assert np.all(traces['loss_after']<=traces['loss_before']+1e-6+1e-5*traces['loss_before']),sid
            f=root/'shots'/f'{sid}.npz'; t=root/'shots'/f'{sid}_inner.npz'
            np.savez_compressed(f,times=s['times'],**curves); np.savez_compressed(t,**traces)
            seals.append(dict(shot=sid,prediction_sha256=sha(f),inner_sha256=sha(t)))
            if (index+1)%20==0 or index+1==len(ids):
                progress=dict(shots=index+1,seconds=time.monotonic()-start)
                _durable_json(root/'progress.json',progress); print(progress,flush=True)
    _durable_json(root/'prediction_seals.json',seals); outcomes=[]
    for index,r in enumerate(seals):
        sid=r['shot']; assert sha(root/'shots'/f'{sid}.npz')==r['prediction_sha256']
        outcomes.append(outcome_record(sid,reader('EAST',sid),r['prediction_sha256'],index))
        access.append(dict(shot=sid,phase='after_all_prediction_seals'))
    _durable_json(root/'outcomes.json',outcomes); _durable_json(root/'label_access.json',access)
    for name,m in models.items(): assert_unchanged(m,states[name]); assert_unchanged(m.base,original)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=evaluate(root,protocol,outcomes)
    completed=dict(status='COMPLETED_SMOKE' if smoke else 'COMPLETED_CENTERED_WINDOW_EXPERIMENT',metrics=result,
        source_unchanged=True,parent_backbone_unchanged=True,seconds=time.monotonic()-start,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',completed)
    print({k:v for k,v in completed.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); main(args.root,args.smoke)
