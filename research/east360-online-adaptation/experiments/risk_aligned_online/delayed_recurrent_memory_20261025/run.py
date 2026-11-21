"""Prespecified CUDA screen: delayed recurrent memory, no alarm/inter updates."""
import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch
import zipfile
import numpy as np
import torch

from .. import io23
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,signal,transform,
    REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..crossfit_teacher_20261025.fit import load_teachers, TEACHERS
from ..operational_bag_loss_20261025 import run as evaluator
from ..operational_bag_loss_20261025.learning import operational_losses, historical_gate
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.run import public
from .model import ARMS,DiagnosticMemory,RecurrentAdapter
from .learning import prepare_example,train,predict,eligible,original_predictions

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/crossfit_policy_cal10_20261025'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/延迟诊断记忆_事前约定_20261025.md'
GEOMETRY=io23.ROOT/'experiments/risk_aligned_online/runs/clock_channel_loss_cal10_20261025/geometry'


def evaluate(root,protocol):
    with patch.object(evaluator,'ARMS',ARMS):return evaluator.evaluate(root,protocol)


def history_evaluation(teachers,memory,threshold,root,check_time):
    rows={'off':[],'memory':[]}
    for t in teachers:
        adapter=RecurrentAdapter(t['model'].base).cuda();adapter.memory=memory
        for sid in t['held']:
            old=READ(GEOMETRY/f'{sid}.json');raw=signal(sid)
            s=transform({k:raw[k] for k in ('values','available')},t['mapping'])
            curves,_,_=predict(adapter,s,arms=('off','memory'))
            np.savez_compressed(root/'historical'/f'{sid}.npz',times=raw['times'],**curves)
            _,valid=io23.labels_at(raw['times'],old['is_disrupt'],old['down'],125.)
            for arm in rows:
                rows[arm].append(dict(shot=sid,times=raw['times'][valid],scores=curves[arm][valid],
                                      is_disrupt=old['is_disrupt'],down=old['down']))
            check_time()
    losses={};detail={};scores={}
    for arm,rs in rows.items():
        losses[arm],detail[arm]=operational_losses(rs,threshold)
        metric=metrics(rs,threshold,lead=30.)
        scores[arm]=dict(public(metric),very_early_over3000=sum(
            r['timely'] and r['lead_ms']>3000 for r in metric['predictions']))
    deltas=[dict(shot=a['shot'],flag=a['flag'],delta=b['loss']-a['loss'])
            for a,b in zip(detail['off'],detail['memory'])]
    result=dict(losses=losses,per_shot_losses=detail,deltas=deltas,counts=scores,
                passed=historical_gate(deltas,scores['memory'],scores['off']))
    _durable_json(root/'historical_gate.json',result)
    return result


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    old=READ(PREVIOUS/'protocol.json')
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==old['parent_sha256']
    assert sha(TEACHERS/'completed.json')=='166190042ab812799a7af448f1d3d251a01112bca1703f5948be8004983b712e'
    assert READ(TEACHERS/'audit.json')['status']=='passed'
    assert not set(old['preparation_ids']+old['shot_ids'])&set(old['excluded360_ids'])
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic()
    parent,_=make_parent();deployed=models_for_calibration(parent)['mapped_fixed']
    original=state_cpu(deployed);teachers=load_teachers(parent)
    originals=[state_cpu(t['model']) for t in teachers]
    memory=DiagnosticMemory(64).cuda()
    assert sum(p.numel() for p in memory.parameters())==7093
    root.mkdir(parents=True)
    for sub in ('shots','updates','training','historical'):(root/sub).mkdir()
    files=set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_delayed_recurrent_memory.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=old['shot_ids'],preparation_ids=old['preparation_ids'],excluded360_ids=old['excluded360_ids'],
        threshold=old['threshold'],arms=('off',)+ARMS,source_sha256=PARENT_SHA256,parent_sha256=old['parent_sha256'],
        mapping_sha256=sha(REFERENCE/'input_map.json'),teachers_completed_sha256=sha(TEACHERS/'completed.json'),
        plan_sha256=sha(PLAN),epochs=10,outer_lr=.001,optimizer='Adam',seed=131,slow_parameters=7093,
        patch_samples=8,direct_horizon_patches=4,first_affected_index=40,memory_radius=1.,forecast_loss_weight=.02,
        slow_selection='minimum historical selected-endpoint risk objective; epoch0 included',
        reset='prior once per shot; no cross-shot writeback',inter_steps=0,alarm_updates=0,
        backend='all arms math SDPA, MHA fastpath disabled, endpoint batch32; comparator direct original tensors, not cloned functional parameters',
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in old['preparation_ids']+old['shot_ids']},
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        scope='DEVELOPMENT_ONLY_NOT_COMPLETE_CONTINUAL_SYSTEM',independent_confirmation=False,event_labels_added=False)
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def check_time():
        if time.monotonic()-start>1800:raise TimeoutError('declared30minute limit')
    def deny(*args,**kwargs):raise AssertionError('new outcomes forbidden until prediction seals')
    with patch.object(io23,'read_outcome',side_effect=deny):
        examples={};receipts=[]
        for t in teachers:
            adapter=RecurrentAdapter(t['model'].base).cuda();adapter.memory=memory
            for sid in t['held']:
                raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},t['mapping'])
                o=READ(GEOMETRY/f'{sid}.json')
                e=prepare_example(adapter,s,raw['times'],o['is_disrupt'],o['down'],old['threshold'])
                examples[sid]=e
                receipts.append(dict(shot=sid,fold=t['fold'],**{k:e[k] for k in
                    ('baseline','selected_ends','full_candidate_count','frozen_peak_full','frozen_peak_selected')}))
                print(dict(stage='cache',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True);check_time()
        _durable_json(root/'selected_endpoints.json',receipts)
        def progress(r):
            _durable_json(root/'training'/f'epoch{r["epoch"]:02d}.json',r)
            print(dict(stage='training',epoch=r['epoch'],objective=r['objective'],forecast=r['forecast_loss'],
                       seconds=round(time.monotonic()-start,2)),flush=True);check_time()
        best=train(memory,examples,old['threshold'],progress=progress)
        torch.save(best['state'],root/'memory.pt')
        _durable_json(root/'selection.json',{k:v for k,v in best.items() if k!='state'})
        _durable_json(root/'memory_seal.json',dict(sha256=sha(root/'memory.pt'),new_outcomes_accessed=0))
        history=history_evaluation(teachers,memory,old['threshold'],root,check_time)
        print(dict(stage='historical_gate',passed=history['passed'],seconds=round(time.monotonic()-start,2)),flush=True)
        for t,saved in zip(teachers,originals):assert_unchanged(t['model'],saved)
        adapter=RecurrentAdapter(deployed.base).cuda();adapter.memory=memory
        mapping=READ(REFERENCE/'input_map.json');seals=[];backends={}
        for sid in old['shot_ids']:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates,logits=predict(adapter,s)
            baseline=original_predictions(deployed.base,s)
            np.testing.assert_array_equal(curves['off'],baseline)
            for arm in ARMS:np.testing.assert_array_equal(curves[arm][:40],baseline[:40])
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:
                backends[str(sid)]=dict(max_abs_probability=float(np.max(np.abs(z['off']-baseline))),
                    changed_threshold_points=int(np.sum((z['off']>=old['threshold'])!=(baseline>=old['threshold']))))
            out=root/'shots'/f'{sid}.npz';np.savez_compressed(out,times=raw['times'],**curves,**{a+'_logit':v for a,v in logits.items()})
            up=root/'updates'/f'{sid}.json';_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(out),updates_sha256=sha(up)))
            print(dict(stage='prediction',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True);check_time()
        _durable_json(root/'prediction_seals.json',seals)
        _durable_json(root/'backend_comparison.json',backends)
    results=evaluate(root,protocol)
    assert_unchanged(deployed,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='completed',seconds=time.monotonic()-start,results=results,
              selected_epoch=best['epoch'],historical_gate_passed=history['passed'],
              eligible_for360=eligible(results,history['passed']),source_parent_and_teachers_unchanged=True,
              scientific_goal_achieved=False,
              artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:main(args.out)
    except Exception as exc:
        if args.out.exists():_durable_json(args.out/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
