"""20 historical shots learn23 weights, followed by sealed ten-shot evaluation."""
import argparse
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
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,CALIBRATION_IDS,PAIR,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import predict_many
from ..signal_driven_span_20261024 import run as evaluator
from .learning import ARMS,collect_shot,fit_policy,predict,eligible

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/部署时钟通道损失学习_事前约定_20261024.md'


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();prior=READ(PAIR/'protocol.json');history=prior['preparation_ids']
    assert len(history)==20 and not set(history+CALIBRATION_IDS)&set(prior['shot_ids'])
    assert not set(history)&set(CALIBRATION_IDS)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==prior['parent_sha256']
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed']
    original=state_cpu(model);mapping=READ(REFERENCE/'input_map.json')
    root.mkdir(parents=True)
    for name in ('geometry','shots','updates'): (root/name).mkdir()
    files=set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_clock_channel_loss.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=CALIBRATION_IDS,preparation_ids=history,excluded360_ids=prior['shot_ids'],
        threshold=.12540414929389954,arms=('off',)+ARMS,groups=['safe','early','danger'],
        source_sha256=PARENT_SHA256,parent_sha256=prior['parent_sha256'],
        mapping_sha256=sha(REFERENCE/'input_map.json'),plan_sha256=sha(PLAN),
        step_bound=.01,first_update=256,period=128,parent_reset_every_update=True,
        inter_steps=0,alarm_updates=0,independent_confirmation=False,
        event_labels_added_in_this_run=False,
        label_contract='20 historical outcomes for weights; ten development outcomes only after all prediction seals',
        scope='DEVELOPMENT_LOSS_DIRECTION_STUDY_NOT_COMPLETE_ONLINE_SYSTEM',
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in history+CALIBRATION_IDS},
        torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    reader=io23.read_outcome
    def only_history(device,sid):
        if device!='EAST' or sid not in history:raise AssertionError('only declared historical outcomes may train weights')
        return reader(device,sid)
    rows=[]
    with patch.object(io23,'read_outcome',side_effect=only_history):
        for sid in history:
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            flag,down=io23.read_outcome('EAST',sid)
            batch,events=collect_shot(model,mapped,raw['times'],sid,flag,down)
            rows.extend(batch)
            _durable_json(root/'geometry'/f'{sid}.json',dict(rows=batch,events=events,is_disrupt=bool(flag),down=down))
            print(dict(stage='historical_geometry',shot=sid,rows=len(batch),seconds=round(time.monotonic()-start,2)),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('declared30minute limit exceeded')
    policy,fits=fit_policy(rows);_durable_json(root/'policy.json',policy);_durable_json(root/'fit.json',fits)
    _durable_json(root/'policy_seal.json',dict(policy_sha256=sha(root/'policy.json'),fit_sha256=sha(root/'fit.json'),
        rows=len(rows),calibration_outcomes_accessed=0))
    print(dict(stage='fit',fits=fits,policy=policy),flush=True)
    results=None;passed=False
    if fits['block']['margin']>1e-8:
        seals=[]
        def deny(*args,**kwargs):raise AssertionError('no current labels before all prediction seals')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in CALIBRATION_IDS:
                raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
                curves,updates=predict(model,mapped,policy)
                with torch.no_grad():baseline=predict_many(model,dict(parent=mapped),device='cuda')['parent']
                np.testing.assert_array_equal(curves['off'],baseline)
                for arm in ARMS:np.testing.assert_array_equal(curves[arm][:256],baseline[:256])
                path=root/'shots'/f'{sid}.npz';np.savez_compressed(path,times=raw['times'],**curves)
                up=root/'updates'/f'{sid}.json';_durable_json(up,updates)
                seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                print(dict(stage='prediction',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True)
                if time.monotonic()-start>1800:raise TimeoutError('declared30minute limit exceeded')
        _durable_json(root/'prediction_seals.json',seals)
        with patch.object(evaluator,'ARMS',ARMS):results=evaluator.evaluate(root,protocol)
        passed=eligible(results)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',historical_rows=len(rows),seconds=time.monotonic()-start,
        training_margin=fits['block']['margin'],results=results,eligible_for360=passed,
        source_and_parent_unchanged=True,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result)
    print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
