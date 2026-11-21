"""Fixed ten-shot development experiment; no policy fitting or risk retuning."""
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
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import predict_many
from ..signal_driven_span_20261024 import run as evaluator
from .learning import ARMS,predict,eligible

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/finite_conditional_loss_cal10_20261025'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/预测误差水平与变化拆分_事前约定_20261025.md'


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    previous=READ(PREVIOUS/'completed.json');assert READ(PREVIOUS/'audit.json')['status']=='passed'
    for name,digest in previous['artifact_sha256'].items():assert sha(PREVIOUS/name)==digest,name
    old=READ(PREVIOUS/'protocol.json')
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');start=time.monotonic();root.mkdir(parents=True)
    for sub in ('shots','updates'):(root/sub).mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_forecast_profile.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=old['shot_ids'],preparation_ids=old['preparation_ids'],excluded360_ids=old['excluded360_ids'],
        source_sha256=PARENT_SHA256,parent_sha256=sha(REFERENCE/'fold0/mapped_fixed/prepared.pt'),
        mapping_sha256=sha(REFERENCE/'input_map.json'),previous_completed_sha256=sha(PREVIOUS/'completed.json'),
        plan_sha256=sha(PLAN),arms=('off',)+ARMS,threshold=old['threshold'],update_every=128,first_update=256,
        parameter_reset='parent every update',step_norm=.001,inter_steps=0,alarm_updates=0,
        labels='all prediction files sealed before evaluation outcome reads',new_history_training=False,
        scope='DEVELOPMENT_ONLY_NOT_COMPLETE_CONTINUAL_SYSTEM',independent_confirmation=False,event_labels_added=False,
        input_sha256=old['input_sha256'],torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('no outcomes before complete prediction seal')
    seals=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in protocol['shot_ids']:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s)
            with torch.no_grad():baseline=predict_many(model,dict(parent=s),device='cuda')['parent']
            np.testing.assert_array_equal(curves['off'],baseline)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:np.testing.assert_array_equal(curves['off'],z['off'])
            for arm in ARMS:np.testing.assert_array_equal(curves[arm][:256],baseline[:256])
            out=root/'shots'/f'{sid}.npz';np.savez_compressed(out,times=raw['times'],**curves)
            up=root/'updates'/f'{sid}.json';_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(out),updates_sha256=sha(up)))
            print(dict(stage='prediction',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True)
            if time.monotonic()-start>1200:raise TimeoutError('declared20minute screen limit exceeded')
    _durable_json(root/'prediction_seals.json',seals)
    with patch.object(evaluator,'ARMS',ARMS):results=evaluator.evaluate(root,protocol)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',seconds=time.monotonic()-start,results=results,
        eligible_for360=eligible(results),source_and_parent_unchanged=True,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
