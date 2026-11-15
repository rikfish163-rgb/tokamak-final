"""Full TEST360 sham-feedback ablation with the fixed held-out calibration."""
import argparse
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
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import REFERENCE, REFERENCE_SHA, models_for_calibration
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .control import predict_sham

CALIBRATION=io23.ROOT/'experiments/risk_aligned_online/runs/heldout_threshold30_test360_20261023'
CALIBRATION_SHA='db0d680546ba185af26decd27e0180dfb2c46bb56c5816d2105f639999e4c231'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/真实变化与持平目标_事前对照_20261023.md'
ARMS=('static','real_intra','persistence_sham')


def main(root):
    if root.exists(): raise FileExistsError(root)
    assert sha(REFERENCE/'completed.json')==REFERENCE_SHA
    assert sha(CALIBRATION/'completed.json')==CALIBRATION_SHA
    for base in (REFERENCE,CALIBRATION):
        for name,digest in READ(base/'completed.json')['artifact_sha256'].items():assert sha(base/name)==digest,name
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,source_threshold=make_parent();model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
    rp=READ(REFERENCE/'protocol.json'); mapping=READ(REFERENCE/'input_map.json')
    rule=READ(CALIBRATION/'thresholds.json')['mapped_fixed']['0.2'];assert rule['threshold'] is not None
    root.mkdir(parents=True);(root/'shots').mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_persistence_feedback_control.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(shot_ids=rp['shot_ids'],source_sha256=PARENT_SHA256,reference_sha256=REFERENCE_SHA,
        calibration_sha256=CALIBRATION_SHA,threshold=rule['threshold'],source_threshold=source_threshold,
        arms=ARMS,plan_sha256=sha(PLAN),input_sha256=rp['input_sha256'],
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        scope='development-exposed matched-target ablation; not a proposed deployment policy',
        independent_confirmation=False,neural_training_steps=0,extra_target_labels=0)
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('outcomes forbidden while predicting')
    receipts=[];steps=0
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in p['shot_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            scores,updates,_=predict_sham(model,signal)
            with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(z['times'],s['times'])
                np.testing.assert_array_equal(scores['static'],z['mapped_fixed_static'])
                curves=dict(static=z['mapped_fixed_static'].copy(),real_intra=z['mapped_fixed_intra'].copy(),persistence_sham=scores['intra'])
            old=READ(REFERENCE/'fold0/shots'/f'{sid}_updates.json')['mapped_fixed']
            assert [(v['last_observation'],v['first_affected_prediction'],v['gradient_steps'],v['observed_targets']) for v in updates]==[(v['last_observation'],v['first_affected_prediction'],v['gradient_steps'],v['observed_targets']) for v in old]
            np.testing.assert_array_equal(curves['static'][:256],curves['persistence_sham'][:256])
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path),update_sha256=sha(up)));steps+=sum(v['gradient_steps'] for v in updates)
            if len(receipts)%20==0:print(dict(stage='predict',shots=len(receipts),seconds=time.monotonic()-start),flush=True)
    assert_unchanged(model,before);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'prediction_receipts.json',receipts)
    rows={a:[] for a in ARMS}
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz') as z:
            for a in ARMS:rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={}
    for label,threshold in (('calibrated',p['threshold']),('source_threshold',source_threshold)):
        for a in ARMS:
            r=metrics(rows[a],threshold,lead=30.);r['very_early_over3000']=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions']);results[a+'/'+label]=r
    _durable_json(root/'per_shot.json',results)
    done=dict(status='COMPLETED',metrics={k:public(v) for k,v in results.items()},online_steps=steps,
        source_unchanged=True,seconds=time.monotonic()-start,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
