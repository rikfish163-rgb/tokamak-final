"""All360 retrospective diagnostic, NEVER report its scores as online gain."""
import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.run import read_signal
from ..continual_episode_20261023.run import labeled
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import REFERENCE,models_for_calibration
from ..history_matched_rule_20261023.probe import PREVIOUS,PREVIOUS_SHA
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.run import public
from .learning import predict

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/当前编码器正确监督能力诊断_事前约定_20261023.md'


def main(root,limit=None):
    if root.exists():raise FileExistsError(root)
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    assert sha(PREVIOUS/'completed.json')==PREVIOUS_SHA
    old=READ(PREVIOUS/'protocol.json');previous_done=READ(PREVIOUS/'completed.json')
    assert sha(PREVIOUS/'per_shot.json')==previous_done['artifact_sha256']['per_shot.json']
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];saved=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');order=old['shot_ids'] if limit is None else old['shot_ids'][:limit]
    files={Path(__file__),Path(__file__).with_name('learning.py'),
           io23.ROOT/'experiments/risk_aligned_online/tests/test_observed_oracle.py'}
    protocol=dict(scope='NONDEPLOYABLE_LABEL_ORACLE; DEVELOPED360; NOT A PERFORMANCE UPPER BOUND',
        shot_ids=order,threshold=old['threshold'],source_sha256=PARENT_SHA256,
        preparation_ids=old['preparation_ids'],calibration_ids=old['calibration_ids'],
        previous_sha256=PREVIOUS_SHA,plan_sha256=sha(PLAN),arms=['static','oracle1','oracle4'],
        learning_rate=.001,clip_norm=1.,first_update=256,stride=128,observed_windows_per_update=8,
        labels_accessed_before_current_shot=True,future_signals_in_gradient=False,event_labels_used=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in order})
    root.mkdir(parents=True);(root/'shots').mkdir();_durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):archive.write(f,str(f.relative_to(io23.ROOT)))
    rows={a:[] for a in protocol['arms']};receipts=[]
    for number,sid in enumerate(order,1):
        if time.monotonic()-started>1500:raise TimeoutError('25min diagnostic budget')
        # Intentionally NONCAUSAL: known outcomes construct oracle supervision.
        s=labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping)
        curves,receipt=predict(model,{k:s[k] for k in ('values','available')},s['y'],s['valid'])
        with np.load(PREVIOUS/'shots'/f'{sid}.npz') as old_curve:
            np.testing.assert_array_equal(s['times'],old_curve['times'])
            curves['static']=old_curve['static'].copy()
        for a in curves:
            np.testing.assert_allclose(curves[a][:256],curves['static'][:256],atol=2e-7,rtol=1e-6)
            rows[a].append(dict(shot=sid,is_disrupt=s['is_disrupt'],down=s['down'],
                times=s['times'][s['valid']],scores=curves[a][s['valid']]))
        np.savez_compressed(root/'shots'/f'{sid}.npz',times=s['times'],valid=s['valid'],**curves)
        receipts.append(dict(shot=sid,updates=receipt))
        if number%20==0:print(dict(scope='ORACLE_DIAGNOSTIC',shots=number,seconds=time.monotonic()-started),flush=True)
    results={a:metrics(rs,protocol['threshold'],lead=30.) for a,rs in rows.items()}
    assert_unchanged(model,saved);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'per_shot.json',results);_durable_json(root/'updates.json',receipts)
    artifacts=[root/'protocol.json',root/'per_shot.json',root/'updates.json',root/'code_snapshot.zip',*sorted((root/'shots').glob('*.npz'))]
    done=dict(status='COMPLETED_NONDEPLOYABLE_ORACLE_DIAGNOSTIC',metrics={a:public(v) for a,v in results.items()},
        seconds=time.monotonic()-started,source_unchanged=True,persistent_risk_model_unchanged=True,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in artifacts})
    _durable_json(root/'completed.json',done);print(done['metrics'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--limit',type=int)
    args=p.parse_args();main(args.output,args.limit)
